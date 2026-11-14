"""The CivicLens ingest pipeline: analyse → cluster → score → persist."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import math
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

from . import claude_vision, geo, severity as sev, vision
from .config import settings
from .db import department_map, service

log = logging.getLogger("civiclens.pipeline")

OPEN_STATUSES = ("submitted", "verified", "assigned", "in_progress")


# ---------------------------------------------------------------------------
# pgvector helpers (PostgREST speaks the text form)
# ---------------------------------------------------------------------------


def to_vec(arr: np.ndarray | None) -> str | None:
    if arr is None:
        return None
    return "[" + ",".join(f"{float(x):.6f}" for x in np.asarray(arr).ravel()) + "]"


def from_vec(value) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return np.asarray(value, dtype=np.float32)
    try:
        return np.asarray(json.loads(value), dtype=np.float32)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def upload_image(bucket: str, data: bytes, content_type: str = "image/jpeg") -> str:
    ext = {"image/png": "png", "image/webp": "webp"}.get(content_type, "jpg")
    path = f"{datetime.now(timezone.utc):%Y/%m}/{uuid.uuid4().hex}.{ext}"
    store = service().storage.from_(bucket)
    store.upload(path, data, {"content-type": content_type, "upsert": "false"})
    url = store.get_public_url(path)
    return url.rstrip("?")


# ---------------------------------------------------------------------------
# Pending analyses: a signed scan token, so submitting does not re-run the AI
# ---------------------------------------------------------------------------
# The analysis travels to the browser and back as an HMAC-signed, compressed
# token instead of living in this process's memory. Serverless functions (Vercel)
# don't share memory between requests, so an in-memory cache would lose scans.

_PENDING_TTL = 30 * 60


class AlreadySubmitted(Exception):
    """The same scan was submitted twice (double tap, retry)."""


def _token_key() -> bytes:
    return hashlib.sha256(("civiclens-scan:" + settings.supabase_secret_key).encode()).digest()


def remember(analysis: dict, user_id: str) -> str:
    payload = {"a": analysis, "u": user_id, "exp": int(time.time() + _PENDING_TTL)}
    raw = zlib.compress(json.dumps(payload, separators=(",", ":"), default=float).encode(), 9)
    body = base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
    sig = hmac.new(_token_key(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def recall(token: str, user_id: str) -> dict | None:
    try:
        body, sig = token.rsplit(".", 1)
        good = hmac.new(_token_key(), body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, good):
            return None
        payload = json.loads(zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))))
    except Exception:
        return None
    if payload.get("u") != user_id or payload.get("exp", 0) < time.time():
        return None
    return payload["a"]


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------


def find_duplicate(
    embedding: np.ndarray | None, lat: float, lng: float, category: str, data: bytes | None = None
) -> dict | None:
    if embedding is None:
        return find_duplicate_by_claude(data, lat, lng, category) if data else None
    try:
        res = service().rpc(
            "match_complaints",
            {
                "query_embedding": to_vec(embedding),
                "lat": lat,
                "lng": lng,
                "radius_m": settings.dup_radius_m,
                "category": category,
            },
        ).execute()
    except Exception as exc:
        log.warning("match_complaints failed: %s", exc)
        return None

    for row in res.data or []:
        if float(row.get("similarity") or 0) >= settings.dup_similarity:
            return row
    return None


DUP_CANDIDATES = 3
DUP_MIN_LIKELIHOOD = 0.6


def find_duplicate_by_claude(data: bytes, lat: float, lng: float, category: str) -> dict | None:
    """Without CLIP (Vercel): find open cases of the same kind within the radius and
    let Claude compare photos. No nearby case = no extra Claude call."""
    radius = settings.dup_radius_m
    dlat = radius / 111_320
    dlng = radius / (111_320 * max(0.01, math.cos(math.radians(lat))))
    try:
        rows = (
            service()
            .table("complaints")
            .select("id,lat,lng,visual_severity,report_count,created_at,cover_image_url")
            .eq("category", category)
            .in_("status", list(OPEN_STATUSES))
            .gte("lat", lat - dlat).lte("lat", lat + dlat)
            .gte("lng", lng - dlng).lte("lng", lng + dlng)
            .execute()
            .data
            or []
        )
    except Exception as exc:
        log.warning("nearby case lookup failed: %s", exc)
        return None

    near = []
    for row in rows:
        d = geo.haversine_m(lat, lng, float(row["lat"]), float(row["lng"]))
        if d <= radius and row.get("cover_image_url"):
            near.append((d, row))
    near.sort(key=lambda t: t[0])

    cands, images = [], []
    for d, row in near[:DUP_CANDIDATES]:
        raw = fetch_image(row["cover_image_url"])
        if raw is None:
            continue
        try:
            images.append(vision.open_image(raw))
            cands.append((d, row))
        except Exception:
            continue
    if not cands:
        return None

    res = claude_vision.same_problem(
        vision.open_image(data), images, vision.CATEGORY_LABELS.get(category, category), radius
    )
    if not res or res["index"] < 0 or res["likelihood"] < DUP_MIN_LIKELIHOOD:
        return None
    d, row = cands[res["index"]]
    return {**row, "similarity": res["likelihood"], "distance_m": d}


# ---------------------------------------------------------------------------
# Analyse
# ---------------------------------------------------------------------------


def analyze(data: bytes, lat: float, lng: float, content_type: str, user_id: str = "") -> dict:
    """Run CV + proximity + duplicate search. Nothing is written to the
    complaint tables yet — only the image is stored."""
    # the school/hospital lookup doesn't depend on the photo, so it runs while the AI looks
    pool = ThreadPoolExecutor(max_workers=1)
    sensitive_job = pool.submit(geo.nearby_sensitive, lat, lng)
    pool.shutdown(wait=False)

    result = vision.analyze_bytes(data)
    if not result["accepted"]:
        return {
            "accepted": False,
            "reason": result["reason"],
            "ai_engine": result["engine"],
            "scores": result["scores"],
            "confidence": round(float(result["confidence"]), 4),
        }

    category: str = result["category"]
    emb: np.ndarray | None = result["embedding"]

    dup = find_duplicate(emb, lat, lng, category, data)
    sensitive = sensitive_job.result()

    report_count = int(dup["report_count"]) + 1 if dup else 1
    created_at = dup["created_at"] if dup else datetime.now(timezone.utc).isoformat()
    base_visual = (
        max(int(dup["visual_severity"]), int(result["visual_severity"])) if dup
        else int(result["visual_severity"])
    )

    scored = sev.compute(base_visual, report_count, created_at, sensitive)
    image_url = upload_image("report-images", data, content_type)

    departments = department_map()
    analysis = {
        "accepted": True,
        "category": category,
        "category_label": vision.CATEGORY_LABELS[category],
        "confidence": round(float(result["confidence"]), 4),
        "department_slug": result["department_slug"],
        "department_id": departments.get(result["department_slug"]),
        "department_name": DEPARTMENT_NAMES[result["department_slug"]],
        "visual_severity": base_visual,
        "severity_expected": result["severity_expected"],
        "severity_distribution": result["severity_distribution"],
        "coverage": result["coverage"],
        "ai_engine": result["engine"],
        "description": result.get("description"),
        "final_severity": scored["final_severity"],
        "priority_score": scored["priority_score"],
        "boosts": scored["boosts"],
        "near_sensitive_place": sensitive,
        "sensitive_label": geo.describe_sensitive(sensitive),
        "duplicate": (
            {
                "complaint_id": dup["id"],
                "report_count": int(dup["report_count"]),
                "similarity": round(float(dup["similarity"]), 4),
                "distance_m": int(round(float(dup["distance_m"]))),
            }
            if dup
            else None
        ),
        "report_count": report_count,
        "explanation": sev.explain(
            category,
            base_visual,
            result["coverage"],
            report_count,
            sensitive,
            scored["boosts"],
            scored["final_severity"],
            result.get("description"),
        ),
        "image_url": image_url,
        "lat": lat,
        "lng": lng,
        "scores": result["scores"],
        "_embedding": to_vec(emb),
    }
    analysis["analysis_id"] = remember(analysis, user_id)
    return analysis


DEPARTMENT_NAMES = {
    "roads": "Roads & Infrastructure",
    "sanitation": "Sanitation & Waste",
    "lighting": "Street Lighting & Electrical",
    "water": "Water & Sewerage",
}


# ---------------------------------------------------------------------------
# Submit
# ---------------------------------------------------------------------------


def submit(analysis: dict, user_id: str, note: str | None, address: str | None) -> dict:
    """Create a new complaint or attach the report to the duplicate cluster."""
    db = service()
    if db.table("reports").select("id").eq("image_url", analysis["image_url"]).limit(1).execute().data:
        raise AlreadySubmitted("this report was already sent")
    category = analysis["category"]
    lat, lng = float(analysis["lat"]), float(analysis["lng"])
    emb = analysis["_embedding"]
    dup = analysis.get("duplicate")

    if dup:
        complaint_id = dup["complaint_id"]
        current = (
            db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
        )
        report_count = int(current["report_count"]) + 1
        visual = max(int(current["visual_severity"]), int(analysis["visual_severity"]))
        sensitive = current.get("near_sensitive_place") or analysis["near_sensitive_place"]
        scored = sev.compute(visual, report_count, current["created_at"], sensitive)
        patch = {
            "report_count": report_count,
            "visual_severity": visual,
            "final_severity": scored["final_severity"],
            "priority_score": scored["priority_score"],
            "near_sensitive_place": sensitive,
            "ai_summary": sev.summary_text(
                category, visual, analysis["coverage"], current.get("address"), sensitive, report_count,
                analysis.get("description"),
            ),
        }
        db.table("complaints").update(patch).eq("id", complaint_id).execute()
        complaint = {**current, **patch}
        merged = True
    else:
        sensitive = analysis["near_sensitive_place"]
        scored = sev.compute(analysis["visual_severity"], 1, datetime.now(timezone.utc), sensitive)
        row = {
            "category": category,
            "department_id": analysis["department_id"],
            "title": sev.title_text(category, address),
            "ai_summary": sev.summary_text(
                category, analysis["visual_severity"], analysis["coverage"], address, sensitive, 1,
                analysis.get("description"),
            ),
            "lat": lat,
            "lng": lng,
            "address": address,
            "visual_severity": analysis["visual_severity"],
            "final_severity": scored["final_severity"],
            "priority_score": scored["priority_score"],
            "report_count": 1,
            "status": "submitted",
            "near_sensitive_place": sensitive,
            "embedding": emb,
            "cover_image_url": analysis["image_url"],
        }
        complaint = db.table("complaints").insert(row).execute().data[0]
        complaint_id = complaint["id"]
        merged = False

    report = (
        db.table("reports")
        .insert(
            {
                "user_id": user_id,
                "complaint_id": complaint_id,
                "image_url": analysis["image_url"],
                "lat": lat,
                "lng": lng,
                "note": (note or "").strip() or None,
                "ai_raw": {
                    "category": category,
                    "confidence": analysis["confidence"],
                    "visual_severity": analysis["visual_severity"],
                    "severity_expected": analysis["severity_expected"],
                    "coverage": analysis["coverage"],
                    "scores": analysis["scores"],
                    "engine": analysis.get("ai_engine"),
                    "description": analysis.get("description"),
                    "merged_into_existing": merged,
                },
                "embedding": emb,
            }
        )
        .execute()
        .data[0]
    )

    if not merged:
        db.table("status_events").insert(
            {
                "complaint_id": complaint_id,
                "status": "submitted",
                "note": "Report received and analysed by CivicLens AI.",
                "actor_id": user_id,
                "image_url": analysis["image_url"],
            }
        ).execute()

    if not (complaint.get("near_sensitive_place") or {}).get("checked"):
        refresh_sensitive_async(complaint_id)

    complaint.pop("embedding", None)
    return {"complaint": complaint, "report": {**report, "embedding": None}, "merged": merged}


# ---------------------------------------------------------------------------
# Upvote ("I see this too")
# ---------------------------------------------------------------------------


def refresh_sensitive(complaint_id: str) -> None:
    """Fill in the school/hospital/trunk-road check that the scan could not
    finish in time, then re-score the complaint. Runs off the request path."""
    db = service()
    try:
        row = (
            db.table("complaints")
            .select("id,lat,lng,status,visual_severity,report_count,created_at,near_sensitive_place,category,address")
            .eq("id", complaint_id)
            .single()
            .execute()
            .data
        )
    except Exception:
        return
    if not row or (row.get("near_sensitive_place") or {}).get("checked"):
        return

    sensitive = geo.nearby_sensitive(
        float(row["lat"]), float(row["lng"]), budget=geo.OVERPASS_BUDGET_BG, force=True
    )
    if not sensitive.get("checked"):
        return

    scored = sev.compute(
        int(row["visual_severity"]), int(row["report_count"]), row["created_at"], sensitive
    )
    try:
        db.table("complaints").update(
            {
                "near_sensitive_place": sensitive,
                "final_severity": scored["final_severity"],
                "priority_score": scored["priority_score"],
            }
        ).eq("id", complaint_id).execute()
        log.info("proximity filled in for %s (found=%s)", complaint_id[:8], sensitive["found"])
    except Exception as exc:
        log.warning("proximity refresh failed for %s: %s", complaint_id[:8], exc)


def refresh_sensitive_async(complaint_id: str) -> None:
    threading.Thread(target=refresh_sensitive, args=(complaint_id,), daemon=True).start()


def upvote(complaint_id: str, user_id: str) -> dict:
    db = service()
    current = db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
    if current["status"] in ("resolved", "rejected"):
        return {"complaint": {k: v for k, v in current.items() if k != "embedding"},
                "already": False, "closed": True}

    existing = (
        db.table("reports")
        .select("id")
        .eq("complaint_id", complaint_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
        .data
    )
    if existing:
        return {"complaint": {k: v for k, v in current.items() if k != "embedding"},
                "already": True, "closed": False}

    db.table("reports").insert(
        {
            "user_id": user_id,
            "complaint_id": complaint_id,
            "image_url": current.get("cover_image_url"),
            "lat": current["lat"],
            "lng": current["lng"],
            "note": "Confirmed by another resident ('I see this too').",
            "ai_raw": {"upvote": True},
        }
    ).execute()

    report_count = int(current["report_count"]) + 1
    scored = sev.compute(
        int(current["visual_severity"]),
        report_count,
        current["created_at"],
        current.get("near_sensitive_place"),
    )
    patch = {
        "report_count": report_count,
        "final_severity": scored["final_severity"],
        "priority_score": scored["priority_score"],
    }
    db.table("complaints").update(patch).eq("id", complaint_id).execute()
    return {"complaint": {**{k: v for k, v in current.items() if k != "embedding"}, **patch},
            "already": False, "closed": False}


# ---------------------------------------------------------------------------
# Status changes, resolution verification
# ---------------------------------------------------------------------------

STATUS_FLOW = ("submitted", "verified", "assigned", "in_progress", "resolved", "rejected")

STATUS_NOTES = {
    "verified": "A city officer reviewed the report and confirmed the issue.",
    "assigned": "The issue was assigned to the responsible department crew.",
    "in_progress": "Work has started on site.",
    "resolved": "The issue has been fixed and verified.",
    "rejected": "The report was closed without action.",
}


def set_status(complaint_id: str, status: str, note: str | None, actor_id: str) -> dict:
    if status not in STATUS_FLOW:
        raise ValueError("unknown status")
    db = service()
    patch: dict = {"status": status}
    if status == "resolved":
        patch["resolved_at"] = datetime.now(timezone.utc).isoformat()
    else:
        patch["resolved_at"] = None
    db.table("complaints").update(patch).eq("id", complaint_id).execute()
    db.table("status_events").insert(
        {
            "complaint_id": complaint_id,
            "status": status,
            "note": (note or "").strip() or STATUS_NOTES.get(status),
            "actor_id": actor_id,
        }
    ).execute()
    row = db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
    row.pop("embedding", None)
    return row


def fetch_image(url: str | None) -> bytes | None:
    """Download the original report photo so Claude can compare before and after."""
    if not url:
        return None
    try:
        import httpx

        r = httpx.get(url, timeout=20, follow_redirects=True)
        r.raise_for_status()
        return r.content
    except Exception as exc:
        log.warning("could not fetch the original photo (%s); CLIP will check the fix", exc)
        return None


class CaseClosed(Exception):
    """Raised when someone tries to verify a fix on a case that is no longer open."""


def verify_resolution(complaint_id: str, data: bytes, content_type: str, official_id: str) -> dict:
    db = service()
    current = db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
    if current["status"] in ("resolved", "rejected"):
        # a stale tab or a double tap must not add a "failed" entry to a closed case's history
        raise CaseClosed(f"this case is already {'fixed' if current['status'] == 'resolved' else 'closed'}")
    before = from_vec(current.get("embedding"))  # None for cases filed without CLIP
    before_photo = fetch_image(current.get("cover_image_url")) if claude_vision.enabled() else None

    check = vision.verify_resolution(data, before, current["category"], before_photo)
    after_url = upload_image("resolution-images", data, content_type)

    db.table("resolutions").insert(
        {
            "complaint_id": complaint_id,
            "official_id": official_id,
            "after_image_url": after_url,
            "verification_score": check["verification_score"],
            "verified": check["verified"],
            "ai_notes": check["ai_notes"],
        }
    ).execute()

    if check["verified"]:
        db.table("complaints").update(
            {"status": "resolved", "resolved_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", complaint_id).execute()
        db.table("status_events").insert(
            {
                "complaint_id": complaint_id,
                "status": "resolved",
                "note": check["ai_notes"],
                "actor_id": official_id,
                "image_url": after_url,
            }
        ).execute()
    else:
        db.table("status_events").insert(
            {
                "complaint_id": complaint_id,
                "status": current["status"],
                "note": f"AI verification failed — {check['ai_notes']}",
                "actor_id": official_id,
                "image_url": after_url,
            }
        ).execute()

    row = db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
    row.pop("embedding", None)
    return {
        "verified": check["verified"],
        "verification_score": check["verification_score"],
        "issue_confidence": check["issue_confidence"],
        "same_place_similarity": check["same_place_similarity"],
        "ai_notes": check["ai_notes"],
        "ai_engine": check["engine"],
        "after_image_url": after_url,
        "complaint": row,
    }


def force_resolve(complaint_id: str, official_id: str, note: str | None) -> dict:
    db = service()
    last = (
        db.table("resolutions")
        .select("after_image_url")
        .eq("complaint_id", complaint_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    image_url = last[0]["after_image_url"] if last else None
    db.table("complaints").update(
        {"status": "resolved", "resolved_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", complaint_id).execute()
    db.table("status_events").insert(
        {
            "complaint_id": complaint_id,
            "status": "resolved",
            "note": (note or "").strip()
            or "Marked resolved by a city officer, overriding the AI check.",
            "actor_id": official_id,
            "image_url": image_url,
        }
    ).execute()
    row = db.table("complaints").select("*").eq("id", complaint_id).single().execute().data
    row.pop("embedding", None)
    return row


# ---------------------------------------------------------------------------
# Hourly severity recompute
# ---------------------------------------------------------------------------


def recompute_open_severities() -> int:
    db = service()
    rows = (
        db.table("complaints")
        .select("id,lat,lng,visual_severity,report_count,created_at,near_sensitive_place,final_severity,priority_score")
        .in_("status", list(OPEN_STATUSES))
        .execute()
        .data
        or []
    )

    # retry a handful of proximity checks that Overpass could not answer earlier
    pending = [r for r in rows if not (r.get("near_sensitive_place") or {}).get("checked")][:10]
    for row in pending:
        refresh_sensitive(row["id"])
    if pending:
        rows = (
            db.table("complaints")
            .select("id,lat,lng,visual_severity,report_count,created_at,near_sensitive_place,final_severity,priority_score")
            .in_("status", list(OPEN_STATUSES))
            .execute()
            .data
            or []
        )

    changed = 0
    for row in rows:
        scored = sev.compute(
            int(row["visual_severity"]),
            int(row["report_count"]),
            row["created_at"],
            row.get("near_sensitive_place"),
        )
        if (
            scored["final_severity"] != row["final_severity"]
            or abs(scored["priority_score"] - float(row["priority_score"])) > 1e-6
        ):
            db.table("complaints").update(
                {
                    "final_severity": scored["final_severity"],
                    "priority_score": scored["priority_score"],
                }
            ).eq("id", row["id"]).execute()
            changed += 1
    log.info("severity recompute: %s/%s complaints updated", changed, len(rows))
    return changed
