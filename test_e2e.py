"""End-to-end check of the CivicLens API against the live Supabase project.

Creates a throwaway citizen, submits two similar reports (expecting a merge),
drives the official status flow, runs resolution verification, then deletes
everything it created.

Start the backend first, then run:  .venv/Scripts/python test_e2e.py

The test photos are synthetic, so the backend must judge them with CLIP:
start it with VISION_ENGINE=clip (or no ANTHROPIC_API_KEY). The script refuses
to run against a Claude-judging backend, so it never spends API credit.
"""
import os, sys, uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import httpx
from app.config import settings
from app.db import service
from test_pipeline import synth_road, synth_garbage, _jpeg

API = os.environ.get("CIVICLENS_API", "http://127.0.0.1:8000")
LAT, LNG = 28.63150, 77.21670
db = service()
created_user = None
ok = True

def step(name, good, extra=""):
    global ok
    ok = ok and good
    print(f"{'PASS' if good else 'FAIL'}  {name}{('  — ' + extra) if extra else ''}")

def token_for(email, password):
    r = httpx.post(
        f"{settings.supabase_url}/auth/v1/token?grant_type=password",
        headers={"apikey": settings.supabase_publishable_key, "Content-Type": "application/json"},
        json={"email": email, "password": password}, timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]

vision = httpx.get(f"{API}/api/health", timeout=30).json().get("vision", {})
if vision.get("engine") == "claude":
    print("The backend is judging photos with Claude. Restart it with VISION_ENGINE=clip "
          "before running this test, so it doesn't spend API credit on synthetic images.")
    sys.exit(2)

try:
    email = f"e2e-{uuid.uuid4().hex[:8]}@civiclens.test"
    pw = "E2eTest!2026"
    res = db.auth.admin.create_user({
        "email": email, "password": pw, "email_confirm": True,
        "user_metadata": {"full_name": "E2E Tester"},
    })
    created_user = res.user.id
    step("citizen created + auto-confirmed", bool(created_user))

    prof = db.table("profiles").select("role,full_name").eq("id", created_user).execute().data
    step("profiles trigger fired", bool(prof) and prof[0]["role"] == "citizen",
         str(prof[0] if prof else None))

    citizen = token_for(email, pw)
    official = token_for(settings.official_email, settings.official_password)
    step("both roles can log in", bool(citizen and official))

    me = httpx.get(f"{API}/api/me", headers={"Authorization": f"Bearer {official}"}, timeout=30).json()
    step("backend resolves the official role", me.get("role") == "official", str(me))

    # ---- 1st report --------------------------------------------------------
    img1 = _jpeg(synth_road("big"))
    r = httpx.post(f"{API}/api/analyze",
                   headers={"Authorization": f"Bearer {citizen}"},
                   files={"image": ("a.jpg", img1, "image/jpeg")},
                   data={"lat": LAT, "lng": LNG}, timeout=180)
    step("POST /api/analyze", r.status_code == 200, f"HTTP {r.status_code}")
    a1 = r.json()
    step("category + department routed", a1["category"] == "pothole" and a1["department_slug"] == "roads",
         f"{a1['category']} -> {a1['department_name']}, severity {a1['visual_severity']}, "
         f"coverage {a1['coverage']:.0%}, sensitive={a1['near_sensitive_place']['found']}")
    step("image uploaded to storage", a1["image_url"].startswith("http"), a1["image_url"][:72] + "...")
    step("no duplicate on an empty table", a1["duplicate"] is None)

    r = httpx.post(f"{API}/api/reports", headers={"Authorization": f"Bearer {citizen}"},
                   json={"analysis_id": a1["analysis_id"], "note": "E2E test report", "address": "Connaught Place, New Delhi"},
                   timeout=60)
    step("POST /api/reports", r.status_code == 200, f"HTTP {r.status_code}")
    sub1 = r.json()
    cid = sub1["complaint"]["id"]
    step("complaint created", not sub1["merged"] and sub1["complaint"]["report_count"] == 1)
    ev = db.table("status_events").select("status").eq("complaint_id", cid).execute().data
    step("timeline seeded", [e["status"] for e in ev] == ["submitted"], str([e["status"] for e in ev]))

    # ---- duplicate merge ---------------------------------------------------
    r = httpx.post(f"{API}/api/analyze", headers={"Authorization": f"Bearer {citizen}"},
                   files={"image": ("b.jpg", _jpeg(synth_road("big")), "image/jpeg")},
                   data={"lat": LAT + 0.0002, "lng": LNG}, timeout=180)
    a2 = r.json()
    step("duplicate detected within 50 m", a2.get("duplicate") is not None,
         f"similarity {a2['duplicate']['similarity']:.3f} at {a2['duplicate']['distance_m']} m"
         if a2.get("duplicate") else "no match")
    r = httpx.post(f"{API}/api/reports", headers={"Authorization": f"Bearer {citizen}"},
                   json={"analysis_id": a2["analysis_id"], "note": None, "address": None}, timeout=60)
    sub2 = r.json()
    step("second report merged into the cluster",
         sub2["merged"] and sub2["complaint"]["report_count"] == 2,
         f"report_count={sub2['complaint']['report_count']}, severity={sub2['complaint']['final_severity']}, "
         f"priority={sub2['complaint']['priority_score']}")

    # ---- different category nearby should NOT merge ------------------------
    r = httpx.post(f"{API}/api/analyze", headers={"Authorization": f"Bearer {citizen}"},
                   files={"image": ("c.jpg", _jpeg(synth_garbage()), "image/jpeg")},
                   data={"lat": LAT, "lng": LNG}, timeout=180)
    a3 = r.json()
    step("different category stays its own case",
         a3.get("accepted") and a3["category"] == "garbage" and a3["duplicate"] is None,
         f"{a3.get('category')} -> {a3.get('department_name')}")

    # ---- rejection of a non-issue -----------------------------------------
    r = httpx.post(f"{API}/api/analyze", headers={"Authorization": f"Bearer {citizen}"},
                   files={"image": ("d.jpg", _jpeg(synth_road(None)), "image/jpeg")},
                   data={"lat": LAT, "lng": LNG}, timeout=180)
    step("non-issue photo rejected with 422", r.status_code == 422,
         (r.json().get("detail", {}) or {}).get("reason", "")[:70])

    # ---- official flow -----------------------------------------------------
    for st in ("verified", "assigned", "in_progress"):
        r = httpx.post(f"{API}/api/complaints/{cid}/status", headers={"Authorization": f"Bearer {official}"},
                       json={"status": st, "note": None}, timeout=60)
        if r.status_code != 200:
            step(f"official set status -> {st}", False, f"HTTP {r.status_code} {r.text[:90]}")
            break
    else:
        row = db.table("complaints").select("status").eq("id", cid).single().execute().data
        ev = db.table("status_events").select("status").eq("complaint_id", cid).order("created_at").execute().data
        step("official status flow", row["status"] == "in_progress" and len(ev) == 4,
             f"{[e['status'] for e in ev]}")

    r = httpx.post(f"{API}/api/complaints/{cid}/status", headers={"Authorization": f"Bearer {citizen}"},
                   json={"status": "resolved", "note": None}, timeout=60)
    step("citizen blocked from changing status", r.status_code == 403, f"HTTP {r.status_code}")

    # ---- verification ------------------------------------------------------
    # the problem is still there -> must NOT verify
    r = httpx.post(f"{API}/api/complaints/{cid}/verify", headers={"Authorization": f"Bearer {official}"},
                   files={"image": ("after.jpg", _jpeg(synth_road("big")), "image/jpeg")}, timeout=180)
    v = r.json()
    step("after-photo still showing the issue is refused", r.status_code == 200 and not v["verified"],
         f"issue_conf={v.get('issue_confidence')}, same_place={v.get('same_place_similarity')}")
    step("failed check does NOT resolve the case",
         db.table("complaints").select("status").eq("id", cid).single().execute().data["status"] == "in_progress")
    step("failed check is still written to the audit trail",
         any("verification failed" in (e["note"] or "").lower()
             for e in db.table("status_events").select("note").eq("complaint_id", cid).execute().data))

    # the road is clean now -> must verify and resolve
    r = httpx.post(f"{API}/api/complaints/{cid}/verify", headers={"Authorization": f"Bearer {official}"},
                   files={"image": ("after2.jpg", _jpeg(synth_road(None)), "image/jpeg")}, timeout=180)
    v2 = r.json()
    step("clean after-photo verifies the fix", v2["verified"],
         f"issue_conf={v2.get('issue_confidence')}, same_place={v2.get('same_place_similarity')}, "
         f"score={v2.get('verification_score')}")
    row = db.table("complaints").select("status,resolved_at").eq("id", cid).single().execute().data
    step("verified fix resolves + stamps resolved_at",
         row["status"] == "resolved" and bool(row["resolved_at"]))
    step("resolution row stored with the after image",
         bool(db.table("resolutions").select("after_image_url,verified").eq("complaint_id", cid)
              .eq("verified", True).execute().data))

    n_before = len(db.table("status_events").select("id").eq("complaint_id", cid).execute().data)
    r = httpx.post(f"{API}/api/complaints/{cid}/verify", headers={"Authorization": f"Bearer {official}"},
                   files={"image": ("late.jpg", _jpeg(synth_road("big")), "image/jpeg")}, timeout=180)
    n_after = len(db.table("status_events").select("id").eq("complaint_id", cid).execute().data)
    step("verifying a fixed case is refused and leaves history alone",
         r.status_code == 409 and n_after == n_before, f"HTTP {r.status_code}")

    r = httpx.post(f"{API}/api/complaints/{cid}/force-resolve", headers={"Authorization": f"Bearer {official}"},
                   json={"note": "E2E override"}, timeout=60)
    step("override endpoint works", r.json()["status"] == "resolved")

    # ---- upvote on a closed case ------------------------------------------
    r = httpx.post(f"{API}/api/complaints/{cid}/upvote", headers={"Authorization": f"Bearer {citizen}"}, timeout=60)
    step("upvote on a closed case is a no-op", r.json().get("closed") is True)

    # ---- hourly recompute --------------------------------------------------
    r = httpx.post(f"{API}/api/recompute", headers={"Authorization": f"Bearer {official}"}, timeout=120)
    step("severity recompute endpoint", r.status_code == 200, str(r.json()))

finally:
    print("\ncleaning up…")
    try:
        rows = db.table("complaints").select("id").in_("id", []).execute()
    except Exception:
        pass
    if created_user:
        mine = db.table("reports").select("complaint_id").eq("user_id", created_user).execute().data or []
        ids = {r["complaint_id"] for r in mine}
        for cid_ in ids:
            db.table("complaints").delete().eq("id", cid_).execute()
        db.auth.admin.delete_user(created_user)
        print(f"  removed {len(ids)} test complaint(s) and the test citizen")

print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
sys.exit(0 if ok else 1)
