"""Claude (Sonnet 5.5) vision for CivicLens.

Claude looks at the photo and judges it: is there a civic problem, which kind,
how bad, and a one-line description a person can read. CLIP still runs for
every photo because duplicate merging needs its 512-number fingerprint.

Claude is used only when ANTHROPIC_API_KEY is set, VISION_ENGINE is "auto" and
today's call count is under CLAUDE_DAILY_LIMIT. Any failure (no internet,
credit used up, bad key, refusal, odd reply) returns None and the caller falls
back to CLIP, so the app never breaks because of the API.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import threading
from datetime import datetime, timezone

from PIL import Image

from .config import settings

log = logging.getLogger("civiclens.claude")

CATEGORIES = ["pothole", "garbage", "streetlight", "waterlogging", "sewage"]

_lock = threading.Lock()
_usage = {"day": "", "calls": 0}
_client = None


# ---------------------------------------------------------------------------
# plumbing
# ---------------------------------------------------------------------------


def enabled() -> bool:
    return settings.vision_engine.lower() == "auto" and bool(settings.anthropic_api_key.strip())


def status() -> dict:
    with _lock:
        today = datetime.now(timezone.utc).date().isoformat()
        calls = _usage["calls"] if _usage["day"] == today else 0
    return {
        "engine": "claude" if enabled() else "clip",
        "model": settings.claude_model if enabled() else None,
        "calls_today": calls,
        "daily_limit": settings.claude_daily_limit,
    }


def _take_call() -> bool:
    """Count one API call against today's cap. False once the cap is reached."""
    with _lock:
        today = datetime.now(timezone.utc).date().isoformat()
        if _usage["day"] != today:
            _usage["day"], _usage["calls"] = today, 0
        if _usage["calls"] >= settings.claude_daily_limit:
            return False
        _usage["calls"] += 1
        return True


def _get_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(
            api_key=settings.anthropic_api_key.strip(), timeout=45.0, max_retries=1
        )
    return _client


def _image_block(img: Image.Image) -> dict:
    """Shrink to ~1092 px on the long side (Claude's sweet spot, fewer tokens) and send as JPEG."""
    img = img.copy()
    img.thumbnail((1092, 1092))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
        },
    }


def _ask(content: list[dict], schema: dict) -> dict | None:
    if not enabled():
        return None
    if not _take_call():
        log.warning("Claude daily limit (%s) reached — using CLIP", settings.claude_daily_limit)
        return None

    import anthropic

    try:
        response = _get_client().messages.create(
            model=settings.claude_model,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": schema},
            },
            messages=[{"role": "user", "content": content}],
        )
    except anthropic.AuthenticationError:
        log.error("Claude rejected the API key — using CLIP")
        return None
    except anthropic.PermissionDeniedError as exc:
        log.error("Claude permission denied (%s) — using CLIP", exc.message)
        return None
    except anthropic.RateLimitError:
        log.warning("Claude rate limited — using CLIP")
        return None
    except anthropic.APIStatusError as exc:
        # 400 with a billing message lands here when the credit runs out
        log.error("Claude API error %s: %s — using CLIP", exc.status_code, exc.message)
        return None
    except anthropic.APIConnectionError:
        log.warning("Claude unreachable — using CLIP")
        return None

    if response.stop_reason != "end_turn":
        log.warning("Claude stopped with %s — using CLIP", response.stop_reason)
        return None
    text = next((b.text for b in response.content if b.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        log.warning("Claude reply was not JSON — using CLIP")
        return None
    u = response.usage
    log.info("Claude %s: %s in / %s out tokens", settings.claude_model, u.input_tokens, u.output_tokens)
    return data


# ---------------------------------------------------------------------------
# prompts + schemas
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the photo inspector for CivicLens, an app where citizens report \
problems on public streets so the city can fix them. You judge photos honestly and in plain, \
everyday English that anyone can understand. Never guess at things you cannot see."""

ANALYZE_PROMPT = """A citizen just took this photo to report a problem on a public street.

Decide:
1. is_issue: does the photo clearly show one of these public problems?
   - pothole: holes, cracks or broken surface on a road or pavement
   - garbage: dumped rubbish, litter piles, overflowing public bins
   - streetlight: a broken, damaged, leaning or dark street light
   - waterlogging: standing water or flooding on a street
   - sewage: overflowing drains, open manholes, sewage on the street
   Answer false for selfies, food, pets, indoor rooms, screenshots, documents, sky, \
clean streets, or anything that is not one of the problems above.
2. category: the best matching problem (pick the closest one even when is_issue is false).
3. confidence: 0 to 1, how sure you are about is_issue and category together.
4. severity: 1 to 5 for how bad it looks.
   1 = tiny (hairline crack, one bag of litter, a puddle at the edge)
   2 = small (small pothole, small pile at the kerb, one dead lamp on a lit street)
   3 = moderate (pothole that jolts a car, pile across the pavement, water across a lane)
   4 = large (pothole across half a lane, rubbish spilling into the road, street mostly dark, \
deep water across the street)
   5 = severe (road barely drivable, rubbish across the whole road, whole street dark, \
vehicles under water, raw sewage across the street)
5. coverage: 0 to 1, roughly how much of the picture the problem takes up.
6. description: one short sentence (under 20 words) describing only the problem, \
e.g. "Deep pothole across most of the left lane, with loose gravel around it." \
Never mention or describe people, faces, vehicles' number plates or anything identifying.
7. reason: if is_issue is false, one friendly sentence telling the citizen why and what to \
photograph instead. If is_issue is true, an empty string."""

ANALYZE_SCHEMA = {
    "type": "object",
    "properties": {
        "is_issue": {"type": "boolean"},
        "category": {"type": "string", "enum": CATEGORIES},
        "confidence": {"type": "number"},
        "severity": {"type": "integer"},
        "coverage": {"type": "number"},
        "description": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["is_issue", "category", "confidence", "severity", "coverage", "description", "reason"],
    "additionalProperties": False,
}

VERIFY_PROMPT = """The FIRST photo is the original report of a {label}. \
The SECOND photo was taken by a city official who says the problem is now fixed.

Decide:
1. same_place: is the second photo taken at the same spot? Look at fixed things like kerbs, \
walls, buildings, poles, trees and road markings. Lighting, angle and time of day may differ.
2. same_place_likelihood: 0 to 1, the chance the two photos show the same spot (0 = clearly a different place, 1 = clearly the same place). It must agree with same_place.
3. issue_still_present: is the {label} still visible in the second photo?
4. issue_confidence: 0 to 1, how likely it is that the {label} is still there \
(0 = clearly fixed, 1 = clearly still there). It must agree with issue_still_present.
5. notes: one or two short plain sentences for the case history explaining your decision, \
e.g. "The pothole has been filled with fresh asphalt; the kerb and lamp post match the original photo." Never mention or describe people.
"""

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "same_place": {"type": "boolean"},
        "same_place_likelihood": {"type": "number"},
        "issue_still_present": {"type": "boolean"},
        "issue_confidence": {"type": "number"},
        "notes": {"type": "string"},
    },
    "required": ["same_place", "same_place_likelihood", "issue_still_present", "issue_confidence", "notes"],
    "additionalProperties": False,
}


def _unit(x, default: float = 0.5) -> float:
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# entry points
# ---------------------------------------------------------------------------


def analyze(img: Image.Image) -> dict | None:
    data = _ask([_image_block(img), {"type": "text", "text": ANALYZE_PROMPT}], ANALYZE_SCHEMA)
    if data is None or data.get("category") not in CATEGORIES:
        return None
    try:
        level = int(max(1, min(5, int(data["severity"]))))
    except (TypeError, ValueError):
        level = 2
    return {
        "is_issue": bool(data["is_issue"]),
        "category": data["category"],
        "confidence": _unit(data["confidence"]),
        "severity": level,
        "coverage": _unit(data["coverage"], 0.3),
        "description": str(data.get("description") or "").strip()[:240],
        "reason": str(data.get("reason") or "").strip()[:300],
    }


def verify(before: Image.Image, after: Image.Image, label: str) -> dict | None:
    data = _ask(
        [
            _image_block(before),
            _image_block(after),
            {"type": "text", "text": VERIFY_PROMPT.format(label=label.lower())},
        ],
        VERIFY_SCHEMA,
    )
    if data is None:
        return None
    return {
        "same_place": bool(data["same_place"]),
        "same_place_likelihood": _unit(data["same_place_likelihood"]),
        "issue_still_present": bool(data["issue_still_present"]),
        "issue_confidence": _unit(data["issue_confidence"]),
        "notes": str(data.get("notes") or "").strip()[:500],
    }


SAME_PROBLEM_PROMPT = """The FIRST photo is a new citizen report of a {label}. \
The {n} photo(s) after it are open reports of a {label} already filed within {radius} metres.

Is the new photo showing the SAME physical problem as one of the existing reports (the same \
pothole, the same rubbish pile, the same lamp post, the same flooded stretch)? Look at fixed \
things like kerbs, walls, buildings, poles, trees and road markings. Angle, distance and lighting \
may differ, and the problem may have grown or shrunk a little.

match: the number of the matching existing photo (1 = the second photo overall, 2 = the third, \
and so on), or 0 if none of them is the same problem.
likelihood: 0 to 1, the chance that your match is right (if match is 0, the chance that the best \
candidate is the same problem)."""

SAME_PROBLEM_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "integer"},
        "likelihood": {"type": "number"},
    },
    "required": ["match", "likelihood"],
    "additionalProperties": False,
}


def same_problem(new: Image.Image, existing: list[Image.Image], label: str, radius_m: float) -> dict | None:
    """Duplicate check without CLIP: is the new photo one of the nearby open reports?
    Returns {"index": 0-based index into existing or -1, "likelihood": float}."""
    if not existing:
        return None
    content = [_image_block(new)] + [_image_block(im) for im in existing]
    content.append({
        "type": "text",
        "text": SAME_PROBLEM_PROMPT.format(label=label.lower(), n=len(existing), radius=int(radius_m)),
    })
    data = _ask(content, SAME_PROBLEM_SCHEMA)
    if data is None:
        return None
    try:
        match = int(data["match"])
    except (TypeError, ValueError):
        return None
    index = match - 1 if 1 <= match <= len(existing) else -1
    return {"index": index, "likelihood": _unit(data["likelihood"])}
