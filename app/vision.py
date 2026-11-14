"""Computer vision for CivicLens: Claude judges, CLIP fingerprints.

When ANTHROPIC_API_KEY is set, Claude (Sonnet 5.5, see claude_vision.py) decides
the category, severity and fix check and writes a plain-English description.
CLIP (openai/clip-vit-base-patch32, local CPU) always runs too, because its
embeddings power duplicate merging, and it takes over completely whenever
Claude is off or unavailable.

  * zero-shot category classification over civic-issue prompt sets
  * category-specific severity ladders (expected value over 5 levels)
  * 512-d image embeddings used for duplicate clustering and
    before/after location matching
"""

from __future__ import annotations

import importlib.util
import io
import logging
import threading
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageOps

from . import claude_vision
from .config import settings

log = logging.getLogger("civiclens.vision")

# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------

CATEGORY_PROMPTS: dict[str, list[str]] = {
    "pothole": [
        "a photo of a pothole in the road",
        "a street photo of broken asphalt with a hole in it",
        "a damaged road surface with a crater",
        "a close up of cracked and crumbling road pavement",
    ],
    "garbage": [
        "a photo of a pile of garbage on the street",
        "a heap of trash bags dumped on the roadside",
        "an overflowing public rubbish bin with litter around it",
        "scattered household waste on a city pavement",
    ],
    "streetlight": [
        "a photo of a broken street light",
        "a damaged street lamp post leaning over the road",
        "a street lamp with a smashed or missing light fitting",
        "a dark street at night with a street light that is not working",
    ],
    "waterlogging": [
        "a flooded street with standing rain water",
        "a waterlogged road with a large puddle covering the lane",
        "a city street under water after heavy rain",
        "a submerged road where vehicles drive through water",
    ],
    "sewage": [
        "a photo of sewage water overflowing onto the street",
        "an open manhole with dirty black waste water spilling out",
        "a blocked drain with foul sewage flooding the pavement",
        "a leaking sewer pipe spilling waste water on the road",
    ],
    "none": [
        "a photo of a clean tidy street with nothing wrong",
        "a smooth well maintained asphalt road in good condition",
        "an empty clean street with no damage and no rubbish",
        "a freshly paved road surface with no cracks or holes",
        "a selfie of a person",
        "a photo of a plate of food",
        "an indoor photo of a living room",
        "a photo of a pet cat or dog",
        "a screenshot of a computer screen",
        "a photo of the sky or clouds",
        "a product photo on a white background",
        "a photo of a document or piece of paper",
    ],
}

ISSUE_CATEGORIES = [c for c in CATEGORY_PROMPTS if c != "none"]

CATEGORY_LABELS = {
    "pothole": "Pothole / road damage",
    "garbage": "Garbage pile",
    "streetlight": "Broken streetlight",
    "waterlogging": "Waterlogging / flooding",
    "sewage": "Sewage overflow",
}

DEPARTMENT_FOR_CATEGORY = {
    "pothole": "roads",
    "garbage": "sanitation",
    "streetlight": "lighting",
    "waterlogging": "water",
    "sewage": "water",
}

# Five-step ladders, level 1 .. level 5
SEVERITY_LADDERS: dict[str, list[str]] = {
    "pothole": [
        "a tiny shallow crack in the road surface",
        "a small shallow pothole in the road",
        "a medium sized pothole deep enough to jolt a car",
        "a large deep pothole covering half of the road lane",
        "a huge deep pothole covering most of the road, impossible to drive around",
    ],
    "garbage": [
        "a small bag of litter on the pavement",
        "a small pile of garbage bags at the kerb",
        "a medium garbage pile taking up the pavement",
        "a large garbage heap spilling into the road",
        "a massive garbage dump spilling across the whole road",
    ],
    "streetlight": [
        "a single street lamp with a minor fault, the street is still lit",
        "one broken street light on an otherwise lit street",
        "a broken street light leaving part of the street dark",
        "several broken street lights leaving the street mostly dark",
        "an entire street in complete darkness with all street lights broken",
    ],
    "waterlogging": [
        "a small shallow puddle at the edge of the road",
        "a shallow patch of water covering part of one lane",
        "standing water covering a whole lane of the road",
        "deep water covering the full width of the street",
        "severe flooding submerging the entire street and vehicles",
    ],
    "sewage": [
        "a small damp patch of waste water near a drain",
        "a small amount of sewage water seeping onto the pavement",
        "sewage water spreading across the pavement and kerb",
        "sewage flooding a large part of the road",
        "raw sewage flooding the entire street",
    ],
}

COVERAGE_PROMPTS = [
    "the problem covers a very small part of the picture",
    "the problem covers a small part of the picture",
    "the problem covers about half of the picture",
    "the problem covers most of the picture",
    "the problem covers the entire picture",
]


# ---------------------------------------------------------------------------
# Model holder
# ---------------------------------------------------------------------------


@dataclass
class _Engine:
    model: object | None = None
    processor: object | None = None
    category_protos: dict[str, np.ndarray] = field(default_factory=dict)  # (n_prompts, 512)
    ladder_protos: dict[str, np.ndarray] = field(default_factory=dict)
    coverage_protos: np.ndarray | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def ready(self) -> bool:
        return self.model is not None


_engine = _Engine()


class VisionUnavailable(Exception):
    """Neither Claude nor CLIP could judge the photo (e.g. on Vercel with Claude down)."""


def clip_available() -> bool:
    """CLIP runs only where torch + transformers are installed (not on Vercel)."""
    return bool(
        settings.clip_enabled
        and importlib.util.find_spec("torch")
        and importlib.util.find_spec("transformers")
    )


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.clip(n, 1e-8, None)


def _encode_text(prompts: list[str]) -> np.ndarray:
    import torch

    inputs = _engine.processor(text=prompts, return_tensors="pt", padding=True)  # type: ignore[operator]
    with torch.no_grad():
        feats = _engine.model.get_text_features(**inputs)  # type: ignore[union-attr]
    return _normalize(feats.cpu().numpy().astype(np.float32))


def load_model() -> None:
    """Load CLIP and pre-compute every text prototype. Called once at startup."""
    if _engine.ready:
        return
    with _engine.lock:
        if _engine.ready:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor

        torch.set_num_threads(max(1, (torch.get_num_threads() or 4)))
        log.info("loading CLIP model %s ...", settings.clip_model)
        _engine.processor = CLIPProcessor.from_pretrained(settings.clip_model)
        model = CLIPModel.from_pretrained(settings.clip_model)
        model.eval()
        _engine.model = model

        # one embedding per prompt; a class scores as its BEST matching prompt,
        # which keeps the heterogeneous "none" class (selfie, food, clean road,
        # screenshot ...) from being diluted by averaging.
        for cat, prompts in CATEGORY_PROMPTS.items():
            _engine.category_protos[cat] = _encode_text(prompts)

        for cat, ladder in SEVERITY_LADDERS.items():
            _engine.ladder_protos[cat] = _encode_text(ladder)  # (5, 512)

        _engine.coverage_protos = _encode_text(COVERAGE_PROMPTS)
        log.info("CLIP ready")


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    z = x * temperature
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def open_image(data: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def encode_image(img: Image.Image) -> np.ndarray:
    """512-d unit-norm CLIP image embedding."""
    load_model()
    import torch

    inputs = _engine.processor(images=img, return_tensors="pt")  # type: ignore[operator]
    with torch.no_grad():
        feats = _engine.model.get_image_features(**inputs)  # type: ignore[union-attr]
    return _normalize(feats.cpu().numpy().astype(np.float32))[0]


def classify(emb: np.ndarray) -> dict:
    """Zero-shot category scores from an image embedding."""
    load_model()
    cats = list(CATEGORY_PROMPTS.keys())
    sims = np.array(
        [float(np.max(_engine.category_protos[c] @ emb)) for c in cats], dtype=np.float32
    )
    probs = _softmax(sims, temperature=100.0)
    scores = {c: round(float(p), 4) for c, p in zip(cats, probs)}
    top = max(scores, key=scores.get)  # type: ignore[arg-type]
    issue_only = {c: scores[c] for c in ISSUE_CATEGORIES}
    best_issue = max(issue_only, key=issue_only.get)  # type: ignore[arg-type]
    return {
        "scores": scores,
        "top": top,
        "top_confidence": scores[top],
        "best_issue": best_issue,
        "best_issue_confidence": issue_only[best_issue],
        "none_confidence": scores["none"],
    }


def coverage(emb: np.ndarray) -> float:
    """0..1 estimate of how much of the frame the problem occupies."""
    load_model()
    sims = _engine.coverage_protos @ emb  # type: ignore[operator]
    probs = _softmax(np.asarray(sims, dtype=np.float32), temperature=50.0)
    expected = float((probs * np.array([0.1, 0.3, 0.5, 0.75, 0.95])).sum())
    return round(expected, 3)


def severity(emb: np.ndarray, category: str) -> dict:
    """Category-specific severity ladder → softmax-weighted expected value."""
    load_model()
    ladder = _engine.ladder_protos.get(category)
    if ladder is None:
        return {"level": 2, "expected": 2.0, "distribution": []}
    sims = np.asarray(ladder @ emb, dtype=np.float32)
    probs = _softmax(sims, temperature=50.0)
    expected = float((probs * np.array([1, 2, 3, 4, 5], dtype=np.float32)).sum())
    level = int(max(1, min(5, round(expected))))
    return {
        "level": level,
        "expected": round(expected, 3),
        "distribution": [round(float(p), 4) for p in probs],
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_normalize(a), _normalize(b)))


# ---------------------------------------------------------------------------
# High level entry points
# ---------------------------------------------------------------------------


def analyze_bytes(data: bytes) -> dict:
    """Classify + score an uploaded photo. Claude first, CLIP as the fallback."""
    img = open_image(data)
    clip = clip_available()
    emb = encode_image(img) if clip else None  # CLIP fingerprint for duplicate merging
    ai = claude_vision.analyze(img)
    if ai is None and not clip:
        raise VisionUnavailable(
            "The photo check is busy right now. Please try again in a minute."
        )

    if ai is not None:
        accepted = ai["is_issue"] and ai["confidence"] >= settings.min_category_confidence
        category = ai["category"] if accepted else None
        # keep the same score shape the app already stores, centred on Claude's pick
        scores = {c: (round(ai["confidence"], 4) if c == ai["category"] else 0.0) for c in CATEGORY_PROMPTS}
        scores["none"] = round(1 - ai["confidence"], 4) if not ai["is_issue"] else 0.0
        out: dict = {
            "engine": "claude",
            "accepted": accepted,
            "category": category,
            "category_label": CATEGORY_LABELS.get(category or "", ""),
            "confidence": ai["confidence"],
            "scores": scores,
            "none_confidence": scores["none"],
            "embedding": emb,
            "width": img.width,
            "height": img.height,
            "description": ai["description"],
        }
        if not accepted:
            out["reason"] = ai["reason"] or (
                "We could not find a civic issue in this photo. Try a clear, well-lit shot of "
                "the pothole, rubbish, streetlight or water, taken a few steps back."
            )
            return out
        out.update(
            {
                "visual_severity": ai["severity"],
                "severity_expected": float(ai["severity"]),
                "severity_distribution": [1.0 if n == ai["severity"] else 0.0 for n in range(1, 6)],
                "coverage": round(ai["coverage"], 3),
                "department_slug": DEPARTMENT_FOR_CATEGORY[category],  # type: ignore[index]
            }
        )
        return out

    cls = classify(emb)  # type: ignore[arg-type]
    accepted = cls["top"] != "none" and cls["top_confidence"] >= settings.min_category_confidence
    category = cls["best_issue"] if accepted else None

    out = {
        "engine": "clip",
        "accepted": accepted,
        "category": category,
        "category_label": CATEGORY_LABELS.get(category or "", ""),
        "confidence": cls["top_confidence"] if accepted else cls["best_issue_confidence"],
        "scores": cls["scores"],
        "none_confidence": cls["none_confidence"],
        "embedding": emb,
        "width": img.width,
        "height": img.height,
        "description": None,
    }

    if not accepted:
        out["reason"] = (
            "We could not find a civic issue in this photo. Try a clear, well-lit shot of "
            "the pothole, rubbish, streetlight or water, taken a few steps back."
        )
        return out

    sev = severity(emb, category)  # type: ignore[arg-type]
    cov = coverage(emb)
    out.update(
        {
            "visual_severity": sev["level"],
            "severity_expected": sev["expected"],
            "severity_distribution": sev["distribution"],
            "coverage": cov,
            "department_slug": DEPARTMENT_FOR_CATEGORY[category],  # type: ignore[index]
        }
    )
    return out


def verify_resolution(
    after_bytes: bytes,
    before_embedding: np.ndarray | None,
    category: str,
    before_bytes: bytes | None = None,
) -> dict:
    """Check an official's 'after' photo: same place, issue gone.
    Claude compares the two photos when it can; CLIP is the fallback."""
    img = open_image(after_bytes)
    clip = clip_available()
    emb = encode_image(img) if clip else None
    label = CATEGORY_LABELS.get(category, category)

    ai = None
    if before_bytes:
        try:
            before_img = open_image(before_bytes)
        except Exception:
            before_img = None
        if before_img is not None:
            ai = claude_vision.verify(before_img, img, label)
    if ai is not None:
        verified = bool(ai["same_place"] and not ai["issue_still_present"])
        score = round(0.65 * (1 - ai["issue_confidence"]) + 0.35 * ai["same_place_likelihood"], 3)
        return {
            "engine": "claude",
            "verified": verified,
            "verification_score": score,
            "issue_confidence": round(ai["issue_confidence"], 4),
            "same_place_similarity": round(ai["same_place_likelihood"], 4),
            "ai_notes": ai["notes"] or ("Fix confirmed by photo check." if verified else "Fix not confirmed by photo check."),
            "embedding": emb,
        }

    if not clip or before_embedding is None:
        raise VisionUnavailable(
            "The photo check is busy right now. Please try again in a minute, "
            "or mark it fixed anyway."
        )

    cls = classify(emb)  # type: ignore[arg-type]

    issue_conf = float(cls["scores"].get(category, 0.0))
    same_place = cosine(emb, before_embedding)

    issue_cleared = issue_conf < settings.resolved_issue_max_confidence
    place_ok = same_place >= settings.same_place_min_similarity

    # verification_score blends "issue gone" with "same location"
    cleared_component = max(0.0, min(1.0, (settings.resolved_issue_max_confidence * 2 - issue_conf) / (settings.resolved_issue_max_confidence * 2)))
    place_component = max(0.0, min(1.0, (same_place - 0.3) / 0.6))
    score = round(0.65 * cleared_component + 0.35 * place_component, 3)

    verified = bool(issue_cleared and place_ok)

    if verified:
        notes = (
            f"After-photo checked: no {label.lower()} detected any more "
            f"({issue_conf:.0%} confidence it is still there) and the scene matches the original "
            f"location ({same_place:.0%} similarity). Marked as genuinely resolved."
        )
    elif not place_ok:
        notes = (
            f"The after-photo does not look like the same place ({same_place:.0%} similarity, "
            f"we need at least {settings.same_place_min_similarity:.0%}). "
            "Please photograph the issue from the same spot as the original report."
        )
    else:
        notes = (
            f"The {label.lower()} is still visible in the after-photo "
            f"({issue_conf:.0%} confidence, needs to drop below "
            f"{settings.resolved_issue_max_confidence:.0%}). Verification failed."
        )

    return {
        "engine": "clip",
        "verified": verified,
        "verification_score": score,
        "issue_confidence": round(issue_conf, 4),
        "same_place_similarity": round(same_place, 4),
        "ai_notes": notes,
        "embedding": emb,
    }
