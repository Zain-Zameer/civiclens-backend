"""Quick smoke test for the CivicLens CV pipeline.

    .venv/Scripts/python test_pipeline.py                 # synthetic images
    .venv/Scripts/python test_pipeline.py photo1.jpg ...  # your own photos

Checks: model loads, categories classify, severity ladders produce sensible
levels, embeddings cluster duplicates, Overpass lookup degrades gracefully,
and the resolution verifier accepts a cleaned-up "after" shot.
"""

from __future__ import annotations

import io
import random
import sys

from PIL import Image, ImageDraw, ImageFilter

from app import geo, severity as sev, vision
from app.config import settings

# this script inspects CLIP's own scores, so never spend Claude credit here
settings.vision_engine = "clip"


def _jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def synth_road(pothole: str | None = "big") -> Image.Image:
    """A crude synthetic road scene — enough to exercise the pipeline."""
    random.seed(7)
    img = Image.new("RGB", (640, 480), (122, 124, 128))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 170], fill=(168, 186, 206))          # sky
    d.rectangle([0, 170, 640, 480], fill=(96, 98, 102))          # asphalt
    for y in range(190, 480, 46):                                 # lane markings
        d.rectangle([300, y, 340, y + 22], fill=(226, 226, 220))
    for _ in range(900):                                          # tarmac grain
        x, y = random.randint(0, 639), random.randint(170, 479)
        g = random.randint(80, 120)
        d.point((x, y), fill=(g, g, g + 2))
    if pothole == "big":
        d.ellipse([170, 300, 470, 430], fill=(32, 30, 30))
        d.ellipse([190, 312, 450, 414], fill=(18, 17, 17))
        d.ellipse([240, 340, 400, 392], fill=(8, 8, 9))
    elif pothole == "small":
        d.ellipse([290, 372, 356, 404], fill=(34, 32, 32))
        d.ellipse([300, 378, 346, 398], fill=(16, 16, 16))
    return img.filter(ImageFilter.GaussianBlur(0.6))


def synth_garbage() -> Image.Image:
    random.seed(3)
    img = Image.new("RGB", (640, 480), (118, 120, 124))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 640, 150], fill=(150, 160, 172))
    d.rectangle([0, 150, 640, 480], fill=(104, 104, 108))
    palette = [(28, 28, 32), (46, 46, 52), (168, 160, 140), (92, 120, 76), (190, 90, 70)]
    for _ in range(140):
        x, y = random.randint(40, 600), random.randint(230, 450)
        w, h = random.randint(28, 90), random.randint(22, 70)
        d.ellipse([x, y, x + w, y + h], fill=random.choice(palette))
    return img.filter(ImageFilter.GaussianBlur(0.8))


def line(char: str = "-") -> None:
    print(char * 74)


def report(name: str, data: bytes) -> dict:
    res = vision.analyze_bytes(data)
    print(f"\n{name}")
    line()
    if not res["accepted"]:
        print(f"  REJECTED  (top confidence {res['confidence']:.3f})")
        print(f"  reason: {res['reason']}")
        print("  scores:", {k: round(v, 3) for k, v in res["scores"].items()})
        return res
    print(f"  category        : {res['category']}  ({res['confidence']:.1%} confident)")
    print(f"  department      : {res['department_slug']}")
    print(f"  visual severity : {res['visual_severity']}  (expected {res['severity_expected']:.2f})")
    print(f"  coverage        : {res['coverage']:.0%}")
    print("  ladder          :", [round(p, 3) for p in res["severity_distribution"]])
    print("  scores          :", {k: round(v, 3) for k, v in sorted(res["scores"].items(), key=lambda kv: -kv[1])[:3]})
    return res


def main(paths: list[str]) -> int:
    print("loading CLIP (first run downloads ~600 MB) ...")
    vision.load_model()
    print("model ready\n")

    results: dict[str, dict] = {}

    if paths:
        for p in paths:
            with open(p, "rb") as fh:
                results[p] = report(p, fh.read())
    else:
        results["big pothole"] = report("synthetic: large pothole", _jpeg(synth_road("big")))
        results["small pothole"] = report("synthetic: small pothole", _jpeg(synth_road("small")))
        results["garbage"] = report("synthetic: garbage pile", _jpeg(synth_garbage()))
        results["clean road"] = report("synthetic: clean road (should be rejected)", _jpeg(synth_road(None)))

    # ---------------------------------------------------------- duplicates
    accepted = [(k, v) for k, v in results.items() if v.get("accepted")]
    if len(accepted) >= 2:
        print("\nembedding similarity matrix")
        line()
        names = [k for k, _ in accepted]
        print("            " + "".join(f"{n[:10]:>12}" for n in names))
        for na, va in accepted:
            row = "".join(f"{vision.cosine(va['embedding'], vb['embedding']):>12.3f}" for _, vb in accepted)
            print(f"{na[:11]:<12}{row}")
        print(f"\n  duplicate threshold = {vision.settings.dup_similarity} within "
              f"{vision.settings.dup_radius_m:.0f} m and the same category")

    # ------------------------------------------------------------ severity
    print("\nseverity escalation")
    line()
    base = 3
    for n, age, sensitive in [
        (1, None, None),
        (4, None, None),
        (6, None, None),
        (1, "2026-10-01T00:00:00Z", None),
        (6, "2026-10-01T00:00:00Z", {"found": True}),
    ]:
        out = sev.compute(base, n, age, sensitive)
        print(f"  visual {base}  reports {n:<2} age {str(age)[:10]:<10} sensitive {bool(sensitive and sensitive.get('found')):<5}"
              f" -> final {out['final_severity']}  priority {out['priority_score']:.2f}  {out['boosts']}")

    # ------------------------------------------------------------- overpass
    print("\nOverpass proximity lookup (Connaught Place, New Delhi)")
    line()
    info = geo.nearby_sensitive(28.6315, 77.2167)
    print(f"  checked={info['checked']}  found={info['found']}  boost={info['boost']}")
    for item in info["items"][:5]:
        print(f"    - {item['kind']:<12} {item['name'][:34]:<36} {item['distance_m']:>4} m")
    if not info["checked"]:
        print("    (Overpass unreachable — the pipeline continues without the boost, as designed)")

    # --------------------------------------------------------- verification
    first = next((v for v in results.values() if v.get("accepted")), None)
    if first:
        print("\nresolution verification")
        line()
        clean = _jpeg(synth_road(None))
        out = vision.verify_resolution(clean, first["embedding"], first["category"])
        print(f"  after = clean road   verified={out['verified']}  score={out['verification_score']:.2f}")
        print(f"    issue_conf={out['issue_confidence']:.3f}  same_place={out['same_place_similarity']:.3f}")
        print(f"    {out['ai_notes']}")
        same = _jpeg(synth_road("big"))
        out2 = vision.verify_resolution(same, first["embedding"], first["category"])
        print(f"\n  after = same pothole verified={out2['verified']}  score={out2['verification_score']:.2f}")
        print(f"    {out2['ai_notes']}")

    print("\nall pipeline stages ran.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
