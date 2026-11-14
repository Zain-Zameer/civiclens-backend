"""Severity escalation and the officials' priority score.

final = visual severity
        + 0.5 (3-4 reports)  | + 1.0 (5+ reports)
        + 0.5 (open > 7 days) | + 1.0 (open > 14 days)
        + 1.0 (within 200 m of a school / hospital / trunk road)
capped at 5.

priority_score is the same number before rounding, so the officials' queue
can sort finely.
"""

from __future__ import annotations

from datetime import datetime, timezone

from .vision import CATEGORY_LABELS


def _parse(ts: str | datetime | None) -> datetime | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    txt = ts.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def count_boost(report_count: int) -> float:
    if report_count >= 5:
        return 1.0
    if report_count >= 3:
        return 0.5
    return 0.0


def age_boost(created_at: str | datetime | None, now: datetime | None = None) -> float:
    dt = _parse(created_at)
    if dt is None:
        return 0.0
    now = now or datetime.now(timezone.utc)
    days = (now - dt).total_seconds() / 86400.0
    if days > 14:
        return 1.0
    if days > 7:
        return 0.5
    return 0.0


def compute(
    visual_severity: float,
    report_count: int,
    created_at: str | datetime | None,
    sensitive: dict | None,
    now: datetime | None = None,
) -> dict:
    cb = count_boost(report_count)
    ab = age_boost(created_at, now)
    sb = 1.0 if (sensitive or {}).get("found") else 0.0

    raw = float(visual_severity) + cb + ab + sb
    priority = max(1.0, min(5.0, raw))
    final = int(max(1, min(5, round(priority))))
    return {
        "final_severity": final,
        "priority_score": round(priority, 3),
        "boosts": {"reports": cb, "age": ab, "sensitive": sb},
    }


SEVERITY_WORDS = {1: "minor", 2: "low", 3: "moderate", 4: "serious", 5: "critical"}


def explain(
    category: str,
    visual_severity: int,
    coverage: float,
    report_count: int,
    sensitive: dict | None,
    boosts: dict,
    final_severity: int,
    description: str | None = None,
) -> str:
    label = CATEGORY_LABELS.get(category, category).lower()
    if description:
        bits = [_sentence(description)]
    else:
        bits = [
            f"The photo shows a {SEVERITY_WORDS[visual_severity]} {label}, taking up roughly "
            f"{round(coverage * 100)}% of the frame."
        ]
    if boosts.get("reports"):
        bits.append(f"{report_count} people have reported it, which raises the priority.")
    if boosts.get("age"):
        bits.append("It has been open for a long time without a fix.")
    if boosts.get("sensitive"):
        from .geo import describe_sensitive

        where = describe_sensitive(sensitive)
        bits.append(f"It sits in a sensitive spot — {where}." if where else
                    "It sits close to a school, hospital or main road.")
    bits.append(f"Overall severity: {final_severity} of 5 ({SEVERITY_WORDS[final_severity]}).")
    return " ".join(bits)


def summary_text(
    category: str,
    visual_severity: int,
    coverage: float,
    address: str | None,
    sensitive: dict | None,
    report_count: int,
    description: str | None = None,
) -> str:
    """A short inspector's note."""
    label = CATEGORY_LABELS.get(category, category).lower()
    size = {
        1: "a very small",
        2: "a small",
        3: "a moderate",
        4: "a large",
        5: "a severe",
    }[visual_severity]
    where = f" on {address.split(',')[0]}" if address else ""
    if description:
        note = f"Inspection note: {_sentence(description)}"
    else:
        note = f"Inspection note: {size} {label}{where}, covering about {round(coverage * 100)}% of the view."
    if (sensitive or {}).get("found"):
        from .geo import describe_sensitive

        d = describe_sensitive(sensitive)
        if d:
            note += f" {d.capitalize()}, so public risk is raised."
    if report_count > 1:
        note += f" Confirmed by {report_count} separate reports."
    return note


def _sentence(text: str) -> str:
    text = text.strip()
    if text and text[-1] not in ".!?":
        text += "."
    return text[:1].upper() + text[1:]


def title_text(category: str, address: str | None) -> str:
    label = CATEGORY_LABELS.get(category, category)
    if address:
        street = address.split(",")[0].strip()
        if street:
            return f"{label} — {street}"
    return label
