"""OpenStreetMap lookups: nearby sensitive places (Overpass) + reverse geocode."""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed

import httpx

from .config import settings

log = logging.getLogger("civiclens.geo")

SENSITIVE_RADIUS_M = 200
# The scan must stay snappy, so the request path waits only a few seconds.
# When no mirror answers in time the complaint is created without the boost and
# a background refresh (longer budget) fills it in and re-scores the case.
OVERPASS_TIMEOUT = 13.0      # per mirror
OVERPASS_BUDGET = 4.5        # inside a user request
OVERPASS_BUDGET_BG = 13.0    # background refresh

_cache: dict[tuple[float, float], tuple[float, dict]] = {}
_CACHE_TTL = 60 * 60 * 24  # a day; the world does not move much

OVERPASS_MIRRORS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
)


def _endpoints() -> list[str]:
    """Primary endpoint from settings first, then the public mirrors."""
    seen, out = set(), []
    for url in (settings.overpass_url, *OVERPASS_MIRRORS):
        if url and url not in seen:
            seen.add(url)
            out.append(url)
    return out


_AMENITIES = "school|hospital|clinic|college|university|kindergarten|childcare"
_HIGHWAYS = "motorway|trunk|primary"


def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _query(lat: float, lng: float) -> str:
    r = SENSITIVE_RADIUS_M
    return f"""
[out:json][timeout:12];
(
  node(around:{r},{lat},{lng})["amenity"~"^({_AMENITIES})$"];
  way(around:{r},{lat},{lng})["amenity"~"^({_AMENITIES})$"];
  way(around:{r},{lat},{lng})["highway"~"^({_HIGHWAYS})$"];
);
out tags center 30;
""".strip()


def _ask_overpass(endpoint: str, query: str) -> dict | None:
    try:
        with httpx.Client(timeout=OVERPASS_TIMEOUT, headers={"User-Agent": settings.user_agent}) as client:
            res = client.post(endpoint, data={"data": query})
        if res.status_code != 200:
            log.info("overpass %s -> HTTP %s", endpoint, res.status_code)
            return None
        return res.json()
    except Exception as exc:
        log.info("overpass %s unavailable (%s)", endpoint, type(exc).__name__)
        return None


def _race_overpass(query: str, budget: float) -> dict | None:
    """Ask every mirror at once and take the first good answer.

    Public Overpass instances are frequently slow or rate-limited, so querying
    them one after another could add half a minute to a scan. Racing them keeps
    the whole lookup inside `budget` seconds.
    """
    endpoints = _endpoints()
    pool = ThreadPoolExecutor(max_workers=len(endpoints))
    futures = [pool.submit(_ask_overpass, e, query) for e in endpoints]
    try:
        for fut in as_completed(futures, timeout=budget):
            payload = fut.result()
            if payload is not None:
                return payload
    except FuturesTimeout:
        log.info("overpass budget of %ss exhausted", budget)
    finally:
        # never block the request on the slow mirrors still in flight
        pool.shutdown(wait=False, cancel_futures=True)
    return None


def nearby_sensitive(lat: float, lng: float, budget: float | None = None,
                     force: bool = False) -> dict:
    """Schools / hospitals / trunk roads within 200 m.

    Returns a jsonb-ready dict. Never raises — a dead Overpass just means
    no boost ("checked": false).
    """
    key = (round(lat, 4), round(lng, 4))
    now = time.time()
    hit = _cache.get(key)
    if hit and hit[0] > now and not force:
        return hit[1]

    result: dict = {"checked": False, "found": False, "items": [], "boost": 0.0,
                    "radius_m": SENSITIVE_RADIUS_M}
    payload = _race_overpass(_query(lat, lng), budget or OVERPASS_BUDGET)

    if payload is None:
        log.info("no Overpass mirror answered in time — continuing without the proximity boost")
        _cache[key] = (now + 300, result)  # short retry window
        return result

    items: list[dict] = []
    for el in payload.get("elements", []):
        tags = el.get("tags", {}) or {}
        center = el.get("center") or {}
        elat = el.get("lat", center.get("lat"))
        elng = el.get("lon", center.get("lon"))
        if elat is None or elng is None:
            continue
        dist = haversine_m(lat, lng, float(elat), float(elng))
        if dist > SENSITIVE_RADIUS_M:
            continue

        amenity = tags.get("amenity")
        highway = tags.get("highway")
        if amenity:
            kind = amenity
            name = tags.get("name") or amenity.replace("_", " ").title()
        elif highway:
            kind = "major_road"
            name = tags.get("name") or f"{highway.title()} road"
        else:
            continue
        items.append({"kind": kind, "name": name, "distance_m": int(round(dist))})

    # de-duplicate by (kind, name), keep nearest
    best: dict[tuple[str, str], dict] = {}
    for it in items:
        k = (it["kind"], it["name"])
        if k not in best or it["distance_m"] < best[k]["distance_m"]:
            best[k] = it
    items = sorted(best.values(), key=lambda i: i["distance_m"])[:8]

    result = {
        "checked": True,
        "found": bool(items),
        "items": items,
        "boost": 1.0 if items else 0.0,
        "radius_m": SENSITIVE_RADIUS_M,
    }
    _cache[key] = (now + _CACHE_TTL, result)
    return result


def reverse_geocode(lat: float, lng: float) -> str | None:
    try:
        with httpx.Client(timeout=10.0, headers={"User-Agent": settings.user_agent}) as client:
            res = client.get(
                settings.nominatim_url,
                params={"lat": lat, "lon": lng, "format": "jsonv2", "zoom": 18,
                        "addressdetails": 1},
            )
            res.raise_for_status()
            return res.json().get("display_name")
    except Exception as exc:
        log.info("nominatim unavailable: %s", exc)
        return None


def describe_sensitive(info: dict | None) -> str:
    if not info or not info.get("found"):
        return ""
    items = info.get("items") or []
    pretty = {
        "school": "a school",
        "kindergarten": "a kindergarten",
        "college": "a college",
        "university": "a university",
        "hospital": "a hospital",
        "clinic": "a clinic",
        "childcare": "a childcare centre",
        "major_road": "a main road",
    }
    first = items[0]
    label = pretty.get(first["kind"], first["kind"].replace("_", " "))
    return f"{label} ({first['name']}) is {first['distance_m']} m away"
