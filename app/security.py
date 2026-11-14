"""Caller authentication.

The access token is validated against GoTrue (so we never need the project's
JWT secret), then the role is read from app_metadata with a fallback to the
profiles table. Results are cached briefly to keep request latency low.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status

from .db import anon, service

_CACHE: dict[str, tuple[float, "Caller"]] = {}
_TTL = 60.0


@dataclass
class Caller:
    id: str
    email: str
    role: str
    full_name: str = ""

    @property
    def is_official(self) -> bool:
        return self.role == "official"


def _bearer(request: Request) -> str:
    header = request.headers.get("authorization") or ""
    if not header.lower().startswith("bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    token = header.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing bearer token")
    return token


def current_user(request: Request) -> Caller:
    token = _bearer(request)
    now = time.time()
    hit = _CACHE.get(token)
    if hit and hit[0] > now:
        return hit[1]

    try:
        res = anon().auth.get_user(token)
    except Exception:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")

    user = getattr(res, "user", None)
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired session")

    app_meta = user.app_metadata or {}
    user_meta = user.user_metadata or {}
    role = str(app_meta.get("role") or "").strip()
    full_name = str(user_meta.get("full_name") or "")

    if role != "official":
        try:
            prof = (
                service()
                .table("profiles")
                .select("role,full_name")
                .eq("id", user.id)
                .limit(1)
                .execute()
                .data
            )
            if prof:
                role = prof[0].get("role") or "citizen"
                full_name = prof[0].get("full_name") or full_name
        except Exception:
            pass

    caller = Caller(id=user.id, email=user.email or "", role=role or "citizen", full_name=full_name)
    _CACHE[token] = (now + _TTL, caller)
    if len(_CACHE) > 512:
        for k in [k for k, v in _CACHE.items() if v[0] <= now]:
            _CACHE.pop(k, None)
    return caller


def require_official(caller: Caller = Depends(current_user)) -> Caller:
    if not caller.is_official:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Officials only")
    return caller
