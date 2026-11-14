"""Supabase clients.

`service()` uses the secret key and bypasses RLS — it must never be exposed
to the browser. `anon()` uses the publishable key and is only used to validate
a caller's access token against GoTrue.
"""

from __future__ import annotations

from functools import lru_cache

from supabase import Client, create_client

from .config import settings


@lru_cache
def service() -> Client:
    return create_client(settings.supabase_url, settings.supabase_secret_key)


@lru_cache
def anon() -> Client:
    key = settings.supabase_publishable_key or settings.supabase_secret_key
    return create_client(settings.supabase_url, key)


def department_map() -> dict[str, str]:
    """slug -> department uuid"""
    rows = service().table("departments").select("id,slug").execute().data or []
    return {r["slug"]: r["id"] for r in rows}
