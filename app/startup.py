"""One-time bootstrap run on every backend boot (idempotent).

1. Create the public storage buckets if they are missing.
2. Seed the single official account through the Admin API with
   email_confirm=True and role="official" in app_metadata.
"""

from __future__ import annotations

import logging

from .config import settings
from .db import service

log = logging.getLogger("civiclens.startup")

BUCKETS = ("report-images", "resolution-images")


def ensure_buckets() -> None:
    client = service()
    try:
        existing = {b.id if hasattr(b, "id") else b["id"] for b in client.storage.list_buckets()}
    except Exception as exc:  # pragma: no cover - network
        log.warning("could not list storage buckets: %s", exc)
        existing = set()

    for name in BUCKETS:
        if name in existing:
            log.info("bucket ok: %s", name)
            continue
        try:
            client.storage.create_bucket(
                name,
                options={
                    "public": True,
                    "file_size_limit": 10 * 1024 * 1024,
                    "allowed_mime_types": ["image/jpeg", "image/png", "image/webp"],
                },
            )
            log.info("bucket created: %s", name)
        except Exception as exc:
            msg = str(exc)
            if "already exists" in msg.lower() or "duplicate" in msg.lower():
                log.info("bucket ok: %s", name)
            else:
                log.warning("bucket %s could not be created: %s", name, msg)


def _find_user_by_email(email: str):
    client = service()
    try:
        users = client.auth.admin.list_users()
    except Exception as exc:  # pragma: no cover - network
        log.warning("could not list auth users: %s", exc)
        return None
    for u in users or []:
        if (getattr(u, "email", "") or "").lower() == email.lower():
            return u
    return None


def seed_official() -> None:
    email = settings.official_email
    client = service()
    existing = _find_user_by_email(email)

    if existing is not None:
        log.info("official account already present (%s)", email)
        user_id = existing.id
    else:
        try:
            res = client.auth.admin.create_user(
                {
                    "email": email,
                    "password": settings.official_password,
                    "email_confirm": True,
                    "user_metadata": {"full_name": settings.official_name},
                    "app_metadata": {"role": "official"},
                }
            )
            user_id = res.user.id
            log.info("official account created (%s)", email)
        except Exception as exc:
            log.warning("official account could not be created: %s", exc)
            return

    # Make sure the profile row exists and carries the official role.
    try:
        client.table("profiles").upsert(
            {"id": user_id, "full_name": settings.official_name, "role": "official"},
            on_conflict="id",
        ).execute()
    except Exception as exc:
        log.warning("official profile upsert failed: %s", exc)


def bootstrap() -> None:
    ensure_buckets()
    seed_official()
