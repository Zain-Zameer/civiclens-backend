from __future__ import annotations

import logging

import hmac

from fastapi import APIRouter, Depends, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .. import geo, pipeline
from ..config import settings
from ..vision import VisionUnavailable
from ..db import service
from ..security import Caller, current_user, require_official

log = logging.getLogger("civiclens.api")
router = APIRouter(prefix="/api")

# Vercel caps a request body at 4.5 MB; the app shrinks photos to ~1 MB before upload
MAX_BYTES = 4 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}


async def _read_image(file: UploadFile) -> tuple[bytes, str]:
    ctype = (file.content_type or "image/jpeg").split(";")[0]
    if ctype not in ALLOWED:
        raise HTTPException(415, "Please upload a JPEG, PNG or WebP photo.")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The photo was empty.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "That photo is larger than 4 MB.")
    return data, ctype


# ---------------------------------------------------------------------------


@router.get("/health")
def health() -> dict:
    from .. import claude_vision
    from ..vision import _engine

    return {"ok": True, "model_loaded": _engine.ready, "vision": claude_vision.status()}


@router.get("/me")
def me(caller: Caller = Depends(current_user)) -> dict:
    return {"id": caller.id, "email": caller.email, "role": caller.role,
            "full_name": caller.full_name}


@router.get("/departments")
def departments() -> list[dict]:
    return service().table("departments").select("id,slug,name,icon").order("name").execute().data or []


@router.get("/geocode")
def geocode(lat: float, lng: float) -> dict:
    return {"address": geo.reverse_geocode(lat, lng)}


# --------------------------------------------------------------- citizen ----


@router.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    lat: float = Form(...),
    lng: float = Form(...),
    caller: Caller = Depends(current_user),
) -> dict:
    data, ctype = await _read_image(image)
    try:
        result = pipeline.analyze(data, lat, lng, ctype, caller.id)
    except VisionUnavailable as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception("analyze failed")
        raise HTTPException(500, f"The scan could not finish: {exc}")
    if not result["accepted"]:
        raise HTTPException(422, detail=result)
    # the signed analysis_id carries the embedding; the readable fields leave it out
    return {k: v for k, v in result.items() if k != "_embedding"}


class SubmitBody(BaseModel):
    analysis_id: str
    note: str | None = Field(default=None, max_length=500)
    address: str | None = Field(default=None, max_length=300)


@router.post("/reports")
def create_report(body: SubmitBody, caller: Caller = Depends(current_user)) -> dict:
    analysis = pipeline.recall(body.analysis_id, caller.id)
    if analysis is None:
        raise HTTPException(410, "That scan expired. Please take the photo again.")
    try:
        return pipeline.submit(analysis, caller.id, body.note, body.address)
    except pipeline.AlreadySubmitted:
        raise HTTPException(409, "This report was already sent.")
    except Exception as exc:
        log.exception("submit failed")
        raise HTTPException(500, f"The report could not be saved: {exc}")


@router.post("/complaints/{complaint_id}/upvote")
def upvote(complaint_id: str, caller: Caller = Depends(current_user)) -> dict:
    try:
        return pipeline.upvote(complaint_id, caller.id)
    except Exception as exc:
        log.exception("upvote failed")
        raise HTTPException(400, f"Could not add your voice: {exc}")


# -------------------------------------------------------------- official ----


class StatusBody(BaseModel):
    status: str
    note: str | None = Field(default=None, max_length=500)


@router.post("/complaints/{complaint_id}/status")
def set_status(
    complaint_id: str, body: StatusBody, caller: Caller = Depends(require_official)
) -> dict:
    if body.status not in pipeline.STATUS_FLOW:
        raise HTTPException(400, "Unknown status")
    try:
        return pipeline.set_status(complaint_id, body.status, body.note, caller.id)
    except Exception as exc:
        log.exception("status change failed")
        raise HTTPException(400, f"Could not update the status: {exc}")


@router.post("/complaints/{complaint_id}/verify")
async def verify(
    complaint_id: str,
    image: UploadFile = File(...),
    caller: Caller = Depends(require_official),
) -> dict:
    data, ctype = await _read_image(image)
    try:
        return pipeline.verify_resolution(complaint_id, data, ctype, caller.id)
    except pipeline.CaseClosed as exc:
        raise HTTPException(409, str(exc).capitalize() + ".")
    except VisionUnavailable as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception("verification failed")
        raise HTTPException(400, f"Verification could not run: {exc}")


class OverrideBody(BaseModel):
    note: str | None = Field(default=None, max_length=500)


@router.post("/complaints/{complaint_id}/force-resolve")
def force_resolve(
    complaint_id: str, body: OverrideBody, caller: Caller = Depends(require_official)
) -> dict:
    try:
        return pipeline.force_resolve(complaint_id, caller.id, body.note)
    except Exception as exc:
        raise HTTPException(400, f"Could not override: {exc}")


@router.post("/recompute")
def recompute(caller: Caller = Depends(require_official)) -> dict:
    return {"updated": pipeline.recompute_open_severities()}


@router.get("/cron/recompute", include_in_schema=False)
def cron_recompute(authorization: str | None = Header(default=None)) -> dict:
    """Called by Vercel Cron (vercel.json), which sends "Authorization: Bearer <CRON_SECRET>"."""
    secret = settings.cron_secret.strip()
    if not secret or not hmac.compare_digest(authorization or "", f"Bearer {secret}"):
        raise HTTPException(401, "Unauthorized")
    return {"updated": pipeline.recompute_open_severities()}
