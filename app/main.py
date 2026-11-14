from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routes.api import router as api_router
from .startup import bootstrap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("civiclens")


async def _severity_loop() -> None:
    from .pipeline import recompute_open_severities

    while True:
        try:
            await asyncio.sleep(settings.recompute_interval_seconds)
            await asyncio.to_thread(recompute_open_severities)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover
            log.warning("severity recompute failed: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .vision import clip_available, load_model

    # On Vercel every cold start runs this, so the one-time Supabase setup is skipped
    # there (it already ran locally) unless BOOTSTRAP_ON_START=true.
    run_bootstrap = settings.bootstrap_on_start
    if run_bootstrap is None:
        run_bootstrap = not settings.serverless
    if run_bootstrap:
        await asyncio.to_thread(bootstrap)
    if clip_available():
        await asyncio.to_thread(load_model)

    # serverless functions freeze between requests; Vercel Cron drives the recompute there
    task = None
    if settings.enable_scheduler and not settings.serverless:
        task = asyncio.create_task(_severity_loop())
    from . import claude_vision

    v = claude_vision.status()
    log.info(
        "CivicLens backend ready — photo judging: %s",
        (f"Claude {v['model']} ({'CLIP fallback' if clip_available() else 'no CLIP'}, "
         f"{v['daily_limit']} calls/day)") if v["engine"] == "claude" else "CLIP only",
    )
    try:
        yield
    finally:
        if task:
            task.cancel()


app = FastAPI(title="CivicLens API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_list,
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/")
def root() -> dict:
    return {"service": "CivicLens API", "docs": "/docs"}
