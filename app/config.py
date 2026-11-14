import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(Path(__file__).resolve().parent.parent / ".env", ".env"), env_file_encoding="utf-8", extra="ignore")

    supabase_url: str
    supabase_secret_key: str
    supabase_publishable_key: str = ""

    official_email: str = "official@civiclens.app"
    official_password: str = "CivicLens@2026"
    official_name: str = "City Operations Officer"

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # extra origins as a regex (local network dev by default); add your Vercel domain via CORS_ORIGINS
    cors_origin_regex: str = r"http://(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+):\d+"

    clip_model: str = "openai/clip-vit-base-patch32"

    # Claude vision: judges photos when a key is set; CLIP is the fallback.
    # VISION_ENGINE=clip forces CLIP only (the e2e tests use this).
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-5-5"
    vision_engine: str = "auto"
    claude_daily_limit: int = 150
    # CLIP needs torch + transformers (requirements-clip.txt). On Vercel they are not
    # installed, so CLIP switches itself off and Claude does all the photo work.
    clip_enabled: bool = True
    overpass_url: str = "https://overpass-api.de/api/interpreter"
    nominatim_url: str = "https://nominatim.openstreetmap.org/reverse"
    user_agent: str = "CivicLens/1.0 (civic issue reporting app)"

    # duplicate clustering
    dup_radius_m: float = 50.0
    dup_similarity: float = 0.82

    # classification gates
    min_category_confidence: float = 0.35
    resolved_issue_max_confidence: float = 0.30
    same_place_min_similarity: float = 0.60

    enable_scheduler: bool = True
    recompute_interval_seconds: int = 3600
    # Vercel Cron calls GET /api/cron/recompute with "Authorization: Bearer <CRON_SECRET>"
    cron_secret: str = ""
    # create buckets + seed the official on startup (skipped on Vercel unless set to true)
    bootstrap_on_start: bool | None = None

    @property
    def serverless(self) -> bool:
        return bool(os.environ.get("VERCEL"))

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings = get_settings()
