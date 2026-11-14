"""Vercel entrypoint. Vercel looks for a FastAPI instance named `app` in index.py.

Locally, keep using:  python -m uvicorn app.main:app --reload --port 8000
"""
from app.main import app  # noqa: F401
