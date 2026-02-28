"""FastAPI dependency injection helpers."""

from __future__ import annotations

from functools import lru_cache

from sec_llm.config import Settings
from sec_llm.sec.client import EdgarClient


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_edgar_client() -> EdgarClient:
    settings = get_settings()
    return EdgarClient(
        identity=settings.edgar_identity,
        cache_ttl=settings.sec_cache_ttl_seconds,
    )
