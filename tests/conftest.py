"""Shared test fixtures."""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("SEC_LLM_OPENAI_API_KEY", "sk-test")

from sec_llm.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(openai_api_key="test-key", edgar_identity="Test User test@example.com")
