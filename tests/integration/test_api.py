"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sec_llm.config import Settings
from sec_llm.main import create_app


@pytest.fixture
def client() -> TestClient:
    settings = Settings(openai_api_key="sk-test", edgar_identity="Test test@example.com")
    app = create_app(settings)
    return TestClient(app)


class TestHealthEndpoint:
    def test_health(self, client: TestClient):
        response = client.get("/api/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChatEndpoint:
    def test_chat_rejects_long_message(self, client: TestClient):
        response = client.post(
            "/api/chat",
            json={"message": "x" * 2001},
        )
        assert response.status_code == 422  # Pydantic validation error
