"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch

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
    def test_chat_out_of_scope_returns_422(self, client: TestClient):
        """Balance sheet queries should be rejected by the guardrail → 422."""
        # Simulate the guardrail raising its exception (with output_info attribute)
        class FakeOutput:
            output_info = "Balance sheet not supported."

        class FakeGuardrailResult:
            output = FakeOutput()
            guardrail = type("G", (), {"__class__": type("G", (), {"__name__": "scope_guardrail"})()})()

        from agents import InputGuardrailTripwireTriggered
        exc = InputGuardrailTripwireTriggered(FakeGuardrailResult())

        with patch("sec_llm.api.chat.run_conversation", new=AsyncMock(side_effect=exc)):
            response = client.post(
                "/api/chat",
                json={"message": "Show me Apple's balance sheet"},
            )
        assert response.status_code == 422

    def test_chat_returns_answer_and_citations(self, client: TestClient):
        """Successful query returns answer and citations fields."""
        with patch(
            "sec_llm.api.chat.run_conversation",
            new=AsyncMock(return_value=("Apple's revenue was $383.3 billion.", [])),
        ):
            response = client.post(
                "/api/chat",
                json={"message": "What was Apple's revenue in FY2023?"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "383" in data["answer"]
