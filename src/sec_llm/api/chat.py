"""POST /api/chat – main query endpoint."""

from __future__ import annotations

import time
from typing import Any

from agents import InputGuardrailTripwireTriggered
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from sec_llm.dependencies import get_settings
from sec_llm.guardrails import sanitize_input
from sec_llm.models import CompanyNotFoundError, ComputationError, FilingNotFoundError
from sec_llm.runner import run_conversation

router = APIRouter()

# Simple per-IP rate limiting state
_rate_state: dict[str, list[float]] = {}


def _check_rate_limit(ip: str, max_per_minute: int) -> None:
    now = time.monotonic()
    window_start = now - 60.0
    hits = _rate_state.get(ip, [])
    hits = [t for t in hits if t > window_start]
    if len(hits) >= max_per_minute:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again shortly.")
    hits.append(now)
    _rate_state[ip] = hits


class ChatRequest(BaseModel):
    message: str
    conversation_history: list[dict[str, Any]] = []


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]] = []


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, req: Request) -> ChatResponse:
    """Process a natural language financial query."""
    settings = get_settings()

    # Rate limiting
    client_ip = req.client.host if req.client else "unknown"
    _check_rate_limit(client_ip, settings.rate_limit_per_minute)

    sanitized = sanitize_input(request.message)

    try:
        answer, citations = await run_conversation(sanitized, request.conversation_history)
        return ChatResponse(answer=answer, citations=citations)
    except InputGuardrailTripwireTriggered as exc:
        detail = str(exc.guardrail_result.output.output_info) if exc.guardrail_result else str(exc)
        raise HTTPException(status_code=422, detail=detail)
    except CompanyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FilingNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ComputationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
