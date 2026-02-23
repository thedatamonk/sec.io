"""POST /api/chat – main query endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from sec_llm.dependencies import get_pipeline, get_settings
from sec_llm.guardrails import check_scope, sanitize_input
from sec_llm.models import (
    AnalysisResponse,
    CompanyNotFoundError,
    ComputationError,
    FilingNotFoundError,
    LLMError,
    MetricNotAvailableError,
    UserQuery,
)

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


@router.post("/api/chat", response_model=AnalysisResponse)
async def chat(query: UserQuery, request: Request) -> AnalysisResponse:
    """Process a natural language financial query."""
    settings = get_settings()

    # Rate limiting
    client_ip = request.client.host if request.client else "unknown"
    _check_rate_limit(client_ip, settings.rate_limit_per_minute)

    # Sanitize and scope-check
    sanitized = sanitize_input(query.message)
    scope_error = check_scope(sanitized)
    if scope_error:
        raise HTTPException(status_code=422, detail=scope_error)

    pipeline = get_pipeline()

    try:
        return await pipeline.process(query)
    except CompanyNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except FilingNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except MetricNotAvailableError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except ComputationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
