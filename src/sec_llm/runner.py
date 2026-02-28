"""Conversational runner: wraps OpenAI Agents SDK Runner for one chat turn."""

from __future__ import annotations

import json
import logging
from typing import Any

from agents import OpenAIProvider, RunConfig, Runner

from sec_llm.agent import sec_agent

logger = logging.getLogger(__name__)


async def run_conversation(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]]]:
    """Run one turn of the conversation and return the agent's answer plus citations.

    Args:
        message: The sanitized user message for this turn.
        history: Prior conversation turns as [{role, content}, ...] dicts.

    Returns:
        (answer, citations) where citations is a list of dicts extracted from
        any get_income_statement tool results in this run.
    """
    from sec_llm.dependencies import get_settings
    settings = get_settings()
    run_config = RunConfig(model_provider=OpenAIProvider(api_key=settings.openai_api_key))

    messages = history + [{"role": "user", "content": message}]
    result = await Runner.run(sec_agent, messages, run_config=run_config)
    answer = result.final_output or ""
    citations = _extract_citations(result)
    return answer, citations


def _extract_citations(result: Any) -> list[dict[str, Any]]:
    """Extract source citations from get_income_statement tool call results."""
    citations: list[dict[str, Any]] = []

    for item in result.new_items:
        if getattr(item, "type", None) != "tool_call_output_item":
            continue

        content = getattr(item, "output", None)
        if not content:
            continue

        # Content may be a string (JSON) or already a dict
        data: dict[str, Any] | None = None
        if isinstance(content, str):
            try:
                data = json.loads(content)
            except (json.JSONDecodeError, ValueError):
                continue
        elif isinstance(content, dict):
            data = content

        if data is None:
            continue

        metadata = data.get("metadata")
        if not metadata:
            continue

        citations.append({
            "ticker": metadata.get("ticker", ""),
            "filing_type": metadata.get("filing_type", ""),
            "fiscal_period": _fiscal_period_label(metadata),
            "filing_date": str(metadata.get("filing_date") or ""),
        })

    return citations


def _fiscal_period_label(metadata: dict[str, Any]) -> str:
    quarter = metadata.get("quarter")
    fiscal_year = metadata.get("fiscal_year", "")
    if quarter:
        return f"Q{quarter} FY{fiscal_year}"
    return f"FY{fiscal_year}"
