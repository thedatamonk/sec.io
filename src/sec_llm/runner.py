"""Conversational runner: wraps OpenAI Agents SDK Runner for one chat turn."""

from __future__ import annotations

import logging
from datetime import date as _date
from typing import Any

from agents import OpenAIProvider, RunConfig, Runner

from sec_llm.agent import sec_agent
from sec_llm.agents.scratchpad import BossResponse, SingleAgentResponse

logger = logging.getLogger(__name__)


def _year_context() -> str:
    today = _date.today()
    yr = today.year
    return (
        f"Today's date is {today.isoformat()}. "
        f'"Last year" = FY{yr - 1}. "This year" = FY{yr}. '
        f'"Most recent" annual data = FY{yr - 1}.'
    )


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

    messages = [{"role": "system", "content": _year_context()}] + history + [{"role": "user", "content": message}]
    result = await Runner.run(sec_agent, messages, run_config=run_config)
    single_response: SingleAgentResponse = result.final_output
    answer = single_response.final_answer
    citations = [c.model_dump() for c in single_response.citations]
    return answer, citations


async def run_multi_agent_conversation(
    message: str,
    history: list[dict[str, str]],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Run one turn using the multi-agent Boss-Worker system.

    Args:
        message: The sanitized user message for this turn.
        history: Prior conversation turns as [{role, content}, ...] dicts.

    Returns:
        (answer, citations, scratchpad) where scratchpad is a dict from the
        Boss agent's structured BossResponse output.
    """
    from sec_llm.agents.boss import boss_agent
    from sec_llm.dependencies import get_settings

    settings = get_settings()
    run_config = RunConfig(model_provider=OpenAIProvider(api_key=settings.openai_api_key))

    messages = [{"role": "system", "content": _year_context()}] + history + [{"role": "user", "content": message}]
    result = await Runner.run(boss_agent, messages, run_config=run_config)
    boss_response: BossResponse = result.final_output
    answer = boss_response.final_answer
    citations = [c.model_dump() for c in boss_response.citations]
    scratchpad = boss_response.scratchpad.model_dump()
    return answer, citations, scratchpad
