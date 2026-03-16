"""Researcher agent — SEC data retrieval specialist."""

from __future__ import annotations

from pathlib import Path

from agents import Agent

from sec_llm.agent import get_income_statement
from sec_llm.agents.tools.researcher_tools import search_alternate_filings

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _get_model() -> str:
    from sec_llm.dependencies import get_settings
    return get_settings().agent_model


researcher_agent = Agent(
    name="Researcher",
    instructions=(_PROMPTS_DIR / "researcher_system.txt").read_text(),
    tools=[get_income_statement, search_alternate_filings],
    model=_get_model(),
)
