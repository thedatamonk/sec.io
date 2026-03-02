"""Quant Analyst agent — numerical computation specialist."""

from __future__ import annotations

from pathlib import Path

from agents import Agent

from sec_llm.agent import aggregate_quarters, compute_growth, compute_margin
from sec_llm.agents.tools.quant_tools import compute_ratio, compute_trend_analysis

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _get_model() -> str:
    from sec_llm.dependencies import get_settings
    return get_settings().agent_model


quant_agent = Agent(
    name="Quant Analyst",
    instructions=(_PROMPTS_DIR / "quant_system.txt").read_text(),
    tools=[compute_growth, compute_margin, aggregate_quarters, compute_ratio, compute_trend_analysis],
    model=_get_model(),
)
