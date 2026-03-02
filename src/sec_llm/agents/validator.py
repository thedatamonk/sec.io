"""Validator agent — data integrity and cross-reference specialist."""

from __future__ import annotations

from pathlib import Path

from agents import Agent

from sec_llm.agents.tools.validator_tools import (
    cross_reference_net_income,
    validate_metric_consistency,
)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _get_model() -> str:
    from sec_llm.dependencies import get_settings
    return get_settings().agent_model


validator_agent = Agent(
    name="Validator",
    instructions=(_PROMPTS_DIR / "validator_system.txt").read_text(),
    tools=[cross_reference_net_income, validate_metric_consistency],
    model=_get_model(),
)
