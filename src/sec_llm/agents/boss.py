"""Boss agent — orchestrator of the multi-agent financial analyst system."""

from __future__ import annotations

from pathlib import Path

from agents import Agent, AgentOutputSchema

from sec_llm.agents.quant import quant_agent
from sec_llm.agents.researcher import researcher_agent
from sec_llm.agents.scratchpad import BossResponse
from sec_llm.agents.validator import validator_agent

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _get_boss_model() -> str:
    from sec_llm.dependencies import get_settings
    return get_settings().boss_model


def _build_boss_agent() -> Agent:
    return Agent(
        name="Boss Financial Analyst",
        instructions=(_PROMPTS_DIR / "boss_system.txt").read_text(),
        tools=[
            researcher_agent.as_tool(
                tool_name="call_researcher",
                tool_description=(
                    "Delegate data retrieval tasks to the Researcher agent. "
                    "Use for fetching income statements from SEC EDGAR (10-K/10-Q). "
                    "Also handles alternate filing searches when primary fetch fails. "
                    "Pass a clear task description including ticker, fiscal year, and what data is needed."
                ),
            ),
            quant_agent.as_tool(
                tool_name="call_quant_analyst",
                tool_description=(
                    "Delegate numerical computation tasks to the Quant Analyst agent. "
                    "Use for growth rates, margins, ratios, trend analysis, and quarter aggregations. "
                    "Pass all required numeric values and periods in the task description."
                ),
            ),
            validator_agent.as_tool(
                tool_name="call_validator",
                tool_description=(
                    "Delegate data validation tasks to the Validator agent. "
                    "Use to cross-reference net income consistency or validate metric consistency "
                    "across multiple sources. Pass all values and sources in the task description."
                ),
            ),
        ],
        model=_get_boss_model(),
        output_type=AgentOutputSchema(BossResponse, strict_json_schema=False),
    )


boss_agent = _build_boss_agent()
