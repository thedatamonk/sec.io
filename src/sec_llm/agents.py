"""LLM-powered agents: clarification, planning, and summarization."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sec_llm.compute import ALL_TOOL_NAMES
from sec_llm.models import (
    ClarificationResponse,
    ClarifiedQuery,
    ComputationError,
    ExecutionPlan,
    LLMError,
    UserQuery,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_CLARIFICATION_PROMPT = (_PROMPTS_DIR / "clarification_system.txt").read_text()
_PLANNER_PROMPT = (_PROMPTS_DIR / "planner_system.txt").read_text()
_SUMMARIZER_PROMPT = (_PROMPTS_DIR / "summarizer_system.txt").read_text()

_CONFIDENCE_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Clarification Agent
# ---------------------------------------------------------------------------

class ClarificationAgentImpl:
    """Clarification agent that uses OpenAI structured output."""

    def __init__(self, client: Any, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def clarify(self, query: UserQuery) -> ClarificationResponse:
        return await self._llm_clarify(query)

    async def _llm_clarify(self, query: UserQuery) -> ClarificationResponse:
        """Use OpenAI structured output to clarify the query."""
        try:
            messages: list[dict[str, str]] = [{"role": "system", "content": _CLARIFICATION_PROMPT}]

            for entry in query.conversation_history:
                messages.append(entry)

            messages.append({"role": "user", "content": query.message})

            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=messages,
                response_format=ClarificationResponse,
                temperature=0.0,
            )

            result = response.choices[0].message.parsed
            if result is None:
                raise LLMError("LLM returned empty clarification response")

            if result.confidence < _CONFIDENCE_THRESHOLD and not result.needs_clarification:
                result.needs_clarification = True
                result.follow_up_question = result.follow_up_question or (
                    "Could you provide more details? I need a specific ticker, "
                    "metric, and time period to look up financial data."
                )

            return result
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"Clarification agent failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Planner Agent
# ---------------------------------------------------------------------------

class PlannerAgentImpl:
    """Planner agent that uses OpenAI structured output."""

    def __init__(self, client: Any, model: str = "gpt-4o"):
        self._client = client
        self._model = model

    async def plan(self, query: ClarifiedQuery) -> ExecutionPlan:
        return await self._llm_plan(query)

    async def _llm_plan(self, query: ClarifiedQuery) -> ExecutionPlan:
        """Use OpenAI structured output to generate an execution plan."""
        try:
            user_content = (
                f"Ticker: {query.ticker}\n"
                f"Query type: {query.query_type.value}\n"
                f"Metrics: {[m.value for m in query.metrics]}\n"
                f"Periods: {[p.model_dump() for p in query.periods]}\n"
                f"Original: {query.original_message}"
            )

            response = self._client.beta.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": _PLANNER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                response_format=ExecutionPlan,
                temperature=0.0,
            )

            plan = response.choices[0].message.parsed
            if plan is None:
                raise LLMError("LLM returned empty execution plan")

            for step in plan.steps:
                if step.tool not in ALL_TOOL_NAMES:
                    raise ComputationError(f"LLM proposed unknown tool: {step.tool}")

            return plan
        except (LLMError, ComputationError):
            raise
        except Exception as exc:
            raise LLMError(f"Planner agent failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Summarizer Agent
# ---------------------------------------------------------------------------

class SummarizerAgentImpl:
    """Summarizer agent that uses a cheaper LLM model to narrate results."""

    def __init__(self, client: Any, model: str = "gpt-4o-mini"):
        self._client = client
        self._model = model

    async def summarize(
        self,
        query: ClarifiedQuery,
        raw_data: list[dict[str, Any]],
        computations: list[dict[str, Any]],
    ) -> str:
        return await self._llm_summarize(query, raw_data, computations)

    async def _llm_summarize(
        self,
        query: ClarifiedQuery,
        raw_data: list[dict[str, Any]],
        computations: list[dict[str, Any]],
    ) -> str:
        """Use OpenAI to generate a natural language summary."""
        try:
            user_content = (
                f"Query: {query.original_message}\n"
                f"Ticker: {query.ticker}\n"
                f"Metrics: {[m.value for m in query.metrics]}\n\n"
                f"Raw data:\n{json.dumps(raw_data, indent=2, default=str)}\n\n"
                f"Computations:\n{json.dumps(computations, indent=2, default=str)}"
            )

            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": _SUMMARIZER_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
            )

            return response.choices[0].message.content or ""
        except Exception as exc:
            raise LLMError(f"Summarizer agent failed: {exc}") from exc

