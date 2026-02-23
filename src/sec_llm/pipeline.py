"""Main query pipeline and execution engine."""

from __future__ import annotations

import logging
from typing import Any, Protocol

from sec_llm.compute import ALL_TOOL_NAMES, COMPUTE_REGISTRY, DATA_TOOLS
from sec_llm.formatter import (
    build_citations,
    build_computations,
    build_raw_data,
    format_visualization,
)
from sec_llm.guardrails import build_truth_set, verify_summary
from sec_llm.models import (
    AnalysisResponse,
    ClarificationResponse,
    ClarifiedQuery,
    ComputationError,
    ExecutionPlan,
    GuardrailInfo,
    IncomeStatementData,
    PlanStep,
    UserQuery,
)
from sec_llm.sec.client import EdgarClient

logger = logging.getLogger(__name__)

_NULL_STRINGS = {"", "null", "none", "None"}


# ---------------------------------------------------------------------------
# Protocol types (used by QueryPipeline)
# ---------------------------------------------------------------------------

class ClarificationAgent(Protocol):
    async def clarify(self, query: UserQuery) -> ClarificationResponse: ...


class PlannerAgent(Protocol):
    async def plan(self, query: ClarifiedQuery) -> ExecutionPlan: ...


class SummarizerAgent(Protocol):
    async def summarize(
        self,
        query: ClarifiedQuery,
        raw_data: list[dict[str, Any]],
        computations: list[dict[str, Any]],
    ) -> str: ...


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class ExecutionPlanExecutor:
    """Executes an ExecutionPlan, routing each step to the right handler."""

    def __init__(self, edgar_client: EdgarClient):
        self._edgar = edgar_client

    async def execute(self, plan: ExecutionPlan) -> list[dict[str, Any]]:
        """Execute all plan steps in order. Returns a list of step results."""
        self._validate_plan(plan)

        results: list[dict[str, Any]] = []
        step_outputs: dict[int, Any] = {}

        for step in plan.steps:
            try:
                output = await self._execute_step(step, step_outputs)
                step_outputs[step.step_id] = output
                results.append({
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "success": True,
                    "output": output,
                })
            except Exception as exc:
                logger.error("Step %d (%s) failed: %s", step.step_id, step.tool, exc)
                results.append({
                    "step_id": step.step_id,
                    "tool": step.tool,
                    "success": False,
                    "error": str(exc),
                })

        return results

    async def _execute_step(
        self, step: PlanStep, prior_outputs: dict[int, Any]
    ) -> Any:
        args_dict = {a.name: a.value for a in step.args}
        resolved_args = self._resolve_args(args_dict, prior_outputs)

        if step.tool in DATA_TOOLS:
            return await self._execute_data_step(step.tool, resolved_args)
        elif step.tool in COMPUTE_REGISTRY:
            return self._execute_compute_step(step.tool, resolved_args)
        else:
            raise ComputationError(f"Unknown tool: {step.tool}")

    async def _execute_data_step(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "get_income_statement":
            quarter_raw = args.get("quarter", "")
            quarter = int(quarter_raw) if quarter_raw not in _NULL_STRINGS else None
            fy_raw = args["fiscal_year"]
            if fy_raw in _NULL_STRINGS:
                raise ComputationError("fiscal_year is required")
            result = await self._edgar.get_income_statement(
                ticker=args["ticker"],
                fiscal_year=int(fy_raw),
                quarter=quarter,
            )
            return result
        raise ComputationError(f"Unknown data tool: {tool}")

    def _execute_compute_step(self, tool: str, args: dict[str, Any]) -> Any:
        fn = COMPUTE_REGISTRY[tool]
        return fn(**args)

    def _resolve_args(
        self, args: dict[str, Any], prior_outputs: dict[int, Any]
    ) -> dict[str, Any]:
        """Resolve $step:N:field references to prior step outputs."""
        resolved = {}
        for key, value in args.items():
            if isinstance(value, str) and value.startswith("$step:"):
                resolved[key] = self._dereference(value, prior_outputs)
            elif isinstance(value, list):
                resolved[key] = [
                    self._dereference(v, prior_outputs)
                    if isinstance(v, str) and v.startswith("$step:")
                    else v
                    for v in value
                ]
            else:
                resolved[key] = value
        return resolved

    def _dereference(self, ref: str, prior_outputs: dict[int, Any]) -> Any:
        """Resolve a $step:N:field reference."""
        parts = ref.split(":")
        if len(parts) < 3:
            raise ComputationError(f"Invalid step reference: {ref}")

        step_id = int(parts[1])
        field_path = parts[2:]

        if step_id not in prior_outputs:
            raise ComputationError(f"Step {step_id} output not found for reference: {ref}")

        output = prior_outputs[step_id]
        for field in field_path:
            if isinstance(output, IncomeStatementData):
                metric_val = output.get_metric(field)
                if metric_val is not None:
                    output = metric_val
                elif hasattr(output, field):
                    output = getattr(output, field)
                else:
                    raise ComputationError(
                        f"Field '{field}' not found on IncomeStatementData"
                    )
            elif isinstance(output, dict):
                output = output[field]
            elif hasattr(output, field):
                output = getattr(output, field)
            else:
                raise ComputationError(f"Cannot resolve field '{field}' on {type(output)}")

        return output

    @staticmethod
    def _validate_plan(plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if step.tool not in ALL_TOOL_NAMES:
                raise ComputationError(f"Unknown tool in plan: {step.tool}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class QueryPipeline:
    """Orchestrates the full query flow."""

    def __init__(
        self,
        clarifier: ClarificationAgent,
        planner: PlannerAgent,
        summarizer: SummarizerAgent,
        executor: ExecutionPlanExecutor,
    ):
        self._clarifier = clarifier
        self._planner = planner
        self._summarizer = summarizer
        self._executor = executor

    async def process(self, query: UserQuery) -> AnalysisResponse:
        """Run the full pipeline for a user query."""
        # Step 1: Clarification
        clarification = await self._clarifier.clarify(query)

        if clarification.needs_clarification:
            return AnalysisResponse(
                needs_clarification=True,
                follow_up_question=clarification.follow_up_question,
            )

        clarified = clarification.clarified_query
        assert clarified is not None

        # Step 2: Planning
        plan = await self._planner.plan(clarified)
        logger.info("Execution plan: %s", plan.model_dump_json(indent=2))

        # Step 3: Execution
        step_results = await self._executor.execute(plan)

        # Step 4: Build structured response
        raw_data = build_raw_data(step_results)
        computations = build_computations(step_results)
        citations = build_citations(step_results)
        primary_metric = clarified.metrics[0].value
        visualization = format_visualization(clarified.query_type, primary_metric, step_results)

        # Step 5: Summarize
        summary = await self._summarizer.summarize(clarified, raw_data, computations)

        # Step 6: Hallucination check (soft mode)
        truth_set = build_truth_set(raw_data, computations)
        unverified = verify_summary(summary, truth_set)

        return AnalysisResponse(
            raw_data=raw_data,
            computations=computations,
            summary=summary,
            citations=citations,
            visualization=visualization,
            guardrails=GuardrailInfo(
                llm_computed_math=False,
                unverified_numbers=unverified,
            ),
        )
