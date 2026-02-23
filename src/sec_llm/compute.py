"""Compute functions (growth, margins, aggregation) and tool registry."""

from __future__ import annotations

from typing import Any, Callable

from sec_llm.models import (
    AggregationResult,
    ComputationError,
    GrowthResult,
    MarginResult,
)


def compute_growth(
    metric_name: str,
    current_value: float,
    previous_value: float,
    current_period: str,
    previous_period: str,
) -> GrowthResult:
    """Compute growth rate between two periods."""
    if previous_value == 0:
        raise ComputationError(
            f"Cannot compute growth for {metric_name}: "
            f"previous period value is zero ({previous_period})"
        )
    growth_rate = (current_value - previous_value) / previous_value
    return GrowthResult(
        metric_name=metric_name,
        current_value=current_value,
        previous_value=previous_value,
        current_period=current_period,
        previous_period=previous_period,
        growth_rate=round(growth_rate, 6),
        growth_percentage=round(growth_rate * 100, 2),
        formula=f"({current_value:,.2f} - {previous_value:,.2f}) / {previous_value:,.2f} = {growth_rate * 100:.2f}%",
    )


def compute_margin(
    metric_name: str,
    numerator: float,
    revenue: float,
    period: str,
) -> MarginResult:
    """Compute a margin ratio (numerator / revenue)."""
    if revenue == 0:
        raise ComputationError(
            f"Cannot compute margin for {metric_name}: revenue is zero ({period})"
        )
    margin_rate = numerator / revenue
    return MarginResult(
        metric_name=metric_name,
        numerator=numerator,
        revenue=revenue,
        period=period,
        margin_rate=round(margin_rate, 6),
        margin_percentage=round(margin_rate * 100, 2),
        formula=f"{numerator:,.2f} / {revenue:,.2f} = {margin_rate * 100:.2f}%",
    )


def aggregate_quarters(
    metric_name: str,
    quarter_data: list[dict[str, float | str]],
    method: str = "sum",
) -> AggregationResult:
    """Aggregate quarterly values by sum or average.

    quarter_data is a list of dicts with keys "period" (str) and "value" (float).
    """
    if not quarter_data:
        raise ComputationError(f"No quarter data provided for aggregation of {metric_name}")

    values = [d["value"] for d in quarter_data]
    periods = [str(d["period"]) for d in quarter_data]

    if method == "sum":
        result = sum(values)
        formula = " + ".join(f"{v:,.2f}" for v in values) + f" = {result:,.2f}"
    elif method == "average":
        result = sum(values) / len(values)
        formula = (
            "(" + " + ".join(f"{v:,.2f}" for v in values) + f") / {len(values)} = {result:,.2f}"
        )
    else:
        raise ComputationError(f"Unknown aggregation method: {method}")

    return AggregationResult(
        metric_name=metric_name,
        values=values,
        periods=periods,
        method=method,
        result=round(result, 2),
        formula=formula,
    )


# Maps tool names (used by the Planner LLM) → Python callables
COMPUTE_REGISTRY: dict[str, Callable[..., Any]] = {
    "compute_yoy_growth": compute_growth,
    "compute_qoq_growth": compute_growth,
    "compute_margin": compute_margin,
    "aggregate_quarters": aggregate_quarters,
}

# Data-retrieval tools (async, handled separately by the executor)
DATA_TOOLS: set[str] = {"get_income_statement"}

ALL_TOOL_NAMES: set[str] = set(COMPUTE_REGISTRY.keys()) | DATA_TOOLS
