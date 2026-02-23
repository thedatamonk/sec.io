"""Format execution results into chart/table payloads for the frontend."""

from __future__ import annotations

from typing import Any

from sec_llm.models import (
    AggregationResult,
    GrowthResult,
    IncomeStatementData,
    MarginResult,
    QueryType,
    SourceCitation,
    VisualizationPayload,
)


def format_visualization(
    query_type: QueryType,
    metric: str,
    step_results: list[dict[str, Any]],
) -> VisualizationPayload | None:
    """Build a VisualizationPayload from execution results."""
    if query_type == QueryType.direct_retrieval:
        return _format_single_value(metric, step_results)
    elif query_type == QueryType.growth_comparison:
        return _format_comparison(metric, step_results)
    elif query_type == QueryType.time_series:
        return _format_timeseries(metric, step_results)
    return None


def _format_single_value(
    metric: str, step_results: list[dict[str, Any]]
) -> VisualizationPayload | None:
    data = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, IncomeStatementData):
            value = output.get_metric(metric)
            if value is not None:
                data.append({"period": output.period_label, "value": value})
    if not data:
        return None
    return VisualizationPayload(chart_type="single_value", metric=metric, data=data)


def _format_comparison(
    metric: str, step_results: list[dict[str, Any]]
) -> VisualizationPayload | None:
    data = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, GrowthResult):
            data.append({
                "period": output.current_period,
                "value": output.current_value,
                "previous_period": output.previous_period,
                "previous_value": output.previous_value,
                "growth_percentage": output.growth_percentage,
                "formula": output.formula,
            })
        elif isinstance(output, IncomeStatementData):
            value = output.get_metric(metric)
            if value is not None:
                data.append({"period": output.period_label, "value": value})
    if not data:
        return None
    return VisualizationPayload(chart_type="comparison", metric=metric, data=data)


def _format_timeseries(
    metric: str, step_results: list[dict[str, Any]]
) -> VisualizationPayload | None:
    data = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, IncomeStatementData):
            value = output.get_metric(metric)
            if value is not None:
                data.append({"period": output.period_label, "value": value})
        elif isinstance(output, AggregationResult):
            for period, value in zip(output.periods, output.values):
                data.append({"period": period, "value": value})
    if not data:
        return None
    return VisualizationPayload(chart_type="timeseries", metric=metric, data=data)


def build_citations(step_results: list[dict[str, Any]]) -> list[SourceCitation]:
    """Extract source citations from all income statement results."""
    citations = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, IncomeStatementData):
            m = output.metadata
            citations.append(SourceCitation(
                ticker=m.ticker,
                filing_type=m.filing_type,
                filing_date=str(m.filing_date) if m.filing_date else None,
                fiscal_period=output.period_label,
            ))
    return citations


def build_raw_data(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract raw financial data dicts from step results."""
    raw = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, IncomeStatementData):
            raw.append(output.model_dump(mode="json"))
    return raw


def build_computations(step_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract computation result dicts from step results."""
    computations = []
    for result in step_results:
        output = result.get("output")
        if isinstance(output, (GrowthResult, MarginResult, AggregationResult)):
            computations.append(output.model_dump(mode="json"))
    return computations
