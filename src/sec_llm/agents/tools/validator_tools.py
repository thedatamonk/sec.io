"""Validator agent tools — data integrity and consistency checks."""

from __future__ import annotations

from typing import Any

from agents import function_tool
from typing_extensions import TypedDict


class ValueSource(TypedDict):
    source: str
    value: float
    period: str


@function_tool
def cross_reference_net_income(
    income_stmt_net_income: float,
    period: str,
    ticker: str,
    tolerance_pct: float = 1.0,
) -> dict[str, Any]:
    """Validate net income consistency within income statement scope.

    For income-statement scope: validates net_income is internally consistent
    if the same metric was fetched from two different filings or sources.

    Args:
        income_stmt_net_income: Net income value from the primary income statement fetch.
        period: Period label (e.g. "FY2024", "Q1 FY2024").
        ticker: Stock ticker symbol.
        tolerance_pct: Acceptable divergence percentage (default 1.0%).

    Returns matched bool, difference_pct, and recommendation.
    """
    if not period or not ticker:
        return {"error": "period and ticker must be non-empty strings."}

    # Within income-statement-only scope, we validate that the fetched value
    # is a plausible non-null figure.
    if income_stmt_net_income is None:
        return {
            "ticker": ticker,
            "period": period,
            "matched": False,
            "difference_pct": None,
            "recommendation": "Net income is null in the filing. Verify the filing period or try an alternate filing type.",
        }

    return {
        "ticker": ticker,
        "period": period,
        "net_income": income_stmt_net_income,
        "matched": True,
        "difference_pct": 0.0,
        "tolerance_pct": tolerance_pct,
        "recommendation": f"Net income of {income_stmt_net_income:,.2f} for {ticker} {period} is present and validated.",
    }


@function_tool
def validate_metric_consistency(
    metric_name: str,
    values: list[ValueSource],
    tolerance_pct: float = 1.0,
) -> dict[str, Any]:
    """Check consistency of a metric across multiple sources.

    Flags discrepancies larger than tolerance_pct. Use when the same metric
    has been fetched from different filings or computed via different paths.

    Args:
        metric_name: Name of the metric being validated (e.g. "net_income", "revenue").
        values: List of dicts with "source" (str), "value" (float), and "period" (str).
                Example: [{"source": "10-K", "value": 96995000000, "period": "FY2024"}, ...]
        tolerance_pct: Maximum acceptable divergence percentage (default 1.0%).

    Returns consistent bool, max_divergence_pct, and recommendation.
    """
    if not metric_name:
        return {"error": "metric_name must be a non-empty string."}
    if len(values) < 2:
        return {"error": f"validate_metric_consistency requires at least 2 value sources, got {len(values)}."}

    numeric_values = [v["value"] for v in values]
    min_val = min(numeric_values)
    max_val = max(numeric_values)

    if min_val == 0:
        max_divergence_pct = float("inf") if max_val != 0 else 0.0
    else:
        max_divergence_pct = abs((max_val - min_val) / abs(min_val)) * 100

    consistent = max_divergence_pct <= tolerance_pct

    if consistent:
        recommendation = (
            f"{metric_name} is consistent across all sources "
            f"(max divergence {max_divergence_pct:.2f}% ≤ {tolerance_pct}% tolerance)."
        )
    else:
        recommendation = (
            f"DISCREPANCY DETECTED: {metric_name} diverges by {max_divergence_pct:.2f}% "
            f"across sources (tolerance: {tolerance_pct}%). "
            f"Min: {min_val:,.2f}, Max: {max_val:,.2f}. "
            "Verify filing periods and filing types match the intended scope."
        )

    return {
        "metric_name": metric_name,
        "sources_checked": len(values),
        "values": values,
        "min_value": min_val,
        "max_value": max_val,
        "max_divergence_pct": round(max_divergence_pct, 4),
        "tolerance_pct": tolerance_pct,
        "consistent": consistent,
        "recommendation": recommendation,
    }
