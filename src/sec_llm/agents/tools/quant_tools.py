"""Quant analyst tools — advanced numerical computation utilities."""

from __future__ import annotations

from typing import Any

from agents import function_tool
from typing_extensions import TypedDict


class DataPoint(TypedDict):
    period: str
    value: float


@function_tool
def compute_ratio(
    metric_name: str,
    numerator: float,
    denominator: float,
    period: str,
) -> dict[str, Any]:
    """Compute a ratio between two metrics.

    Use for R&D intensity (r&d_expense / revenue), cost ratios, or any
    numerator/denominator relationship not covered by compute_margin.

    Args:
        metric_name: Name of the ratio (e.g. "rd_intensity", "sga_ratio").
        numerator: The numerator value.
        denominator: The denominator value.
        period: Label for the period (e.g. "FY2024").

    Returns ratio (decimal), percentage, and a human-readable formula.
    """
    if not metric_name or not period:
        return {"error": "metric_name and period must be non-empty strings."}
    if denominator == 0:
        return {"error": f"Cannot compute {metric_name} for {period}: denominator is zero."}

    ratio = numerator / denominator
    percentage = ratio * 100

    return {
        "metric_name": metric_name,
        "period": period,
        "ratio": ratio,
        "percentage": round(percentage, 4),
        "formula": f"{metric_name} = {numerator:,.2f} / {denominator:,.2f} = {percentage:.2f}%",
    }


@function_tool
def compute_trend_analysis(
    metric_name: str,
    data_points: list[DataPoint],
) -> dict[str, Any]:
    """Compute trend analysis across multiple periods.

    Requires at least 2 data points. Calculates period-over-period growth rates,
    CAGR, min/max/avg, and trend direction.

    Args:
        metric_name: Name of the metric being analyzed (e.g. "revenue").
        data_points: List of dicts with "period" (str) and "value" (float) keys,
                     ordered chronologically oldest to newest.
                     Example: [{"period": "FY2022", "value": 394.3e9}, ...]

    Returns period_growth_rates, cagr, min/max/avg values, and trend_direction.
    """
    if not metric_name:
        return {"error": "metric_name must be a non-empty string."}
    if len(data_points) < 2:
        return {"error": f"compute_trend_analysis requires at least 2 data points, got {len(data_points)}."}

    values = [dp["value"] for dp in data_points]
    periods = [dp["period"] for dp in data_points]

    # Period-over-period growth rates
    growth_rates = []
    for i in range(1, len(values)):
        prior = values[i - 1]
        current = values[i]
        if prior == 0:
            growth_pct = None
            formula = f"Cannot compute: prior period {periods[i-1]} value is zero"
        else:
            growth_rate = (current - prior) / abs(prior)
            growth_pct = round(growth_rate * 100, 4)
            formula = f"({current:,.2f} - {prior:,.2f}) / {abs(prior):,.2f} = {growth_pct:.2f}%"
        growth_rates.append({
            "from_period": periods[i - 1],
            "to_period": periods[i],
            "growth_percentage": growth_pct,
            "formula": formula,
        })

    # CAGR over the full range
    n_periods = len(values) - 1
    first_val = values[0]
    last_val = values[-1]
    if first_val > 0 and last_val > 0 and n_periods > 0:
        cagr = ((last_val / first_val) ** (1 / n_periods) - 1) * 100
        cagr = round(cagr, 4)
        cagr_formula = f"({last_val:,.2f} / {first_val:,.2f})^(1/{n_periods}) - 1 = {cagr:.2f}%"
    else:
        cagr = None
        cagr_formula = "CAGR not computable (non-positive values)"

    # Summary stats
    min_val = min(values)
    max_val = max(values)
    avg_val = sum(values) / len(values)

    # Trend direction
    valid_growth = [gr["growth_percentage"] for gr in growth_rates if gr["growth_percentage"] is not None]
    if valid_growth:
        positive = sum(1 for g in valid_growth if g > 0)
        if positive == len(valid_growth):
            trend_direction = "consistently_growing"
        elif positive == 0:
            trend_direction = "consistently_declining"
        elif positive > len(valid_growth) / 2:
            trend_direction = "generally_growing"
        else:
            trend_direction = "generally_declining"
    else:
        trend_direction = "indeterminate"

    return {
        "metric_name": metric_name,
        "periods_analyzed": periods,
        "period_growth_rates": growth_rates,
        "cagr_percentage": cagr,
        "cagr_formula": cagr_formula,
        "min_value": min_val,
        "max_value": max_val,
        "avg_value": round(avg_val, 2),
        "trend_direction": trend_direction,
    }
