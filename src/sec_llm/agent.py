"""SEC Financial Analyst agent definition using OpenAI Agents SDK."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from typing_extensions import TypedDict

from agents import Agent, GuardrailFunctionOutput, RunContextWrapper, function_tool, input_guardrail

from sec_llm.compute import aggregate_quarters as _aggregate_quarters
from sec_llm.compute import compute_growth as _compute_growth
from sec_llm.compute import compute_margin as _compute_margin
from sec_llm.guardrails import check_scope, sanitize_input

_PROMPTS_DIR = Path(__file__).parent / "prompts"


# ---------------------------------------------------------------------------
# Typed schema for aggregate_quarters tool input
# ---------------------------------------------------------------------------

class QuarterDataPoint(TypedDict):
    period: str
    value: float


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool
async def get_income_statement(
    ticker: str,
    fiscal_year: int,
    quarter: int | None = None,
) -> dict[str, Any]:
    """Fetch income statement data for a company from SEC EDGAR 10-K or 10-Q filings.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "MSFT", "NVDA").
        fiscal_year: The fiscal year as an integer (e.g. 2023, 2024).
        quarter: Optional quarter number 1-4. Pass null for annual (10-K) data.

    Returns a dict with: revenue, cost_of_revenue, gross_profit, operating_income,
    net_income, eps_basic, eps_diluted (all in dollars), plus filing metadata.
    """
    if not ticker or not ticker.strip():
        return {"error": "ticker must be a non-empty string (e.g. 'AAPL')."}
    if not (2000 <= fiscal_year <= 2030):
        return {"error": f"fiscal_year {fiscal_year} is out of the supported range (2000–2030)."}
    if quarter is not None and quarter not in (1, 2, 3, 4):
        return {"error": f"quarter must be 1, 2, 3, or 4 — got {quarter}."}

    from sec_llm.dependencies import get_edgar_client

    client = get_edgar_client()
    result = await client.get_income_statement(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        quarter=quarter,
    )
    return result.model_dump()


@function_tool
def compute_growth(
    metric_name: str,
    current_value: float | None,
    prior_value: float | None,
    current_period: str,
    prior_period: str,
) -> dict[str, Any]:
    """Compute year-over-year or quarter-over-quarter growth rate.

    Args:
        metric_name: Name of the metric (e.g. "revenue", "net_income").
        current_value: The value for the more recent period.
        prior_value: The value for the earlier period.
        current_period: Label for the current period (e.g. "FY2024", "Q2 FY2024").
        prior_period: Label for the prior period (e.g. "FY2023", "Q2 FY2023").

    Returns growth_rate (decimal), growth_percentage, and a human-readable formula.
    """
    if current_value is None:
        return {"error": f"Cannot compute growth for {metric_name}: {current_period} value is not reported in this SEC filing."}
    if prior_value is None:
        return {"error": f"Cannot compute growth for {metric_name}: {prior_period} value is not reported in this SEC filing."}
    if not metric_name or not current_period or not prior_period:
        return {"error": "metric_name, current_period, and prior_period must be non-empty strings."}

    result = _compute_growth(
        metric_name=metric_name,
        current_value=current_value,
        previous_value=prior_value,
        current_period=current_period,
        previous_period=prior_period,
    )
    return result.model_dump()


@function_tool
def aggregate_quarters(
    metric_name: str,
    quarter_data: list[QuarterDataPoint],
    method: str = "sum",
) -> dict[str, Any]:
    """Sum or average a metric across multiple quarters.

    Use this for TTM (trailing twelve months) calculations or average quarterly metrics.

    Args:
        metric_name: The metric name (e.g. "revenue", "net_income").
        quarter_data: List of dicts with "period" (str) and "value" (float) keys.
                      Example: [{"period": "Q1 FY2024", "value": 119575000000}, ...]
        method: "sum" for totals (revenue, income), "average" for rates (margins).

    Returns the aggregated result value, method used, and human-readable formula.
    """
    if not quarter_data:
        return {"error": f"No quarter data provided for {metric_name}."}
    if method not in ("sum", "average"):
        return {"error": f"method must be 'sum' or 'average', got '{method}'."}
    result = _aggregate_quarters(
        metric_name=metric_name, quarter_data=quarter_data, method=method
    )
    return result.model_dump()


@function_tool
def compute_margin(
    metric_name: str,
    numerator: float | None,
    revenue: float | None,
    period: str,
) -> dict[str, Any]:
    """Compute a margin ratio (numerator / revenue).

    REQUIRED: Always call this tool for any margin calculation — never compute margins inline.
    Use for gross margin (gross_profit / revenue), operating margin (operating_income / revenue),
    or net margin (net_income / revenue). Report the margin_percentage from the tool result.

    Args:
        metric_name: Name of the margin (e.g. "gross_margin", "operating_margin").
        numerator: The profit figure (e.g. gross_profit value in dollars).
        revenue: Total revenue in dollars.
        period: Label for the period (e.g. "FY2023", "Q1 FY2024").

    Returns margin_rate (decimal), margin_percentage, and a human-readable formula.
    """
    if numerator is None:
        return {"error": f"Cannot compute {metric_name} for {period}: the profit metric value is not reported in this SEC filing."}
    if revenue is None:
        return {"error": f"Cannot compute {metric_name} for {period}: revenue is not reported in this SEC filing."}
    if not metric_name or not period:
        return {"error": "metric_name and period must be non-empty strings."}

    result = _compute_margin(
        metric_name=metric_name,
        numerator=numerator,
        revenue=revenue,
        period=period,
    )
    return result.model_dump()


# ---------------------------------------------------------------------------
# Input guardrail
# ---------------------------------------------------------------------------

@input_guardrail
async def scope_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list,
) -> GuardrailFunctionOutput:
    """Block queries outside the income statement scope."""
    if isinstance(input, str):
        message = input
    elif isinstance(input, list) and input:
        last = input[-1]
        message = last.get("content", "") if isinstance(last, dict) else getattr(last, "content", "")
    else:
        message = ""

    error = check_scope(sanitize_input(str(message)))
    if error:
        return GuardrailFunctionOutput(output_info=error, tripwire_triggered=True)
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered=False)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

def _get_model() -> str:
    from sec_llm.dependencies import get_settings
    return get_settings().agent_model


sec_agent = Agent(
    name="SEC Financial Analyst",
    instructions=(_PROMPTS_DIR / "agent_system.txt").read_text(),
    tools=[get_income_statement, compute_growth, compute_margin, aggregate_quarters],
    input_guardrails=[scope_guardrail],
    model=_get_model(),
)
