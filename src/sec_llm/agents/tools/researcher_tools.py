"""Researcher agent tools — data retrieval utilities."""

from __future__ import annotations

from typing import Any

from agents import function_tool


@function_tool
async def search_alternate_filings(
    ticker: str,
    fiscal_year: int,
    filing_types: list[str],
) -> dict[str, Any]:
    """Search for alternate SEC filings when the primary filing is not found.

    Use this tool when get_income_statement returns a not-found error.
    Searches across multiple filing form types for the given fiscal year.

    Args:
        ticker: Stock ticker symbol (e.g. "AAPL", "MSFT").
        fiscal_year: The fiscal year to search for.
        filing_types: List of form types to search (e.g. ["10-K", "10-K/A", "20-F"]).

    Returns a dict with ticker, fiscal_year, and a list of matching filings
    with form_type, filing_date, and period_of_report.
    """
    if not ticker or not ticker.strip():
        return {"error": "ticker must be a non-empty string."}
    if not (2000 <= fiscal_year <= 2030):
        return {"error": f"fiscal_year {fiscal_year} is out of the supported range (2000–2030)."}
    if not filing_types:
        return {"error": "filing_types must be a non-empty list."}

    from sec_llm.dependencies import get_edgar_client

    client = get_edgar_client()
    return await client.search_alternate_filings(
        ticker=ticker.upper(),
        fiscal_year=fiscal_year,
        filing_types=filing_types,
    )
