"""Extract income statement data from edgartools filing objects."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from sec_llm.models import FilingMetadata, IncomeStatementData
from sec_llm.sec.normalizer import LABEL_CANDIDATES, find_row_value

logger = logging.getLogger(__name__)


def extract_income_statement(
    financials: Any,
    company_name: str,
    ticker: str,
    cik: str,
    filing_type: str,
    filing_date: date | None,
    fiscal_year: int,
    quarter: int | None = None,
) -> IncomeStatementData:
    """Extract income statement metrics from an edgartools Financials object.

    Tries two strategies:
    1. Direct accessor methods on a Financials object (preferred, most reliable).
    2. DataFrame label matching via normalizer (fallback).
    """
    metadata = FilingMetadata(
        company=company_name,
        ticker=ticker,
        cik=cik,
        filing_type=filing_type,
        filing_date=filing_date,
        fiscal_year=fiscal_year,
        quarter=quarter,
    )

    # Strategy 1: Use direct accessor methods if the object supports them
    result = _try_direct_accessors(financials, metadata)
    if result is not None:
        return result

    # Strategy 2: Fall back to DataFrame label matching
    df = _get_income_statement_df(financials)

    return IncomeStatementData(
        metadata=metadata,
        revenue=find_row_value(df, LABEL_CANDIDATES["revenue"]),
        cost_of_revenue=find_row_value(df, LABEL_CANDIDATES["cost_of_revenue"]),
        gross_profit=find_row_value(df, LABEL_CANDIDATES["gross_profit"]),
        operating_income=find_row_value(df, LABEL_CANDIDATES["operating_income"]),
        net_income=find_row_value(df, LABEL_CANDIDATES["net_income"]),
        eps_basic=find_row_value(df, LABEL_CANDIDATES["eps_basic"]),
        eps_diluted=find_row_value(df, LABEL_CANDIDATES["eps_diluted"]),
    )


def _try_direct_accessors(financials: Any, metadata: FilingMetadata) -> IncomeStatementData | None:
    """Try to extract metrics using direct accessor methods on a Financials object.

    Uses get_revenue() and get_net_income() from the Financials object (the only
    income-related accessors edgartools provides), then supplements remaining
    metrics from the income statement DataFrame.
    """
    # Use the only two accessor methods that exist on Financials
    revenue = _safe_call(financials, "get_revenue")
    net_income = _safe_call(financials, "get_net_income")

    if revenue is None and net_income is None:
        return None

    # For remaining metrics, get the income statement DataFrame
    df = _get_income_statement_df(financials)
    gross_profit = find_row_value(df, LABEL_CANDIDATES["gross_profit"])
    operating_income = find_row_value(df, LABEL_CANDIDATES["operating_income"])
    cost_of_revenue = find_row_value(df, LABEL_CANDIDATES["cost_of_revenue"])
    eps_basic = find_row_value(df, LABEL_CANDIDATES["eps_basic"])
    eps_diluted = find_row_value(df, LABEL_CANDIDATES["eps_diluted"])

    return IncomeStatementData(
        metadata=metadata,
        revenue=revenue,
        cost_of_revenue=cost_of_revenue,
        gross_profit=gross_profit,
        operating_income=operating_income,
        net_income=net_income,
        eps_basic=eps_basic,
        eps_diluted=eps_diluted,
    )


def _safe_call(obj: Any, method_name: str) -> float | None:
    """Safely call a method on an object and return the result as a float."""
    method = getattr(obj, method_name, None)
    if method is None or not callable(method):
        return None
    try:
        val = method()
        if val is None:
            return None
        return float(val)
    except (ValueError, TypeError, AttributeError, Exception):
        return None


def _get_income_statement_df(financials: Any) -> Any:
    """Navigate edgartools financials object to get an income statement DataFrame.

    edgartools has varying APIs across versions. This tries common access patterns.
    """
    # If it's already a DataFrame, return it
    try:
        import pandas as pd

        if isinstance(financials, pd.DataFrame):
            return financials
    except ImportError:
        pass

    # Try: financials.income_statement (edgartools Financials object)
    if hasattr(financials, "income_statement"):
        income = financials.income_statement
        if callable(income):
            income = income()
        # The income statement may itself have a .to_dataframe() or .data attribute
        if hasattr(income, "to_dataframe"):
            df = income.to_dataframe()
            return df() if callable(df) else df
        if hasattr(income, "data"):
            data = income.data
            return data() if callable(data) else data
        if hasattr(income, "get_dataframe"):
            df = income.get_dataframe()
            return df() if callable(df) else df
        return income

    # Try: financials.get_income_statement()
    if hasattr(financials, "get_income_statement"):
        result = financials.get_income_statement()
        return result() if callable(result) else result

    return financials
