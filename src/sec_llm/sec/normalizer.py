"""Field name mapping and normalization for edgartools DataFrames."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# Candidate label lists for each income statement metric.
# edgartools DataFrames use varying labels depending on the filing.
LABEL_CANDIDATES: dict[str, list[str]] = {
    "revenue": [
        "Revenue",
        "Revenues",
        "Net Revenue",
        "Net Revenues",
        "Total Revenue",
        "Total Revenues",
        "Net Sales",
        "Total Net Sales",
        "Sales",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
    ],
    "cost_of_revenue": [
        "Cost of Revenue",
        "Cost of Goods Sold",
        "Cost of Sales",
        "Cost of Products Sold",
        "Total Cost of Revenue",
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
    ],
    "gross_profit": [
        "Gross Profit",
        "Gross Margin",
        "GrossProfit",
    ],
    "operating_income": [
        "Operating Income",
        "Operating Income (Loss)",
        "Income from Operations",
        "OperatingIncomeLoss",
    ],
    "net_income": [
        "Net Income",
        "Net Income (Loss)",
        "Net Income Attributable",
        "NetIncomeLoss",
        "Net income",
    ],
    "eps_basic": [
        "Basic EPS",
        "Earnings Per Share, Basic",
        "Basic Earnings Per Share",
        "EarningsPerShareBasic",
    ],
    "eps_diluted": [
        "Diluted EPS",
        "Earnings Per Share, Diluted",
        "Diluted Earnings Per Share",
        "EarningsPerShareDiluted",
    ],
}


def find_row_value(
    df: pd.DataFrame,
    candidates: list[str],
    value_column: str | None = None,
) -> float | None:
    """Search a DataFrame for the first matching label candidate.

    The DataFrame is expected to have a label/concept column (typically the index
    or a column like 'label') and one or more value columns.

    Returns the value as a float, or None if no match is found.
    """
    if df is None or df.empty:
        return None

    # Determine which column(s) contain the row labels
    # Try multiple label sources for better matching across edgartools versions
    label_columns: list[pd.Series] = []

    if "concept" in df.columns:
        label_columns.append(df["concept"])
    if "label" in df.columns:
        label_columns.append(df["label"])
    if isinstance(df.index, pd.Index) and df.index.dtype == object:
        label_columns.append(df.index.to_series())
    if not label_columns:
        # Use first column as labels
        label_columns.append(df.iloc[:, 0])

    # Determine value column: skip known non-value columns
    _NON_VALUE_COLS = {"label", "concept", "level", "abstract", "units"}
    if value_column is not None and value_column in df.columns:
        val_col = value_column
    else:
        numeric_cols = [
            c for c in df.columns
            if c.lower() not in _NON_VALUE_COLS
        ]
        val_col = numeric_cols[-1] if numeric_cols else (df.columns[-1] if len(df.columns) >= 1 else None)
    if val_col is None:
        return None

    # Search for candidates across all label columns (case-insensitive partial match)
    for candidate in candidates:
        candidate_lower = candidate.lower()
        for label_series in label_columns:
            label_lower = label_series.astype(str).str.lower()
            mask = label_lower.str.contains(re.escape(candidate_lower), case=False, na=False)
            matches = df.loc[mask]
            if not matches.empty:
                raw_value = matches.iloc[0][val_col] if val_col in matches.columns else matches.iloc[0, -1]
                return _to_float(raw_value)

    return None


def _to_float(value: Any) -> float | None:
    """Safely convert a value to float."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def format_period_label(fiscal_year: int, quarter: int | None = None) -> str:
    if quarter:
        return f"Q{quarter} FY{fiscal_year}"
    return f"FY{fiscal_year}"
