"""Input guardrails and post-summary hallucination check."""

from __future__ import annotations

import re
from typing import Any

from sec_llm.models import MetricName, SECLLMError


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

SUPPORTED_METRICS = {m.value for m in MetricName}

OUT_OF_SCOPE_KEYWORDS = [
    "balance sheet",
    "cash flow",
    "dcf",
    "discounted cash flow",
    "stock price",
    "share price",
    "market cap",
    "options",
    "derivatives",
    "risk factor",
    "dividend",
    "book value",
    "assets",
    "liabilities",
    "equity",
    "debt",
    "working capital",
]


class ScopeError(SECLLMError):
    """Raised when a query is outside the supported scope."""


def check_scope(message: str) -> str | None:
    """Check if a message contains out-of-scope topics.

    Returns an error message if out of scope, None if within scope.
    """
    msg_lower = message.lower()
    for keyword in OUT_OF_SCOPE_KEYWORDS:
        if keyword in msg_lower:
            return (
                f"This query appears to be about '{keyword}', which is outside "
                "the current scope. This tool supports income statement analysis only: "
                "revenue, net income, EPS, gross margin, and operating income from "
                "10-K and 10-Q filings."
            )
    return None


def sanitize_input(message: str, max_length: int = 2000) -> str:
    """Sanitize user input: truncate and strip control characters."""
    message = message[:max_length]
    message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", message)
    return message.strip()


# ---------------------------------------------------------------------------
# Hallucination check
# ---------------------------------------------------------------------------

def extract_numbers(text: str) -> list[float]:
    """Extract numeric values from a text string.

    Handles formats like:
    - $90.75 billion → 90750000000
    - $234.5 million → 234500000
    - 15.3% → 15.3
    - $6.42 → 6.42
    - 1,234,567 → 1234567
    """
    numbers: list[float] = []

    dollar_scale = re.findall(
        r"\$?([\d,]+\.?\d*)\s*(billion|million|thousand|trillion)?",
        text,
        re.IGNORECASE,
    )
    for num_str, scale in dollar_scale:
        num_str = num_str.replace(",", "")
        try:
            value = float(num_str)
        except ValueError:
            continue

        scale_lower = scale.lower() if scale else ""
        if scale_lower == "trillion":
            value *= 1_000_000_000_000
        elif scale_lower == "billion":
            value *= 1_000_000_000
        elif scale_lower == "million":
            value *= 1_000_000
        elif scale_lower == "thousand":
            value *= 1_000

        numbers.append(value)

    percentages = re.findall(r"([\-\d,]+\.?\d*)%", text)
    for pct_str in percentages:
        pct_str = pct_str.replace(",", "")
        try:
            numbers.append(float(pct_str))
        except ValueError:
            continue

    return numbers


def build_truth_set(
    raw_data: list[dict[str, Any]],
    computations: list[dict[str, Any]],
) -> set[float]:
    """Build a set of all known numeric values from raw data and computations."""
    truth: set[float] = set()

    for data in raw_data:
        for key in ("revenue", "cost_of_revenue", "gross_profit", "operating_income",
                     "net_income", "eps_basic", "eps_diluted"):
            value = data.get(key)
            if value is not None:
                truth.add(float(value))

    for comp in computations:
        for key in ("growth_rate", "growth_percentage", "current_value", "previous_value",
                     "margin_rate", "margin_percentage", "numerator", "revenue",
                     "result"):
            value = comp.get(key)
            if value is not None:
                truth.add(float(value))
        if "values" in comp:
            for v in comp["values"]:
                truth.add(float(v))

    return truth


def verify_summary(
    summary: str,
    truth_set: set[float],
    tolerance: float = 0.001,
) -> list[float]:
    """Verify that numbers in the summary match known truth values.

    Returns a list of unverified numbers (numbers found in the summary that
    don't match any known value within the tolerance).
    """
    if not summary or not truth_set:
        return []

    extracted = extract_numbers(summary)
    unverified: list[float] = []

    for number in extracted:
        if not _matches_any(number, truth_set, tolerance):
            unverified.append(number)

    return unverified


def _matches_any(value: float, truth_set: set[float], tolerance: float) -> bool:
    """Check if a value matches any value in the truth set within tolerance."""
    if value == 0:
        return 0.0 in truth_set

    for truth_val in truth_set:
        if truth_val == 0:
            if value == 0:
                return True
            continue
        relative_diff = abs(value - truth_val) / abs(truth_val)
        if relative_diff <= tolerance:
            return True

    return False
