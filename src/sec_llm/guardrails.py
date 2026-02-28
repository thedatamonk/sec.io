"""Input guardrails: scope enforcement and input sanitization."""

from __future__ import annotations

import re

from sec_llm.models import SECLLMError


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

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
                "the current scope. This agent supports income statement analysis only: "
                "revenue, net income, EPS, gross margin, and operating income from "
                "10-K and 10-Q filings."
            )
    return None


def sanitize_input(message: str, max_length: int = 2000) -> str:
    """Sanitize user input: truncate and strip control characters."""
    message = message[:max_length]
    message = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", message)
    return message.strip()
