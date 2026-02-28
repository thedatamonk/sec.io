"""Pydantic schemas: errors and financial data models."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class SECLLMError(Exception):
    """Base exception for SEC-LLM."""


class CompanyNotFoundError(SECLLMError):
    """Raised when a ticker does not map to a valid company."""


class FilingNotFoundError(SECLLMError):
    """Raised when a requested filing cannot be found."""


class MetricNotAvailableError(SECLLMError):
    """Raised when a metric is not present in a filing."""


class ComputationError(SECLLMError):
    """Raised when a deterministic computation fails."""


class LLMError(SECLLMError):
    """Raised when LLM interaction fails."""


# ---------------------------------------------------------------------------
# Financial data schemas
# ---------------------------------------------------------------------------

class FilingMetadata(BaseModel):
    company: str
    ticker: str = Field(..., min_length=1, max_length=5)
    cik: str = ""
    filing_type: str  # "10-K" or "10-Q"
    filing_date: date | None = None
    fiscal_year: int = Field(..., ge=2000, le=2030)
    quarter: int | None = Field(None, ge=1, le=4)
    fiscal_period_end: date | None = None


class IncomeStatementData(BaseModel):
    metadata: FilingMetadata
    revenue: float | None = None
    cost_of_revenue: float | None = None
    gross_profit: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    eps_basic: float | None = None
    eps_diluted: float | None = None

    @property
    def period_label(self) -> str:
        m = self.metadata
        if m.quarter:
            return f"Q{m.quarter} FY{m.fiscal_year}"
        return f"FY{m.fiscal_year}"

    def get_metric(self, metric_name: str) -> float | None:
        """Look up a metric value by name string."""
        mapping = {
            "revenue": self.revenue,
            "net_income": self.net_income,
            "eps": self.eps_diluted,
            "gross_margin": self.gross_profit,  # raw value; margin computed separately
            "operating_income": self.operating_income,
        }
        return mapping.get(metric_name)
