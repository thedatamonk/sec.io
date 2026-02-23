"""All Pydantic schemas: errors, queries, plans, financial data, computations, responses."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any

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
# Query schemas
# ---------------------------------------------------------------------------

class QueryType(str, Enum):
    direct_retrieval = "direct_retrieval"
    growth_comparison = "growth_comparison"
    time_series = "time_series"


class MetricName(str, Enum):
    revenue = "revenue"
    net_income = "net_income"
    eps = "eps"
    gross_margin = "gross_margin"
    operating_income = "operating_income"


class FiscalPeriod(BaseModel):
    fiscal_year: int = Field(..., ge=2000, le=2030)
    quarter: int | None = Field(None, ge=1, le=4)

    @property
    def label(self) -> str:
        if self.quarter:
            return f"Q{self.quarter} FY{self.fiscal_year}"
        return f"FY{self.fiscal_year}"


class UserQuery(BaseModel):
    message: str = Field(..., max_length=2000)
    conversation_history: list[dict[str, str]] = Field(default_factory=list)


class ClarifiedQuery(BaseModel):
    ticker: str = Field(..., min_length=1, max_length=5, pattern=r"^[A-Z]{1,5}$")
    query_type: QueryType
    metrics: list[MetricName] = Field(..., min_length=1)
    periods: list[FiscalPeriod] = Field(..., min_length=1)
    original_message: str


class ClarificationResponse(BaseModel):
    """LLM returns either a clarified query or a follow-up question."""

    needs_clarification: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    follow_up_question: str | None = None
    clarified_query: ClarifiedQuery | None = None


# ---------------------------------------------------------------------------
# Execution plan schemas
# ---------------------------------------------------------------------------

class ToolCallArg(BaseModel):
    """Single key-value argument for a tool call.

    All values are strings to satisfy OpenAI structured output
    (which requires additionalProperties: false on all objects).
    """

    name: str
    value: str


class PlanStep(BaseModel):
    step_id: int = Field(..., ge=0)
    tool: str
    args: list[ToolCallArg]
    depends_on: list[int] = Field(default_factory=list)
    description: str = ""


class ExecutionPlan(BaseModel):
    steps: list[PlanStep] = Field(..., min_length=1, max_length=10)
    reasoning: str = ""


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


# ---------------------------------------------------------------------------
# Computation result schemas
# ---------------------------------------------------------------------------

class GrowthResult(BaseModel):
    metric_name: str
    current_value: float
    previous_value: float
    current_period: str
    previous_period: str
    growth_rate: float  # as decimal, e.g. 0.15 = 15%
    growth_percentage: float  # as percentage, e.g. 15.0
    formula: str  # human-readable formula string


class MarginResult(BaseModel):
    metric_name: str
    numerator: float
    revenue: float
    period: str
    margin_rate: float  # as decimal
    margin_percentage: float  # as percentage
    formula: str


class AggregationResult(BaseModel):
    metric_name: str
    values: list[float]
    periods: list[str]
    method: str  # "sum" or "average"
    result: float
    formula: str


# ---------------------------------------------------------------------------
# API response schemas
# ---------------------------------------------------------------------------

class SourceCitation(BaseModel):
    ticker: str
    filing_type: str
    filing_date: str | None = None
    fiscal_period: str


class VisualizationPayload(BaseModel):
    chart_type: str  # "single_value", "comparison", "timeseries"
    metric: str
    data: list[dict[str, Any]]


class GuardrailInfo(BaseModel):
    llm_computed_math: bool = False
    unverified_numbers: list[float] = Field(default_factory=list)


class AnalysisResponse(BaseModel):
    raw_data: list[dict[str, Any]] = Field(default_factory=list)
    computations: list[dict[str, Any]] = Field(default_factory=list)
    summary: str = ""
    citations: list[SourceCitation] = Field(default_factory=list)
    visualization: VisualizationPayload | None = None
    guardrails: GuardrailInfo = Field(default_factory=GuardrailInfo)

    # If the pipeline needs clarification instead of an answer
    needs_clarification: bool = False
    follow_up_question: str | None = None
