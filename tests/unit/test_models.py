"""Tests for Pydantic model validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sec_llm.models import (
    AnalysisResponse,
    ClarificationResponse,
    ClarifiedQuery,
    ExecutionPlan,
    FilingMetadata,
    FiscalPeriod,
    GrowthResult,
    IncomeStatementData,
    MarginResult,
    MetricName,
    PlanStep,
    QueryType,
    SourceCitation,
    ToolCallArg,
    UserQuery,
)


class TestFiscalPeriod:
    def test_annual_label(self):
        fp = FiscalPeriod(fiscal_year=2024)
        assert fp.label == "FY2024"

    def test_quarterly_label(self):
        fp = FiscalPeriod(fiscal_year=2024, quarter=2)
        assert fp.label == "Q2 FY2024"

    def test_invalid_year_too_low(self):
        with pytest.raises(ValidationError):
            FiscalPeriod(fiscal_year=1999)

    def test_invalid_year_too_high(self):
        with pytest.raises(ValidationError):
            FiscalPeriod(fiscal_year=2031)

    def test_invalid_quarter(self):
        with pytest.raises(ValidationError):
            FiscalPeriod(fiscal_year=2024, quarter=5)


class TestUserQuery:
    def test_valid_query(self):
        q = UserQuery(message="What was Apple's revenue in FY2024?")
        assert q.message == "What was Apple's revenue in FY2024?"

    def test_too_long_message(self):
        with pytest.raises(ValidationError):
            UserQuery(message="x" * 2001)


class TestClarifiedQuery:
    def test_valid(self):
        cq = ClarifiedQuery(
            ticker="AAPL",
            query_type=QueryType.direct_retrieval,
            metrics=[MetricName.revenue],
            periods=[FiscalPeriod(fiscal_year=2024)],
            original_message="test",
        )
        assert cq.ticker == "AAPL"

    def test_invalid_ticker_lowercase(self):
        with pytest.raises(ValidationError):
            ClarifiedQuery(
                ticker="aapl",
                query_type=QueryType.direct_retrieval,
                metrics=[MetricName.revenue],
                periods=[FiscalPeriod(fiscal_year=2024)],
                original_message="test",
            )

    def test_invalid_ticker_too_long(self):
        with pytest.raises(ValidationError):
            ClarifiedQuery(
                ticker="AAPLXX",
                query_type=QueryType.direct_retrieval,
                metrics=[MetricName.revenue],
                periods=[FiscalPeriod(fiscal_year=2024)],
                original_message="test",
            )


class TestExecutionPlan:
    def test_valid_plan(self):
        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id=0,
                    tool="get_income_statement",
                    args=[ToolCallArg(name="ticker", value="AAPL")],
                ),
            ],
            reasoning="Fetch income statement",
        )
        assert len(plan.steps) == 1

    def test_empty_plan_rejected(self):
        with pytest.raises(ValidationError):
            ExecutionPlan(steps=[])

    def test_too_many_steps_rejected(self):
        steps = [
            PlanStep(step_id=i, tool="compute_margin", args=[]) for i in range(11)
        ]
        with pytest.raises(ValidationError):
            ExecutionPlan(steps=steps)


class TestIncomeStatementData:
    def test_get_metric(self):
        data = IncomeStatementData(
            metadata=FilingMetadata(
                company="Apple Inc.",
                ticker="AAPL",
                filing_type="10-K",
                fiscal_year=2024,
            ),
            revenue=100_000.0,
            net_income=25_000.0,
        )
        assert data.get_metric("revenue") == 100_000.0
        assert data.get_metric("net_income") == 25_000.0
        assert data.get_metric("nonexistent") is None

    def test_period_label_annual(self):
        data = IncomeStatementData(
            metadata=FilingMetadata(
                company="Apple", ticker="AAPL", filing_type="10-K", fiscal_year=2024
            ),
        )
        assert data.period_label == "FY2024"

    def test_period_label_quarterly(self):
        data = IncomeStatementData(
            metadata=FilingMetadata(
                company="Apple", ticker="AAPL", filing_type="10-Q", fiscal_year=2024, quarter=2
            ),
        )
        assert data.period_label == "Q2 FY2024"


class TestClarificationResponse:
    def test_needs_clarification(self):
        resp = ClarificationResponse(
            needs_clarification=True,
            confidence=0.4,
            follow_up_question="Which fiscal year?",
        )
        assert resp.needs_clarification is True
        assert resp.clarified_query is None

    def test_resolved(self):
        resp = ClarificationResponse(
            needs_clarification=False,
            confidence=0.95,
            clarified_query=ClarifiedQuery(
                ticker="AAPL",
                query_type=QueryType.direct_retrieval,
                metrics=[MetricName.revenue],
                periods=[FiscalPeriod(fiscal_year=2024)],
                original_message="test",
            ),
        )
        assert resp.clarified_query is not None


class TestAnalysisResponse:
    def test_default_guardrails(self):
        resp = AnalysisResponse()
        assert resp.guardrails.llm_computed_math is False
        assert resp.guardrails.unverified_numbers == []

    def test_with_citations(self):
        resp = AnalysisResponse(
            citations=[
                SourceCitation(
                    ticker="AAPL",
                    filing_type="10-K",
                    fiscal_period="FY2024",
                )
            ]
        )
        assert len(resp.citations) == 1
