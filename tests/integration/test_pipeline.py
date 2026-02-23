"""Integration tests for the query pipeline (mocked LLM, mocked SEC)."""

from __future__ import annotations

import pytest

from sec_llm.models import (
    ClarificationResponse,
    ClarifiedQuery,
    ExecutionPlan,
    FilingMetadata,
    FiscalPeriod,
    IncomeStatementData,
    MetricName,
    PlanStep,
    QueryType,
    ToolCallArg,
    UserQuery,
)
from sec_llm.pipeline import ExecutionPlanExecutor, QueryPipeline


class FakeClarifier:
    """Always returns a resolved query (no clarification needed)."""

    def __init__(self, clarified: ClarifiedQuery):
        self._clarified = clarified

    async def clarify(self, query: UserQuery) -> ClarificationResponse:
        return ClarificationResponse(
            needs_clarification=False,
            confidence=0.95,
            clarified_query=self._clarified,
        )


class FakeSummarizer:
    """Returns a fixed summary string."""

    async def summarize(
        self,
        query: ClarifiedQuery,
        raw_data: list,
        computations: list,
    ) -> str:
        metrics = [m.value for m in query.metrics]
        return f"Fake summary for {query.ticker}: {', '.join(metrics)}"


class FakePlanner:
    """Returns a hardcoded plan."""

    def __init__(self, plan: ExecutionPlan):
        self._plan = plan

    async def plan(self, query: ClarifiedQuery) -> ExecutionPlan:
        return self._plan


class FakeEdgarClient:
    """Returns fake income statement data."""

    def __init__(self, data_map: dict[str, IncomeStatementData]):
        self._data = data_map

    async def get_income_statement(
        self, ticker: str, fiscal_year: int, quarter: int | None = None
    ) -> IncomeStatementData:
        key = f"{ticker}:{fiscal_year}:{quarter}"
        return self._data[key]


def _make_income_data(
    fiscal_year: int,
    quarter: int | None = None,
    revenue: float = 100_000.0,
    net_income: float = 25_000.0,
) -> IncomeStatementData:
    return IncomeStatementData(
        metadata=FilingMetadata(
            company="Apple Inc.",
            ticker="AAPL",
            filing_type="10-Q" if quarter else "10-K",
            fiscal_year=fiscal_year,
            quarter=quarter,
        ),
        revenue=revenue,
        net_income=net_income,
    )


class TestPipelineDirectRetrieval:
    async def test_direct_retrieval(self):
        clarified = ClarifiedQuery(
            ticker="AAPL",
            query_type=QueryType.direct_retrieval,
            metrics=[MetricName.revenue],
            periods=[FiscalPeriod(fiscal_year=2024)],
            original_message="What was Apple's revenue in FY2024?",
        )

        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id=0,
                    tool="get_income_statement",
                    args=[
                        ToolCallArg(name="ticker", value="AAPL"),
                        ToolCallArg(name="fiscal_year", value="2024"),
                    ],
                    description="Fetch FY2024",
                )
            ],
            reasoning="Direct retrieval",
        )

        fake_data = {"AAPL:2024:None": _make_income_data(2024, revenue=391_000_000_000.0)}
        fake_edgar = FakeEdgarClient(fake_data)
        executor = ExecutionPlanExecutor.__new__(ExecutionPlanExecutor)
        executor._edgar = fake_edgar

        summarizer = FakeSummarizer()

        pipeline = QueryPipeline(
            clarifier=FakeClarifier(clarified),
            planner=FakePlanner(plan),
            summarizer=summarizer,
            executor=executor,
        )

        result = await pipeline.process(UserQuery(message="test"))

        assert result.needs_clarification is False
        assert len(result.raw_data) == 1
        assert result.raw_data[0]["revenue"] == 391_000_000_000.0
        assert len(result.citations) == 1
        assert result.citations[0].ticker == "AAPL"
        assert result.guardrails.llm_computed_math is False
        assert "revenue" in result.summary.lower()


class TestPipelineGrowthComparison:
    async def test_growth_comparison(self):
        clarified = ClarifiedQuery(
            ticker="AAPL",
            query_type=QueryType.growth_comparison,
            metrics=[MetricName.revenue],
            periods=[FiscalPeriod(fiscal_year=2023), FiscalPeriod(fiscal_year=2024)],
            original_message="Apple revenue growth FY2023 to FY2024",
        )

        plan = ExecutionPlan(
            steps=[
                PlanStep(
                    step_id=0,
                    tool="get_income_statement",
                    args=[
                        ToolCallArg(name="ticker", value="AAPL"),
                        ToolCallArg(name="fiscal_year", value="2023"),
                    ],
                ),
                PlanStep(
                    step_id=1,
                    tool="get_income_statement",
                    args=[
                        ToolCallArg(name="ticker", value="AAPL"),
                        ToolCallArg(name="fiscal_year", value="2024"),
                    ],
                ),
                PlanStep(
                    step_id=2,
                    tool="compute_yoy_growth",
                    args=[
                        ToolCallArg(name="metric_name", value="revenue"),
                        ToolCallArg(name="current_value", value="$step:1:revenue"),
                        ToolCallArg(name="previous_value", value="$step:0:revenue"),
                        ToolCallArg(name="current_period", value="FY2024"),
                        ToolCallArg(name="previous_period", value="FY2023"),
                    ],
                    depends_on=[0, 1],
                ),
            ],
            reasoning="Growth comparison",
        )

        fake_data = {
            "AAPL:2023:None": _make_income_data(2023, revenue=383_000_000_000.0),
            "AAPL:2024:None": _make_income_data(2024, revenue=391_000_000_000.0),
        }
        fake_edgar = FakeEdgarClient(fake_data)
        executor = ExecutionPlanExecutor.__new__(ExecutionPlanExecutor)
        executor._edgar = fake_edgar

        pipeline = QueryPipeline(
            clarifier=FakeClarifier(clarified),
            planner=FakePlanner(plan),
            summarizer=FakeSummarizer(),
            executor=executor,
        )

        result = await pipeline.process(UserQuery(message="test"))

        assert result.needs_clarification is False
        assert len(result.raw_data) == 2
        assert len(result.computations) == 1
        assert result.computations[0]["metric_name"] == "revenue"
        assert result.computations[0]["growth_percentage"] > 0
        assert "formula" in result.computations[0]


