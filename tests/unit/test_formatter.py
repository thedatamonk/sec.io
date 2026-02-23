"""Tests for visualization formatter."""

from __future__ import annotations

from sec_llm.formatter import (
    build_citations,
    build_computations,
    build_raw_data,
    format_visualization,
)
from sec_llm.models import (
    FilingMetadata,
    GrowthResult,
    IncomeStatementData,
    QueryType,
)


def _make_income_data(fiscal_year: int, quarter: int | None = None, revenue: float = 100.0):
    return IncomeStatementData(
        metadata=FilingMetadata(
            company="Apple Inc.",
            ticker="AAPL",
            filing_type="10-Q" if quarter else "10-K",
            fiscal_year=fiscal_year,
            quarter=quarter,
        ),
        revenue=revenue,
        net_income=revenue * 0.25,
    )


class TestFormatVisualization:
    def test_single_value(self):
        data = _make_income_data(2024, revenue=100_000.0)
        results = [{"step_id": 0, "tool": "get_income_statement", "success": True, "output": data}]
        viz = format_visualization(QueryType.direct_retrieval, "revenue", results)
        assert viz is not None
        assert viz.chart_type == "single_value"
        assert len(viz.data) == 1
        assert viz.data[0]["value"] == 100_000.0

    def test_comparison(self):
        growth = GrowthResult(
            metric_name="revenue",
            current_value=120.0,
            previous_value=100.0,
            current_period="FY2024",
            previous_period="FY2023",
            growth_rate=0.2,
            growth_percentage=20.0,
            formula="(120.00 - 100.00) / 100.00 = 20.00%",
        )
        results = [
            {"step_id": 0, "success": True, "output": _make_income_data(2023, revenue=100.0)},
            {"step_id": 1, "success": True, "output": _make_income_data(2024, revenue=120.0)},
            {"step_id": 2, "tool": "compute_yoy_growth", "success": True, "output": growth},
        ]
        viz = format_visualization(QueryType.growth_comparison, "revenue", results)
        assert viz is not None
        assert viz.chart_type == "comparison"

    def test_timeseries(self):
        results = [
            {"step_id": i, "success": True, "output": _make_income_data(2024, quarter=i + 1, revenue=100.0 + i * 10)}
            for i in range(4)
        ]
        viz = format_visualization(QueryType.time_series, "revenue", results)
        assert viz is not None
        assert viz.chart_type == "timeseries"
        assert len(viz.data) == 4

    def test_no_data_returns_none(self):
        viz = format_visualization(QueryType.direct_retrieval, "revenue", [])
        assert viz is None


class TestBuildCitations:
    def test_extracts_citations(self):
        data = _make_income_data(2024)
        results = [{"output": data}]
        citations = build_citations(results)
        assert len(citations) == 1
        assert citations[0].ticker == "AAPL"
        assert citations[0].fiscal_period == "FY2024"


class TestBuildRawData:
    def test_extracts_raw(self):
        data = _make_income_data(2024, revenue=500.0)
        results = [{"output": data}]
        raw = build_raw_data(results)
        assert len(raw) == 1
        assert raw[0]["revenue"] == 500.0


class TestBuildComputations:
    def test_extracts_growth(self):
        growth = GrowthResult(
            metric_name="revenue",
            current_value=120.0,
            previous_value=100.0,
            current_period="FY2024",
            previous_period="FY2023",
            growth_rate=0.2,
            growth_percentage=20.0,
            formula="test",
        )
        results = [{"output": growth}]
        comps = build_computations(results)
        assert len(comps) == 1
        assert comps[0]["growth_percentage"] == 20.0
