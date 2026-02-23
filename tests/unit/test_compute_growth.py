"""Tests for growth computation functions."""

from __future__ import annotations

import pytest

from sec_llm.compute import compute_growth
from sec_llm.models import ComputationError


class TestComputeGrowth:
    def test_positive_growth(self):
        result = compute_growth(
            metric_name="revenue",
            current_value=120.0,
            previous_value=100.0,
            current_period="FY2024",
            previous_period="FY2023",
        )
        assert result.growth_percentage == 20.0
        assert result.growth_rate == pytest.approx(0.2)
        assert result.metric_name == "revenue"
        assert "FY2024" in result.current_period
        assert "120" in result.formula

    def test_negative_growth(self):
        result = compute_growth(
            metric_name="net_income",
            current_value=80.0,
            previous_value=100.0,
            current_period="FY2024",
            previous_period="FY2023",
        )
        assert result.growth_percentage == -20.0
        assert result.growth_rate == pytest.approx(-0.2)

    def test_zero_growth(self):
        result = compute_growth(
            metric_name="revenue",
            current_value=100.0,
            previous_value=100.0,
            current_period="FY2024",
            previous_period="FY2023",
        )
        assert result.growth_percentage == 0.0

    def test_zero_denominator_raises(self):
        with pytest.raises(ComputationError, match="previous period value is zero"):
            compute_growth(
                metric_name="revenue",
                current_value=100.0,
                previous_value=0.0,
                current_period="FY2024",
                previous_period="FY2023",
            )

    def test_large_growth(self):
        result = compute_growth(
            metric_name="revenue",
            current_value=1_000_000.0,
            previous_value=500_000.0,
            current_period="FY2024",
            previous_period="FY2023",
        )
        assert result.growth_percentage == 100.0

    def test_qoq_growth(self):
        result = compute_growth(
            metric_name="revenue",
            current_value=110.0,
            previous_value=100.0,
            current_period="Q2 FY2024",
            previous_period="Q1 FY2024",
        )
        assert result.growth_percentage == 10.0

    def test_qoq_zero_denominator_raises(self):
        with pytest.raises(ComputationError):
            compute_growth(
                metric_name="revenue",
                current_value=100.0,
                previous_value=0.0,
                current_period="Q2 FY2024",
                previous_period="Q1 FY2024",
            )
