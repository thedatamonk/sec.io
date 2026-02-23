"""Tests for margin computation functions."""

from __future__ import annotations

import pytest

from sec_llm.compute import compute_margin
from sec_llm.models import ComputationError


class TestComputeMargin:
    def test_gross_margin(self):
        result = compute_margin(
            metric_name="gross_margin",
            numerator=60.0,
            revenue=100.0,
            period="FY2024",
        )
        assert result.margin_percentage == 60.0
        assert result.margin_rate == pytest.approx(0.6)
        assert result.formula == "60.00 / 100.00 = 60.00%"

    def test_net_margin(self):
        result = compute_margin(
            metric_name="net_margin",
            numerator=25.0,
            revenue=100.0,
            period="Q1 FY2024",
        )
        assert result.margin_percentage == 25.0

    def test_negative_margin(self):
        result = compute_margin(
            metric_name="operating_margin",
            numerator=-10.0,
            revenue=100.0,
            period="FY2024",
        )
        assert result.margin_percentage == -10.0

    def test_zero_revenue_raises(self):
        with pytest.raises(ComputationError, match="revenue is zero"):
            compute_margin(
                metric_name="gross_margin",
                numerator=50.0,
                revenue=0.0,
                period="FY2024",
            )
