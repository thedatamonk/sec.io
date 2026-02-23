"""Tests for aggregation computation functions."""

from __future__ import annotations

import pytest

from sec_llm.compute import aggregate_quarters
from sec_llm.models import ComputationError


class TestAggregateQuarters:
    def test_sum_aggregation(self):
        data = [
            {"period": "Q1 FY2024", "value": 100.0},
            {"period": "Q2 FY2024", "value": 120.0},
            {"period": "Q3 FY2024", "value": 110.0},
            {"period": "Q4 FY2024", "value": 130.0},
        ]
        result = aggregate_quarters("revenue", data, method="sum")
        assert result.result == 460.0
        assert result.method == "sum"
        assert len(result.values) == 4
        assert len(result.periods) == 4

    def test_average_aggregation(self):
        data = [
            {"period": "Q1 FY2024", "value": 100.0},
            {"period": "Q2 FY2024", "value": 200.0},
        ]
        result = aggregate_quarters("revenue", data, method="average")
        assert result.result == 150.0
        assert result.method == "average"

    def test_empty_data_raises(self):
        with pytest.raises(ComputationError, match="No quarter data"):
            aggregate_quarters("revenue", [], method="sum")

    def test_unknown_method_raises(self):
        with pytest.raises(ComputationError, match="Unknown aggregation method"):
            aggregate_quarters("revenue", [{"period": "Q1", "value": 10.0}], method="median")

    def test_formula_sum(self):
        data = [
            {"period": "Q1", "value": 10.0},
            {"period": "Q2", "value": 20.0},
        ]
        result = aggregate_quarters("revenue", data, method="sum")
        assert "10.00 + 20.00 = 30.00" in result.formula
