"""Tests for SEC normalizer functions."""

from __future__ import annotations

import pandas as pd

from sec_llm.sec.normalizer import find_row_value, format_period_label


class TestFindRowValue:
    def test_exact_match(self):
        df = pd.DataFrame(
            {"label": ["Revenue", "Net Income", "EPS"], "value": [100.0, 25.0, 1.5]}
        )
        df = df.set_index("label")
        assert find_row_value(df, ["Revenue"]) == 100.0

    def test_case_insensitive(self):
        df = pd.DataFrame({"label": ["TOTAL REVENUE", "Net Income"], "value": [200.0, 50.0]})
        df = df.set_index("label")
        assert find_row_value(df, ["total revenue"]) == 200.0

    def test_partial_match(self):
        df = pd.DataFrame(
            {"label": ["Total Net Revenue", "Net Income (Loss)"], "value": [300.0, 75.0]}
        )
        df = df.set_index("label")
        assert find_row_value(df, ["Net Revenue"]) == 300.0

    def test_no_match(self):
        df = pd.DataFrame({"label": ["Revenue"], "value": [100.0]})
        df = df.set_index("label")
        assert find_row_value(df, ["Operating Income"]) is None

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        assert find_row_value(df, ["Revenue"]) is None

    def test_none_dataframe(self):
        assert find_row_value(None, ["Revenue"]) is None

    def test_candidate_priority(self):
        """First matching candidate wins."""
        df = pd.DataFrame(
            {"label": ["Net Sales", "Total Revenue"], "value": [150.0, 200.0]}
        )
        df = df.set_index("label")
        # "Revenue" matches "Total Revenue", "Net Sales" matches "Net Sales"
        assert find_row_value(df, ["Net Sales", "Total Revenue"]) == 150.0


class TestFormatPeriodLabel:
    def test_annual(self):
        assert format_period_label(2024) == "FY2024"

    def test_quarterly(self):
        assert format_period_label(2024, 2) == "Q2 FY2024"
