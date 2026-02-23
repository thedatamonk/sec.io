"""Tests for hallucination detection guardrails."""

from __future__ import annotations

from sec_llm.guardrails import build_truth_set, extract_numbers, verify_summary


class TestExtractNumbers:
    def test_dollar_billions(self):
        nums = extract_numbers("Revenue was $90.75 billion")
        assert any(abs(n - 90_750_000_000) < 1 for n in nums)

    def test_dollar_millions(self):
        nums = extract_numbers("Net income of $234.5 million")
        assert any(abs(n - 234_500_000) < 1 for n in nums)

    def test_percentage(self):
        nums = extract_numbers("Growth was 15.3%")
        assert 15.3 in nums

    def test_plain_number(self):
        nums = extract_numbers("EPS of $6.42")
        assert any(abs(n - 6.42) < 0.01 for n in nums)

    def test_negative_percentage(self):
        nums = extract_numbers("Declined -5.2%")
        assert -5.2 in nums

    def test_no_numbers(self):
        assert extract_numbers("No numbers here") == []

    def test_comma_formatted(self):
        nums = extract_numbers("Revenue was $1,234,567")
        assert any(abs(n - 1234567) < 1 for n in nums)


class TestBuildTruthSet:
    def test_from_raw_data(self):
        raw = [{"revenue": 100.0, "net_income": 25.0}]
        truth = build_truth_set(raw, [])
        assert 100.0 in truth
        assert 25.0 in truth

    def test_from_computations(self):
        comps = [{"growth_percentage": 20.0, "current_value": 120.0, "previous_value": 100.0}]
        truth = build_truth_set([], comps)
        assert 20.0 in truth
        assert 120.0 in truth

    def test_ignores_none(self):
        raw = [{"revenue": None, "net_income": 25.0}]
        truth = build_truth_set(raw, [])
        assert 25.0 in truth
        assert len(truth) == 1


class TestVerifySummary:
    def test_all_verified(self):
        truth = {100.0, 25.0, 20.0}
        summary = "Revenue was $100.00, net income $25.00, growth 20.0%"
        unverified = verify_summary(summary, truth)
        assert unverified == []

    def test_unverified_number(self):
        truth = {100.0}
        summary = "Revenue was $100.00 and profit was $999.00"
        unverified = verify_summary(summary, truth)
        assert any(abs(n - 999.0) < 1 for n in unverified)

    def test_empty_summary(self):
        assert verify_summary("", {100.0}) == []

    def test_empty_truth_set(self):
        assert verify_summary("Revenue was $100", set()) == []

    def test_tolerance(self):
        truth = {100.0}
        # 100.05 is within 0.1% of 100.0
        summary = "Value was $100.05"
        unverified = verify_summary(summary, truth, tolerance=0.001)
        assert unverified == []
