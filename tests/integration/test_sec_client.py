"""Integration tests for EdgarClient (hits real EDGAR API)."""

from __future__ import annotations

import pytest

from sec_llm.models.errors import CompanyNotFoundError
from sec_llm.sec.client import EdgarClient


@pytest.fixture
def edgar_client() -> EdgarClient:
    return EdgarClient(identity="SEC-LLM Test test@example.com", cache_ttl=60)


@pytest.mark.slow
class TestEdgarClientIntegration:
    async def test_get_company_info(self, edgar_client: EdgarClient):
        info = await edgar_client.get_company_info("AAPL")
        assert info["ticker"] == "AAPL"
        assert "Apple" in info["name"]
        assert info["cik"]

    async def test_invalid_ticker(self, edgar_client: EdgarClient):
        with pytest.raises(CompanyNotFoundError):
            await edgar_client.get_company_info("ZZZZZ")

    async def test_get_income_statement_annual(self, edgar_client: EdgarClient):
        data = await edgar_client.get_income_statement("AAPL", fiscal_year=2023)
        assert data.metadata.ticker == "AAPL"
        assert data.metadata.filing_type == "10-K"
        # Apple should have revenue
        assert data.revenue is not None
        assert data.revenue > 0
