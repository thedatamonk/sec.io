"""EdgarClient: async wrapper around edgartools (which is synchronous)."""

from __future__ import annotations

import asyncio
import functools
import os
from datetime import date
from typing import Any

from sec_llm.models import CompanyNotFoundError, FilingNotFoundError, IncomeStatementData
from sec_llm.sec.cache import TTLCache
from sec_llm.sec.extractor import extract_income_statement


class EdgarClient:
    """Async-friendly facade over edgartools."""

    def __init__(self, identity: str, cache_ttl: int = 900):
        # Set identity before importing edgartools
        os.environ.setdefault("EDGAR_IDENTITY", identity)
        self._cache = TTLCache(ttl_seconds=cache_ttl)

    async def _run_sync(self, fn, *args, **kwargs):
        """Run a blocking function in the default executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))

    async def get_company_info(self, ticker: str) -> dict[str, Any]:
        """Look up basic company information by ticker."""
        cache_key = f"company:{ticker}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        info = await self._run_sync(self._fetch_company_info, ticker)
        self._cache.set(cache_key, info)
        return info

    async def get_income_statement(
        self,
        ticker: str,
        fiscal_year: int,
        quarter: int | None = None,
    ) -> IncomeStatementData:
        """Fetch and parse an income statement for a given period."""
        cache_key = f"income:{ticker}:{fiscal_year}:{quarter}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        result = await self._run_sync(
            self._fetch_income_statement, ticker, fiscal_year, quarter
        )
        self._cache.set(cache_key, result)
        return result

    @staticmethod
    def _fetch_company_info(ticker: str) -> dict[str, Any]:
        from edgar import Company

        try:
            company = Company(ticker)
        except Exception as exc:
            raise CompanyNotFoundError(f"Company not found for ticker: {ticker}") from exc

        cik_str = str(company.cik)
        if cik_str.startswith("-") or company.name.startswith("Entity -"):
            raise CompanyNotFoundError(f"Company not found for ticker: {ticker}")

        exchange = getattr(company, "exchange", None)
        exchanges = [exchange] if exchange else []

        return {
            "name": company.name,
            "ticker": ticker.upper(),
            "cik": str(company.cik),
            "exchanges": exchanges,
            "sic": str(getattr(company, "sic", "")),
            "sic_description": getattr(company, "industry", ""),
            "category": str(getattr(company, "filer_category", "")),
            "entity_type": getattr(company, "entity_type", ""),
        }

    @staticmethod
    def _fetch_income_statement(
        ticker: str,
        fiscal_year: int,
        quarter: int | None = None,
    ) -> IncomeStatementData:
        from edgar import Company

        try:
            company = Company(ticker)
        except Exception as exc:
            raise CompanyNotFoundError(f"Company not found for ticker: {ticker}") from exc

        filing_type = "10-Q" if quarter else "10-K"
        filings = company.get_filings(form=filing_type)

        if filings is None or len(filings) == 0:
            raise FilingNotFoundError(
                f"No {filing_type} filings found for {ticker}"
            )

        # Find the filing matching the requested fiscal year / quarter
        target_filing = _find_matching_filing(filings, fiscal_year, quarter)
        if target_filing is None:
            raise FilingNotFoundError(
                f"No {filing_type} filing found for {ticker} "
                f"FY{fiscal_year}" + (f" Q{quarter}" if quarter else "")
            )

        # Get financials via filing.obj().financials (preferred, has direct accessors)
        # Fall back to xbrl().statements if .obj() is unavailable
        financials = None
        try:
            filing_obj = target_filing.obj()
            if filing_obj is not None:
                financials = getattr(filing_obj, "financials", None)
        except Exception:
            pass

        if financials is None:
            xbrl = target_filing.xbrl()
            if xbrl is None:
                raise FilingNotFoundError(
                    f"No XBRL data available for {ticker} {filing_type} "
                    f"FY{fiscal_year}" + (f" Q{quarter}" if quarter else "")
                )
            financials = xbrl.statements

        filing_date_val = getattr(target_filing, "filing_date", None)
        if isinstance(filing_date_val, str):
            filing_date_val = date.fromisoformat(filing_date_val)

        return extract_income_statement(
            financials=financials,
            company_name=company.name,
            ticker=ticker.upper(),
            cik=str(company.cik),
            filing_type=filing_type,
            filing_date=filing_date_val,
            fiscal_year=fiscal_year,
            quarter=quarter,
        )


def _find_matching_filing(filings, fiscal_year: int, quarter: int | None) -> Any | None:
    """Find a filing matching the requested fiscal year and quarter.

    Uses filing date heuristics since edgartools doesn't always expose
    fiscal period metadata directly.
    """
    for filing in filings:
        filing_date_val = getattr(filing, "filing_date", None)
        if filing_date_val is None:
            continue

        if isinstance(filing_date_val, str):
            try:
                filing_date_val = date.fromisoformat(filing_date_val)
            except ValueError:
                continue

        # For annual (10-K): match by year
        if quarter is None:
            # 10-K filings are typically filed in the first few months after fiscal year end
            # A filing dated in early FY+1 corresponds to FY
            if filing_date_val.year == fiscal_year or filing_date_val.year == fiscal_year + 1:
                # Check period of report if available
                period_of_report = getattr(filing, "period_of_report", None)
                if period_of_report:
                    if isinstance(period_of_report, str):
                        try:
                            period_of_report = date.fromisoformat(period_of_report)
                        except ValueError:
                            period_of_report = None
                    if period_of_report and period_of_report.year == fiscal_year:
                        return filing
                # Fallback: trust the year range
                if filing_date_val.year in (fiscal_year, fiscal_year + 1):
                    return filing
        else:
            # For quarterly (10-Q): match quarter by period_of_report month
            period_of_report = getattr(filing, "period_of_report", None)
            if period_of_report:
                if isinstance(period_of_report, str):
                    try:
                        period_of_report = date.fromisoformat(period_of_report)
                    except ValueError:
                        continue

                # Map calendar month → approximate quarter
                # This is a rough heuristic; fiscal year ends vary by company
                month_to_quarter = {
                    3: 1, 4: 2, 6: 2, 7: 3, 9: 3, 10: 4, 12: 4,
                    1: 1, 2: 1, 5: 2, 8: 3, 11: 4,
                }
                report_quarter = month_to_quarter.get(period_of_report.month)
                report_year = period_of_report.year

                if report_year == fiscal_year and report_quarter == quarter:
                    return filing

    return None
