"""Company lookup endpoint."""
from typing import Any

from fastapi import APIRouter, HTTPException

from sec_llm.dependencies import get_edgar_client
from sec_llm.models import CompanyNotFoundError

router = APIRouter()


@router.get("/api/company/{ticker}")
async def get_company(ticker: str) -> dict[str, Any]:
    """Look up company information by ticker symbol."""
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 5:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    client = get_edgar_client()
    try:
        return await client.get_company_info(ticker)
    except CompanyNotFoundError:
        raise HTTPException(status_code=404, detail=f"Company not found: {ticker}")
