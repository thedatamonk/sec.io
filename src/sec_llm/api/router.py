"""Aggregates all sub-routers."""
from fastapi import APIRouter

from sec_llm.api.chat import router as chat_router
from sec_llm.api.company import router as company_router
from sec_llm.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(company_router, tags=["company"])
api_router.include_router(chat_router, tags=["chat"])
