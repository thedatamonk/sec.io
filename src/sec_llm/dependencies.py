"""FastAPI dependency injection helpers."""

from __future__ import annotations

from functools import lru_cache

from sec_llm.config import Settings
from sec_llm.pipeline import ExecutionPlanExecutor, QueryPipeline
from sec_llm.sec.client import EdgarClient


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_edgar_client() -> EdgarClient:
    settings = get_settings()
    return EdgarClient(
        identity=settings.edgar_identity,
        cache_ttl=settings.sec_cache_ttl_seconds,
    )


def get_openai_client():
    """Get a configured OpenAI client."""
    from openai import OpenAI

    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


@lru_cache
def get_pipeline() -> QueryPipeline:
    """Build the full query pipeline with all agents wired up."""
    settings = get_settings()
    edgar_client = get_edgar_client()
    executor = ExecutionPlanExecutor(edgar_client)

    from sec_llm.agents import ClarificationAgentImpl, PlannerAgentImpl, SummarizerAgentImpl

    openai_client = get_openai_client()

    clarifier = ClarificationAgentImpl(
        client=openai_client,
        model=settings.clarification_model,
    )
    planner = PlannerAgentImpl(
        client=openai_client,
        model=settings.planner_model,
    )
    summarizer = SummarizerAgentImpl(
        client=openai_client,
        model=settings.summarizer_model,
    )

    return QueryPipeline(
        clarifier=clarifier,
        planner=planner,
        summarizer=summarizer,
        executor=executor,
    )
