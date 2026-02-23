"""Application configuration via environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_config = {"env_prefix": "SEC_LLM_"}

    openai_api_key: str
    edgar_identity: str = "SEC-LLM POC dev@example.com"

    planner_model: str = "gpt-4o"
    summarizer_model: str = "gpt-4o-mini"
    clarification_model: str = "gpt-4o"

    cors_origins: list[str] = ["http://localhost:3000"]
    sec_cache_ttl_seconds: int = 900  # 15 minutes

    rate_limit_per_minute: int = 20
