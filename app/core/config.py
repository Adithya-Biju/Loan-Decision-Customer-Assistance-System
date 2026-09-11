"""
Centralized application configuration.

Every other layer reads settings from here rather than calling os.getenv()
directly. This is the single source of truth for environment-driven config.
"""

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    # ---------------------------------------------------------
    # LLM provider
    # ---------------------------------------------------------

    llm_provider: str = "openrouter"

    # OpenRouter
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Agent model
    agent_model: str = "nex-agi/nex-n2.5-mini:free"
    agent_temperature: float = 0.1
    agent_max_tool_iterations: int = 8

    # ---------------------------------------------------------
    # Embeddings
    # ---------------------------------------------------------

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # ---------------------------------------------------------
    # PostgreSQL
    # ---------------------------------------------------------

    database_url: str = (
        "postgresql+psycopg2://"
        "hdfc_admin:change_me_locally"
        "@localhost:5432/hdfc_loans"
    )

    # ---------------------------------------------------------
    # Qdrant
    # ---------------------------------------------------------

    qdrant_url: str = "http://localhost:6333"

    qdrant_api_key: str

    qdrant_collection: str = "hdfc_policy"

    # ---------------------------------------------------------
    # Observability
    # ---------------------------------------------------------

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ---------------------------------------------------------
    # App
    # ---------------------------------------------------------

    app_env: str = "development"
    log_level: str = "INFO"

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ---------------------------------------------------------
    # Guardrails / business rules
    # ---------------------------------------------------------

    manual_review_score_threshold: int = 60

    risk_score_escalation_threshold: int = 80

    max_ratio_sanity_cap: float = 20.0


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor.

    The .env file is parsed once per process.
    """
    return Settings()