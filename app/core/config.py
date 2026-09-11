"""
Centralized application configuration.

Every other layer (database, services, agents) reads settings from here rather
than calling os.getenv() directly. This is the single source of truth for
environment-driven config, which matters for the guardrail thresholds too --
we want RISK_SCORE_ESCALATION_THRESHOLD defined once, not hardcoded in
multiple services.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM provider ---
    llm_provider: str = "openrouter"

    # --- OpenRouter (OpenAI-compatible endpoint; used for open-weight models
    # like Nex-N2.5-Pro without needing local GPU) ---
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # --- Agent-specific model config ---
    # Kept separate from llm_model above so the agent layer can use a
    # different model/provider than any other LLM usage in the app without
    # a code change -- just an env var.
    agent_model: str = "nex-agi/nex-n2.5-mini:free"
    agent_temperature: float = 0.1  # low temperature -- consistent tool-calling matters more than creativity here
    agent_max_tool_iterations: int = 8  # safety cap on the ReAct loop; prevents runaway tool-call chains

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = 384

    # --- PostgreSQL ---
    database_url: str = (
        "postgresql+psycopg2://hdfc_admin:change_me_locally@localhost:5432/hdfc_loans"
    )

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "hdfc"

    # --- Observability ---
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # --- App ---
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # --- Guardrails / business rules ---
    # Kept in config (not hardcoded in the service) so thresholds can be
    # tuned per environment without a code change -- e.g. a stricter
    # threshold in production than in a demo environment.
    #
    # Two distinct thresholds, deliberately different bars:
    #   manual_review_score_threshold  -> Agent 1's requires_manual_review
    #                                      flag on a single LoanAnalysis
    #                                      (broader -- "a human should glance
    #                                      at this")
    #   risk_score_escalation_threshold -> Agent 3's requires_human_review in
    #                                      FinalRecommendation (stricter --
    #                                      "the AI recommendation alone is
    #                                      not sufficient, block on a human")
    manual_review_score_threshold: int = 60
    risk_score_escalation_threshold: int = 80

    # This is a SANITY cap, not a display-flattering cap. It exists only to
    # neutralize the ~11 rows where Annual_Household_Income = 0, which sends
    # the raw ratio up to ~38,000 -- clearly a divide-by-zero artifact, not
    # real risk signal. It must stay well above any realistic-but-high value
    # (e.g. an 8.8x loan-to-income ratio is legitimately high-risk and must
    # pass through unchanged for scoring to work) -- setting this too low
    # (e.g. 3.0) would silently erase genuine risk signal before the scoring
    # function ever sees it.
    max_ratio_sanity_cap: float = 20.0


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings accessor. FastAPI dependencies should call this rather
    than instantiating Settings() directly, so the .env file is only parsed
    once per process.
    """
    return Settings()
