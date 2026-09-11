"""
LLM client factory.

Centralizing this means every agent gets its model the same way, and
swapping providers (OpenRouter -> OpenAI -> Gemini -> Anthropic, per the
spec's explicit "OpenAI / Gemini / Claude / Open-weight model" flexibility
requirement) is a config change, not a code change in every agent file.

OpenRouter is OpenAI-API-compatible, so it uses the same ChatOpenAI client
as a real OpenAI key would -- only base_url and api_key differ.
"""
from langchain_openai import ChatOpenAI

from app.core.config import get_settings

settings = get_settings()


def get_agent_chat_model() -> ChatOpenAI:
    """
    Returns the chat model used by agents for tool-calling.

    Temperature is deliberately low (see core/config.py) -- for a
    tool-calling agent, consistent, repeatable tool selection matters more
    than varied phrasing. This does NOT affect calculate_risk_score's
    output, which never goes through the LLM at all (see
    app/agents/loan_analysis_agent.py for why).
    """
    if settings.llm_provider == "openrouter":
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "LLM_PROVIDER is 'openrouter' but OPENROUTER_API_KEY is not set. "
                "Add it to your .env file."
            )
        return ChatOpenAI(
            model=settings.agent_model,
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            temperature=settings.agent_temperature,
            default_headers={
                # OpenRouter uses these purely for its public leaderboard
                # attribution -- harmless to omit, but good practice.
                "HTTP-Referer": "https://github.com/hdfc-genai-loan-system",
                "X-Title": "HDFC Loan Intelligence System",
            },
        )

    if settings.llm_provider == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("LLM_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
            temperature=settings.agent_temperature,
        )

    raise NotImplementedError(
        f"LLM_PROVIDER '{settings.llm_provider}' is not wired up yet. "
        "Supported right now: 'openrouter', 'openai'. Gemini/Anthropic "
        "clients follow the same pattern -- add a branch here when needed."
    )
