"""
FastAPI application entrypoint.

Run locally with: uvicorn app.main:app --reload --port 8000
Swagger docs at:  http://localhost:8000/docs
"""
from fastapi import FastAPI

from app.core.config import get_settings
from app.routers import loan_router

from app.core.lifespan import lifespan

settings = get_settings()

app = FastAPI(
    title="HDFC Intelligent Loan Decision & Customer Assistance System",
    description=(
        "Multi-agent GenAI system producing AI-assisted loan recommendations "
        "for human loan-officer validation. Not a final lending decision "
        "system -- see /docs for endpoint details."
    ),
    version="0.1.0",
    lifespan=lifespan
)

app.include_router(loan_router.router)
# Phase 2/3 will add: assistant_router (chat), knowledge_router (RAG),
# analytics_router (NL->SQL), loans_router (search).


@app.get("/health", tags=["System"])
def health_check() -> dict:
    return {"status": "ok", "env": settings.app_env}
