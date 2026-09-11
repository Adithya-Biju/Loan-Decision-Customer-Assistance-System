"""
Pydantic schemas -- the contracts every layer agrees on.

Two categories live here:
1. API request/response envelopes (what FastAPI routers accept/return)
2. Agent structured outputs (what Agent 1/2/3 are constrained to produce --
   these were designed in the agent-specification pass and are locked in
   here so the LLM's tool-calling / structured-output mode has a concrete
   schema to target, not a description in a prompt).

Field-level sensitivity is enforced at the repository layer (repositories
never SELECT from `verification` or expose religion/gender into any object
that could reach an LLM), not here -- these schemas simply never HAVE those
fields, so there's no field to accidentally leak.
"""
from enum import Enum

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Audience(str, Enum):
    LOAN_OFFICER = "loan_officer"
    CUSTOMER = "customer"


class CoverageStatus(str, Enum):
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_COVERED = "not_covered"


# ---------------------------------------------------------------------------
# API request envelopes
# ---------------------------------------------------------------------------

class LoanAnalyzeRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        examples=["Analyze loan HDFC100125"],
        description="Natural-language loan analysis request"
    )


class ChatRequest(BaseModel):
    session_id: str = Field(..., examples=["session_001"])
    query: str = Field(..., min_length=1, max_length=2000)


class KnowledgeQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class AnalyticsQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class LoanSearchParams(BaseModel):
    status: str | None = None
    min_cibil: int | None = None
    max_cibil: int | None = None
    branch: str | None = None


# ---------------------------------------------------------------------------
# Agent 1 -- Loan Analysis Agent output
# ---------------------------------------------------------------------------

class LoanAnalysis(BaseModel):
    loan_id: str
    loan_status: str
    financial_risk: RiskLevel
    risk_score: int = Field(..., ge=0, le=100)
    risk_factors: list[str]
    positive_factors: list[str]
    requires_manual_review: bool


# ---------------------------------------------------------------------------
# Agent 2 -- Policy & Knowledge Agent output
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    source: str
    section: str


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    coverage: CoverageStatus


# ---------------------------------------------------------------------------
# Agent 3 -- Decision Review & Customer Agent output
# ---------------------------------------------------------------------------

class SupportingFactor(BaseModel):
    factor: str
    policy_basis: str


class FinalRecommendation(BaseModel):
    loan_id: str
    audience: Audience = Audience.LOAN_OFFICER
    decision_summary: str
    supporting_factors: list[SupportingFactor]
    requires_human_review: bool
    escalation_reason: str | None = None
    disclaimer: str = (
        "This is an AI-generated recommendation for loan officer validation, "
        "not a final lending decision."
    )


class CustomerGuidance(BaseModel):
    loan_id: str
    audience: Audience = Audience.CUSTOMER
    recommendations: list[str]
    tone: str = "supportive"
    disclaimer: str = (
        "These are general suggestions based on this application's data, "
        "not a guarantee of future approval."
    )


# ---------------------------------------------------------------------------
# Shared error envelope
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    error: str
    detail: str
    loan_id: str | None = None

# ---------------------------------------------------------------------------
# LangGraph -- Query Routing
# ---------------------------------------------------------------------------

class RoutingDecision(BaseModel):
    intent: str = Field(
        ...,
        description=(
            "The user's intent. Must be one of: "
            "general, policy, loan_analysis, decision_review, "
            "customer_assistance, unsupported."
        ),
    )

    loan_id: str | None = Field(
        default=None,
        description=(
            "HDFC loan ID relevant to the request, such as HDFC100125. "
            "Use the loan ID from the current query when explicitly provided. "
            "For follow-up questions, reuse the active loan ID from context "
            "when appropriate."
        ),
    )