from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages

from app.models.schemas import (
    CustomerGuidance,
    FinalRecommendation,
    LoanAnalysis,
    RAGResponse,
)


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    query: str
    loan_id: str | None
    mode: str
    loan_analysis: LoanAnalysis | None
    policy_response: RAGResponse | None
    final_recommendation: FinalRecommendation | None
    customer_guidance: CustomerGuidance | None
    final_answer: str | None
    requires_human_review: bool
    escalation_reason: str | None
    error: str | None
    retry_count: int