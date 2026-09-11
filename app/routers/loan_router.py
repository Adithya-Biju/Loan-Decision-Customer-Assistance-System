"""
HTTP API routes.

The API exposes:

1. Agent 1 - Loan Analysis
2. Agent 2 - Policy & Knowledge
3. Agent 3 - Decision Review / Customer Assistance
4. End-to-end LangGraph workflow

The router is responsible only for:
- HTTP request validation
- Calling the appropriate service/agent
- Translating application errors into HTTP errors
- Returning responses
"""

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    ChatRequest,
    CustomerGuidance,
    FinalRecommendation,
    KnowledgeQueryRequest,
    LoanAnalyzeRequest,
    LoanAnalysis,
    RAGResponse,
)

from app.services.policy_service import policy_service
from app.services.decision_service import decision_service
from app.services.workflow_service import workflow_service

from app.agents.loan_analysis_agent import run_loan_analysis


router = APIRouter(
    prefix="/api/v1",
)


# ============================================================
# AGENT 1 — LOAN ANALYSIS
# ============================================================

@router.post(
    "/loan/analyze",
    response_model=LoanAnalysis,
    tags=["Agent 1 - Loan Analysis"],
)
def analyze_loan(
    request: LoanAnalyzeRequest,
) -> LoanAnalysis:
    """
    Run Agent 1 independently.

    Agent 1:
    - resolves the loan ID
    - retrieves required loan information
    - invokes deterministic loan analysis logic
    """

    try:

        result = run_loan_analysis(
            query=request.query,
        )

        if result.error:

            raise ValueError(result.error)

        if result.loan_analysis is None:

            raise ValueError(
                "Loan analysis could not be completed."
            )

        return result.loan_analysis

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Loan analysis failed: {exc}",
        )


# ============================================================
# AGENT 2 — POLICY & KNOWLEDGE
# ============================================================

@router.post(
    "/knowledge/query",
    response_model=RAGResponse,
    tags=["Agent 2 - Policy & Knowledge"],
)
def query_policy(
    request: KnowledgeQueryRequest,
) -> RAGResponse:
    """
    Run Agent 2 independently.

    Agent 2:
    - retrieves relevant policy chunks from Qdrant
    - generates a grounded answer
    - validates the answer
    - returns policy citations
    """

    try:

        return policy_service.query_policy(
            query=request.query,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Policy agent failed: {exc}",
        )


# ============================================================
# AGENT 3 — DECISION REVIEW / CUSTOMER ASSISTANCE
# ============================================================

@router.post(
    "/decision/review",
    response_model=FinalRecommendation | CustomerGuidance,
    tags=["Agent 3 - Decision Review"],
)
def decision_review(
    request: ChatRequest,
) -> FinalRecommendation | CustomerGuidance:
    """
    Run Agent 3 independently.

    DecisionService handles:
    - Loan Analysis
    - Policy & Knowledge
    - Decision Review / Customer Assistance
    """

    try:

        return decision_service.review(
            query=request.query,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision review failed: {exc}",
        )


# ============================================================
# END-TO-END LANGGRAPH WORKFLOW
# ============================================================

@router.post(
    "/loan/workflow",
    tags=["LangGraph - End to End"],
)
def loan_workflow(
    request: ChatRequest,
):
    """
    Run the end-to-end LangGraph workflow.

    WorkflowService handles execution of the graph.

    session_id is passed to LangGraph so the checkpointer
    can maintain conversation state across requests.
    """

    try:

        result = workflow_service.run(
            query=request.query,
            session_id=request.session_id,
        )

    except ValueError as exc:

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Workflow failed: {exc}",
        )

    if result.get("error"):

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    return result