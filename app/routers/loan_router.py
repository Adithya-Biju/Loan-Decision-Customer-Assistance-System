"""
Router layer -- pure HTTP surface. Declares the route, its request/response
schema, and delegates everything else to the controller. No business logic,
no direct service or repository calls, ever.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.controllers import loan_controller
from app.core.database import get_db
from app.models.schemas import LoanAnalyzeRequest, LoanAnalysis

router = APIRouter(prefix="/api/v1/loan", tags=["Loan Analysis"])


@router.post("/analyze", response_model=LoanAnalysis)
def analyze_loan(
    request: LoanAnalyzeRequest,
    db: Session = Depends(get_db),
) -> LoanAnalysis:
    """
    Analyze the financial risk of a single loan application.

    This currently calls the deterministic scoring service directly.
    Phase 2 will route this same call through the LangGraph-orchestrated
    Loan Analysis Agent instead, which wraps this exact service function
    as a tool -- the HTTP contract here won't change.
    """
    return loan_controller.handle_analyze_loan(db, request.loan_id)
