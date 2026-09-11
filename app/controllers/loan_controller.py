"""
Controller layer -- sits between the router (HTTP concerns) and the service
(business logic). Its job: validate input shape, call the service, and
translate the service's plain-Python result (or None) into something the
router can turn into an HTTP response. No SQL lives here, no route
decorators live here -- just orchestration and error shaping.
"""
import re

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.schemas import LoanAnalysis
from app.services import loan_service

# Loan IDs are HDFC followed by digits, per the dataset's actual format
# (HDFC100001-HDFC101000). Validating this here means a malformed ID never
# reaches the database layer at all.
LOAN_ID_PATTERN = re.compile(r"^HDFC\d{6}$")


def handle_analyze_loan(db: Session, loan_id: str) -> LoanAnalysis:
    """
    Raises HTTPException(422) for a malformed loan_id, HTTPException(404)
    if it's well-formed but doesn't exist, otherwise returns the
    LoanAnalysis produced by the service.
    """
    if not LOAN_ID_PATTERN.match(loan_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"'{loan_id}' is not a valid loan ID format (expected HDFC followed by 6 digits).",
        )

    result = loan_service.analyze_loan(db, loan_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No loan application found with ID '{loan_id}'.",
        )

    return result
