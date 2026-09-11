"""
Repository layer -- the only place raw SQL/ORM queries happen for loan data.

Design rule: every function here returns a plain dict containing ONLY
SAFE or INTERNAL tier fields (see db_models.py docstring for tier
definitions). RESTRICTED fields (verification table) and PROTECTED fields
(religion, gender) are never selected here at all -- not filtered out after
the fact, never queried in the first place. This means there's no field to
accidentally forward into an LLM prompt later; the data simply isn't in the
object.

This same layer is used by both:
  - app/services/loan_service.py (the FastAPI request path)
  - app/tools/postgres_tools.py (the LangChain tool-calling path for Agent 1)
so the two paths can never drift into inconsistent query logic.
"""
from sqlalchemy.orm import Session

from app.models.db_models import (
    CustomerFeedback,
    EmploymentDetail,
    FinancialDetail,
    Loan,
    LoanHistory,
)


def loan_exists(db: Session, loan_id: str) -> bool:
    return db.query(Loan.loan_id).filter(Loan.loan_id == loan_id).first() is not None


def get_loan_details(db: Session, loan_id: str) -> dict | None:
    """Core loan facts. No customer PII touched."""
    loan = db.query(Loan).filter(Loan.loan_id == loan_id).first()
    if loan is None:
        return None
    return {
        "loan_id": loan.loan_id,
        "loan_amount": loan.loan_amount,
        "loan_term_months": loan.loan_term_months,
        "purpose_of_loan": loan.purpose_of_loan,
        "loan_status": loan.loan_status,
        "property_area": loan.property_area,
        "region_branch": loan.region_branch,
    }


def get_customer_financials(db: Session, loan_id: str) -> dict | None:
    """
    Returns sanity-capped ratio values only (debt_to_income_ratio,
    loan_to_annual_income), capped at settings.max_ratio_sanity_cap (20.0).
    This is NOT a display-flattening cap -- genuine high-risk values like an
    8.8x loan-to-income ratio pass through unchanged. It only neutralizes
    the handful of rows where Annual_Household_Income = 0, which sends the
    raw ratio into the tens of thousands (a divide-by-zero artifact, not
    real signal). The raw uncapped values exist in the table for audit
    purposes but are deliberately not selected here.
    """
    fin = db.query(FinancialDetail).filter(FinancialDetail.loan_id == loan_id).first()
    if fin is None:
        return None
    return {
        "loan_id": loan_id,
        "applicant_income": fin.applicant_income,
        "coapplicant_income": fin.coapplicant_income,
        "annual_household_income": fin.annual_household_income,
        "debt_to_income_ratio": fin.debt_to_income_ratio_capped,
        "loan_to_annual_income": fin.loan_to_annual_income_capped,
        "existing_emis": fin.existing_emis,
        "monthly_expense": fin.monthly_expense,
        "asset_value": fin.asset_value,
    }


def get_credit_history(db: Session, loan_id: str) -> dict | None:
    hist = db.query(LoanHistory).filter(LoanHistory.loan_id == loan_id).first()
    if hist is None:
        return None
    return {
        "loan_id": loan_id,
        "credit_history": hist.credit_history,
        "cibil_score": hist.cibil_score,
        "default_history_count": hist.default_history_count,
    }


def get_previous_loans(db: Session, loan_id: str) -> dict | None:
    """
    IMPORTANT: number_of_previous_loans is an attribute reported on THIS
    application, not a join across other real historical rows for the same
    person -- the dataset has no reliable customer identity to join on
    (see db_models.py module docstring). The `note` field exists so callers
    (especially the LLM-facing tool wrapper) surface this caveat rather than
    implying a linked history.
    """
    hist = db.query(LoanHistory).filter(LoanHistory.loan_id == loan_id).first()
    if hist is None:
        return None
    return {
        "loan_id": loan_id,
        "number_of_previous_loans": hist.number_of_previous_loans,
        "note": (
            "This count is self-reported on this application record, not "
            "verified against other linked applications."
        ),
    }


def get_employment_details(db: Session, loan_id: str) -> dict | None:
    emp = db.query(EmploymentDetail).filter(EmploymentDetail.loan_id == loan_id).first()
    if emp is None:
        return None
    return {
        "loan_id": loan_id,
        "employment_status": emp.employment_status,
        "employment_length_years": emp.employment_length_years,
        "organization_type": emp.organization_type,
        "business_type": emp.business_type,
        "occupation": emp.occupation,
    }


def get_customer_feedback(db: Session, loan_id: str) -> dict | None:
    """Used by Agent 3 for the customer-improvement scenario, not Agent 1."""
    fb = db.query(CustomerFeedback).filter(CustomerFeedback.loan_id == loan_id).first()
    if fb is None:
        return None
    return {
        "loan_id": loan_id,
        "customer_feedback": fb.customer_feedback,
        "customer_sentiment": fb.customer_sentiment,
    }


def get_full_analysis_record(db: Session, loan_id: str) -> dict | None:
    """
    Convenience aggregator: one round trip's worth of the four tables Agent 1
    needs, instead of four separate tool calls when the caller (the risk
    scoring service) needs all of them at once. Agent 1 still has the
    individual tools available for cases where the LLM only needs one field
    (e.g. "what's the CIBIL score on this loan?").
    """
    if not loan_exists(db, loan_id):
        return None
    return {
        "loan_details": get_loan_details(db, loan_id),
        "financials": get_customer_financials(db, loan_id),
        "credit": get_credit_history(db, loan_id),
        "previous_loans": get_previous_loans(db, loan_id),
        "employment": get_employment_details(db, loan_id),
    }
