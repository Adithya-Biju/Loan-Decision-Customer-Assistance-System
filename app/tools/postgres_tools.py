"""
LangChain tool wrappers for Postgres access -- this is what Agent 1 actually
calls. Each tool opens its own short-lived DB session (tools run outside a
FastAPI request context, so there's no request-scoped session to reuse the
way app.core.database.get_db provides one to routers).

Every function here is a thin wrapper around app.repositories.loan_repository
-- the query logic itself lives in exactly one place (the repository),
shared with the FastAPI request path. These wrappers add:
  1. The @tool decorator + docstring the LLM actually reads to decide when
     to call each one (docstrings here are load-bearing, not just comments)
  2. Loan ID format validation before ever touching the database
  3. Deliberately plain, LLM-readable error strings instead of raising --
     a tool that raises stops the agent's loop; a tool that returns
     "loan not found" lets the LLM react (e.g. tell the user, or try a
     corrected ID if the user gave a follow-up).
"""
import json
import re

from app.core.database import SessionLocal
from app.repositories import loan_repository
from app.guardrails.pii import sanitize_for_llm

try:
    from langchain_core.tools import tool
except ImportError:  # pragma: no cover - only hit if langchain isn't installed yet
    def tool(func):  # type: ignore
        """Fallback no-op decorator so this module still imports for testing
        the plain Python logic before langchain-core is installed."""
        return func


LOAN_ID_PATTERN = re.compile(r"^HDFC\d{6}$")


def _validate_loan_id(loan_id: str) -> str | None:
    """Returns an error string if malformed, None if the format is valid."""
    if not LOAN_ID_PATTERN.match(loan_id.strip().upper()):
        return (
            f"'{loan_id}' is not a valid loan ID format. Loan IDs look like "
            f"'HDFC100245' (HDFC followed by 6 digits). Ask the user to confirm "
            f"the correct loan ID rather than guessing."
        )
    return None


@tool
def get_loan_details(loan_id: str) -> str:
    """
    Retrieve core loan facts: loan amount, term, purpose, status, property
    area, and branch. Call this first for any question about a specific
    loan application to confirm the loan_id actually exists before calling
    the more specific tools below.
    """
    error = _validate_loan_id(loan_id)
    if error:
        return error

    db = SessionLocal()
    try:
        result = loan_repository.get_loan_details(db, loan_id.strip().upper())
    finally:
        db.close()

    if result is None:
        return f"No loan application found with ID '{loan_id}'. Do not invent details for it."
    return json.dumps(sanitize_for_llm(result))


@tool
def get_customer_financials(loan_id: str) -> str:
    """
    Retrieve income, debt-to-income ratio, loan-to-income ratio, existing
    EMIs, monthly expenses, and asset value for a specific loan application.
    Call this when analyzing financial risk. Note: debt_to_income_ratio and
    loan_to_annual_income are already sanity-capped -- use them as-is,
    do not recompute or second-guess them.
    """
    error = _validate_loan_id(loan_id)
    if error:
        return error

    db = SessionLocal()
    try:
        result = loan_repository.get_customer_financials(db, loan_id.strip().upper())
    finally:
        db.close()

    if result is None:
        return f"No financial record found for loan ID '{loan_id}'."
    return json.dumps(sanitize_for_llm(result))


@tool
def get_credit_history(loan_id: str) -> str:
    """
    Retrieve CIBIL score, credit history flag, and default history count
    for a specific loan application. Call this when analyzing
    creditworthiness or when the user asks about CIBIL score, credit
    history, or past defaults.
    """
    error = _validate_loan_id(loan_id)
    if error:
        return error

    db = SessionLocal()
    try:
        result = loan_repository.get_credit_history(db, loan_id.strip().upper())
    finally:
        db.close()

    if result is None:
        return f"No credit history record found for loan ID '{loan_id}'."
    return json.dumps(sanitize_for_llm(result))


@tool
def get_previous_loans(loan_id: str) -> str:
    """
    Retrieve the number of previous loans reported on a specific
    application. IMPORTANT: this is a self-reported count on this one
    application record, not a verified link to other real applications by
    the same person -- say so if you mention this figure, don't imply a
    verified loan history.
    """
    error = _validate_loan_id(loan_id)
    if error:
        return error

    db = SessionLocal()
    try:
        result = loan_repository.get_previous_loans(db, loan_id.strip().upper())
    finally:
        db.close()

    if result is None:
        return f"No previous-loan record found for loan ID '{loan_id}'."
    return json.dumps(sanitize_for_llm(result))


@tool
def get_employment_details(loan_id: str) -> str:
    """
    Retrieve employment status, employment length, organization type,
    business type, and occupation for a specific loan application. Call
    this when analyzing employment stability as a risk factor.
    """
    error = _validate_loan_id(loan_id)
    if error:
        return error

    db = SessionLocal()
    try:
        result = loan_repository.get_employment_details(db, loan_id.strip().upper())
    finally:
        db.close()

    if result is None:
        return f"No employment record found for loan ID '{loan_id}'."
    return json.dumps(sanitize_for_llm(result))


# The full set Agent 1 is bound to. Deliberately does NOT include
# calculate_risk_score -- see app/agents/loan_analysis_agent.py docstring
# for why the scoring step is never left to the LLM's discretion.
LOAN_ANALYSIS_TOOLS = [
    get_loan_details,
    get_customer_financials,
    get_credit_history,
    get_previous_loans,
    get_employment_details
]


@tool
def get_customer_feedback(loan_id: str) -> str:
    """
    Retrieve customer feedback and sentiment for a loan application.

    Sensitive customer information is not returned.
    """
    db = SessionLocal()

    try:
        feedback = loan_repository.get_customer_feedback(
            db,
            loan_id,
        )

        if feedback is None:
            return json.dumps({
                "loan_id": loan_id,
                "feedback": None,
                "sentiment": None,
            })

        return json.dumps({
            "loan_id": loan_id,
            "feedback": feedback.get("feedback"),
            "sentiment": feedback.get("sentiment"),
        })

    finally:
        db.close()