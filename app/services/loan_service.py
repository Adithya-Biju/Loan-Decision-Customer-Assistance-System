"""
Loan analysis business logic.

The core function here, calculate_risk_score(), is DELIBERATELY plain
Python with no LLM call involved. This is the function Agent 1 will be
required to call as a tool (Phase 2) rather than reasoning about a risk
number itself -- see the agent-specification doc's note on why scoring must
be deterministic (reproducibility, and it gives the critic node something
concrete to check for hallucination against).

Because it's plain Python, it's fully unit-testable right now, before any
agent/LLM wiring exists -- see tests/test_loan_service.py.
"""
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.schemas import LoanAnalysis, RiskLevel
from app.repositories import loan_repository

settings = get_settings()


def calculate_risk_score(record: dict) -> dict:
    """
    Pure function: combined record in, scoring breakdown out.

    `record` is the dict shape produced by
    loan_repository.get_full_analysis_record() -- i.e.
    {"loan_details": {...}, "financials": {...}, "credit": {...},
     "previous_loans": {...}, "employment": {...}}

    Returns a dict with risk_score, risk_level, risk_factors,
    positive_factors, requires_manual_review -- everything LoanAnalysis
    needs except the loan_id itself.

    Rubric (documented in agent_specifications.md, reproduced here as the
    single source of truth -- the doc should link back to this function,
    not duplicate the numbers):

        CIBIL score       < 650            +20   |  >= 750           positive
                           650-749          +8
        Credit history     0 (none)         +15   |  1 (established)  positive
        Debt-to-income     > 0.8            +20   |  < 0.2            positive
                           0.4-0.8          +8
        Employment         Unemployed       +25   |  Salaried/Retired positive
                           Self-Employed    +8
        Loan-to-income     > 6              +15   |  < 1.5            positive
                           3-6              +6
        Default history    >= 1 (capped 2)  +20 each, max +40  | 0    positive

    risk_score is capped at 100. requires_manual_review uses the broader
    Agent-1-level threshold (settings.manual_review_score_threshold),
    which is intentionally lower/more permissive than Agent 3's stricter
    human-escalation bar (settings.risk_score_escalation_threshold) -- see
    core/config.py for why these are two different numbers.
    """
    financials = record["financials"]
    credit = record["credit"]
    employment = record["employment"]

    risk_points = 0
    risk_factors: list[str] = []
    positive_factors: list[str] = []

    # --- CIBIL score ---
    cibil = credit["cibil_score"]
    if cibil < 650:
        risk_points += 20
        risk_factors.append(f"Below-average CIBIL score ({cibil})")
    elif cibil < 750:
        risk_points += 8
    else:
        positive_factors.append(f"Strong CIBIL score ({cibil})")

    # --- Credit history ---
    if credit["credit_history"] == 0:
        risk_points += 15
        risk_factors.append("No established credit history")
    else:
        positive_factors.append("Established credit history")

    # --- Debt-to-income ratio (already capped by the repository) ---
    dti = financials["debt_to_income_ratio"]
    if dti > 0.8:
        risk_points += 20
        risk_factors.append(f"Very high debt-to-income ratio ({dti:.2f}x)")
    elif dti > 0.4:
        risk_points += 8
        risk_factors.append(f"Elevated debt-to-income ratio ({dti:.2f}x)")
    elif dti < 0.2:
        positive_factors.append(f"Low debt-to-income ratio ({dti:.2f}x)")

    # --- Employment status ---
    emp_status = employment["employment_status"]
    if emp_status == "Unemployed":
        risk_points += 25
        risk_factors.append("Applicant is currently unemployed")
    elif emp_status == "Self-Employed":
        risk_points += 8
        risk_factors.append("Self-employed income can be less predictable than salaried income")
    else:
        positive_factors.append(f"Stable {emp_status.lower()} income")

    # --- Loan-to-income ratio (already capped by the repository) ---
    lti = financials["loan_to_annual_income"]
    if lti > 6:
        risk_points += 15
        risk_factors.append(f"High loan-to-income ratio ({lti:.2f}x annual household income)")
    elif lti > 3:
        risk_points += 6
        risk_factors.append(f"Elevated loan-to-income ratio ({lti:.2f}x annual household income)")
    elif lti < 1.5:
        positive_factors.append(f"Conservative loan amount relative to income ({lti:.2f}x)")

    # --- Default history ---
    defaults = credit["default_history_count"]
    if defaults >= 1:
        risk_points += min(defaults, 2) * 20
        risk_factors.append(f"{defaults} prior loan default(s) on record")
    else:
        positive_factors.append("No prior loan defaults")

    risk_score = min(risk_points, 100)

    if risk_score < 40:
        risk_level = RiskLevel.LOW
    elif risk_score > 70:
        risk_level = RiskLevel.HIGH
    else:
        risk_level = RiskLevel.MEDIUM

    requires_manual_review = (
        risk_score >= settings.manual_review_score_threshold
        or defaults >= 1
        or (emp_status == "Unemployed" and dti > 0.8)
    )

    return {
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "positive_factors": positive_factors,
        "requires_manual_review": requires_manual_review,
    }


def analyze_loan(db: Session, loan_id: str) -> LoanAnalysis | None:
    """
    Full pipeline: fetch the record via the repository, score it, assemble
    the LoanAnalysis schema. Returns None if the loan_id doesn't exist --
    the controller layer is responsible for turning that into a 404, not
    this function (services stay HTTP-agnostic).
    """
    record = loan_repository.get_full_analysis_record(db, loan_id)
    if record is None:
        return None

    scoring = calculate_risk_score(record)

    return LoanAnalysis(
        loan_id=loan_id,
        financial_risk=scoring["risk_level"],
        risk_score=scoring["risk_score"],
        risk_factors=scoring["risk_factors"],
        positive_factors=scoring["positive_factors"],
        requires_manual_review=scoring["requires_manual_review"],
    )
