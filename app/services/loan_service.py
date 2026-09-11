"""
Loan analysis business logic.

The core function here, calculate_risk_score(), is deliberately plain
Python with no LLM call involved.

The risk calculation is deterministic so that:
- results are reproducible
- the logic can be unit tested
- Agent 1 cannot hallucinate a risk score
- downstream critic/review nodes have a concrete result to validate

Repository access is handled by analyze_loan().
"""

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models.schemas import LoanAnalysis, RiskLevel
from app.repositories import loan_repository


settings = get_settings()


class LoanService:

    def calculate_risk_score(self, record: dict) -> dict:
        """
        Pure deterministic risk calculation.

        Parameters
        ----------
        record:
            Combined record returned by
            loan_repository.get_full_analysis_record()

        Returns
        -------
        dict:
            Risk score, risk level, risk factors, positive factors,
            and manual-review flag.
        """

        financials = record["financials"]
        credit = record["credit"]
        employment = record["employment"]

        risk_points = 0

        risk_factors: list[str] = []
        positive_factors: list[str] = []

        # --------------------------------------------------------
        # CIBIL SCORE
        # --------------------------------------------------------

        cibil = credit["cibil_score"]

        if cibil < 650:
            risk_points += 20

            risk_factors.append(
                f"Below-average CIBIL score ({cibil})"
            )

        elif cibil < 750:
            risk_points += 8

        else:
            positive_factors.append(
                f"Strong CIBIL score ({cibil})"
            )

        # --------------------------------------------------------
        # CREDIT HISTORY
        # --------------------------------------------------------

        if credit["credit_history"] == 0:

            risk_points += 15

            risk_factors.append(
                "No established credit history"
            )

        else:

            positive_factors.append(
                "Established credit history"
            )

        # --------------------------------------------------------
        # DEBT-TO-INCOME RATIO
        # --------------------------------------------------------

        dti = financials["debt_to_income_ratio"]

        if dti > 0.8:

            risk_points += 20

            risk_factors.append(
                f"Very high debt-to-income ratio ({dti:.2f}x)"
            )

        elif dti > 0.4:

            risk_points += 8

            risk_factors.append(
                f"Elevated debt-to-income ratio ({dti:.2f}x)"
            )

        elif dti < 0.2:

            positive_factors.append(
                f"Low debt-to-income ratio ({dti:.2f}x)"
            )

        # --------------------------------------------------------
        # EMPLOYMENT STATUS
        # --------------------------------------------------------

        emp_status = employment["employment_status"]

        if emp_status == "Unemployed":

            risk_points += 25

            risk_factors.append(
                "Applicant is currently unemployed"
            )

        elif emp_status == "Self-Employed":

            risk_points += 8

            risk_factors.append(
                "Self-employed income can be less predictable "
                "than salaried income"
            )

        else:

            positive_factors.append(
                f"Stable {emp_status.lower()} income"
            )

        # --------------------------------------------------------
        # LOAN-TO-INCOME RATIO
        # --------------------------------------------------------

        lti = financials["loan_to_annual_income"]

        if lti > 6:

            risk_points += 15

            risk_factors.append(
                f"High loan-to-income ratio "
                f"({lti:.2f}x annual household income)"
            )

        elif lti > 3:

            risk_points += 6

            risk_factors.append(
                f"Elevated loan-to-income ratio "
                f"({lti:.2f}x annual household income)"
            )

        elif lti < 1.5:

            positive_factors.append(
                f"Conservative loan amount relative to income "
                f"({lti:.2f}x)"
            )

        # --------------------------------------------------------
        # DEFAULT HISTORY
        # --------------------------------------------------------

        defaults = credit["default_history_count"]

        if defaults >= 1:

            risk_points += min(defaults, 2) * 20

            risk_factors.append(
                f"{defaults} prior loan default(s) on record"
            )

        else:

            positive_factors.append(
                "No prior loan defaults"
            )

        # --------------------------------------------------------
        # FINAL SCORE
        # --------------------------------------------------------

        risk_score = min(risk_points, 100)

        if risk_score < 40:

            risk_level = RiskLevel.LOW

        elif risk_score > 70:

            risk_level = RiskLevel.HIGH

        else:

            risk_level = RiskLevel.MEDIUM

        # --------------------------------------------------------
        # MANUAL REVIEW
        # --------------------------------------------------------

        requires_manual_review = (
            risk_score >= settings.manual_review_score_threshold
            or defaults >= 1
            or (
                emp_status == "Unemployed"
                and dti > 0.8
            )
        )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "positive_factors": positive_factors,
            "requires_manual_review": requires_manual_review,
        }

    def analyze_loan(
        self,
        db: Session,
        loan_id: str,
    ) -> LoanAnalysis | None:
        """
        Fetch a loan record and perform deterministic risk analysis.

        Returns None when the loan does not exist.

        HTTP error handling is intentionally not performed here.
        The controller/router layer handles HTTP status codes.
        """

        record = loan_repository.get_full_analysis_record(
            db,
            loan_id,
        )

        if record is None:
            return None

        scoring = self.calculate_risk_score(record)

        return LoanAnalysis(
            loan_id=loan_id,
            loan_status=record["loan_details"]["loan_status"],
            financial_risk=scoring["risk_level"],
            risk_score=scoring["risk_score"],
            risk_factors=scoring["risk_factors"],
            positive_factors=scoring["positive_factors"],
            requires_manual_review=scoring["requires_manual_review"],
        )

loan_service = LoanService()