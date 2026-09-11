from app.agents.decision_agent import run_customer_review_agent
from app.agents.loan_analysis_agent import run_loan_analysis
from app.agents.policy_agent import run_policy_agent

from app.models.schemas import (
    CustomerGuidance,
    FinalRecommendation,
)


class DecisionService:
    """
    Application service for decision review and customer assistance.

    Responsible for coordinating:
        Agent 1 -> Agent 2 -> Agent 3

    Loan ID extraction and validation are handled by the
    application workflow/agent layer, not the HTTP router.
    """

    def review(
        self,
        query: str,
    ) -> FinalRecommendation | CustomerGuidance:

        analysis_result = run_loan_analysis(
            query=query,
        )

        if (
            analysis_result.error
            or analysis_result.loan_analysis is None
        ):
            raise ValueError(
                analysis_result.error
                or "Loan analysis failed."
            )

        loan_analysis = analysis_result.loan_analysis

        loan_id = loan_analysis.loan_id

        policy_response = run_policy_agent(
            query,
        )

        result = run_customer_review_agent(
            query=query,
            loan_id=loan_id,
            loan_analysis=loan_analysis,
            policy_response=policy_response,
        )

        if isinstance(result, dict) and result.get("error"):
            raise ValueError(
                result["error"]
            )

        return result


decision_service = DecisionService()