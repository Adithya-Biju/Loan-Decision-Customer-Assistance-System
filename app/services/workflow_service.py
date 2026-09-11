from app.graph.loan_workflow import run_loan_workflow
from app.guardrails.input_guardrail import validate_user_input


class WorkflowService:

    def run(
        self,
        query: str,
        session_id: str,
        loan_id: str | None = None,
    ):
        is_allowed, error = validate_user_input(query)

        if not is_allowed:
            return {
                "error": error,
                "guardrail_triggered": True,
            }

        return run_loan_workflow(
            query=query,
            session_id=session_id,
            loan_id=loan_id,
        )


workflow_service = WorkflowService()