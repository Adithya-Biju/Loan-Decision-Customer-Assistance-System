from app.graph.loan_workflow import run_loan_workflow


class WorkflowService:
    """
    Application service for the end-to-end LangGraph workflow.
    """

    def run(
        self,
        query: str,
        session_id: str,
        loan_id: str | None = None,
    ):
        return run_loan_workflow(
            query=query,
            session_id=session_id,
            loan_id=loan_id,
        )


workflow_service = WorkflowService()