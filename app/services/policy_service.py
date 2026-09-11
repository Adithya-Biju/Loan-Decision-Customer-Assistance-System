from app.agents.policy_agent import run_policy_agent
from app.models.schemas import RAGResponse


class PolicyService:
    """
    Application service for policy and knowledge operations.

    Responsibilities:
        - Execute Agent 2
        - Return structured policy responses

    The actual RAG logic remains inside the Policy Agent.
    """

    def query_policy(self, query: str) -> RAGResponse:
        return run_policy_agent(query)


policy_service = PolicyService()