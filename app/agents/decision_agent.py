import json
import re

from app.core.llm import get_agent_chat_model
from app.models.schemas import (
    CustomerGuidance,
    FinalRecommendation,
    RAGResponse,
    LoanAnalysis,
)
from app.agents.loan_analysis_agent import run_loan_analysis
from app.agents.policy_agent import run_policy_agent
from app.tools.postgres_tools import get_customer_feedback


LOAN_ID_PATTERN = re.compile(r"^HDFC\d{6}$")


SYSTEM_PROMPT = """
You are the Decision Review & Customer Agent for HDFC's loan intelligence
system.

You are the final interaction layer for loan officers and customers.

You combine information from:
- Loan Analysis Agent
- Policy & Knowledge Agent
- Customer feedback

You NEVER invent financial facts or policy claims.

Every factual statement about an application must come from the
Loan Analysis Agent.

Every policy statement must come from the Policy & Knowledge Agent.

Rules:

1. For a specific loan_id, always obtain Loan Analysis Agent output first.

2. For decision-review questions such as:
   - why was this loan rejected
   - why is this application risky
   - explain this decision

   retrieve relevant policy information as well.

3. For customer-improvement questions such as:
   - what can I improve
   - what should I fix before applying again

   use financial analysis, relevant policy information, and customer
   feedback when available.

4. Never expose:
   - Aadhaar
   - phone number
   - email
   - PIN code
   - customer name
   - religion
   - gender

5. Never invent a loan decision.

6. This system produces an AI-assisted recommendation for human validation,
   not a final lending decision.

7. For decision-review responses, clearly state when human review is required.

8. Customer-facing responses must be supportive and must not guarantee
   future approval.

9. Only use facts contained in the supplied Agent 1, Agent 2, and feedback
   outputs.
"""


def _extract_loan_id(text: str) -> str | None:
    match = re.search(r"\bHDFC\d{6}\b", text.upper())

    if match:
        return match.group(0)

    return None


def _is_customer_improvement_query(query: str) -> bool:
    lowered = query.lower()

    patterns = [
        "what can i improve",
        "what should i improve",
        "how can i improve",
        "improve before applying",
        "improve my application",
        "improve before reapplying",
        "what should i fix",
        "what can i fix",
    ]

    return any(pattern in lowered for pattern in patterns)


def _is_decision_review_query(query: str) -> bool:
    lowered = query.lower()

    patterns = [
        "why was",
        "why is",
        "why did",
        "why rejected",
        "why was this rejected",
        "why is this risky",
        "explain the decision",
        "explain this decision",
        "review this loan",
    ]

    return any(pattern in lowered for pattern in patterns)


def _contains_prompt_injection(text: str) -> bool:
    patterns = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(the\s+)?system\s+prompt",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"show\s+(me\s+)?your\s+instructions",
        r"developer\s+message",
        r"system\s+message",
    ]

    lowered = text.lower()

    return any(
        re.search(pattern, lowered)
        for pattern in patterns
    )


def determine_escalation(
    loan_analysis: LoanAnalysis,
    policy_response: RAGResponse | None,
    query: str,
) -> tuple[bool, str | None]:

    if loan_analysis.risk_score >= 80:
        return (
            True,
            "Computed risk score meets the high-risk escalation threshold.",
        )

    if loan_analysis.risk_score >= 60:
        if loan_analysis.requires_manual_review:
            return (
                True,
                "The loan analysis requires manual review.",
            )

    if policy_response is not None:
        if policy_response.coverage in {
            "not_covered",
            "partially_supported",
        }:
            return (
                True,
                "Policy coverage is insufficient to fully ground the explanation.",
            )

    lowered = query.lower()

    dispute_patterns = [
        "that's not my cibil",
        "that is not my cibil",
        "wrong cibil",
        "incorrect cibil",
        "my cibil is wrong",
        "wrong information",
        "incorrect information",
    ]

    if any(pattern in lowered for pattern in dispute_patterns):
        return (
            True,
            "The customer disputes application information.",
        )

    return False, None


def _build_policy_query(
    query: str,
    loan_analysis: LoanAnalysis,
) -> str:

    factors = ", ".join(
        loan_analysis.risk_factors
    )

    return (
        f"Explain the policy relevance of these financial risk factors: "
        f"{factors}. User question: {query}"
    )


def _build_decision_prompt(
    query: str,
    loan_analysis: LoanAnalysis,
    policy_response: RAGResponse,
    requires_human_review: bool,
    escalation_reason: str | None,
) -> str:

    policy_context = json.dumps(
        policy_response.model_dump(),
        indent=2,
    )

    analysis_context = json.dumps(
        loan_analysis.model_dump(),
        indent=2,
    )

    return f"""
You are generating a loan-officer-facing decision review.

USER QUESTION:
{query}

LOAN ANALYSIS:
{analysis_context}

POLICY RESPONSE:
{policy_context}

HUMAN REVIEW:
{requires_human_review}

ESCALATION REASON:
{escalation_reason}

Generate a concise explanation.

Requirements:

- Explain the existing application status using ONLY the supplied data.
- Connect the major risk factors to the retrieved policy information.
- Do not invent policy rules.
- Do not invent financial values.
- Do not claim that your analysis itself approved or rejected the loan.
- Clearly state whether human review is required.
- This is an AI-assisted recommendation for human validation.
"""


def _build_customer_prompt(
    query: str,
    loan_analysis: LoanAnalysis,
    policy_response: RAGResponse,
    feedback: dict | None,
) -> str:

    analysis_context = json.dumps(
        loan_analysis.model_dump(),
        indent=2,
    )

    policy_context = json.dumps(
        policy_response.model_dump(),
        indent=2,
    )

    feedback_context = json.dumps(
        feedback or {},
        indent=2,
    )

    return f"""
You are generating customer-facing guidance.

USER QUESTION:
{query}

FINANCIAL ANALYSIS:
{analysis_context}

POLICY INFORMATION:
{policy_context}

CUSTOMER FEEDBACK:
{feedback_context}

Generate 3 to 5 practical recommendations.

Requirements:

- Use only the supplied information.
- Focus on actionable improvements.
- Be supportive and respectful.
- Do not expose internal information.
- Do not mention risk scores unless necessary.
- Do not guarantee loan approval.
- Do not claim that improving one factor guarantees approval.
- Do not invent financial or policy facts.
"""


def _generate_final_recommendation(
    prompt: str,
) -> FinalRecommendation:

    model = get_agent_chat_model()

    structured_model = model.with_structured_output(
        FinalRecommendation
    )

    return structured_model.invoke(prompt)


def _generate_customer_guidance(
    prompt: str,
) -> CustomerGuidance:

    model = get_agent_chat_model()

    structured_model = model.with_structured_output(
        CustomerGuidance
    )

    return structured_model.invoke(prompt)


def run_customer_review_agent(
    query: str,
    loan_id: str | None = None,
):
    if not query.strip():
        return {
            "error": "Please provide a loan review or customer assistance question."
        }

    if _contains_prompt_injection(query):
        return {
            "error": (
                "I cannot provide internal instructions or sensitive "
                "customer information."
            )
        }

    resolved_loan_id = loan_id or _extract_loan_id(query)

    if not resolved_loan_id:
        return {
            "error": (
                "Please provide a valid loan ID so I can review "
                "the application."
            )
        }

    resolved_loan_id = resolved_loan_id.upper()

    if not LOAN_ID_PATTERN.match(resolved_loan_id):
        return {
            "error": f"'{resolved_loan_id}' is not a valid loan ID."
        }

    loan_result = run_loan_analysis(
        query=query,
        loan_id_hint=resolved_loan_id,
    )

    if loan_result.error or loan_result.loan_analysis is None:
        return {
            "error": loan_result.error
            or "Loan analysis could not be completed.",
            "loan_id": resolved_loan_id,
        }

    loan_analysis = loan_result.loan_analysis

    customer_query = _is_customer_improvement_query(query)
    decision_query = _is_decision_review_query(query)

    if customer_query:
        policy_query = (
            "How can an applicant improve the financial factors "
            "identified in this loan analysis before applying again?"
        )

        policy_response = run_policy_agent(policy_query)

        feedback_raw = get_customer_feedback.invoke(
            {"loan_id": resolved_loan_id}
        )

        try:
            feedback = json.loads(feedback_raw)
        except (json.JSONDecodeError, TypeError):
            feedback = None

        prompt = _build_customer_prompt(
            query=query,
            loan_analysis=loan_analysis,
            policy_response=policy_response,
            feedback=feedback,
        )

        return _generate_customer_guidance(prompt)

    if decision_query:
        policy_query = _build_policy_query(
            query=query,
            loan_analysis=loan_analysis,
        )

        policy_response = run_policy_agent(policy_query)

        requires_human_review, escalation_reason = (
            determine_escalation(
                loan_analysis=loan_analysis,
                policy_response=policy_response,
                query=query,
            )
        )

        prompt = _build_decision_prompt(
            query=query,
            loan_analysis=loan_analysis,
            policy_response=policy_response,
            requires_human_review=requires_human_review,
            escalation_reason=escalation_reason,
        )

        return _generate_final_recommendation(prompt)

    return {
        "error": (
            "I can help review a specific loan or provide "
            "customer improvement guidance."
        )
    }