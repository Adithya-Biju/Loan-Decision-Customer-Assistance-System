"""
Agent 1 -- Loan Analysis Agent.

Division of responsibility, deliberately narrow:

- The LLM's ONLY job is retrieval:
    - understand the user's query
    - identify which loan_id the user is referring to
    - call the required PostgreSQL retrieval tools

- The LLM NEVER calculates or states:
    - risk score
    - risk level
    - final lending conclusion

Risk scoring is performed deterministically by LoanService after
the retrieval loop finishes.

This keeps the risk calculation:
- reproducible
- unit-testable
- independent of LLM behavior
- safe from LLM hallucination
"""

import json
from dataclasses import dataclass, field

from langgraph.prebuilt import create_react_agent

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.llm import get_agent_chat_model

from app.models.schemas import LoanAnalysis

from app.repositories import loan_repository

# IMPORTANT:
# Import the SERVICE INSTANCE, not the loan_service module.
from app.services.loan_service import loan_service

from app.tools.postgres_tools import (
    LOAN_ANALYSIS_TOOLS,
    LOAN_ID_PATTERN,
)


settings = get_settings()


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Loan Analysis Agent for HDFC's loan intelligence system.

Your ONLY job is retrieval.

You never calculate, state, imply, or guess:
- a risk score
- a risk level
- an approval/rejection conclusion
- an overall financial risk opinion

Those are calculated separately by deterministic Python code after
you finish retrieving the required data.

If the user asks:
- "Is this risky?"
- "Why was this rejected?"
- "Will this loan be approved?"

you STILL only retrieve the relevant information.

Rules:

1. IDENTIFY THE LOAN

Identify which loan_id the user's question refers to.

Loan IDs look like:

HDFC100245

If the user does not mention a loan ID and none is available from
conversation context, do not guess or invent one.

2. CONFIRM THE LOAN

Always call get_loan_details first when a loan ID is available.

3. FULL ANALYSIS

For a full analysis request such as:

- "Analyze this loan"
- "Why is this loan risky?"
- "What are the risk indicators?"
- "Why was this application rejected?"

retrieve ALL of:

- get_loan_details
- get_customer_financials
- get_credit_history
- get_previous_loans
- get_employment_details

Do not stop after only one or two tools.

4. NARROW QUESTIONS

For a narrow question such as:

- "What's the CIBIL score?"
- "What's the employment status?"

you may call only the relevant retrieval tool(s).

5. TOOL ERRORS

If a tool reports that the loan does not exist:

- do not guess another ID
- do not modify the ID
- do not retry with an invented ID

Report the error plainly.

6. SENSITIVE INFORMATION

Never mention or ask about:

- Aadhaar
- phone number
- email
- PIN code
- religion
- gender

These fields are intentionally not exposed.

7. NO RISK OPINION

After retrieval, provide only a short confirmation of what
information was retrieved.

Do not summarize the financial picture.

Do not calculate risk.

Do not provide a lending recommendation.
"""


# ============================================================
# RESULT OBJECT
# ============================================================

@dataclass
class LoanAnalysisAgentResult:
    loan_id: str | None
    loan_analysis: LoanAnalysis | None

    tools_called: list[str] = field(default_factory=list)

    gap_filled_tools: list[str] = field(default_factory=list)

    agent_message: str = ""

    error: str | None = None


# ============================================================
# REQUIRED RETRIEVAL TOOLS
# ============================================================

_RETRIEVAL_TOOL_NAMES = [
    "get_loan_details",
    "get_customer_financials",
    "get_credit_history",
    "get_previous_loans",
    "get_employment_details",
]


# ============================================================
# BUILD AGENT
# ============================================================

def build_loan_analysis_agent():
    """
    Build the compiled LangGraph ReAct agent.

    The agent only has access to PostgreSQL retrieval tools.
    calculate_risk_score is deliberately NOT exposed as an LLM tool.
    """

    model = get_agent_chat_model()

    return create_react_agent(
        model,
        tools=LOAN_ANALYSIS_TOOLS,
        state_modifier=SYSTEM_PROMPT,
    )


# ============================================================
# EXTRACT TOOL RESULTS
# ============================================================

def _extract_tool_results(
    messages: list,
) -> tuple[str | None, dict[str, dict]]:
    """
    Extract:

    1. The loan ID selected by the LLM from its tool calls.
    2. Successful PostgreSQL tool results.

    The loan ID comes from the actual tool call arguments rather than
    manually extracting it from the user's free-form query.
    """

    loan_id: str | None = None

    results: dict[str, dict] = {}

    for msg in messages:

        # --------------------------------------------------------
        # Extract loan ID from tool call arguments
        # --------------------------------------------------------

        tool_calls = getattr(msg, "tool_calls", None)

        if tool_calls and loan_id is None:

            for call in tool_calls:

                args = call.get("args", {})

                candidate = args.get("loan_id")

                if candidate:

                    loan_id = candidate.strip().upper()

                    break

        # --------------------------------------------------------
        # Extract successful tool result
        # --------------------------------------------------------

        msg_type = getattr(msg, "type", None)

        if msg_type != "tool":
            continue

        tool_name = getattr(msg, "name", None)

        content = getattr(msg, "content", None)

        if (
            tool_name in _RETRIEVAL_TOOL_NAMES
            and isinstance(content, str)
        ):

            try:

                parsed = json.loads(content)

                if isinstance(parsed, dict):

                    results[tool_name] = parsed

            except (json.JSONDecodeError, TypeError):

                # Tool errors are plain strings rather than
                # JSON payloads, so they are intentionally ignored.
                pass

    return loan_id, results


# ============================================================
# GAP FILLING
# ============================================================

def _fill_gaps(
    loan_id: str,
    results: dict[str, dict],
) -> tuple[dict[str, dict], list[str]]:
    """
    Defensive completeness check.

    If the LLM fails to call one of the retrieval tools required
    for analysis, fetch the missing information directly from the
    repository.

    This does NOT calculate risk.

    It only guarantees that deterministic scoring receives the
    required data.
    """

    gap_filled: list[str] = []

    db = SessionLocal()

    try:

        fetchers = {
            "get_loan_details":
                loan_repository.get_loan_details,

            "get_customer_financials":
                loan_repository.get_customer_financials,

            "get_credit_history":
                loan_repository.get_credit_history,

            "get_previous_loans":
                loan_repository.get_previous_loans,

            "get_employment_details":
                loan_repository.get_employment_details,
        }

        for tool_name, fetch_fn in fetchers.items():

            if tool_name in results:
                continue

            fetched = fetch_fn(
                db,
                loan_id,
            )

            if fetched is not None:

                results[tool_name] = fetched

                gap_filled.append(tool_name)

    finally:

        db.close()

    return results, gap_filled


# ============================================================
# MAIN AGENT ENTRYPOINT
# ============================================================

def run_loan_analysis(
    query: str,
    loan_id_hint: str | None = None,
) -> LoanAnalysisAgentResult:
    """
    Run Agent 1.

    Parameters
    ----------
    query:
        Natural-language user query.

    loan_id_hint:
        Optional validated loan ID supplied by another layer,
        such as LangGraph conversation state.

    The LLM still performs the retrieval tool calls.

    The deterministic risk calculation is performed only after
    retrieval has completed.
    """

    # --------------------------------------------------------
    # Validate optional loan ID hint
    # --------------------------------------------------------

    effective_query = query

    if loan_id_hint:

        normalized_loan_id = loan_id_hint.strip().upper()

        if not LOAN_ID_PATTERN.fullmatch(normalized_loan_id):

            return LoanAnalysisAgentResult(
                loan_id=None,
                loan_analysis=None,
                error=(
                    f"'{loan_id_hint}' is not a valid loan ID format."
                ),
            )

        effective_query = (
            f"{query}\n\n"
            f"The application in question is {normalized_loan_id}."
        )

    # --------------------------------------------------------
    # Build Agent
    # --------------------------------------------------------

    try:

        agent = build_loan_analysis_agent()

    except Exception as exc:

        return LoanAnalysisAgentResult(
            loan_id=loan_id_hint,
            loan_analysis=None,
            error=f"Failed to initialize loan analysis agent: {exc}",
        )

    # --------------------------------------------------------
    # Run ReAct retrieval loop
    # --------------------------------------------------------

    try:

        output = agent.invoke(
            {
                "messages": [
                    ("user", effective_query)
                ]
            },
            config={
                "recursion_limit": (
                    settings.agent_max_tool_iterations * 2 + 2
                )
            },
        )

    except Exception as exc:

        return LoanAnalysisAgentResult(
            loan_id=loan_id_hint,
            loan_analysis=None,
            error=f"Agent execution failed: {exc}",
        )

    # --------------------------------------------------------
    # Extract results
    # --------------------------------------------------------

    messages = output.get(
        "messages",
        [],
    )

    loan_id, results = _extract_tool_results(
        messages
    )

    loan_id = loan_id or loan_id_hint

    # --------------------------------------------------------
    # Extract final AI confirmation
    # --------------------------------------------------------

    final_ai_text = ""

    for msg in reversed(messages):

        if (
            getattr(msg, "type", None) == "ai"
            and getattr(msg, "content", None)
        ):

            final_ai_text = msg.content

            break

    # --------------------------------------------------------
    # No loan identified
    # --------------------------------------------------------

    if loan_id is None:

        return LoanAnalysisAgentResult(
            loan_id=None,
            loan_analysis=None,
            tools_called=list(results.keys()),
            agent_message=final_ai_text,
            error=(
                "The agent could not identify which loan "
                "this question refers to."
            ),
        )

    # --------------------------------------------------------
    # No successful retrieval
    # --------------------------------------------------------

    if not results:

        return LoanAnalysisAgentResult(
            loan_id=loan_id,
            loan_analysis=None,
            tools_called=[],
            agent_message=final_ai_text,
            error=(
                f"No loan application found with ID "
                f"'{loan_id}'."
            ),
        )

    # --------------------------------------------------------
    # Fill missing retrieval data
    # --------------------------------------------------------

    completed_results, gap_filled = _fill_gaps(
        loan_id,
        results,
    )

    # --------------------------------------------------------
    # Required deterministic scoring inputs
    # --------------------------------------------------------

    financials = completed_results.get(
        "get_customer_financials"
    )

    credit = completed_results.get(
        "get_credit_history"
    )

    employment = completed_results.get(
        "get_employment_details"
    )

    if any(
        value is None
        for value in (
            financials,
            credit,
            employment,
        )
    ):

        return LoanAnalysisAgentResult(
            loan_id=loan_id,
            loan_analysis=None,
            tools_called=list(results.keys()),
            gap_filled_tools=gap_filled,
            agent_message=final_ai_text,
            error=(
                f"Incomplete data for loan '{loan_id}' -- "
                "cannot compute a risk score."
            ),
        )

    # --------------------------------------------------------
    # Loan details are required for final LoanAnalysis
    # --------------------------------------------------------

    loan_details = completed_results.get(
        "get_loan_details"
    )

    if loan_details is None:

        return LoanAnalysisAgentResult(
            loan_id=loan_id,
            loan_analysis=None,
            tools_called=list(results.keys()),
            gap_filled_tools=gap_filled,
            agent_message=final_ai_text,
            error=(
                f"Loan details are missing for loan "
                f"'{loan_id}'."
            ),
        )

    # --------------------------------------------------------
    # DETERMINISTIC RISK CALCULATION
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # The LLM does not perform this calculation.
    #
    # We call the LoanService instance here.
    # --------------------------------------------------------

    scoring = loan_service.calculate_risk_score(
        {
            "financials": financials,
            "credit": credit,
            "employment": employment,
        }
    )

    # --------------------------------------------------------
    # Build final structured Agent 1 result
    # --------------------------------------------------------

    loan_analysis = LoanAnalysis(
        loan_id=loan_id,

        loan_status=loan_details["loan_status"],

        financial_risk=scoring["risk_level"],

        risk_score=scoring["risk_score"],

        risk_factors=scoring["risk_factors"],

        positive_factors=scoring["positive_factors"],

        requires_manual_review=scoring[
            "requires_manual_review"
        ],
    )

    # --------------------------------------------------------
    # Return
    # --------------------------------------------------------

    return LoanAnalysisAgentResult(
        loan_id=loan_id,

        loan_analysis=loan_analysis,

        tools_called=list(results.keys()),

        gap_filled_tools=gap_filled,

        agent_message=final_ai_text,

        error=None,
    )
