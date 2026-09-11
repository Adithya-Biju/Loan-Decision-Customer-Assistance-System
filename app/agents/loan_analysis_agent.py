"""
Agent 1 -- Loan Analysis Agent.

Division of responsibility, deliberately narrow:
  - The LLM's ONLY job is retrieval: read the user's query, figure out which
    loan_id it refers to (genuine agentic parsing -- no regex extraction
    from free text, see postgres_tools.py and the conversation that led
    here), and call whichever of the 5 Postgres tools are needed.
  - The LLM NEVER states a risk score, risk level, or conclusion. This
    isn't just a prompt instruction -- calculate_risk_score is not exposed
    to it as a callable tool at all, so there's no path by which the LLM's
    own judgment becomes the risk number. That number always comes from
    app.services.loan_service.calculate_risk_score, deterministically, in
    this file's code, after the ReAct loop finishes.

Why no narrative text field: the spec's own "Expected Output" for Agent 1
is pure structured JSON, no prose explanation. Customer/officer-friendly
narrative is explicitly Agent 3's job. Keeping Agent 1 output-only-schema
also sidesteps a real failure mode: if we let the LLM freely narrate a risk
opinion in the same turn, it could state a conclusion that disagrees with
the deterministic score (e.g. LLM says "seems moderate risk" while the
rubric says HIGH) -- a self-inconsistency the critic would have to catch
later. Not producing that text at all is more robust than trying to
detect the mismatch after the fact.
"""
import json
from dataclasses import dataclass, field

from langgraph.prebuilt import create_react_agent

from app.core.llm import get_agent_chat_model
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.models.schemas import LoanAnalysis
from app.repositories import loan_repository
from app.services import loan_service
from app.tools.postgres_tools import LOAN_ANALYSIS_TOOLS, LOAN_ID_PATTERN

settings = get_settings()

SYSTEM_PROMPT = """You are the Loan Analysis Agent for HDFC's loan intelligence system.

Your ONLY job is retrieval. You never state, imply, or guess a risk score,
risk level, or overall conclusion about an application -- that is computed
separately by deterministic code after you finish retrieving data. If the
user asks "is this risky" or "why was this rejected", your job is still
just to retrieve the relevant fields, not to answer that question yourself.

Rules:
1. Identify which loan_id the user's question refers to. Loan IDs look like
   "HDFC100245". If the user doesn't mention one and none is available from
   context, say so and ask for it -- do not guess or invent one.
2. Always call get_loan_details first to confirm the loan_id exists.
3. For a full analysis request (e.g. "analyze this loan", "why is this
   risky", "what are the risk indicators"), call ALL of: get_loan_details,
   get_customer_financials, get_credit_history, get_previous_loans, and
   get_employment_details. Do not stop after only one or two tools for a
   full-analysis request.
4. For a narrow question (e.g. "what's the CIBIL score on this loan"), you
   may call only the relevant tool(s).
5. If a tool returns an error message (e.g. "no loan found"), do not
   retry with a guessed or corrected loan_id on your own -- report the
   error plainly so the user can confirm the correct ID.
6. Never mention or ask about Aadhaar, phone number, email, PIN code,
   religion, or gender -- these tools don't expose them and you should
   never imply you have access to them.
7. When you've finished retrieving what's needed, give a brief one-line
   confirmation of what you retrieved. Do not summarize the financial
   picture or offer an opinion on risk.
"""


@dataclass
class LoanAnalysisAgentResult:
    loan_id: str | None
    loan_analysis: LoanAnalysis | None
    tools_called: list[str] = field(default_factory=list)
    gap_filled_tools: list[str] = field(default_factory=list)
    agent_message: str = ""
    error: str | None = None


_RETRIEVAL_TOOL_NAMES = [
    "get_loan_details",
    "get_customer_financials",
    "get_credit_history",
    "get_previous_loans",
    "get_employment_details",
]


def build_loan_analysis_agent():
    """
    Builds the compiled LangGraph ReAct agent. Kept as a separate function
    (rather than a module-level singleton) so tests can construct a fresh
    instance, and so a future graph-orchestration layer can build this once
    and reuse it as a node.
    """
    model = get_agent_chat_model()
    return create_react_agent(model, tools=LOAN_ANALYSIS_TOOLS, state_modifier=SYSTEM_PROMPT)


def _extract_tool_results(messages: list) -> tuple[str | None, dict[str, dict]]:
    """
    Walks the agent's message history and pulls out:
      - the loan_id actually used (from the first tool call's arguments --
        this is the LLM's own extraction, not a regex match)
      - a dict of {tool_name: parsed_result} for every successful tool call

    Tool messages that come back as plain error strings (not valid JSON,
    per postgres_tools.py's convention of json.dumps() on success only)
    are skipped here -- they indicate the tool didn't find data, which the
    caller handles separately.
    """
    loan_id: str | None = None
    results: dict[str, dict] = {}

    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls and loan_id is None:
            for call in tool_calls:
                args = call.get("args", {})
                candidate = args.get("loan_id")
                if candidate:
                    loan_id = candidate.strip().upper()
                    break

        msg_type = getattr(msg, "type", None)
        if msg_type == "tool":
            tool_name = getattr(msg, "name", None)
            content = getattr(msg, "content", None)
            if tool_name in _RETRIEVAL_TOOL_NAMES and isinstance(content, str):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        results[tool_name] = parsed
                except (json.JSONDecodeError, TypeError):
                    pass  # error string from the tool, not a data payload

    return loan_id, results


def _fill_gaps(loan_id: str, results: dict[str, dict]) -> tuple[dict[str, dict], list[str]]:
    """
    Defensive completeness check: if the LLM didn't call every retrieval
    tool it should have for a full analysis, fetch the missing pieces
    directly through the repository. This guards against an unreliable or
    under-eager model (relevant for a free-tier model of unverified
    consistency) silently producing an incomplete risk analysis.

    Returns the completed results dict and the list of tool names that had
    to be filled in this way, so callers/observability can see when the
    LLM under-performed its own system prompt instructions.
    """
    gap_filled: list[str] = []
    db = SessionLocal()
    try:
        fetchers = {
            "get_loan_details": loan_repository.get_loan_details,
            "get_customer_financials": loan_repository.get_customer_financials,
            "get_credit_history": loan_repository.get_credit_history,
            "get_previous_loans": loan_repository.get_previous_loans,
            "get_employment_details": loan_repository.get_employment_details,
        }
        for tool_name, fetch_fn in fetchers.items():
            if tool_name not in results:
                fetched = fetch_fn(db, loan_id)
                if fetched is not None:
                    results[tool_name] = fetched
                    gap_filled.append(tool_name)
    finally:
        db.close()

    return results, gap_filled


def run_loan_analysis(query: str, loan_id_hint: str | None = None) -> LoanAnalysisAgentResult:
    """
    Main entrypoint. `loan_id_hint` lets a caller that already has a
    validated loan_id (e.g. the /api/v1/loan/analyze endpoint, where the
    ID arrived as an explicit, regex-validated request field -- see
    app/controllers/loan_controller.py) pass it along as context, without
    bypassing the agent's own tool-calling loop. The agent still decides
    which tools to call and in what order; it's simply told which
    application the question concerns instead of having to parse it out of
    a bare loan_id-only string, which is an awkward thing to ask an LLM to
    "converse" about.
    """
    effective_query = query
    if loan_id_hint:
        if not LOAN_ID_PATTERN.match(loan_id_hint.strip().upper()):
            return LoanAnalysisAgentResult(
                loan_id=None,
                loan_analysis=None,
                error=f"'{loan_id_hint}' is not a valid loan ID format.",
            )
        effective_query = f"{query}\n\n(The application in question is {loan_id_hint.strip().upper()}.)"

    agent = build_loan_analysis_agent()

    try:
        output = agent.invoke(
            {"messages": [("user", effective_query)]},
            config={"recursion_limit": settings.agent_max_tool_iterations * 2 + 2},
        )
    except Exception as exc:  
        return LoanAnalysisAgentResult(
            loan_id=loan_id_hint,
            loan_analysis=None,
            error=f"Agent execution failed: {exc}",
        )

    messages = output.get("messages", [])
    loan_id, results = _extract_tool_results(messages)
    loan_id = loan_id or loan_id_hint

    final_ai_text = ""
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "ai" and getattr(msg, "content", None):
            final_ai_text = msg.content
            break

    if loan_id is None:
        return LoanAnalysisAgentResult(
            loan_id=None,
            loan_analysis=None,
            tools_called=list(results.keys()),
            agent_message=final_ai_text,
            error="The agent could not identify which loan this question refers to.",
        )

    if "get_loan_details" not in results and not results:
        # No tool succeeded at all -- most likely the loan_id doesn't exist.
        return LoanAnalysisAgentResult(
            loan_id=loan_id,
            loan_analysis=None,
            agent_message=final_ai_text,
            error=f"No loan application found with ID '{loan_id}'.",
        )

    completed_results, gap_filled = _fill_gaps(loan_id, results)

    required = {"financials": completed_results.get("get_customer_financials"),
                "credit": completed_results.get("get_credit_history"),
                "employment": completed_results.get("get_employment_details")}
    if any(v is None for v in required.values()):
        return LoanAnalysisAgentResult(
            loan_id=loan_id,
            loan_analysis=None,
            tools_called=list(results.keys()),
            gap_filled_tools=gap_filled,
            agent_message=final_ai_text,
            error=f"Incomplete data for loan '{loan_id}' -- cannot compute a risk score.",
        )

    scoring = loan_service.calculate_risk_score(
        {
            "financials": required["financials"],
            "credit": required["credit"],
            "employment": required["employment"],
        }
    )

    loan_analysis = LoanAnalysis(
        loan_id=loan_id,
        financial_risk=scoring["risk_level"],
        risk_score=scoring["risk_score"],
        risk_factors=scoring["risk_factors"],
        positive_factors=scoring["positive_factors"],
        requires_manual_review=scoring["requires_manual_review"],
    )

    return LoanAnalysisAgentResult(
        loan_id=loan_id,
        loan_analysis=loan_analysis,
        tools_called=list(results.keys()),
        gap_filled_tools=gap_filled,
        agent_message=final_ai_text,
    )
