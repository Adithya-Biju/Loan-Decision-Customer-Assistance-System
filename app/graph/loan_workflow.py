from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.decision_agent import run_customer_review_agent
from app.agents.loan_analysis_agent import run_loan_analysis
from app.agents.policy_agent import run_policy_agent
from app.graph.router import route_query
from app.graph.state import AgentState


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> AgentState:

    query = state["query"]

    # Loan ID stored in the checkpoint from a previous turn.
    previous_loan_id = state.get("loan_id")

    # -----------------------------------------------------------------------
    # Build conversation context.
    #
    # The router needs previous messages for questions such as:
    #
    # User: Analyze loan HDFC100125
    # User: Why?
    #
    # The second query has no loan ID, so the router needs the context.
    # -----------------------------------------------------------------------

    messages = state.get("messages", [])

    conversation_context = "\n".join(
        f"{message.__class__.__name__}: {message.content}"
        for message in messages[-6:]
    )

    try:
        decision = route_query(
            query=query,
            conversation_context=conversation_context,
            previous_loan_id=previous_loan_id,
        )

    except Exception as exc:

        return {
            "mode": "unsupported",
            "error": f"Unable to determine request intent: {exc}",
        }

    loan_id = decision.loan_id
    mode = decision.intent

    # -----------------------------------------------------------------------
    # These operations require a specific loan.
    # -----------------------------------------------------------------------

    if mode in {
        "loan_analysis",
        "decision_review",
        "customer_assistance",
    } and not loan_id:

        return {
            "mode": "unsupported",
            "error": (
                "Please provide a valid HDFC loan ID "
                "for this request."
            ),
        }

    return {
        "loan_id": loan_id,
        "mode": mode,
        "error": None,
    }


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

def general_node(state: AgentState) -> AgentState:

    return {
        "final_answer": (
            "Hello! I can help you analyze a loan application, "
            "answer loan policy questions, or provide guidance "
            "on improving an application."
        )
    }


# ---------------------------------------------------------------------------
# Agent 1 -- Loan Analysis
# ---------------------------------------------------------------------------

def loan_analysis_node(state: AgentState) -> AgentState:

    loan_id = state.get("loan_id")

    if not loan_id:
        return {
            "error": "A valid loan ID is required for loan analysis."
        }

    result = run_loan_analysis(
        query=state["query"],
        loan_id_hint=loan_id,
    )

    if result.error:
        return {
            "loan_analysis": None,
            "error": result.error,
        }

    return {
        "loan_id": result.loan_id,
        "loan_analysis": result.loan_analysis,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Agent 2 -- Policy & Knowledge
# ---------------------------------------------------------------------------

def policy_node(state: AgentState) -> AgentState:

    loan_analysis = state.get("loan_analysis")

    # ---------------------------------------------------------
    # Standalone policy question.
    # ---------------------------------------------------------

    if state["mode"] == "policy":

        try:
            policy_response = run_policy_agent(
                state["query"]
            )

        except Exception as exc:

            return {
                "error": f"Policy retrieval failed: {exc}",
            }

        return {
            "policy_response": policy_response,
        }

    # ---------------------------------------------------------
    # Customer assistance / decision review both require
    # Agent 1 analysis first.
    # ---------------------------------------------------------

    if loan_analysis is None:

        return {
            "error": "Loan analysis is required before policy retrieval."
        }

    # ---------------------------------------------------------
    # Customer improvement guidance.
    # ---------------------------------------------------------

    if state["mode"] == "customer_assistance":

        policy_query = (
            "How can an applicant improve the financial factors "
            "identified in this loan analysis before applying again?"
        )

    # ---------------------------------------------------------
    # Decision review.
    # ---------------------------------------------------------

    else:

        risk_factors = ", ".join(
            loan_analysis.risk_factors
        )

        policy_query = (
            f"Explain the policy relevance of these loan risk factors: "
            f"{risk_factors}. "
            f"User question: {state['query']}"
        )

    try:

        policy_response = run_policy_agent(
            policy_query
        )

    except Exception as exc:

        return {
            "error": f"Policy retrieval failed: {exc}",
        }

    return {
        "policy_response": policy_response,
    }


# ---------------------------------------------------------------------------
# Agent 3 -- Decision Review & Customer
# ---------------------------------------------------------------------------

def decision_node(state: AgentState) -> AgentState:

    loan_id = state.get("loan_id")

    if not loan_id:

        return {
            "error": "A valid loan ID is required."
        }

    result = run_customer_review_agent(
        query=state["query"],
        loan_id=loan_id,
        loan_analysis=state.get("loan_analysis"),
        policy_response=state.get("policy_response"),
    )

    if isinstance(result, dict) and result.get("error"):

        return {
            "error": result["error"],
        }

    if state["mode"] == "customer_assistance":

        return {
            "customer_guidance": result,
        }

    return {
        "final_recommendation": result,
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def route_after_router(state: AgentState) -> str:

    return state["mode"]


def should_continue_after_analysis(state: AgentState) -> str:

    if state.get("error"):
        return "error"

    # Pure loan analysis does NOT need Agents 2 or 3.
    if state["mode"] == "loan_analysis":
        return "final"

    # Decision review / customer assistance need policy context.
    return "policy"


def should_continue_after_policy(state: AgentState) -> str:

    if state.get("error"):
        return "error"

    # Standalone policy question is already complete.
    if state["mode"] == "policy":
        return "final"

    # Otherwise Agent 3 combines Agent 1 + Agent 2.
    return "decision"


def should_continue_after_decision(state: AgentState) -> str:

    if state.get("error"):
        return "error"

    final_recommendation = state.get(
        "final_recommendation"
    )

    if (
        final_recommendation
        and final_recommendation.requires_human_review
    ):
        return "human_review"

    return "final"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def error_node(state: AgentState) -> AgentState:

    return state


# ---------------------------------------------------------------------------
# Human review
# ---------------------------------------------------------------------------

def human_review_node(state: AgentState) -> AgentState:

    return {
        "requires_human_review": True,
    }


# ---------------------------------------------------------------------------
# Final response
# ---------------------------------------------------------------------------

def final_node(state: AgentState) -> AgentState:

    # ---------------------------------------------------------
    # Loan analysis
    # ---------------------------------------------------------

    if state["mode"] == "loan_analysis":

        loan_analysis = state.get("loan_analysis")

        if loan_analysis:

            return {
                "final_answer": (
                    f"Loan {loan_analysis.loan_id}\n\n"

                    f"Status: "
                    f"{loan_analysis.loan_status}\n"

                    f"Risk Level: "
                    f"{loan_analysis.financial_risk.value}\n"

                    f"Risk Score: "
                    f"{loan_analysis.risk_score}/100\n\n"

                    f"Risk Factors:\n"

                    + "\n".join(
                        f"- {factor}"
                        for factor in loan_analysis.risk_factors
                    )

                    + "\n\n"

                    f"Positive Factors:\n"

                    + "\n".join(
                        f"- {factor}"
                        for factor in loan_analysis.positive_factors
                    )

                    + "\n\n"

                    f"Manual Review Required: "
                    f"{'Yes' if loan_analysis.requires_manual_review else 'No'}"
                )
            }

    # ---------------------------------------------------------
    # Policy
    # ---------------------------------------------------------

    if state["mode"] == "policy":

        policy_response = state.get(
            "policy_response"
        )

        if policy_response:

            return {
                "final_answer": policy_response.answer
            }

    # ---------------------------------------------------------
    # Customer assistance
    # ---------------------------------------------------------

    if state["mode"] == "customer_assistance":

        customer_guidance = state.get(
            "customer_guidance"
        )

        if customer_guidance:

            return {
                "final_answer": "\n".join(
                    customer_guidance.recommendations
                )
            }

    # ---------------------------------------------------------
    # Decision review
    # ---------------------------------------------------------

    if state["mode"] == "decision_review":

        final_recommendation = state.get(
            "final_recommendation"
        )

        if final_recommendation:

            return {
                "final_answer": (
                    final_recommendation.decision_summary
                )
            }

    return state


# ---------------------------------------------------------------------------
# Build LangGraph
# ---------------------------------------------------------------------------

def build_loan_workflow():

    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node(
        "loan_analysis_agent",
        loan_analysis_node,
    )
    graph.add_node(
        "policy_agent",
        policy_node,
    )
    graph.add_node(
        "decision_agent",
        decision_node,
    )
    graph.add_node(
        "human_review",
        human_review_node,
    )
    graph.add_node(
        "final",
        final_node,
    )
    graph.add_node(
        "workflow_error",
        error_node,
    )

    # START -> Router
    graph.add_edge(
        START,
        "router",
    )

    # Router -> appropriate path
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "general": "general",
            "policy": "policy_agent",
            "loan_analysis": "loan_analysis_agent",
            "decision_review": "loan_analysis_agent",
            "customer_assistance": "loan_analysis_agent",
            "unsupported": "workflow_error",
        },
    )

    # General -> END
    graph.add_edge(
        "general",
        END,
    )

    # Agent 1 -> Agent 2 OR Final
    graph.add_conditional_edges(
        "loan_analysis_agent",
        should_continue_after_analysis,
        {
            "policy": "policy_agent",
            "final": "final",
            "error": "workflow_error",
        },
    )

    # Agent 2 -> Agent 3 OR Final
    graph.add_conditional_edges(
        "policy_agent",
        should_continue_after_policy,
        {
            "decision": "decision_agent",
            "final": "final",
            "error": "workflow_error",
        },
    )

    # Agent 3 -> Human Review OR Final
    graph.add_conditional_edges(
        "decision_agent",
        should_continue_after_decision,
        {
            "human_review": "human_review",
            "final": "final",
            "error": "workflow_error",
        },
    )

    # Human Review -> Final
    graph.add_edge(
        "human_review",
        "final",
    )

    # Final -> END
    graph.add_edge(
        "final",
        END,
    )

    # Error -> END
    graph.add_edge(
        "workflow_error",
        END,
    )

    # Checkpointing / conversation memory
    checkpointer = MemorySaver()

    return graph.compile(
        checkpointer=checkpointer,
    )


loan_workflow = build_loan_workflow()


# ---------------------------------------------------------------------------
# Public workflow entry point
# ---------------------------------------------------------------------------

def run_loan_workflow(
    query: str,
    session_id: str,
    loan_id: str | None = None,
):

    initial_state: AgentState = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ],
        "query": query,
        "mode": "",
        "retry_count": 0,
    }

    if loan_id:
        initial_state["loan_id"] = loan_id

    return loan_workflow.invoke(
        initial_state,
        config={
            "configurable": {
                "thread_id": session_id,
            }
        },
    )
