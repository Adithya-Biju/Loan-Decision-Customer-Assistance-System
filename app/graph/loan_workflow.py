import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents.decision_agent import run_customer_review_agent
from app.agents.loan_analysis_agent import run_loan_analysis
from app.agents.policy_agent import run_policy_agent
from app.graph.state import AgentState


LOAN_ID_PATTERN = re.compile(r"\bHDFC\d{6}\b")


def router_node(state: AgentState) -> AgentState:

    query = state["query"]
    loan_id = state.get("loan_id")

    if not loan_id:
        match = LOAN_ID_PATTERN.search(query.upper())

        if match:
            loan_id = match.group(0)

    lowered = query.lower()

    # -------------------------
    # GENERAL
    # -------------------------

    general_patterns = [
        "hello",
        "hi",
        "hey",
        "thanks",
        "thank you",
        "good morning",
        "good afternoon",
        "good evening",
    ]

    if any(pattern in lowered for pattern in general_patterns):
        return {
            "loan_id": loan_id,
            "mode": "general",
        }

    # -------------------------
    # CUSTOMER ASSISTANCE
    # -------------------------

    improvement_patterns = [
        "what can i improve",
        "what should i improve",
        "how can i improve",
        "improve before applying",
        "improve my application",
        "improve before reapplying",
        "what should i fix",
        "what can i fix",
    ]

    if any(pattern in lowered for pattern in improvement_patterns):

        if not loan_id:
            return {
                "mode": "unsupported",
                "error": (
                    "Please provide a loan ID for "
                    "customer-specific improvement guidance."
                ),
            }

        return {
            "loan_id": loan_id,
            "mode": "customer_assistance",
        }

    # -------------------------
    # DECISION REVIEW
    # -------------------------

    decision_patterns = [
        "why was",
        "why is",
        "why did",
        "why rejected",
        "explain the decision",
        "explain this decision",
        "review this loan",
        "review the loan",
        "review this application",
        "why is this risky",
    ]

    if any(pattern in lowered for pattern in decision_patterns):

        if not loan_id:
            return {
                "mode": "unsupported",
                "error": "Please provide a loan ID for decision review.",
            }

        return {
            "loan_id": loan_id,
            "mode": "decision_review",
        }

    # -------------------------
    # LOAN ANALYSIS
    # -------------------------

    analysis_patterns = [
        "analyze loan",
        "analyze this loan",
        "analyze the loan",
        "analyze application",
        "analyze this application",
        "analyze the application",
        "risk analysis",
        "financial analysis",
    ]

    if any(pattern in lowered for pattern in analysis_patterns):

        if not loan_id:
            return {
                "mode": "unsupported",
                "error": "Please provide a loan ID for loan analysis.",
            }

        return {
            "loan_id": loan_id,
            "mode": "loan_analysis",
        }

    # -------------------------
    # POLICY
    # -------------------------

    policy_patterns = [
        "policy",
        "eligibility",
        "eligible",
        "cibil",
        "credit score",
        "debt to income",
        "dti",
        "loan to income",
        "lti",
        "documents",
        "documentation",
        "guarantor",
        "employment requirement",
    ]

    if any(pattern in lowered for pattern in policy_patterns):

        return {
            "loan_id": loan_id,
            "mode": "policy",
        }

    # -------------------------
    # UNKNOWN
    # -------------------------

    return {
        "loan_id": loan_id,
        "mode": "unsupported",
        "error": (
            "I can help analyze a specific loan, "
            "answer policy questions, or review a loan decision."
        ),
    }


def general_node(state: AgentState) -> AgentState:
    return {
        "final_answer": (
            "Hello! I can help you analyze a loan application, "
            "answer loan policy questions, or provide guidance "
            "on improving an application."
        )
    }


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


def policy_node(state: AgentState) -> AgentState:
    loan_analysis = state.get("loan_analysis")

    if state["mode"] == "policy":
        try:
            policy_response = run_policy_agent(state["query"])
        except Exception as exc:
            return {
                "error": f"Policy retrieval failed: {exc}",
            }

        return {
            "policy_response": policy_response,
        }

    if loan_analysis is None:
        return {
            "error": "Loan analysis is required before policy retrieval."
        }

    if state["mode"] == "customer_assistance":
        policy_query = (
            "How can an applicant improve the financial factors "
            "identified in this loan analysis before applying again?"
        )
    else:
        risk_factors = ", ".join(loan_analysis.risk_factors)

        policy_query = (
            f"Explain the policy relevance of these loan risk factors: "
            f"{risk_factors}. "
            f"User question: {state['query']}"
        )

    try:
        policy_response = run_policy_agent(policy_query)
    except Exception as exc:
        return {
            "error": f"Policy retrieval failed: {exc}",
        }

    return {
        "policy_response": policy_response,
    }


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


def route_after_router(state: AgentState) -> str:
    return state["mode"]


def should_continue_after_analysis(state: AgentState) -> str:

    if state.get("error"):
        return "error"

    if state["mode"] == "loan_analysis":
        return "final"

    return "policy"

def should_continue_after_policy(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    if state["mode"] == "policy":
        return "final"

    return "decision"


def should_continue_after_decision(state: AgentState) -> str:
    if state.get("error"):
        return "error"

    final_recommendation = state.get("final_recommendation")

    if (
        final_recommendation
        and final_recommendation.requires_human_review
    ):
        return "human_review"

    return "final"


def error_node(state: AgentState) -> AgentState:
    return state


def human_review_node(state: AgentState) -> AgentState:
    return {
        "requires_human_review": True,
    }


def final_node(state: AgentState) -> AgentState:

    if state["mode"] == "loan_analysis":
        loan_analysis = state.get("loan_analysis")

        if loan_analysis:
            return {
                "final_answer": (
                    f"Loan {loan_analysis.loan_id}\n\n"
                    f"Status: {loan_analysis.loan_status}\n"
                    f"Risk Level: {loan_analysis.financial_risk.value}\n"
                    f"Risk Score: {loan_analysis.risk_score}/100\n\n"
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

    if state["mode"] == "policy":
        policy_response = state.get("policy_response")

        if policy_response:
            return {
                "final_answer": policy_response.answer
            }

    if state["mode"] == "customer_assistance":
        customer_guidance = state.get("customer_guidance")

        if customer_guidance:
            return {
                "final_answer": "\n".join(
                    customer_guidance.recommendations
                )
            }

    if state["mode"] == "decision_review":
        final_recommendation = state.get("final_recommendation")

        if final_recommendation:
            return {
                "final_answer": final_recommendation.decision_summary
            }

    return state


def build_loan_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("router", router_node)
    graph.add_node("general", general_node)
    graph.add_node("loan_analysis_agent", loan_analysis_node)
    graph.add_node("policy_agent", policy_node)
    graph.add_node("decision_agent", decision_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("final", final_node)
    graph.add_node("workflow_error", error_node)

    graph.add_edge(START, "router")

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

    graph.add_edge("general", END)

    graph.add_conditional_edges(
    "loan_analysis_agent",
    should_continue_after_analysis,
        {
        "policy": "policy_agent",
        "final": "final",
        "error": "workflow_error",
        },
    )

    graph.add_conditional_edges(
        "policy_agent",
        should_continue_after_policy,
        {
            "decision": "decision_agent",
            "final": "final",
            "error": "workflow_error",
        },
    )

    graph.add_conditional_edges(
        "decision_agent",
        should_continue_after_decision,
        {
            "human_review": "human_review",
            "final": "final",
            "error": "workflow_error",
        },
    )

    graph.add_edge("human_review", "final")
    graph.add_edge("final", END)
    graph.add_edge("workflow_error", END)

    checkpointer = MemorySaver()

    return graph.compile(
        checkpointer=checkpointer,
    )


loan_workflow = build_loan_workflow()


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