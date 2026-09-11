import re

from app.core.llm import get_agent_chat_model
from app.models.schemas import RoutingDecision


LOAN_ID_PATTERN = re.compile(r"\bHDFC\d{6}\b")


ROUTER_PROMPT = """
You are the routing component of an HDFC loan intelligence system.

Your job is ONLY to classify the user's request into one of these intents:

- general:
  Greetings, thanks, casual conversation.

- loan_analysis:
  The user wants a financial/risk analysis of a specific loan application.
  Examples:
  "Analyze loan HDFC100125"
  "What's the risk of this application?"
  "Give me a financial analysis"

- decision_review:
  The user wants to understand, explain, or review an existing loan decision.
  Examples:
  "Why was this loan rejected?"
  "Why is this application risky?"
  "Explain the decision"
  "Review this loan"

- customer_assistance:
  The user wants to know how they can improve their application.
  Examples:
  "What can I improve?"
  "How can I improve my application?"
  "What should I fix before reapplying?"

- policy:
  The user asks about loan policies, eligibility, CIBIL, DTI,
  documentation, employment requirements, guarantors, etc.
  Examples:
  "What CIBIL score is required?"
  "What documents are needed?"
  "What is the DTI requirement?"

- unsupported:
  The request is unrelated to this loan intelligence system.

IMPORTANT:

1. Use the conversation context when interpreting short follow-up questions.

2. If the user says something like:
   "Analyze loan HDFC100125"
   followed by:
   "Why?"
   then "Why?" refers to HDFC100125 and should be classified as
   decision_review.

3. If a loan ID appears in the CURRENT user query, always use that loan ID.
   Never prefer an older loan ID from conversation history.

4. If the current query has no loan ID but conversation context contains
   the previously discussed loan ID, you may reuse that loan ID.

5. Do not invent loan IDs.

6. Return ONLY the structured routing decision.
"""


def route_query(
    query: str,
    conversation_context: str = "",
    previous_loan_id: str | None = None,
) -> RoutingDecision:

    model = get_agent_chat_model().with_structured_output(
        RoutingDecision
    )

    # Explicit loan ID in CURRENT query always wins.
    match = LOAN_ID_PATTERN.search(query.upper())

    current_loan_id = match.group(0) if match else None

    effective_context = conversation_context.strip()

    if not effective_context:
        effective_context = "No previous conversation context."

    prompt = f"""
{ROUTER_PROMPT}

Previous conversation:
{effective_context}

Previously active loan ID:
{previous_loan_id or "None"}

Current user query:
{query}

Remember:
- Current query loan ID > previous loan ID.
- Do not invent a loan ID.
"""

    decision = model.invoke(prompt)

    # Safety: if the user explicitly supplied an ID,
    # always preserve it.
    if current_loan_id:
        decision.loan_id = current_loan_id

    # If the model did not return an ID but we have an active one,
    # preserve the active loan for follow-up questions.
    elif decision.loan_id is None and previous_loan_id:
        decision.loan_id = previous_loan_id

    return decision