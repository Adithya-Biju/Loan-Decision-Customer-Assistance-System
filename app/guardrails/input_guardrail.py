"""
Input guardrails for user requests.

These checks run BEFORE the request reaches any LLM.

The goal is to prevent users from requesting sensitive customer
information such as Aadhaar, phone numbers, email addresses, PINs, etc.
"""

import re


SENSITIVE_REQUEST_PATTERNS = [
    r"\baadhaar\b",
    r"\baadhar\b",
    r"\baadhaar\s*(number|no)\b",
    r"\baadhar\s*(number|no)\b",
    r"\bphone\s*(number|no)\b",
    r"\bmobile\s*(number|no)\b",
    r"\bemail\s*(address|id)?\b",
    r"\bp(in|incode)\b",
    r"\bpostal\s*code\b",
]


SENSITIVE_REQUEST_MESSAGE = (
    "I can't provide or retrieve sensitive personal information "
    "such as Aadhaar numbers, phone numbers, email addresses, or PINs."
)


def validate_user_input(query: str) -> tuple[bool, str | None]:
    """
    Validate a user query before it reaches an LLM.

    Returns:
        (True, None) if the request is allowed.
        (False, error_message) if the request should be blocked.
    """

    normalized_query = query.strip().lower()

    if not normalized_query:
        return False, "Query cannot be empty."

    for pattern in SENSITIVE_REQUEST_PATTERNS:
        if re.search(pattern, normalized_query):
            return False, SENSITIVE_REQUEST_MESSAGE

    return True, None