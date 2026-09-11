"""
PII protection for LLM-facing data.

Sensitive customer identifiers must never be exposed to the LLM.
Database/repository code may still contain these fields, but anything
returned to an LLM tool is sanitized first.
"""

from typing import Any


SENSITIVE_FIELDS = {
    "aadhaar",
    "aadhar",
    "aadhaar_number",
    "aadhar_number",
    "phone",
    "phone_number",
    "mobile",
    "mobile_number",
    "email",
    "email_address",
    "pin",
    "pincode",
    "postal_code",
}


def sanitize_for_llm(data: Any) -> Any:
    """
    Recursively remove sensitive fields from data before it is
    passed into an LLM context.
    """

    if isinstance(data, dict):
        return {
            key: sanitize_for_llm(value)
            for key, value in data.items()
            if key.lower() not in SENSITIVE_FIELDS
        }

    if isinstance(data, list):
        return [sanitize_for_llm(item) for item in data]

    if isinstance(data, tuple):
        return tuple(sanitize_for_llm(item) for item in data)

    return data