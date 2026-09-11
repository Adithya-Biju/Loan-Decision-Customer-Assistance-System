from enum import Enum

from pydantic import BaseModel, Field

from app.core.llm import get_agent_chat_model


class ValidationResult(str, Enum):
    FULLY_SUPPORTED = "fully_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    NOT_COVERED = "not_covered"


class RAGValidation(BaseModel):
    result: ValidationResult
    reason: str = Field(
        description="Brief explanation for the classification."
    )


def validate_rag_answer(
    query: str,
    answer: str,
    chunks: list[dict],
) -> RAGValidation:

    if not chunks:
        return RAGValidation(
            result=ValidationResult.NOT_COVERED,
            reason="No policy information was retrieved.",
        )

    context = "\n\n---\n\n".join(
        chunk.get("text", "").strip()
        for chunk in chunks
        if chunk.get("text", "").strip()
    )

    model = get_agent_chat_model()

    structured_model = model.with_structured_output(RAGValidation)

    prompt = f"""
You are the critic/validator for a policy RAG system.

Determine whether the generated answer is supported ONLY by the
retrieved knowledge-base context.

USER QUESTION:
{query}

GENERATED ANSWER:
{answer}

RETRIEVED KNOWLEDGE:
{context}

CLASSIFICATION RULES:

FULLY_SUPPORTED:
Every important claim in the answer is directly supported by the
retrieved knowledge.

PARTIALLY_SUPPORTED:
Some important claims are supported, but at least one important
claim is not adequately supported.

NOT_COVERED:
The retrieved knowledge does not adequately answer the question.

IMPORTANT:
- Do not use outside knowledge.
- Do not judge whether the answer is generally true.
- Judge only whether the retrieved context supports the answer.
- Ignore similarity scores.
- Return a short reason explaining your classification.
"""

    return structured_model.invoke(prompt)