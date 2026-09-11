import json

from langchain_core.tools import tool

from app.rag.retriever import retrieve_policy_chunks
from app.rag.reranker import rerank_policy_chunks


@tool
def search_policy_knowledge(query: str) -> str:
    """
    Search the policy knowledge base for information relevant to a
    loan policy or eligibility question.

    Returns relevant policy excerpts together with source metadata
    required for citations.
    """

    chunks = retrieve_policy_chunks(
        query=query,
        top_k=5,
    )

    chunks = rerank_policy_chunks(
        chunks,
        top_k=3,
    )

    if not chunks:
        return json.dumps({
            "results": [],
            "message": (
                "No relevant policy information was found "
                "in the knowledge base."
            ),
        })

    results = []

    for chunk in chunks:
        results.append({
            "text": chunk.get("text", "").strip(),
            "source": chunk.get("source", ""),
            "title": chunk.get("title", ""),
            "section": chunk.get("section", ""),
            "chunk_id": (
                str(chunk.get("chunk_id"))
                if chunk.get("chunk_id") is not None
                else None
            ),
            "score": chunk.get("score", 0.0),
            "domain": chunk.get("domain", ""),
            "version": chunk.get("version", ""),
        })

    return json.dumps({
        "results": results
    })