def rerank_policy_chunks(
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict]:
    """
    Deduplicate and select the strongest policy chunks.

    Current implementation:
    - removes duplicate chunks
    - sorts by Qdrant similarity score
    - returns top_k
    """

    seen = set()
    unique_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        # Use source + section + text to identify duplicates
        key = (
            chunk.get("source", ""),
            chunk.get("section", ""),
            text,
        )

        if key in seen:
            continue

        seen.add(key)
        unique_chunks.append(chunk)

    unique_chunks.sort(
        key=lambda x: x.get("score", 0.0),
        reverse=True,
    )

    return unique_chunks[:top_k] 