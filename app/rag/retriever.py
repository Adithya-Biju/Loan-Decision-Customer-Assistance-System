from app.core.config import get_settings
from app.rag.embeddings import embed_text
from app.core.qdrant import get_qdrant_client


def retrieve_policy_chunks(
    query: str,
    top_k: int = 5
) -> list[dict]:
    """
    Retrieve the most relevant policy chunks from Qdrant.
    """

    settings = get_settings()
    client = get_qdrant_client()

    query_vector = embed_text(query)

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    ).points

    chunks = []

    for result in results:
        payload = result.payload or {}

        chunks.append(
            {
                "score": result.score,
                "text": payload.get("text", ""),
                "source": payload.get("source", ""),
                "title": payload.get("title", ""),
                "section": payload.get("section", ""),
                "domain": payload.get("domain", ""),
                "version": payload.get("version", ""),
                "chunk_index": payload.get("chunk_index")
            }
        )

    return chunks