from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_settings


@lru_cache
def get_qdrant_client() -> QdrantClient:
    settings = get_settings()

    return QdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120
    )


def ensure_collection() -> None:
    settings = get_settings()
    client = get_qdrant_client()

    collections = client.get_collections().collections

    if settings.qdrant_collection not in {
        collection.name for collection in collections
    }:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dimension,
                distance=Distance.COSINE,
            ),
        )