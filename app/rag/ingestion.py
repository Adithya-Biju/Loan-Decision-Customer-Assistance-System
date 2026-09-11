from pathlib import Path
from uuid import uuid4

from qdrant_client.models import PointStruct

from app.core.config import get_settings
from app.rag.embeddings import embed_texts
from app.core.qdrant import ensure_collection, get_qdrant_client


KNOWLEDGE_BASE_DIR = Path("data/knowledge_base")

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120


def split_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    """
    Simple semantic-ish chunking.

    We first preserve Markdown sections and then split large
    sections into overlapping chunks.
    """

    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= chunk_size:
            current = f"{current}\n\n{paragraph}".strip()
        else:
            if current:
                chunks.append(current)

            # Keep overlap from previous chunk
            if overlap > 0 and current:
                current = current[-overlap:] + "\n\n" + paragraph
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def extract_metadata(file_path: Path, chunk: str) -> dict:
    """
    Extract basic metadata from a Markdown knowledge-base file.
    """

    lines = chunk.splitlines()

    title = file_path.stem.replace("_", " ").title()
    section = "General"

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("# "):
            title = stripped[2:].strip()

        elif stripped.startswith("## "):
            section = stripped[3:].strip()

    return {
        "source": file_path.name,
        "title": title,
        "section": section,
        "domain": file_path.stem,
        "version": "1.0",
    }


def load_documents() -> list[dict]:
    """
    Load all Markdown documents from the knowledge base
    and convert them into chunks with metadata.
    """

    documents = []

    files = sorted(KNOWLEDGE_BASE_DIR.glob("*.md"))

    if not files:
        raise RuntimeError(
            f"No Markdown files found in {KNOWLEDGE_BASE_DIR}"
        )

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")

        chunks = split_text(text)

        for chunk_index, chunk in enumerate(chunks):
            metadata = extract_metadata(file_path, chunk)

            metadata["chunk_index"] = chunk_index

            documents.append(
                {
                    "text": chunk,
                    "metadata": metadata,
                }
            )

    return documents


def ingest_knowledge_base() -> None:
    """
    Embed all knowledge-base chunks and store them in Qdrant.
    """

    settings = get_settings()

    print("Loading knowledge base...")

    documents = load_documents()

    print(f"Documents/chunks loaded: {len(documents)}")

    texts = [document["text"] for document in documents]

    print("Generating embeddings...")

    vectors = embed_texts(texts)

    print(f"Generated {len(vectors)} embeddings")

    ensure_collection()

    client = get_qdrant_client()

    points = []

    for document, vector in zip(documents, vectors):
        payload = {
            "text": document["text"],
            **document["metadata"],
        }

        points.append(
            PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload=payload,
            )
        )

    print("Uploading vectors to Qdrant...")

    client.upsert(
        collection_name=settings.qdrant_collection,
        points=points,
    )

    print()
    print("Knowledge base ingestion complete.")
    print(f"Collection : {settings.qdrant_collection}")
    print(f"Vectors    : {len(points)}")
    print(f"Dimensions : {settings.embedding_dimension}")


if __name__ == "__main__":
    ingest_knowledge_base()