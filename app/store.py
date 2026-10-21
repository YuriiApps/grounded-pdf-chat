import warnings

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from app.config import Settings

SOURCE_KEY = "metadata.source"


def make_client(s: Settings) -> QdrantClient:
    if s.qdrant_url:
        return QdrantClient(url=s.qdrant_url)
    if s.qdrant_path == ":memory:":
        return QdrantClient(location=":memory:")
    return QdrantClient(path=s.qdrant_path)


def ensure_collection(client: QdrantClient, name: str, embeddings: Embeddings) -> None:
    if client.collection_exists(name):
        return
    dim = len(embeddings.embed_query("dimension probe"))
    client.create_collection(
        name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    # Index on the file name: fast deletes on re-index and filtering by document.
    # Embedded Qdrant ignores payload indexes and warns about it, so the warning is silenced here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        client.create_payload_index(name, SOURCE_KEY, models.PayloadSchemaType.KEYWORD)


def make_store(client: QdrantClient, name: str, embeddings: Embeddings) -> QdrantVectorStore:
    ensure_collection(client, name, embeddings)
    return QdrantVectorStore(client=client, collection_name=name, embedding=embeddings)


def delete_source(client: QdrantClient, name: str, source: str) -> None:
    client.delete(
        name,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[models.FieldCondition(key=SOURCE_KEY, match=models.MatchValue(value=source))]
            )
        ),
    )


def count_points(client: QdrantClient, name: str) -> int:
    return client.count(name, exact=True).count
