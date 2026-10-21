from fastapi import Depends, FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_qdrant import QdrantVectorStore
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from app import rag
from app.config import Settings, get_settings
from app.deps import get_client, get_llm, get_store
from app.ingest import ingest_dir
from app.store import count_points

app = FastAPI(title="Grounded PDF Chat", version="1.0.0")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class Source(BaseModel):
    ref: int
    source: str
    page: int
    section: str
    score: float
    snippet: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    found: bool
    sources: list[Source]


@app.get("/health")
def health(
    client: QdrantClient = Depends(get_client),
    store: QdrantVectorStore = Depends(get_store),
    s: Settings = Depends(get_settings),
):
    return {"status": "ok", "chunks": count_points(client, s.collection)}


@app.post("/chat", response_model=ChatResponse)
def chat(
    req: ChatRequest,
    store: QdrantVectorStore = Depends(get_store),
    llm: BaseChatModel = Depends(get_llm),
    s: Settings = Depends(get_settings),
):
    return rag.answer(req.question, store, llm, s)


@app.post("/ingest")
def ingest(
    store: QdrantVectorStore = Depends(get_store),
    client: QdrantClient = Depends(get_client),
    s: Settings = Depends(get_settings),
):
    files = ingest_dir(store, client, s)
    return {"files": files, "total_chunks": count_points(client, s.collection)}
