from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from app.config import get_settings
from app.store import make_client, make_store


@lru_cache
def get_client() -> QdrantClient:
    return make_client(get_settings())


@lru_cache
def get_store() -> QdrantVectorStore:
    s = get_settings()
    embeddings = OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key or None)
    return make_store(get_client(), s.collection, embeddings)


@lru_cache
def get_llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(model=s.chat_model, temperature=0, api_key=s.openai_api_key or None)
