from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # Qdrant server URL (Docker). When empty, Qdrant runs embedded and stores data in qdrant_path.
    qdrant_url: str = ""
    qdrant_path: str = "./qdrant_data"
    collection: str = "docs"

    pdf_dir: str = "./data/pdfs"
    chunk_size: int = 900
    chunk_overlap: int = 150

    top_k: int = 5
    # Cosine similarity below this is treated as "not relevant". Passages under it never reach the LLM.
    score_threshold: float = 0.3


@lru_cache
def get_settings() -> Settings:
    return Settings()
