"""Offline test doubles: the pipeline runs end to end without an OpenAI key."""
import hashlib
import math
import re
import sys
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from app.config import Settings  # noqa: E402
from app.store import make_client, make_store  # noqa: E402
from make_sample_pdfs import build  # noqa: E402

STOP = {"the", "a", "an", "is", "of", "to", "and", "in", "for", "on", "it", "what", "how", "do", "does", "i", "my", "can", "be"}


class HashEmbeddings(Embeddings):
    """Bag-of-words vectors hashed into a fixed size. Crude, but deterministic and meaningful enough for tests."""

    dim = 512

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in STOP:
                continue
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


@pytest.fixture
def settings(tmp_path) -> Settings:
    pdf_dir = tmp_path / "pdfs"
    build(pdf_dir)
    return Settings(qdrant_url="", qdrant_path=":memory:", pdf_dir=str(pdf_dir), score_threshold=0.15, top_k=3)


@pytest.fixture
def client(settings):
    return make_client(settings)


@pytest.fixture
def store(client, settings):
    return make_store(client, settings.collection, HashEmbeddings())
