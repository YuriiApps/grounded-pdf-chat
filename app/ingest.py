"""PDF -> sections -> chunks -> Qdrant.

Run directly to index every PDF in PDF_DIR:  python -m app.ingest
"""
import re
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from qdrant_client import QdrantClient

from app.config import Settings
from app.store import delete_source

# A short line without closing punctuation, optionally numbered ("2.1 Returns"), starting with a capital.
HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][^.!?:;]{2,70}$")


def read_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(str(path))
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def split_sections(text: str, carry: str = "") -> list[tuple[str, str]]:
    """Split page text at heading-like lines. `carry` is the heading still open from the previous page."""
    sections: list[tuple[str, list[str]]] = [(carry, [])]
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if HEADING.match(line) and len(line.split()) <= 10:
            sections.append((line, []))
        else:
            sections[-1][1].append(line)
    return [(title, " ".join(body)) for title, body in sections if body]


def chunk_pdf(path: Path, s: Settings) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = []
    heading = ""
    for page_no, text in read_pages(path):
        for title, body in split_sections(text, carry=heading):
            heading = title
            for piece in splitter.split_text(body):
                # The heading travels with every chunk: it carries meaning the body text often lacks.
                content = f"{title}\n{piece}" if title else piece
                chunks.append(
                    Document(
                        page_content=content,
                        metadata={"source": path.name, "page": page_no, "section": title},
                    )
                )
    return chunks


def chunk_id(doc: Document, index: int) -> str:
    key = f"{doc.metadata['source']}:{doc.metadata['page']}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def ingest_file(path: Path, store: QdrantVectorStore, client: QdrantClient, s: Settings) -> int:
    chunks = chunk_pdf(path, s)
    # Replace, never append: an updated PDF must not leave its old version searchable.
    delete_source(client, s.collection, path.name)
    if chunks:
        store.add_documents(chunks, ids=[chunk_id(c, i) for i, c in enumerate(chunks)])
    return len(chunks)


def ingest_dir(store: QdrantVectorStore, client: QdrantClient, s: Settings) -> dict[str, int]:
    pdfs = sorted(Path(s.pdf_dir).glob("*.pdf"))
    return {p.name: ingest_file(p, store, client, s) for p in pdfs}


if __name__ == "__main__":
    from app.config import get_settings
    from app.deps import get_client, get_store

    settings = get_settings()
    result = ingest_dir(get_store(), get_client(), settings)
    for name, n in result.items():
        print(f"{name}: {n} chunks")
    print(f"Indexed {len(result)} files.")
