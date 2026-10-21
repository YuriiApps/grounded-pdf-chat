import re

from langchain_core.language_models import BaseChatModel
from langchain_qdrant import QdrantVectorStore

from app.config import Settings

NOT_FOUND = "I couldn't find the answer to this in the documents."

SYSTEM_PROMPT = """You answer questions for a support team using ONLY the numbered passages provided.

Rules:
- Every sentence of your answer must be supported by a passage. Cite it right after the sentence as [1], [2], etc.
- Do not use outside knowledge, and do not guess.
- If the passages do not contain the answer, reply with exactly: NOT_FOUND
- Keep answers short and factual."""


def format_context(hits) -> str:
    return "\n\n".join(
        f"[{i}] ({doc.metadata['source']}, page {doc.metadata['page']})\n{doc.page_content}"
        for i, (doc, _score) in enumerate(hits, start=1)
    )


def not_found(question: str) -> dict:
    return {"question": question, "answer": NOT_FOUND, "found": False, "sources": []}


def answer(question: str, store: QdrantVectorStore, llm: BaseChatModel, s: Settings) -> dict:
    hits = store.similarity_search_with_score(question, k=s.top_k)
    # Guard 1: nothing relevant retrieved -> say so without asking the LLM.
    hits = [(doc, score) for doc, score in hits if score >= s.score_threshold]
    if not hits:
        return not_found(question)

    reply = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Passages:\n{format_context(hits)}\n\nQuestion: {question}"),
        ]
    )
    text = str(reply.content).strip()
    if text.startswith("NOT_FOUND"):
        return not_found(question)

    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 1 <= int(n) <= len(hits)})
    # Guard 2: an answer that cites nothing is not grounded, so it is not shown.
    if not cited:
        return not_found(question)

    sources = []
    for n in cited:
        doc, score = hits[n - 1]
        sources.append(
            {
                "ref": n,
                "source": doc.metadata["source"],
                "page": doc.metadata["page"],
                "section": doc.metadata.get("section", ""),
                "score": round(float(score), 3),
                "snippet": doc.page_content[:300],
            }
        )
    return {"question": question, "answer": text, "found": True, "sources": sources}
