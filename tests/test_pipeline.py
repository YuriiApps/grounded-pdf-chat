from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app import rag
from app.config import get_settings
from app.deps import get_client, get_llm, get_store
from app.ingest import chunk_pdf, ingest_dir, ingest_file, split_sections
from app.main import app
from app.store import count_points


def test_split_sections_detects_headings():
    text = "Keep Warm Mode\nPress and hold the Warm button.\n2. Refund Timing\nRefunds are issued in 5 days."
    assert split_sections(text) == [
        ("Keep Warm Mode", "Press and hold the Warm button."),
        ("2. Refund Timing", "Refunds are issued in 5 days."),
    ]


def test_chunks_keep_source_page_and_heading(settings):
    chunks = chunk_pdf(Path(settings.pdf_dir) / "brightleaf-kettle-k2-guide.pdf", settings)
    descaling = [c for c in chunks if c.metadata["section"] == "Descaling"]
    assert descaling, [c.metadata for c in chunks]
    assert descaling[0].metadata == {"source": "brightleaf-kettle-k2-guide.pdf", "page": 2, "section": "Descaling"}
    assert descaling[0].page_content.startswith("Descaling\n")


def test_reindex_replaces_instead_of_duplicating(store, client, settings):
    first = ingest_dir(store, client, settings)
    total = count_points(client, settings.collection)
    assert total == sum(first.values())
    ingest_dir(store, client, settings)
    assert count_points(client, settings.collection) == total


def test_reindex_drops_chunks_removed_from_updated_pdf(store, client, settings, tmp_path):
    path = Path(settings.pdf_dir) / "brightleaf-kettle-k2-guide.pdf"
    ingest_file(path, store, client, settings)
    before = count_points(client, settings.collection)
    # "Update" the document with a shorter one under the same name.
    (Path(settings.pdf_dir) / "brightleaf-support-faq.pdf").replace(path)
    ingest_file(path, store, client, settings)
    after = count_points(client, settings.collection)
    assert after < before
    hits = store.similarity_search("descale vinegar hard water", k=5)
    assert all("Descaling" not in h.page_content for h in hits)


def test_answer_with_citation(store, client, settings):
    ingest_dir(store, client, settings)
    llm = FakeListChatModel(responses=["Descale every 4 weeks in hard water areas [1]."])
    res = rag.answer("How often should I descale the kettle in hard water?", store, llm, settings)
    assert res["found"] is True
    assert res["sources"][0]["source"] == "brightleaf-kettle-k2-guide.pdf"
    assert res["sources"][0]["page"] == 2


def test_out_of_scope_question_never_reaches_llm(store, client, settings):
    ingest_dir(store, client, settings)
    llm = FakeListChatModel(responses=["This must not be returned [1]."])
    res = rag.answer("quantum chromodynamics lattice simulation", store, llm, settings)
    assert res == rag.not_found("quantum chromodynamics lattice simulation")


def test_llm_not_found_and_uncited_answers_are_refused(store, client, settings):
    ingest_dir(store, client, settings)
    q = "How much does express delivery cost?"
    assert rag.answer(q, store, FakeListChatModel(responses=["NOT_FOUND"]), settings)["found"] is False
    assert rag.answer(q, store, FakeListChatModel(responses=["About 10 euros."]), settings)["found"] is False
    assert rag.answer(q, store, FakeListChatModel(responses=["It costs 9.90 EUR [9]."]), settings)["found"] is False


def test_api_chat_and_ingest(store, client, settings):
    app.dependency_overrides = {
        get_store: lambda: store,
        get_client: lambda: client,
        get_settings: lambda: settings,
        get_llm: lambda: FakeListChatModel(responses=["Express delivery costs 9.90 EUR [1]."]),
    }
    try:
        api = TestClient(app)
        ingested = api.post("/ingest").json()
        assert len(ingested["files"]) == 3 and ingested["total_chunks"] > 0
        assert api.get("/health").json()["chunks"] == ingested["total_chunks"]

        res = api.post("/chat", json={"question": "How much does express delivery cost?"}).json()
        assert res["found"] is True
        assert res["sources"][0]["source"] == "brightleaf-support-faq.pdf"

        assert api.post("/chat", json={"question": ""}).status_code == 422
    finally:
        app.dependency_overrides = {}
