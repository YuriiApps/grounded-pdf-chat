# Grounded PDF Chat

A RAG chatbot that answers **only from your PDFs**, cites the document and page for every answer, and says
"not found" instead of guessing.

**Stack:** FastAPI · LangChain · Qdrant · OpenAI · Streamlit · Docker

The repo ships with three sample PDFs from a fictional company (Brightleaf Home: a kettle guide, a returns and
warranty policy, a support FAQ), so it runs without any real data.

## How it works

```
PDF ──► pages ──► sections (split at headings) ──► chunks + metadata ──► Qdrant
                                                   {source, page, section}

question ──► top-k search ──► drop chunks below score threshold ──► LLM with numbered passages
                                     │ nothing left                         │
                                     ▼                                      ▼
                                 "not found"            answer with [1] [2] citations ──► sources (file, page)
```

Accuracy and anti-hallucination decisions:

- **Heading-aware chunking.** Text is split at headings first, then by size. Each chunk starts with its heading,
  so a chunk like "every 4 weeks in hard water areas" still knows it is about *Descaling*.
- **Citations from metadata.** Every chunk carries the file name and page number; the API returns them with the answer.
- **Two "not found" guards.** If no passage passes the similarity threshold, the LLM is never called. If the LLM
  answers without citing a passage, the answer is not shown.
- **Re-indexing replaces.** Chunk IDs are derived from file + page + position, and old chunks of a file are deleted
  before it is indexed again, so an updated PDF never leaves its old version searchable.
- **Measured, not assumed.** `eval/` runs a question set with expected sources, including questions that are *not*
  in the documents.

## Run with Docker

```bash
cp .env.example .env          # add your OPENAI_API_KEY
python scripts/make_sample_pdfs.py   # or put your own PDFs into data/pdfs/
docker compose up --build
curl -X POST http://localhost:8000/ingest
```

- Chat UI: http://localhost:8501
- API docs: http://localhost:8000/docs

## Run without Docker

Qdrant runs embedded (data in `./qdrant_data`) when `QDRANT_URL` is empty.

```bash
python -m venv .venv && .venv/Scripts/activate    # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                              # add your OPENAI_API_KEY
python scripts/make_sample_pdfs.py
python -m app.ingest
uvicorn app.main:app --reload
streamlit run ui/streamlit_app.py                 # in a second terminal
```

## API

| Method | Path | What it does |
|---|---|---|
| `POST` | `/chat` | `{"question": "..."}` → `answer`, `found`, `sources[]` (file, page, section, score, snippet) |
| `POST` | `/ingest` | Re-indexes every PDF in `PDF_DIR` |
| `GET` | `/health` | Status and number of indexed chunks |

Example response (the shape is exact, the values are illustrative):

```json
{
  "question": "How much does express delivery cost?",
  "answer": "Express delivery takes 1 to 2 business days and costs 9.90 EUR [1].",
  "found": true,
  "sources": [{"ref": 1, "source": "brightleaf-support-faq.pdf", "page": 1, "section": "Delivery", "score": 0.61, "snippet": "..."}]
}
```

## Evaluation

```bash
python eval/run_eval.py      # API must be running
```

Writes `eval/report.md` with three scores: correct source cited, expected fact present in the answer, and
out-of-scope questions answered with "not found". Add your own cases to `eval/questions.json`.

## Tests

The pipeline is tested offline, without an OpenAI key: a deterministic bag-of-words embedding and a fake chat model
stand in for OpenAI, and Qdrant runs in memory.

```bash
pip install pytest httpx
pytest
```

## Configuration

All settings are environment variables (see `.env.example`): models, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`,
`SCORE_THRESHOLD`, Qdrant URL or local path.
