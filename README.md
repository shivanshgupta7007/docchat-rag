# DocChat - RAG Document Q&A

**Live demo:** https://docchat-rag-eno1.onrender.com (free tier - first load after idle takes ~30-60s)

Upload PDFs, then ask questions and get answers **grounded in the documents**, with page-level citations. Built to demonstrate a production-style Retrieval-Augmented Generation pipeline: FastAPI backend, LangChain-style chunking, ChromaDB vector store, Gemini embeddings + generation, and a Streamlit chat UI.

![Architecture](docs/architecture.md)

## What it does

1. **Ingest** - a PDF is split into pages, cleaned, and cut into overlapping chunks (~800 chars, 150-char overlap) that respect paragraph boundaries.
2. **Index** - each chunk is embedded with Gemini's `text-embedding-004` and stored in ChromaDB with its source file and page number.
3. **Answer** - a question is embedded, the top-k most similar chunks are retrieved, and Gemini (`gemini-2.0-flash`) answers using *only* those chunks, citing them as `[1]`, `[2]`, ... The API returns the sources alongside the answer so every claim is checkable.

If the documents don't contain the answer, DocChat says so instead of guessing.

## Stack

Python · FastAPI · ChromaDB · Gemini API (embeddings + generation) · Streamlit · pytest

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your GEMINI_API_KEY from https://aistudio.google.com/apikey
```

## Run the API

```bash
uvicorn app.main:app --reload
# POST /documents (multipart PDF upload)
# POST /ask  {"question": "What does the refund policy say?"}
# GET  /health
```

## Run the UI

```bash
streamlit run ui/streamlit_app.py
```

## Tests

```bash
pytest          # chunker and prompt-assembly tests (no API key needed)
```

## Design notes

- **Asymmetric retrieval**: queries and documents are embedded with different task types (`retrieval_query` vs `retrieval_document`) - measurably better recall.
- **Overlap**: the tail of each chunk repeats at the start of the next, so ideas straddling a boundary survive.
- **Citations by construction**: metadata (file, page) travels with every chunk from extraction to the answer payload.
- **Lazy imports**: heavy dependencies load only at runtime, keeping the unit tests fast and dependency-light.
