"""FastAPI surface for DocChat.

POST /documents  - upload a PDF, returns how many chunks were indexed
POST /ask        - {question} -> {answer, sources}
GET  /health     - liveness + index size
"""
from __future__ import annotations

import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .rag import RAGPipeline

app = FastAPI(title="DocChat - RAG document Q&A", version="1.0.0")
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health() -> dict:
    try:
        count = get_pipeline().store.count()
    except Exception:
        count = -1  # pipeline not initialised yet (e.g. missing API key)
    return {"status": "ok", "chunks_indexed": count}


@app.post("/documents")
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    contents = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        added = get_pipeline().ingest_pdf(tmp_path, source_name=file.filename)
    finally:
        os.unlink(tmp_path)
    return {"indexed_chunks": added, "source": file.filename}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    return get_pipeline().ask(req.question)
