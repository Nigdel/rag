import psycopg
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import rag

app = FastAPI(title="Local RAG")


class Question(BaseModel):
    question: str


@app.post("/ask")
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question is empty.")
    try:
        return {"answer": rag.answer(body.question)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {e}")


@app.get("/health")
def health():
    try:
        with psycopg.connect(rag.DATABASE_URL) as conn:
            chunks = conn.execute("SELECT count(*) FROM documents").fetchone()[0]
        requests.get(f"{rag.OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {"status": "ok", "chunks": chunks}