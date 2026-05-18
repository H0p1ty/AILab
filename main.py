import uuid

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse as _JSONResponse
from pydantic import BaseModel

from rag.embedder import EmbeddingError, embed
from rag.llm import LLMError, generate
from rag.pdf_loader import chunk_text, load_pdf
from rag.retriever import add_chunks, delete_doc, list_doc_ids, retrieve
from rag import session_store


class JSONResponse(_JSONResponse):
    media_type = "application/json; charset=utf-8"


app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation over PDFs using Llama 3.2 + ChromaDB",
    default_response_class=JSONResponse,
)


def build_search_query(question: str, history: list[dict]) -> str:
    """Augment search query with recent conversation context for better retrieval on follow-ups."""
    if len(history) < 2:
        return question
    last_answer = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    return f"{last_answer[:300]} {question}" if last_answer else question


# ---------- request / response models ----------

class QueryRequest(BaseModel):
    question: str
    n_results: int = 5
    session_id: str | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    session_id: str


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_indexed: int
    session_id: str


# ---------- endpoints ----------

@app.post("/upload", response_model=UploadResponse, summary="Upload a PDF and index it")
async def upload_pdf(file: UploadFile = File(...), session_id: str | None = Form(None)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    content = await file.read()
    text = load_pdf(content)
    if not text.strip():
        raise HTTPException(status_code=422, detail="Could not extract text from the PDF.")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=422, detail="PDF produced no text chunks after processing.")

    try:
        embeddings = await embed(chunks)
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Is it running on localhost:11434?")

    sid = session_id or str(uuid.uuid4())
    session_store.create_session(sid)
    doc_id = str(uuid.uuid4())
    add_chunks(doc_id, sid, chunks, embeddings)

    return UploadResponse(doc_id=doc_id, filename=file.filename, chunks_indexed=len(chunks), session_id=sid)


@app.post("/query", response_model=QueryResponse, summary="Ask a question against indexed documents")
async def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    # Create or retrieve session
    sid = req.session_id or str(uuid.uuid4())
    session_store.create_session(sid)
    history = session_store.get_history(sid)

    # Expand search query with recent context for better retrieval on follow-ups
    search_query = build_search_query(req.question, history)

    try:
        q_embedding = await embed([search_query])
    except EmbeddingError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Is it running on localhost:11434?")

    sources = retrieve(q_embedding[0], sid, n_results=req.n_results)

    try:
        answer = await generate(req.question, sources, history)
    except LLMError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Cannot reach Ollama. Is it running on localhost:11434?")

    session_store.append_messages(sid, [
        {"role": "user", "content": req.question},
        {"role": "assistant", "content": answer},
    ])

    return QueryResponse(answer=answer, sources=sources, session_id=sid)


@app.get("/documents/{session_id}", summary="List indexed document IDs")
def get_documents(session_id: str):
    return {"documents": list_doc_ids(session_id)}


@app.delete("/documents/{session_id}/{doc_id}", summary="Remove a document and its chunks from the index")
def remove_document(session_id: str, doc_id: str):
    delete_doc(session_id, doc_id)
    return {"deleted": {"session_id": session_id, "doc_id": doc_id}}


@app.get("/sessions/{session_id}", summary="Get conversation history for a session")
def get_session(session_id: str):
    if not session_store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"session_id": session_id, "history": session_store.get_history(session_id)}


@app.delete("/sessions/{session_id}", summary="Delete a session and its history")
def delete_session(session_id: str):
    session_store.delete_session(session_id)
    return {"deleted": session_id}


@app.get("/health", summary="Check Ollama connectivity")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            models = [m["name"] for m in resp.json().get("models", [])]
        return {"status": "ok", "ollama_models": models}
    except httpx.ConnectError:
        return {"status": "ollama_unreachable"}
