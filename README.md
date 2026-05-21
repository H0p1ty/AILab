# AILab — Local RAG API

A local Retrieval-Augmented Generation (RAG) API. Upload PDF documents, then ask questions about them in a persistent, multi-turn conversation — all running locally via [Ollama](https://ollama.com), no external API keys required.

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) running on `localhost:11434` with the following models pulled:

```bash
ollama pull nomic-embed-text
ollama pull mistral:7b
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
```

## Running

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`. Interactive docs are at `http://localhost:8000/docs`.

---

## Usage

### Sessions

Every conversation happens inside a **session**. A session ties together:
- uploaded documents (only visible within that session)
- conversation history (used by the model for follow-up questions)

A session is created automatically when you upload a document or send your first query. Use the returned `session_id` in all subsequent requests to continue the same conversation.

---

### Endpoints

#### `POST /upload` — Upload a PDF

Upload a PDF to index it for retrieval. Optionally pass an existing `session_id` to add the document to a running session; otherwise a new session is created.

```bash
# Start a new session
curl -X POST http://localhost:8000/upload \
  -F "file=@notes.pdf"

# Add to an existing session
curl -X POST http://localhost:8000/upload \
  -F "file=@notes.pdf" \
  -F "session_id=<your-session-id>"
```

Response:
```json
{
  "doc_id": "...",
  "filename": "notes.pdf",
  "chunks_indexed": 42,
  "session_id": "..."
}
```

Limits: PDF only, max 5 MB.

---

#### `POST /query` — Ask a question

Send a question, optionally with a `session_id` to continue a previous conversation. Without indexed documents the model answers from general knowledge.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is covered in chapter 3?", "session_id": "<your-session-id>"}'
```

Request body:

| Field | Type | Default | Description |
|---|---|---|---|
| `question` | string | required | Max 2000 characters |
| `session_id` | string | null | Omit to start a new session |
| `n_results` | int | 5 | Number of document chunks to retrieve (1–20) |

Response:
```json
{
  "answer": "...",
  "sources": ["chunk text 1", "chunk text 2"],
  "session_id": "..."
}
```

---

#### `GET /documents/{session_id}` — List documents in a session

```bash
curl http://localhost:8000/documents/<session-id>
```

```json
{"documents": ["doc-id-1", "doc-id-2"]}
```

---

#### `DELETE /documents/{session_id}/{doc_id}` — Remove a document

Deletes the document's chunks from the vector store and unregisters it from the session.

```bash
curl -X DELETE http://localhost:8000/documents/<session-id>/<doc-id>
```

---

#### `GET /sessions/{session_id}` — Get conversation history

```bash
curl http://localhost:8000/sessions/<session-id>
```

```json
{
  "session_id": "...",
  "history": [
    {"role": "user", "content": "What is chapter 3 about?"},
    {"role": "assistant", "content": "Chapter 3 covers ..."}
  ]
}
```

---

#### `DELETE /sessions/{session_id}` — Delete a session

Removes all conversation history and deletes all documents that were uploaded in the session.

```bash
curl -X DELETE http://localhost:8000/sessions/<session-id>
```

---

#### `GET /health` — Check Ollama connectivity

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok", "ollama_models": ["mistral:7b", "nomic-embed-text"]}
```

---

## Architecture

```
PDF upload → text extraction (pymupdf) → chunking (512 words, 64 overlap)
          → embed (nomic-embed-text) → ChromaDB (session-scoped)

Query → embed → cosine similarity search (ChromaDB, session-scoped)
      → retrieved chunks + conversation history → mistral:7b → answer
```

| Component | Role |
|---|---|
| [Ollama](https://ollama.com) | Local LLM and embedding server |
| ChromaDB | Vector store (persistent, `./chroma_db`) |
| SQLite | Session and document metadata (`./sessions.db`) |

## Project structure

```
main.py              # FastAPI app and endpoints
rag/
  pdf_loader.py      # PDF text extraction and chunking
  embedder.py        # Embedding via Ollama nomic-embed-text
  retriever.py       # ChromaDB vector store operations
  llm.py             # LLM generation via Ollama mistral:7b
  session_store.py   # SQLite session and document tracking
requirements.txt
```
