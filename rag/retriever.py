from typing import List

import chromadb

_client = chromadb.PersistentClient(path="./chroma_db")


def _collection():
    return _client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(doc_id: str, session_id: str, chunks: List[str], embeddings: List[List[float]]) -> None:
    col = _collection()
    ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"doc_id": doc_id, "session_id": session_id}] * len(chunks)
    col.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def retrieve(embedding: List[float], session_id: str, n_results: int = 5) -> List[str]:
    col = _collection()
    where = {"session_id": session_id}
    count = len(col.get(where=where)["ids"])
    if count == 0:
        return []
    results = col.query(
        query_embeddings=[embedding],
        n_results=min(n_results, count),
        where=where,
    )
    return results["documents"][0] if results["documents"] else []


def list_doc_ids(session_id: str) -> List[str]:
    col = _collection()
    kwargs = {"where": {"session_id": session_id}} if session_id else {}
    result = col.get(include=["metadatas"], **kwargs)
    if not result["metadatas"]:
        return []
    return sorted({m["doc_id"] for m in result["metadatas"]})


def delete_doc(session_id: str, doc_id: str) -> None:
    col = _collection()
    col.delete(where={
        "$and": [
            {"doc_id": doc_id},
            {"session_id": session_id}
        ]
    })
