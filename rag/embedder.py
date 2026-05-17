import logging
from typing import List

import httpx

OLLAMA_BASE = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
logger = logging.getLogger(__name__)


class EmbeddingError(Exception):
    pass


async def embed(texts: List[str]) -> List[List[float]]:
    embeddings = []
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for text in texts:
                # Ensure text is proper UTF-8
                if isinstance(text, bytes):
                    text = text.decode("utf-8", errors="replace")
                text = text.encode("utf-8").decode("utf-8")

                try:
                    resp = await client.post(
                        f"{OLLAMA_BASE}/api/embeddings",
                        json={"model": EMBED_MODEL, "prompt": text},
                    )
                    resp.raise_for_status()
                    embeddings.append(resp.json()["embedding"])
                except httpx.HTTPStatusError as e:
                    logger.error(f"Ollama embedding failed: {e.response.status_code} {e.response.text}")
                    raise EmbeddingError(f"Ollama error: {e.response.status_code}")
                except httpx.TimeoutException:
                    logger.error("Embedding request timed out")
                    raise EmbeddingError("Embedding request timed out (Ollama may be busy)")
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama")
        raise EmbeddingError("Cannot reach Ollama on localhost:11434")
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during embedding: {e}")
        raise EmbeddingError(f"HTTP error: {str(e)}")
    return embeddings
