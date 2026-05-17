import logging

import httpx

OLLAMA_BASE = "http://localhost:11434"
LLM_MODEL = "mistral:7b"
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant engaged in a multi-turn conversation about documents. "
    "Each user message includes retrieved document context followed by a question. "
    "Use that context as your primary source of facts. "
    "For follow-up or clarification questions, you may also draw on prior conversation turns. "
    "If a factual answer cannot be found in either the context or the conversation history, say so."
)


class LLMError(Exception):
    pass


async def generate(question: str, context_chunks: list[str], history: list[dict]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    user_msg = f"Context:\n{context}\n\nQuestion: {question}"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_msg})

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            try:
                resp = await client.post(
                    f"{OLLAMA_BASE}/api/chat",
                    json={"model": LLM_MODEL, "messages": messages, "stream": False},
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"].strip()
            except httpx.HTTPStatusError as e:
                logger.error(f"Ollama chat failed: {e.response.status_code} {e.response.text}")
                raise LLMError(f"Ollama error: {e.response.status_code}")
            except httpx.TimeoutException:
                logger.error("LLM request timed out")
                raise LLMError("LLM request timed out (Ollama may be busy or model is slow)")
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama")
        raise LLMError("Cannot reach Ollama on localhost:11434")
    except httpx.HTTPError as e:
        logger.error(f"HTTP error during LLM generation: {e}")
        raise LLMError(f"HTTP error: {str(e)}")
