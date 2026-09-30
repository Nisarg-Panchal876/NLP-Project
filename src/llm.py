from __future__ import annotations

from langchain_ollama import ChatOllama

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL, get_ollama_api_key


def build_llm() -> ChatOllama:
    """Create an authenticated Ollama Cloud chat client."""
    api_key = get_ollama_api_key()
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}},
    )


def generate_answer(prompt: str) -> str:
    """Generate a final answer from a prompt with the Ollama Cloud model."""
    llm = build_llm()
    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        detail = str(exc)
        raise RuntimeError(
            f"Ollama Cloud request failed for model {OLLAMA_MODEL!r} at {OLLAMA_BASE_URL}. "
            f"Provider response: {detail}"
        ) from exc
