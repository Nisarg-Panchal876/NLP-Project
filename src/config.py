from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"


def load_environment() -> None:
    """Load environment variables from the project .env file."""
    load_dotenv(BASE_DIR / ".env", override=True)


load_environment()

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
EMBEDDING_TASK = os.getenv("EMBEDDING_TASK", "feature-extraction")
HF_TOKEN_ENV_VAR = "HUGGINGFACEHUB_API_TOKEN"
OLLAMA_API_KEY_ENV_VAR = "OLLAMA_API_KEY"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "glm-5.3-flash:cloud")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "5"))


def get_hf_token() -> str:
    """Return the Hugging Face token or raise a clear error."""
    load_environment()
    token = os.getenv(HF_TOKEN_ENV_VAR)
    normalized = (token or "").strip()
    if not normalized or normalized.lower() in {"your_token_here", "your_hf_token_here"}:
        raise ValueError(
            f"Missing Hugging Face API token. Set the {HF_TOKEN_ENV_VAR} environment variable in your .env file with a real token."
        )
    return normalized


def get_ollama_api_key() -> str:
    """Return the Ollama Cloud API key or raise a clear error."""
    load_environment()
    token = os.getenv(OLLAMA_API_KEY_ENV_VAR)
    normalized = (token or "").strip()
    if not normalized or normalized.lower() in {"your_token_here", "your_ollama_api_key_here"}:
        raise ValueError(
            f"Missing Ollama Cloud API key. Set the {OLLAMA_API_KEY_ENV_VAR} environment variable in your .env file."
        )
    return normalized


def ensure_pdf_dir() -> Path:
    if not PDF_DIR.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_DIR}. Create it and place your PDFs inside it."
        )
    return PDF_DIR
