from __future__ import annotations

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from .config import EMBEDDING_MODEL, EMBEDDING_TASK, get_hf_token


def build_embeddings() -> HuggingFaceEndpointEmbeddings:
    """Build the Hugging Face hosted embedding model configured via env variables."""
    token = get_hf_token()
    try:
        return HuggingFaceEndpointEmbeddings(
            model=EMBEDDING_MODEL,
            task=EMBEDDING_TASK,
            huggingfacehub_api_token=token,
        )
    except ValueError as exc:
        raise ValueError(
            "The chosen Hugging Face embedding model/task combination is not compatible with the current hosted endpoint API. "
            "Set EMBEDDING_MODEL and EMBEDDING_TASK in .env to a supported model such as sentence-transformers/all-MiniLM-L6-v2 with feature-extraction. "
            f"Current config: model={EMBEDDING_MODEL}, task={EMBEDDING_TASK}."
        ) from exc
    except Exception as exc:
        raise ValueError(
            "The Hugging Face token in HUGGINGFACEHUB_API_TOKEN is invalid, expired, or not accepted by the inference endpoint. "
            "Create a new token at https://huggingface.co/settings/tokens and update the value in .env."
        ) from exc
