from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from .config import TOP_K


def retrieve_top_chunks(vectorstore, query: str, top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """Retrieve the top-k relevant chunks with score metadata for later inspection."""
    results = vectorstore.similarity_search_with_score(query=query, k=top_k)
    retrieved: List[Dict[str, Any]] = []
    for rank, (document, score) in enumerate(results, start=1):
        metadata = dict(document.metadata or {})
        retrieved.append(
            {
                "rank": rank,
                "source_filename": metadata.get("source_filename", "unknown.pdf"),
                "page": metadata.get("page", 1),
                "chunk_id": metadata.get("chunk_id", f"chunk-{rank}"),
                "document_id": metadata.get("document_id", "unknown"),
                "text": document.page_content,
                "score": float(score),
            }
        )
    return retrieved
