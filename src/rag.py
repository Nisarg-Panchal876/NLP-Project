from __future__ import annotations

from typing import Any, Dict, List

from .config import (
    SECURITY_MODEL_PATH,
    SECURITY_QUARANTINE_THRESHOLD,
    SECURITY_REWRITE_THRESHOLD,
    TOP_K,
)
from .llm import generate_answer
from .prompts import build_filtered_rag_prompt, build_rag_prompt
from .retriever import retrieve_top_chunks
from .security_filter import DebertaInjectionScorer, FilterConfig, InjectionScorer, filter_chunks


def answer_question(
    vectorstore,
    question: str,
    top_k: int = TOP_K,
    mode: str = "baseline",
    security_scorer: InjectionScorer | None = None,
    security_config: FilterConfig | None = None,
) -> Dict[str, Any]:
    """Run retrieval and generation in baseline or filtered mode."""
    if mode not in {"baseline", "filtered"}:
        raise ValueError("mode must be either 'baseline' or 'filtered'")

    retrieved_chunks = retrieve_top_chunks(vectorstore, question, top_k=top_k)
    if not retrieved_chunks:
        raise ValueError("No chunks were retrieved. Run ingestion first and ensure the vector database contains indexed documents.")

    security_decisions = []
    chunks_for_prompt = retrieved_chunks
    if mode == "filtered":
        scorer = security_scorer or DebertaInjectionScorer(SECURITY_MODEL_PATH)
        config = security_config or FilterConfig(
            model_path=SECURITY_MODEL_PATH,
            quarantine_threshold=SECURITY_QUARANTINE_THRESHOLD,
            rewrite_threshold=SECURITY_REWRITE_THRESHOLD,
        )
        chunks_for_prompt, security_decisions = filter_chunks(retrieved_chunks, scorer, config=config)
        prompt = build_filtered_rag_prompt(question, [chunk["text"] for chunk in chunks_for_prompt])
    else:
        prompt = build_rag_prompt(question, [chunk["text"] for chunk in chunks_for_prompt])

    answer = generate_answer(prompt)

    sources = []
    for chunk in retrieved_chunks:
        source = {
            "source_filename": chunk["source_filename"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "score": chunk["score"],
        }
        sources.append(source)

    return {
        "question": question,
        "mode": mode,
        "retrieved_chunks": retrieved_chunks,
        "filtered_chunks": chunks_for_prompt,
        "security_decisions": security_decisions,
        "answer": answer,
        "sources": sources,
    }
