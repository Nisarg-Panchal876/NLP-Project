from __future__ import annotations

from typing import Any, Dict, List

from .config import TOP_K
from .llm import generate_answer
from .prompts import build_rag_prompt
from .retriever import retrieve_top_chunks


def answer_question(vectorstore, question: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Run retrieval and generation for a user question."""
    retrieved_chunks = retrieve_top_chunks(vectorstore, question, top_k=top_k)
    if not retrieved_chunks:
        raise ValueError("No chunks were retrieved. Run ingestion first and ensure the vector database contains indexed documents.")

    prompt = build_rag_prompt(question, [chunk["text"] for chunk in retrieved_chunks])
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
        "retrieved_chunks": retrieved_chunks,
        "answer": answer,
        "sources": sources,
    }
