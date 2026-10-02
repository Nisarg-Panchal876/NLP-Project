from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K, load_environment
from src.embeddings import build_embeddings
from src.rag import answer_question
from src.vectorstore import create_vectorstore, get_vectorstore_count, persist_documents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "evaluation" / "artifacts" / "corpus_manifest.json"
DEFAULT_QA = ROOT / "evaluation" / "qa_manifest.template.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "artifacts" / "rag_results.jsonl"


def prompt_hash(answer: dict[str, Any]) -> str:
    context = "\n".join(chunk["text"] for chunk in answer["filtered_chunks"])
    return hashlib.sha256(f"{answer['mode']}\n{answer['question']}\n{context}".encode("utf-8")).hexdigest()


def build_documents(manifest: dict[str, Any]) -> list[Document]:
    documents = []
    for chunk in manifest["clean_chunks"] + manifest.get("contaminated_chunks", []) + manifest["poison_chunks"]:
        metadata = {"chunk_id": chunk["chunk_id"], "source_filename": chunk["source_filename"] or "poisoned_chunks.csv", "page": chunk["page"] or 0, "document_id": chunk["document_id"] or chunk["chunk_id"], "corpus_role": chunk["corpus_role"], "label": chunk["label"]}
        documents.append(Document(page_content=chunk["text"], metadata=metadata))
    return documents


def load_questions(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions", [])
    if not questions:
        raise ValueError("QA manifest contains no questions")
    if payload.get("review_status") != "REVIEWED":
        print("Warning: QA manifest is not REVIEWED; do not report answer-correctness metrics.")
    return questions


def run(manifest: dict[str, Any], questions: list[dict[str, Any]], output: Path, top_k: int, seed: int) -> None:
    load_environment()
    embeddings = build_embeddings()
    collection_name = f"{COLLECTION_NAME}_evaluation_{seed}"
    vectorstore = create_vectorstore(embeddings, collection_name=collection_name, persist_directory=CHROMA_DIR)
    if get_vectorstore_count(vectorstore) == 0:
        vectorstore = persist_documents(build_documents(manifest), embeddings, collection_name=collection_name, persist_directory=CHROMA_DIR)
    existing = {json.loads(line)["trial_id"] for line in output.read_text(encoding="utf-8").splitlines() if line.strip()} if output.exists() else set()
    trials = [(question, mode) for question in questions for mode in ("baseline", "filtered")]
    random.Random(seed).shuffle(trials)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for question, mode in trials:
            trial_id = f"{question['id']}::{mode}"
            if trial_id in existing:
                continue
            started = time.perf_counter()
            result = answer_question(vectorstore, question["question"], top_k=top_k, mode=mode)
            record = {"trial_id": trial_id, "question_id": question["id"], "topic": question.get("topic"), "source_document": question.get("source_document"), "mode": mode, "question": question["question"], "reference_answer": question.get("reference_answer", ""), "answer": result["answer"], "retrieved_chunks": result["retrieved_chunks"], "filtered_chunks": result["filtered_chunks"], "security_decisions": result["security_decisions"], "prompt_hash": prompt_hash(result), "elapsed_seconds": time.perf_counter() - started, "top_k": top_k, "seed": seed}
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
            handle.flush()
            print(f"completed {trial_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired baseline and filtered RAG trials.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--qa", type=Path, default=DEFAULT_QA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(json.loads(args.manifest.read_text(encoding="utf-8")), load_questions(args.qa), args.output, args.top_k, args.seed)


if __name__ == "__main__":
    main()
