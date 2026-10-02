from __future__ import annotations

import argparse

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K, load_environment
from src.embeddings import build_embeddings
from src.rag import answer_question
from src.vectorstore import create_vectorstore, get_vectorstore_count


def print_chunk(chunk: dict, heading: str) -> None:
    print(f"\n{heading}")
    print(f"Source: {chunk.get('source_filename', 'unknown')}")
    print(f"Page: {chunk.get('page', 'unknown')} | Chunk: {chunk.get('chunk_id', 'unknown')}")
    print(f"Score/Distance: {chunk.get('score', 'unknown')}")
    print("Text:")
    print(chunk.get("text", ""))


def print_pipeline_details(result: dict, mode: str) -> None:
    print("\n" + "=" * 72)
    print(f"{mode.upper()} PIPELINE DETAILS")
    print("=" * 72)

    for index, chunk in enumerate(result["retrieved_chunks"], start=1):
        print_chunk(chunk, f"Retrieved chunk {index}")

    if mode != "filtered":
        return

    print("\n" + "-" * 72)
    print("SECURITY DECISIONS")
    print("-" * 72)
    for index, decision in enumerate(result["security_decisions"], start=1):
        security = decision["security"]
        anomaly = security["anomaly"]
        print(
            f"\nChunk {index} ({decision.get('chunk_id', 'unknown')}): "
            f"{security['action']} | "
            f"P(factual)={security['factual_probability']:.4f} | "
            f"P(injection)={security['injection_probability']:.4f}"
        )
        print(f"Anomaly: {anomaly['severity']} | Reasons: {', '.join(anomaly['reasons']) or 'none'}")
        if "rewritten_text" in security:
            print("Rewritten text:")
            print(security["rewritten_text"])

    print("\n" + "-" * 72)
    print("CHUNKS SENT TO LLM")
    print("-" * 72)
    if result["filtered_chunks"]:
        for index, chunk in enumerate(result["filtered_chunks"], start=1):
            print_chunk(chunk, f"Retained/rewritten chunk {index}")
    else:
        print("No chunks were retained.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the SecRAG baseline or filtered pipeline.")
    parser.add_argument("--mode", choices=("baseline", "filtered"), default="baseline")
    args = parser.parse_args()

    load_environment()
    embeddings = build_embeddings()
    vectorstore = create_vectorstore(
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DIR,
    )

    if get_vectorstore_count(vectorstore) == 0:
        print("No indexed documents found in Chroma. Run 'python ingest.py' first.")
        return

    question = input("Enter your question: ").strip()
    if not question:
        print("No question entered.")
        return

    result = answer_question(vectorstore, question, top_k=TOP_K, mode=args.mode)
    print_pipeline_details(result, args.mode)
    print(f"\n{result['answer'].strip()}\n")
    if args.mode == "filtered":
        quarantined = sum(
            decision["security"]["action"] == "QUARANTINE"
            for decision in result["security_decisions"]
        )
        rewritten = sum(
            decision["security"]["action"] == "REWRITE"
            for decision in result["security_decisions"]
        )
        print(f"Security actions: {quarantined} quarantined, {rewritten} rewritten, {len(result['filtered_chunks'])} retained.")


if __name__ == "__main__":
    main()
