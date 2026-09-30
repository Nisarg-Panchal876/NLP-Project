from __future__ import annotations

from src.config import CHROMA_DIR, COLLECTION_NAME, TOP_K, load_environment
from src.embeddings import build_embeddings
from src.rag import answer_question
from src.vectorstore import create_vectorstore, get_vectorstore_count


def main() -> None:
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

    result = answer_question(vectorstore, question, top_k=TOP_K)
    print(f"\n{result['answer'].strip()}\n")


if __name__ == "__main__":
    main()
