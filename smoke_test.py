from __future__ import annotations

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR, load_environment
from src.embeddings import build_embeddings
from src.ingestion import chunk_documents, load_pdf_documents
from src.rag import answer_question
from src.vectorstore import persist_documents


def main() -> None:
    load_environment()

    documents = load_pdf_documents(PDF_DIR)
    assert documents, "PDF loading failed: no documents were loaded."

    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    assert chunks, "Document chunking failed: no chunks were created."

    embeddings = build_embeddings()
    vectorstore = persist_documents(
        chunks[:3],
        embeddings,
        collection_name="smoke_test_collection",
        persist_directory=CHROMA_DIR,
    )

    results = vectorstore.similarity_search_with_score("attendance policy", k=1)
    assert results, "Chroma retrieval failed: no chunks were retrieved."

    answer = answer_question(vectorstore, "What is the attendance policy?", top_k=1)
    assert answer["answer"], "LLM answer generation failed: no answer returned."

    print("Smoke test passed: PDFs loaded, chunks created, embeddings generated, Chroma stored and retrieved, Ollama Cloud answered.")


if __name__ == "__main__":
    main()
