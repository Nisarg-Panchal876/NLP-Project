from __future__ import annotations

from src.config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION_NAME, PDF_DIR, load_environment
from src.embeddings import build_embeddings
from src.ingestion import chunk_documents, load_pdf_documents
from src.vectorstore import persist_documents


def main() -> None:
    load_environment()
    documents = load_pdf_documents(PDF_DIR)
    chunks = chunk_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = build_embeddings()
    persist_documents(chunks, embeddings, collection_name=COLLECTION_NAME, persist_directory=CHROMA_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(documents)} PDFs into {CHROMA_DIR}.")
    print(f"Collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    main()
