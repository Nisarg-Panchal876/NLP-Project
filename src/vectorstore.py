from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .config import CHROMA_DIR, COLLECTION_NAME


def create_vectorstore(embedding_function, collection_name: str = COLLECTION_NAME, persist_directory: str | Path = CHROMA_DIR):
    """Create or open a Chroma collection with the given embedding function."""
    persist_path = str(Path(persist_directory))
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_path,
    )


def persist_documents(documents: Iterable[Document], embedding_function, collection_name: str = COLLECTION_NAME, persist_directory: str | Path = CHROMA_DIR):
    """Persist the documents to the local Chroma database."""
    persist_path = str(Path(persist_directory))
    database = Chroma.from_documents(
        documents=list(documents),
        embedding=embedding_function,
        collection_name=collection_name,
        persist_directory=persist_path,
    )
    return database


def get_vectorstore_count(vectorstore) -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        return 0
