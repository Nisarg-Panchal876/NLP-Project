from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR, ensure_pdf_dir


def _normalize_document_metadata(document: Document) -> Document:
    metadata = dict(document.metadata or {})
    source = metadata.get("source")
    source_filename = Path(source).name if source else "unknown.pdf"
    metadata.setdefault("source_filename", source_filename)
    metadata.setdefault("document_id", source_filename)
    metadata.setdefault("page", metadata.get("page", 1))
    document.metadata = metadata
    return document


def load_pdf_documents(pdf_dir: Path | str = PDF_DIR) -> List[Document]:
    """Load all PDF files from the configured folder."""
    resolved_dir = Path(pdf_dir)
    ensure_pdf_dir()
    loader = DirectoryLoader(str(resolved_dir), glob="*.pdf", loader_cls=PyPDFLoader, show_progress=False)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No PDFs found in {resolved_dir}. Add PDF files to that folder before ingesting.")
    return [_normalize_document_metadata(doc) for doc in documents]


def chunk_documents(documents: Iterable[Document], chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[Document]:
    """Split loaded documents into recursive chunks while preserving document metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=False,
    )
    chunks: List[Document] = splitter.split_documents(list(documents))

    for idx, chunk in enumerate(chunks):
        metadata = dict(chunk.metadata or {})
        source_name = metadata.get("source_filename", "unknown.pdf")
        page = metadata.get("page", 1)
        metadata["chunk_index"] = idx
        metadata["chunk_id"] = f"{Path(source_name).stem.upper()}-{page}-{str(idx).zfill(3)}"
        metadata["source_filename"] = source_name
        metadata["document_id"] = metadata.get("document_id", source_name)
        chunk.metadata = metadata

    return chunks
