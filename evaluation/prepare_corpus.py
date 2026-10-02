from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR
from src.ingestion import chunk_documents, load_pdf_documents

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTACKS = ROOT / "poisoned_chunks.csv"
DEFAULT_OUTPUT = ROOT / "evaluation" / "artifacts" / "corpus_manifest.json"


def stable_id(prefix: str, text: str) -> str:
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def normalize_for_matching(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def clean_chunk_records(pdf_dir: Path) -> list[dict[str, Any]]:
    chunks = chunk_documents(load_pdf_documents(pdf_dir), chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    records = []
    for chunk in chunks:
        metadata = dict(chunk.metadata or {})
        text = chunk.page_content
        records.append({"chunk_id": metadata.get("chunk_id", stable_id("clean", text)), "source_filename": metadata.get("source_filename", "unknown.pdf"), "page": metadata.get("page", 1), "document_id": metadata.get("document_id", "unknown"), "text": text, "label": 0, "corpus_role": "clean"})
    return records


def attack_records(csv_path: Path) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"poisoned text", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
    records = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        text = (row.get("poisoned text") or "").strip()
        if not text:
            raise ValueError(f"Empty poison text at CSV row {index + 1}")
        if text in seen:
            raise ValueError(f"Duplicate poison text at CSV row {index + 1}")
        seen.add(text)
        records.append({"chunk_id": stable_id(f"poison-{index:03d}", text), "source_filename": None, "page": None, "document_id": None, "text": text, "label": 1, "corpus_role": "poison", "attack_id": index, "training_included": True})
    return records


def build_manifest(pdf_dir: Path, attacks_csv: Path) -> dict[str, Any]:
    clean = clean_chunk_records(pdf_dir)
    attacks = attack_records(attacks_csv)
    attack_texts = [(record["attack_id"], normalize_for_matching(record["text"])) for record in attacks]
    uncontaminated: list[dict[str, Any]] = []
    contaminated: list[dict[str, Any]] = []
    for chunk in clean:
        matched_attack_ids = [attack_id for attack_id, attack_text in attack_texts if attack_text and attack_text in normalize_for_matching(chunk["text"])]
        if matched_attack_ids:
            contaminated_chunk = dict(chunk)
            contaminated_chunk["label"] = 1
            contaminated_chunk["corpus_role"] = "contaminated_pdf"
            contaminated_chunk["matched_attack_ids"] = matched_attack_ids
            contaminated.append(contaminated_chunk)
        else:
            uncontaminated.append(chunk)
    return {
        "schema_version": 2,
        "seed": 42,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "clean_source": str(pdf_dir),
        "attack_source": str(attacks_csv),
        "attack_source_note": "Project attack rows were included in DeBERTa training and are a challenge set, not an unbiased held-out test set.",
        "labeling_note": "PDF chunks containing whitespace-normalized poison CSV text are labeled contaminated_pdf rather than clean.",
        "clean_chunks": uncontaminated,
        "contaminated_chunks": contaminated,
        "poison_chunks": attacks,
        "counts": {
            "clean_chunks": len(uncontaminated),
            "contaminated_pdf_chunks": len(contaminated),
            "poison_chunks": len(attacks),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a deterministic SecRAG evaluation corpus manifest.")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--attacks", type=Path, default=DEFAULT_ATTACKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = build_manifest(args.pdf_dir, args.attacks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(manifest["counts"], sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
