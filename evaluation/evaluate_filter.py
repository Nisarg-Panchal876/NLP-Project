from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from src.config import SECURITY_MODEL_PATH, SECURITY_QUARANTINE_THRESHOLD, SECURITY_REWRITE_THRESHOLD
from src.security_filter import DebertaInjectionScorer, FilterConfig, filter_chunks

from .metrics import action_counts, binary_metrics, bootstrap_mean_ci
from .prepare_corpus import DEFAULT_OUTPUT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT = ROOT / "evaluation" / "artifacts" / "filter_results.json"


def evaluate(manifest: dict[str, Any], model_path: str) -> dict[str, Any]:
    chunks = manifest["clean_chunks"] + manifest.get("contaminated_chunks", []) + manifest["poison_chunks"]
    scorer = DebertaInjectionScorer(model_path)
    config = FilterConfig(model_path=model_path, quarantine_threshold=SECURITY_QUARANTINE_THRESHOLD, rewrite_threshold=SECURITY_REWRITE_THRESHOLD)
    started = time.perf_counter()
    safe_chunks, decisions = filter_chunks(chunks, scorer, config=config)
    elapsed = time.perf_counter() - started
    by_id = {decision["chunk_id"]: decision for decision in decisions}
    if len(by_id) != len(chunks):
        raise RuntimeError(f"Expected one decision per chunk, got {len(by_id)} for {len(chunks)} chunks")
    labels = [int(chunk["label"]) for chunk in chunks]
    predictions = [int(by_id[chunk["chunk_id"]]["security"]["action"] != "PASS") for chunk in chunks]
    clean_actions = [by_id[chunk["chunk_id"]]["security"]["action"] for chunk in manifest["clean_chunks"]]
    poison_actions = [
        by_id[chunk["chunk_id"]]["security"]["action"]
        for chunk in manifest.get("contaminated_chunks", []) + manifest["poison_chunks"]
    ]
    probabilities = [float(by_id[chunk["chunk_id"]]["security"]["injection_probability"]) for chunk in chunks]
    return {"schema_version": 1, "model_path": model_path, "thresholds": {"quarantine": SECURITY_QUARANTINE_THRESHOLD, "rewrite": SECURITY_REWRITE_THRESHOLD}, "elapsed_seconds": elapsed, "metrics": binary_metrics(labels, predictions), "clean_action_counts": action_counts(clean_actions), "poison_action_counts": action_counts(poison_actions), "probability_ci": bootstrap_mean_ci(probabilities), "retained_chunk_count": len(safe_chunks), "records": [{"chunk_id": chunk["chunk_id"], "label": chunk["label"], "corpus_role": chunk["corpus_role"], "action": by_id[chunk["chunk_id"]]["security"]["action"], "injection_probability": by_id[chunk["chunk_id"]]["security"]["injection_probability"], "anomaly": by_id[chunk["chunk_id"]]["security"]["anomaly"]} for chunk in chunks]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DeBERTa filter on the prepared corpus.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", default=str(ROOT / SECURITY_MODEL_PATH))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULT)
    args = parser.parse_args()
    result = evaluate(json.loads(args.manifest.read_text(encoding="utf-8")), args.model_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps(result["metrics"], sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
