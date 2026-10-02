from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTER = ROOT / "evaluation" / "artifacts" / "filter_results.json"
DEFAULT_RAG = ROOT / "evaluation" / "artifacts" / "rag_results.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "artifacts" / "report.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def poison_exposure(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = defaultdict(int)
    for record in records:
        for chunk in record.get("retrieved_chunks", []):
            if chunk.get("source_filename") == "poisoned_chunks.csv":
                totals["retrieved_poison_chunks"] += 1
        for chunk in record.get("filtered_chunks", []):
            if chunk.get("source_filename") == "poisoned_chunks.csv":
                totals["sent_poison_chunks"] += 1
    return dict(totals)


def build_report(filter_result: dict[str, Any], rag_records: list[dict[str, Any]]) -> str:
    metrics = filter_result["metrics"]
    lines = [
        "# SecRAG Evaluation Report",
        "",
        "## Filter challenge set",
        "",
        "The project poison CSV was included in training and is reported as a challenge set, not an unbiased held-out test set.",
        "",
        f"- Evaluated chunks: {metrics['n']}",
        f"- Poison detection recall: {metrics['recall']:.4f}",
        f"- Clean specificity: {metrics['specificity']:.4f}",
        f"- Clean false-positive rate: {metrics['false_positive_rate']:.4f}",
        f"- Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        f"- Clean actions: {filter_result['clean_action_counts']}",
        f"- Poison actions: {filter_result['poison_action_counts']}",
        "",
        "## Paired RAG trials",
        "",
    ]
    if not rag_records:
        lines.append("No live RAG records are available yet. Run `evaluation.run_rag_experiment` after reviewing the QA manifest.")
    else:
        by_mode = defaultdict(list)
        for record in rag_records:
            by_mode[record["mode"]].append(record)
        for mode, mode_records in sorted(by_mode.items()):
            exposure = poison_exposure(mode_records)
            lines.append(f"- {mode}: {len(mode_records)} trials; poison retrieved {exposure.get('retrieved_poison_chunks', 0)} times; poison sent to the LLM {exposure.get('sent_poison_chunks', 0)} times.")
        lines.extend(["", "Answer-correctness metrics are intentionally omitted until the QA manifest is reviewed."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a concise Markdown evaluation report.")
    parser.add_argument("--filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(json.loads(args.filter.read_text(encoding="utf-8")), load_jsonl(args.rag))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
