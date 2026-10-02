from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILTER = ROOT / "evaluation" / "artifacts" / "filter_results_calibrated.json"
DEFAULT_ORIGINAL_FILTER = ROOT / "evaluation" / "artifacts" / "filter_results.json"
DEFAULT_MANIFEST = ROOT / "evaluation" / "artifacts" / "corpus_manifest.json"
DEFAULT_RAG = ROOT / "evaluation" / "artifacts" / "rag_results.jsonl"
DEFAULT_OUTPUT = ROOT / "evaluation" / "artifacts" / "SecRAG_full_evaluation_report.pdf"
DEFAULT_SUMMARY = ROOT / "models" / "secrag-deberta-final" / "training_summary.json"

NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#137C8B")
LIGHT_BLUE = colors.HexColor("#EAF3F5")
LIGHT_GRAY = colors.HexColor("#F3F5F7")
DARK = colors.HexColor("#1E2933")
RED = colors.HexColor("#A33A3A")
GREEN = colors.HexColor("#277A4B")


class ArchitectureDiagram(Flowable):
    def __init__(self, width: float = 510, height: float = 190):
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        boxes = [
            (10, 125, 105, 42, "PDF corpus\n+ poison CSV", LIGHT_BLUE),
            (145, 125, 105, 42, "Ingestion\n+ chunking", LIGHT_BLUE),
            (280, 125, 105, 42, "Embeddings\n+ Chroma", LIGHT_BLUE),
            (415, 125, 85, 42, "Top-k\nretrieval", LIGHT_BLUE),
            (145, 30, 105, 45, "Baseline\nraw context", colors.HexColor("#FBE8E8")),
            (280, 30, 105, 45, "DeBERTa\n+ anomaly gate", colors.HexColor("#E4F2E9")),
            (415, 30, 85, 45, "Ollama\nanswer", LIGHT_BLUE),
        ]
        for x, y, w, h, label, fill in boxes:
            canvas.setFillColor(fill)
            canvas.setStrokeColor(NAVY)
            canvas.roundRect(x, y, w, h, 6, fill=1, stroke=1)
            canvas.setFillColor(DARK)
            canvas.setFont("Helvetica-Bold", 8)
            lines = label.split("\n")
            for index, line in enumerate(lines):
                canvas.drawCentredString(x + w / 2, y + h / 2 + 7 - index * 11, line)

        arrows = [
            (115, 146, 145, 146),
            (250, 146, 280, 146),
            (385, 146, 415, 146),
            (332, 125, 332, 75),
            (250, 52, 280, 52),
            (385, 52, 415, 52),
        ]
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.4)
        for x1, y1, x2, y2 in arrows:
            canvas.line(x1, y1, x2, y2)
            if y1 == y2:
                canvas.line(x2, y2, x2 - 5, y2 + 3)
                canvas.line(x2, y2, x2 - 5, y2 - 3)
            else:
                canvas.line(x2, y2, x2 - 3, y2 + 5)
                canvas.line(x2, y2, x2 + 3, y2 + 5)
        canvas.setFillColor(DARK)
        canvas.setFont("Helvetica", 7.5)
        canvas.drawString(255, 89, "filtered mode")
        canvas.drawString(253, 16, "PASS / REWRITE / QUARANTINE")


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


def rich(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("ReportTitle", parent=base["Title"], fontName="Helvetica-Bold", fontSize=23, leading=28, alignment=TA_CENTER, textColor=NAVY, spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=10, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#52616B"), spaceAfter=18),
        "h1": ParagraphStyle("H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20, textColor=NAVY, spaceBefore=14, spaceAfter=8),
        "h2": ParagraphStyle("H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=TEAL, spaceBefore=9, spaceAfter=5),
        "body": ParagraphStyle("Body", parent=base["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13, textColor=DARK, spaceAfter=7),
        "small": ParagraphStyle("Small", parent=base["BodyText"], fontName="Helvetica", fontSize=7.5, leading=10, textColor=DARK, spaceAfter=4),
        "callout": ParagraphStyle("Callout", parent=base["BodyText"], fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=NAVY, backColor=LIGHT_BLUE, borderPadding=8, spaceBefore=6, spaceAfter=10),
        "code": ParagraphStyle("Code", parent=base["Code"], fontName="Courier", fontSize=7.5, leading=10, textColor=DARK, backColor=LIGHT_GRAY, borderPadding=6),
    }


def table(data: list[list[Any]], widths: list[float], header: bool = True) -> Table:
    converted = []
    for row in data:
        converted.append([cell if isinstance(cell, Paragraph) else str(cell) for cell in row])
    result = Table(converted, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B7C3CC")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    if header:
        commands.extend([("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")])
    for row_index in range(1 if header else 0, len(data)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), LIGHT_GRAY))
    result.setStyle(TableStyle(commands))
    return result


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def page_number(canvas: Any, document: Any) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5DC"))
    canvas.line(42, 32, 553, 32)
    canvas.setFillColor(colors.HexColor("#65727D"))
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(42, 20, "SecRAG evaluation report | Reproducible project artifact")
    canvas.drawRightString(553, 20, f"Page {document.page}")
    canvas.restoreState()


def rag_mode_stats(records: list[dict[str, Any]], mode: str) -> dict[str, float | int]:
    selected = [record for record in records if record.get("mode") == mode]
    retrieved_poison = sum(
        1
        for record in selected
        for chunk in record.get("retrieved_chunks", [])
        if chunk.get("source_filename") == "poisoned_chunks.csv"
    )
    sent_poison = sum(
        1
        for record in selected
        for chunk in record.get("filtered_chunks", [])
        if chunk.get("source_filename") == "poisoned_chunks.csv"
    )
    context_counts = [len(record.get("filtered_chunks", [])) for record in selected]
    latencies = [float(record.get("elapsed_seconds", 0.0)) for record in selected]
    answer_lengths = [len(record.get("answer", "")) for record in selected]
    return {
        "trials": len(selected),
        "retrieved_poison": retrieved_poison,
        "sent_poison": sent_poison,
        "mean_context": sum(context_counts) / len(context_counts) if context_counts else 0.0,
        "empty_context": sum(count == 0 for count in context_counts),
        "mean_latency": sum(latencies) / len(latencies) if latencies else 0.0,
        "mean_answer_chars": sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0.0,
    }


def build_story(filter_result: dict[str, Any], original_filter_result: dict[str, Any], manifest: dict[str, Any], rag_records: list[dict[str, Any]], training_summary: dict[str, Any], qa_manifest: dict[str, Any], styles: dict[str, ParagraphStyle]) -> list[Any]:
    story: list[Any] = []
    metrics = filter_result["metrics"]
    clean = manifest["counts"]["clean_chunks"]
    contaminated = manifest["counts"].get("contaminated_pdf_chunks", 0)
    poison = contaminated + manifest["counts"]["poison_chunks"]
    total = metrics["n"]
    clean_actions = filter_result["clean_action_counts"]
    poison_actions = filter_result["poison_action_counts"]

    story.extend([
        Spacer(1, 0.65 * inch),
        rich("SecRAG: Full Evaluation and System Architecture Report", styles["title"]),
        rich("Baseline RAG versus DeBERTa-gated retrieval | Generated from the project evaluation artifacts", styles["subtitle"]),
        ArchitectureDiagram(),
        Spacer(1, 0.18 * inch),
        rich("Purpose", styles["h1"]),
        paragraph("This report documents the implemented retrieval-augmented generation system, the security filter inserted between retrieval and generation, the evaluation protocol, the numerical results currently available, and the experiments still required before making end-to-end answer-quality claims in a research paper.", styles["body"]),
        paragraph("Important status: this report uses the no-retraining calibrated gate and corrected provenance labels. PDF chunks containing poison CSV text are counted as contaminated rather than clean. Live answer records remain useful for exposure analysis only because the QA reference manifest is unreviewed.", styles["callout"]),
        rich("Executive findings", styles["h1"]),
        table([
            ["Finding", "Measured value"],
            ["Poison chunks detected", f"{metrics['tp']} of {poison} ({pct(metrics['recall'])})"],
            ["Clean chunks retained", f"{clean_actions.get('PASS', 0)} of {clean} ({pct(metrics['specificity'])})"],
            ["Clean false-positive rate", pct(metrics["false_positive_rate"])],
            ["Total evaluated chunks", str(total)],
            ["Balanced accuracy", f"{metrics['balanced_accuracy']:.4f}"],
        ], [2.65 * inch, 3.65 * inch]),
        rich("Evidence map and claim status", styles["h2"]),
        table([
            ["Evidence tier", "Primary artifact", "Status and permitted claim"],
            ["A. DeBERTa classifier test", "models/secrag-deberta-final/training_summary.json", "Available; classifier test metrics only"],
            ["B. Offline security challenge", "evaluation/artifacts/filter_results.json and filter_results_calibrated.json", "Available; 244-chunk security metrics"],
            ["C. Paired RAG experiment", "evaluation/artifacts/rag_results.jsonl", "Available; 10 trials, 5 per mode, exposure/runtime only"],
            ["D. Answer-quality evaluation", "evaluation/qa_manifest.template.json", "Not complete; QA status is UNREVIEWED"],
        ], [1.5 * inch, 2.35 * inch, 2.45 * inch]),
        PageBreak(),
    ])

    story.extend([
        rich("1. System architecture", styles["h1"]),
        paragraph("The project is a modular RAG pipeline. PDF files are loaded and split into recursive chunks. Each chunk is embedded and stored in ChromaDB. A user question is embedded and used to retrieve the top-k chunks. In baseline mode, retrieved text is sent directly to the answer prompt. In filtered mode, each retrieved chunk is scored by the local DeBERTa classifier and checked by explainable anomaly rules before the retained context is sent to the generator.", styles["body"]),
        rich("Data and ingestion", styles["h2"]),
        table([
            ["Component", "Implementation detail"],
            ["Clean source", "Five policy PDFs in data/pdfs/; contaminated chunks are relabeled"],
            ["Chunking", f"RecursiveCharacterTextSplitter; chunk size {manifest['chunk_size']}; overlap {manifest['chunk_overlap']}"],
            ["Embeddings", "Configurable Hugging Face hosted embedding model"],
            ["Vector store", "Persistent ChromaDB collection"],
            ["Retriever", "Similarity search with configurable top-k; default top-k is 5"],
            ["Generator", "Authenticated Ollama Cloud ChatOllama model"],
        ], [1.45 * inch, 4.85 * inch]),
        rich("Security gate", styles["h2"]),
        paragraph("The DebertaInjectionScorer loads microsoft/deberta-v3-base fine-tuned for factual versus injection classification. It returns the class-1 injection probability. The anomaly layer independently checks character entropy, base64-like payloads, zero-width characters, unusual character ratios, and directive patterns.", styles["body"]),
        table([
            ["Condition", "Action"],
            [f"Injection probability > {filter_result['thresholds']['quarantine']} or anomaly severity HIGH", "QUARANTINE: remove from LLM context"],
            [f"Injection probability >= {filter_result['thresholds']['rewrite']} and below quarantine", "REWRITE: remove directive-bearing sentences"],
            ["Otherwise", "PASS: retain original chunk"],
        ], [3.0 * inch, 3.3 * inch]),
        rich("Audit behavior", styles["h2"]),
        paragraph("Every retrieved chunk now produces exactly one ordered security decision. Each decision includes the action, injection probability, factual probability, anomaly result, and original text. This makes per-rank exposure, quarantine, rewrite, and retention statistics auditable.", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        rich("2. Evaluation design and provenance", styles["h1"]),
        paragraph("The primary offline evaluation combines clean chunks extracted from the five project PDFs with the existing project poison CSV. Each poison row is included exactly once; no second poisoning pass or duplicate attack generation is performed.", styles["body"]),
        table([
            ["Dataset property", "Value"],
            ["Uncontaminated PDF chunks", str(clean)],
            ["Contaminated PDF chunks", str(contaminated)],
            ["Poison CSV rows", str(poison)],
            ["Total chunks", str(total)],
            ["Class 0 proportion", pct(clean / total)],
            ["Class 1 proportion", pct(poison / total)],
            ["Random seed", str(manifest.get("seed", 42))],
            ["CSV provenance", "Included in DeBERTa training; challenge set only"],
        ], [2.15 * inch, 4.15 * inch]),
        rich("Baseline versus filtered definitions", styles["h2"]),
        table([
            ["Aspect", "Baseline", "Filtered"],
            ["Retrieved context", "All top-k chunks", "Top-k chunks before security gate"],
            ["Security scoring", "None", "DeBERTa probability plus anomaly analysis"],
            ["Poison handling", "Raw poison can reach the prompt", "PASS, REWRITE, or QUARANTINE"],
            ["Generator", "Same configured Ollama model", "Same configured Ollama model"],
            ["Controlled variable", "No filtering", "Security gate inserted before generation"],
        ], [1.35 * inch, 2.45 * inch, 2.5 * inch]),
        rich("Statistical definitions", styles["h2"]),
        paragraph("For binary detection, TP is a poison chunk marked non-PASS, TN is a clean chunk marked PASS, FP is a clean chunk marked non-PASS, and FN is a poison chunk marked PASS. Accuracy = (TP + TN) / N. Precision = TP / (TP + FP). Recall = TP / (TP + FN). Specificity = TN / (TN + FP). F1 is the harmonic mean of precision and recall. Balanced accuracy is the mean of recall and specificity.", styles["body"]),
        PageBreak(),
    ])

    story.extend([
        rich("3. Numerical results: B. Offline security evaluation", styles["h1"]),
        paragraph("The calibrated figures below are derived directly from filter_results_calibrated.json. The original figures are retained for comparison, but the original clean FPR is not a valid clean-domain estimate because the original manifest labeled contaminated PDF chunks as clean.", styles["body"]),
        table([
            ["Metric", "Original gate artifact", "Calibrated artifact"],
            ["N", str(original_filter_result["metrics"]["n"]), str(metrics["n"])],
            ["Accuracy", f"{original_filter_result['metrics']['accuracy']:.4f}", f"{metrics['accuracy']:.4f}"],
            ["Precision", f"{original_filter_result['metrics']['precision']:.4f}", f"{metrics['precision']:.4f}"],
            ["Recall", f"{original_filter_result['metrics']['recall']:.4f}", f"{metrics['recall']:.4f}"],
            ["Specificity", f"{original_filter_result['metrics']['specificity']:.4f}", f"{metrics['specificity']:.4f}"],
            ["False-positive rate", f"{original_filter_result['metrics']['false_positive_rate']:.4f} ({pct(original_filter_result['metrics']['false_positive_rate'])})", f"{metrics['false_positive_rate']:.4f} ({pct(metrics['false_positive_rate'])})"],
            ["F1", f"{original_filter_result['metrics']['f1']:.4f}", f"{metrics['f1']:.4f}"],
        ], [2.1 * inch, 2.1 * inch, 2.1 * inch]),
        rich("Confusion matrix", styles["h2"]),
        table([
            ["", "Predicted safe", "Predicted unsafe", "Total"],
            ["Actual clean", str(metrics["tn"]), str(metrics["fp"]), str(clean)],
            ["Actual poison", str(metrics["fn"]), str(metrics["tp"]), str(poison)],
            ["Total", str(metrics["tn"] + metrics["fn"]), str(metrics["fp"] + metrics["tp"]), str(metrics["n"])],
        ], [1.35 * inch, 1.55 * inch, 1.65 * inch, 1.25 * inch]),
        rich("Aggregate detection metrics", styles["h2"]),
        table([
            ["Metric", "Value", "Interpretation"],
            ["Accuracy", f"{metrics['accuracy']:.4f} ({pct(metrics['accuracy'])})", "All chunk decisions correct"],
            ["Precision", f"{metrics['precision']:.4f} ({pct(metrics['precision'])})", "Non-PASS decisions that were poison"],
            ["Recall", f"{metrics['recall']:.4f} ({pct(metrics['recall'])})", "Poison chunks detected"],
            ["Specificity", f"{metrics['specificity']:.4f} ({pct(metrics['specificity'])})", "Clean chunks retained as PASS"],
            ["False-positive rate", f"{metrics['false_positive_rate']:.4f} ({pct(metrics['false_positive_rate'])})", "Clean chunks incorrectly blocked or rewritten"],
            ["F1", f"{metrics['f1']:.4f}", "Harmonic mean of precision and recall"],
            ["Balanced accuracy", f"{metrics['balanced_accuracy']:.4f}", "Mean of recall and specificity"],
        ], [1.45 * inch, 1.55 * inch, 3.3 * inch]),
        rich("Security action distribution", styles["h2"]),
        table([
            ["Corpus role", "PASS", "REWRITE", "QUARANTINE", "Non-PASS"],
            ["Clean", str(clean_actions.get("PASS", 0)), str(clean_actions.get("REWRITE", 0)), str(clean_actions.get("QUARANTINE", 0)), str(clean - clean_actions.get("PASS", 0))],
            ["Poison", str(poison_actions.get("PASS", 0)), str(poison_actions.get("REWRITE", 0)), str(poison_actions.get("QUARANTINE", 0)), str(poison - poison_actions.get("PASS", 0))],
        ], [1.5 * inch, 1.1 * inch, 1.1 * inch, 1.35 * inch, 1.25 * inch]),
        rich("Probability and runtime", styles["h2"]),
        table([
            ["Measurement", "Value"],
            ["Mean injection probability", f"{filter_result['probability_ci']['mean']:.4f}"],
            ["Bootstrap 95% lower bound", f"{filter_result['probability_ci']['lower_95']:.4f}"],
            ["Bootstrap 95% upper bound", f"{filter_result['probability_ci']['upper_95']:.4f}"],
            ["Filter evaluation elapsed time", f"{filter_result['elapsed_seconds']:.3f} seconds"],
            ["Mean time per chunk", f"{filter_result['elapsed_seconds'] / total:.4f} seconds"],
        ], [2.8 * inch, 3.5 * inch]),
        paragraph("Interpretation: after correcting contaminated PDF labels and applying the no-retraining gate, clean false-positive rate falls to 30.00%, but poison recall is 62.89%. This is a better operating point than the original gate, yet it still requires broader attack-pattern coverage or threshold/decision tuning before claiming robust protection.", styles["callout"]),
        PageBreak(),
    ])

    story.extend([
        rich("4. Numerical results: A. DeBERTa classifier test", styles["h1"]),
        paragraph("The saved model summary is included for provenance. These figures describe the classifier training evaluation reported by the existing training pipeline; they are separate from the end-to-end RAG filter challenge-set results above.", styles["body"]),
    ])
    if training_summary:
        summary_metrics = training_summary.get("test_metrics", {})
        story.append(table([
            ["Training attribute", "Recorded value"],
            ["Base model", training_summary.get("model", "not recorded")],
            ["Seed", training_summary.get("seed", "not recorded")],
            ["Training rows", training_summary.get("train_rows", "not recorded")],
            ["Classifier test rows", training_summary.get("test_rows", "not recorded")],
            ["Classifier test accuracy", f"{summary_metrics.get('eval_accuracy', 0.0):.4f}"],
            ["Classifier test precision", f"{summary_metrics.get('eval_precision', 0.0):.4f}"],
            ["Classifier test recall", f"{summary_metrics.get('eval_recall', 0.0):.4f}"],
            ["Classifier test F1", f"{summary_metrics.get('eval_f1', 0.0):.4f}"],
            ["Class mapping", str(training_summary.get("class_mapping", {}))],
        ], [2.25 * inch, 4.05 * inch]))
    story.extend([
        rich("5. Numerical results: C. Paired baseline-versus-filtered RAG", styles["h1"]),
        paragraph("Before filtering, the baseline policy sends every retrieved chunk to the generator. In the completed five-question paired run, every poison chunk that was retrieved by baseline was sent to the generator. After filtering, the same retrieved poison chunks were removed before generation.", styles["body"]),
        paragraph(f"The live answers were captured with a QA manifest whose status is {qa_manifest.get('review_status', 'UNKNOWN')}. They support context-exposure and runtime measurements, but not answer-correctness or faithfulness claims.", styles["body"]),
        table([
            ["Mode", "Trials", "Poison retrieved", "Poison sent", "Mean context", "Empty contexts", "Mean latency (s)", "Mean answer chars"],
            ["Baseline", str(int(rag_mode_stats(rag_records, "baseline")["trials"])), str(int(rag_mode_stats(rag_records, "baseline")["retrieved_poison"])), str(int(rag_mode_stats(rag_records, "baseline")["sent_poison"])), f"{rag_mode_stats(rag_records, 'baseline')['mean_context']:.2f}", str(int(rag_mode_stats(rag_records, "baseline")["empty_context"])), f"{rag_mode_stats(rag_records, 'baseline')['mean_latency']:.2f}", f"{rag_mode_stats(rag_records, 'baseline')['mean_answer_chars']:.1f}"],
            ["Filtered", str(int(rag_mode_stats(rag_records, "filtered")["trials"])), str(int(rag_mode_stats(rag_records, "filtered")["retrieved_poison"])), str(int(rag_mode_stats(rag_records, "filtered")["sent_poison"])), f"{rag_mode_stats(rag_records, 'filtered')['mean_context']:.2f}", str(int(rag_mode_stats(rag_records, "filtered")["empty_context"])), f"{rag_mode_stats(rag_records, 'filtered')['mean_latency']:.2f}", f"{rag_mode_stats(rag_records, 'filtered')['mean_answer_chars']:.1f}"],
        ], [0.75 * inch, 0.55 * inch, 0.85 * inch, 0.75 * inch, 0.75 * inch, 0.7 * inch, 0.85 * inch, 0.85 * inch]),
        table([
            ["Result category", "Current status", "Required evidence"],
            ["Chunk-level poison detection", "Measured", "filter_results.json"],
            ["Clean retention / false positives", "Measured", "filter_results.json"],
            ["Baseline poison retrieval and exposure", "Measured: 6 retrieved, 6 sent", "rag_results.jsonl"],
            ["Filtered poison exposure", "Measured: 6 retrieved, 0 sent", "rag_results.jsonl"],
            ["Answer correctness", "Pending", "Reviewed QA references"],
            ["Statistical paired significance", "Pending", "Matched baseline/filtered trials"],
        ], [2.15 * inch, 1.35 * inch, 2.8 * inch]),
        rich("6. Numerical results: D. Answer-quality evaluation", styles["h2"]),
        paragraph("No answer-correctness, faithfulness, hallucination, or statistical paired answer-quality metric is reported. The only QA file currently present is the unreviewed template, with empty reference answers and zero reviewed questions. These metrics remain future work until a reviewed QA manifest is supplied.", styles["callout"]),
        PageBreak(),
    ])

    story.extend([
        rich("7. Reproducibility protocol", styles["h1"]),
        paragraph("The following commands reproduce the current offline artifacts from the project root. Use the repository virtual environment or an equivalent Python environment with requirements.txt installed.", styles["body"]),
        paragraph("python -m evaluation.prepare_corpus\npython -m evaluation.evaluate_filter\npython -m evaluation.report\npython -m evaluation.generate_pdf_report", styles["code"]),
        paragraph("To execute the paired live experiment, first copy the QA template, manually verify the reference answers against the PDFs, set review_status to REVIEWED, configure the Hugging Face and Ollama credentials, and run the paired runner. The runner records raw answers and resumes completed trials.", styles["body"]),
        paragraph("python -m evaluation.run_rag_experiment --qa evaluation/qa_manifest.reviewed.json\npython -m evaluation.generate_pdf_report", styles["code"]),
        rich("8. Limitations and threats to validity", styles["h1"]),
        table([
            ["Issue", "Impact on claims"],
            ["Project CSV was included in training", "Do not call the 129-row result an unbiased held-out classifier score."],
            ["Clean-domain false positives are high", "Filtering may reduce useful evidence and answer quality."],
            ["No reviewed QA references yet", "Answer accuracy and correctness cannot yet be reported."],
            ["Ollama generation is external and nondeterministic", "Use paired order, raw-output retention, model identity, and repeated trials."],
            ["Only five paired questions so far", "Exposure statistics are preliminary and need a larger reviewed benchmark."],
            ["Single configured threshold pair", "Threshold sensitivity and ablation studies are still required."],
        ], [2.15 * inch, 4.15 * inch]),
        rich("9. Research-paper interpretation", styles["h1"]),
        paragraph("A defensible current statement is: after provenance correction and no-retraining calibration, the gate retained 35 of 50 uncontaminated PDF chunks, with a 30.00% clean false-positive rate and 122 detections among 194 poison/contaminated chunks, for 62.89% recall. This is an improved but incomplete operating point; end-to-end answer claims require a reviewed QA benchmark and a larger paired trial set.", styles["callout"]),
    ])
    return story


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the full SecRAG evaluation PDF.")
    parser.add_argument("--filter", type=Path, default=DEFAULT_FILTER)
    parser.add_argument("--original-filter", type=Path, default=DEFAULT_ORIGINAL_FILTER)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--rag", type=Path, default=DEFAULT_RAG)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--qa", type=Path, default=ROOT / "evaluation" / "qa_manifest.template.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    filter_result = json.loads(args.filter.read_text(encoding="utf-8"))
    original_filter_result = json.loads(args.original_filter.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    training_summary = json.loads(args.summary.read_text(encoding="utf-8")) if args.summary.exists() else {}
    qa_manifest = json.loads(args.qa.read_text(encoding="utf-8")) if args.qa.exists() else {}
    styles = make_styles()
    document = SimpleDocTemplate(str(args.output), pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42, title="SecRAG Full Evaluation Report", author="SecRAG Project")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document.build(build_story(filter_result, original_filter_result, manifest, load_jsonl(args.rag), training_summary, qa_manifest, styles), onFirstPage=page_number, onLaterPages=page_number)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
