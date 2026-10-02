# SecRAG evaluation

This folder contains reproducible measurements for the raw baseline and DeBERTa-filtered RAG pipelines.

## Offline preparation

Run from the project root:

```powershell
python -m evaluation.prepare_corpus
python -m evaluation.evaluate_filter
python -m evaluation.generate_pdf_report
```

The first command loads the five policy PDFs using production chunking settings and records every row from `poisoned_chunks.csv` exactly once. The second command evaluates the local DeBERTa model and writes auditable per-chunk decisions.

The corpus builder also whitespace-normalizes the CSV attacks and checks whether they already occur inside PDF chunks. Matching PDF chunks are labeled `contaminated_pdf`, not clean, so the clean false-positive rate is not inflated by poisoned source material.

The project CSV was included in filter training. Its results are an attack challenge-set result, not an unbiased held-out classifier result. Use untouched BIPIA/deepset test data for generalization claims.

## QA review

Copy `qa_manifest.template.json` to a reviewed manifest, fill in each reference answer from the named PDF, and set `review_status` and each `reviewed` field to `REVIEWED`/`true`. Do not report answer-correctness metrics from the template.

## Paired RAG trials

After reviewing the QA manifest and configuring Hugging Face/Ollama credentials:

```powershell
python -m evaluation.run_rag_experiment --qa evaluation/qa_manifest.reviewed.json
```

The runner uses the same isolated vector collection, question, embedding configuration, and `top_k` for both modes. It randomizes paired trial order, preserves raw answers and retrieved context as JSONL, and resumes trials already present in the output file.

Generated files are written below `evaluation/artifacts/`: `corpus_manifest.json`, `filter_results.json`, and `rag_results.jsonl`. These may contain provider outputs and should not be committed if sensitive. No API keys are written.

The full paper-support PDF is `evaluation/artifacts/SecRAG_full_evaluation_report.pdf`. Regenerate it after updating the filter or live RAG artifacts.
