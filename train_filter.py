from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import jsonlines
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


SEED = 42
MODEL_NAME = "microsoft/deberta-v3-base"


def load_contexts(path: Path) -> list[str]:
    with jsonlines.open(path) as reader:
        return [row["context"] for row in reader]


def load_attacks(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return [attack for variants in raw.values() for attack in variants]


def build_bipia_split(
    context_files: list[Path],
    attack_file: Path,
    poisoned_count: int,
    seed: int,
) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    contexts = [context for path in context_files for context in load_contexts(path)]
    attacks = load_attacks(attack_file)
    insertions = (
        lambda context, attack: f"{context}\n{attack}",
        lambda context, attack: f"{attack}\n{context}",
    )

    poisoned: set[str] = set()
    while len(poisoned) < poisoned_count:
        context = rng.choice(contexts)
        attack = rng.choice(attacks)
        poisoned.add(rng.choice(insertions)(context, attack))
    return list(poisoned), list(dict.fromkeys(contexts))


def build_data(bipia_dir: Path, poisoned_csv: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct = load_dataset("deepset/prompt-injections")
    direct_train = direct["train"].to_pandas()[["text", "label"]]
    direct_test = direct["test"].to_pandas()[["text", "label"]]
    direct_train["source"] = "deepset"
    direct_test["source"] = "deepset"

    train_poison, train_clean = build_bipia_split(
        [bipia_dir / "email/train.jsonl", bipia_dir / "table/train.jsonl"],
        bipia_dir / "text_attack_train.json",
        poisoned_count=1500,
        seed=seed,
    )
    test_poison, test_clean = build_bipia_split(
        [bipia_dir / "email/test.jsonl", bipia_dir / "table/test.jsonl"],
        bipia_dir / "text_attack_test.json",
        poisoned_count=300,
        seed=seed,
    )
    bipia_train = pd.DataFrame({"text": train_clean + train_poison, "label": [0] * len(train_clean) + [1] * len(train_poison), "source": "bipia"})
    bipia_test = pd.DataFrame({"text": test_clean + test_poison, "label": [0] * len(test_clean) + [1] * len(test_poison), "source": "bipia"})

    project_poison = pd.read_csv(poisoned_csv)[["poisoned text", "label"]].rename(columns={"poisoned text": "text"})
    project_poison["label"] = 1
    project_poison["source"] = "project_csv"

    train_df = pd.concat([direct_train, bipia_train, project_poison], ignore_index=True)
    test_df = pd.concat([direct_test, bipia_test], ignore_index=True)
    train_df = train_df.drop_duplicates("text").sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.drop_duplicates("text").sample(frac=1, random_state=seed).reset_index(drop=True)
    train_texts = set(train_df["text"])
    overlap_count = test_df["text"].isin(train_texts).sum()
    if overlap_count:
        print(f"Removed {overlap_count} duplicate test rows to prevent train/test leakage.")
        test_df = test_df[~test_df["text"].isin(train_texts)].reset_index(drop=True)
    return train_df, test_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa for SecRAG injection detection.")
    parser.add_argument("--bipia-dir", type=Path, default=Path("BIPIA/benchmark"))
    parser.add_argument("--poisoned-csv", type=Path, default=Path("poisoned_chunks.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/secrag-deberta-final"))
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    train_df, test_df = build_data(args.bipia_dir, args.poisoned_csv, SEED)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    train_dataset = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True).rename_column("label", "labels")
    test_dataset = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize, batched=True).rename_column("label", "labels")
    columns = ["input_ids", "attention_mask", "labels"]
    train_dataset.set_format(type="torch", columns=columns)
    test_dataset.set_format(type="torch", columns=columns)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, dtype=torch.float32)

    def metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
        return {"accuracy": accuracy_score(labels, predictions), "precision": precision, "recall": recall, "f1": f1}

    training_args = TrainingArguments(
        output_dir=str(args.output_dir.parent / "training-checkpoints"),
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        learning_rate=2e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        report_to="none",
        seed=SEED,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=metrics)
    trainer.train()
    evaluation = trainer.evaluate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_summary.json").write_text(
        json.dumps({"model": MODEL_NAME, "seed": SEED, "train_rows": len(train_df), "test_rows": len(test_df), "test_metrics": evaluation}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved model to {args.output_dir}")
    print(evaluation)


if __name__ == "__main__":
    main()