from __future__ import annotations

import math
import random
from collections import Counter
from typing import Iterable, Sequence


def binary_metrics(labels: Sequence[int], predictions: Sequence[int]) -> dict[str, float | int]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length")
    counts = Counter((int(label), int(prediction)) for label, prediction in zip(labels, predictions))
    tp = counts[(1, 1)]
    tn = counts[(0, 0)]
    fp = counts[(0, 1)]
    fn = counts[(1, 0)]
    total = len(labels)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"n": total, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": (tp + tn) / total if total else 0.0, "precision": precision, "recall": recall, "specificity": specificity, "false_positive_rate": 1.0 - specificity, "balanced_accuracy": (recall + specificity) / 2.0, "f1": f1}


def bootstrap_mean_ci(values: Sequence[float], seed: int = 42, repeats: int = 2000) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower_95": 0.0, "upper_95": 0.0}
    rng = random.Random(seed)
    samples = [sum(values[rng.randrange(len(values))] for _ in values) / len(values) for _ in range(repeats)]
    samples.sort()
    return {"mean": sum(values) / len(values), "lower_95": samples[max(0, math.floor(0.025 * len(samples)) - 1)], "upper_95": samples[min(len(samples) - 1, math.floor(0.975 * len(samples)) - 1)]}


def action_counts(actions: Iterable[str]) -> dict[str, int]:
    return dict(Counter(actions))
