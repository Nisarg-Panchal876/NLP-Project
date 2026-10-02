from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Protocol

from .anomaly import DIRECTIVE_PATTERNS, AnomalyResult, analyze_text


class InjectionScorer(Protocol):
    def score(self, text: str) -> float:
        """Return the probability that text contains an injection."""


@dataclass(frozen=True)
class FilterConfig:
    model_path: str | None = None
    quarantine_threshold: float = 0.85
    rewrite_threshold: float = 0.50


@dataclass(frozen=True)
class ChunkDecision:
    action: str
    injection_probability: float
    factual_probability: float
    anomaly: AnomalyResult
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "injection_probability": self.injection_probability,
            "factual_probability": self.factual_probability,
            "anomaly": self.anomaly.to_dict(),
            "text": self.text,
        }


class DebertaInjectionScorer:
    """Lazy local DeBERTa inference wrapper.

    The tokenizer and model are loaded on first use and then reused for all
    chunks handled by this process.
    """

    def __init__(self, model_path: str | Path):
        self.model_path = str(model_path)
        self._tokenizer: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Filtered mode requires torch and transformers. Install the training/inference dependencies first."
            ) from exc

        model_path = Path(self.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"DeBERTa model directory does not exist: {model_path}. Train/export the model before using filtered mode."
            )
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self._model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self._model.eval()
        self._torch = torch

    def score(self, text: str) -> float:
        self._load()
        inputs = self._tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt",
        )
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        return float(self._torch.softmax(logits, dim=-1)[0][1].item())


def rewrite_text(text: str) -> str:
    """Remove sentences that address an assistant instead of stating facts."""
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    safe_sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
        and not any(re.search(pattern, sentence, re.IGNORECASE | re.DOTALL) for pattern in DIRECTIVE_PATTERNS)
    ]
    return " ".join(safe_sentences)


def decide_chunk(
    text: str,
    scorer: InjectionScorer,
    config: FilterConfig | None = None,
) -> ChunkDecision:
    config = config or FilterConfig()
    injection_probability = min(max(float(scorer.score(text)), 0.0), 1.0)
    anomaly = analyze_text(text)
    obfuscation_detected = any(
        reason in anomaly.reasons
        for reason in ("high_entropy", "base64_like_payload", "zero_width_characters", "unusual_character_ratio")
    )
    ai_directive_detected = "directive_pattern" in anomaly.reasons

    # A high classifier score alone is insufficient because ordinary policy prose
    # is out of domain for the injection-trained classifier.
    if obfuscation_detected or (ai_directive_detected and injection_probability > config.quarantine_threshold):
        action = "QUARANTINE"
    elif ai_directive_detected:
        action = "REWRITE"
    else:
        action = "PASS"

    return ChunkDecision(
        action=action,
        injection_probability=injection_probability,
        factual_probability=1.0 - injection_probability,
        anomaly=anomaly,
        text=text,
    )


def filter_chunks(
    chunks: Iterable[Dict[str, Any]],
    scorer: InjectionScorer,
    config: FilterConfig | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return safe context chunks and auditable per-chunk decisions."""
    safe_chunks: List[Dict[str, Any]] = []
    decisions: List[Dict[str, Any]] = []
    for chunk in chunks:
        decision = decide_chunk(chunk["text"], scorer, config=config)
        audit = dict(chunk)
        audit["security"] = decision.to_dict()
        if decision.action == "PASS":
            safe_chunks.append(chunk)
        elif decision.action == "REWRITE":
            rewritten_text = rewrite_text(chunk["text"])
            if rewritten_text:
                rewritten_chunk = dict(chunk)
                rewritten_chunk["text"] = rewritten_text
                safe_chunks.append(rewritten_chunk)
                audit["security"]["rewritten_text"] = rewritten_text
        decisions.append(audit)

    return safe_chunks, decisions