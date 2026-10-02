from __future__ import annotations

import base64
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Dict, List


DIRECTIVE_PATTERNS = (
    r"\b(?:system|automated|retrieval|knowledge|language)\s+(?:directive|instruction|rule|note|guidance)\b",
    r"\b(?:assistant|agent|language model|automated system|automated response system|retrieval agent|retrieval engine|knowledge assistant|knowledge tool|language tool|virtual support assistant|information retrieval agent|AI|Al)\b.{0,100}\b(?:must|should|required to|instruct|tell|advise|inform|rephrase|state|respond|answer|output|append|prepend)\b",
    r"\bif\s+(?:a user|the query|the prompt)\s+(?:asks|contains|includes)\b.{0,120}\b(?:answer|state|respond|output|tell|instruct|assistant|agent|model|system)\b",
    r"\b(?:ignore|disregard)\s+(?:all\s+)?(?:previous|prior|system|user)\s+instructions?\b",
    r"\b(?:output|append|prepend|terminate|rephrase)\b.{0,80}\b(?:answer|response|text|assistant|agent|model)\b",
)


@dataclass(frozen=True)
class AnomalyResult:
    entropy: float
    base64_ratio: float
    zero_width_count: int
    non_alphanumeric_ratio: float
    directive_matches: int
    severity: str
    reasons: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def shannon_entropy(text: str) -> float:
    """Return character entropy in bits, or zero for empty text."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((count / length) * math.log2(count / length) for count in counts.values())


def _base64_ratio(text: str) -> float:
    candidates = re.findall(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/])", text)
    if not candidates:
        return 0.0
    encoded_length = 0
    for candidate in candidates:
        try:
            base64.b64decode(candidate, validate=True)
        except (ValueError, TypeError):
            continue
        encoded_length += len(candidate)
    return encoded_length / max(len(text), 1)


def analyze_text(
    text: str,
    *,
    entropy_threshold: float = 5.2,
    base64_ratio_threshold: float = 0.12,
    non_alphanumeric_threshold: float = 0.55,
) -> AnomalyResult:
    """Extract explainable indicators for obfuscated or directive-bearing text."""
    entropy = shannon_entropy(text)
    zero_width_count = len(re.findall(r"[\u200B-\u200D\uFEFF]", text))
    non_alphanumeric_ratio = sum(
        not character.isalnum() and not character.isspace() for character in text
    ) / max(len(text), 1)
    base64_ratio = _base64_ratio(text)
    directive_matches = sum(
        bool(re.search(pattern, text, re.IGNORECASE | re.DOTALL))
        for pattern in DIRECTIVE_PATTERNS
    )

    reasons: List[str] = []
    if entropy >= entropy_threshold and len(text) >= 80:
        reasons.append("high_entropy")
    if base64_ratio >= base64_ratio_threshold:
        reasons.append("base64_like_payload")
    if zero_width_count:
        reasons.append("zero_width_characters")
    if non_alphanumeric_ratio >= non_alphanumeric_threshold and len(text) >= 40:
        reasons.append("unusual_character_ratio")
    if directive_matches:
        reasons.append("directive_pattern")

    return AnomalyResult(
        entropy=entropy,
        base64_ratio=base64_ratio,
        zero_width_count=zero_width_count,
        non_alphanumeric_ratio=non_alphanumeric_ratio,
        directive_matches=directive_matches,
        severity="HIGH" if reasons else "OK",
        reasons=reasons,
    )