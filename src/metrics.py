"""Evaluation metrics for question answering (F1, Exact Match).

Based on the official SQuAD evaluation script:
https://rajpurkar.github.io/SQuAD-explorer/
"""

from __future__ import annotations

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = "".join(ch for ch in s if ch not in string.punctuation)
    return " ".join(s.split())


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return float(pred_tokens == gt_tokens)
    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match(prediction: str, ground_truth: str) -> float:
    """Check if normalized prediction exactly matches ground truth."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def max_over_ground_truths(metric_fn, prediction: str, ground_truths: list[str]) -> float:
    """Take the max metric score over all ground truth answers."""
    return max(metric_fn(prediction, gt) for gt in ground_truths)
