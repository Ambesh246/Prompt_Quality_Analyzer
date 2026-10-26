"""Small reusable evaluation helpers for classifier tests and training reports."""
from __future__ import annotations
from typing import Any, Sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], positive_probabilities: Sequence[float]
) -> dict[str, float | None]:
    if not (len(y_true) == len(y_pred) == len(positive_probabilities)):
        raise ValueError("y_true, y_pred and positive_probabilities must have equal lengths")
    if len(y_true) == 0:
        raise ValueError("At least one labelled example is required to calculate metrics")
    metrics: dict[str, float | None] = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
    }
    try:
        metrics["roc_auc"] = round(float(roc_auc_score(y_true, positive_probabilities)), 4)
    except ValueError:
        metrics["roc_auc"] = None
    return metrics
