"""Per-class metrics for multi-label molecular classification.

The entry point is :func:`per_class_metrics`, which treats every column as an
independent binary problem.  Scores are **probabilities / continuous scores**:
ROC AUC and average precision receive the raw scores, never thresholded or
``argmax``-ed labels.

Degenerate columns:

* zero positive examples — precision / recall / F1 / AP / ROC AUC / balanced
  accuracy are ``NaN`` (undefined), and excluded from macro averages;
* zero negative examples — AP / ROC AUC / balanced accuracy are ``NaN``
  (no ranking problem exists); P/R/F1 remain defined.

``NaN`` scores (e.g. invalid SMILES with no prediction) are ignored per class.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
]

COUNT_COLUMNS = ["n_pos", "n_neg", "n_pred_pos", "tp", "fp", "fn", "tn"]


def _as_matrix(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be 2-dimensional (N, C), got shape {array.shape}")
    return array


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def per_class_metrics(
    y_true,
    y_score,
    labels: Sequence[str] | None = None,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Return one row of binary metrics per class.

    Parameters
    ----------
    y_true:
        ``(N, C)`` multi-hot ground truth (values compared against 0.5).
    y_score:
        ``(N, C)`` per-class probabilities / continuous scores.
    labels:
        Optional class names, length ``C``.  Defaults to ``label_0 .. label_{C-1}``.
    threshold:
        Decision threshold for precision / recall / F1 / balanced accuracy.
        ROC AUC and average precision are threshold-independent.
    """
    truth = _as_matrix(y_true, "y_true")
    scores = _as_matrix(y_score, "y_score")
    if truth.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {truth.shape} and {scores.shape}"
        )
    n_samples, n_classes = truth.shape
    if labels is None:
        labels = [f"label_{index}" for index in range(n_classes)]
    labels = [str(label) for label in labels]
    if len(labels) != n_classes:
        raise ValueError(f"labels must have length {n_classes}, got {len(labels)}")

    truth_bool = truth > 0.5
    threshold = float(threshold)
    predicted = scores >= threshold

    rows = []
    for index, label in enumerate(labels):
        column_truth = truth_bool[:, index]
        column_scores = scores[:, index]
        finite = np.isfinite(column_scores)
        column_truth = column_truth[finite]
        column_scores = column_scores[finite]
        column_predicted = predicted[finite, index]

        n_pos = int(np.sum(column_truth))
        n_neg = int(np.sum(~column_truth))
        n_pred_pos = int(np.sum(column_predicted))
        tp = int(np.sum(column_truth & column_predicted))
        fp = int(np.sum(~column_truth & column_predicted))
        fn = int(np.sum(column_truth & ~column_predicted))
        tn = int(np.sum(~column_truth & ~column_predicted))

        if n_pos == 0:
            rows.append(
                {
                    "label": label,
                    "n_pos": 0,
                    "n_neg": n_neg,
                    "n_pred_pos": n_pred_pos,
                    "tp": 0,
                    "fp": fp,
                    "fn": 0,
                    "tn": tn,
                    "precision": np.nan,
                    "recall": np.nan,
                    "f1": np.nan,
                    "average_precision": np.nan,
                    "roc_auc": np.nan,
                    "balanced_accuracy": np.nan,
                }
            )
            continue

        precision, recall, f1 = _prf(tp, fp, fn)

        if n_neg == 0:
            # No negative examples: ranking metrics and specificity are undefined.
            average_precision = np.nan
            roc_auc = np.nan
            balanced_accuracy = np.nan
        else:
            average_precision = float(average_precision_score(column_truth, column_scores))
            roc_auc = float(roc_auc_score(column_truth, column_scores))
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            balanced_accuracy = 0.5 * (recall + specificity)

        rows.append(
            {
                "label": label,
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_pred_pos": n_pred_pos,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": average_precision,
                "roc_auc": roc_auc,
                "balanced_accuracy": balanced_accuracy,
            }
        )

    frame = pd.DataFrame(rows)
    frame.attrs["threshold"] = threshold
    frame.attrs["n_samples"] = n_samples
    return frame


def summarize(per_class: pd.DataFrame) -> dict:
    """Aggregate a :func:`per_class_metrics` frame into a JSON-friendly dict.

    Macro averages skip ``NaN`` (undefined) classes and report how many classes
    were excluded per metric.  Micro P/R/F1/balanced accuracy are computed from
    the flattened confusion matrix: the per-class ``tp`` / ``fp`` / ``fn`` /
    ``tn`` counts are summed first, then the metrics are derived from the
    totals.  Micro AP / ROC AUC need the raw scores and are filled in by
    :func:`evaluate_predictions` (left as ``None`` here).
    """
    if per_class.empty:
        raise ValueError("per_class frame is empty")

    macro: dict[str, float | None] = {}
    excluded: dict[str, int] = {}
    for metric in METRIC_COLUMNS:
        column = pd.to_numeric(per_class[metric], errors="coerce")
        valid = column.dropna()
        macro[metric] = float(valid.mean()) if len(valid) else None
        excluded[metric] = int(len(column) - len(valid))

    tp = int(per_class["tp"].sum())
    fp = int(per_class["fp"].sum())
    fn = int(per_class["fn"].sum())
    tn = int(per_class["tn"].sum())
    precision, recall, f1 = _prf(tp, fp, fn)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "threshold": float(per_class.attrs.get("threshold", 0.5)),
        "n_samples": int(per_class.attrs.get("n_samples", 0)),
        "n_classes": int(len(per_class)),
        "zero_positive": int((per_class["n_pos"] == 0).sum()),
        "zero_negative": int((per_class["n_neg"] == 0).sum()),
        "macro": macro,
        "excluded": excluded,
        "micro": {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "average_precision": None,
            "roc_auc": None,
            "balanced_accuracy": 0.5 * (recall + specificity),
        },
    }


def fill_micro_ranking(y_true, y_score, summary: dict) -> dict:
    """Fill micro AP / ROC AUC on the flattened ``(sample, class)`` matrix.

    Scores that are not finite are dropped.  If the flattened truth is a single
    class, both values stay ``None`` (sklearn would warn otherwise).
    """
    truth = (_as_matrix(y_true, "y_true") > 0.5).ravel()
    scores = _as_matrix(y_score, "y_score").ravel()
    finite = np.isfinite(scores)
    truth = truth[finite]
    scores = scores[finite]
    if truth.any() and (~truth).any():
        summary["micro"]["average_precision"] = float(
            average_precision_score(truth, scores)
        )
        summary["micro"]["roc_auc"] = float(roc_auc_score(truth, scores))
    return summary


__all__ = [
    "COUNT_COLUMNS",
    "METRIC_COLUMNS",
    "per_class_metrics",
    "summarize",
    "fill_micro_ranking",
]
