"""Metrics contract: P/R/F1 at a threshold, AP/AUROC on continuous scores."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from evaluation.metrics import per_class_metrics, summarize


def test_perfect_predictions_are_all_ones():
    y_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
        ]
    )
    y_score = y_true.astype(float)

    frame = per_class_metrics(y_true, y_score)
    assert list(frame["label"]) == ["label_0", "label_1", "label_2"]
    for metric in ["precision", "recall", "f1", "average_precision", "roc_auc", "balanced_accuracy"]:
        assert np.allclose(frame[metric], 1.0), metric

    summary = summarize(frame)
    for metric in ["precision", "recall", "f1", "average_precision", "roc_auc", "balanced_accuracy"]:
        assert math.isclose(summary["macro"][metric], 1.0), metric
        assert summary["excluded"][metric] == 0
    assert summary["micro"]["f1"] == 1.0
    assert summary["zero_positive"] == 0
    assert summary["zero_negative"] == 0


def test_zero_positive_class_is_nan_and_excluded_from_macro():
    # Column 1 has no positive example; column 0 is perfect.
    y_true = np.array([[1, 0], [0, 0], [1, 0], [0, 0]])
    y_score = np.array([[0.9, 0.1], [0.1, 0.2], [0.8, 0.3], [0.2, 0.4]])

    frame = per_class_metrics(y_true, y_score, labels=["seen", "absent"])
    absent = frame.loc[frame["label"] == "absent"].iloc[0]
    assert absent["n_pos"] == 0
    for metric in ["precision", "recall", "f1", "average_precision", "roc_auc", "balanced_accuracy"]:
        assert math.isnan(absent[metric]), metric

    summary = summarize(frame)
    assert summary["zero_positive"] == 1
    # Macro averages only over classes with defined values.
    assert summary["macro"]["f1"] == 1.0
    assert summary["macro"]["precision"] == 1.0
    assert summary["excluded"]["f1"] == 1
    assert summary["excluded"]["roc_auc"] == 1


def test_auroc_uses_probabilities_not_thresholded_labels():
    # One class, perfect ranking, but every score falls below 0.5: thresholded
    # labels are all zero (F1 = 0) while AUC/AP stay perfect.
    y_true = np.array([[0], [0], [1], [1]])
    y_score = np.array([[0.20], [0.30], [0.49], [0.48]])

    frame = per_class_metrics(y_true, y_score, labels=["class"])
    row = frame.iloc[0]
    assert row["f1"] == 0.0
    assert row["precision"] == 0.0
    assert row["recall"] == 0.0
    assert row["roc_auc"] == 1.0
    assert row["average_precision"] == 1.0
    # Had the hard labels been used as scores, AUC would be nan (single class).
    hard = (y_score >= 0.5).astype(float)
    assert len(np.unique(hard)) == 1


def test_threshold_changes_f1_but_not_ap_or_auroc():
    y_true = np.array([[0], [1]])
    y_score = np.array([[0.4], [0.6]])

    low = per_class_metrics(y_true, y_score, threshold=0.3).iloc[0]
    high = per_class_metrics(y_true, y_score, threshold=0.5).iloc[0]

    assert high["f1"] == 1.0
    assert low["f1"] < 1.0
    assert low["average_precision"] == high["average_precision"] == 1.0
    assert low["roc_auc"] == high["roc_auc"] == 1.0


def test_zero_negative_class_only_loses_ranking_metrics():
    y_true = np.array([[1, 1], [1, 1], [1, 1]])
    y_score = np.array([[0.9, 0.9], [0.8, 0.7], [0.6, 0.55]])

    frame = per_class_metrics(y_true, y_score, labels=["all_pos", "all_pos_2"])
    row = frame.iloc[0]
    assert row["precision"] == 1.0
    assert row["recall"] == 1.0
    assert row["f1"] == 1.0
    assert math.isnan(row["roc_auc"])
    assert math.isnan(row["average_precision"])
    assert math.isnan(row["balanced_accuracy"])

    summary = summarize(frame)
    assert summary["zero_negative"] == 2
    assert summary["macro"]["f1"] == 1.0
    assert summary["macro"]["roc_auc"] is None


def test_summary_is_json_friendly():
    y_true = np.array([[1], [0]])
    y_score = np.array([[0.9], [0.1]])
    frame = per_class_metrics(y_true, y_score)
    summary = summarize(frame)
    assert isinstance(summary["macro"], dict)
    assert isinstance(summary["micro"], dict)
    assert summary["n_samples"] == 2
    assert summary["n_classes"] == 1
    assert isinstance(frame, pd.DataFrame)
