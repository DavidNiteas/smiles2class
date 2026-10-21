"""Report generation: existing predictions CSV and the CLI surface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import evaluate_predictions, main

REPO_ROOT = Path(__file__).resolve().parents[2]
LABEL_COLUMNS = ["label_0", "label_1"]
TRUTH = [
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0],
]
PREDICTIONS = [
    [0.9, 0.1],
    [0.2, 0.8],
    [0.7, 0.6],
    [0.1, 0.2],
]


def _write_csvs(tmp_path: Path):
    input_path = tmp_path / "labels.csv"
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame(
        {"smiles": ["CCO", "CCC", "CCN", "CCCl"], **{
            label: [row[index] for row in TRUTH]
            for index, label in enumerate(LABEL_COLUMNS)
        }}
    ).to_csv(input_path, index=False)
    pd.DataFrame(
        {"smiles": ["CCO", "CCC", "CCN", "CCCl"], **{
            label: [row[index] for row in PREDICTIONS]
            for index, label in enumerate(LABEL_COLUMNS)
        }}
    ).to_csv(predictions_path, index=False)
    return input_path, predictions_path


def test_evaluate_predictions_dataframes():
    truth = pd.DataFrame(TRUTH, columns=LABEL_COLUMNS)
    scores = pd.DataFrame(PREDICTIONS, columns=LABEL_COLUMNS)

    per_class, summary = evaluate_predictions(truth, scores)

    assert list(per_class["label"]) == LABEL_COLUMNS
    assert len(per_class) == 2
    assert summary["n_samples"] == 4
    assert summary["n_classes"] == 2
    for metric in ["precision", "recall", "f1"]:
        assert summary["macro"][metric] == 1.0
        assert summary["micro"][metric] == 1.0
    assert summary["threshold"] == 0.5


def test_cli_with_existing_predictions(tmp_path):
    input_path, predictions_path = _write_csvs(tmp_path)
    prefix = tmp_path / "report"

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--predictions",
            str(predictions_path),
            "--output",
            str(prefix),
        ]
    )
    assert exit_code == 0

    per_class_path = tmp_path / "report_per_class.csv"
    summary_path = tmp_path / "report_summary.json"
    assert per_class_path.exists()
    assert summary_path.exists()

    per_class = pd.read_csv(per_class_path)
    assert list(per_class.columns)[:4] == ["label", "n_pos", "n_neg", "n_pred_pos"]
    assert list(per_class["label"]) == LABEL_COLUMNS
    for metric in ["precision", "recall", "f1", "average_precision", "roc_auc", "balanced_accuracy"]:
        assert metric in per_class.columns
        assert np.allclose(per_class[metric], 1.0)

    summary = json.loads(summary_path.read_text())
    assert summary["threshold"] == 0.5
    assert summary["n_samples"] == 4
    assert summary["n_classes"] == 2
    assert summary["zero_positive"] == 0
    assert summary["zero_negative"] == 0
    assert summary["macro"]["f1"] == 1.0
    assert summary["micro"]["f1"] == 1.0
    assert summary["micro"]["roc_auc"] == 1.0


def test_cli_label_map_renames_classes(tmp_path):
    input_path, predictions_path = _write_csvs(tmp_path)
    label_map = tmp_path / "label_map.json"
    label_map.write_text(json.dumps(["lipid", "steroid"]))
    prefix = tmp_path / "mapped"

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--predictions",
                str(predictions_path),
                "--label-map",
                str(label_map),
                "--output",
                str(prefix),
            ]
        )
        == 0
    )
    per_class = pd.read_csv(tmp_path / "mapped_per_class.csv")
    assert list(per_class["label"]) == ["lipid", "steroid"]


def test_cli_threshold_changes_report(tmp_path):
    input_path = tmp_path / "labels.csv"
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame({"smiles": ["CCO", "CCC"], "label_0": [1, 0]}).to_csv(
        input_path, index=False
    )
    pd.DataFrame({"smiles": ["CCO", "CCC"], "label_0": [0.4, 0.6]}).to_csv(
        predictions_path, index=False
    )

    assert (
        main(
            [
                "--input",
                str(input_path),
                "--predictions",
                str(predictions_path),
                "--threshold",
                "0.3",
                "--output",
                str(tmp_path / "t03"),
            ]
        )
        == 0
    )
    summary = json.loads((tmp_path / "t03_summary.json").read_text())
    assert summary["threshold"] == 0.3
    assert summary["micro"]["f1"] < 1.0


def test_cli_rejects_mismatched_rows(tmp_path):
    input_path, predictions_path = _write_csvs(tmp_path)
    short = pd.read_csv(predictions_path).iloc[:2]
    short.to_csv(predictions_path, index=False)
    with pytest.raises(ValueError, match="rows"):
        main(
            [
                "--input",
                str(input_path),
                "--predictions",
                str(predictions_path),
                "--output",
                str(tmp_path / "bad"),
            ]
        )


def test_module_help_runs():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [sys.executable, "-m", "evaluation.evaluate", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--predictions" in result.stdout
    assert "--label-map" in result.stdout
