"""Evaluate per-class predictions and write a report.

Two modes:

* existing predictions — ``--predictions preds.csv`` (no model is loaded);
* direct inference — ``--model main --model-path <model_dir>``.

Rows of the prediction table must be aligned with the input table (same order).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import fill_micro_ranking, per_class_metrics, summarize

_METADATA_COLUMNS = {"status", "id", "index", "Unnamed: 0"}


def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"unsupported input format {suffix!r}: use CSV or parquet")


def _numbered_label_columns(columns) -> list[str]:
    numbered: list[tuple[int, str]] = []
    for column in columns:
        match = re.fullmatch(r"label_(\d+)", str(column))
        if match:
            numbered.append((int(match.group(1)), str(column)))
    numbered.sort()
    return [column for _, column in numbered]


def _load_label_map(path: str | Path | None):
    if path is None:
        return None
    return json.loads(Path(path).read_text())


def _label_columns(
    frame: pd.DataFrame,
    smiles_column: str,
    label_map,
) -> tuple[list[str], list[str]]:
    """Return ``(truth_column_names, display_labels)``."""
    if isinstance(label_map, dict):
        missing = [column for column in label_map if column not in frame.columns]
        if missing:
            raise ValueError(f"label-map columns not found in input: {missing}")
        columns = list(label_map)
        return columns, [str(label_map[column]) for column in columns]

    if isinstance(label_map, list):
        columns = _numbered_label_columns(frame.columns)
        if len(columns) == len(label_map):
            return columns, [str(label) for label in label_map]
        missing = [label for label in label_map if label not in frame.columns]
        if missing:
            raise ValueError(
                "label-map is a list of length "
                f"{len(label_map)}, but the input has {len(columns)} label_N columns "
                f"and lacks: {missing}"
            )
        columns = [str(label) for label in label_map]
        return columns, columns

    columns = _numbered_label_columns(frame.columns)
    if columns:
        return columns, columns

    columns = [
        str(column)
        for column in frame.columns
        if column != smiles_column and column not in _METADATA_COLUMNS
    ]
    if not columns:
        raise ValueError("no label columns found in the input")
    return columns, columns


def _truth_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    truth = frame.loc[:, columns].to_numpy(dtype=np.float64)
    if not np.isfinite(truth).all():
        raise ValueError("ground-truth label columns contain missing or non-finite values")
    return truth


def _prediction_matrix(
    scores: np.ndarray,
    score_labels: list[str] | None,
    truth_columns: list[str],
) -> np.ndarray:
    """Align an ``(N, M)`` score matrix with the truth columns."""
    if scores.ndim != 2:
        raise ValueError(f"predictions must be 2-dimensional, got shape {scores.shape}")
    if score_labels is not None and set(truth_columns) <= set(score_labels):
        index = [score_labels.index(column) for column in truth_columns]
        return scores[:, index]
    if scores.shape[1] == len(truth_columns):
        return scores
    raise ValueError(
        f"prediction matrix has {scores.shape[1]} columns, but the input has "
        f"{len(truth_columns)} label columns"
    )


def _model_scores(
    smiles: list,
    model: str,
    model_path: str | Path,
) -> tuple[np.ndarray, list[str]]:
    from smiles2class.classifier import MainClassifier, ViceClassifier

    if model == "main":
        classifier = MainClassifier.from_pretrained(model_path)
    elif model == "vice":
        classifier = ViceClassifier.from_pretrained(model_path)
    else:
        raise ValueError(f"unknown model {model!r}: expected 'main' or 'vice'")
    frame = classifier.predict(smiles)
    return frame.to_numpy(), list(frame.columns)


def evaluate_predictions(
    y_true,
    y_score,
    labels=None,
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict]:
    """Compute ``(per_class_df, summary)`` for aligned truth/score matrices.

    ``y_true`` and ``y_score`` may be ``(N, C)`` arrays or DataFrames.  When both
    are frames, prediction columns are aligned to the truth columns by name when
    possible, otherwise positionally.
    """
    if isinstance(y_true, pd.DataFrame):
        if labels is None:
            labels = [str(column) for column in y_true.columns]
        if isinstance(y_score, pd.DataFrame) and set(labels) <= set(
            map(str, y_score.columns)
        ):
            y_score = y_score.loc[:, list(labels)]
        else:
            y_score = np.asarray(y_score)
        y_true = y_true.to_numpy()

    truth = np.asarray(y_true, dtype=np.float64)
    scores = np.asarray(y_score, dtype=np.float64)
    if truth.ndim != 2 or scores.ndim != 2:
        raise ValueError("y_true and y_score must be 2-dimensional")
    if truth.shape != scores.shape:
        raise ValueError(
            f"y_true and y_score must have the same shape, got {truth.shape} and {scores.shape}"
        )
    names = [str(label) for label in labels] if labels is not None else None
    per_class = per_class_metrics(truth, scores, labels=names, threshold=threshold)
    summary = summarize(per_class)
    fill_micro_ranking(truth, scores, summary)
    return per_class, summary


def _run(args) -> dict:
    frame = _read_table(args.input)
    if args.smiles_column not in frame.columns:
        raise ValueError(f"column {args.smiles_column!r} not found in {args.input}")

    label_map = _load_label_map(args.label_map)
    truth_columns, display_labels = _label_columns(frame, args.smiles_column, label_map)
    truth = _truth_matrix(frame, truth_columns)

    if args.predictions:
        pred_frame = _read_table(args.predictions)
        missing = [column for column in truth_columns if column not in pred_frame.columns]
        if missing:
            raise ValueError(f"prediction table is missing label columns: {missing}")
        if len(pred_frame) != len(frame):
            raise ValueError(
                f"prediction table has {len(pred_frame)} rows, input has {len(frame)}"
            )
        scores = pred_frame.loc[:, truth_columns].to_numpy(dtype=np.float64)
    else:
        if not args.model_path:
            raise ValueError("--model-path is required unless --predictions is given")
        smiles = frame[args.smiles_column].tolist()
        raw_scores, model_labels = _model_scores(
            smiles, args.model, args.model_path
        )
        if len(raw_scores) != len(frame):
            raise ValueError(
                f"model returned {len(raw_scores)} rows, input has {len(frame)}"
            )
        scores = _prediction_matrix(raw_scores, model_labels, truth_columns)

    per_class, summary = evaluate_predictions(
        truth, scores, labels=display_labels, threshold=args.threshold
    )

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    per_class_path = prefix.with_name(prefix.name + "_per_class.csv")
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    per_class.to_csv(per_class_path, index=False)
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    return {
        "per_class": str(per_class_path),
        "summary": str(summary_path),
        "rows": len(per_class),
        "threshold": summary["threshold"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-class evaluation report for smiles2class predictions."
    )
    parser.add_argument("--input", required=True, help="CSV/parquet with SMILES + labels")
    parser.add_argument(
        "--predictions",
        help="CSV/parquet with the same label columns (skips model loading)",
    )
    parser.add_argument(
        "--model",
        choices=["main", "vice"],
        default="main",
        help="model used when --predictions is absent (default: main)",
    )
    parser.add_argument("--model-path", help="model directory")
    parser.add_argument("--output", required=True, help="output prefix")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--label-map",
        help="JSON list of class names or JSON object mapping columns to names",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = _run(args)
    print(
        f"wrote {result['per_class']} and {result['summary']} "
        f"({result['rows']} classes, threshold={result['threshold']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
