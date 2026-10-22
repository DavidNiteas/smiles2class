# Evaluation

[中文](../zh-CN/08-evaluation.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Reference

**Prerequisites:** [Training](./04-training.md).

`evaluation/` computes per-class metrics from a multi-hot ground-truth matrix and predicted probabilities, and writes a report. The evaluation code is not included in the wheel; install its extra first:

```bash
pip install -e ".[eval]"      # scikit-learn, for AP / ROC AUC
```

## Metric conventions

Each class is an independent one-vs-rest binary problem:

| Metric | Meaning | Threshold-dependent |
|---|---|---|
| precision | Of the samples predicted positive, the share that are positive | yes |
| recall | Of the positive samples, the share predicted positive | yes |
| F1 | Harmonic mean of precision and recall | yes |
| average precision (AP) | Area under the precision-recall curve | no |
| ROC AUC | Ranking separability of positives vs negatives | no |
| balanced accuracy | (recall + specificity) / 2 | yes |

Rules:

- **Threshold**: `0.5` by default, configurable with `--threshold`. It only affects precision / recall / F1 / balanced accuracy.
- **AUROC and AP take continuous scores (probabilities)**: the raw per-class probability is passed to the metric. `argmax` or thresholded 0/1 labels are not used as scores, because thresholding discards ranking information and makes AUROC meaningless.
- **Multi-label**: truth is multi-hot, so one molecule may belong to several classes; classes are scored independently.
- **Invalid scores**: `NaN` predictions (e.g. an invalid SMILES with no output) are ignored per class and do not contribute to that class's metrics.

### Zero positives and zero negatives

- **Zero-positive class** (never positive in the truth): precision / recall / F1 / AP / ROC AUC / balanced accuracy are undefined and recorded as `NaN` in the per-class table;
- **Zero-negative class** (positive in every row): AP / ROC AUC / balanced accuracy are undefined and recorded as `NaN`; precision / recall / F1 are still computed at the threshold;
- `zero_positive` / `zero_negative` in the summary count these degenerate classes; `excluded` counts, per metric, how many classes were dropped because of `NaN`;
- **macro** averages only the defined (non-`NaN`) classes; undefined classes are not counted as zero. If every class is undefined for a metric, the macro value is `null`;
- **micro** sums the per-class confusion matrices (tp / fp / fn / tn) first and derives P/R/F1 and balanced accuracy from the flattened matrix; micro AP / ROC AUC are computed over all flattened `(sample, class)` scores.

## Two ways to run

### 1. Existing predictions

The input table holds SMILES and truth label columns; the prediction table holds the same label columns. Row order must match.

```bash
python -m evaluation.evaluate \
  --input labels.csv \
  --predictions preds.csv \
  --output report
```

### 2. Run the model directly

Without `--predictions`, the tool runs inference from a model directory. `--model` is `main` (default) or `vice`.

```bash
python -m evaluation.evaluate \
  --input labels.csv \
  --model main \
  --model-path /path/to/model_dir \
  --output report
```

Other options:

- `--threshold`: decision threshold, default `0.5`;
- `--smiles-column`: SMILES column name, default `smiles`; for the `labels.parquet` produced by `pipelines.hmdb.labels`, the column is `SMILES`, so pass `--smiles-column SMILES` explicitly;
- `--label-map`: optional JSON. A list names `label_0..N` columns positionally (or matches same-named columns); an object maps `{column: display name}`. Without it, `label_0..N` columns are preferred, otherwise every column except SMILES and `status` / `id` is treated as a label.

### Label column conventions

Truth label columns may be:

- class names (same as the model's `label_map.json`);
- `label_0`, `label_1`, … (matched to model outputs by position);
- arbitrary names, mapped to readable class names with `--label-map`.

## Reports

`--output` is a prefix that produces two files (no model or data side products):

- `<prefix>_per_class.csv`: one row per class with
  `label, n_pos, n_neg, n_pred_pos, tp, fp, fn, tn, precision, recall, f1, average_precision, roc_auc, balanced_accuracy`;
- `<prefix>_summary.json`: aggregate summary:

```json
{
  "threshold": 0.5,
  "n_samples": 4,
  "n_classes": 2,
  "zero_positive": 0,
  "zero_negative": 0,
  "macro": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "average_precision": 1.0,
    "roc_auc": 1.0,
    "balanced_accuracy": 1.0
  },
  "excluded": {
    "precision": 0,
    "recall": 0,
    "f1": 0,
    "average_precision": 0,
    "roc_auc": 0,
    "balanced_accuracy": 0
  },
  "micro": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "average_precision": 1.0,
    "roc_auc": 1.0,
    "balanced_accuracy": 1.0
  }
}
```

## Fast and slow tests

The evaluation fast tests need neither a model nor data:

```bash
PYTHONPATH=src:. pytest tests/evaluation/ -q
```

They cover the metric conventions (perfect predictions, zero-positive classes, threshold vs AP/AUROC) and the CLI on synthetic CSVs.

Full and slow suites:

```bash
PYTHONPATH=src:. pytest tests/ -q                 # fast tests
PYTHONPATH=src:. pytest tests/ -q -m slow         # slow tests only
```

Fast tests use synthetic data and can run in CI; full model evaluation needs local models and a truth table, so it should run offline with reports under `evaluation/reports/` (gitignored).

**Next:** [Why code-only](./09-why-code-only.md)
