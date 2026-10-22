# AGENTS.md

Instructions for AI assistants working in `smiles2class`.

## What this repo is

`smiles2class` converts SMILES into probabilities for 426 molecular classes. It
provides a `main` classifier, an optional `vice` classifier, HMDB/LIPID MAPS data
pipelines, training, evaluation and a CLI.

It does not ship training data, pretrained models or model weights. The encoder
comes from `unimol-lite`, a lightweight modified implementation of Uni-Mol; it is
not the official Uni-Mol release.

## First, figure out what the user wants

| User asks for | What to do |
|---|---|
| "Predict these SMILES" | Check for a model directory; if none, train one first |
| "Train a model" | Check data, build the 426-label table, split, prepare, train |
| "Build a dataset" | Obtain HMDB / LIPID MAPS, clean, label, split |
| "Reproduce the reported result" | Ask for the data version, split and model directory; never invent metrics |
| "Just run something" | Use a small synthetic smoke test and label it as a demo |

Do not start a long training run before checking data columns, label counts and
the split. Do not invent a result if no data or model is available.

## Hard rules

- Do not commit data, model weights, run artifacts or generated reports.
- Do not fabricate data, labels or metrics.
- Do not download from unofficial mirrors or bypass access controls.
- Do not substitute official Uni-Mol for `unimol-lite`.
- Do not reorder, add or remove the 426 labels; the order is part of the model.
- Do not report a metric you did not compute. State the command and the split.
- Keep changes small and run the checks before answering.

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip

# Prediction with an existing trained model
python -m pip install -e ".[vice]"

# Data pipeline, training and evaluation
python -m pip install -e ".[data,train,vice,eval]"
```

Verify:

```bash
python -c "import smiles2class; print(smiles2class.__version__)"
python -m pipelines.hmdb.clean --help
python -m training.main.prepare --help
```

There is no CI in this repo. Run the checks locally.

## Data sources: talk to the user first

The repo does not include data. Ask before downloading anything:

- Do you already have HMDB / LIPID MAPS tables, or should we obtain them?
- Which HMDB release and which LIPID MAPS export?
- Do you accept the source terms and redistribution restrictions?
- Do you need LIPID MAPS, or is HMDB-only enough for the first run?
- Where should raw data, work files and runs be stored?

| Source | Official page | Download | Notes |
|---|---|---|---|
| **HMDB** | <https://hmdb.ca/downloads> | Full metabolite table (CSV preferred) | 426 labels come from `Alternative Parents`; required for training. Academic terms; do not redistribute. |
| **LIPID MAPS** | <https://www.lipidmaps.org/databases/lmsd/download> | `LMSD Structure-data file (SDF)` | Optional; SDF is CC BY 4.0. Convert to CSV before use. |

Recommendations:

- **First run:** HMDB only. It is enough to build the 426-label table.
- **Add lipid coverage:** HMDB + LIPID MAPS. Note that LIPID MAPS rows have no
  `Alternative Parents`, so their labels are all zero. Confirm with the user
  before merging.
- **Do not** substitute a random dataset or fabricate `Alternative Parents`.
- **Do not** use the LIPID MAPS TSV for labels: it lacks taxonomy fields. Use
  the SDF.

HMDB may return a Cloudflare challenge or 403. Ask the user to download in a
browser; do not try to bypass it.

## Data gate

The HMDB table needs at least:

```text
HMDB ID, SMILES, IUPAC Name, Kingdom, Super Class, Class, Sub Class, Alternative Parents
```

Check before running anything:

```bash
python - <<'PY'
import pandas as pd

frame = pd.read_csv("data/raw/hmdb.csv", nrows=5)
required = [
    "HMDB ID", "SMILES", "IUPAC Name",
    "Kingdom", "Super Class", "Class", "Sub Class",
    "Alternative Parents",
]
missing = [column for column in required if column not in frame.columns]
print("columns:", list(frame.columns))
print("missing:", missing)
if missing:
    raise SystemExit("HMDB table is missing required columns")
PY
```

For LIPID MAPS, convert the SDF to CSV with RDKit and check for `LM_ID`,
`NAME`, `SMILES`, `CATEGORY`, `MAIN_CLASS`, `SUB_CLASS`, `CLASS_LEVEL4`.

## Build the 426-label table

```bash
python -m pipelines.hmdb.clean \
  --input data/raw/hmdb.csv \
  --output work/train_compounds.parquet \
  --keep-alternative-parents

python -m pipelines.hmdb.labels \
  --input work/train_compounds.parquet \
  --output work/label_map.json \
  --labels-output work/labels.parquet

python -m pipelines.hmdb.split \
  --input work/labels.parquet \
  --output work/splits.json \
  --stats-output work/split_stats.json
```

Checks:

- `work/label_map.json` has exactly 426 entries.
- `work/labels.parquet` has `id`, `SMILES` and `label_0..label_425`.
- train / vali / test have no overlap.
- With 2–3 rows, the split may put everything in test; use enough rows for a
  non-empty train split.

## Train

```bash
python -m training.main.prepare \
  --labels work/labels.parquet \
  --splits work/splits.json \
  --output work/dataset.pkl \
  --smiles-column SMILES \
  --workers 8

python -m training.main.train \
  --dataset work/dataset.pkl \
  --unimol <unimol_lite_encoder_dir> \
  --training-config training/main/configs/default.yaml \
  --output runs/main \
  --export
```

`--unimol` must point at a UniMol-Lite encoder directory. If the user has none,
ask where it should come from; do not substitute official Uni-Mol weights.

Optional vice model:

```bash
python -m training.vice.prepare \
  --input work/labels.parquet \
  --model-dir runs/main \
  --output work/vice.parquet \
  --smiles-column SMILES

python -m training.vice.train \
  --dataset work/vice.parquet \
  --output runs/main/vice_model \
  --label-map runs/main/label_map.json
```

Output:

```text
runs/main/
├── main_model/config.json
├── main_model/model.safetensors
└── label_map.json
```

## Predict and evaluate

```bash
smiles2class \
  --model main \
  --model-path runs/main \
  --input-csv input.csv \
  --output preds.csv
```

```bash
python -m evaluation.evaluate \
  --input work/labels.parquet \
  --model main \
  --model-path runs/main \
  --smiles-column SMILES \
  --output evaluation/reports/main
```

`status="invalid"` rows have `NaN` probabilities; `status="degraded_2d"` rows are
out of distribution and should be filtered or down-weighted.

## Missing dependencies

| Error | Fix |
|---|---|
| `ModuleNotFoundError: autogluon` | `python -m pip install -e ".[vice]"` |
| `ModuleNotFoundError: pytorch_lightning` | `python -m pip install -e ".[train]"` |
| `ModuleNotFoundError: sklearn` | `python -m pip install -e ".[eval]"` |
| `ModuleNotFoundError: pyarrow` | `python -m pip install -e ".[data]"` |

## Before you answer

Run what is relevant:

```bash
pytest tests/ -q
python -c "import smiles2class; print(smiles2class.__version__)"
git status --short
```

Remove any data, model or report artifact from the commit.

When you report results, include:

- the command,
- the data source and version,
- the split (train/validation/test),
- the metrics and threshold,
- what was not run or not available.

## If the user says "just give me the result"

1. Look for a model directory or attached data. If neither exists, explain that
   the repo ships no weights and ask for the data.
2. If they only want to see the pipeline work, run a small synthetic smoke test
   and label it as a demo.
3. Never make up scores or predictions.

## 中文速览

先确认用户是要训练、预测、建数据集，还是只跑示例。本仓库不提供数据或预训练模型，
预测前必须先训练。数据只从 HMDB / LIPID MAPS 官方页面获取，先确认许可、版本和保存
路径；HMDB 遇到 Cloudflare/403 时让用户在浏览器下载。426 标签来自 HMDB 的
`Alternative Parents`，训练表必须保留该列。`labels.parquet` 的 SMILES 列名是大写，
训练和评测要显式传 `--smiles-column SMILES`；主模型 prepare 的参数是 `--labels`。
不要编造数据、指标或模型，不要提交 `data/`、`work/`、`runs/` 或报告。
