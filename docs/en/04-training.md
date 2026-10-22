# Training

[中文](../zh-CN/04-training.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 2 — Use

**Prerequisites:** [Data pipeline](./03-data-pipeline.md).

Training code lives in `training/` and is **not installed with the wheel**; run it from the repository root. Its output follows the model-directory contract used by inference.

## Install

```shell
pip install -e ".[train,vice]"
```

`[train]` provides PyTorch-Lightning and torchmetrics; vice training also needs `[vice]` (AutoGluon).

## Main model

### 1. Prepare the data

After `pipelines/` produces `labels.parquet` and `splits.json`:

```shell
python -m training.main.prepare \
  --labels work/labels.parquet \
  --splits work/splits.json \
  --output work/dataset.pkl \
  --smiles-column SMILES \
  --workers 8
```

`prepare` turns each SMILES into the four tensors with `unimol_lite` and assembles a trainable DataModule per split.

The table produced by `pipelines.hmdb.labels` uses an uppercase `SMILES` column; keep `--smiles-column SMILES` for the default pipeline output, and pass the actual column name for custom tables.

### 2. Train

```shell
python -m training.main.train \
  --dataset work/dataset.pkl \
  --unimol base_unimol \
  --training-config training/main/configs/default.yaml \
  --output runs/main \
  --export
```

- `--unimol` points at a UniMol-Lite encoder directory (not official Uni-Mol);
- `--export` writes:

```text
runs/main/
├── main_model/{config.json, model.safetensors}
└── label_map.json
```

- Batch size, learning rate, epochs, precision and accelerator come from `training/main/configs/default.yaml`;
- `--max-steps` / `--max-epochs` are available for debugging.

### 3. Use

```python
from smiles2class import MainClassifier
clf = MainClassifier.from_pretrained("runs/main")
```

## Vice model

### 1. Build embeddings

```shell
python -m training.vice.prepare \
  --input work/labels.parquet \
  --model-dir runs/main \
  --output work/vice.parquet \
  --smiles-column SMILES
```

- The input may carry `cluster_id`; when absent it is set to 0 (clustering only affects majority-class downsampling);
- Output columns: `vec_0..vec_511`, `label_*`, `cluster_id`.

### 2. Train per class

```shell
python -m training.vice.train \
  --dataset work/vice.parquet \
  --output runs/main/vice_model \
  --label-map runs/main/label_map.json
```

- One `TabularPredictor` per class, written to `runs/main/vice_model/<i>/`;
- Presets, time limit, CPU count and split ratios come from `training/vice/configs/default.yaml`;
- `--max-classes` / `--time-limit` are available for debugging;
- For subset training, `<output>/label_map.json` contains only the trained labels; for a full run use the model-root `label_map.json` produced by main-model training.

### 3. Use

```python
from smiles2class import ViceClassifier
clf = ViceClassifier.from_pretrained("runs/main")
```

## Model-directory contract

```text
<model_dir>/
├── label_map.json          # 426 class names; order is part of the model
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 AutoGluon predictor directories
```

## Reproducibility and limits

- Training code is not in the wheel; `pip install smiles2class` does not pull the training stack;
- Training data is supplied by the caller; the repository ships no raw or derived data;
- All entry points accept a fixed seed; full training is not run in CI;
- The main model runs on CPU or GPU; vice predictors are AutoGluon tabular models on CPU;
- Training 426 vice predictors is per-class work; runtime scales with the number of classes and rows.

**Next:** [Models](./05-models.md)
