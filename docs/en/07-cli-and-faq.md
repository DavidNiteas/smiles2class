# CLI and FAQ

[中文](../zh-CN/07-cli-and-faq.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 3 — Reference

**Prerequisites:** [Quickstart](./02-quickstart.md).

## CLI

```shell
smiles2class \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv \
  --smiles-column smiles \
  --workers 8 \
  --timeout 30 \
  --device cpu \
  --batch-size 64
```

| Option | Default | Description |
|---|---|---|
| `--model` | `main` | `main` / `vice` |
| `--model-path`, `-m` | required | Model directory |
| `--input-csv`, `-i` | required | Input CSV |
| `--output`, `-o` | required | Output CSV |
| `--smiles-column`, `-s` | `smiles` | SMILES column |
| `--workers` | `8` | Featurization threads |
| `--timeout` | `30.0` | Per-conformer budget |
| `--device` | `cpu` | `cpu` / `cuda` |
| `--batch-size` | `64` | Main-model forward batch |
| `--vice-workers` | `1` | Vice predictor parallelism |

The output CSV has `smiles`, `status` and the 426 probability columns; the row count matches the input.

## FAQ

### A row of NaNs

When a SMILES cannot be parsed, that row has `status="invalid"` and all 426 probabilities are `NaN`; other rows are unaffected.

### Using `degraded_2d` rows

The main model is trained on 3D conformers, so 2D is out-of-distribution input. Filter these rows by `status` or de-prioritize them downstream.

### Duplicate SMILES

Duplicate rows are not merged. Each row is featurized and predicted independently, and the output row count matches the input.

### Vice-model runtime

The vice model contains 426 AutoGluon ensembles and evaluates them sequentially by default. Runtime depends on hardware, model directory and cache state. Keep the model directory on an SSD and pre-warm at service start where needed.

### GPU support

The main model supports `device="cuda"`; the vice predictors are tabular AutoGluon models and run on CPU.

### Probability normalisation

The 426 outputs are independent binary probabilities and do not sum to one; normalisation changes their semantics.

## Development and tests

```shell
pip install -e ".[dev,vice]"
pytest tests/ -q
```

Tests read the model directory from `S2C_MODEL_DIR` and skip model-dependent tests when it is unset.

**Next:** [Evaluation](./08-evaluation.md)
