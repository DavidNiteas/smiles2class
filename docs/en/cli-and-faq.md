# CLI and FAQ

[中文](../zh/cli-and-faq.md) | **English**

## CLI

```shell
smiles2class predict \
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

### Why is a row all NaN?

The SMILES could not be parsed. `status="invalid"`; other rows are unaffected.

### Can I use `degraded_2d` rows?

With care. The main model was trained on 3D conformers; 2D is out of distribution. Filter by `status` or de-prioritize downstream.

### Are duplicate SMILES collapsed?

No. Each row is featurized and predicted independently.

### Why is the vice model so slow?

It evaluates 426 AutoGluon ensembles sequentially, about 47 s per call at steady state; cold start on an HDD is slower. Use SSD/NVMe or pre-warm.

### Can it use a GPU?

The main model supports `device="cuda"`; the vice predictors are tabular AutoGluon models and stay on CPU.

### Should I normalise the probabilities?

No. The 426 outputs are independent binary probabilities and do not sum to one; normalising changes the semantics.

## Development and tests

```shell
pip install -e ".[dev,vice]"
pytest tests/ -q                  # skips the full vice golden by default
S2C_FULL_VICE=1 pytest tests/ -q  # includes the 426-predictor golden
```

Tests read the model directory from `S2C_MODEL_DIR`; if unset they use `<repo>/../model/uni_classify` and skip when it is absent.
