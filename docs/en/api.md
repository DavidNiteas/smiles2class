# API

[中文](../zh/api.md) | **English** · [Documentation home](../../README.md#en)

## MainClassifier

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained(model_dir, device="cpu")
probs, metas = clf.predict(
    smiles,
    workers=8,
    timeout=30.0,
    coord_mode="auto",
    batch_size=64,
    return_meta=True,
)
```

| Parameter | Default | Description |
|---|---|---|
| `model_dir` | — | Directory containing `label_map.json` and `main_model/` |
| `device` | `"cpu"` | `"cpu"` or `"cuda"` |
| `workers` | `8` | Featurization threads |
| `timeout` | `30.0` | Per-conformer wall-clock budget (s); `None` disables |
| `coord_mode` | `"auto"` | `auto` / `3d` / `2d` |
| `batch_size` | `64` | Main-model forward batch size |
| `return_meta` | `False` | Also return per-row status |

Returns:

- `return_meta=False`: `pandas.DataFrame` of shape `(N, 426)`, columns = class names;
- `return_meta=True`: `(DataFrame, list[dict])`, with `meta` in input order.

## ViceClassifier

```python
from smiles2class import ViceClassifier

clf = ViceClassifier.from_pretrained(model_dir, device="cpu")
probs, metas = clf.predict(
    smiles,
    workers=8,
    timeout=30.0,
    coord_mode="auto",
    batch_size=64,
    vice_workers=1,
    return_meta=True,
)
```

- Runs the main model first to get `(N, 512)` embeddings, then the 426 predictors;
- `vice_workers` defaults to `1` (sequential); larger values are allowed but not guaranteed to be faster;
- The output is `(N, 426)`; it reuses the main model's embedding but remains an independent per-class result.

## Row alignment and edge cases

| Input | Behaviour |
|---|---|
| Valid SMILES | Normal probabilities |
| Duplicate SMILES | Two rows; not collapsed |
| Invalid SMILES | That row is all `NaN`, `status="invalid"` |
| Empty list | `(0, 426)` and empty `meta` |
| 2D fallback | Normal probabilities, `status="degraded_2d"` |
| Timeout with `on_timeout="error"` | Row is `None` / `NaN`, `status="timeout_error"` |

## The `meta` status model

| `status` | Meaning |
|---|---|
| `ok` | Normal (3D or explicit 2D) |
| `degraded_2d` | 3D failed or timed out; fell back to 2D |
| `timeout_error` | Timed out with `on_timeout="error"` |
| `invalid` | SMILES could not be parsed |
| `skipped` | Timed out with `on_timeout="skip"` |
| `error` | Unexpected exception caught by the batch layer |

`meta` also contains `coord_mode`, `reason` and `elapsed_s`.

> `degraded_2d` is out-of-distribution input: the main model is trained on 3D conformers. Filter or de-prioritize these rows by `status`.

## Labels

```python
from smiles2class import LABELS, NUM_LABELS

assert NUM_LABELS == 426
```

The order of `LABELS` is aligned with the model output columns and **must not be reordered**.
