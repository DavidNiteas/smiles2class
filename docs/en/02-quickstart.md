# Getting started

[中文](../zh-CN/02-quickstart.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Start

**Prerequisites:** [Installation](./01-installation.md).

## Model directory

```text
<model_dir>/
├── label_map.json          # 426 class names; the order is part of the model
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 AutoGluon predictor directories (vice only)
```

Model files are not distributed with the repository; train a model following [Training](04-training.md).

## Minimal example: main

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(
    ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "not-a-smiles"],
    return_meta=True,
)
print(probs.shape)                     # (3, 426)
print([m["status"] for m in meta])     # ['ok', 'ok', 'invalid']
print(probs.iloc[2].isna().all())      # True
```

## Minimal example: vice

```python
from smiles2class import ViceClassifier

clf = ViceClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO"], return_meta=True)
print(probs.shape)                     # (1, 426)
```

The vice classifier runs the main model first to obtain embeddings, then walks the 426 predictors. The first prediction can take a while.

## Command line

```shell
smiles2class \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv
```

The output CSV contains `smiles`, `status` and the 426 probability columns.

## Next steps

- Parameters and return values: [API](06-api.md)
- Model architecture and probability semantics: [Models](05-models.md)
- Command line and common questions: [CLI and FAQ](07-cli-and-faq.md)
- Agent-driven installation, data preparation and training: [AGENTS.md](../../AGENTS.md)

**Next:** [Data pipeline](./03-data-pipeline.md)
