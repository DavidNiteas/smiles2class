# Getting started

[中文](../zh/getting-started.md) | **English**

## Requirements

- Python 3.10 – 3.11;
- `rdkit>=2025.9,<2026`, `torch>=2.0`, `transformers>=4.47,<5`, `numpy<2`;
- the vice model additionally needs `autogluon.tabular==1.2.0` and `scikit-learn==1.5.1`.

## Installation

```shell
# Main model only
pip install "git+ssh://git@github.com/MetaEngine-Dev/smiles2class.git@v0.1.0"

# Main + vice
pip install "smiles2class[vice] @ git+ssh://git@github.com/MetaEngine-Dev/smiles2class.git@v0.1.0"
```

> The repository is private, hence the SSH URL. Once public, use HTTPS:
> `pip install "git+https://github.com/MetaEngine-Dev/smiles2class.git@v0.1.0"`.
> The `unimol-lite@v0.1.0` dependency is also private for now.

Development install:

```shell
git clone git@github.com:MetaEngine-Dev/smiles2class.git
cd smiles2class
pip install -e ".[dev,vice]"
pytest tests/ -q
```

## Model directory

```text
<model_dir>/
├── label_map.json          # 426 class names; the order is part of the model
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 AutoGluon predictor directories (vice only)
```

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
