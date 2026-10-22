# Installation

[中文](../zh-CN/01-installation.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 1 — Start

**Prerequisites:** Python `>=3.10,<3.12` (3.10 or 3.11) and `pip`.

## Install from Git

Main model only:

```shell
python -m pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

Main and vice model:

```shell
python -m pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

With an SSH key configured, `git+ssh://git@github.com/DavidNiteas/smiles2class.git@v0.1.1` can be used instead.

This also installs `unimol-lite@v0.2.1`. UniMol-Lite is a lightweight, modified implementation of Uni-Mol, not the official Uni-Mol release.

`smiles2class` and `unimol-lite` both declare `requires-python = ">=3.10,<3.12"`, so a combined install must use Python 3.10 or 3.11.

## Development install

```shell
git clone https://github.com/DavidNiteas/smiles2class.git
cd smiles2class
python -m pip install -e ".[dev]"
```

With the vice model:

```shell
python -m pip install -e ".[dev,vice]"
```

## Optional extras

| Extra | Purpose |
|---|---|
| `vice` | 426 per-class AutoGluon predictors |
| `data` | HMDB / LIPID MAPS pipelines (`pyarrow`, `PyYAML`) |
| `train` | Main and vice training (PyTorch Lightning, AutoGluon) |
| `eval` | Per-class evaluation (scikit-learn) |
| `dev` | Tests and development tools |

## Verify the install

```shell
python -c "import smiles2class; print(smiles2class.__version__)"
smiles2class --help
```

## Model files

This project does not provide pretrained models and does not distribute model weights. Train a model following [Training](04-training.md) to produce a model directory:

```text
<model_dir>/
├── label_map.json
├── main_model/
└── vice_model/        # vice only
```

## Build a wheel

```shell
python -m pip install build
python -m build
python -m pip install dist/smiles2class-*.whl
```

## Common installation problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: autogluon` | Install `.[vice]` or `.[train,vice]` |
| `ModuleNotFoundError: rdkit` | Check the Python version and network, then reinstall dependencies |
| Python 3.12 or newer | Use Python 3.10 or 3.11 |
| Build failure | Upgrade `pip` and confirm `hatchling` is available |

**Next:** [Quickstart](./02-quickstart.md)
