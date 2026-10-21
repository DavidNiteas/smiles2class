# smiles2class overview

[中文](../zh/index.md) | **English**

`smiles2class` turns SMILES into **probabilities for 426 molecular classes**. It ships two **independent** implementations:

- **main**: a UniMol encoder plus 426 class-query decoders — one model, all classes;
- **vice**: 426 per-class AutoGluon classifiers over the main model's 512-dimensional embedding.

## Features

- **Order-preserving**: one output row per input row; duplicate SMILES are kept;
- **Explicit status**: invalid SMILES, 2D degradation and timeouts are surfaced, never silent;
- **Independent implementations**: use main alone, or add the `[vice]` extra;
- **Pip-installable**: hatchling build, `smiles2class` console entry point;
- **CPU-first**: inference works on CPU; `device="cuda"` is available for the main model.

## Documentation map

| Page | What you will find |
|---|---|
| [Getting started](getting-started.md) | Installation, model directory, minimal examples, CLI |
| [API](api.md) | Classes and signatures, parameters, return values, status model |
| [Models](models.md) | Architecture, labels, probability semantics, runtime and limits |
| [Evaluation](evaluation.md) | Per-class metrics, AUROC/AP, reports |
| [CLI and FAQ](cli-and-faq.md) | Command line, common questions, development |

## Version and license

Version `0.1.0`; code under MIT. Class names come from HMDB `Alternative Parents` (ClassyFire terms); the selection and order of the 426 classes are part of the model.
