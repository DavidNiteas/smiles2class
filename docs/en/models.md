# Models

[中文](../zh/models.md) | **English** · [Documentation home](../../README.md#en)

## Main model

> The encoder comes from the companion library UniMol-Lite, not official Uni-Mol. UniMol-Lite is a lightweight, modified implementation of Uni-Mol.

- A Uni-Mol-style 15-layer encoder (512-dim) from UniMol-Lite plus a 3-layer TransformerDecoder;
- 426 class-query tokens, each projected by `Linear(512, 2)` into a yes/no probability;
- The output is **426 independent binary probabilities**, not a 426-way mutually exclusive softmax; values do not sum to one across classes;
- The 512-dim CLS embedding is also returned for the vice model.

Loading uses `main_model/config.json` and `main_model/model.safetensors` with a strict state-dict check.

## Vice model

- 426 `autogluon.tabular.TabularPredictor` instances, one per class;
- Each predictor contains 13 base models plus a `WeightedEnsemble`;
- Input is the main model's 512-dim embedding, with columns `vec_0..vec_511`;
- Output is `(N, 426)`, one probability per class;
- Sequential by default: 426 ensembles are evaluated one after another.

## Label space

- The 426 names come from HMDB `Alternative Parents` (ClassyFire terms);
- The **selection and order** of the 426 classes are part of the model: row k of `label_map.json` matches output column k;
- Reordering or adding classes requires retraining, not just editing JSON.

## Probability semantics

- Both implementations return **per-class independent probabilities**; there is no cross-class normalisation;
- Main and vice were trained under different distributions (natural prevalence vs balanced sampling), so their probability scales differ.

## Runtime and limits

| Item | Note |
|---|---|
| Main model | runs on CPU or GPU |
| Vice model | 426 ensembles evaluated sequentially; the first call may be slow |
| Vice directory | large; SSD/NVMe storage is preferable |
| Model files | not shipped; `from_pretrained` points at a local directory |

- The vice cost is dominated by the 426 ensembles;
- Pre-warming at service start and using a local cache reduce first-call latency;
- Runtime depends on hardware, model directory and input distribution.
