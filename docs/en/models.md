# Models

[中文](../zh/models.md) | **English**

## Main model

- Uni-Mol 15-layer encoder (512-dim) plus a 3-layer TransformerDecoder;
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

## Performance and limits

| Item | Reference |
|---|---|
| Main model, one molecule (CPU) | ~75 ms |
| Vice, one prediction (CPU) | ~47 s |
| Vice, first call | may be slower, depending on disk and cache |
| Vice directory size | ~15 GB |

- The main cost of the vice model is running 426 ensembles;
- Put the vice directory on SSD/NVMe, or pre-warm it at service start;
- Model files are not shipped with the repository; `from_pretrained` points at a local directory.
