# Why code-only

[中文](../zh-CN/09-why-code-only.md) | **English** · [Documentation home](../../README.md#english)

**Level:** 0 — Scope

**Prerequisites:** None.

This repository contains code, configuration and documentation only. It does not include training data, pretrained models or model weights.

## Data

The 426 labels come from HMDB `Alternative Parents` (ClassyFire terms). HMDB data is subject to its redistribution terms, so the repository cannot ship raw or derived tables. To reproduce, obtain the data from the [HMDB downloads page](https://hmdb.ca/downloads) under its terms; LIPID MAPS is an optional supplement, described in the [data pipeline](./03-data-pipeline.md).

## Models

Trained model directories (`label_map.json` + `main_model/`) are not distributed with the repository. Model size, training cost and data version depend on your own data; retraining locally is more controllable than passing around a fixed bundle. See [Training](./04-training.md).

## Just want results

Prepare data and train a model first, then predict with the CLI:

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

Without a model directory, start from [Quickstart](./02-quickstart.md), then complete data preparation and training following the [data pipeline](./03-data-pipeline.md) and [Training](./04-training.md).

**Next:** [Installation](./01-installation.md)
