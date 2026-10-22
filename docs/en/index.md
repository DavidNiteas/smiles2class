# smiles2class documentation (English)

**Language:** English · [简体中文](../zh-CN/index.md)

`smiles2class` converts SMILES into probabilities for 426 molecular classes. It provides two independent implementations (`main` and `vice`) and a full data/training/evaluation pipeline. The repository is code-only: it does not ship HMDB/LIPID MAPS data or trained weights.

## Pick a starting point

| If you want to... | Read |
|---|---|
| Understand why data and models are not shipped | [Why code-only](./09-why-code-only.md) |
| Install and run a small example | [Installation](./01-installation.md) → [Quickstart](./02-quickstart.md) |
| Use your own data | [Data pipeline](./03-data-pipeline.md) → [Training](./04-training.md) |
| Understand the model and probability semantics | [Models](./05-models.md) |
| Look up the Python API or CLI | [API](./06-api.md) · [CLI and FAQ](./07-cli-and-faq.md) |
| Measure model quality | [Evaluation](./08-evaluation.md) |

## How to read these docs

- **First time here?** [Install](./01-installation.md), then run the [Quickstart](./02-quickstart.md).
- **Need a model?** There are no pretrained weights. Prepare data, then [train](./04-training.md).
- **Already trained a model?** Use the [API](./06-api.md) or [CLI](./07-cli-and-faq.md) to predict.
- **Need to trust a metric?** Read [Evaluation](./08-evaluation.md).

## Not in this repo

- HMDB / LIPID MAPS raw or derived tables
- trained model directories or weights
- internal work orders or development notes

Authors are in [`AUTHORS`](../../AUTHORS). What is shipped is in [`NOTICE`](../../NOTICE).

## Authors and license

- David Niteas `<david.ql.d@foxmail.com>`
- License: MIT (code only)

**Next:** [Installation](./01-installation.md)
