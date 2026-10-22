# smiles2class

**English** · [简体中文](#zh-cn)

<a id="english"></a>

## English

`smiles2class` converts SMILES into probabilities for 426 molecular classes. It ships two independent implementations:

- **main**: a UniMol-Lite encoder plus 426 class-query decoders, returning `(N, 426)` probabilities and a 512-dimensional embedding;
- **vice**: 426 per-class AutoGluon classifiers over the main model's embedding, returning `(N, 426)` probabilities.

### Docs

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

The docs are ordered by how deep you want to go:

| If you need to... | Read |
|---|---|
| Run something once | [Installation](docs/en/01-installation.md) → [Quickstart](docs/en/02-quickstart.md) |
| Use your own data | [Data pipeline](docs/en/03-data-pipeline.md) → [Training](docs/en/04-training.md) |
| Understand the model | [Models](docs/en/05-models.md) |
| Look up a command or signature | [API](docs/en/06-api.md) · [CLI and FAQ](docs/en/07-cli-and-faq.md) |
| Check whether a result is trustworthy | [Evaluation](docs/en/08-evaluation.md) |
| Understand why only code is shipped | [Why code-only](docs/en/09-why-code-only.md) |

### Just want the result?

The repository does not include training data or trained model weights. To get predictions, prepare data, train a model, then run the CLI. If an AI assistant is doing the work, point it at [`AGENTS.md`](./AGENTS.md); it will ask for the data source and license before downloading anything.

### What's here

- `main` and `vice` classifiers, plus a small CLI and Python API
- HMDB / LIPID MAPS data pipelines: cleaning, 426-label multi-hot, splits
- Main and vice model training, evaluation and report generation
- Explicit per-row status (`ok`, `invalid`, `degraded_2d`, `timeout_error`)

### No data or model files

- The repository does not ship HMDB or LIPID MAPS raw/derived data. HMDB data is subject to its redistribution terms; obtain it from the [official downloads page](https://hmdb.ca/downloads).
- The repository does not ship trained model directories or weights. Train on your own data; the model directory contract is in [Training](docs/en/04-training.md).
- The encoder comes from `unimol-lite`, a lightweight modified implementation of Uni-Mol, not the official Uni-Mol release.

### Install

```bash
python -m pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"

# With the vice model
python -m pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

For development:

```bash
git clone https://github.com/DavidNiteas/smiles2class.git
cd smiles2class
python -m pip install -e ".[dev,vice]"
```

### Commands

```bash
# Prepare data (after obtaining HMDB/LIPID MAPS tables)
python -m pipelines.hmdb.clean --input data/raw/hmdb.csv --output work/train_compounds.parquet --keep-alternative-parents
python -m pipelines.hmdb.labels --input work/train_compounds.parquet --output work/label_map.json --labels-output work/labels.parquet
python -m pipelines.hmdb.split --input work/labels.parquet --output work/splits.json

# Train the main model
python -m training.main.prepare --labels work/labels.parquet --splits work/splits.json --output work/dataset.pkl --smiles-column SMILES
python -m training.main.train --dataset work/dataset.pkl --unimol <unimol_lite_encoder_dir> --training-config training/main/configs/default.yaml --output runs/main --export

# Predict
smiles2class --model main --model-path runs/main --input-csv input.csv --output preds.csv
```

### Status

The pipeline and CLI are covered by tests. Tests use synthetic data and do not require model weights. There is no CI in this repository.

### Authors

- David Niteas `<david.ql.d@foxmail.com>`

See [`AUTHORS`](./AUTHORS). What's shipped is described in [`NOTICE`](./NOTICE).

### License

MIT. See [`LICENSE`](./LICENSE). Citation metadata is in [`CITATION.cff`](./CITATION.cff).

---

<a id="zh-cn"></a>

## 简体中文

[English](#english) · **简体中文**

`smiles2class` 把 SMILES 转成 426 个分子类别的概率，提供两种独立实现：

- **main**：UniMol-Lite encoder + 426 个类查询解码器，输出 `(N, 426)` 概率与 512 维 embedding；
- **vice**：426 个 AutoGluon 逐类分类器，输入为主模型的 embedding，输出 `(N, 426)` 概率。

### 文档

- **English:** [Documentation index](docs/en/index.md)
- **简体中文：** [文档索引](docs/zh-CN/index.md)

文档按使用深度排列：

| 你要做什么 | 看哪里 |
|---|---|
| 先跑通一次 | [安装](docs/zh-CN/01-installation.md) → [快速开始](docs/zh-CN/02-quickstart.md) |
| 用自己的数据 | [数据管线](docs/zh-CN/03-data-pipeline.md) → [训练](docs/zh-CN/04-training.md) |
| 理解模型 | [模型说明](docs/zh-CN/05-models.md) |
| 查命令或函数签名 | [API](docs/zh-CN/06-api.md) · [CLI 与 FAQ](docs/zh-CN/07-cli-and-faq.md) |
| 判断结果能不能信 | [评测](docs/zh-CN/08-evaluation.md) |
| 理解为什么只发代码 | [为什么只发代码](docs/zh-CN/09-why-code-only.md) |

### 只想要结果？

仓库不包含训练数据，也不包含训练好的模型权重。要得到预测，需要先准备数据、训练模型，再运行 CLI。希望 agent 代跑时，把 [`AGENTS.md`](./AGENTS.md) 交给它；它会先与你确认数据来源与许可，再下载和执行。

### 仓库里有什么

- `main` / `vice` 两套分类器，以及小型 CLI 与 Python API
- HMDB / LIPID MAPS 数据管线：清洗、426 标签 multi-hot、划分
- 主/副模型训练、评测与报告生成
- 显式逐行状态（`ok`、`invalid`、`degraded_2d`、`timeout_error`）

### 没有数据或模型文件

- 仓库不分发 HMDB / LIPID MAPS 原始或派生数据。HMDB 数据受再分发条款限制，请从 [官方下载页](https://hmdb.ca/downloads) 获取。
- 仓库不分发训练好的模型目录或权重。请用自己的数据训练；模型目录契约见[训练](docs/zh-CN/04-training.md)。
- 编码器来自 `unimol-lite`，它是 Uni-Mol 的轻量修改版实现，不是官方 Uni-Mol。

### 安装

```bash
python -m pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"

# 需要副模型时
python -m pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

开发安装：

```bash
git clone https://github.com/DavidNiteas/smiles2class.git
cd smiles2class
python -m pip install -e ".[dev,vice]"
```

### 常用命令

```bash
# 准备数据（先从 HMDB / LIPID MAPS 获取原始表）
python -m pipelines.hmdb.clean --input data/raw/hmdb.csv --output work/train_compounds.parquet --keep-alternative-parents
python -m pipelines.hmdb.labels --input work/train_compounds.parquet --output work/label_map.json --labels-output work/labels.parquet
python -m pipelines.hmdb.split --input work/labels.parquet --output work/splits.json

# 训练主模型
python -m training.main.prepare --labels work/labels.parquet --splits work/splits.json --output work/dataset.pkl --smiles-column SMILES
python -m training.main.train --dataset work/dataset.pkl --unimol <unimol_lite_encoder_dir> --training-config training/main/configs/default.yaml --output runs/main --export

# 预测
smiles2class --model main --model-path runs/main --input-csv input.csv --output preds.csv
```

### 状态

流程与 CLI 有测试覆盖；测试使用合成数据，不需要模型权重。仓库没有 CI。

### 作者

- David Niteas `<david.ql.d@foxmail.com>`

见 [`AUTHORS`](./AUTHORS)。仓库包含什么见 [`NOTICE`](./NOTICE)。

### 许可证

MIT，见 [`LICENSE`](./LICENSE)。引用元数据见 [`CITATION.cff`](./CITATION.cff)。
