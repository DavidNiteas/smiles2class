# smiles2class

[中文](#zh) | [English](#en)

<a id="zh"></a>

## 中文

SMILES → 426 个分子类别的概率。提供两种独立实现：

- **main**：UniMol-Lite encoder + 426 个类查询解码器，输出 `(N, 426)` 概率与 512 维 embedding；
- **vice**：426 个 AutoGluon 逐类分类器，输入为主模型的 embedding，输出 `(N, 426)` 概率。

> **关于 Uni-Mol 实现与依赖**
> 编码器来自配套库 **UniMol-Lite**（依赖名 `unimol-lite`，导入名 `unimol_lite`）。UniMol-Lite 是 Uni-Mol 的轻量修改版实现，**不是** Uni-Mol 官方版本。
> 本项目的依赖是 `unimol-lite@v0.2.1`，不会引入官方 Uni-Mol 实现。需要官方实现、权重或支持时，请使用 [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol)。

<a id="zh-results"></a>

### 只想要结果？

本项目不提供预训练模型。要得到预测，需要先用自有数据训练模型：

1. 按下一节完成基础安装；训练还需要 `.[train,vice]`，见[训练](docs/zh/training.md)。
2. 按[数据管线](docs/zh/data-pipeline.md)获取 HMDB / LIPID MAPS 数据，完成清洗、426 标签与 train / vali / test 划分。
3. 按[训练](docs/zh/training.md)训练主模型（可选副模型），得到模型目录（`label_map.json` + `main_model/`）。
4. 准备 `input.csv`，至少包含一列 `smiles`。
5. 运行：

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

6. `preds.csv` 包含 `smiles`、`status` 与 426 个概率列。

已经按上述流程训练出模型目录时，可直接从第 4 步开始。希望 agent 代跑？把 [agent.md](agent.md) 交给 agent；它会先与你确认数据来源、许可和存放路径，再执行安装、数据准备、训练与预测。

### 安装

```shell
pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
# 需要副模型时：
pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

安装时会一并安装 `unimol-lite@v0.2.1`。本项目不提供预训练模型；`from_pretrained(model_dir)` 需要指向用户自己训练得到的模型目录。

### 快速开始

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO", "c1ccccc1"], return_meta=True)
print(probs.shape)  # (2, 426)
```

- 输出与输入等长保序，重复 SMILES 不折叠；
- 非法 SMILES 的行 `status="invalid"`，概率为 `NaN`；
- 2D 降级的行 `status="degraded_2d"`。

模型目录：

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录（仅 vice 需要）
```

命令行入口：

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

### 文档导航

主页只保留安装与最小示例；参数、训练、评测等细节在对应页面展开。建议从“入门”开始：

| 阶段 | 页面 | 内容 |
|---|---|---|
| 入门 | [快速开始](docs/zh/getting-started.md) | 环境、安装、模型目录、最小示例、CLI |
| 使用 | [API](docs/zh/api.md) | `MainClassifier` / `ViceClassifier`、参数、返回结构、状态模型 |
| 使用 | [模型说明](docs/zh/models.md) | 主/副模型结构、标签空间、概率语义、运行时限制 |
| 使用 | [CLI 与 FAQ](docs/zh/cli-and-faq.md) | 命令行、常见问题、开发测试 |
| 进阶 | [训练](docs/zh/training.md) | 主/副模型训练、模型目录契约 |
| 进阶 | [数据管线](docs/zh/data-pipeline.md) | HMDB / LIPID MAPS 清洗、426 标签 multi-hot、划分 |
| 进阶 | [评测](docs/zh/evaluation.md) | 逐类指标、AUROC/AP、报告字段、快慢测 |
| 快捷 | [只想要结果？](#zh-results) | 从数据准备到训练、预测的最短路径 |
| Agent | [agent.md](agent.md) | 从安装到数据准备、训练或推理的 agent 流程 |

### 数据

HMDB 数据受再分发条款限制，本仓库不提供原始或派生数据；复现数据管线或训练时需自行前往 HMDB 官网按条款下载。

### 作者与许可

作者：David Niteas

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

<a id="en"></a>

## English

SMILES → probabilities for 426 molecular classes. Two independent implementations:

- **main**: a UniMol-Lite encoder plus 426 class-query decoders, returning `(N, 426)` probabilities and a 512-dimensional embedding;
- **vice**: 426 per-class AutoGluon classifiers over the main model's embedding, returning `(N, 426)` probabilities.

> **Uni-Mol implementation and dependency**
> The encoder comes from the companion library **UniMol-Lite** (dependency name `unimol-lite`, import name `unimol_lite`). UniMol-Lite is a lightweight, modified implementation of Uni-Mol; it is **not** the official Uni-Mol release.
> The `smiles2class` dependency is pinned to `unimol-lite@v0.2.1` and does not install or invoke the official Uni-Mol implementation. For the official implementation, weights and support, see [deepmodeling/Uni-Mol](https://github.com/deepmodeling/Uni-Mol).

<a id="en-results"></a>

### Just want results?

This project does not provide pretrained models. To get predictions, train a model on your own data first:

1. Complete the base installation in the next section; training also needs `.[train,vice]`, as described in [Training](docs/en/training.md).
2. Obtain HMDB / LIPID MAPS data and run cleaning, 426-label generation and train / vali / test splitting as described in the [data pipeline](docs/en/data-pipeline.md).
3. Train the main model (and optionally the vice model) following [Training](docs/en/training.md) to produce a model directory (`label_map.json` + `main_model/`).
4. Prepare `input.csv` with at least a `smiles` column.
5. Run:

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

6. `preds.csv` contains `smiles`, `status` and the 426 probability columns.

If you have already trained a model directory with the steps above, start from step 4. Want an agent to run the flow? Hand [agent.md](agent.md) to the agent. It will first confirm the data source, license and storage path with you, then install, prepare data, train and predict.

### Installation

```shell
pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
# With the vice model:
pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

This also installs `unimol-lite@v0.2.1`. This project does not provide pretrained models; `from_pretrained(model_dir)` must point at a model directory trained by the user.

### Quick start

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO", "c1ccccc1"], return_meta=True)
print(probs.shape)  # (2, 426)
```

- Output rows match input rows in length and order; duplicate SMILES are kept;
- Rows with invalid SMILES have `status="invalid"` and `NaN` probabilities;
- Rows degraded to 2D have `status="degraded_2d"`.

Model directory:

```text
<model_dir>/
├── label_map.json          # 426 class names; the order is part of the model
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 AutoGluon predictor directories (vice only)
```

Command line:

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

### Documentation

The homepage keeps only installation and a minimal example; parameters, training and evaluation are expanded in the linked pages. Start with “Getting started”:

| Level | Page | Contents |
|---|---|---|
| Getting started | [Getting started](docs/en/getting-started.md) | Requirements, installation, model directory, minimal examples, CLI |
| Usage | [API](docs/en/api.md) | `MainClassifier` / `ViceClassifier`, parameters, return values, status model |
| Usage | [Models](docs/en/models.md) | Main/vice architecture, label space, probability semantics, runtime limits |
| Usage | [CLI and FAQ](docs/en/cli-and-faq.md) | Command line, common questions, development and tests |
| Advanced | [Training](docs/en/training.md) | Main/vice training, model-directory contract |
| Advanced | [Data pipeline](docs/en/data-pipeline.md) | HMDB / LIPID MAPS cleaning, 426-label multi-hot, splits |
| Advanced | [Evaluation](docs/en/evaluation.md) | Per-class metrics, AUROC/AP, report fields, fast/slow tests |
| Shortcut | [Just want results?](#en-results) | Shortest path from data preparation to training and prediction |
| Agent | [agent.md](agent.md) | Agent flow from installation to data preparation, training or inference |

### Data

HMDB data is subject to its redistribution terms. This repository provides no raw or derived HMDB data; reproducing the data pipeline or training requires downloading the data from the official HMDB website under its terms.

### Author and license

Author: David Niteas

This project is released under the MIT License. See [LICENSE](LICENSE).
