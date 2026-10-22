# smiles2class 文档（简体中文）

**语言：** [English](../en/index.md) · 简体中文

`smiles2class` 把 SMILES 转成 426 个分子类别的概率。它提供 `main` 与 `vice` 两种独立实现，以及完整的数据、训练与评测流程。仓库只发代码：不附带 HMDB / LIPID MAPS 数据，也不附带训练好的模型权重。

## 从哪开始

| 你想做什么 | 阅读顺序 |
|---|---|
| 理解为什么不发布数据/模型 | [为什么只发代码](./09-why-code-only.md) |
| 安装并跑通一个小例子 | [安装](./01-installation.md) → [快速开始](./02-quickstart.md) |
| 使用自己的数据 | [数据管线](./03-data-pipeline.md) → [训练](./04-training.md) |
| 理解模型与概率语义 | [模型说明](./05-models.md) |
| 查 Python API 或 CLI | [API](./06-api.md) · [CLI 与 FAQ](./07-cli-and-faq.md) |
| 评估模型质量 | [评测](./08-evaluation.md) |

## 怎么读这些文档

- **第一次用：** [安装](./01-installation.md)，然后跑 [快速开始](./02-quickstart.md)。
- **需要模型？** 仓库没有预训练权重。先准备数据，再[训练](./04-training.md)。
- **已经训练出模型？** 用 [API](./06-api.md) 或 [CLI](./07-cli-and-faq.md) 预测。
- **要判断指标可不可信：** 读[评测](./08-evaluation.md)。

## 你需要自备

- HMDB / LIPID MAPS 原始数据表：从官方页面按条款下载，见[数据管线](./03-data-pipeline.md)。
- 训练好的模型目录：本仓库不附带权重，按[训练](./04-training.md)生成。

作者见 [`AUTHORS`](../../AUTHORS)，仓库包含什么见 [`NOTICE`](../../NOTICE)。

## 作者与许可

- David Niteas `<david.ql.d@foxmail.com>`
- 许可证：MIT（仅代码）

**下一页：** [安装](./01-installation.md)
