# 为什么只发代码

**中文** | [English](../en/09-why-code-only.md) · [返回主页](../../README.md#zh-cn)

**层级：** 0 — 范围

**前置条件：** 无。

本仓库只包含代码、配置和文档，不包含训练数据，也不包含预训练模型或模型权重。

## 数据

426 个标签来自 HMDB `Alternative Parents`（ClassyFire 术语）。HMDB 数据受其再分发条款限制，仓库不能转存原始表或派生表。复现时需要从 [HMDB 下载页](https://hmdb.ca/downloads) 按条款获取数据；LIPID MAPS 是可选补充，见[数据管线](./03-data-pipeline.md)。

## 模型

训练好的模型目录（`label_map.json` + `main_model/`）不随仓库分发。模型体积、训练成本和数据版本都由用户自己的数据决定；在本机重训比传递一个固定 bundle 更可控。训练流程见[训练](./04-training.md)。

## 只想要结果

先准备数据并训练模型，再用 CLI 预测：

```shell
smiles2class --model main --model-path <model_dir> --input-csv input.csv --output preds.csv
```

没有模型目录时，从[快速开始](./02-quickstart.md)进入，按[数据管线](./03-data-pipeline.md)与[训练](./04-training.md)完成训练。

**下一页：** [安装](./01-installation.md)
