# 快速开始

**中文** | [English](../en/02-quickstart.md) · [返回主页](../../README.md#zh-cn)

**层级：** 1 — 开始

**前置条件：** [安装](./01-installation.md)。

## 模型目录

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录（仅 vice 需要）
```

模型文件不随仓库分发，需要按[训练](04-training.md)自行训练生成。

## 最小示例：主模型

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(
    ["CCO", "CC(=O)Oc1ccccc1C(=O)O", "not-a-smiles"],
    return_meta=True,
)
print(probs.shape)                     # (3, 426)
print([m["status"] for m in meta])     # ['ok', 'ok', 'invalid']
print(probs.iloc[2].isna().all())      # True
```

## 最小示例：副模型

```python
from smiles2class import ViceClassifier

clf = ViceClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO"], return_meta=True)
print(probs.shape)                     # (1, 426)
```

副模型会先跑主模型拿 embedding，再走 426 个 predictor；首次预测可能较慢。

## 命令行

```shell
smiles2class \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv
```

输出 CSV 包含 `smiles`、`status` 与 426 个概率列。

## 下一步

- 查参数与返回结构：[API](06-api.md)
- 了解模型与概率语义：[模型说明](05-models.md)
- 命令行与常见问题：[CLI 与 FAQ](07-cli-and-faq.md)
- 需要 agent 代跑安装、数据准备与训练：[AGENTS.md](../../AGENTS.md)

**下一页：** [数据管线](./03-data-pipeline.md)
