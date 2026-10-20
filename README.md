# smiles2class

SMILES → 分子类别概率。当前提供两种独立实现：

- **main**：主模型（UniMol encoder + 426 个类查询解码器），输出 `(N, 426)` 概率与 512 维 embedding；
- **vice**：副模型（426 个 AutoGluon 逐类二分类器，输入为主模型的 embedding），输出 `(N, 426)` 概率。

## 文档 / Documentation

- [中文文档](docs/zh/index.md) · [English documentation](docs/en/index.md)
- [完整索引](docs/README.md)

## 安装

```shell
pip install "git+ssh://git@github.com/MetaEngine-Dev/smiles2class.git@v0.1.0"
# 需要副模型时：
pip install "smiles2class[vice] @ git+ssh://git@github.com/MetaEngine-Dev/smiles2class.git@v0.1.0"
```

模型文件不随仓库分发，`from_pretrained(model_dir)` 需要指向本地模型目录（目录结构见下）。

## 模型目录

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录（仅 vice 需要）
```

## 使用

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO", "c1ccccc1"], return_meta=True)
```

```python
from smiles2class import ViceClassifier

clf = ViceClassifier.from_pretrained("<model_dir>")
probs, meta = clf.predict(["CCO", "c1ccccc1"], return_meta=True)
```

- 输出与输入**等长保序**；重复 SMILES 不折叠；
- 非法 SMILES 对应行 `status="invalid"`，概率为 `NaN`；
- 降级 2D 的行 `status="degraded_2d"`（特征化层给出）。

> 副模型的 426 个 predictor 默认顺序执行；`--vice-workers > 1` 可选，但在机械硬盘上实测无明显加速，且可能受 I/O 抖动影响。

## CLI

```shell
smiles2class predict \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv
```

## 开发文档

- 设计路线：`_dev/设计路线.md`
- 工单：`_dev/仓库建立/`、`_dev/主模型移植/`、`_dev/副模型移植/`

## 许可

MIT（代码）。模型文件与训练数据的许可另行确认。
