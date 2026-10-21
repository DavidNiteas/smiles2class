# 快速开始

**中文** | [English](../en/getting-started.md)

## 环境要求

- Python 3.10 – 3.11；
- `rdkit>=2025.9,<2026`、`torch>=2.0`、`transformers>=4.47,<5`、`numpy<2`；
- 副模型额外需要 `autogluon.tabular==1.2.0` 与 `scikit-learn==1.5.1`。

## 安装

```shell
# 只使用主模型
pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"

# 同时使用副模型
pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

> 维护者可将 HTTPS 地址替换为
> `git+ssh://git@github.com/DavidNiteas/smiles2class.git@v0.1.1`。
> 依赖 `unimol-lite` 从其公开 HTTPS 地址的 `v0.2.1` 安装。

开发模式：

```shell
git clone https://github.com/DavidNiteas/smiles2class.git
cd smiles2class
pip install -e ".[dev,vice]"
pytest tests/ -q
```

## 模型目录

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录（仅 vice 需要）
```

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
