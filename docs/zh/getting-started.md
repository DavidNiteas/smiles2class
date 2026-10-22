# 快速开始

**中文** | [English](../en/getting-started.md) · [返回主页](../../README.md#zh)

## 环境要求

- Python 3.10 – 3.11；
- `rdkit>=2025.9,<2026`、`torch>=2.0`、`transformers>=4.47,<5`、`numpy<2`；
- 编码器由 `unimol-lite`（导入名 `unimol_lite`）提供；它是 Uni-Mol 的轻量修改版实现，不是官方 Uni-Mol；
- 副模型额外需要 `autogluon.tabular==1.2.0` 与 `scikit-learn==1.5.1`。

## 安装

```shell
# 只使用主模型
pip install "git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"

# 同时使用副模型
pip install "smiles2class[vice] @ git+https://github.com/DavidNiteas/smiles2class.git@v0.1.1"
```

> 已配置 SSH 密钥时，也可使用
> `git+ssh://git@github.com/DavidNiteas/smiles2class.git@v0.1.1`。
> `unimol-lite` 依赖固定为 `v0.2.1`（Uni-Mol 的轻量修改版实现，不是官方 Uni-Mol）。

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

## 下一步

- 查参数与返回结构：[API](api.md)
- 了解模型与概率语义：[模型说明](models.md)
- 命令行与常见问题：[CLI 与 FAQ](cli-and-faq.md)
- 需要 agent 代跑安装、数据准备与训练：[agent.md](../../agent.md)
