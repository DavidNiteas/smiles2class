# 训练

**中文** | [English](../en/training.md)

训练代码在 `training/`，**不随 wheel 安装**；从仓库根目录运行。训练产物符合推理侧的模型目录契约。

## 安装

```shell
pip install -e ".[train,vice]"
```

`[train]` 提供 PyTorch-Lightning、torchmetrics 等；副模型训练还需要 `[vice]`（AutoGluon）。

## 主模型

### 1. 准备数据

由 `pipelines/` 产出 `labels.parquet` 与 `splits.json` 后：

```shell
python -m training.main.prepare \
  --input work/labels.parquet \
  --splits work/splits.json \
  --output work/dataset.pkl \
  --workers 8
```

`prepare` 用 `unimol_lite` 把每条 SMILES 转成四个张量，并按 split 组装成可训练的 DataModule。

### 2. 训练

```shell
python -m training.main.train \
  --dataset work/dataset.pkl \
  --unimol base_unimol \
  --training-config training/main/configs/default.yaml \
  --output runs/main \
  --export
```

- `--unimol` 指向 Uni-Mol encoder 目录；
- `--export` 写出：

```text
runs/main/
├── main_model/{config.json, model.safetensors}
└── label_map.json
```

- 训练超参（batch size、lr、epochs、precision、accelerator 等）在 `training/main/configs/default.yaml`；
- 调试可用 `--max-steps` / `--max-epochs` 限制步数。

### 3. 使用

```python
from smiles2class import MainClassifier
clf = MainClassifier.from_pretrained("runs/main")
```

## 副模型

### 1. 生成 embedding

```shell
python -m training.vice.prepare \
  --input work/labels.parquet \
  --model-dir runs/main \
  --output work/vice.parquet
```

- 输入表可带 `cluster_id`；没有时填 0（聚类分层仅影响多数类下采样，不影响能否训练）；
- 输出：`vec_0..vec_511`、`label_*`、`cluster_id`。

### 2. 逐类训练

```shell
python -m training.vice.train \
  --dataset work/vice.parquet \
  --output runs/main/vice_model \
  --label-map runs/main/label_map.json
```

- 每个类一个 `TabularPredictor`，写入 `runs/main/vice_model/<i>/`；
- 训练参数（presets、time limit、CPU 数、切分比例）在 `training/vice/configs/default.yaml`；
- 调试可用 `--max-classes` / `--time-limit`；
- 子集训练时，`<output>/label_map.json` 只包含实际训练的标签；全量训练请使用主模型产出的模型根 `label_map.json`。

### 3. 使用

```python
from smiles2class import ViceClassifier
clf = ViceClassifier.from_pretrained("runs/main")
```

## 模型目录契约

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录
```

## 复现与限制

- 训练代码不进入 wheel；`pip install smiles2class` 不会引入训练栈；
- 训练需要自备数据；仓库不附带原始/派生数据；
- 所有入口支持固定 seed；完整训练不在 CI 中运行；
- 主模型可 CPU/GPU；副模型是 AutoGluon 表格模型，走 CPU；
- 426 个副模型 predictor 的训练是逐类进行的，耗时与类别数、数据量成正比。
