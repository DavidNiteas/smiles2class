# 评测

**中文** | [English](../en/08-evaluation.md) · [返回主页](../../README.md#zh-cn)

**层级：** 3 — 参考

**前置条件：** [训练](./04-training.md)。

`evaluation/` 提供逐类评测：把真值 multi-hot 矩阵与预测概率对齐，输出每类指标和整体汇总。评测代码不包含在 wheel 中，需要额外安装依赖：

```bash
pip install -e ".[eval]"      # scikit-learn，用于 AP / ROC AUC
```

## 指标口径

每个类别视为一个独立的二分类问题（一对多），逐类计算：

| 指标 | 含义 | 是否依赖阈值 |
|---|---|---|
| precision | 预测为正的样本中真正为正的比例 | 是 |
| recall | 真正为正的样本中被预测为正的比例 | 是 |
| F1 | precision 与 recall 的调和平均 | 是 |
| average precision（AP） | precision-recall 曲线下的面积 | 否 |
| ROC AUC | 正负样本按分数排序的可分性 | 否 |
| balanced accuracy | (recall + specificity) / 2 | 是 |

约定：

- **阈值**：默认 `0.5`，可用 `--threshold` 修改。阈值只影响 precision / recall / F1 / balanced accuracy。
- **AUROC 与 AP 使用连续分数（概率）**：直接把每类的预测概率交给指标函数，不使用 `argmax` 或阈值化后的 0/1 作为分数。阈值化的 0/1 会丢失排序信息，使 AUROC 失去意义。
- **多标签**：真值为 multi-hot，一个分子可以同时属于多个类；各类互不影响。
- **无效分数**：预测中的 `NaN`（如非法 SMILES 没有输出）按类逐列忽略，不参与该类的统计。

### 零正例与零负例

- **零正例类**（该类在真值中从未出现）：precision / recall / F1 / AP / ROC AUC / balanced accuracy 均无定义，逐类表中记 `NaN`；
- **零负例类**（该类在真值中全部为正）：AP / ROC AUC / balanced accuracy 无定义记 `NaN`；precision / recall / F1 仍按阈值计算；
- 汇总中的 `zero_positive` / `zero_negative` 给出这两类退化类别的数量，`excluded` 给出每个指标因 `NaN` 被排除的类数；
- **macro** 平均只对非 `NaN` 的类取平均，不把无定义类当作 0 计入；若某指标所有类都无定义，macro 值为 `null`；
- **micro** 把各类的混淆矩阵（tp / fp / fn / tn）先相加，再按展平后的矩阵计算 P/R/F1 与 balanced accuracy；micro AP / ROC AUC 在展平后的全部 `(样本, 类别)` 分数上计算。

## 两种用法

### 1. 已有预测

输入表含 SMILES 与真值标签列，预测表含相同的标签列；两张表行顺序必须一致。

```bash
python -m evaluation.evaluate \
  --input labels.csv \
  --predictions preds.csv \
  --output report
```

### 2. 直接跑模型

不给 `--predictions` 时，用模型目录做推理。`--model` 可选 `main`（默认）或 `vice`。

```bash
python -m evaluation.evaluate \
  --input labels.csv \
  --model main \
  --model-path /path/to/model_dir \
  --output report
```

其他参数：

- `--threshold`：判定阈值，默认 `0.5`；
- `--smiles-column`：SMILES 列名，默认 `smiles`；若输入是 `pipelines.hmdb.labels` 产出的 `labels.parquet`，列名为 `SMILES`，需显式传 `--smiles-column SMILES`；
- `--label-map`：可选 JSON。给列表时按位置命名 `label_0..N` 列（或匹配同名标签列）；给对象时按 `{列名: 显示名}` 重命名。不提供时，优先使用 `label_0..N` 形式的列，否则使用除 SMILES 与 `status` / `id` 之外的全部列。

### 标签列约定

真值标签列的列名可以是：

- 类名（与模型 `label_map.json` 中的名字一致）；
- `label_0`, `label_1`, …（按位置对应模型输出）；
- 任意列名，配合 `--label-map` 映射为可读类名。

## 报告

`--output` 是前缀，生成两个文件（不会写模型或数据副产物）：

- `<prefix>_per_class.csv`：每类一行，字段为
  `label, n_pos, n_neg, n_pred_pos, tp, fp, fn, tn, precision, recall, f1, average_precision, roc_auc, balanced_accuracy`；
- `<prefix>_summary.json`：整体汇总，结构如下：

```json
{
  "threshold": 0.5,
  "n_samples": 4,
  "n_classes": 2,
  "zero_positive": 0,
  "zero_negative": 0,
  "macro": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "average_precision": 1.0,
    "roc_auc": 1.0,
    "balanced_accuracy": 1.0
  },
  "excluded": {
    "precision": 0,
    "recall": 0,
    "f1": 0,
    "average_precision": 0,
    "roc_auc": 0,
    "balanced_accuracy": 0
  },
  "micro": {
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0,
    "average_precision": 1.0,
    "roc_auc": 1.0,
    "balanced_accuracy": 1.0
  }
}
```

## 快测与慢测

评测相关的快测不需要模型或数据：

```bash
PYTHONPATH=src:. pytest tests/evaluation/ -q
```

它们覆盖指标口径（完美预测、零正例、阈值与 AP/AUROC 的关系）以及基于合成 CSV 的 CLI 输出。

全量测试与需要真实模型/数据的慢测：

```bash
PYTHONPATH=src:. pytest tests/ -q                 # 快测
PYTHONPATH=src:. pytest tests/ -q -m slow         # 只跑慢测
```

快测使用合成数据，可在 CI 中运行；完整模型评测需要本地模型与真值表，应离线运行，产物写到 `evaluation/reports/`（已 gitignore）。

**下一页：** [为什么只发代码](./09-why-code-only.md)
