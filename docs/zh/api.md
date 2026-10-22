# API

**中文** | [English](../en/api.md) · [返回主页](../../README.md#zh)

## MainClassifier

```python
from smiles2class import MainClassifier

clf = MainClassifier.from_pretrained(model_dir, device="cpu")
probs, metas = clf.predict(
    smiles,
    workers=8,
    timeout=30.0,
    coord_mode="auto",
    batch_size=64,
    return_meta=True,
)
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_dir` | — | 含 `label_map.json` 与 `main_model/` 的目录 |
| `device` | `"cpu"` | `"cpu"` 或 `"cuda"` |
| `workers` | `8` | 特征化线程数 |
| `timeout` | `30.0` | 每个构象的墙钟预算（秒）；`None` 关闭 |
| `coord_mode` | `"auto"` | `auto` / `3d` / `2d` |
| `batch_size` | `64` | 主模型前向批大小 |
| `return_meta` | `False` | 是否同时返回每行的状态 |

返回：

- `return_meta=False`：`pandas.DataFrame`，形状 `(N, 426)`，列 = 类名；
- `return_meta=True`：`(DataFrame, list[dict])`，`meta` 与输入等长保序。

## ViceClassifier

```python
from smiles2class import ViceClassifier

clf = ViceClassifier.from_pretrained(model_dir, device="cpu")
probs, metas = clf.predict(
    smiles,
    workers=8,
    timeout=30.0,
    coord_mode="auto",
    batch_size=64,
    vice_workers=1,
    return_meta=True,
)
```

- 先调用主模型拿到 `(N, 512)` embedding，再走 426 个 predictor；
- `vice_workers` 默认 `1`（顺序执行）；更大的值不保证更快；
- 输出同样是 `(N, 426)`，逐类概率。

## 行对齐与边界

| 输入 | 行为 |
|---|---|
| 合法 SMILES | 正常概率 |
| 重复 SMILES | 保留两行，不折叠 |
| 非法 SMILES | 该行全 `NaN`，`status="invalid"` |
| 空列表 | 返回 `(0, 426)` 与空 `meta` |
| 2D 降级 | 正常概率，`status="degraded_2d"` |
| 超时且 `on_timeout="error"` | 该行 `None` / `NaN`，`status="timeout_error"` |

## `meta` 状态模型

| `status` | 含义 |
|---|---|
| `ok` | 正常（3D 或显式 2D） |
| `degraded_2d` | 3D 失败或超时，已降级 2D |
| `timeout_error` | 超时，且 `on_timeout="error"` |
| `invalid` | SMILES 无法解析 |
| `skipped` | 超时，且 `on_timeout="skip"` |
| `error` | 批量层捕获的未预期异常 |

`meta` 另含 `coord_mode`、`reason`、`elapsed_s`。

> `degraded_2d` 属于训练分布外输入：主模型使用 3D 构象训练。下游应按 `status` 过滤或降权。

## 标签

```python
from smiles2class import LABELS, NUM_LABELS

assert NUM_LABELS == 426
```

`LABELS` 的顺序与模型输出列严格对应，**不得重排**。
