# CLI 与 FAQ

**中文** | [English](../en/cli-and-faq.md)

## CLI

```shell
smiles2class \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv \
  --smiles-column smiles \
  --workers 8 \
  --timeout 30 \
  --device cpu \
  --batch-size 64
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `--model` | `main` | `main` / `vice` |
| `--model-path`, `-m` | 必填 | 模型目录 |
| `--input-csv`, `-i` | 必填 | 输入 CSV |
| `--output`, `-o` | 必填 | 输出 CSV |
| `--smiles-column`, `-s` | `smiles` | SMILES 列名 |
| `--workers` | `8` | 特征化线程数 |
| `--timeout` | `30.0` | 每构象墙钟预算 |
| `--device` | `cpu` | `cpu` / `cuda` |
| `--batch-size` | `64` | 主模型前向批大小 |
| `--vice-workers` | `1` | 副模型 predictor 并行度 |

输出 CSV：`smiles`、`status`、426 个概率列；行数与输入一致。

## FAQ

### 为什么某一行是 NaN？

SMILES 无法解析。`status="invalid"`，该行概率全为 `NaN`，其它行不受影响。

### `degraded_2d` 能用吗？

需要谨慎。主模型按 3D 构象训练；2D 是分布外输入。可以按 `status` 过滤，或在下游降权。

### 重复 SMILES 会合并吗？

不会。每行独立特征化、独立推理；重复行输出两行。

### 副模型为什么这么慢？

426 个 AutoGluon 集成逐个推理，单次预测可能较慢，取决于硬件、模型目录与缓存状态。建议把模型目录放在 SSD 上，或在服务启动时预热。

### 能用 GPU 吗？

主模型支持 `device="cuda"`；副模型是 AutoGluon 表格模型，仍走 CPU。

### 概率要归一化吗？

不要。426 个输出是独立二分类概率，跨类不归一。任何“归一化”都会改变语义。

## 开发与测试

```shell
pip install -e ".[dev,vice]"
pytest tests/ -q
```

测试用 `S2C_MODEL_DIR` 指定模型目录；未设置时默认 `<repo>/../model/uni_classify`，不存在则自动跳过。
