# CLI 与 FAQ

**中文** | [English](../en/cli-and-faq.md) · [返回主页](../../README.md#zh)

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

### 某行概率全为 NaN

SMILES 无法解析时，该行 `status="invalid"`，426 个概率全为 `NaN`，其它行不受影响。

### `degraded_2d` 的使用

主模型按 3D 构象训练，2D 属于分布外输入。下游应按 `status` 过滤，或对这类行降权。

### 重复 SMILES

重复行不合并，各自独立特征化、独立推理，输出行数与输入一致。

### 副模型的推理耗时

副模型包含 426 个 AutoGluon 集成，默认逐个推理；耗时取决于硬件、模型目录与缓存状态。模型目录宜放在 SSD 上，并可在服务启动时预热。

### GPU 支持

主模型支持 `device="cuda"`；副模型是 AutoGluon 表格模型，使用 CPU。

### 概率归一化

426 个输出是独立二分类概率，跨类不归一；归一化会改变概率语义。

## 开发与测试

```shell
pip install -e ".[dev,vice]"
pytest tests/ -q
```

测试通过 `S2C_MODEL_DIR` 指定模型目录；未设置时跳过依赖模型的测试。
