# pipelines · 数据管线

`pipelines/` 是 smiles2class 的**仓库内**数据准备工具，把 HMDB / LIPID MAPS 原始表处理成：

1. 清洗（可选合并）后的化合物表；
2. 与 `src/smiles2class/labels.json` 逐项一致的 426 标签 multi-hot 表；
3. train / vali / test 划分。

## 定位

- **不进 wheel**：`pyproject.toml` 的 wheel 只打包 `src/smiles2class`，本目录随 sdist 分发；
- **不附带数据**：仓库不含任何原始或派生数据表；
- **全部参数化**：路径、列名、比例、seed 均来自命令行或 `pipelines/hmdb/configs/default.yaml`，代码不写死机器路径。

## 目录

```text
pipelines/
├── __init__.py                # 配置加载 / 表格读写等公共助手
├── hmdb/
│   ├── clean.py               # HMDB 清洗 + LIPID MAPS 转换 + 合并
│   ├── labels.py              # Alternative Parents -> 426 标签 multi-hot
│   ├── split.py               # 多标签 greedy 20/20 划分
│   └── configs/default.yaml   # 列名、ID 候选、比例、seed
└── README.md
```

## 依赖

- `pandas`；
- `.parquet` 读写需要 `pyarrow`；
- 读取 YAML 配置需要 `PyYAML`（`pip install -e ".[data]"` 已包含）。

## 快速开始

```shell
# 1. 清洗 + 合并（合并交付表默认丢弃 Alternative Parents）
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --lipidmaps-input raw/lipidmaps.csv \
  --output work/compounds.parquet

# 训练用表：保留 Alternative Parents
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --output work/hmdb_train.parquet \
  --keep-alternative-parents

# 2. 426 标签 multi-hot（同时写 label_map.json 与 labels.parquet）
python -m pipelines.hmdb.labels \
  --input work/hmdb_train.parquet \
  --output work/label_map.json \
  --labels-output work/labels.parquet

# 3. 多标签划分
python -m pipelines.hmdb.split \
  --input work/labels.parquet \
  --output work/splits.json \
  --stats-output work/split_stats.json
```

所有命令都支持 `--help`；`--config your.yaml` 会深度合并到默认配置之上。

## 行为要点

- `clean` 不带 `--lipidmaps-input` 时只清洗 HMDB（保留全部行，含空 SMILES）；
- LIPID MAPS 转换会丢弃 SMILES 为空的行；
- `labels` 的 `--output` 是 `label_map.json`，multi-hot 表默认写到同目录的 `labels.parquet`；
- `split` 的 `--output` 固定为 `{"train": [...], "vali": [...], "test": [...]}`，每标签覆盖统计走 `--stats-output`。

详细输入 schema / 输出契约见 [`docs/zh/data-pipeline.md`](../docs/zh/data-pipeline.md)。
