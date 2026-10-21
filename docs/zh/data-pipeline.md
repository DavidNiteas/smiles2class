# 数据管线

**中文** | [English](../en/data-pipeline.md)

`pipelines/` 是仓库内的数据准备工具：把 HMDB / LIPID MAPS 原始表处理成清洗表、426 标签 multi-hot 表与 train / vali / test 划分。它**不随 wheel 安装**（只随 sdist 分发），仓库也**不附带任何原始或派生数据**——所有输入由调用方提供。

## 依赖

- `pandas`；
- `.parquet` 读写需要 `pyarrow`；
- 读取 YAML 配置需要 `PyYAML`（`pip install -e ".[data]"` 已包含）。

## 输入 schema

### HMDB 原始表（`.csv` / `.parquet`）

| 列 | 必需 | 含义 |
|---|---|---|
| `HMDB ID` | 是 | 分子 ID；清洗后按它排序 |
| `SMILES` | 是 | 结构；原样复制到 `CANON_SMILES` |
| `IUPAC Name` | 合并时必需 | 统一表的 `Name` |
| `Kingdom` / `Super Class` / `Class` / `Sub Class` | 合并时必需 | 分类字段 |
| `Alternative Parents` | 标签步骤必需 | ClassyFire 父类列表 |

### LIPID MAPS 原始表（可选）

`LM_ID`、`NAME`、`SMILES`、`CATEGORY`、`MAIN_CLASS`、`SUB_CLASS`、`CLASS_LEVEL4`；`SMILES` 为空的行会被丢弃。

### 标签表输入（`pipelines.hmdb.labels`）

需要 `SMILES` 与 `Alternative Parents` 两列。ID 列按顺序取第一个存在的候选：`id` → `ID` → `HMDB ID`；都不存在时使用行号，并给出 warning。

`Alternative Parents` 单元格支持三种写法：

```text
['Organic oxides', 'Carbonyl compounds']      # Python list 字符串
["Organic oxides", "Carbonyl compounds"]      # JSON 数组
Organic oxides; Carbonyl compounds            # ; 或 | 分隔
```

裸逗号也接受，但仅当拆分后的每一段都是已知标签时才生效（部分 ClassyFire 名称本身含逗号，如 `1,2-diols`）。

列名映射写在 `pipelines/hmdb/configs/default.yaml` 的 `columns` 段；自定义表结构时复制该文件并用 `--config` 覆盖即可。

## 三个命令

```shell
# 1. 清洗 + 合并
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --lipidmaps-input raw/lipidmaps.csv \
  --output work/compounds.parquet

# 训练用表：保留 Alternative Parents
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --output work/hmdb_train.parquet \
  --keep-alternative-parents

# 2. 426 标签 multi-hot
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

- 输出格式由后缀决定（`.csv` / `.parquet`）；
- 不带 `--lipidmaps-input` 时 `clean` 只清洗 HMDB，保留全部行（包括空 SMILES）；
- 合并交付表默认丢弃 `Alternative Parents`，训练表加 `--keep-alternative-parents` 保留。

## 输出契约

### `clean`

不带 LIPID MAPS 时：原始 HMDB 列 + `CANON_SMILES`，按 `HMDB ID` 升序，`Alternative Parents` 默认删除。

带 LIPID MAPS 时，统一为：

| 列 | 来源 |
|---|---|
| `ID` | `HMDB ID` / `LM_ID` |
| `Name` | `IUPAC Name` / `NAME` |
| `SMILES` | 原样 |
| `Kingdom` | `Kingdom` / `CATEGORY` |
| `Super Class` | `Super Class` / `MAIN_CLASS` |
| `Class` | `Class` / `SUB_CLASS` |
| `Sub Class` | `Sub Class` / `CLASS_LEVEL4` |

HMDB 行在前，LIPID MAPS 行在后，不去重；加 `--keep-alternative-parents` 时末尾多一列 `Alternative Parents`（LIPID MAPS 行为空）。

### `labels`

- `label_map.json`：426 个类名，**与权威文件逐项一致**（含顺序）；
- multi-hot 表（默认 `labels.parquet`）：`id`、`SMILES`、`label_0` … `label_425`，值为 0/1。

如果某个标签在输入中从未出现，会给出 warning（不静默丢弃）；`label_map.json` 本身始终是完整词表。

### `split`

`splits.json` 只有三个键：

```json
{"train": ["id..."], "vali": ["id..."], "test": ["id..."]}
```

`--stats-output` 另存统计：各集合大小与占比、参数（seed / 比例）、每标签覆盖、无样本标签、未达 20% 目标的标签、无标签行数。

划分算法（按标签覆盖的贪心划分）：

1. 按 `label_0 … label_425` 顺序遍历标签；
2. 该标签尚未分配的分子优先补 test，直到该标签的 test 覆盖达到 20%；
3. 再补 vali，直到达到 20%；
4. 其余进 train；一个分子只进一个集合；
5. 没有被任何标签覆盖的分子统一进 train，保证三集合构成完整划分。

算法本身无随机性，结果确定；`--seed` 仅记录在统计中（为后续引入随机留位）。

## 标签权威

426 个类名及其顺序来自 `src/smiles2class/labels.json`，与模型权重和 `label_map.json` 严格对齐。**不要修改顺序或增删类名**——它属于模型语义的一部分。`pipelines.hmdb.labels` 只读取该文件，不重新推导标签。

## 配置

默认配置：`pipelines/hmdb/configs/default.yaml`。`--config` 指定用户配置时，会深度合并到默认配置之上，只需写想覆盖的键。

| 段 | 内容 |
|---|---|
| `clean` / `labels` / `split` | 可选默认路径（仓库内留空，运行时由命令行提供） |
| `split.seed` / `test_ratio` / `vali_ratio` | 划分参数，默认 `42` / `0.2` / `0.2` |
| `labels.path` / `expected_count` | 标签权威路径（默认仓库内文件）与数量校验（426） |
| `columns` | HMDB / LIPID MAPS / 统一表 / 标签 / 划分的列名映射 |

也可以完全用命令行参数驱动，不修改配置文件。
