# smiles2class Agent 工作流

本文件面向 AI agent，也适用于希望按固定流程操作的使用者。目标：在不假设数据或模型已经存在的前提下，理解项目、完成安装、与用户确认数据获取方案，并把推理或训练跑起来。

本仓库不提供原始/派生数据，也不分发模型权重。agent 不得伪造数据、猜造标签或从非官方镜像下载数据。

## 0. 不可违反的约束

- **数据只从官方来源获取**：HMDB 与 LIPID MAPS。不要使用网盘、论坛附件或第三方镜像。
- **不要绕过访问控制**：HMDB 可能返回 Cloudflare 验证页。遇到 403 或验证页时，请用户在浏览器中下载，不要尝试绕过。
- **先确认再下载**：下载前与用户确认数据来源、许可条款、保存路径和数据规模。
- **不要提交数据或模型**：`data/`、`work/`、`runs/` 均视为本地工作目录。
- **先小样本验证**：全量训练前，先用足够产生非空 train split 的样本（通常至少几十行）跑通 `clean → labels → split → prepare`。只有 2–3 行时，split 可能把全部样本放进 test，导致 `prepare` 失败。
- **不要替换实现**：编码器依赖是 `unimol-lite`，不是官方 Uni-Mol；不要用官方 Uni-Mol 替换。
- **不要静默修复**：缺列、标签数不对、划分异常时停止并向用户报告，不要擅自改标签或补齐数据。

## 1. 项目速览

`smiles2class` 把 SMILES 转换为 426 个分子类别的独立概率。

- `main`：UniMol-Lite encoder + 426 个类查询解码器，输出 `(N, 426)` 概率与 512 维 embedding；
- `vice`：426 个 AutoGluon 逐类分类器，输入为主模型的 embedding，输出 `(N, 426)` 概率。

关键目录：

| 路径 | 作用 |
|---|---|
| `src/smiles2class/` | 推理库、CLI、模型与标签 |
| `pipelines/` | HMDB / LIPID MAPS 清洗、426 标签 multi-hot、划分 |
| `training/` | 主模型与副模型训练 |
| `evaluation/` | 逐类指标与评测报告 |
| `docs/zh/`、`docs/en/` | 深入文档 |
| `data/` | 用户放置原始数据（不提交） |
| `work/` | 清洗与训练中间产物（不提交） |
| `runs/` | 训练输出与模型目录（不提交） |

`main` 的编码器来自配套库 `unimol-lite`（导入名 `unimol_lite`），它是 Uni-Mol 的轻量修改版实现，不是官方 Uni-Mol。

## 2. 开工前必须问用户

先提问并等待回答，不要直接开始下载或训练：

1. 目标是**只要预测结果**，还是**复现/自己训练**？
2. 是否已有模型目录？目录中应有 `label_map.json` 与 `main_model/`（副模型还需要 `vice_model/`）。
3. 是否已有 HMDB / LIPID MAPS 原始数据？数据是什么版本、什么格式、放在哪里？
4. 是否接受从 HMDB 与 LIPID MAPS 官网下载，并遵守其许可与再分发条款？
5. 机器的 Python 版本、CPU/GPU、内存与磁盘空间是否满足训练需求？
6. 是否允许创建 `data/`、`work/`、`runs/` 并写入这些目录？

### 2.1 路径判定

| 用户情况 | 执行路径 |
|---|---|
| 已有模型目录，只要预测 | 跳到第 5 节“只推理” |
| 没有模型目录，只要预测 | 停止：说明模型权重不随仓库分发，请用户向作者获取；除非用户明确要求训练，否则不要自行训练 |
| 有数据，想训练/复现 | 第 3 节安装 → 第 4 节数据验收 → 第 6 节训练 |
| 没有数据，想训练/复现 | 第 3 节安装 → 第 4 节数据获取 → 第 6 节训练 |
| 只想先看看代码/文档 | 阅读 `README.md` 与 `docs/`，不执行下载或训练 |

## 3. 安装与自检

要求 Python 3.10 或 3.11。

```shell
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

# 只做推理
pip install -e .

# 需要副模型
pip install -e ".[vice]"

# 需要数据管线、训练、评测
pip install -e ".[data,train,vice,eval]"
```

自检：

```shell
python -c "import smiles2class; print(smiles2class.__version__)"
python -c "import unimol_lite; print('unimol_lite ok')"

# 安装了 data / train 依赖时再检查：
python -m pipelines.hmdb.clean --help
python -m training.main.prepare --help
```

若 `python -m pipelines...` 报 `ModuleNotFoundError`，确认当前目录是仓库根目录，或设置 `PYTHONPATH=src:.`。

## 4. 数据方案

### 4.1 数据需求

| 来源 | 官方页面 | 下载什么 | 用途 | 许可提示 |
|---|---|---|---|---|
| HMDB | <https://hmdb.ca/downloads> | 全量代谢物表（优先 CSV；导出名随版本变化） | 426 标签来自 `Alternative Parents`，训练/复现必需 | 按页面条款；不再分发原始或派生数据 |
| LIPID MAPS | <https://www.lipidmaps.org/databases/lmsd/download> | `LMSD Structure-data file (SDF)` | 可选，补充脂类结构 | SDF 标注为 CC BY 4.0 |

HMDB 表至少需要这些列（列名可在 `pipelines/hmdb/configs/default.yaml` 中映射）：

```text
HMDB ID, SMILES, IUPAC Name, Kingdom, Super Class, Class, Sub Class, Alternative Parents
```

LIPID MAPS 转换后至少需要：

```text
LM_ID, NAME, SMILES, CATEGORY, MAIN_CLASS, SUB_CLASS, CLASS_LEVEL4
```

注意：LIPID MAPS 官方下载的 TSV 只有 `LM_ID`、`INCHIKEY`、`SMILES`，没有分类字段；请下载 SDF，再按下文转换。

### 4.2 与用户讨论数据获取

先把下面几点说清楚，得到确认后再下载：

- **只用 HMDB**：可以完成 426 标签与训练；推荐先用这条路径跑通。
- **HMDB + LIPID MAPS**：会合并脂类结构，但 LIPID MAPS 行没有 `Alternative Parents`，标签全为 0。若用户不希望这些行参与训练，先用 HMDB-only 表，或明确约定如何处理。
- **数据版本**：记录下载日期与文件名，写入 `work/agent-runbook.md`。
- **许可**：提醒用户 HMDB 数据受再分发条款限制；仓库不转存、不再分发。
- **磁盘与时间**：HMDB 全量表和构象生成都可能较大；先估算空间，再决定是否全量。

### 4.3 HMDB 下载指引

1. 让用户在浏览器打开 <https://hmdb.ca/downloads>。
2. 找到 **Metabolite Data**（或同义的代谢物数据下载区）。
3. 下载**全量代谢物表**，优先选择 CSV 导出；不要只下载单个代谢物页面或示例数据。
4. 保存为 `data/raw/hmdb.csv`（或用户指定的其他路径）。
5. 如果页面只提供 XML，先让用户确认是否有 CSV/TSV 导出；不要直接把 XML 当作 CSV 喂给管线。

HMDB 可能对自动请求返回 403/Cloudflare 验证页。此时应停止自动抓取，让用户在浏览器完成下载。

下载后做列检查：

```shell
python - <<'PY'
import pandas as pd

path = "data/raw/hmdb.csv"
df = pd.read_csv(path, nrows=5)
required = [
    "HMDB ID", "SMILES", "IUPAC Name",
    "Kingdom", "Super Class", "Class", "Sub Class",
    "Alternative Parents",
]
missing = [column for column in required if column not in df.columns]
print("columns:", list(df.columns))
print("missing:", missing)
if missing:
    raise SystemExit("HMDB table is missing required columns")
PY
```

### 4.4 LIPID MAPS 下载与转换

1. 让用户打开 <https://www.lipidmaps.org/databases/lmsd/download>。
2. 下载 **LMSD Structure-data file (SDF)**（不是只有 ID 的 TSV）。
3. 保存为 `data/raw/LMSD.sdf`。
4. 安装依赖后，用下面的脚本转成管线需要的 CSV：

```shell
mkdir -p data/raw
python - <<'PY'
from pathlib import Path

import pandas as pd
from rdkit import Chem

source = Path("data/raw/LMSD.sdf")
target = Path("data/raw/lipidmaps.csv")
fields = ["LM_ID", "NAME", "SMILES", "CATEGORY", "MAIN_CLASS", "SUB_CLASS", "CLASS_LEVEL4"]

rows = []
for mol in Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False):
    if mol is None:
        continue
    row = {field: mol.GetProp(field) if mol.HasProp(field) else "" for field in fields}
    if not row["SMILES"]:
        row["SMILES"] = Chem.MolToSmiles(mol)
    rows.append(row)

frame = pd.DataFrame(rows, columns=fields)
frame = frame[frame["SMILES"].str.strip().ne("")]
frame.to_csv(target, index=False)
print(f"wrote {len(frame)} rows to {target}")
PY
```

转换后检查空 SMILES 与必需列。LIPID MAPS 是可选数据；用户不想合并时可跳过这一步。

### 4.5 数据验收

```shell
python - <<'PY'
import pandas as pd

for path, required in [
    ("data/raw/hmdb.csv", ["HMDB ID", "SMILES", "Alternative Parents"]),
    ("data/raw/lipidmaps.csv", ["LM_ID", "NAME", "SMILES", "CATEGORY", "MAIN_CLASS", "SUB_CLASS", "CLASS_LEVEL4"]),
]:
    df = pd.read_csv(path, nrows=5)
    missing = [column for column in required if column not in df.columns]
    print(path, "columns ok" if not missing else f"missing: {missing}")
PY
```

任一文件缺列时停止，先修正数据或调整列名映射，不要继续训练。

## 5. 路径 A：只推理（已有模型目录）

模型目录结构：

```text
<model_dir>/
├── label_map.json          # 426 个类名，顺序即模型语义
├── main_model/             # config.json + model.safetensors
└── vice_model/             # 426 个 AutoGluon predictor 目录（仅 vice 需要）
```

准备输入 CSV，至少包含一列 `smiles`：

```csv
smiles
CCO
c1ccccc1
```

运行：

```shell
smiles2class \
  --model main \
  --model-path <model_dir> \
  --input-csv input.csv \
  --output preds.csv
```

检查结果：

```shell
python - <<'PY'
import pandas as pd

frame = pd.read_csv("preds.csv")
print(frame.shape)
print(frame["status"].value_counts(dropna=False))
PY
```

输出行数应与输入行数一致。`status="invalid"` 的行概率为 `NaN`；`status="degraded_2d"` 属于分布外输入，应按 `status` 过滤或降权。

## 6. 路径 B：数据准备与训练

以下命令均在仓库根目录执行，默认使用 `data/raw/` 与 `work/`。

### 6.1 清洗与合并

只使用 HMDB（推荐先跑通）：

```shell
python -m pipelines.hmdb.clean \
  --input data/raw/hmdb.csv \
  --output work/train_compounds.parquet \
  --keep-alternative-parents
```

HMDB + LIPID MAPS：

```shell
python -m pipelines.hmdb.clean \
  --input data/raw/hmdb.csv \
  --lipidmaps-input data/raw/lipidmaps.csv \
  --output work/train_compounds.parquet \
  --keep-alternative-parents
```

训练表必须带 `--keep-alternative-parents`，否则 426 标签无法生成。

### 6.2 生成 426 标签

```shell
python -m pipelines.hmdb.labels \
  --input work/train_compounds.parquet \
  --output work/label_map.json \
  --labels-output work/labels.parquet
```

检查：

```shell
python - <<'PY'
import json
import pandas as pd

labels = json.load(open("work/label_map.json", encoding="utf-8"))
table = pd.read_parquet("work/labels.parquet")
print("labels:", len(labels))
print("table:", table.shape)
print("first columns:", table.columns[:5].tolist())
assert len(labels) == 426
PY
```

`pipelines.hmdb.labels` 产出的 SMILES 列名为大写 `SMILES`。后续训练与评测需要显式使用 `--smiles-column SMILES`。

### 6.3 划分

```shell
python -m pipelines.hmdb.split \
  --input work/labels.parquet \
  --output work/splits.json \
  --stats-output work/split_stats.json
```

检查 `train` / `vali` / `test` 无重叠，并集等于全表 ID。

### 6.4 主模型训练

```shell
python -m training.main.prepare \
  --labels work/labels.parquet \
  --splits work/splits.json \
  --output work/dataset.pkl \
  --smiles-column SMILES \
  --workers 8
```

```shell
python -m training.main.train \
  --dataset work/dataset.pkl \
  --unimol <unimol_lite_encoder_dir> \
  --training-config training/main/configs/default.yaml \
  --output runs/main \
  --export
```

`--unimol` 指向 UniMol-Lite 的 encoder 目录，不是官方 Uni-Mol。用户没有该目录时先询问来源；不要用官方 Uni-Mol 权重替代。

训练完成后确认：

```text
runs/main/
├── main_model/config.json
├── main_model/model.safetensors
└── label_map.json
```

### 6.5 副模型训练（可选）

```shell
python -m training.vice.prepare \
  --input work/labels.parquet \
  --model-dir runs/main \
  --output work/vice.parquet \
  --smiles-column SMILES
```

```shell
python -m training.vice.train \
  --dataset work/vice.parquet \
  --output runs/main/vice_model \
  --label-map runs/main/label_map.json
```

副模型是 426 个 AutoGluon 集成，CPU 训练，耗时较长。先确认用户有时间预算。

## 7. 评测

```shell
python -m evaluation.evaluate \
  --input work/labels.parquet \
  --model main \
  --model-path runs/main \
  --smiles-column SMILES \
  --output evaluation/reports/main
```

如果使用自定义真值表，按实际 SMILES 列名传 `--smiles-column`。评测产物写到 `evaluation/reports/`，不要提交。

## 8. 检查点与失败处理

| 现象 | 处理 |
|---|---|
| `pip install` 失败 | 检查 Python 版本（3.10/3.11）与网络；不要降级到不受支持的 Python |
| HMDB 403 / Cloudflare | 让用户在浏览器下载；不要绕过 |
| `missing required column(s)` | 修正源表或 `--config` 列名映射；不要凭空补列 |
| `labels.parquet` 只有少数标签 | 输入数据太少或 `Alternative Parents` 解析失败；先在 `work/labels.parquet` 检查非零标签数 |
| `split` 后 train 为空 | 数据量太小；增加样本后重跑，不要用 2–3 行数据做 `prepare`/训练 |
| `training.main.prepare` 找不到 `smiles` | 显式传 `--smiles-column SMILES` |
| `training.main.prepare` 参数错误 | 入口参数是 `--labels`，不是 `--input` |
| 训练显存/内存不足 | 降低 `batch_size`、`max_atoms` 或先用子集；向用户确认 |
| 预测行数不一致 | 检查输入 CSV 是否有空行或列名错误 |

任何一步失败时，先把命令、错误摘要和已确认的数据路径写入 `work/agent-runbook.md`，再向用户报告。

## 9. 交付物与记录

建议在 `work/agent-runbook.md` 记录：

- 用户目标与选择（推理 / 训练 / 只跑通）；
- 数据来源、下载日期、文件名与许可确认；
- 行数与列检查结果；
- 执行的命令与输出路径；
- 训练配置、硬件、耗时；
- 预测/评测结果摘要与异常状态计数。

不要提交 `data/`、`work/`、`runs/`、`evaluation/reports/` 或任何模型权重。
