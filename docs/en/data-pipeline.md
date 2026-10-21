# Data pipeline

[中文](../zh/data-pipeline.md) | **English**

`pipelines/` turns raw HMDB / LIPID MAPS tables into a cleaned compound table, a 426-label multi-hot table and train / vali / test splits. It is a **repo-local** tool: it is **not shipped in the wheel** (only in the sdist), and the repository ships **no raw or derived data** — every input is supplied by the caller.

HMDB data is subject to its redistribution terms. This repository does not provide raw or derived HMDB data. Users who need to reproduce the pipeline should obtain the required data from the official HMDB website and follow its terms.

## Requirements

- `pandas`;
- `pyarrow` for `.parquet`;
- `PyYAML` for configs (`pip install -e ".[dev]"` includes it).

## Input schema

### Raw HMDB table (`.csv` / `.parquet`)

| Column | Required | Meaning |
|---|---|---|
| `HMDB ID` | yes | molecule id; the cleaned table is sorted by it |
| `SMILES` | yes | structure; copied verbatim into `CANON_SMILES` |
| `IUPAC Name` | when merging | becomes `Name` |
| `Kingdom` / `Super Class` / `Class` / `Sub Class` | when merging | taxonomy fields |
| `Alternative Parents` | for labels | ClassyFire parent terms |

### Raw LIPID MAPS table (optional)

`LM_ID`, `NAME`, `SMILES`, `CATEGORY`, `MAIN_CLASS`, `SUB_CLASS`, `CLASS_LEVEL4`; rows without SMILES are dropped.

### Label input (`pipelines.hmdb.labels`)

Requires `SMILES` and `Alternative Parents`. The first existing id candidate wins: `id` → `ID` → `HMDB ID`; otherwise positional ids are used with a warning.

`Alternative Parents` cells may be written as:

```text
['Organic oxides', 'Carbonyl compounds']      # Python list literal
["Organic oxides", "Carbonyl compounds"]      # JSON array
Organic oxides; Carbonyl compounds            # ; or | separated
```

A bare comma is accepted only when every resulting piece is a known label (some ClassyFire terms contain commas, e.g. `1,2-diols`).

Column mappings live under `columns` in `pipelines/hmdb/configs/default.yaml`; copy it and pass `--config` for a different table layout.

## Three commands

```shell
# 1. wash + merge
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --lipidmaps-input raw/lipidmaps.csv \
  --output work/compounds.parquet

# training table: keep Alternative Parents
python -m pipelines.hmdb.clean \
  --input raw/hmdb.csv \
  --output work/hmdb_train.parquet \
  --keep-alternative-parents

# 2. 426-label multi-hot
python -m pipelines.hmdb.labels \
  --input work/hmdb_train.parquet \
  --output work/label_map.json \
  --labels-output work/labels.parquet

# 3. multi-label split
python -m pipelines.hmdb.split \
  --input work/labels.parquet \
  --output work/splits.json \
  --stats-output work/split_stats.json
```

- The output format follows the suffix (`.csv` / `.parquet`);
- without `--lipidmaps-input`, `clean` only washes HMDB and keeps every row (including empty SMILES);
- the merged delivery table drops `Alternative Parents` by default; add `--keep-alternative-parents` for training tables.

## Output contracts

### `clean`

Without LIPID MAPS: the original HMDB columns plus `CANON_SMILES`, sorted by `HMDB ID`, with `Alternative Parents` dropped by default.

With LIPID MAPS, the unified schema is:

| Column | Source |
|---|---|
| `ID` | `HMDB ID` / `LM_ID` |
| `Name` | `IUPAC Name` / `NAME` |
| `SMILES` | unchanged |
| `Kingdom` | `Kingdom` / `CATEGORY` |
| `Super Class` | `Super Class` / `MAIN_CLASS` |
| `Class` | `Class` / `SUB_CLASS` |
| `Sub Class` | `Sub Class` / `CLASS_LEVEL4` |

HMDB rows come first, LIPID MAPS rows after; no de-duplication. With `--keep-alternative-parents` the table ends with an `Alternative Parents` column (empty for LIPID MAPS rows).

### `labels`

- `label_map.json`: the 426 class names, **item for item identical** to the authority file (including order);
- multi-hot table (defaults to `labels.parquet`): `id`, `SMILES`, `label_0` … `label_425` with 0/1 values.

A label that never occurs in the input produces a warning (it is never dropped silently); `label_map.json` always stays the full vocabulary.

### `split`

`splits.json` has exactly three keys:

```json
{"train": ["id..."], "vali": ["id..."], "test": ["id..."]}
```

`--stats-output` adds split sizes/ratios, parameters (seed / ratios), per-label coverage, labels without samples, labels below the 20% target, and the number of unlabelled rows.

Algorithm (greedy coverage split):

1. walk `label_0 … label_425` in order;
2. assign the label's unassigned molecules to test until that label's test coverage reaches 20%;
3. then to vali until it reaches 20%;
4. everything else goes to train; a molecule belongs to one split only;
5. molecules covered by no label go to train, so the three splits always form a complete partition.

The algorithm has no randomness and the output is deterministic; `--seed` is only recorded in the stats (reserved for future randomised variants).

## Label authority

The 426 class names and their order come from `src/smiles2class/labels.json`, aligned with the model weights and `label_map.json`. **Do not reorder or add/remove classes** — the order is part of the model semantics. `pipelines.hmdb.labels` only reads that file and never re-derives labels.

## Config

Defaults: `pipelines/hmdb/configs/default.yaml`. A `--config` file is deep-merged over the defaults, so it only needs the keys you want to override.

| Section | Content |
|---|---|
| `clean` / `labels` / `split` | optional default paths (left empty in the repo; pass paths on the CLI) |
| `split.seed` / `test_ratio` / `vali_ratio` | split parameters, defaults `42` / `0.2` / `0.2` |
| `labels.path` / `expected_count` | label authority path (defaults to the repo file) and expected size (426) |
| `columns` | column mappings for HMDB / LIPID MAPS / unified / labels / split |

Everything can also be driven from the command line without editing the config.
