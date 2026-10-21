"""Fixtures for the repo-local data pipelines.

All data here is synthetic (12 molecules, 3 labels).  No real HMDB / LIPID MAPS
tables are needed, and the repository never ships any.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipelines import load_config  # noqa: E402
from pipelines.hmdb.labels import build_multi_hot  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "pipelines" / "hmdb" / "configs" / "default.yaml"
COLUMNS = load_config(DEFAULT_CONFIG)["columns"]

AUTHORITY = json.loads(
    (REPO_ROOT / "src" / "smiles2class" / "labels.json").read_text(encoding="utf-8")
)
LABEL_A, LABEL_B, LABEL_C = AUTHORITY[0], AUTHORITY[1], AUTHORITY[2]

# (HMDB ID, SMILES, parents, encoding) -- A on rows 1-6, B on 3/5/7/8, C on 9-12.
ROWS: list[tuple[str, str, list[str], str]] = [
    ("HMDB00001", "CCO", [LABEL_A], "python"),
    ("HMDB00002", "CCN", [LABEL_A], "json"),
    ("HMDB00003", "CC(=O)O", [LABEL_A, LABEL_B], "json"),
    ("HMDB00004", "c1ccccc1", [LABEL_A], "python"),
    ("HMDB00005", "CC(=O)Oc1ccccc1C(=O)O", [LABEL_A, LABEL_B], "delimiter"),
    ("HMDB00006", "CC(C)C", [LABEL_A], "python"),
    ("HMDB00007", "C1CCCCC1", [LABEL_B], "json"),
    ("HMDB00008", "CCOC(=O)C", [LABEL_B], "plain"),
    ("HMDB00009", "CCCCCCCC", [LABEL_C], "python"),
    ("HMDB00010", "CCCCCCCCC", [LABEL_C], "json"),
    ("HMDB00011", "O=C=O", [LABEL_C], "python"),
    ("HMDB00012", "NCC(=O)O", [LABEL_C], "delimiter"),
]


def encode_parents(parents: list[str], encoding: str) -> str:
    if encoding == "python":
        return repr(parents)
    if encoding == "json":
        return json.dumps(parents)
    if encoding == "delimiter":
        return "; ".join(parents)
    if encoding == "plain":
        return parents[0]
    raise ValueError(f"unknown encoding: {encoding}")


def hmdb_raw_frame() -> pd.DataFrame:
    # Deliberately out of order: wash_hmdb_data must sort by HMDB ID.
    records = [
        {
            "HMDB ID": identifier,
            "SMILES": smiles,
            "IUPAC Name": f"Compound {identifier[-2:]}",
            "Kingdom": "Organic compounds",
            "Super Class": "Lipids and lipid-like molecules",
            "Class": "Fatty Acyls",
            "Sub Class": "Fatty acids",
            "Alternative Parents": encode_parents(parents, encoding),
            "Unrelated": "x",
        }
        for identifier, smiles, parents, encoding in reversed(ROWS)
    ]
    return pd.DataFrame(records)


def lipidmaps_raw_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "LM_ID": "LM0001",
                "NAME": "LM one",
                "SMILES": "CCCC",
                "CATEGORY": "Lipids",
                "MAIN_CLASS": "Fatty Acyls",
                "SUB_CLASS": "Fatty acids",
                "CLASS_LEVEL4": "Straight chain",
            },
            {
                "LM_ID": "LM0002",
                "NAME": "LM two",
                "SMILES": "CCCCCC",
                "CATEGORY": "Lipids",
                "MAIN_CLASS": "Glycerolipids",
                "SUB_CLASS": "Monoradylglycerols",
                "CLASS_LEVEL4": "Triacylglycerols",
            },
            {
                # Empty SMILES: transform_lipidmaps_data must drop this row.
                "LM_ID": "LM0003",
                "NAME": "LM without smiles",
                "SMILES": "",
                "CATEGORY": "Lipids",
                "MAIN_CLASS": "Fatty Acyls",
                "SUB_CLASS": "Fatty acids",
                "CLASS_LEVEL4": "Branched",
            },
        ]
    )


@pytest.fixture
def columns() -> dict:
    return COLUMNS


@pytest.fixture
def authority() -> list[str]:
    return list(AUTHORITY)


@pytest.fixture
def label_a() -> str:
    return LABEL_A


@pytest.fixture
def label_b() -> str:
    return LABEL_B


@pytest.fixture
def label_c() -> str:
    return LABEL_C


@pytest.fixture
def hmdb_frame() -> pd.DataFrame:
    return hmdb_raw_frame()


@pytest.fixture
def lipidmaps_frame() -> pd.DataFrame:
    return lipidmaps_raw_frame()


@pytest.fixture
def hmdb_raw(tmp_path: Path) -> Path:
    path = tmp_path / "hmdb_raw.csv"
    hmdb_raw_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def lipidmaps_raw(tmp_path: Path) -> Path:
    path = tmp_path / "lipidmaps_raw.csv"
    lipidmaps_raw_frame().to_csv(path, index=False)
    return path


@pytest.fixture
def multi_hot_frame(columns) -> pd.DataFrame:
    """12-row table already reduced to the three fixture labels."""
    frame = pd.DataFrame(
        {
            "id": [row[0] for row in ROWS],
            "SMILES": [row[1] for row in ROWS],
            "Alternative Parents": [list(row[2]) for row in ROWS],
        }
    )
    return build_multi_hot(frame, [LABEL_A, LABEL_B, LABEL_C], columns)


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT
