from __future__ import annotations

import os
from pathlib import Path

import pytest

# Repo layout: <workspace>/smiles2class/tests/conftest.py
DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "model" / "uni_classify"
MODEL_DIR = Path(os.environ.get("S2C_MODEL_DIR", DEFAULT_MODEL_DIR))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs the full 426-predictor vice model")


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "main_model" / "model.safetensors").exists():
        pytest.skip(f"model directory not available: {MODEL_DIR}")
    return MODEL_DIR
