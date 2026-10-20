"""Golden test for the main-model port.

The fixture was generated with the original ``uni_classify`` implementation
(``UniClassifyModel.predict``); the port must reproduce it bit-for-bit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from smiles2class import MainClassifier

GOLDEN = Path(__file__).resolve().parent / "golden" / "main.npz"


def test_main_golden(model_dir):
    if not GOLDEN.exists():
        pytest.skip(f"missing golden file: {GOLDEN}")
    data = np.load(GOLDEN, allow_pickle=True)
    smiles = [str(s) for s in data["smiles"]]

    classifier = MainClassifier.from_pretrained(model_dir)
    probs, metas = classifier.predict(smiles, workers=2, timeout=60.0, return_meta=True)

    assert [meta["status"] for meta in metas] == ["ok"] * len(smiles)
    np.testing.assert_array_equal(probs.to_numpy(), data["probs"])


def test_invalid_and_duplicate_rows(model_dir):
    classifier = MainClassifier.from_pretrained(model_dir)
    probs, metas = classifier.predict(
        ["CCO", "not-a-smiles", "CCO"], workers=2, timeout=60.0, return_meta=True
    )

    assert probs.shape == (3, 426)
    assert [meta["status"] for meta in metas] == ["ok", "invalid", "ok"]
    assert np.isnan(probs.to_numpy()[1]).all()
    # Duplicate SMILES are featurized independently; RDKit conformer generation
    # can differ at the 1e-8 level between calls, so compare with a tight tolerance.
    np.testing.assert_allclose(probs.to_numpy()[0], probs.to_numpy()[2], rtol=0, atol=1e-6)


def test_empty_input(model_dir):
    classifier = MainClassifier.from_pretrained(model_dir)
    probs, metas = classifier.predict([], return_meta=True)
    assert probs.shape == (0, 426)
    assert metas == []
