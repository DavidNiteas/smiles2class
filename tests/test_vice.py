"""Vice-model tests.

The full 426-predictor golden is slow (minutes and ~10 GB); it is opt-in via
``S2C_FULL_VICE=1``. The default test verifies the wrapper against a single
predictor so the inference plumbing is covered without loading 426 models.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from smiles2class.model import load_main_model
from smiles2class.vice import ViceModel
from unimol_lite.data import smiles2unimolitem
from unimol_lite.data.collate_fn import unimol_collate_fn


def test_vice_wrapper_matches_predictor(model_dir):
    from autogluon.tabular import TabularDataset, TabularPredictor

    n = 5
    predictors = [
        TabularPredictor.load(str(model_dir / "vice_model" / str(i)), verbosity=0)
        for i in range(n)
    ]
    wrapper = ViceModel(predictors, [f"class_{i}" for i in range(n)])

    model = load_main_model(model_dir / "main_model")
    item = smiles2unimolitem("CCO", coord_mode="auto")
    batch = unimol_collate_fn([item])
    import torch

    with torch.no_grad():
        _, embeds = model.predict(**batch)
    embeds = embeds.numpy()

    wrapped = wrapper.predict(embeds, workers=1)
    frame = pd.DataFrame(embeds, columns=[f"vec_{i}" for i in range(embeds.shape[1])])
    direct = np.column_stack(
        [
            np.asarray(
                predictor.predict_proba(TabularDataset(frame), as_multiclass=False),
                dtype=np.float64,
            )
            for predictor in predictors
        ]
    )

    assert wrapped.shape == (1, n)
    np.testing.assert_array_equal(wrapped, direct)


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("S2C_FULL_VICE") != "1", reason="set S2C_FULL_VICE=1")
def test_vice_full_golden(model_dir):
    from pathlib import Path

    import torch

    from smiles2class import ViceClassifier

    golden = Path(__file__).resolve().parent / "golden" / "vice.npz"
    if not golden.exists():
        pytest.skip("missing vice golden")

    data = np.load(golden, allow_pickle=True)
    smiles = [str(s) for s in data["smiles"]]
    classifier = ViceClassifier.from_pretrained(model_dir)
    probs, metas = classifier.predict(
        smiles, workers=1, timeout=60.0, vice_workers=1, return_meta=True
    )
    assert [meta["status"] for meta in metas] == ["ok"] * len(smiles)
    np.testing.assert_allclose(probs.to_numpy(), data["probs"], rtol=0, atol=1e-9)

