"""Vice model: 426 per-class AutoGluon predictors over the main-model embedding."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


class ViceModel:
    def __init__(self, predictors, labels):
        self.predictors = predictors
        self.labels = labels

    @classmethod
    def from_pretrained(cls, model_path: str | Path, label_path: str | Path | None = None):
        from autogluon.tabular import TabularPredictor

        model_path = Path(model_path)
        label_path = Path(label_path) if label_path is not None else model_path.parent / "label_map.json"
        labels = json.loads(Path(label_path).read_text())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictors = [
                TabularPredictor.load(str(model_path / str(i)), verbosity=0)
                for i in range(len(labels))
            ]
        return cls(predictors, labels)

    def predict(self, mol_embeds: np.ndarray, workers: int = 1) -> np.ndarray:
        """Return ``(N, 426)`` per-class positive probabilities.

        .. note::
           Autogluon's ``TabularPredictor.predict_proba`` is not reliably
           thread-safe in this build: calling 426 predictors from a thread pool
           can block indefinitely (observed: CPU time 26 s over 16 min, state
           ``Sl``).  ``workers > 1`` is therefore accepted for API compatibility
           but falls back to sequential execution with a warning.
        """
        from autogluon.tabular import TabularDataset

        if workers > 1:
            warnings.warn(
                "autogluon predictors are not thread-safe here; "
                "falling back to sequential predict_proba",
                RuntimeWarning,
                stacklevel=2,
            )

        mol_embeds = np.asarray(mol_embeds)
        frame = pd.DataFrame(
            mol_embeds, columns=[f"vec_{i}" for i in range(mol_embeds.shape[1])]
        )
        inputs = TabularDataset(frame)

        def _one(predictor):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return np.asarray(
                    predictor.predict_proba(inputs, as_multiclass=False), dtype=np.float64
                )

        columns = [_one(p) for p in self.predictors]
        return np.stack(columns, axis=1)
