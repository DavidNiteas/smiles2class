"""Vice model: 426 per-class AutoGluon predictors over the main-model embedding."""

from __future__ import annotations

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

LoadMode = Literal["lazy", "eager"]
_LOAD_MODES = ("lazy", "eager")


class ViceModel:
    def __init__(self, predictors, labels, load_mode: LoadMode = "lazy", feature_dim: int = 512):
        if load_mode not in _LOAD_MODES:
            raise ValueError(f"load_mode must be one of {_LOAD_MODES}, got {load_mode!r}")
        self.predictors = predictors
        self.labels = labels
        self.load_mode = load_mode
        self.feature_dim = feature_dim
        if load_mode == "eager":
            self.warmup()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        label_path: str | Path | None = None,
        load_mode: LoadMode = "lazy",
        feature_dim: int = 512,
    ):
        from autogluon.tabular import TabularPredictor

        model_path = Path(model_path)
        label_path = (
            Path(label_path) if label_path is not None else model_path.parent / "label_map.json"
        )
        labels = json.loads(Path(label_path).read_text())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            predictors = [
                TabularPredictor.load(str(model_path / str(i)), verbosity=0)
                for i in range(len(labels))
            ]
        return cls(predictors, labels, load_mode=load_mode, feature_dim=feature_dim)

    def _inputs(self, mol_embeds: np.ndarray):
        from autogluon.tabular import TabularDataset

        frame = pd.DataFrame(
            mol_embeds, columns=[f"vec_{i}" for i in range(mol_embeds.shape[1])]
        )
        return TabularDataset(frame)

    @staticmethod
    def _one(predictor, inputs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(
                predictor.predict_proba(inputs, as_multiclass=False), dtype=np.float64
            )

    def warmup(self) -> None:
        """Force every predictor to deserialize its models into process memory."""
        dummy = np.zeros((1, self.feature_dim), dtype=np.float64)
        inputs = self._inputs(dummy)
        for predictor in self.predictors:
            self._one(predictor, inputs)

    def predict(self, mol_embeds: np.ndarray, workers: int = 1) -> np.ndarray:
        """Return ``(N, 426)`` per-class positive probabilities.

        ``workers`` is passed to a thread pool when greater than 1.  On the
        model host (rotational HDD) concurrent execution gives no speedup for
        cold caches, so the default is sequential.
        """
        mol_embeds = np.asarray(mol_embeds)
        inputs = self._inputs(mol_embeds)

        def _one(predictor):
            return self._one(predictor, inputs)

        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                columns = list(pool.map(_one, self.predictors))
        else:
            columns = [_one(predictor) for predictor in self.predictors]
        return np.stack(columns, axis=1)
