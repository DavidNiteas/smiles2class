"""Public classifiers: main and vice (no fusion)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from unimol_lite.data import unimol_collate_fn

from .featurize import featurize
from .labels import LABELS
from .model import load_main_model
from .vice import ViceModel
from .weights import resolve_model_dir


def _chunks(seq, size):
    for start in range(0, len(seq), size):
        yield seq[start : start + size]


class MainClassifier:
    """Main model: SMILES → ``(N, 426)`` per-class probabilities."""

    def __init__(self, model, labels: list[str], device: str = "cpu"):
        self.model = model
        self.labels = labels
        self.device = device

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: str = "cpu", labels: list[str] | None = None
    ) -> "MainClassifier":
        model_dir = resolve_model_dir(model_dir)
        labels = labels or json.loads((model_dir / "label_map.json").read_text())
        model = load_main_model(model_dir / "main_model", device=device)
        return cls(model, labels, device=device)

    def _predict(self, smiles, workers, timeout, coord_mode, batch_size):
        items, metas = featurize(
            smiles, workers=workers, timeout=timeout, coord_mode=coord_mode
        )
        n = len(items)
        probs = np.full((n, len(self.labels)), np.nan, dtype=np.float64)
        embeds = np.full((n, self.model.configs.encoder_embed_dim), np.nan, dtype=np.float64)

        valid = [(i, item) for i, item in enumerate(items) if item is not None]
        for chunk in _chunks(valid, batch_size):
            batch = unimol_collate_fn([item for _, item in chunk])
            batch = {key: value.to(self.device) for key, value in batch.items()}
            batch_probs, batch_embeds = self.model.predict(**batch)
            batch_probs = batch_probs.detach().cpu().numpy()
            batch_embeds = batch_embeds.detach().cpu().numpy()
            for row, (index, _) in enumerate(chunk):
                probs[index] = batch_probs[row]
                embeds[index] = batch_embeds[row]
        return probs, embeds, metas

    def predict(
        self,
        smiles,
        *,
        workers: int | None = 8,
        timeout: float | None = 30.0,
        coord_mode: str = "auto",
        batch_size: int = 64,
        return_meta: bool = False,
    ):
        smiles = list(smiles)
        probs, _embeds, metas = self._predict(
            smiles, workers, timeout, coord_mode, batch_size
        )
        frame = pd.DataFrame(probs, columns=self.labels)
        if return_meta:
            return frame, metas
        return frame


class ViceClassifier:
    """Vice model: SMILES → main embedding → 426 per-class predictors."""

    def __init__(self, main: MainClassifier, vice: ViceModel):
        self.main = main
        self.vice = vice
        self.labels = main.labels

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        device: str = "cpu",
        labels: list[str] | None = None,
    ) -> "ViceClassifier":
        model_dir = resolve_model_dir(model_dir)
        main = MainClassifier.from_pretrained(model_dir, device=device, labels=labels)
        vice = ViceModel.from_pretrained(model_dir / "vice_model", model_dir / "label_map.json")
        return cls(main, vice)

    def predict(
        self,
        smiles,
        *,
        workers: int | None = 8,
        timeout: float | None = 30.0,
        coord_mode: str = "auto",
        batch_size: int = 64,
        vice_workers: int = 1,
        return_meta: bool = False,
    ):
        smiles = list(smiles)
        _main_probs, embeds, metas = self.main._predict(
            smiles, workers, timeout, coord_mode, batch_size
        )
        probs = np.full((len(smiles), len(self.labels)), np.nan, dtype=np.float64)
        valid = ~np.isnan(embeds).any(axis=1)
        if valid.any():
            probs[valid] = self.vice.predict(embeds[valid], workers=vice_workers)
        frame = pd.DataFrame(probs, columns=self.labels)
        if return_meta:
            return frame, metas
        return frame


__all__ = ["MainClassifier", "ViceClassifier", "LABELS"]
