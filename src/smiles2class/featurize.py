"""SMILES → Uni-Mol item, via the published ``unimol-lite`` package."""

from __future__ import annotations

from typing import Iterable

from unimol_lite import featurize_batch


def featurize(
    smiles: Iterable[str],
    *,
    workers: int | None = 8,
    timeout: float | None = 30.0,
    coord_mode: str = "auto",
):
    """Featurize a batch, returning ``(items, metas)`` in input order."""
    results = featurize_batch(
        list(smiles), workers=workers, timeout=timeout, coord_mode=coord_mode
    )
    items = [item for item, _ in results]
    metas = [meta for _, meta in results]
    return items, metas
