"""Smoke tests for the migrated main-model training code.

One training step on eight synthetic molecules with a tiny model, then export
to the inference-side model directory and load it back with
``MainClassifier``.  No full training runs here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from smiles2class import MainClassifier
from smiles2class.model import UniClassifyConfig, UniClassifyModel
from training.main.lightning import TrainingConfig
from training.main.prepare import prepare
from training.main.train import run_training

SMILES = ["CCO", "CCN", "CCC", "CCCl", "CCBr", "c1ccccc1", "CC(=O)O", "CC(C)O"]
LABELS = ["class_a", "class_b", "class_c"]
SPLITS = {"train": [0, 1, 2, 3], "vali": [4, 5], "test": [6, 7]}


def tiny_model_config() -> UniClassifyConfig:
    return UniClassifyConfig(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_embed_dim=64,
        decoder_ffn_embed_dim=128,
        decoder_attention_heads=4,
        decoder_token_num=len(LABELS),
        only_train_decoder=True,
        emb_dropout=0.0,
        dropout=0.0,
        attention_dropout=0.0,
        mode="infer",
    )


@pytest.fixture(scope="module")
def tiny_unimol(tmp_path_factory) -> Path:
    from unimol_lite.model import UniMolModel
    from unimol_lite.model.config import UniMolConfig

    path = tmp_path_factory.mktemp("unimol")
    config = UniMolConfig(
        encoder_layers=1,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=128,
        encoder_attention_heads=4,
        emb_dropout=0.0,
        dropout=0.0,
        attention_dropout=0.0,
    )
    UniMolModel(config).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("train_data")
    frame = pd.DataFrame(
        {
            "id": list(range(len(SMILES))),
            "smiles": SMILES,
            "label_0": [1, 0, 1, 0, 1, 0, 1, 0],
            "label_1": [0, 1, 0, 1, 0, 1, 0, 1],
            "label_2": [1, 1, 1, 0, 0, 1, 0, 0],
        }
    )
    frame.to_parquet(root / "labels.parquet")
    (root / "label_map.json").write_text(json.dumps(LABELS), encoding="utf-8")
    (root / "splits.json").write_text(json.dumps(SPLITS), encoding="utf-8")
    datamodule = prepare(
        root / "labels.parquet",
        root / "splits.json",
        root / "dataset.pkl",
        workers=2,
        timeout=60.0,
        batch_size=2,
    )
    return root, datamodule


def test_forward_returns_per_class_logits(prepared):
    _root, datamodule = prepared
    batch = datamodule.trainset.collate_fn([datamodule.trainset[0], datamodule.trainset[1]])
    model = UniClassifyModel(tiny_model_config())
    with torch.no_grad():
        logits = model(**batch)
    assert logits.shape == (2, len(LABELS), 2)


def test_one_step_train_export_and_reload(prepared, tiny_unimol, tmp_path):
    _root, datamodule = prepared
    settings = TrainingConfig(
        batch_size=2,
        learning_rate=1e-3,
        accelerator="cpu",
        devices=1,
        precision="32",
        max_epochs=1,
        max_steps=-1,
        seed=0,
    )
    output = tmp_path / "model"
    _module, trainer = run_training(
        datamodule,
        tiny_model_config(),
        settings,
        unimol=tiny_unimol,
        output=output,
        limit_train_batches=1,
        export=True,
        labels=LABELS,
    )

    assert trainer.global_step == 1
    assert trainer.checkpoint_callback.best_model_path
    assert (output / "main_model" / "config.json").is_file()
    assert (output / "main_model" / "model.safetensors").is_file()
    assert json.loads((output / "label_map.json").read_text(encoding="utf-8")) == LABELS

    classifier = MainClassifier.from_pretrained(output)
    probs = classifier.predict(SMILES, workers=2, timeout=60.0)
    assert probs.shape == (len(SMILES), len(LABELS))
    assert np.isfinite(probs.to_numpy()).all()
    assert ((probs.to_numpy() >= 0) & (probs.to_numpy() <= 1)).all()
