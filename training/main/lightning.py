"""Training-time config and ``LightningModule`` for the main model.

The model definition itself lives in :mod:`smiles2class.model`; this module adds
only loss / optimiser / metrics / Lightning hooks.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, F1Score, Precision, Recall

from smiles2class.model import UniClassifyConfig, UniClassifyModel


@dataclass
class TrainingConfig:
    """Optimisation / trainer hyper-parameters (ported from ``UniClassifyTrainingConfig``)."""

    batch_size: int = 32
    dataloader_num_workers: int = 0
    learning_rate: float = 1e-5
    scheduler_monitor: str = "val_loss"
    scheduler_monitor_mode: str = "min"
    scheduler_factor: float = 0.1
    scheduler_patience: int = 5
    scheduler_interval: str = "epoch"
    scheduler_frequency: int = 1
    scheduler_strict: bool = True
    scheduler_name: str = "val_loss"
    accelerator: str = "auto"
    devices: Union[int, str, list] = 1
    precision: Union[str, int] = "bf16"
    max_epochs: int = 100
    max_steps: int = -1
    seed: int = 42

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TrainingConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown training config keys: {unknown}")
        return cls(**{key: value for key, value in data.items() if key in known})

    @classmethod
    def from_json_file(cls, path: str | Path) -> "TrainingConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "TrainingConfig":
        import yaml

        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def _as_training_config(value: Union[TrainingConfig, Mapping[str, Any], None]) -> TrainingConfig:
    if value is None:
        return TrainingConfig()
    if isinstance(value, TrainingConfig):
        return value
    return TrainingConfig.from_dict(value)


def _as_model_config(value: Union[UniClassifyConfig, Mapping[str, Any], None]) -> UniClassifyConfig:
    if value is None:
        return UniClassifyConfig()
    if isinstance(value, UniClassifyConfig):
        return value
    return UniClassifyConfig(**dict(value))


def model_config_from_dict(data: Mapping[str, Any]) -> UniClassifyConfig:
    """Build a :class:`UniClassifyConfig` from a plain mapping."""
    return _as_model_config(data)


class UniClassifyLightningModule(pl.LightningModule):
    """Lightning wrapper around :class:`~smiles2class.model.UniClassifyModel`."""

    def __init__(
        self,
        model_config: Union[UniClassifyConfig, Mapping[str, Any]],
        training_config: Union[TrainingConfig, Mapping[str, Any]],
        model: Optional[UniClassifyModel] = None,
    ):
        super().__init__()
        self.model_config = _as_model_config(model_config)
        self.training_config = _as_training_config(training_config)
        # Checkpoint hyper-parameters stay plain so ``load_from_checkpoint`` works.
        self.save_hyperparameters(
            {
                "model_config": self.model_config.to_dict(),
                "training_config": asdict(self.training_config),
            },
            ignore=["model"],
        )
        self.model = model if model is not None else UniClassifyModel(self.model_config)
        self.accuracy = Accuracy(task="binary", num_classes=2)
        self.precision = Precision(task="binary", num_classes=2)
        self.recall = Recall(task="binary", num_classes=2)
        self.f1 = F1Score(task="binary", num_classes=2)
        self.loss_weights = None

    @classmethod
    def from_unimol(
        cls,
        unimol_path: str | Path,
        model_config: Union[UniClassifyConfig, Mapping[str, Any]],
        training_config: Union[TrainingConfig, Mapping[str, Any]],
        *args,
        **kwargs,
    ) -> "UniClassifyLightningModule":
        configs = _as_model_config(model_config)
        model = UniClassifyModel.from_unimol(unimol_path, configs, *args, **kwargs)
        return cls(configs, training_config, model=model)

    def init_loss_weights(self, training_labels: torch.Tensor):
        true_prob = training_labels.sum(dim=0) / training_labels.shape[0]
        false_prob = 1 - true_prob
        w_true = torch.pow(true_prob, -1) / (torch.pow(true_prob, -1) + torch.pow(false_prob, -1))
        w_false = torch.pow(false_prob, -1) / (torch.pow(true_prob, -1) + torch.pow(false_prob, -1))
        self.loss_weights = torch.stack([w_false, w_true], dim=0)
        self.loss_weights_index = torch.arange(self.loss_weights.shape[1])

    def merge_loss(self, loss: torch.Tensor, labels: torch.Tensor):
        if self.loss_weights is None:
            return loss.mean()
        weights = self.loss_weights.to(loss.device)
        index = self.loss_weights_index.to(loss.device)
        loss = (loss * weights[labels, index].view(-1)).mean()
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=loss.device)
        return loss

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.model.forward(*args, **kwargs)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        batch = dict(batch)
        labels = batch.pop("label")
        class_logits = self.forward(**batch)
        if self.loss_weights is None:
            loss = F.cross_entropy(class_logits.view(-1, 2), labels.view(-1))
        else:
            loss = F.cross_entropy(class_logits.view(-1, 2), labels.view(-1), reduction="none")
            loss = self.merge_loss(loss, labels)

        preds = class_logits.argmax(dim=-1).view(-1)
        targets = labels.view(-1)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", self.accuracy(preds, targets))
        self.log(f"{stage}_precision", self.precision(preds, targets))
        self.log(f"{stage}_recall", self.recall(preds, targets))
        self.log(f"{stage}_f1", self.f1(preds, targets), prog_bar=True)
        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "test")

    def configure_optimizers(self):
        if self.model_config.only_train_decoder:
            parameters = self.model.class_decoder.parameters()
        else:
            parameters = self.parameters()
        optimizer = torch.optim.Adam(parameters, lr=self.training_config.learning_rate)
        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode=self.training_config.scheduler_monitor_mode,
                factor=self.training_config.scheduler_factor,
                patience=self.training_config.scheduler_patience,
            ),
            "monitor": self.training_config.scheduler_monitor,
            "interval": self.training_config.scheduler_interval,
            "frequency": self.training_config.scheduler_frequency,
            "strict": self.training_config.scheduler_strict,
            "name": self.training_config.scheduler_name,
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
