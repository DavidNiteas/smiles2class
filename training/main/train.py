"""Train the main model.

    python -m training.main.train --dataset data.pkl --unimol base_unimol \
        --training-config training/main/configs/default.yaml \
        --output runs/main --export

Writes Lightning checkpoints under ``<output>/checkpoints`` and, with
``--export``, a model directory loadable by ``MainClassifier.from_pretrained``:

    <output>/main_model/{config.json, model.safetensors}
    <output>/label_map.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional

import pytorch_lightning as pl
import typer
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from safetensors.torch import save_file

from smiles2class.model import UniClassifyConfig
from training.main.dataset import UniClassifyDataModule
from training.main.lightning import (
    TrainingConfig,
    UniClassifyLightningModule,
    model_config_from_dict,
)

DEFAULT_CONFIG = Path(__file__).with_name("configs") / "default.yaml"

app = typer.Typer(add_completion=False, help="Train the smiles2class main model.")


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"config must be a mapping: {path}")
    return dict(data)


def load_model_config(default: Mapping[str, Any], path: Path | str | None) -> UniClassifyConfig:
    """Merge the default ``model:`` section with an optional override."""
    merged = dict(default)
    if path is not None:
        path = Path(path)
        if path.suffix.lower() == ".json":
            return UniClassifyConfig.from_json_file(str(path))
        data = _read_yaml(path)
        override = data.get("model") if isinstance(data.get("model"), Mapping) else data
        merged.update(dict(override))
    return model_config_from_dict(merged)


def load_training_config(default: Mapping[str, Any], path: Path | str | None) -> TrainingConfig:
    """Read the ``training:`` section (or a flat file) of a YAML config."""
    if path is None:
        return TrainingConfig.from_dict(dict(default))
    data = _read_yaml(path)
    section = data.get("training") if isinstance(data.get("training"), Mapping) else data
    if not section:
        return TrainingConfig.from_dict(dict(default))
    return TrainingConfig.from_dict(dict(section))


def default_sections() -> tuple[dict[str, Any], dict[str, Any]]:
    data = _read_yaml(DEFAULT_CONFIG)
    model = data.get("model") if isinstance(data.get("model"), Mapping) else {}
    training = data.get("training") if isinstance(data.get("training"), Mapping) else data
    return dict(model), dict(training)


def build_trainer(
    training_config: TrainingConfig,
    output: Path,
    *,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    limit_train_batches: Optional[int | float] = None,
    limit_val_batches: Optional[int | float] = None,
    enable_progress_bar: bool = False,
) -> pl.Trainer:
    checkpoint = ModelCheckpoint(
        dirpath=str(output / "checkpoints"),
        filename="model-{epoch:02d}-{val_f1:.4f}",
        monitor="val_f1",
        mode="max",
        save_top_k=1,
        save_last=True,
    )
    return pl.Trainer(
        default_root_dir=str(output),
        accelerator=training_config.accelerator,
        devices=training_config.devices,
        max_epochs=training_config.max_epochs if max_epochs is None else max_epochs,
        max_steps=training_config.max_steps if max_steps is None else max_steps,
        precision=training_config.precision,
        callbacks=[checkpoint],
        logger=False,
        enable_progress_bar=enable_progress_bar,
        enable_model_summary=enable_progress_bar,
        limit_train_batches=limit_train_batches if limit_train_batches is not None else 1.0,
        limit_val_batches=limit_val_batches if limit_val_batches is not None else 1.0,
    )


def export_model(
    module: UniClassifyLightningModule,
    output: Path | str,
    labels: list[str],
) -> Path:
    """Write the inference-side model directory and return it."""
    output = Path(output)
    main_dir = output / "main_model"
    main_dir.mkdir(parents=True, exist_ok=True)
    configs = module.model.configs
    (main_dir / "config.json").write_text(configs.to_json_string(use_diff=False), encoding="utf-8")
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in module.model.state_dict().items()
    }
    save_file(state, str(main_dir / "model.safetensors"))
    (output / "label_map.json").write_text(
        json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output


def run_training(
    datamodule: UniClassifyDataModule,
    model_config: UniClassifyConfig,
    training_config: TrainingConfig,
    *,
    unimol: Path | str | None = None,
    output: Path | str = "runs/main",
    resume: Path | str | None = None,
    max_epochs: Optional[int] = None,
    max_steps: Optional[int] = None,
    limit_train_batches: Optional[int | float] = None,
    limit_val_batches: Optional[int | float] = None,
    export: bool = False,
    labels: Optional[list[str]] = None,
    enable_progress_bar: bool = False,
) -> tuple[UniClassifyLightningModule, pl.Trainer]:
    """Fit ``model_config`` on ``datamodule``; returns ``(module, trainer)``."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(training_config.seed, workers=True)

    datamodule.BatchSize = training_config.batch_size
    datamodule.NumWorkers = training_config.dataloader_num_workers

    if resume is not None:
        module = UniClassifyLightningModule.load_from_checkpoint(str(resume), map_location="cpu")
    elif unimol is not None:
        module = UniClassifyLightningModule.from_unimol(unimol, model_config, training_config)
    else:
        module = UniClassifyLightningModule(model_config, training_config)

    trainer = build_trainer(
        training_config,
        output,
        max_epochs=max_epochs,
        max_steps=max_steps,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
        enable_progress_bar=enable_progress_bar,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=str(resume) if resume is not None else None)

    if export:
        best_path = getattr(trainer.checkpoint_callback, "best_model_path", "") or ""
        exported = module
        if best_path:
            exported = UniClassifyLightningModule.load_from_checkpoint(best_path, map_location="cpu")
        labels = labels or datamodule.labels
        if not labels:
            raise ValueError("no label names available; pass --label-map or rerun prepare.py")
        export_model(exported, output, list(labels))
        typer.echo(f"exported model directory: {output}")
    return module, trainer


@app.command()
def main(
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Pickled datamodule from prepare.py."),
    model_config: Optional[Path] = typer.Option(
        None, "--model-config", "-m", help="Model config (.json or .yaml with a model: section)."
    ),
    training_config: Optional[Path] = typer.Option(
        None, "--training-config", "-t", help="Training YAML (training: section or flat)."
    ),
    unimol: Optional[Path] = typer.Option(
        None, "--unimol", "-u", help="Pre-trained Uni-Mol directory; omitted = random encoder."
    ),
    output: Path = typer.Option(Path("runs/main"), "--output", "-o", help="Run directory."),
    resume: Optional[Path] = typer.Option(
        None, "--resume", "-r", help="Lightning checkpoint to resume from."
    ),
    export: bool = typer.Option(
        False, "--export", "-e", help="Write model_dir/{label_map.json, main_model/}."
    ),
    max_epochs: Optional[int] = typer.Option(None, "--max-epochs"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
):
    model_section, training_section = default_sections()
    configs = load_model_config(model_section, model_config)
    settings = load_training_config(training_section, training_config)

    datamodule = UniClassifyDataModule.from_pickle(str(dataset))
    run_training(
        datamodule,
        configs,
        settings,
        unimol=unimol,
        output=output,
        resume=resume,
        max_epochs=max_epochs,
        max_steps=max_steps,
        export=export,
    )


if __name__ == "__main__":
    app()
