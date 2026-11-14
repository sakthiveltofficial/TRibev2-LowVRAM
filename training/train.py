"""
train.py — Training CLI for TRIBE-local Multimodal Brain Encoder
===============================================================
Entry-point script for training the lightweight fusion and brain prediction
layers with PyTorch Lightning on RTX 5060 Ti (8GB VRAM) or CPU.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from ..data.dataset import TribeLocalDataModule
from .pl_module import TribeLocalModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("tribe_local.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TRIBE-local Brain Encoder")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/local_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for fMRI and stimulus data (overrides config)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["voxel", "roi"],
        default=None,
        help="Target prediction mode: 'voxel' (20484 vertices) or 'roi' (360 ROIs)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (default: 1 for 8GB VRAM)",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Maximum training epochs (default: 15)",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=None,
        help="Enable 16-bit mixed precision (FP16)",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    p = Path(config_path)
    if not p.is_absolute():
        # Look relative to current working directory or tribe_local
        candidate1 = Path.cwd() / p
        candidate2 = Path(__file__).resolve().parent.parent / p
        if candidate1.exists():
            p = candidate1
        elif candidate2.exists():
            p = candidate2

    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    args = parse_args()

    # 1. Load base configuration
    cfg = load_config(args.config)

    # 2. Apply CLI overrides
    if args.data_root:
        cfg["data"]["data_root"] = args.data_root
    if args.mode:
        cfg["brain_head"]["mode"] = args.mode
    if args.batch_size:
        cfg["hardware"]["batch_size"] = args.batch_size
    if args.max_epochs:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.checkpoint_dir:
        cfg["training"]["checkpoint_dir"] = args.checkpoint_dir

    # 3. Set random seed (reproducibility matching TRIBE v2: seed 33)
    seed = cfg.get("seed", 33)
    pl.seed_everything(seed, workers=True)

    # 4. Hardware and Precision Setup
    use_gpu = torch.cuda.is_available() and not args.cpu
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1

    precision = cfg.get("hardware", {}).get("precision", "16-mixed")
    if args.fp16 is not None:
        precision = "16-mixed" if args.fp16 else "32"
    if not use_gpu:
        precision = "32"  # FP16 not natively supported on CPU for all ops

    logger.info(
        f"Initializing TRIBE-local training: device={accelerator}, precision={precision}, "
        f"mode={cfg.get('brain_head', {}).get('mode')}, batch_size={cfg.get('hardware', {}).get('batch_size')}"
    )

    if use_gpu:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Target GPU: {gpu_name} ({total_vram_gb:.2f} GB VRAM)")

    # 5. Initialize DataModule
    batch_size = cfg.get("hardware", {}).get("batch_size", 1)
    num_workers = cfg.get("hardware", {}).get("num_workers", 2)
    data_root = cfg.get("data", {}).get("data_root", "data/")
    cache_dir = cfg.get("data", {}).get("cache_dir", "~/.tribe_cache")
    mode = cfg.get("brain_head", {}).get("mode", "voxel")

    datamodule = TribeLocalDataModule(
        data_root=data_root,
        cache_dir=cache_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        mode=mode,
        seed=seed,
    )
    datamodule.setup()

    # 6. Initialize Model
    model = TribeLocalModule(config=cfg)

    # 7. Setup Checkpointing and Logging
    ckpt_dir = Path(cfg.get("training", {}).get("checkpoint_dir", "checkpoints/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(cfg.get("training", {}).get("log_dir", "logs/"))
    log_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="tribe-{epoch:02d}-{val_pearson:.3f}",
        monitor="val/pearson",
        mode="max",
        save_top_k=3,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")
    early_stop = EarlyStopping(
        monitor="val/pearson",
        patience=cfg.get("training", {}).get("early_stopping_patience", 5),
        mode="max",
    )

    tb_logger = TensorBoardLogger(save_dir=str(log_dir), name="tribe_local")

    # 8. Setup Trainer
    max_epochs = cfg.get("training", {}).get("max_epochs", 15)
    grad_clip = cfg.get("training", {}).get("gradient_clip_val", 1.0)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        max_epochs=max_epochs,
        gradient_clip_val=grad_clip,
        callbacks=[checkpoint_callback, lr_monitor, early_stop],
        logger=tb_logger,
        log_every_n_steps=5,
    )

    # 9. Train!
    logger.info("Starting training loop...")
    trainer.fit(
        model,
        datamodule=datamodule,
        ckpt_path=args.resume_from,
    )

    logger.info(f"Training completed. Best model saved at: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
