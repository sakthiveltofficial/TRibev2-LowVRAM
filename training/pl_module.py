"""
pl_module.py — PyTorch Lightning Module for TRIBE-local Brain Encoding
======================================================================
Wraps frozen multimodal encoders (or cached features), cross-attention fusion,
and subject-specific brain prediction heads into an integrated LightningModule.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.lr_scheduler import OneCycleLR

from ..brain.losses import BrainLoss
from ..brain.metrics import BrainMetrics
from ..brain.subject_layers import BrainPredictionHead
from ..fusion.cross_attention_fusion import TribeLocalFusion

logger = logging.getLogger(__name__)


class TribeLocalModule(pl.LightningModule):
    """LightningModule for end-to-end multimodal brain encoding.

    Key design principles:
    1. Pretrained unimodal encoders (VideoMAE, AST, DeepSeek) are kept FROZEN.
    2. Only ModalityProjectors, MultimodalTransformerFusion, and BrainPredictionHead
       contain trainable parameters.
    3. Supports direct feature input (from pre-extracted cache) or on-the-fly encoding.
    4. Gradient checkpointing and mixed-precision (FP16) enabled for 8GB VRAM budget.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        video_encoder: Optional[nn.Module] = None,
        audio_encoder: Optional[nn.Module] = None,
        text_encoder: Optional[Any] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.cfg = config

        # 1. Encoders (kept frozen)
        self.video_encoder = video_encoder
        self.audio_encoder = audio_encoder
        self.text_encoder = text_encoder

        # Freeze encoder parameters if provided as nn.Modules
        for enc in (self.video_encoder, self.audio_encoder):
            if enc is not None and isinstance(enc, nn.Module):
                enc.eval()
                for p in enc.parameters():
                    p.requires_grad = False

        # 2. Extract dimensions from config
        enc_cfg = self.cfg.get("encoders", {})
        d_video = enc_cfg.get("video", {}).get("output_dim", 384)
        d_audio = enc_cfg.get("audio", {}).get("output_dim", 768)
        d_text = enc_cfg.get("text", {}).get("output_dim", 1536)

        fusion_cfg = self.cfg.get("fusion", {})
        hidden = fusion_cfg.get("hidden", 512)
        low_rank = fusion_cfg.get("low_rank", 1024)
        n_layers = fusion_cfg.get("n_transformer_layers", 4)
        n_heads = fusion_cfg.get("n_heads", 8)
        dropout = fusion_cfg.get("dropout", 0.1)
        modality_dropout = fusion_cfg.get("modality_dropout", 0.3)
        max_seq_len = fusion_cfg.get("max_seq_len", 1024)

        # 3. Fusion module
        self.fusion = TribeLocalFusion(
            encoder_dims={"video": d_video, "audio": d_audio, "text": d_text},
            hidden=hidden,
            low_rank=low_rank,
            num_transformer_layers=n_layers,
            num_heads=n_heads,
            dropout=dropout,
            modality_dropout=modality_dropout,
            max_seq_len=max_seq_len,
        )

        # 4. Brain prediction head
        brain_cfg = self.cfg.get("brain_head", {})
        self.head_mode = brain_cfg.get("mode", "voxel")
        n_vertices = brain_cfg.get("n_vertices", 20484)
        n_rois = brain_cfg.get("n_rois", 360)
        n_subjects = brain_cfg.get("n_subjects", 25)
        target_trs = brain_cfg.get("target_trs", 100)
        subj_dropout = brain_cfg.get("subject_dropout", 0.1)

        self.head = BrainPredictionHead(
            mode=self.head_mode,
            r=low_rank,
            n_vertices=n_vertices,
            n_rois=n_rois,
            n_subjects=n_subjects,
            target_trs=target_trs,
            subject_dropout=subj_dropout,
        )

        # 5. Loss & Metrics
        loss_alpha = self.cfg.get("loss", {}).get("alpha", 0.8)
        self.loss_fn = BrainLoss(alpha=loss_alpha)

        target_dim = n_rois if self.head_mode == "roi" else n_vertices
        self.val_metrics = BrainMetrics(n_vertices=target_dim, n_subjects=n_subjects)

        self.cpu_offload = self.cfg.get("hardware", {}).get("cpu_offload_encoders", True)

    def forward(
        self,
        features: Dict[str, Optional[Tensor]],
        subject_ids: Tensor,
    ) -> Tensor:
        """Run fusion and brain head forward pass.

        Args:
            features: dict with keys "video", "audio", "text" → (B, D, T)
            subject_ids: (B,) integer tensor

        Returns:
            fMRI prediction: (B, n_vertices, target_trs) or (B, n_rois, target_trs)
        """
        # 1. Multimodal fusion: (B, 1024, T_common)
        fused = self.fusion(features)

        # 2. Prediction head: (B, n_targets, target_trs)
        pred = self.head(fused, subject_ids)
        return pred

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Tensor:
        features = batch["features"]
        fmri_target = batch["fmri"]
        subject_ids = batch["subject_ids"]

        # Forward pass
        pred = self(features, subject_ids)

        # Compute composite loss (handles zero-masking internally)
        loss_dict = self.loss_fn(pred, fmri_target)
        loss = loss_dict["total"]

        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=len(subject_ids))
        self.log("train/mse", loss_dict["mse"], on_step=False, on_epoch=True, batch_size=len(subject_ids))
        self.log("train/pearson_loss", loss_dict["pearson_loss"], on_step=False, on_epoch=True, batch_size=len(subject_ids))

        return loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        features = batch["features"]
        fmri_target = batch["fmri"]
        subject_ids = batch["subject_ids"]

        pred = self(features, subject_ids)
        loss_dict = self.loss_fn(pred, fmri_target)

        self.log("val/loss", loss_dict["total"], on_epoch=True, prog_bar=True, batch_size=len(subject_ids))
        self.log("val/mse", loss_dict["mse"], on_epoch=True, batch_size=len(subject_ids))

        # Accumulate metrics
        self.val_metrics.update(pred, fmri_target, subject_ids)

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        # Log main Pearson metric (matches TRIBE v2 monitor: val/pearson)
        self.log("val/pearson", metrics["mean_pearson"], prog_bar=True)
        self.log("val/median_pearson", metrics["median_pearson"])
        self.log("val/r2", metrics["r2"])

        # Log individual subject correlations
        for k, v in metrics.items():
            if k.startswith("subj_"):
                self.log(f"val/{k}", v)

        self.val_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        opt_cfg = self.cfg.get("optimizer", {})
        lr = float(opt_cfg.get("lr", 1.0e-4))
        wd = float(opt_cfg.get("weight_decay", 0.0))

        # Collect only parameters that require gradients (fusion + brain head)
        trainable_params = [p for p in self.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=wd)

        sched_cfg = self.cfg.get("scheduler", {})
        sched_name = sched_cfg.get("name", "OneCycleLR")

        if sched_name == "OneCycleLR":
            # Estimate total steps from trainer or max_epochs
            max_epochs = self.cfg.get("training", {}).get("max_epochs", 15)
            # Default fallback steps per epoch = 100 if dataloader not attached yet
            steps_per_epoch = getattr(self.trainer, "estimated_stepping_batches", max_epochs * 100) // max(1, max_epochs)
            total_steps = max_epochs * max(1, steps_per_epoch)

            scheduler = OneCycleLR(
                optimizer,
                max_lr=lr,
                total_steps=total_steps,
                pct_start=float(sched_cfg.get("pct_start", 0.1)),
                div_factor=25.0,
                final_div_factor=1000.0,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        return {"optimizer": optimizer}
