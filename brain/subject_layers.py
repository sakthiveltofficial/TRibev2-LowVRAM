"""
Subject-specific prediction heads for TRIBE-inspired brain encoding.

Memory-efficient alternative to TRIBE v2's giant SubjectLayers parameter
  (n_subjects, r=2048, n_vertices=20484) which requires ~4 GB for 25 subjects.

Shapes throughout (our lighter r=1024 variant):
    Fusion output  : (B, 1024,     T)
    fMRI target    : (B, 20484,    100)   — fsaverage5, 100 TRs/segment
    ROI target     : (B, 360,      100)   — HCP-MMP1 parcellation
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class LightweightSubjectHead(nn.Module):
    """Memory-efficient TRIBE-compatible vertex-level prediction head.

    Replaces TRIBE v2's monolithic ``nn.Parameter(n_subjects, r, n_vertices)``
    with a decomposed design:
      - One shared decoder (r → n_vertices)  saved ~95 % of the parameter count.
      - Per-subject style embeddings (n_subjects, 128) that modulate the
        shared decoder via FiLM-like scale + offset in the r-dim bottleneck.

    Args:
        r            : Bottleneck dimension from fusion model (default 1024).
        n_vertices   : Number of cortical vertices (default 20484 = 2×10242,
                       fsaverage5 hemispheres).
        n_subjects   : Dataset subjects to track (default 25).
        target_trs   : Target TR count for temporal pooling (default 100).
        subject_dropout: Probability of zeroing the per-subject offset during
                       training (regularises subject-specificity, default 0.1).

    Forward shapes::

        x           : (B, r, T)       — fusion encoder output
        subject_ids : (B,)            — integer indices in [0, n_subjects)
        → output    : (B, n_vertices, target_trs)
    """

    def __init__(
        self,
        r: int = 1024,
        n_vertices: int = 20484,
        n_subjects: int = 25,
        target_trs: int = 100,
        subject_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.r = r
        self.n_vertices = n_vertices
        self.n_subjects = n_subjects
        self.target_trs = target_trs
        self.subject_dropout_p = subject_dropout

        # --- Shared decoder: (r → n_vertices), no bias (bias absorbed by subject_bias)
        self.shared_decoder = nn.Linear(r, n_vertices, bias=False)

        # --- Per-subject style components
        # Embedding → compact 128-d style vector per subject
        self.subject_embeddings = nn.Embedding(n_subjects, 128)
        # Adapter: style → bottleneck scale offset
        self.subject_adapter = nn.Sequential(
            nn.Linear(128, r),
            nn.GELU(),
        )
        # FiLM-like scale per subject in bottleneck dim
        self.subject_scale = nn.Parameter(torch.ones(n_subjects, r))
        # Per-subject additive bias in vertex space
        self.subject_bias = nn.Parameter(torch.zeros(n_subjects, n_vertices))

        # Temporal alignment
        self.pooler = nn.AdaptiveAvgPool1d(target_trs)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.xavier_uniform_(self.shared_decoder.weight)
        nn.init.normal_(self.subject_embeddings.weight, std=0.02)
        for layer in self.subject_adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor, subject_ids: Tensor) -> Tensor:
        """
        Args:
            x           : (B, r, T)   fusion encoder output (may be fp16)
            subject_ids : (B,)        integer subject indices

        Returns:
            Tensor of shape (B, n_vertices, target_trs)
        """
        # --- Subject style modulation
        subject_emb = self.subject_embeddings(subject_ids)        # (B, 128)
        subject_offset = self.subject_adapter(subject_emb)        # (B, r)
        subject_sc = self.subject_scale[subject_ids]              # (B, r)

        # Subject dropout: zero out offsets during training
        if self.training and self.subject_dropout_p > 0.0:
            drop_mask = torch.bernoulli(
                torch.full(
                    (subject_offset.size(0), 1),
                    1.0 - self.subject_dropout_p,
                    device=subject_offset.device,
                    dtype=subject_offset.dtype,
                )
            )  # (B, 1)
            subject_offset = subject_offset * drop_mask

        # FiLM: scale + offset in bottleneck, broadcast over T
        # (B, r, T) = (B, r, T) * (B, r, 1) + (B, r, 1)
        x_subj = x * subject_sc.unsqueeze(-1) + subject_offset.unsqueeze(-1)  # (B, r, T)

        # Decode vertices
        x_subj = x_subj.transpose(1, 2)                          # (B, T, r)
        pred = self.shared_decoder(x_subj)                       # (B, T, n_vertices)

        # Add per-subject bias (broadcast over T)
        pred = pred + self.subject_bias[subject_ids].unsqueeze(1)  # (B, T, n_vertices)

        # Reshape and pool to target TRs
        pred = pred.transpose(1, 2)                              # (B, n_vertices, T)
        pred = self.pooler(pred)                                  # (B, n_vertices, target_trs)
        return pred


class ROIHead(nn.Module):
    """ROI-level (parcellation) prediction head.

    Predicts activity for each of ``n_rois`` ROIs (HCP-MMP1 = 360 by default)
    rather than individual vertices. Useful as an auxiliary head or for
    datasets with ROI-level ground truth.

    Args:
        r          : Bottleneck dimension (default 1024).
        n_rois     : Number of ROIs (default 360, HCP-MMP1 parcellation).
        n_subjects : Number of subjects (default 25).
        target_trs : Target TR count (default 100).

    Forward shapes::

        x           : (B, r, T)       — fusion encoder output
        subject_ids : (B,)            — integer subject indices
        → output    : (B, n_rois, target_trs)
    """

    def __init__(
        self,
        r: int = 1024,
        n_rois: int = 360,
        n_subjects: int = 25,
        target_trs: int = 100,
    ) -> None:
        super().__init__()
        self.r = r
        self.n_rois = n_rois
        self.n_subjects = n_subjects
        self.target_trs = target_trs

        self.roi_decoder = nn.Linear(r, n_rois)
        self.subject_roi_bias = nn.Parameter(torch.zeros(n_subjects, n_rois))
        self.pooler = nn.AdaptiveAvgPool1d(target_trs)

        nn.init.xavier_uniform_(self.roi_decoder.weight)
        nn.init.zeros_(self.roi_decoder.bias)

    def forward(self, x: Tensor, subject_ids: Tensor) -> Tensor:
        """
        Args:
            x           : (B, r, T)   fusion encoder output (may be fp16)
            subject_ids : (B,)        integer subject indices

        Returns:
            Tensor of shape (B, n_rois, target_trs)
        """
        x = x.transpose(1, 2)                                   # (B, T, r)
        pred = self.roi_decoder(x)                              # (B, T, n_rois)
        pred = pred + self.subject_roi_bias[subject_ids].unsqueeze(1)  # (B, T, n_rois)
        pred = pred.transpose(1, 2)                             # (B, n_rois, T)
        pred = self.pooler(pred)                                # (B, n_rois, target_trs)
        return pred


class BrainPredictionHead(nn.Module):
    """Unified brain prediction head wrapping vertex-level and ROI-level heads.

    Args:
        mode         : ``"voxel"`` for LightweightSubjectHead (default),
                       ``"roi"`` for ROIHead, or ``"both"`` to run both.
        r            : Bottleneck dimension (default 1024).
        n_vertices   : Number of cortical vertices (default 20484).
        n_rois       : Number of ROIs (default 360).
        n_subjects   : Number of subjects (default 25).
        target_trs   : Target TR count (default 100).
        subject_dropout : Subject dropout probability (default 0.1).

    Forward shapes::

        x           : (B, r, T)
        subject_ids : (B,)
        → "voxel"   : (B, n_vertices, target_trs)
        → "roi"     : (B, n_rois, target_trs)
        → "both"    : dict{"voxel": ..., "roi": ...}
    """

    def __init__(
        self,
        mode: str = "voxel",
        r: int = 1024,
        n_vertices: int = 20484,
        n_rois: int = 360,
        n_subjects: int = 25,
        target_trs: int = 100,
        subject_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        assert mode in ("voxel", "roi", "both"), f"mode must be 'voxel', 'roi', or 'both', got {mode!r}"
        self.mode = mode

        if mode in ("voxel", "both"):
            self.voxel_head = LightweightSubjectHead(
                r=r,
                n_vertices=n_vertices,
                n_subjects=n_subjects,
                target_trs=target_trs,
                subject_dropout=subject_dropout,
            )
        if mode in ("roi", "both"):
            self.roi_head = ROIHead(
                r=r,
                n_rois=n_rois,
                n_subjects=n_subjects,
                target_trs=target_trs,
            )

    def forward(
        self, x: Tensor, subject_ids: Tensor
    ) -> Tensor | dict[str, Tensor]:
        """
        Args:
            x           : (B, r, T)   fusion encoder output
            subject_ids : (B,)        integer subject indices

        Returns:
            Tensor (voxel or roi mode) or dict with keys "voxel" and "roi" (both mode).
        """
        if self.mode == "voxel":
            return self.voxel_head(x, subject_ids)
        elif self.mode == "roi":
            return self.roi_head(x, subject_ids)
        else:  # "both"
            return {
                "voxel": self.voxel_head(x, subject_ids),
                "roi": self.roi_head(x, subject_ids),
            }
