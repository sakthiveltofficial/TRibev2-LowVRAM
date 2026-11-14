"""
tribe_local.fusion
==================
Feature fusion pipeline for the TRIBE-inspired multimodal brain encoder.

Submodules:
    projectors            – ModalityProjectors: per-modality MLP + cat → (B, T, 512)
    temporal_align        – TemporalAligner: interpolate all modalities to common T
    cross_attention_fusion – CrossAttentionBlock, MultimodalTransformerFusion,
                             TribeLocalFusion (top-level, (B,D,T) → (B,1024,T))

Typical usage::

    from tribe_local.fusion import TribeLocalFusion

    fusion = TribeLocalFusion(
        encoder_dims={"video": 384, "audio": 768, "text": 1536},
        hidden=512,
        low_rank=1024,
    )
    out = fusion({"video": v, "audio": a, "text": t})
    # out: (B, 1024, T_common)
"""

from .projectors import ModalityProjectors
from .temporal_align import TemporalAligner
from .cross_attention_fusion import (
    CrossAttentionBlock,
    MultimodalTransformerFusion,
    TribeLocalFusion,
)

__all__ = [
    "ModalityProjectors",
    "TemporalAligner",
    "CrossAttentionBlock",
    "MultimodalTransformerFusion",
    "TribeLocalFusion",
]
