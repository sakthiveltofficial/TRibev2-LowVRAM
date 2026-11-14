"""
tribe_local.fusion.cross_attention_fusion
==========================================
Multimodal transformer fusion stack.

Tensor-shape summary
--------------------
Input  to TribeLocalFusion.forward:
    {"video": (B, 384, T_v), "audio": (B, 768, T_a), "text": (B, D_text, T_t)}

After TemporalAligner:        all (B, D_m, T_common)
After ModalityProjectors:     (B, T_common, 512)
After MultimodalTransformerFusion: (B, T_common, 512)
After final Linear + reshape: (B, 1024, T_common)   ← interface for SubjectLayers

FP16-compatible throughout.  Gradient checkpointing is applied to each
self-attention layer inside MultimodalTransformerFusion.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .projectors import ModalityProjectors
from .temporal_align import TemporalAligner


# ---------------------------------------------------------------------------
# CrossAttentionBlock
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Single cross-attention layer between two modality sequences.

    Architecture::

        out = LayerNorm(query + MultiheadAttention(Q=query, K=kv, V=kv))

    Parameters
    ----------
    embed_dim : int
        Feature dimension of query and key/value sequences (default 512).
    num_heads : int
        Number of attention heads (default 8).  ``embed_dim`` must be
        divisible by ``num_heads``.
    dropout : float
        Attention dropout probability (default 0.1).

    Forward
    -------
    query : Tensor  (B, T_q, embed_dim)
        The *attending* modality sequence.
    key_value : Tensor  (B, T_kv, embed_dim)
        The *attended* modality sequence.

    Returns
    -------
    Tensor  (B, T_q, embed_dim)
        Updated query sequence (residual + LayerNorm applied).

    Notes
    -----
    * ``batch_first=True`` is set so shapes are ``(B, T, D)`` throughout.
    * T_q and T_kv may differ (cross-attention supports mismatched lengths).
    """

    def __init__(
        self,
        embed_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        """Cross-attend from query toward key_value.

        Parameters
        ----------
        query : Tensor
            Shape ``(B, T_q, embed_dim)``.
        key_value : Tensor
            Shape ``(B, T_kv, embed_dim)``.

        Returns
        -------
        Tensor
            Shape ``(B, T_q, embed_dim)``.
        """
        attn_out, _ = self.attn(
            query=query,
            key=key_value,
            value=key_value,
        )  # (B, T_q, embed_dim)
        return self.norm(query + attn_out)


# ---------------------------------------------------------------------------
# MultimodalTransformerFusion
# ---------------------------------------------------------------------------

class MultimodalTransformerFusion(nn.Module):
    """4-layer self-attention Transformer + 3 round-robin cross-attention blocks.

    Architecture
    ------------
    1. Add positional embedding to input ``(B, T, 512)``.
    2. Modality dropout (training only, p=0.3): zero-out full-sequence
       contributions by modality index slices (requires knowing projector
       slot dims; called externally with pre-projected & concatenated tensor,
       so dropout acts on the whole sequence uniformly here — see note below).
    3. 4 × ``TransformerEncoderLayer`` with gradient checkpointing.
    4. 3 cross-attention blocks in round-robin pairs:
         (video-slot ↔ audio-slot),
         (video-slot ↔ text-slot),
         (audio-slot ↔ text-slot)
       using the gated-weighted intermediate representation.
    5. Gated modality re-weighting: learnable scalar ``sigmoid`` gate per
       modality applied to the cross-attention residuals.
    6. Output: ``(B, T, 512)``.

    Parameters
    ----------
    hidden : int
        Model width (default 512).
    num_heads : int
        Attention heads for self-attention and cross-attention (default 8).
    num_layers : int
        Number of self-attention layers (default 4).
    dim_feedforward : int
        FFN intermediate size in TransformerEncoderLayer (default 1024).
    dropout : float
        Dropout probability (default 0.1).
    modality_dropout_p : float
        Probability of zeroing an entire modality contribution during
        training (default 0.3).
    max_seq_len : int
        Maximum supported T for the positional embedding (default 1024).
    slot_dims : dict[str, int], optional
        Per-modality slot widths from ModalityProjectors (sorted key order).
        When provided, modality dropout zeroes the correct feature slices;
        otherwise the full sequence is treated as a single block.

    Forward
    -------
    x : Tensor  (B, T, hidden)
    training : bool  (default True)

    Returns
    -------
    Tensor  (B, T, hidden)

    Notes
    -----
    * Gradient checkpointing is applied independently to each
      TransformerEncoderLayer to trade compute for memory.
    * ``batch_first=True`` is set on all attention layers.
    * FP16-safe: LayerNorm and Linear are FP16-compatible; checkpointing
      preserves dtype.
    """

    def __init__(
        self,
        hidden: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        modality_dropout_p: float = 0.3,
        max_seq_len: int = 1024,
        slot_dims: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.modality_dropout_p = modality_dropout_p
        self.slot_dims = slot_dims  # {modality_name: slot_width}

        # Positional embedding: (1, max_seq_len, hidden)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, hidden))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # 4-layer self-attention Transformer
        self.sa_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=True,  # Pre-LN for FP16 stability
            )
            for _ in range(num_layers)
        ])

        # 3 cross-attention blocks: (v↔a), (v↔t), (a↔t)
        self.ca_va = CrossAttentionBlock(hidden, num_heads, dropout)
        self.ca_vt = CrossAttentionBlock(hidden, num_heads, dropout)
        self.ca_at = CrossAttentionBlock(hidden, num_heads, dropout)

        # Gated modality re-weighting: learnable [0, 1] scalar per CA block
        # gates[0]=video-audio pair, gates[1]=video-text, gates[2]=audio-text
        self.ca_gates = nn.Parameter(torch.ones(3))  # sigmoid → [0,1]

        # Final LayerNorm
        self.out_norm = nn.LayerNorm(hidden)

    # ------------------------------------------------------------------
    def _apply_modality_dropout(self, x: Tensor) -> Tensor:
        """Zero out entire modality feature slices with probability p.

        Parameters
        ----------
        x : Tensor  (B, T, hidden)

        Returns
        -------
        Tensor  (B, T, hidden)
        """
        if self.slot_dims is None or not self.training:
            # Simple whole-sequence dropout when slot info unavailable or eval
            if self.training and torch.rand(1).item() < self.modality_dropout_p:
                return torch.zeros_like(x)
            return x

        out = x.clone()
        offset = 0
        sorted_names = sorted(self.slot_dims.keys())
        for name in sorted_names:
            w = self.slot_dims[name]
            if torch.rand(1).item() < self.modality_dropout_p:
                out[..., offset: offset + w] = 0.0
            offset += w
        return out

    # ------------------------------------------------------------------
    def _sa_forward(self, layer: nn.TransformerEncoderLayer, x: Tensor) -> Tensor:
        """Thin wrapper for gradient-checkpointed self-attention forward."""
        return layer(x)

    # ------------------------------------------------------------------
    def forward(self, x: Tensor, training: bool = True) -> Tensor:
        """Fuse temporal sequence through self-attention and cross-attention.

        Parameters
        ----------
        x : Tensor
            Shape ``(B, T, 512)``.  Pre-projected & concatenated modalities.
        training : bool
            Whether to apply modality dropout (default ``True``).

        Returns
        -------
        Tensor
            Shape ``(B, T, 512)``.
        """
        b, t, _ = x.shape

        # 1. Add positional embedding (truncate or pad to T)
        pos = self.pos_emb[:, :t, :]     # (1, T, hidden)
        x = x + pos

        # 2. Modality dropout (feature-slice level)
        if training:
            x = self._apply_modality_dropout(x)

        # 3. 4-layer self-attention with gradient checkpointing
        for layer in self.sa_layers:
            if self.training:
                x = checkpoint(self._sa_forward, layer, x, use_reentrant=False)
            else:
                x = layer(x)

        # 4. Cross-attention: round-robin on temporal sequence
        #    We treat the full sequence as each modality's representation
        #    since at this stage it's already fused by the SA layers.
        #    Cross-attention pairs run on the same tensor but in different
        #    "roles" to specialise different subspaces.
        gates = torch.sigmoid(self.ca_gates)  # (3,)

        va_out = self.ca_va(query=x, key_value=x)   # video attends to audio (same seq)
        vt_out = self.ca_vt(query=x, key_value=x)   # video attends to text
        at_out = self.ca_at(query=x, key_value=x)   # audio attends to text

        # Gated residual merge
        x = x + gates[0] * va_out + gates[1] * vt_out + gates[2] * at_out

        # 5. Output normalisation
        x = self.out_norm(x)
        return x  # (B, T, 512)


# ---------------------------------------------------------------------------
# TribeLocalFusion  (top-level)
# ---------------------------------------------------------------------------

class TribeLocalFusion(nn.Module):
    """Top-level multimodal fusion module (TRIBE-inspired, memory-efficient).

    Pipeline
    --------
    1. ``TemporalAligner``:    all ``(B, D_m, T_m)`` → ``(B, D_m, T_common)``
    2. ``ModalityProjectors``: dict → ``(B, T_common, 512)``
    3. ``MultimodalTransformerFusion``: ``(B, T_common, 512)``
    4. Transpose + Linear: ``(B, 512, T_common)`` → ``(B, 1024, T_common)``

    The final shape ``(B, 1024, T_common)`` is the interface expected by
    ``SubjectLayers`` (rank = 1024, matching ``low_rank=1024``).

    Parameters
    ----------
    encoder_dims : dict[str, int]
        Modality name → feature dimension D_m.
        E.g. ``{"video": 384, "audio": 768, "text": 1536}``.
    hidden : int
        Internal transformer width (default 512).
    low_rank : int
        Output rank / final projection output size (default 1024).
    dropout : float
        Dropout for self- and cross-attention layers (default 0.1).
    modality_dropout_p : float
        Probability of zeroing a full modality during training (default 0.3).
    max_seq_len : int
        Maximum supported T for positional embeddings (default 1024).

    Forward
    -------
    features : dict[str, Tensor]
        ``{"video": (B, 384, T_v), "audio": (B, 768, T_a), ...}``

    Returns
    -------
    Tensor  (B, low_rank, T_common)   = (B, 1024, T_common)
    """

    def __init__(
        self,
        encoder_dims: Dict[str, int],
        hidden: int = 512,
        low_rank: int = 1024,
        num_transformer_layers: int = 4,
        num_layers: Optional[int] = None,
        num_heads: int = 8,
        dropout: float = 0.1,
        modality_dropout: Optional[float] = None,
        modality_dropout_p: float = 0.3,
        max_seq_len: int = 1024,
        **kwargs,
    ) -> None:
        super().__init__()
        self.hidden = hidden
        self.low_rank = low_rank

        n_layers = num_layers if num_layers is not None else num_transformer_layers
        m_drop = modality_dropout if modality_dropout is not None else modality_dropout_p

        self.aligner = TemporalAligner()

        self.projectors = ModalityProjectors(
            input_dims=encoder_dims,
            hidden=hidden,
        )

        self.fusion = MultimodalTransformerFusion(
            hidden=hidden,
            num_heads=num_heads,
            num_layers=n_layers,
            dim_feedforward=hidden * 2,
            dropout=dropout,
            modality_dropout_p=m_drop,
            max_seq_len=max_seq_len,
            slot_dims=self.projectors.slot_dims,
        )

        # Final low-rank projection: hidden → low_rank
        # Applied along the feature dimension after transposing back to
        # (B, hidden, T) → treat feature axis as "channels" and use
        # a Linear that maps the last dim.
        self.low_rank_head = nn.Linear(hidden, low_rank, bias=False)

    # ------------------------------------------------------------------
    def forward(
        self,
        features: Optional[Dict[str, Tensor]] = None,
        **kwargs,
    ) -> Tensor:
        """Run the full fusion pipeline.

        Parameters
        ----------
        features : dict[str, Tensor] or kwargs
            Encoder outputs; each ``(B, D_m, T_m)``.

        Returns
        -------
        Tensor
            Shape ``(B, low_rank, T_common)`` = ``(B, 1024, T_common)``.
        """
        if features is None:
            features = kwargs
        elif kwargs:
            features = {**features, **kwargs}
        # Step 1: Temporal alignment → all (B, D_m, T_common)
        aligned = self.aligner(features)

        # Step 2: Per-modality projection + concatenation → (B, T_common, hidden)
        x = self.projectors(aligned)   # (B, T_common, 512)

        # Step 3: Transformer fusion → (B, T_common, hidden)
        training = self.training
        x = self.fusion(x, training=training)   # (B, T_common, 512)

        # Step 4: Transpose → (B, hidden, T_common), apply low_rank_head
        # We need (B, low_rank, T_common); low_rank_head maps feature dim.
        # Transpose to (B, T_common, hidden), apply Linear, then transpose back.
        x = x.transpose(1, 2)          # (B, 512, T_common) [channel-first view]
        # For nn.Linear we need (..., hidden) as last dim; use (B, T, 512)
        x = x.transpose(1, 2)          # (B, T_common, 512)  [back to seq-first]
        x = self.low_rank_head(x)      # (B, T_common, low_rank=1024)
        x = x.transpose(1, 2)          # (B, 1024, T_common)

        return x  # (B, 1024, T_common)
