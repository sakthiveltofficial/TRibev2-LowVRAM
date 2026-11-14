"""
tribe_local.fusion.projectors
==============================
Per-modality MLP projectors that map raw encoder features to a shared
hidden dimension and concatenate them into a single fused representation.

Tensor-shape contract
---------------------
Input  (per modality m):  (B, D_m, T)
After transpose:          (B, T, D_m)
After MLP projection:     (B, T, hidden // n_modalities)
After cat (all modalities):(B, T, hidden)         [= (B, T, 512) by default]

FP16-compatible (LayerNorm + Linear are safe in half precision).
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class ModalityProjectors(nn.Module):
    """Project each modality's features into a shared hidden dimension.

    Each modality gets a small MLP::

        Linear(D_m, slot_dim) → GELU → LayerNorm(slot_dim)

    where ``slot_dim = hidden // n_modalities`` for most modalities and the
    last modality absorbs any remainder so that the concatenated output is
    exactly ``hidden`` wide.

    Missing-modality handling
    -------------------------
    If a modality key is absent from the ``forward`` input dict, a learnable
    zero-replacement embedding (shape ``(1, 1, slot_dim)``) is broadcast over
    ``(B, T)`` instead.

    Parameters
    ----------
    input_dims : dict[str, int]
        Mapping from modality name → feature dimension D_m.
        E.g. ``{"video": 384, "audio": 768, "text": 1536}``.
    hidden : int
        Total concatenated hidden size (default 512).
        Must be ≥ n_modalities so that each slot ≥ 1.

    Attributes
    ----------
    modality_names : list[str]
        Sorted list of modality names (deterministic order for cat).
    slot_dims : dict[str, int]
        Per-modality projection width (sums to ``hidden``).
    projectors : nn.ModuleDict
        One MLP per modality.
    missing_embs : nn.ParameterDict
        Learnable replacement embedding per modality, shape (1, 1, slot_dim).
    """

    def __init__(
        self,
        input_dims: Dict[str, int],
        hidden: int = 512,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must not be empty.")

        self.hidden = hidden
        # Deterministic ordering so cat is reproducible
        self.modality_names: list[str] = sorted(input_dims.keys())
        n = len(self.modality_names)

        # Distribute hidden evenly; last modality absorbs remainder
        base_slot = hidden // n
        remainder = hidden - base_slot * n
        self.slot_dims: Dict[str, int] = {}
        for i, name in enumerate(self.modality_names):
            self.slot_dims[name] = base_slot + (1 if i == n - 1 else 0) * remainder

        # Build per-modality MLP blocks
        projectors: Dict[str, nn.Module] = {}
        missing_embs: Dict[str, nn.Parameter] = {}
        for name in self.modality_names:
            d_in = input_dims[name]
            d_out = self.slot_dims[name]
            projectors[name] = nn.Sequential(
                nn.Linear(d_in, d_out),
                nn.GELU(),
                nn.LayerNorm(d_out),
            )
            # Learnable replacement for a fully-missing modality
            missing_embs[name] = nn.Parameter(torch.zeros(1, 1, d_out))

        self.projectors = nn.ModuleDict(projectors)
        self.missing_embs = nn.ParameterDict(missing_embs)

    # ------------------------------------------------------------------
    def forward(self, features: Dict[str, Tensor]) -> Tensor:
        """Project and fuse all modalities.

        Parameters
        ----------
        features : dict[str, Tensor]
            Subset (or all) of the registered modalities.
            Each value has shape ``(B, D_m, T)``.
            Missing modalities are replaced by the learnable zero embedding.

        Returns
        -------
        Tensor
            Fused features of shape ``(B, T, hidden)`` = ``(B, T, 512)``.

        Notes
        -----
        * T is inferred from the first present modality; all present modalities
          are assumed to share the same T (use TemporalAligner first).
        * The learned missing embedding is broadcast over B and T.
        """
        # Discover B and T from the first present modality
        b, t = None, None
        for name in self.modality_names:
            if name in features:
                b, _, t = features[name].shape
                break
        if b is None:
            raise ValueError("At least one modality must be present in `features`.")

        projected_parts: list[Tensor] = []
        for name in self.modality_names:
            if name in features:
                # (B, D_m, T) → (B, T, D_m) → MLP → (B, T, slot_dim)
                x = features[name].transpose(1, 2)          # (B, T, D_m)
                x = self.projectors[name](x)                # (B, T, slot_dim)
            else:
                # Broadcast learnable zero embedding: (1, 1, slot_dim) → (B, T, slot_dim)
                emb = self.missing_embs[name]               # (1, 1, slot_dim)
                x = emb.expand(b, t, -1)                    # (B, T, slot_dim)
            projected_parts.append(x)

        # Concatenate along feature dim → (B, T, hidden)
        fused = torch.cat(projected_parts, dim=-1)         # (B, T, 512)
        return fused
