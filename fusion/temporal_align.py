"""
tribe_local.fusion.temporal_align
==================================
Temporal alignment utilities for multimodal encoder outputs that have
different time resolutions.

Classes
-------
TemporalAligner
    Resamples all modality tensors ``(B, D, T_m)`` to a common T using
    linear interpolation (``F.interpolate``).

Functions
---------
apply_hrf_offset
    Shifts fMRI targets forward by ``offset_steps`` time-steps to account
    for the haemodynamic-response-function (HRF) lag (~5 s at 2 Hz).

FP16-compatible: ``F.interpolate`` operates on float; inputs are cast
temporarily when needed.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
import torch.nn as nn


class TemporalAligner(nn.Module):
    """Align per-modality tensors to a shared temporal resolution.

    Resampling strategy
    -------------------
    ``F.interpolate`` with ``mode='linear'`` is applied independently per
    modality.  The tensor is treated as a *1-D signal* of length ``T_m``
    with ``D`` channels (batch size absorbed into the leading dimension so
    that interpolate sees ``(B*D, 1, T_m)``).

    Parameters
    ----------
    None (stateless; all configuration is passed at forward-time).

    Forward signature
    -----------------
    forward(features, target_t=None) → dict[str, Tensor]
        * ``features``:  dict mapping modality name → ``(B, D, T_m)``
        * ``target_t``:  desired output time length.  If ``None``, uses
                         ``max(T_m)`` across all present modalities.

    Returns
    -------
    dict[str, Tensor]
        Same keys as input; each value has shape ``(B, D, target_t)``.

    Notes
    -----
    * Modalities whose ``T_m == target_t`` are returned as-is (no copy).
    * ``F.interpolate`` requires ``float32`` or ``float64`` input.
      Half-precision tensors are cast, interpolated, then cast back so the
      rest of the pipeline can remain in FP16.
    """

    def forward(
        self,
        features: Dict[str, Tensor],
        target_t: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Resample all modalities to ``target_t`` time steps.

        Parameters
        ----------
        features : dict[str, Tensor]
            Mapping of modality name → ``(B, D, T_m)``.
        target_t : int, optional
            Target temporal length.  Defaults to ``max(T_m)`` if ``None``.

        Returns
        -------
        dict[str, Tensor]
            Aligned tensors; each ``(B, D, target_t)``.
        """
        # Filter out None values for missing modalities
        valid_features = {k: v for k, v in features.items() if v is not None}
        if not valid_features:
            return {}

        # Determine target_t from present modalities
        if target_t is None:
            target_t = max(v.shape[2] for v in valid_features.values())

        aligned: Dict[str, Tensor] = {}
        for name, x in valid_features.items():
            # x: (B, D, T_m)
            b, d, t_m = x.shape
            if t_m == target_t:
                aligned[name] = x
                continue

            # F.interpolate needs float input; handle fp16 gracefully
            original_dtype = x.dtype
            if original_dtype == torch.float16:
                x = x.float()

            # Reshape to (B*D, 1, T_m) for 1-D linear interpolation
            x_flat = x.reshape(b * d, 1, t_m)
            x_interp = F.interpolate(
                x_flat,
                size=target_t,
                mode="linear",
                align_corners=False,
            )  # (B*D, 1, target_t)

            x_out = x_interp.reshape(b, d, target_t)

            if original_dtype == torch.float16:
                x_out = x_out.half()

            aligned[name] = x_out

        return aligned

    # ------------------------------------------------------------------
    @staticmethod
    def apply_hrf_offset(fmri: Tensor, offset_steps: int = 10) -> Tensor:
        """Shift fMRI target tensor to account for the HRF lag.

        The haemodynamic response peaks ~5 s after neural onset.  At a 2 Hz
        feature sampling rate this corresponds to ``offset_steps = 10``
        steps.  We roll the fMRI target *forward* by ``offset_steps`` so
        that feature step ``t`` is paired with fMRI step ``t + offset``.

        Parameters
        ----------
        fmri : Tensor
            fMRI target tensor of shape ``(B, n_vertices, T)`` or any shape
            whose last dimension is time.
        offset_steps : int
            Number of time-steps to shift (default 10 ≈ 5 s at 2 Hz).

        Returns
        -------
        Tensor
            Shifted tensor; same shape as input.  Time-steps rolled with
            wrap-around (``torch.roll``).  The wrapped boundary steps are
            typically masked out by the loss's valid-TR mask.

        Notes
        -----
        Wrapping is used rather than zero-padding so that the shape stays
        constant; callers should ignore the first ``offset_steps`` or last
        ``offset_steps`` time-steps in the loss computation depending on
        convention (here we roll *left* so the *last* ``offset_steps`` TRs
        wrap and should be masked).

        Example
        -------
        >>> aligner = TemporalAligner()
        >>> fmri_shifted = aligner.apply_hrf_offset(fmri, offset_steps=10)
        """
        # Roll along the last (time) dimension
        return torch.roll(fmri, shifts=-offset_steps, dims=-1)
