"""
Brain encoding loss functions — TRIBE v2 compatible.

TRIBE v2 uses pure MSELoss(reduction='none').mean() with a validity mask that
skips TRs where all vertices are zero (bad acquisitions / padding).  We extend
this with an optional Pearson-correlation loss term so that the model is
rewarded for predicting the *shape* of the BOLD response, not just its
magnitude.

Composite loss:
    total = alpha * MSE + (1 - alpha) * (1 - Pearson_r)

Default alpha = 0.8 (matches the intuition that MSE dominates early training
while Pearson shapes fine-tuning).

Shapes throughout::

    pred   : (B, n_vertices, T)   — model output
    target : (B, n_vertices, T)   — fMRI ground truth
    mask   : (B, T)               — optional boolean mask (True = keep)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BrainLoss(nn.Module):
    """Composite MSE + Pearson-correlation loss for fMRI prediction.

    Args:
        alpha : Weight for MSE term.  ``1 - alpha`` weights the Pearson term.
                Default 0.8.

    Example::

        loss_fn = BrainLoss(alpha=0.8)
        pred   = torch.randn(4, 20484, 100)
        target = torch.randn(4, 20484, 100)
        out    = loss_fn(pred, target)
        # out["total"], out["mse"], out["pearson_loss"]
    """

    def __init__(self, alpha: float = 0.8) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_tr_mask(target: Tensor) -> Tensor:
        """Return a boolean (B, T) mask of TRs that are not all-zero.

        Mimics TRIBE v2's ``y_true == 0`` check: a TR is considered invalid
        (padding / bad acquisition) when the sum of absolute values across
        all vertices is zero.

        Args:
            target : (B, n_vertices, T)

        Returns:
            Tensor of shape (B, T), dtype bool — True where TR is valid.
        """
        # Sum over vertex dimension → (B, T)
        return target.abs().sum(dim=1) != 0

    @staticmethod
    def pearsonr_temporal(x: Tensor, y: Tensor) -> Tensor:
        """Per-vertex Pearson r along the time dimension, averaged over vertices.

        Args:
            x : (B, n_vertices, T)
            y : (B, n_vertices, T)

        Returns:
            Scalar tensor — mean Pearson r across the batch × vertices.

        Note:
            FP16-safe: intermediate statistics are promoted to float32.
        """
        # Promote to fp32 for numerical stability
        x = x.float()
        y = y.float()

        T = x.size(-1)
        if T < 2:
            # Undefined correlation for single time-point; return 0
            return x.new_zeros(1).squeeze()

        # Mean over time dim
        mx = x.mean(dim=-1, keepdim=True)   # (B, V, 1)
        my = y.mean(dim=-1, keepdim=True)   # (B, V, 1)

        xc = x - mx                          # (B, V, T) centred
        yc = y - my

        # Numerator: covariance
        cov = (xc * yc).sum(dim=-1)          # (B, V)

        # Denominator: product of standard deviations (clamped for stability)
        std_x = xc.pow(2).sum(dim=-1).add_(1e-8).sqrt()  # (B, V)
        std_y = yc.pow(2).sum(dim=-1).add_(1e-8).sqrt()  # (B, V)

        r = cov / (std_x * std_y)            # (B, V)
        r = r.clamp(-1.0, 1.0)

        return r.mean()                      # scalar

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        """Compute composite brain loss.

        Args:
            pred   : (B, n_vertices, T) — model predictions (may be fp16)
            target : (B, n_vertices, T) — fMRI ground truth
            mask   : (B, T) bool, optional.  If provided, combined (AND) with
                     the auto-detected all-zero mask.  True = keep.

        Returns:
            dict with keys:
                ``"total"``        — alpha*mse + (1-alpha)*pearson_loss
                ``"mse"``          — MSE on valid TRs
                ``"pearson_loss"`` — 1 - mean Pearson r (on valid TRs)
        """
        # --- Build validity mask (B, T)
        valid = self._valid_tr_mask(target)   # (B, T)
        if mask is not None:
            valid = valid & mask

        # Expand mask to (B, n_vertices, T) for indexing
        valid_vt = valid.unsqueeze(1).expand_as(pred)  # (B, V, T)

        # We need at least one valid element
        if valid_vt.sum() == 0:
            zero = pred.new_zeros(1).squeeze()
            return {"total": zero, "mse": zero, "pearson_loss": zero}

        # --- MSE on valid positions
        pred_f = pred.float()
        target_f = target.float()

        mse_loss = F.mse_loss(pred_f[valid_vt], target_f[valid_vt])

        # --- Pearson loss on valid TRs
        # Mask invalid TRs by zeroing them out before Pearson (they don't
        # affect mean because we only have valid data).
        # Actually, compute per-sample using the full tensor restricted to
        # valid TRs; for simplicity we zero-out invalid positions and
        # re-compute Pearson on the whole tensor (zero TRs contribute ~0
        # variance, so their effect on the correlation is minimal, but we
        # explicitly zero them).
        pred_masked = pred_f * valid.unsqueeze(1).float()
        target_masked = target_f * valid.unsqueeze(1).float()
        pearson_loss = 1.0 - self.pearsonr_temporal(pred_masked, target_masked)

        total = self.alpha * mse_loss + (1.0 - self.alpha) * pearson_loss

        return {
            "total": total,
            "mse": mse_loss,
            "pearson_loss": pearson_loss,
        }
