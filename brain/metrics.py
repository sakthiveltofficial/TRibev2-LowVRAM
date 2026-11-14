"""
Brain encoding evaluation metrics — TRIBE v2 compatible.

OnlinePearsonCorr
    Per-vertex Pearson correlation accumulated over mini-batches via sufficient
    statistics (sum_x, sum_y, sum_x2, sum_y2, sum_xy, n).  Identical interface
    to the TRIBE v2 ``OnlinePearsonCorr`` class so it can be swapped in
    directly.

BrainMetrics
    Convenience wrapper that bundles:
      - Overall mean / median Pearson r   (all vertices)
      - Overall R²                        (all vertices, flatten)
      - Per-subject mean Pearson r        (tracks subject_ids across updates)

All statistics are accumulated in fp32 regardless of the input dtype so that
FP16 training does not cause overflow / underflow in the running sums.

Shapes throughout::

    pred   : (B, n_vertices, T)
    target : (B, n_vertices, T)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor


class OnlinePearsonCorr(nn.Module):
    """Streaming per-vertex Pearson correlation metric.

    Accumulates sufficient statistics across mini-batches so that the full
    dataset correlation can be computed without storing all predictions in
    memory.

    Usage::

        metric = OnlinePearsonCorr(n_vertices=20484)
        for pred, target in loader:
            metric.update(pred, target)   # (B, 20484, T)
        r = metric.compute()              # (20484,)
        metric.reset()

    Args:
        n_vertices : Number of cortical vertices (default 20484).
    """

    def __init__(self, n_vertices: int = 20484, dim: int = 0, **kwargs) -> None:
        super().__init__()
        self.n_vertices = n_vertices
        self.dim = dim
        # Sufficient statistics — always fp32
        self.register_buffer("sum_x",  torch.zeros(n_vertices))
        self.register_buffer("sum_y",  torch.zeros(n_vertices))
        self.register_buffer("sum_x2", torch.zeros(n_vertices))
        self.register_buffer("sum_y2", torch.zeros(n_vertices))
        self.register_buffer("sum_xy", torch.zeros(n_vertices))
        self.register_buffer("n",      torch.zeros(1))

    # ------------------------------------------------------------------
    def update(self, pred: Tensor, target: Tensor) -> None:
        """Accumulate statistics for one mini-batch.

        Args:
            pred   : (B, n_vertices, T)
            target : (B, n_vertices, T)
        """
        # Flatten (B, V, T) → (B*T, V) for per-vertex accumulation
        B, V, T = pred.shape
        x = pred.float().permute(0, 2, 1).reshape(B * T, V)    # (B*T, V)
        y = target.float().permute(0, 2, 1).reshape(B * T, V)  # (B*T, V)

        self.sum_x  += x.sum(dim=0)       # (V,)
        self.sum_y  += y.sum(dim=0)
        self.sum_x2 += x.pow(2).sum(dim=0)
        self.sum_y2 += y.pow(2).sum(dim=0)
        self.sum_xy += (x * y).sum(dim=0)
        self.n      += B * T

    # ------------------------------------------------------------------
    def compute(self) -> Tensor:
        """Compute per-vertex Pearson r from accumulated statistics.

        Returns:
            Tensor of shape (n_vertices,) in float32.
        """
        n = self.n
        # Pearson r = (n*sum_xy - sum_x*sum_y) /
        #             sqrt((n*sum_x2 - sum_x²) * (n*sum_y2 - sum_y²))
        numerator   = n * self.sum_xy  - self.sum_x  * self.sum_y
        denom_x     = n * self.sum_x2 - self.sum_x.pow(2)
        denom_y     = n * self.sum_y2 - self.sum_y.pow(2)
        denominator = (denom_x.clamp(min=0.0) * denom_y.clamp(min=0.0)).sqrt().add_(1e-8)
        r = (numerator / denominator).clamp(-1.0, 1.0)
        return r  # (n_vertices,)

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all accumulated statistics."""
        self.sum_x.zero_()
        self.sum_y.zero_()
        self.sum_x2.zero_()
        self.sum_y2.zero_()
        self.sum_xy.zero_()
        self.n.zero_()


# ---------------------------------------------------------------------------


class BrainMetrics:
    """Composite evaluation metrics for brain encoding.

    Bundles:
        - ``OnlinePearsonCorr`` for overall per-vertex Pearson r
        - Mean / median Pearson r (scalar summaries)
        - R² (coefficient of determination, computed on flattened arrays)
        - Per-subject mean Pearson r tracked via ``subject_ids``

    Usage::

        metrics = BrainMetrics(n_vertices=20484)
        for pred, target, subject_ids in loader:
            metrics.update(pred, target, subject_ids)
        results = metrics.compute()
        # {"mean_pearson": ..., "median_pearson": ..., "r2": ...,
        #  "subj_0_pearson": ..., "subj_1_pearson": ..., ...}
        metrics.reset()

    Args:
        n_vertices : Number of cortical vertices (default 20484).
    """

    def __init__(self, n_vertices: int = 20484) -> None:
        self.n_vertices = n_vertices
        # Global Pearson accumulator
        self._pearson = OnlinePearsonCorr(n_vertices=n_vertices)

        # R² accumulators: sum of squared residuals and total variance
        self._ss_res: float = 0.0
        self._ss_tot: float = 0.0

        # Per-subject Pearson accumulators keyed by subject id (int)
        self._subj_pearson: Dict[int, OnlinePearsonCorr] = {}

    # ------------------------------------------------------------------
    def _get_or_create_subj(self, sid: int) -> OnlinePearsonCorr:
        if sid not in self._subj_pearson:
            self._subj_pearson[sid] = OnlinePearsonCorr(n_vertices=self.n_vertices)
            # Move to same device as global accumulator
            device = self._pearson.sum_x.device
            self._subj_pearson[sid] = self._subj_pearson[sid].to(device)
        return self._subj_pearson[sid]

    # ------------------------------------------------------------------
    def update(self, pred: Tensor, target: Tensor, subject_ids: Tensor) -> None:
        """Accumulate statistics for one mini-batch.

        Args:
            pred        : (B, n_vertices, T) — model predictions (fp16/fp32)
            target      : (B, n_vertices, T) — fMRI ground truth
            subject_ids : (B,)               — integer subject indices
        """
        pred_f   = pred.float()
        target_f = target.float()

        # --- Global Pearson
        self._pearson.update(pred_f, target_f)

        # --- R²: accumulate over all elements (B, V, T)
        res = (target_f - pred_f).pow(2).sum().item()
        mean_target = target_f.mean()
        tot = (target_f - mean_target).pow(2).sum().item()
        self._ss_res += res
        self._ss_tot += tot

        # --- Per-subject Pearson
        unique_ids = subject_ids.unique().tolist()
        for sid in unique_ids:
            sid_int = int(sid)
            mask = (subject_ids == sid_int)              # (B,) bool
            pred_s   = pred_f[mask]                      # (B_s, V, T)
            target_s = target_f[mask]
            self._get_or_create_subj(sid_int).update(pred_s, target_s)

    # ------------------------------------------------------------------
    def compute(self) -> dict[str, float]:
        """Compute all metrics from accumulated statistics.

        Returns:
            dict with keys:
                ``"mean_pearson"``   — mean Pearson r across all vertices (float)
                ``"median_pearson"`` — median Pearson r (float)
                ``"r2"``             — coefficient of determination (float)
                ``"subj_{id}_pearson"`` — mean Pearson r for each seen subject
        """
        r_all = self._pearson.compute()          # (n_vertices,)

        mean_r   = float(r_all.mean().item())
        median_r = float(r_all.median().item())

        r2 = 1.0 - (self._ss_res / (self._ss_tot + 1e-8))

        results: dict[str, float] = {
            "mean_pearson":   mean_r,
            "median_pearson": median_r,
            "r2":             r2,
        }

        # Per-subject
        for sid, acc in sorted(self._subj_pearson.items()):
            r_subj = acc.compute()
            results[f"subj_{sid}_pearson"] = float(r_subj.mean().item())

        return results

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Zero all accumulators."""
        self._pearson.reset()
        self._ss_res = 0.0
        self._ss_tot = 0.0
        for acc in self._subj_pearson.values():
            acc.reset()
        self._subj_pearson.clear()

    # ------------------------------------------------------------------
    def to(self, device: torch.device | str) -> "BrainMetrics":
        """Move internal buffers to ``device``."""
        self._pearson = self._pearson.to(device)
        for acc in self._subj_pearson.values():
            acc.to(device)
        return self
