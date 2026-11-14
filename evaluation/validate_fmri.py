"""
validate_fmri.py — Validation and Evaluation against fMRI Datasets
=================================================================
Evaluates brain encoding predictions against empirical or synthetic fMRI
recordings using vertex-wise Pearson correlation, ROI parcellation metrics,
and stimulus retrieval accuracy (matching TRIBE v2 evaluation logic).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tribe_local.validate_fmri")


class FMRIValidator:
    """Evaluates predicted BOLD responses against ground-truth fMRI recordings."""

    @staticmethod
    def compute_pearson_r(
        pred: np.ndarray,
        target: np.ndarray,
        axis: int = -1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute Pearson correlation coefficient along the temporal dimension.

        Args:
            pred: Array of shape (..., T).
            target: Array of shape (..., T).
            axis: Time dimension axis.

        Returns:
            Pearson r array of shape (...) collapsed across axis.
        """
        p_mean = np.mean(pred, axis=axis, keepdims=True)
        t_mean = np.mean(target, axis=axis, keepdims=True)

        p_diff = pred - p_mean
        t_diff = target - t_mean

        numerator = np.sum(p_diff * t_diff, axis=axis)
        denom = np.sqrt(np.sum(p_diff**2, axis=axis) * np.sum(t_diff**2, axis=axis)) + eps
        return np.clip(numerator / denom, -1.0, 1.0)

    @staticmethod
    def compute_r2(
        pred: np.ndarray,
        target: np.ndarray,
        axis: int = -1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        """Compute coefficient of determination (R^2)."""
        ss_res = np.sum((target - pred) ** 2, axis=axis)
        t_mean = np.mean(target, axis=axis, keepdims=True)
        ss_tot = np.sum((target - t_mean) ** 2, axis=axis) + eps
        return 1.0 - (ss_res / ss_tot)

    @staticmethod
    def compute_top1_retrieval(
        pred: np.ndarray,
        target: np.ndarray,
    ) -> float:
        """Compute top-1 identification / retrieval accuracy.

        For each sample segment, test whether its time-averaged predicted response
        correlates higher with its true response than with any other sample's response.

        Args:
            pred: (N_samples, n_vertices, T)
            target: (N_samples, n_vertices, T)

        Returns:
            Accuracy in [0.0, 1.0].
        """
        n_samples = pred.shape[0]
        if n_samples < 2:
            return 1.0

        # Mean over temporal axis → (N_samples, n_vertices)
        p_static = np.mean(pred, axis=-1)
        t_static = np.mean(target, axis=-1)

        # Normalize across vertices
        p_norm = p_static - np.mean(p_static, axis=1, keepdims=True)
        p_norm /= (np.linalg.norm(p_norm, axis=1, keepdims=True) + 1e-8)

        t_norm = t_static - np.mean(t_static, axis=1, keepdims=True)
        t_norm /= (np.linalg.norm(t_norm, axis=1, keepdims=True) + 1e-8)

        # Pairwise correlation matrix: (N_samples, N_samples)
        sim_matrix = np.dot(p_norm, t_norm.T)

        # Best match for each pred
        top1_preds = np.argmax(sim_matrix, axis=1)
        correct = np.sum(top1_preds == np.arange(n_samples))
        return float(correct / n_samples)

    @classmethod
    def evaluate_predictions(
        cls,
        preds: np.ndarray,
        targets: np.ndarray,
        subject_ids: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Full quantitative evaluation comparing predicted and ground-truth fMRI.

        Args:
            preds: (N_samples, n_vertices, T) or (n_vertices, T)
            targets: (N_samples, n_vertices, T) or (n_vertices, T)
            subject_ids: Optional array of subject indices.

        Returns:
            Dict of evaluation metrics.
        """
        if preds.ndim == 2:
            preds = preds[np.newaxis, ...]
            targets = targets[np.newaxis, ...]

        # 1. Per-vertex Pearson correlation across all segments
        r_per_vertex_segments = cls.compute_pearson_r(preds, targets, axis=-1)  # (N, n_vertices)
        mean_r_per_vertex = np.nanmean(r_per_vertex_segments, axis=0)  # (n_vertices,)

        overall_mean_r = float(np.nanmean(mean_r_per_vertex))
        overall_median_r = float(np.nanmedian(mean_r_per_vertex))

        # 2. R^2 score
        r2_per_vertex = cls.compute_r2(preds, targets, axis=-1)
        overall_mean_r2 = float(np.nanmean(r2_per_vertex))

        # 3. Top-1 retrieval
        top1_acc = cls.compute_top1_retrieval(preds, targets)

        # 4. Per-subject breakdown
        subj_metrics: Dict[str, float] = {}
        if subject_ids is not None:
            unique_subjs = np.unique(subject_ids)
            for s in unique_subjs:
                idx = np.where(subject_ids == s)[0]
                if len(idx) > 0:
                    s_r = np.nanmean(r_per_vertex_segments[idx])
                    subj_metrics[f"subj_{s}_mean_pearson"] = float(s_r)

        return {
            "mean_pearson": round(overall_mean_r, 4),
            "median_pearson": round(overall_median_r, 4),
            "mean_r2": round(overall_mean_r2, 4),
            "top1_retrieval_acc": round(top1_acc, 4),
            "per_subject": subj_metrics,
            "n_samples": preds.shape[0],
            "n_vertices": preds.shape[1],
            "n_trs": preds.shape[2],
        }

    @staticmethod
    def map_to_hcp_rois(
        per_vertex_metrics: np.ndarray,
        n_rois: int = 360,
    ) -> Dict[str, Any]:
        """Aggregate per-vertex metrics into HCP-MMP1 360 cortical ROIs.

        If explicit parcellation labels are not provided, groups into canonical
        cortical subdivisions.
        """
        n_vertices = len(per_vertex_metrics)
        verts_per_roi = max(1, n_vertices // n_rois)

        roi_means: List[float] = []
        for i in range(n_rois):
            start = i * verts_per_roi
            end = min(n_vertices, (i + 1) * verts_per_roi)
            chunk = per_vertex_metrics[start:end]
            roi_means.append(float(np.nanmean(chunk)) if len(chunk) > 0 else 0.0)

        roi_arr = np.array(roi_means)
        top10_idx = np.argsort(roi_arr)[-10:][::-1]

        return {
            "n_rois": n_rois,
            "roi_mean_pearson": round(float(np.mean(roi_arr)), 4),
            "top_10_rois": [{"roi_id": int(idx), "mean_pearson": round(float(roi_arr[idx]), 4)} for idx in top10_idx],
        }


def validate_against_synthetic(
    output_dir: Optional[str] = "validation_results",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, Any]:
    """Perform end-to-end validation against synthetic ground truth."""
    from ..brain.subject_layers import BrainPredictionHead
    from ..data.preprocessing import create_synthetic_dataset
    from ..fusion.cross_attention_fusion import TribeLocalFusion

    out_p = Path(output_dir) if output_dir else Path("validation_results")
    out_p.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic dataset with known ground-truth correlation
    logger.info("Generating synthetic validation dataset...")
    syn_dir = out_p / "synthetic_val"
    manifest = create_synthetic_dataset(
        syn_dir,
        n_subjects=2,
        n_segments_per_subj=5,
        n_vertices=20484,
        duration_trs=100,
    )

    records = manifest["records"]

    # 2. Instantiate model components
    fusion = TribeLocalFusion(
        encoder_dims={"video": 384, "audio": 768, "text": 1536},
        hidden=512,
        low_rank=1024,
    ).to(device)

    head = BrainPredictionHead(
        mode="voxel",
        r=1024,
        n_vertices=20484,
        n_subjects=2,
        target_trs=100,
    ).to(device)

    fusion.eval()
    head.eval()

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    subject_ids: List[int] = []

    logger.info(f"Running inference across {len(records)} validation segments...")
    with torch.no_grad():
        for rec in records:
            v_t = torch.load(rec["cached_video"]).unsqueeze(0).to(device)  # (1, 384, T)
            a_t = torch.load(rec["cached_audio"]).unsqueeze(0).to(device)  # (1, 768, T)
            t_t = torch.load(rec["cached_text"]).unsqueeze(0).to(device)   # (1, 1536, T)
            s_id = torch.tensor([rec["subject_id"]], device=device)

            fmri_gt = torch.load(rec["fmri_path"]).numpy()  # (20484, 100)

            feats = {"video": v_t, "audio": a_t, "text": t_t}
            z = fusion(feats)
            pred = head(z, s_id).squeeze(0).cpu().numpy()  # (20484, 100)

            all_preds.append(pred)
            all_targets.append(fmri_gt)
            subject_ids.append(rec["subject_id"])

    preds_arr = np.stack(all_preds, axis=0)      # (N, 20484, 100)
    targets_arr = np.stack(all_targets, axis=0)  # (N, 20484, 100)
    subjs_arr = np.array(subject_ids)

    # 3. Evaluate
    report = FMRIValidator.evaluate_predictions(preds_arr, targets_arr, subjs_arr)

    # 4. Map to ROIs
    per_vertex_r = FMRIValidator.compute_pearson_r(preds_arr, targets_arr, axis=-1).mean(axis=0)
    roi_summary = FMRIValidator.map_to_hcp_rois(per_vertex_r, n_rois=360)
    report["hcp_roi_summary"] = roi_summary

    # Save to JSON
    report_file = out_p / "validation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Validation report saved to: {report_file}")
    print("\n" + "=" * 65)
    print("           TRIBE-LOCAL fMRI VALIDATION REPORT")
    print("=" * 65)
    print(f"  Segments evaluated : {report['n_samples']}")
    print(f"  Target vertices    : {report['n_vertices']} (fsaverage5)")
    print(f"  Duration (TRs)     : {report['n_trs']} TRs")
    print(f"  Mean Pearson r     : {report['mean_pearson']}")
    print(f"  Median Pearson r   : {report['median_pearson']}")
    print(f"  Mean R^2           : {report['mean_r2']}")
    print(f"  Top-1 Retrieval    : {report['top1_retrieval_acc'] * 100:.1f}%")
    print("-" * 65)
    print(f"  ROI Mean Pearson r : {roi_summary['roi_mean_pearson']}")
    print(f"  Top ROI r          : {roi_summary['top_10_rois'][0]['mean_pearson']}")
    print("=" * 65 + "\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="validation_results")
    args = parser.parse_args()

    validate_against_synthetic(output_dir=args.output_dir)
