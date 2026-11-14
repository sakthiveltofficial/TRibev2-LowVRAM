"""
timeline_analyzer.py — Timeline-Based Cortical Engagement & Retention Analysis
==============================================================================
Analyzes temporal fMRI BOLD predictions across the video timeline to score:
- Continuous viewer cognitive engagement (second-by-second)
- High engagement peaks (hooks, dramatic turns, visual climaxes)
- Low engagement drop-off risks (pacing lulls, cognitive fatigue)
- Functional network breakdown (Attention, Language, Visual, Auditory)
- YouTube Shorts retention and hook diagnostics
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
import yaml

from .inference import TribeLocalPredictor

logger = logging.getLogger("tribe_local.timeline_analyzer")


class TimelineEngagementAnalyzer:
    """Analyzes continuous cortical BOLD predictions along a stimulus timeline."""

    # Functional network parcel indices based on HCP-MMP1 360 parcellation
    # Grouped into 4 cognitive axes relevant to video/audio engagement
    NETWORK_GROUPS = {
        "Attention_Curiosity": [
            # Frontoparietal, Dorsolateral Prefrontal, Intraparietal Sulcus
            8, 9, 10, 11, 23, 24, 85, 86, 95, 96, 140, 141, 188, 189, 203, 204
        ],
        "Language_Comprehension": [
            # Superior Temporal, Broca's area (44, 45), Wernicke's area (TPOJ, PSL)
            34, 35, 75, 76, 102, 103, 104, 133, 134, 174, 175, 214, 215, 255, 256
        ],
        "Visual_Salience": [
            # Early visual (V1, V2, V3), ventral stream (FFC, PIT), MT+
            1, 2, 3, 4, 15, 16, 17, 18, 21, 22, 181, 182, 183, 184, 195, 196
        ],
        "Auditory_Emotion": [
            # Primary auditory (A1), lateral temporal, anterior insula / limbic
            25, 26, 73, 74, 105, 106, 112, 113, 205, 206, 253, 254
        ],
    }

    def __init__(self, predictor: TribeLocalPredictor) -> None:
        self.predictor = predictor

    @classmethod
    def compute_engagement_timecourse(
        cls,
        predicted_bold: np.ndarray,
        duration_secs: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Transform predicted BOLD timecourse into continuous engagement curves.

        Args:
            predicted_bold: Array of shape (n_vertices, T) or (n_rois, T).
            duration_secs: Stimulus duration in seconds (defaults to T * 0.5s at 2Hz).

        Returns:
            Dict containing engagement metrics, timestamps, peaks, and troughs.
        """
        n_features, n_trs = predicted_bold.shape

        if duration_secs is None:
            # Assume 2 Hz (0.5s per time step) or TR=1.0s
            duration_secs = n_trs * 0.55

        time_stamps = np.linspace(0.0, duration_secs, n_trs)

        # 1. Global Engagement Index:
        # Standard deviation of dynamic cortical activations across space at each TR,
        # scaled and normalized to a [0, 100] engagement index.
        spatial_variance = np.std(predicted_bold, axis=0)  # Dynamic cortical dispersion
        spatial_energy = np.mean(np.abs(predicted_bold), axis=0)
        raw_engagement = spatial_variance * 0.6 + spatial_energy * 0.4

        # Normalize to 0-100 scale
        min_e = np.min(raw_engagement)
        max_e = np.max(raw_engagement)
        norm_engagement = (raw_engagement - min_e) / (max_e - min_e + 1e-8) * 100.0

        # Apply moving average temporal smoothing (window = 3 steps)
        kernel_size = min(5, max(1, n_trs // 10))
        if kernel_size > 1:
            kernel = np.ones(kernel_size) / kernel_size
            smooth_engagement = np.convolve(norm_engagement, kernel, mode="same")
        else:
            smooth_engagement = norm_engagement

        # 2. Network-specific engagement breakdown (if 360 ROIs or mapped)
        network_scores: Dict[str, List[float]] = {}
        if n_features == 360:
            for net_name, roi_indices in cls.NETWORK_GROUPS.items():
                valid_idx = [i for i in roi_indices if i < n_features]
                if valid_idx:
                    net_activity = np.mean(np.abs(predicted_bold[valid_idx, :]), axis=0)
                    net_norm = (net_activity - np.min(net_activity)) / (np.max(net_activity) - np.min(net_activity) + 1e-8) * 100.0
                    network_scores[net_name] = [round(float(v), 2) for v in net_norm]
        else:
            # Approximate parcel grouping across 20484 vertices
            step = n_features // 4
            for i, net_name in enumerate(["Visual_Salience", "Auditory_Emotion", "Language_Comprehension", "Attention_Curiosity"]):
                chunk = predicted_bold[i * step : (i + 1) * step, :]
                act = np.mean(np.abs(chunk), axis=0)
                norm_act = (act - np.min(act)) / (np.max(act) - np.min(act) + 1e-8) * 100.0
                network_scores[net_name] = [round(float(v), 2) for v in norm_act]

        # 3. Detect Peaks (Highest Engagement) & Troughs (Lowest Engagement)
        # Sort indices by engagement
        sorted_indices = np.argsort(smooth_engagement)
        trough_indices = sorted_indices[:3]  # 3 lowest points
        peak_indices = sorted_indices[-3:][::-1]  # 3 highest peaks

        peaks = [
            {
                "time_sec": round(float(time_stamps[idx]), 1),
                "timestamp_str": f"{int(time_stamps[idx]//60):02d}:{int(time_stamps[idx]%60):02d}",
                "engagement_score": round(float(smooth_engagement[idx]), 1),
                "percentile": "High (Top 10%)",
            }
            for idx in peak_indices
        ]

        troughs = [
            {
                "time_sec": round(float(time_stamps[idx]), 1),
                "timestamp_str": f"{int(time_stamps[idx]//60):02d}:{int(time_stamps[idx]%60):02d}",
                "engagement_score": round(float(smooth_engagement[idx]), 1),
                "risk_factor": "Drop-off Risk (Lull)",
            }
            for idx in trough_indices
        ]

        # 4. Critical Short-Form Diagnostics
        # Hook (First 3-5 seconds)
        hook_window = max(1, int(n_trs * (3.0 / duration_secs)))
        hook_score = float(np.mean(smooth_engagement[:hook_window]))

        # Mid-video retention dip
        mid_start = int(n_trs * 0.3)
        mid_end = int(n_trs * 0.7)
        mid_score = float(np.mean(smooth_engagement[mid_start:mid_end]))

        # Climax (70% - 90% timeline)
        climax_start = int(n_trs * 0.7)
        climax_end = int(n_trs * 0.9)
        climax_score = float(np.mean(smooth_engagement[climax_start:climax_end]))

        # Final Hook / Loop (last 10%)
        loop_score = float(np.mean(smooth_engagement[-max(1, int(n_trs * 0.1)):]))

        timeline_points = [
            {
                "time_sec": round(float(t), 1),
                "timestamp": f"{int(t//60):02d}:{int(t%60):02d}",
                "engagement": round(float(score), 1),
            }
            for t, score in zip(time_stamps, smooth_engagement)
        ]

        return {
            "duration_secs": round(duration_secs, 1),
            "mean_engagement": round(float(np.mean(smooth_engagement)), 1),
            "hook_score_0_3s": round(hook_score, 1),
            "mid_retention_score": round(mid_score, 1),
            "climax_score": round(climax_score, 1),
            "loop_retention_score": round(loop_score, 1),
            "top_peaks": peaks,
            "drop_off_risks": troughs,
            "timeline": timeline_points,
            "network_breakdown": network_scores,
        }

    def analyze_stimulus(
        self,
        text_input: Optional[str] = None,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        duration_secs: Optional[float] = None,
        subject_id: int = 0,
    ) -> Dict[str, Any]:
        """Run end-to-end prediction and timeline analysis."""
        pred_res = self.predictor.predict(
            text_input=text_input,
            video_path=video_path,
            audio_path=audio_path,
            subject_id=subject_id,
        )

        bold_matrix = pred_res["predicted_bold"]  # (n_features, T)
        analysis = self.compute_engagement_timecourse(bold_matrix, duration_secs=duration_secs)
        analysis["model_latency_ms"] = pred_res["latency_ms"]
        analysis["target_shape"] = pred_res["shape"]
        return analysis


def render_ascii_timeline(analysis: Dict[str, Any], width: int = 50) -> str:
    """Render an ASCII engagement timeline chart."""
    timeline = analysis["timeline"]
    lines: List[str] = []

    lines.append("\n" + "=" * 68)
    lines.append("        TIMELINE ENGAGEMENT CURVE (YouTube Shorts Pacing)")
    lines.append("=" * 68)
    lines.append(f"  Duration: {analysis['duration_secs']}s | Mean Engagement: {analysis['mean_engagement']}/100")
    lines.append(f"  Hook (0-3s): {analysis['hook_score_0_3s']}/100 | Climax: {analysis['climax_score']}/100 | Loop: {analysis['loop_retention_score']}/100")
    lines.append("-" * 68)
    lines.append(" Time   | Engagement [0-100]                      | Status")
    lines.append("-" * 68)

    # Subsample to ~12-15 visual rows
    step = max(1, len(timeline) // 14)
    sampled = timeline[::step]

    for pt in sampled:
        t_str = pt["timestamp"]
        score = pt["engagement"]
        bar_len = int((score / 100.0) * (width - 15))
        bar = "█" * bar_len + "░" * (width - 15 - bar_len)

        status = ""
        if score >= 80:
            status = "🔥 HIGH PEAK"
        elif score <= 25:
            status = "⚠️ DROP RISK"
        elif score >= 60:
            status = "✨ ENGAGED"
        else:
            status = "MODERATE"

        lines.append(f" {t_str}  | {bar} {score:4.1f} | {status}")

    lines.append("-" * 68)
    lines.append(" TOP ENGAGING MOMENTS (Keep viewer glued):")
    for p in analysis["top_peaks"]:
        lines.append(f"   • At {p['timestamp_str']} ({p['time_sec']}s): Score {p['engagement_score']}/100")

    lines.append(" ATTENTION DROP RISKS (Where viewers scroll away):")
    for tr in analysis["drop_off_risks"]:
        lines.append(f"   • At {tr['timestamp_str']} ({tr['time_sec']}s): Score {tr['engagement_score']}/100 (Pacing lull)")

    lines.append("=" * 68 + "\n")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Timeline Engagement Analyzer")
    parser.add_argument("--config", type=str, default="configs/roi_config.yaml")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--duration", type=float, default=55.0)
    parser.add_argument("--output", type=str, default="timeline_engagement.json")
    args = parser.parse_args()

    cfg_p = Path(args.config)
    if not cfg_p.exists():
        cfg_p = Path(__file__).resolve().parent.parent / args.config

    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    predictor = TribeLocalPredictor(config=cfg, device="cpu")
    analyzer = TimelineEngagementAnalyzer(predictor)

    logger.info("Running timeline engagement analysis...")
    report = analyzer.analyze_stimulus(
        text_input=args.text,
        video_path=args.video,
        audio_path=args.audio,
        duration_secs=args.duration,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    chart = render_ascii_timeline(report)
    print(chart)
    logger.info(f"Timeline engagement data saved to: {args.output}")


if __name__ == "__main__":
    main()
