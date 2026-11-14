"""
inference.py — Inference Script for TRIBE-local Brain Encoding
==============================================================
Runs forward inference from multimodal stimulus inputs (video, audio, text)
to predict cortical BOLD responses in fsaverage5 vertex space or HCP-MMP1 ROIs.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch import Tensor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tribe_local.inference")


class TribeLocalPredictor:
    """High-level inference engine for predicting brain activity from multimodal stimuli."""

    def __init__(
        self,
        config: Dict[str, Any],
        checkpoint_path: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        fp16: bool = True,
    ) -> None:
        self.cfg = config
        self.device = device
        self.fp16 = fp16 and (device == "cuda")

        # 1. Initialize Fusion & Brain Head
        from ..brain.subject_layers import BrainPredictionHead
        from ..fusion.cross_attention_fusion import TribeLocalFusion

        enc_cfg = self.cfg.get("encoders", {})
        d_v = enc_cfg.get("video", {}).get("output_dim", 384)
        d_a = enc_cfg.get("audio", {}).get("output_dim", 768)
        d_t = enc_cfg.get("text", {}).get("output_dim", 1536)

        fusion_cfg = self.cfg.get("fusion", {})
        hidden = fusion_cfg.get("hidden", 512)
        low_rank = fusion_cfg.get("low_rank", 1024)

        self.fusion = TribeLocalFusion(
            encoder_dims={"video": d_v, "audio": d_a, "text": d_t},
            hidden=hidden,
            low_rank=low_rank,
        ).to(self.device)

        brain_cfg = self.cfg.get("brain_head", {})
        self.head_mode = brain_cfg.get("mode", "voxel")
        self.head = BrainPredictionHead(
            mode=self.head_mode,
            r=low_rank,
            n_vertices=brain_cfg.get("n_vertices", 20484),
            n_rois=brain_cfg.get("n_rois", 360),
            n_subjects=brain_cfg.get("n_subjects", 25),
            target_trs=brain_cfg.get("target_trs", 100),
        ).to(self.device)

        # 2. Load Weights if checkpoint provided
        if checkpoint_path and Path(checkpoint_path).exists():
            logger.info(f"Loading checkpoint weights from: {checkpoint_path}")
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("state_dict", ckpt)

            # Strip 'fusion.' and 'head.' prefixes if loaded from LightningModule
            fusion_state = {k.replace("fusion.", ""): v for k, v in state_dict.items() if k.startswith("fusion.")}
            head_state = {k.replace("head.", ""): v for k, v in state_dict.items() if k.startswith("head.")}

            if fusion_state:
                self.fusion.load_state_dict(fusion_state, strict=False)
            if head_state:
                self.head.load_state_dict(head_state, strict=False)

        self.fusion.eval()
        self.head.eval()

        # Encoders initialized on-demand to save VRAM
        self._video_extractor = None
        self._audio_extractor = None
        self._text_extractor = None

    def _get_video_extractor(self):
        if self._video_extractor is None:
            from ..encoders.video_encoder import VideoFeatureExtractor
            self._video_extractor = VideoFeatureExtractor(fp16=self.fp16)
        return self._video_extractor

    def _get_audio_extractor(self):
        if self._audio_extractor is None:
            from ..encoders.audio_encoder import AudioFeatureExtractor
            self._audio_extractor = AudioFeatureExtractor()
        return self._audio_extractor

    def _get_text_extractor(self):
        if self._text_extractor is None:
            from ..encoders.text_encoder import TextFeatureExtractor
            self._text_extractor = TextFeatureExtractor()
        return self._text_extractor

    def predict(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text_input: Optional[Union[str, List[Tuple[str, float, float]]]] = None,
        subject_id: int = 0,
        cached_features: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Any]:
        """Generate cortical fMRI predictions from stimulus inputs.

        Args:
            video_path: Path to .mp4 / .mkv video.
            audio_path: Path to .wav / .mp3 audio.
            text_input: String transcript or list of (word, start_s, end_s) tuples.
            subject_id: Integer subject index.
            cached_features: Pre-extracted feature dictionary (optional).

        Returns:
            Dict containing predicted BOLD response, latency, and shape metadata.
        """
        t0 = time.perf_counter()
        features: Dict[str, Optional[Tensor]] = {}

        # 1. Feature extraction or cache loading
        if cached_features is not None:
            for k, v in cached_features.items():
                if v is not None:
                    if v.ndim == 2:
                        v = v.unsqueeze(0)  # (1, D, T)
                    features[k] = v.to(self.device)
                else:
                    features[k] = None
        else:
            # Video
            if video_path and Path(video_path).exists():
                ext = self._get_video_extractor()
                res = ext(video_path)
                features["video"] = res["video"].unsqueeze(0).to(self.device)
            else:
                features["video"] = None

            # Audio
            if audio_path and Path(audio_path).exists():
                ext = self._get_audio_extractor()
                res = ext(audio_path)
                features["audio"] = res["audio"].unsqueeze(0).to(self.device)
            else:
                features["audio"] = None

            # Text
            if text_input:
                ext = self._get_text_extractor()
                res = ext(text_input)
                t_tensor = res["text"]
                if t_tensor.ndim == 2:
                    if t_tensor.shape[-1] == 1536 or t_tensor.shape[-1] == 384:
                        t_tensor = t_tensor.transpose(0, 1)  # (T, D) -> (D, T)
                    t_tensor = t_tensor.unsqueeze(0)  # -> (1, D, T)
                features["text"] = t_tensor.to(self.device)
            else:
                features["text"] = None

        s_tensor = torch.tensor([subject_id], device=self.device)

        # 2. Forward pass through lightweight fusion + brain head
        with torch.inference_mode():
            with torch.cuda.amp.autocast(enabled=self.fp16):
                z = self.fusion(features)  # (1, 1024, T)
                pred = self.head(z, s_tensor)  # (1, n_vertices, target_trs)

        pred_np = pred.squeeze(0).float().cpu().numpy()
        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "predicted_bold": pred_np,
            "mode": self.head_mode,
            "shape": pred_np.shape,
            "subject_id": subject_id,
            "latency_ms": round(latency_ms, 2),
            "device": self.device,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="TRIBE-local Brain Encoding Inference")
    parser.add_argument("--config", type=str, default="configs/local_config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--audio", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--subject-id", type=int, default=0)
    parser.add_argument("--output", type=str, default="predicted_bold.npy")
    args = parser.parse_args()

    # Load config
    cfg_p = Path(args.config)
    if not cfg_p.exists():
        cfg_p = Path(__file__).resolve().parent.parent / args.config
    with open(cfg_p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    predictor = TribeLocalPredictor(
        config=cfg,
        checkpoint_path=args.checkpoint,
    )

    logger.info(f"Running inference for Subject {args.subject_id}...")
    result = predictor.predict(
        video_path=args.video,
        audio_path=args.audio,
        text_input=args.text,
        subject_id=args.subject_id,
    )

    np.save(args.output, result["predicted_bold"])
    logger.info(
        f"Prediction complete! Shape: {result['shape']}, Latency: {result['latency_ms']} ms. "
        f"Saved to: {args.output}"
    )


if __name__ == "__main__":
    main()
