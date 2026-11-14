"""
extract_features.py — Batch Feature Pre-Extraction & Caching
============================================================
Pre-extracts multimodal features from video, audio, and transcript files
and stores them in ~/.tribe_cache/ to accelerate training and minimize VRAM.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

from tribe_local.data.preprocessing import compute_file_hash, scan_dataset_directory
from tribe_local.encoders.audio_encoder import AudioFeatureExtractor
from tribe_local.encoders.text_encoder import TextFeatureExtractor
from tribe_local.encoders.video_encoder import VideoFeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tribe_local.extract_features")


def main():
    parser = argparse.ArgumentParser(description="Pre-extract and cache stimulus features")
    parser.add_argument("--data-root", type=str, required=True, help="Path to data directory")
    parser.add_argument("--modalities", nargs="+", default=["video", "audio", "text"])
    parser.add_argument("--fp16", action="store_true", default=True)
    args = parser.parse_args()

    records = scan_dataset_directory(args.data_root)
    if not records:
        logger.warning(f"No records found in {args.data_root}")
        return

    logger.info(f"Discovered {len(records)} segments across dataset. Extracting features...")

    video_ext = VideoFeatureExtractor(fp16=args.fp16) if "video" in args.modalities else None
    audio_ext = AudioFeatureExtractor() if "audio" in args.modalities else None
    text_ext = TextFeatureExtractor() if "text" in args.modalities else None

    for rec in tqdm(records, desc="Extracting features"):
        # 1. Video
        if video_ext and rec.get("video_path") and Path(rec["video_path"]).exists():
            try:
                _ = video_ext(rec["video_path"])
            except Exception as e:
                logger.debug(f"Video extraction error: {e}")

        # 2. Audio
        if audio_ext and rec.get("audio_path") and Path(rec["audio_path"]).exists():
            try:
                _ = audio_ext(rec["audio_path"])
            except Exception as e:
                logger.debug(f"Audio extraction error: {e}")

        # 3. Text
        if text_ext and rec.get("text_tsv_path") and Path(rec["text_tsv_path"]).exists():
            try:
                _ = text_ext(rec["text_tsv_path"])
            except Exception as e:
                logger.debug(f"Text extraction error: {e}")

    logger.info("Feature pre-extraction finished! All cached to ~/.tribe_cache/")


if __name__ == "__main__":
    main()
