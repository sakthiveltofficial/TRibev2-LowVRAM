"""
tribe_local/encoders/video_encoder.py
======================================
VideoMAE-Small encoder (MCG-NJU/videomae-small, ~22M params) as a frozen
drop-in replacement for V-JEPA2 ViT-G used in TRIBE v2.

Original TRIBE v2:
  V-JEPA2 ViT-G  →  (B, 2, 1408, T≈200) at 2 Hz (2 layers at [0.75, 1.0])

This module:
  VideoMAE-Small  →  (B, 384, T_temporal=8) at 2 Hz

VideoMAE-Small tokenisation with 16 input frames:
  - Spatial patch size : 16×16 on 224×224 images → 14×14 = 196 spatial patches
  - Temporal tube depth: 2 frames per tube → 16 // 2 = 8 temporal positions
  - Total tokens       : 8 × 196 = 1568 (matches HuggingFace output)
  - Encoder dim        : D = 384

Output after temporal-mean pooling over spatial patches:
  (B, 384, 8)   where 8 = T_temporal

Dependencies: torch, transformers, torchvision, einops, numpy
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from transformers import VideoMAEModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_VIDEOMAE_MODEL_ID   = "MCG-NJU/videomae-small"
_VIDEOMAE_FRAMES     = 16        # canonical clip length for VideoMAE-Small
_VIDEOMAE_IMG_SIZE   = 224       # spatial resolution (H = W = 224)
_VIDEOMAE_PATCH_H    = 14        # 224 / 16 = 14 spatial patch rows
_VIDEOMAE_PATCH_W    = 14        # 224 / 16 = 14 spatial patch cols
_VIDEOMAE_TEMP_TUBE  = 2         # temporal tube depth (frames per tube)
_VIDEOMAE_T_TEMPORAL = _VIDEOMAE_FRAMES // _VIDEOMAE_TEMP_TUBE  # 8
_VIDEOMAE_D          = 384       # encoder hidden dim for VideoMAE-Small

_CACHE_DIR = Path.home() / ".tribe_cache" / "video"


# ---------------------------------------------------------------------------
# VideoMAEEncoder
# ---------------------------------------------------------------------------

class VideoMAEEncoder(nn.Module):
    """Frozen VideoMAE-Small encoder.

    Wraps HuggingFace ``VideoMAEModel`` (MCG-NJU/videomae-small) and reduces
    the full token sequence to a compact temporal feature sequence via mean
    pooling over spatial patch tokens.

    Args:
        pretrained (bool): Load pretrained weights from HuggingFace Hub.
        fp16 (bool): Cast model and inputs to ``torch.float16`` during forward.
        cpu_offload (bool): Keep model on CPU; move to GPU only during forward.

    Input shape:
        pixel_values : (B, C=3, T=16, H=224, W=224) – float, normalised

    Output shape:
        (B, D_video=384, T_temporal=8)
    """

    def __init__(
        self,
        pretrained: bool = True,
        fp16: bool = False,
        cpu_offload: bool = False,
    ) -> None:
        super().__init__()

        self.fp16         = fp16
        self.cpu_offload  = cpu_offload
        self.output_dim   = _VIDEOMAE_D
        self.t_temporal   = _VIDEOMAE_T_TEMPORAL  # 8

        # ---- Load model -------------------------------------------------------
        print(f"[VideoMAEEncoder] Loading {_VIDEOMAE_MODEL_ID} …")
        self.model: VideoMAEModel = VideoMAEModel.from_pretrained(
            _VIDEOMAE_MODEL_ID,
        )

        # ---- Freeze all encoder parameters ------------------------------------
        for param in self.model.parameters():
            param.requires_grad = False

        print(
            f"[VideoMAEEncoder] Frozen. "
            f"Params: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f} M"
        )

        # ---- FP16 cast --------------------------------------------------------
        if fp16:
            self.model = self.model.half()

    # -----------------------------------------------------------------------

    def get_output_dim(self) -> int:
        """Return the feature dimension D_video = 384."""
        return self.output_dim

    # -----------------------------------------------------------------------

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract temporal video features.

        Args:
            pixel_values: ``(B, C, T, H, W)`` float tensor, normalised with
                ImageNet mean/std.  T must equal 16.

        Returns:
            Tensor of shape ``(B, 384, 8)`` – temporal feature sequence.

        Shapes at each stage
        --------------------
        pixel_values           : (B, 3, 16, 224, 224)
        HF pixel_values input  : (B, 16, 3, 224, 224)  ← HF convention
        last_hidden_state      : (B, 1568, 384)
          └─ 1568 = 8 temporal positions × 196 spatial patches
        after rearrange        : (B, 8, 196, 384)
        after spatial mean pool: (B, 8, 384)
        output (transposed)    : (B, 384, 8)
        """
        B = pixel_values.shape[0]
        assert pixel_values.shape[1:] == (3, _VIDEOMAE_FRAMES, _VIDEOMAE_IMG_SIZE, _VIDEOMAE_IMG_SIZE), (
            f"Expected (B, 3, {_VIDEOMAE_FRAMES}, {_VIDEOMAE_IMG_SIZE}, {_VIDEOMAE_IMG_SIZE}), "
            f"got {tuple(pixel_values.shape)}"
        )

        # ---- Device / dtype management ----------------------------------------
        target_device = next(self.model.parameters()).device

        if self.cpu_offload:
            # move model to GPU for the forward pass
            gpu_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            self.model.to(gpu_device)
            target_device = gpu_device

        if self.fp16:
            pixel_values = pixel_values.half()

        pixel_values = pixel_values.to(target_device)

        # ---- HuggingFace convention: (B, T, C, H, W) -------------------------
        # Our input is (B, C, T, H, W) → permute
        pixel_values_hf = pixel_values.permute(0, 2, 1, 3, 4)
        # shape: (B, 16, 3, 224, 224)

        # ---- VideoMAE tube masking --------------------------------------------
        # VideoMAEModel expects a bool_masked_pos of shape (B, num_patches//4)
        # = (B, 1568//4) = (B, 392).  We pass all-False (no masking) for inference.
        num_patches = _VIDEOMAE_T_TEMPORAL * _VIDEOMAE_PATCH_H * _VIDEOMAE_PATCH_W  # 1568
        bool_masked_pos = torch.zeros(
            B, num_patches, dtype=torch.bool, device=target_device
        )

        # ---- Forward pass through frozen encoder ------------------------------
        with torch.no_grad():
            outputs = self.model(
                pixel_values=pixel_values_hf,
                bool_masked_pos=bool_masked_pos,
            )

        # last_hidden_state: (B, 1568, 384)
        last_hidden: torch.Tensor = outputs.last_hidden_state

        # ---- Reshape: (B, T_temporal, n_spatial, D) ---------------------------
        # 1568 tokens = 8 temporal × 196 spatial
        n_spatial = _VIDEOMAE_PATCH_H * _VIDEOMAE_PATCH_W  # 196
        features = rearrange(
            last_hidden,
            "b (t s) d -> b t s d",
            t=self.t_temporal,
            s=n_spatial,
        )
        # shape: (B, 8, 196, 384)

        # ---- Temporal-mean pool over spatial patches --------------------------
        features = features.mean(dim=2)
        # shape: (B, 8, 384)

        # ---- Transpose to (B, D, T) -------------------------------------------
        features = features.transpose(1, 2)
        # shape: (B, 384, 8)

        # ---- CPU offload: move model back to CPU ------------------------------
        if self.cpu_offload:
            self.model.to("cpu")
            features = features.cpu()

        return features.float()  # always return float32


# ---------------------------------------------------------------------------
# VideoFeatureExtractor
# ---------------------------------------------------------------------------

class VideoFeatureExtractor:
    """High-level wrapper around ``VideoMAEEncoder``.

    Accepts video file paths **or** raw numpy arrays, preprocesses them,
    runs the encoder in 2-second temporal chunks at 2 Hz, and optionally
    caches results to ``~/.tribe_cache/video/``.

    Args:
        encoder (VideoMAEEncoder | None): Pre-built encoder.  If ``None``,
            one is created with default settings.
        fps (float): Source video frame rate used to determine chunk sizes.
        output_fps (float): Desired feature output rate (Hz).  Default = 2.
        use_cache (bool): Enable MD5-keyed feature caching.
        fp16 (bool): Forward to encoder if constructing one internally.
        cpu_offload (bool): Forward to encoder if constructing one internally.

    Output:
        dict with key ``"video"`` → ``torch.Tensor`` of shape
        ``(T_feature_steps, 384)``
    """

    def __init__(
        self,
        encoder: Optional[VideoMAEEncoder] = None,
        fps: float = 30.0,
        output_fps: float = 2.0,
        use_cache: bool = True,
        fp16: bool = False,
        cpu_offload: bool = False,
    ) -> None:
        self.encoder    = encoder or VideoMAEEncoder(fp16=fp16, cpu_offload=cpu_offload)
        self.fps        = fps
        self.output_fps = output_fps
        self.use_cache  = use_cache

        # frames per 2-second window
        self.frames_per_window = _VIDEOMAE_FRAMES  # 16

        if use_cache:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------

    @staticmethod
    def _md5(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    # -----------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        return _CACHE_DIR / f"{key}.pkl"

    def _load_cache(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        p = self._cache_path(key)
        if p.exists():
            with open(p, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        with open(self._cache_path(key), "wb") as f:
            pickle.dump(value, f)

    # -----------------------------------------------------------------------

    @staticmethod
    def _load_video_as_array(path: Union[str, Path]) -> np.ndarray:
        """Load a video file into a numpy array ``(T, H, W, C)`` uint8.

        Uses ``torchvision.io.read_video`` for format support.

        Args:
            path: Path to a video file (MP4, AVI, MKV, …).

        Returns:
            ``(T, H, W, C)`` uint8 numpy array.
        """
        import torchvision.io as tvio  # lazy import

        frames, _, _ = tvio.read_video(str(path), output_format="THWC", pts_unit="sec")
        return frames.numpy()  # (T, H, W, C) uint8

    # -----------------------------------------------------------------------

    @staticmethod
    def _preprocess_frames(frames: np.ndarray, img_size: int = _VIDEOMAE_IMG_SIZE) -> torch.Tensor:
        """Resize, normalise, and convert frames to a model-ready tensor.

        Args:
            frames: ``(T, H, W, C)`` uint8 numpy array.
            img_size: Target spatial resolution (default 224).

        Returns:
            ``(C, T, H, W)`` float32 tensor normalised with ImageNet mean/std.

        Shapes:
            frames          : (T, H, W, C) uint8
            as tensor       : (T, C, H, W) float [0, 1]
            after resize    : (T, C, img_size, img_size)
            final           : (C, T, img_size, img_size)
        """
        import torchvision.transforms.functional as TF  # lazy import

        mean = torch.tensor(_IMAGENET_MEAN, dtype=torch.float32)[:, None, None]
        std  = torch.tensor(_IMAGENET_STD,  dtype=torch.float32)[:, None, None]

        t_frames = []
        for frame in frames:  # frame: (H, W, C) uint8
            img = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # (C, H, W)
            img = TF.resize(img, [img_size, img_size], antialias=True)
            img = (img - mean) / std
            t_frames.append(img)

        stacked = torch.stack(t_frames, dim=1)  # (C, T, H, W)
        return stacked

    # -----------------------------------------------------------------------

    def _chunk_frames(self, frames_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split a ``(C, T, H, W)`` tensor into 16-frame windows (strided by 8).

        Windows: [0:16], [8:24], [16:32], …  (50 % overlap keeps 2 Hz at 30 fps).
        If the last window is shorter than 16 frames it is padded with zeros.

        Args:
            frames_tensor: ``(C, T, H, W)``

        Returns:
            List of ``(C, 16, H, W)`` tensors.
        """
        C, T, H, W = frames_tensor.shape
        stride = max(1, int(self.fps / self.output_fps / 2))  # ~8 at 30 fps / 2 Hz
        chunks = []
        start = 0
        while start < T:
            end   = start + self.frames_per_window
            chunk = frames_tensor[:, start:end, :, :]          # (C, <=16, H, W)
            if chunk.shape[1] < self.frames_per_window:
                # zero-pad if clip is shorter
                pad = torch.zeros(C, self.frames_per_window - chunk.shape[1], H, W,
                                  dtype=chunk.dtype)
                chunk = torch.cat([chunk, pad], dim=1)
            chunks.append(chunk)
            start += stride
        return chunks

    # -----------------------------------------------------------------------

    def encode(
        self,
        source: Union[str, Path, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Encode a single video source.

        Args:
            source: A video file path **or** a ``(T, H, W, C)`` uint8 numpy
                    array of raw frames.

        Returns:
            ``{"video": Tensor(T_feature_steps, 384)}``

        Output shape:
            T_feature_steps depends on video length and chunk stride.
            Each step corresponds to a 2 Hz feature sample.
            Feature dim = 384 (VideoMAE-Small encoder dim).
        """
        # ---- Cache key --------------------------------------------------------
        cache_key: Optional[str] = None
        if self.use_cache:
            if isinstance(source, (str, Path)):
                with open(str(source), "rb") as f:
                    raw = f.read(1 << 20)  # first 1 MB for hash
                cache_key = self._md5(raw + str(source).encode())
            else:
                cache_key = self._md5(source.tobytes())

            cached = self._load_cache(cache_key)
            if cached is not None:
                return cached

        # ---- Load frames ------------------------------------------------------
        if isinstance(source, (str, Path)):
            frames_np = self._load_video_as_array(source)
        else:
            frames_np = source  # (T, H, W, C) uint8

        # ---- Preprocess -------------------------------------------------------
        frames_t = self._preprocess_frames(frames_np)  # (C, T, H, W)

        # ---- Temporal chunks --------------------------------------------------
        chunks = self._chunk_frames(frames_t)           # list of (C, 16, H, W)

        # ---- Encode batched chunks -------------------------------------------
        all_features: List[torch.Tensor] = []
        self.encoder.eval()
        with torch.no_grad():
            for chunk in chunks:
                # (C, 16, H, W) → (1, C, 16, H, W)
                x = chunk.unsqueeze(0)
                feat = self.encoder(x)  # (1, 384, 8)
                # Take temporal mean over the 8 internal positions → (1, 384)
                feat = feat.mean(dim=-1)  # (1, 384)
                all_features.append(feat.squeeze(0))  # (384,)

        # (T_feature_steps, 384)
        features_cat = torch.stack(all_features, dim=0)

        result = {"video": features_cat}

        if self.use_cache and cache_key is not None:
            self._save_cache(cache_key, result)

        return result

    # -----------------------------------------------------------------------

    def encode_batch(
        self,
        sources: List[Union[str, Path, np.ndarray]],
    ) -> List[Dict[str, torch.Tensor]]:
        """Encode a list of video sources sequentially.

        Args:
            sources: List of file paths or numpy arrays.

        Returns:
            List of dicts, each ``{"video": Tensor(T_steps, 384)}``.
        """
        return [self.encode(src) for src in sources]


# ---------------------------------------------------------------------------
# __main__ — smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("VideoMAEEncoder smoke test")
    print("=" * 60)

    # ---- GPU test ------------------------------------------------------------
    has_gpu = torch.cuda.is_available()

    encoder = VideoMAEEncoder(pretrained=True, fp16=False, cpu_offload=False)

    if has_gpu:
        encoder = encoder.cuda()

    encoder.eval()
    dummy_input = torch.randn(1, 3, _VIDEOMAE_FRAMES, _VIDEOMAE_IMG_SIZE, _VIDEOMAE_IMG_SIZE)

    if has_gpu:
        dummy_input = dummy_input.cuda()
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        output = encoder(dummy_input)

    print(f"\n[GPU mode = {has_gpu}]")
    print(f"Input  shape : {tuple(dummy_input.shape)}")
    print(f"Output shape : {tuple(output.shape)}")
    assert output.shape == (1, _VIDEOMAE_D, _VIDEOMAE_T_TEMPORAL), (
        f"Unexpected output shape: {output.shape}"
    )
    print("✓  Shape assertion passed: (1, 384, 8)")

    if has_gpu:
        vram_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        print(f"GPU VRAM used (peak): {vram_mb:.1f} MB")
    else:
        print("No GPU detected — ran on CPU.")

    # ---- CPU-only test via cpu_offload flag ----------------------------------
    print("\n[CPU offload mode]")
    encoder_cpu = VideoMAEEncoder(pretrained=False, fp16=False, cpu_offload=True)
    encoder_cpu.eval()
    dummy_cpu = torch.randn(1, 3, _VIDEOMAE_FRAMES, _VIDEOMAE_IMG_SIZE, _VIDEOMAE_IMG_SIZE)

    with torch.no_grad():
        out_cpu = encoder_cpu(dummy_cpu)

    print(f"Input  shape : {tuple(dummy_cpu.shape)}")
    print(f"Output shape : {tuple(out_cpu.shape)}")
    assert out_cpu.shape == (1, _VIDEOMAE_D, _VIDEOMAE_T_TEMPORAL)
    print("✓  Shape assertion passed (CPU mode)")

    # ---- VideoFeatureExtractor smoke test ------------------------------------
    print("\n[VideoFeatureExtractor — numpy array input]")
    extractor = VideoFeatureExtractor(encoder=encoder_cpu, use_cache=False)
    fake_video = np.random.randint(0, 255, (32, 224, 224, 3), dtype=np.uint8)
    result = extractor.encode(fake_video)
    video_feat = result["video"]
    print(f"Extractor output shape : {tuple(video_feat.shape)}")
    assert video_feat.ndim == 2 and video_feat.shape[1] == _VIDEOMAE_D
    print("✓  VideoFeatureExtractor shape OK")

    print("\nAll smoke tests passed. ✓")
    sys.exit(0)
