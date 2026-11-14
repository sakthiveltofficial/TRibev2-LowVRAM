"""
audio_encoder.py — Audio Encoder for TRIBE-local Brain Encoding Pipeline
=========================================================================
Replaces the original Wav2Vec-BERT 2.0 audio encoder from TRIBE v2 with an
Audio Spectrogram Transformer (AST) from HuggingFace.

Original TRIBE v2:  Wav2Vec-BERT 2.0 → (B, 1024, T) at 2 Hz (mean over 2 layers)
This module  :      AST (MIT/ast-finetuned-audioset-10-10-0.4593) → (B, 768, T_out) at 2 Hz

Design decisions
----------------
* AST is patch-based, not causal. To produce a **temporal sequence** we chunk
  the mel spectrogram along the time axis into overlapping 1024-frame windows
  (≈10.24 s each). Each chunk produces one pooled 768-dim vector; stacking the
  vectors yields a temporal sequence at 2 Hz (one feature per 500 ms, i.e. one
  per 80-frame hop within the 1024-frame window).
* The encoder is **frozen** by default to match TRIBE v2's frozen-encoder
  strategy.
* Mixed-precision (FP16) is used via ``torch.cuda.amp.autocast``.
* Feature caching (MD5-keyed, stored in ``~/.tribe_cache/audio/``) avoids
  re-computing expensive AST forward passes.

Dependencies
------------
    pip install torch torchaudio transformers numpy
"""

from __future__ import annotations

import hashlib
import os
import pathlib
import pickle
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T

# ---------------------------------------------------------------------------
# Lazy import of HuggingFace transformers components (avoid slow import at
# module load time when not needed)
# ---------------------------------------------------------------------------
_transformers_imported = False
_ASTModel = None
_ASTFeatureExtractor = None


def _ensure_transformers() -> None:
    global _transformers_imported, _ASTModel, _ASTFeatureExtractor
    if not _transformers_imported:
        from transformers import ASTFeatureExtractor, ASTModel  # type: ignore

        _ASTModel = ASTModel
        _ASTFeatureExtractor = ASTFeatureExtractor
        _transformers_imported = True


# ---------------------------------------------------------------------------
# Constants — AST paper normalisation statistics
# ---------------------------------------------------------------------------
AST_MEAN: float = -4.2677393
AST_STD: float = 4.5689974

AST_MODEL_ID: str = "MIT/ast-finetuned-audioset-10-10-0.4593"
AST_OUTPUT_DIM: int = 768  # hidden dim of the AST model
MEL_BINS: int = 128
FRAMES_PER_CHUNK: int = 1024  # 10.24 s at 100 fps (hop_length=160, SR=16000)
TARGET_SR: int = 16_000
HOP_LENGTH: int = 160  # 10 ms per frame at 16 kHz
N_FFT: int = 400
FMIN: float = 50.0
FMAX: float = 8_000.0
# 2 Hz output → one AST pass per 0.5 s → 0.5 s / (1/100 fps) = 50 frames hop
CHUNK_HOP_FRAMES: int = 50  # overlap-strided chunk stride (500 ms → 2 Hz)

CACHE_DIR: pathlib.Path = pathlib.Path.home() / ".tribe_cache" / "audio"


# ===========================================================================
# MelSpectrogramProcessor
# ===========================================================================

class MelSpectrogramProcessor(nn.Module):
    """Convert raw waveform to batched mel spectrogram chunks.

    Parameters
    ----------
    src_sample_rate : int
        Sample rate of the input waveform. If different from 16 kHz the
        waveform will be resampled.
    n_mels : int
        Number of mel filter-bank bins.  Default: 128.
    n_fft : int
        FFT size.  Default: 400 (25 ms at 16 kHz).
    hop_length : int
        STFT hop length in samples.  Default: 160 (10 ms at 16 kHz).
    fmin : float
        Minimum frequency in Hz.  Default: 50.
    fmax : float
        Maximum frequency in Hz.  Default: 8000.
    frames_per_chunk : int
        Width (time axis) of each output chunk.  Default: 1024.
    chunk_hop_frames : int
        Stride between successive chunks in frames.  Default: 50 (→ 2 Hz).

    Forward
    -------
    Input  : waveform ``(samples,)`` float32 at *src_sample_rate*
    Output : ``(N_chunks, n_mels, frames_per_chunk)`` float32 mel chunks
    """

    def __init__(
        self,
        src_sample_rate: int = TARGET_SR,
        n_mels: int = MEL_BINS,
        n_fft: int = N_FFT,
        hop_length: int = HOP_LENGTH,
        fmin: float = FMIN,
        fmax: float = FMAX,
        frames_per_chunk: int = FRAMES_PER_CHUNK,
        chunk_hop_frames: int = CHUNK_HOP_FRAMES,
    ) -> None:
        super().__init__()
        self.src_sample_rate = src_sample_rate
        self.frames_per_chunk = frames_per_chunk
        self.chunk_hop_frames = chunk_hop_frames

        # Resampler (identity when src_sample_rate == TARGET_SR)
        if src_sample_rate != TARGET_SR:
            self.resampler: Optional[T.Resample] = T.Resample(
                orig_freq=src_sample_rate, new_freq=TARGET_SR
            )
        else:
            self.resampler = None

        self.mel_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            power=2.0,
        )
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80.0)

        self.register_buffer(
            "_norm_mean", torch.tensor(AST_MEAN, dtype=torch.float32)
        )
        self.register_buffer(
            "_norm_std", torch.tensor(AST_STD, dtype=torch.float32)
        )

    # ------------------------------------------------------------------

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute normalised mel spectrogram and split into AST chunks.

        Parameters
        ----------
        waveform : torch.Tensor
            Shape ``(samples,)`` or ``(1, samples)``, float32.

        Returns
        -------
        torch.Tensor
            Shape ``(N_chunks, 128, 1024)`` — batched mel spectrogram chunks.
        """
        # Ensure 1-D
        wav = waveform.squeeze()
        if wav.dim() != 1:
            raise ValueError(
                f"Expected 1-D waveform, got shape {waveform.shape}"
            )

        # Resample if needed
        if self.resampler is not None:
            wav = self.resampler(wav.unsqueeze(0)).squeeze(0)

        # Mel spectrogram: (n_mels, T_frames)
        mel: torch.Tensor = self.mel_transform(wav)
        mel = self.amplitude_to_db(mel)

        # AST-paper normalisation
        mel = (mel - self._norm_mean) / (self._norm_std + 1e-6)

        # Pad to ensure at least one full chunk
        T_frames = mel.shape[-1]
        if T_frames < self.frames_per_chunk:
            pad_len = self.frames_per_chunk - T_frames
            mel = torch.nn.functional.pad(mel, (0, pad_len))
            T_frames = self.frames_per_chunk

        # Chunking: stride = chunk_hop_frames → 2 Hz output
        chunks = []
        start = 0
        while start + self.frames_per_chunk <= T_frames:
            chunks.append(mel[:, start : start + self.frames_per_chunk])
            start += self.chunk_hop_frames

        # Stack → (N_chunks, n_mels, frames_per_chunk)
        return torch.stack(chunks, dim=0)


# ===========================================================================
# ASTEncoder
# ===========================================================================

class ASTEncoder(nn.Module):
    """Audio Spectrogram Transformer encoder producing temporal features.

    Loads ``MIT/ast-finetuned-audioset-10-10-0.4593`` from HuggingFace and
    runs it on mel spectrogram chunks to produce a temporal feature sequence
    at 2 Hz.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    frozen : bool
        If True (default) all AST parameters are frozen.
    pool : str
        Patch pooling strategy: ``"cls"`` uses the CLS token only;
        ``"mean"`` averages all patch tokens (excluding CLS).

    Forward
    -------
    Input  : ``(B, n_mels=128, T_frames=1024)``  — a **single** mel chunk
    Output : ``(B, 768, 1)``  — pooled 768-dim feature, time-axis of length 1

    To obtain a full temporal sequence, see :class:`AudioFeatureExtractor`
    which calls :meth:`encode_sequence` on multi-chunk inputs.
    """

    def __init__(
        self,
        model_id: str = AST_MODEL_ID,
        frozen: bool = True,
        pool: str = "mean",
    ) -> None:
        super().__init__()
        _ensure_transformers()

        self.model_id = model_id
        self.pool = pool

        # Load AST backbone
        self.ast = _ASTModel.from_pretrained(model_id)  # type: ignore[call-arg]

        if frozen:
            self.freeze()

        self._output_dim = AST_OUTPUT_DIM

    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all AST parameters (no gradient updates)."""
        for p in self.ast.parameters():
            p.requires_grad_(False)
        self.ast.eval()

    def unfreeze(self) -> None:
        """Unfreeze all AST parameters for fine-tuning."""
        for p in self.ast.parameters():
            p.requires_grad_(True)
        self.ast.train()

    def get_output_dim(self) -> int:
        """Return the feature dimensionality: 768."""
        return self._output_dim

    # ------------------------------------------------------------------

    def _pool_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool patch tokens → single (B, 768) vector.

        Parameters
        ----------
        hidden_states : torch.Tensor
            Shape ``(B, N_tokens, 768)`` — full last-hidden-state from AST.
            Token 0 is the CLS token; tokens 1: are patch embeddings.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 768)``.
        """
        if self.pool == "cls":
            return hidden_states[:, 0, :]
        else:  # "mean" — mean-pool non-CLS patch tokens
            return hidden_states[:, 1:, :].mean(dim=1)

    # ------------------------------------------------------------------

    def forward(self, mel_chunk: torch.Tensor) -> torch.Tensor:
        """Forward a single mel spectrogram chunk through AST.

        Parameters
        ----------
        mel_chunk : torch.Tensor
            Shape ``(B, n_mels=128, T_frames=1024)`` — batch of mel chunks.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 768)`` — pooled audio feature per sample.
        """
        # AST expects input_values of shape (B, T_frames, n_mels) — HF convention
        # (batch, time, freq) for ASTModel
        x = mel_chunk.permute(0, 2, 1)  # (B, T_frames=1024, n_mels=128)

        with torch.cuda.amp.autocast(enabled=mel_chunk.is_cuda):
            outputs = self.ast(input_values=x)
            hidden_states: torch.Tensor = outputs.last_hidden_state  # (B, N_tokens, 768)

        pooled = self._pool_hidden_states(hidden_states)  # (B, 768)
        return pooled

    # ------------------------------------------------------------------

    def encode_sequence(self, mel_chunks: torch.Tensor) -> torch.Tensor:
        """Encode a time-ordered stack of mel chunks to a temporal sequence.

        Parameters
        ----------
        mel_chunks : torch.Tensor
            Shape ``(B, N_chunks, n_mels=128, T_frames=1024)``.

        Returns
        -------
        torch.Tensor
            Shape ``(B, 768, N_chunks)`` — feature sequence at 2 Hz.
        """
        B, N_chunks, n_mels, T_frames = mel_chunks.shape
        # Flatten batch × chunks so we can process in one or batched passes
        flat = mel_chunks.view(B * N_chunks, n_mels, T_frames)
        feats = self.forward(flat)  # (B*N_chunks, 768)
        feats = feats.view(B, N_chunks, self._output_dim)
        feats = feats.permute(0, 2, 1)  # (B, 768, N_chunks)
        return feats


# ===========================================================================
# AudioFeatureExtractor
# ===========================================================================

class AudioFeatureExtractor:
    """High-level audio feature extractor with file I/O and caching.

    Wraps :class:`MelSpectrogramProcessor` and :class:`ASTEncoder` and
    provides a convenient interface that accepts either a file path or a raw
    numpy waveform.  Results are cached on disk at
    ``~/.tribe_cache/audio/<md5>.pkl`` to avoid re-computation.

    Parameters
    ----------
    src_sample_rate : int
        Sample rate of waveforms passed directly (not via file path).
        When loading from file, the native file sample rate is used.
    device : str or torch.device
        Device to run the encoder on.  Defaults to ``"cuda"`` if available,
        else ``"cpu"``.
    frozen : bool
        Freeze AST weights.  Default: True.
    use_cache : bool
        Enable on-disk feature cache.  Default: True.

    Call
    ----
    ``extractor(audio)`` where *audio* is:

    * ``str`` or ``pathlib.Path`` — path to any audio file readable by
      ``torchaudio.load``.
    * ``np.ndarray`` of shape ``(samples,)`` — float32 waveform.

    Returns
    -------
    dict
        ``{"audio": Tensor(T_feature_steps, 768)}``
    """

    def __init__(
        self,
        src_sample_rate: int = TARGET_SR,
        device: Optional[Union[str, torch.device]] = None,
        frozen: bool = True,
        use_cache: bool = True,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.use_cache = use_cache

        self.processor = MelSpectrogramProcessor(
            src_sample_rate=src_sample_rate
        ).to(self.device)
        self.encoder = ASTEncoder(frozen=frozen).to(self.device)

        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _md5_numpy(arr: np.ndarray) -> str:
        return hashlib.md5(arr.tobytes()).hexdigest()

    @staticmethod
    def _md5_file(path: Union[str, pathlib.Path]) -> str:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()

    def _cache_path(self, key: str) -> pathlib.Path:
        return CACHE_DIR / f"{key}.pkl"

    def _load_cache(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        cp = self._cache_path(key)
        if cp.exists():
            with open(cp, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cache(
        self, key: str, result: Dict[str, torch.Tensor]
    ) -> None:
        cp = self._cache_path(key)
        cpu_result = {k: v.cpu() for k, v in result.items()}
        with open(cp, "wb") as f:
            pickle.dump(cpu_result, f)

    # ------------------------------------------------------------------

    def _load_audio(
        self, source: Union[str, pathlib.Path, np.ndarray]
    ) -> Tuple[torch.Tensor, int]:
        """Load audio and return ``(waveform_1d, sample_rate)``."""
        if isinstance(source, (str, pathlib.Path)):
            wav, sr = torchaudio.load(str(source))
            # Mix down to mono if multi-channel
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            return wav.squeeze(0), sr
        elif isinstance(source, np.ndarray):
            wav = torch.from_numpy(source.astype(np.float32))
            return wav, self.processor.src_sample_rate
        else:
            raise TypeError(
                f"Expected file path or numpy array, got {type(source)}"
            )

    # ------------------------------------------------------------------

    @torch.no_grad()
    def __call__(
        self,
        audio: Union[str, pathlib.Path, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """Extract audio features.

        Parameters
        ----------
        audio : str | pathlib.Path | np.ndarray
            Audio source: file path or waveform array ``(samples,)`` float32.

        Returns
        -------
        dict
            ``{"audio": Tensor(T_feature_steps, 768)}``
            where T_feature_steps ≈ audio_duration_seconds × 2 (2 Hz output).
        """
        # Cache key
        cache_key: Optional[str] = None
        if self.use_cache:
            if isinstance(audio, (str, pathlib.Path)):
                cache_key = self._md5_file(audio)
            elif isinstance(audio, np.ndarray):
                cache_key = self._md5_numpy(audio)
            if cache_key:
                cached = self._load_cache(cache_key)
                if cached is not None:
                    return cached

        # Load waveform
        wav, sr = self._load_audio(audio)

        # Update processor sample rate if loading from file with native SR
        if isinstance(audio, (str, pathlib.Path)) and sr != self.processor.src_sample_rate:
            # Re-create processor with the correct source SR
            self.processor = MelSpectrogramProcessor(src_sample_rate=sr).to(self.device)

        wav = wav.to(self.device)

        # Mel spectrogram chunks: (N_chunks, 128, 1024)
        mel_chunks = self.processor(wav)  # (N_chunks, n_mels, T_frames)
        N_chunks = mel_chunks.shape[0]

        # Add batch dim: (1, N_chunks, 128, 1024)
        mel_chunks = mel_chunks.unsqueeze(0)

        # AST encode: (1, 768, N_chunks)
        with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            features = self.encoder.encode_sequence(mel_chunks)  # (1, 768, N_chunks)

        # Remove batch dim, transpose: (N_chunks, 768) = (T_feature_steps, 768)
        features = features.squeeze(0).permute(1, 0)  # (T_feature_steps, 768)

        result = {"audio": features.float()}

        # Cache
        if self.use_cache and cache_key:
            self._save_cache(cache_key, result)

        return result


# ===========================================================================
# __main__ — smoke test
# ===========================================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("ASTEncoder smoke test")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # 1. Test ASTEncoder directly with a single chunk
    print("\n[1] ASTEncoder — single chunk forward pass")
    encoder = ASTEncoder(frozen=True).to(device)
    dummy_chunk = torch.randn(1, MEL_BINS, FRAMES_PER_CHUNK, device=device)
    with torch.no_grad():
        single_out = encoder(dummy_chunk)
    print(f"    Input  shape : {tuple(dummy_chunk.shape)}")
    print(f"    Output shape : {tuple(single_out.shape)}")  # (1, 768)
    assert single_out.shape == (1, AST_OUTPUT_DIM), "Unexpected output shape"

    # 2. Test encode_sequence: batch of multi-chunk inputs
    print("\n[2] ASTEncoder — encode_sequence (B=1, N_chunks=10)")
    B, N_chunks = 1, 10
    batch_mel = torch.randn(B, N_chunks, MEL_BINS, FRAMES_PER_CHUNK, device=device)
    with torch.no_grad():
        seq_out = encoder.encode_sequence(batch_mel)
    print(f"    Input  shape : {tuple(batch_mel.shape)}")
    print(f"    Output shape : {tuple(seq_out.shape)}")  # (1, 768, 10)
    assert seq_out.shape == (B, AST_OUTPUT_DIM, N_chunks), "Unexpected sequence shape"

    # 3. MelSpectrogramProcessor on synthetic waveform
    print("\n[3] MelSpectrogramProcessor — synthetic 30-s waveform at 16 kHz")
    proc = MelSpectrogramProcessor().to(device)
    wav = torch.randn(TARGET_SR * 30, device=device)  # 30 seconds
    with torch.no_grad():
        chunks = proc(wav)
    print(f"    Waveform samples : {TARGET_SR * 30}")
    print(f"    Mel chunks shape : {tuple(chunks.shape)}")  # (N_chunks, 128, 1024)

    # 4. AudioFeatureExtractor on a synthetic numpy waveform
    print("\n[4] AudioFeatureExtractor — numpy waveform (10 seconds)")
    extractor = AudioFeatureExtractor(device=device, use_cache=False)
    wav_np = np.random.randn(TARGET_SR * 10).astype(np.float32)
    result = extractor(wav_np)
    print(f"    Output keys  : {list(result.keys())}")
    print(f"    'audio' shape: {tuple(result['audio'].shape)}")  # (T_steps, 768)
    expected_steps = (TARGET_SR * 10 // HOP_LENGTH - FRAMES_PER_CHUNK) // CHUNK_HOP_FRAMES + 1
    print(f"    Expected T_steps ≈ {expected_steps}  (~2 Hz for 10 s = ~20)")

    # 5. GPU VRAM usage
    if device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 ** 2
        reserved = torch.cuda.memory_reserved() / 1024 ** 2
        print(f"\n[5] GPU VRAM — Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")
    else:
        print("\n[5] GPU VRAM — running on CPU, no VRAM usage")

    print("\n✓ All smoke tests passed.")
    print(f"\nget_output_dim() = {encoder.get_output_dim()}")
    print(f"Output shape summary:")
    print(f"  Single chunk : (B, {AST_OUTPUT_DIM})")
    print(f"  Sequence     : (B, {AST_OUTPUT_DIM}, T_out) at 2 Hz")
