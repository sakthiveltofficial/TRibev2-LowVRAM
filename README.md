# TRIBE-local: Lightweight Multimodal Brain Encoding

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.2+-792ee5.svg)](https://lightning.ai/)
[![Hardware Target](https://img.shields.io/badge/GPU-RTX%205060%20Ti%20(8GB)-76b900.svg)](https://www.nvidia.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Independent Research Implementation inspired by Meta TRIBE v2**  
> Based on architectural principles from: *"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience"* (Meta AI, arXiv:2605.04326).  
> *Note: This project is an independent adaptation re-engineered specifically for consumer GPUs (8GB VRAM) and does NOT claim original Meta TRIBE v2 accuracy without experimental validation. See [DEVIATIONS.md](DEVIATIONS.md) for full details.*

---

## 1. Overview & System Architecture

TRIBE-local is an end-to-end multimodal brain-encoding framework capable of predicting continuous human cortical BOLD responses (fMRI) from simultaneous video, audio, and language stimuli. 

While the original Meta TRIBE v2 required an 80GB enterprise GPU cluster, TRIBE-local replaces all heavyweight foundation backbones with high-efficiency, GPU-compatible alternatives optimized for a **single NVIDIA GeForce RTX 5060 Ti (8GB VRAM)** on **Windows 11 + WSL2 Ubuntu 24.04**.

### Architecture Data Flow

```
                      ┌────────────────────────────────────────┐
                      │          MULTIMODAL STIMULUS           │
                      └───────────────────┬────────────────────┘
                                          │
            ┌─────────────────────────────┼─────────────────────────────┐
            │                             │                             │
            ▼ (Video: 16-frame clip)      ▼ (Audio: 16kHz waveform)     ▼ (Transcript / Text)
  ┌───────────────────┐         ┌───────────────────┐         ┌───────────────────┐
  │  VideoMAE-Small   │         │    AST Encoder    │         │   DeepSeek API    │
  │  (~22M params)    │         │   (~86M params)   │         │ (or MiniLM local) │
  │    [FROZEN]       │         │     [FROZEN]      │         │     [FROZEN]      │
  └─────────┬─────────┘         └─────────┬─────────┘         └─────────┬─────────┘
            │ (B, 384, T_v)               │ (B, 768, T_a)               │ (B, 1536, T_t)
            └─────────────────────────────┼─────────────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    TEMPORAL ALIGNMENT (Linear Interp)  │
                      │    Aligns all streams to 2 Hz grid     │
                      └───────────────────┬────────────────────┘
                                          │ all (B, D_m, T_common)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    MODALITY PROJECTORS (MLPs)          │
                      │    Projects each to hidden // 3 = 170  │
                      │    Concatenates to: (B, T_common, 512) │
                      └───────────────────┬────────────────────┘
                                          │ (B, T, 512)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │   CROSS-ATTENTION TRANSFORMER FUSION   │
                      │   - 4-layer Self-Attention (hidden=512)│
                      │   - Pairwise Cross-Attention (V↔A,V↔T) │
                      │   - Gated Modality Weighting           │
                      │   - Modality Dropout (p=0.3)           │
                      └───────────────────┬────────────────────┘
                                          │ (B, T, 512)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │  LOW-RANK BOTTLENECK HEAD (r = 1024)   │
                      │  Linear(512 → 1024)                    │
                      └───────────────────┬────────────────────┘
                                          │ (B, 1024, T)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │     BRAIN PREDICTION HEAD (Dual-Mode)  │
                      │  • Voxel Mode: Shared Decoder + FiLM   │
                      │    Subject Adaptation → (B, 20484, 100)│
                      │  • ROI Mode: 360 HCP-MMP1 Parcels      │
                      │    Linear Projection → (B, 360, 100)   │
                      └───────────────────┬────────────────────┘
                                          │
                                          ▼
                      ┌────────────────────────────────────────┐
                      │     PREDICTED fMRI BOLD ACTIVITY       │
                      │    Surface Mesh: fsaverage5 (20,484)   │
                      └────────────────────────────────────────┘
```

---

## 2. Component Replacement Matrix

| Component | Meta TRIBE v2 (Original) | TRIBE-local (Ours) | Parameters | VRAM Footprint |
|:---|:---|:---|:---|:---|
| **Video Encoder** | V-JEPA 2 ViT-Giant | **VideoMAE-Small** | ~22M | ~120 MB |
| **Audio Encoder** | Wav2Vec-BERT 2.0 | **AST (Audio Spectrogram Transformer)** | ~86M | ~280 MB |
| **Text Encoder** | LLaMA 3.2-3B | **DeepSeek API** *(fallback: all-MiniLM-L6-v2)* | 0M local *(22M)* | 0 MB *(~80 MB)* |
| **Multimodal Fusion** | 8-layer Transformer (dim=1152) | **4-layer Cross-Attention Transformer (dim=512)** | ~18M | ~110 MB |
| **Low-Rank Bottleneck** | `r = 2048` | **`r = 1024`** | ~0.5M | ~15 MB |
| **Subject Head** | Monolithic matrix `(25, 2048, 20484)` = 4.2 GB | **Shared Linear + FiLM Style Embeddings** | ~21M | ~85 MB |
| **Total Trainable** | ~1.1 Billion | **~40 Million** | **Fits in 8GB VRAM with ~7GB headroom!** |

---

## 3. Installation Guide (WSL2 Ubuntu 24.04 / Windows 11)

### Prerequisites
- Windows 11 with WSL2 enabled (`wsl --version`)
- NVIDIA GPU Driver installed on Windows (NVIDIA CUDA pass-through enabled in WSL2)
- Target: RTX 5060 Ti (8GB VRAM) and at least 16GB system RAM (32GB recommended)

### Step 1: Clone Repository & Setup Environment
In your WSL2 Ubuntu terminal:

```bash
# Clone or navigate to the project
cd /mnt/d/"sakthivel workspace"/Workspace/CRM/tribe_local

# Run automated setup script
chmod +x scripts/setup_wsl.sh
./scripts/setup_wsl.sh
```

Or manually using Conda:

```bash
# Create Conda environment
conda create -n tribe_local python=3.11 -y
conda activate tribe_local

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install requirements
pip install -r requirements.txt
pip install -e .
```

### Step 2: Configure Environment Variables
Copy `.env.example` to `.env` and provide your DeepSeek API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
DEEPSEEK_API_KEY=sk-your-actual-deepseek-api-key-here
TRIBE_CACHE_DIR=~/.tribe_cache
```

*(If `DEEPSEEK_API_KEY` is not provided, the pipeline automatically falls back to local SentenceTransformers `all-MiniLM-L6-v2` with zero configuration required).*

---

## 4. Quickstart & Testing

### 1. Run Automated GPU Benchmark
Profiles VRAM consumption, latency, and throughput on your RTX 5060 Ti:

```bash
python -m tribe_local.evaluation.benchmark --output-dir benchmarks
```

Expected output:
```
============================================================================
  TRIBE-LOCAL BENCHMARK SUMMARY (Target: RTX 5060 Ti 8GB VRAM)
============================================================================
Device: NVIDIA GeForce RTX 5060 Ti | FP16: True
----------------------------------------------------------------------------
Component                  | Total Params | Peak VRAM (MB) | Latency (ms)  
----------------------------------------------------------------------------
fusion                     | 18,378,752   | 112.4          | 12.8          
voxel_head                 | 21,012,388   | 84.6           | 6.2           
roi_head                   | 378,280      | 8.1            | 1.8           
integrated_pipeline        | 39,391,140   | 197.0          | 19.0          
----------------------------------------------------------------------------
8GB VRAM Budget Status: PASS (Fits comfortably in 8GB) (Headroom: 7995.0 MB)
============================================================================
```

### 2. Run Synthetic fMRI Validation
Validates Pearson correlation, R² score, and ROI parcellation mapping against synthetic ground truth:

```bash
python -m tribe_local.evaluation.validate_fmri --output-dir validation_results
```

---

## 5. Training Guide

### Pre-Extract Stimulus Features (Recommended)
Pre-extracting features to disk eliminates redundant forward passes through the frozen vision and audio encoders during training, maximizing GPU throughput:

```bash
python scripts/extract_features.py --data-root data/my_dataset --modalities video audio text
```

### Train Voxel-Mode Model (20,484 Cortical Vertices)
Trains the lightweight cross-attention transformer and vertex-level subject head:

```bash
python -m tribe_local.training.train \
    --config configs/local_config.yaml \
    --data-root data/my_dataset \
    --mode voxel \
    --batch-size 1 \
    --fp16 \
    --max-epochs 15
```

### Train ROI-Mode Model (360 HCP-MMP1 Parcels)
For ultra-fast experimentation or smaller GPUs:

```bash
python -m tribe_local.training.train \
    --config configs/roi_config.yaml \
    --data-root data/my_dataset \
    --mode roi \
    --batch-size 2 \
    --fp16 \
    --max-epochs 15
```

---

## 6. Inference API

### Command Line Interface

```bash
python -m tribe_local.evaluation.inference \
    --config configs/local_config.yaml \
    --checkpoint checkpoints/best_model.ckpt \
    --video sample_stimuli/clip.mp4 \
    --audio sample_stimuli/clip.wav \
    --text "A person is walking down the street." \
    --subject-id 0 \
    --output predicted_bold.npy
```

### Python API

```python
from tribe_local.evaluation.inference import TribeLocalPredictor
import yaml

with open("configs/local_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

predictor = TribeLocalPredictor(
    config=cfg,
    checkpoint_path="checkpoints/best_model.ckpt",
    device="cuda",
    fp16=True,
)

# Run inference on any multimodal input
result = predictor.predict(
    video_path="stimulus.mp4",
    audio_path="stimulus.wav",
    text_input="Dialogue transcribed from scene",
    subject_id=0,
)

print("Predicted fMRI shape:", result["predicted_bold"].shape)  # (20484, 100)
print("Inference latency:", result["latency_ms"], "ms")
```

---

## 7. Streamlit Web UI, REST API & MCP Server

### A. Streamlit Interactive Web Studio
Launch the full interactive dashboard for script generation, real-time engagement graphs, peak/drop diagnostics, and brain surface mapping:

```bash
streamlit run tribe_local/ui/app.py
```
Open your browser at `http://localhost:8501`. Features include:
- **Viral Script & Retention Analyzer:** Enter any topic or click *"Generate with DeepSeek"* to craft high-retention scripts, and view second-by-second engagement graphs, hook ratings, and drop-off risks.
- **Multimodal Cortical Predictor:** Upload video/audio clips and predict cortical fMRI activity in real time.
- **Hardware & VRAM Monitor:** Live RTX 5060 Ti memory profiling.

---

### B. FastAPI REST API Server
Start the high-performance REST API:

```bash
python -m tribe_local.api.server
# Or using uvicorn:
uvicorn tribe_local.api.server:app --host 0.0.0.0 --port 8000
```
Interactive Swagger / OpenAPI documentation is available at `http://localhost:8000/docs`.

**Key Endpoints:**
- `GET /api/v1/health`: GPU and service status check.
- `POST /api/v1/predict`: Predict brain BOLD response from text, video, or audio.
- `POST /api/v1/engagement`: Get second-by-second timeline engagement curves, top peaks, and drop-off risks.
- `POST /api/v1/generate-script`: Generate viral scripts via DeepSeek and immediately compute retention metrics.

---

### C. Model Context Protocol (MCP) Server
Integrate TRIBE-local directly with Claude Desktop, Antigravity, Cursor, or any LLM agent:

```bash
python -m tribe_local.mcp.server
```

**Add to Claude Desktop (`claude_desktop_config.json`):**
```json
{
  "mcpServers": {
    "tribe-local": {
      "command": "python",
      "args": ["-m", "tribe_local.mcp.server"],
      "cwd": "D:\\sakthivel workspace\\Workspace\\CRM"
    }
  }
}
```

**Exposed MCP Tools:**
1. `tribe_predict_brain`: Predict human cortical BOLD responses from multimodal stimuli.
2. `tribe_analyze_engagement`: Calculate timeline engagement curves, hook strength, and drop-off risks.
3. `tribe_generate_viral_script`: Generate scripts with DeepSeek and return engagement analytics.
4. `tribe_get_system_status`: Inspect GPU memory, VRAM headroom, and encoder health.

---

## 8. Supported Datasets

TRIBE-local supports standard fMRI datasets used in the original TRIBE v2 paper:
1. **Algonauts 2025 (Courtois NeuroMod)**: Naturalistic movie and TV sitcom viewing fMRI (4 subjects, TR=1.49s).
2. **Lahner 2024 (BOLD Moments)**: 10 subjects watching short 3-second video clips (OpenNeuro `ds004873`).
3. **Lebel 2023 (Natural Language)**: 8 subjects listening to spoken stories (OpenNeuro `ds003020`).
4. **Wen 2017**: Naturalistic video viewing fMRI (3 subjects, TR=2.0s).

Run `scripts/download_algonauts.sh` for download pointers.

---

## 8. Troubleshooting

| Issue | Cause | Solution |
|:---|:---|:---|
| **CUDA out of memory (OOM)** | Large batch size or unclosed memory cache | Ensure `--batch-size 1` and `--fp16` are active. Pre-extract features with `extract_features.py` to keep vision/audio encoders off GPU during training. |
| **`nvidia-smi` not found in WSL2** | Missing NVIDIA Windows display driver | Update to the latest NVIDIA GeForce Game Ready or Studio Driver on Windows. Do **not** install Linux drivers inside WSL2. |
| **DeepSeek API timeout / rate limit** | Network latency or API quota exhaustion | The text encoder automatically falls back to local `all-MiniLM-L6-v2`. Alternatively, pre-extract text features using `extract_features.py`. |
| **Missing audio libraries (`sndfile`)** | System dependency missing | Run `sudo apt-get install -y libsndfile1 ffmpeg`. |

---

## 9. License & Citations

- **Original TRIBE v2 Paper:**
  > Michael et al., *"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience"*, arXiv:2605.04326 (Meta AI, 2026).
- **Code License:** Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
