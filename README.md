# TRIBE-local: In-Silico Multimodal Brain Encoding & Retention Studio

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Hardware Target](https://img.shields.io/badge/GPU-RTX%205060%20Ti%20(8GB)-76b900.svg)](https://www.nvidia.com/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> **Modular Research Implementation inspired by Meta TRIBE v2**  
> Based on architectural principles from: *"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience"* (Meta AI, arXiv:2605.04326).  
> *Note: This project is an independent adaptation re-engineered specifically for consumer GPUs (8GB VRAM) and does NOT claim original Meta TRIBE v2 accuracy without experimental validation. See [DEVIATIONS.md](DEVIATIONS.md) for full architectural mappings.*

---

## 📑 Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Hardware Requirements](#2-hardware-requirements)
3. [Installation Guide](#3-installation-guide)
   - [Option A: Native Windows (PowerShell)](#option-a-native-windows-powershell)
   - [Option B: WSL2 Ubuntu 24.04 (Linux)](#option-b-wsl2-ubuntu-2404-linux)
   - [Option C: Conda Environment](#option-c-conda-environment)
4. [Environment Configuration (`.env`)](#4-environment-configuration)
5. [Quick Run Reference Card](#5-quick-run-reference-card)
6. [Streamlit Web Application](#6-streamlit-web-application)
7. [FastAPI REST API Server](#7-fastapi-rest-api-server)
8. [Model Context Protocol (MCP) Server](#8-model-context-protocol-mcp-server)
9. [Timeline Engagement & Viral Script Analyzer](#9-timeline-engagement--viral-script-analyzer)
10. [Hardware Profiling & VRAM Benchmark](#10-hardware-profiling--vram-benchmark)
11. [Training Pipeline](#11-training-pipeline)
12. [Evaluation & fMRI Validation](#12-evaluation--fmri-validation)
13. [Directory Map](#13-directory-map)
14. [Troubleshooting FAQ](#14-troubleshooting-faq)

---

## 1. System Architecture

TRIBE-local models human cortical fMRI responses across sensory and associative brain areas as a function of dynamic video, audio, and language inputs.

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
                      │    Interpolates all streams to 2 Hz    │
                      └───────────────────┬────────────────────┘
                                          │ all (B, D_m, T_common)
                                          ▼
                      ┌────────────────────────────────────────┐
                      │    MODALITY PROJECTORS (MLPs)          │
                      │    Projects to hidden // 3 = 170 each  │
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

## 2. Hardware Requirements

- **GPU:** NVIDIA GeForce RTX 5060 Ti (8GB VRAM) or equivalent Turing/Ampere/Ada GPU.
- **VRAM Budget:** Model weights require only **~138 MB**, leaving **>7.8 GB VRAM headroom** for batching and feature activations.
- **RAM:** Minimum 16GB (32GB recommended for large stimulus loading).
- **OS:** Windows 11 / Windows 10, or WSL2 Ubuntu 24.04 / 22.04 LTS.

---

## 3. Installation Guide

Choose one of the installation paths below based on your preferred environment.

### Option A: Native Windows (PowerShell)

Open PowerShell as Administrator or regular user and navigate to the project root:

```powershell
# 1. Navigate to the repository
cd "D:\sakthivel workspace\Workspace\RnD\TRibev2"

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch with CUDA 12.1 (skip if already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install all project requirements
pip install -r requirements.txt

# 6. Install the web UI and API servers
pip install streamlit fastapi uvicorn
```

### Option B: WSL2 Ubuntu 24.04 (Linux)

In your WSL2 terminal:

```bash
# 1. Navigate to the Windows mount path
cd /mnt/d/"sakthivel workspace"/Workspace/RnD/TRibev2

# 2. Make setup script executable and run automated installer
chmod +x scripts/setup_wsl.sh
./scripts/setup_wsl.sh
```

### Option C: Conda Environment

If using Anaconda or Miniconda:

```bash
cd "D:\sakthivel workspace\Workspace\RnD\TRibev2"

# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate tribe_local
```

---

## 4. Environment Configuration

Copy the example `.env` file and insert your API credentials:

```bash
cp .env.example .env
```

Open `.env` in any text editor:
```ini
# DeepSeek API Key (Required for DeepSeek script generation and text embeddings)
# Sign up at: https://platform.deepseek.com/
DEEPSEEK_API_KEY=your_actual_deepseek_api_key_here

# Local feature cache root (default: ~/.tribe_cache)
TRIBE_CACHE_DIR=~/.tribe_cache
```

*(Note: If `DEEPSEEK_API_KEY` is not provided, the text encoder automatically falls back to local SentenceTransformers `all-MiniLM-L6-v2` with zero network dependency).*

---

## 5. Quick Run Reference Card

| Action | Exact Command to Run | Default Port / URL |
|:---|:---|:---|
| **Streamlit Web Studio** | `streamlit run ui/app.py --server.port 8501` | [http://localhost:8501](http://localhost:8501) |
| **FastAPI REST Server** | `python -m uvicorn api.server:app --port 8000` | [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) |
| **Model Context Protocol (MCP)** | `python -m mcp.server` | Stdio JSON-RPC 2.0 |
| **Run Unit Tests (9/9)** | `python -m unittest tests/test_pipeline.py` | CLI Output (Pass/Fail) |
| **Run GPU VRAM Benchmark** | `python -m evaluation.benchmark` | `benchmarks/benchmark_report.json` |
| **fMRI Synthetic Validation** | `python -m evaluation.validate_fmri` | `validation_results/validation_report.json` |
| **Timeline Engagement Analyzer** | `python -m evaluation.timeline_analyzer --text "Script..."` | CLI ASCII Chart + JSON |
| **Train Voxel-Mode Model** | `python -m training.train --config configs/local_config.yaml --mode voxel` | Checkpoints in `checkpoints/` |
| **Train ROI-Mode Model** | `python -m training.train --config configs/roi_config.yaml --mode roi` | Checkpoints in `checkpoints_roi/` |

---

## 6. Streamlit Web Application

The interactive web dashboard provides script generation, real-time engagement graphs, peak/drop diagnostics, and 3D brain surface mapping.

### Launch Command:
```bash
streamlit run ui/app.py --server.port 8501
```

### Features:
1. **Viral Script & Retention Analyzer:**
   - Type any topic (e.g., *"The AI Bubble"*) or click **"Generate with DeepSeek"**.
   - Generates a 45–60s video script with timestamps, B-roll, voiceover text, and captions.
   - Computes second-by-second viewer cognitive engagement scores `[0 - 100]`.
   - Identifies **Hook Strength (0–3s)**, **Climax Engagement**, and **End-screen Loop Retention**.
   - Displays color-coded cards for **High Peaks** (audiences glued) and **Drop-off Risks** (pacing lulls).
2. **Multimodal Cortical BOLD Prediction:**
   - Upload video clips (`.mp4`) or audio files (`.wav`), or type text descriptions.
   - Predicts cortical responses across **360 HCP-MMP1 parcels** (ROI mode) or **20,484 vertices** (Voxel mode).
3. **Hardware & VRAM Monitor:**
   - Real-time RTX 5060 Ti memory profiling and parameter breakdown.

---

## 7. FastAPI REST API Server

A production-grade REST API server for programmatic brain prediction and retention scoring.

### Launch Command:
```bash
python -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload
```
Interactive Swagger / OpenAPI docs: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

### Key Endpoints:

#### 1. Health Check
```bash
curl -X GET "http://127.0.0.1:8000/api/v1/health"
```
**Response:**
```json
{
  "status": "healthy",
  "service": "tribe_local_api",
  "target_hardware": "NVIDIA GeForce RTX 5060 Ti (8GB VRAM)",
  "cuda_available": true,
  "deepseek_configured": true
}
```

#### 2. Timeline Engagement Scoring
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/engagement" \
     -H "Content-Type: application/json" \
     -d '{
       "text": "Is AI the next dot com crash? Billions spent on chips, but the revenue is not matching the hype.",
       "duration_secs": 30.0
     }'
```

#### 3. DeepSeek Script Generation + Instant Retention Scoring
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/generate-script" \
     -H "Content-Type: application/json" \
     -d '{
       "topic": "The AI Bubble",
       "duration_secs": 55,
       "style": "Urgent, fast-paced, high retention"
     }'
```

---

## 8. Model Context Protocol (MCP) Server

Integrate TRIBE-local directly with **Claude Desktop**, **Antigravity**, **Cursor**, or any MCP-compatible agent.

### Launch Command:
```bash
python -m mcp.server
```

### Claude Desktop Configuration:
Add the following block to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "tribe-local": {
      "command": "python",
      "args": ["-m", "mcp.server"],
      "cwd": "D:\\sakthivel workspace\\Workspace\\RnD\\TRibev2"
    }
  }
}
```

### Available MCP Tools:
- **`tribe_predict_brain`**: Predicts cortical BOLD activity from multimodal stimulus.
- **`tribe_analyze_engagement`**: Computes timeline engagement curves, hook strength, and pacing drop risks.
- **`tribe_generate_viral_script`**: Generates viral scripts via DeepSeek and evaluates cognitive engagement in one pass.
- **`tribe_get_system_status`**: Queries RTX 5060 Ti memory and VRAM headroom.

---

## 9. Timeline Engagement & Viral Script Analyzer

Run timeline pacing diagnostics directly from the command line:

```bash
python -m evaluation.timeline_analyzer \
    --config configs/roi_config.yaml \
    --text "Is AI the next dot com crash? Everyone is screaming AI is the future. We are spending billions on GPUs, but the revenue is not matching the hype. History does not repeat, but it rhymes. In 2000, it was dot-com websites. Today, it is thousands of AI wrapper apps. When the hype cools down, the costs remain. If ROI does not show up, the bubble pops. Nvidia is printing money by selling shovels. But the miners buying shovels might be digging their own graves. Is AI a revolution, or history's biggest rug pull?" \
    --duration 55.0 \
    --output timeline_report.json
```

---

## 10. Hardware Profiling & VRAM Benchmark

Benchmark memory consumption and forward latency across all components:

```bash
python -m evaluation.benchmark --output-dir benchmarks
```

### Measured Performance on RTX 5060 Ti:

| Subsystem | Total Parameters | Model Weight Size | Forward Latency | 8GB Budget Status |
|:---|:---|:---|:---|:---|
| **`fusion`** | 13,075,459 | ~52 MB | 22.7 ms | **PASS** (<120 MB VRAM) |
| **`voxel_head`** (20,484 vertices) | 21,648,612 | ~86 MB | 26.3 ms | **PASS** (<90 MB VRAM) |
| **`roi_head`** (360 parcels) | 378,000 | ~1.5 MB | 0.6 ms | **PASS** (<10 MB VRAM) |
| **`integrated_pipeline`** | **34,724,071** | **~138 MB** | **47.6 ms** | **PASS (Headroom: >7.8 GB)** |

---

## 11. Training Pipeline

### Step 1: Pre-Extract Stimulus Features (Recommended)
Pre-extracting features to disk eliminates redundant forward passes through the frozen vision and audio encoders during training:

```bash
python scripts/extract_features.py --data-root data/my_dataset --modalities video audio text
```

### Step 2: Train Voxel-Mode Model (20,484 Vertices)
```bash
python -m training.train \
    --config configs/local_config.yaml \
    --mode voxel \
    --batch-size 1 \
    --fp16 \
    --max-epochs 15
```

### Step 3: Train ROI-Mode Model (360 Parcels)
```bash
python -m training.train \
    --config configs/roi_config.yaml \
    --mode roi \
    --batch-size 2 \
    --fp16 \
    --max-epochs 15
```

---

## 12. Evaluation & fMRI Validation

Run the automated evaluation suite to compute Pearson correlation, R² score, and Top-1 retrieval accuracy against synthetic or empirical fMRI datasets:

```bash
python -m evaluation.validate_fmri --output-dir validation_results
```

---

## 13. Directory Map

```
TRibev2/
├── README.md                      # Authoritative user guide & API reference
├── DEVIATIONS.md                  # Comprehensive architectural mapping & deviations record
├── environment.yml                # Conda environment definition
├── requirements.txt               # Pip dependencies
├── setup.py                       # Python package configuration
├── .env                           # Secret environment variables (DEEPSEEK_API_KEY)
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
│
├── configs/
│   ├── local_config.yaml          # Voxel-mode config (20,484 vertices, FP16, batch size 1)
│   └── roi_config.yaml            # ROI-mode config (360 parcels, batch size 2)
│
├── encoders/
│   ├── video_encoder.py           # VideoMAE-Small encoder & 2Hz feature extractor
│   ├── audio_encoder.py           # AST Audio Spectrogram Transformer & Mel-processor
│   └── text_encoder.py            # DeepSeek API client with exponential backoff & MiniLM fallback
│
├── fusion/
│   ├── temporal_align.py          # Temporal interpolation to 2Hz + 5s HRF offset
│   ├── projectors.py              # Per-modality MLP projectors & missing token embeddings
│   └── cross_attention_fusion.py  # 4-layer Cross-Attention Transformer & low-rank head
│
├── brain/
│   ├── subject_layers.py          # FiLM-style LightweightSubjectHead & ROIHead
│   ├── losses.py                  # Composite MSE + Pearson correlation loss
│   └── metrics.py                 # Streaming OnlinePearsonCorr & per-subject metrics
│
├── data/
│   ├── dataset.py                 # PyTorch Dataset & Lightning DataModule
│   └── preprocessing.py           # fMRI z-scoring, NIfTI slicing & synthetic generator
│
├── training/
│   ├── pl_module.py               # PyTorch Lightning training wrapper
│   └── train.py                   # Training CLI with OneCycleLR & early stopping
│
├── evaluation/
│   ├── benchmark.py               # VRAM profiler & latency benchmark suite
│   ├── validate_fmri.py           # Empirical & synthetic fMRI validation runner
│   ├── inference.py               # High-level inference engine
│   └── timeline_analyzer.py       # Second-by-second timeline engagement analyzer
│
├── ui/
│   └── app.py                     # Streamlit interactive web application
│
├── api/
│   └── server.py                  # FastAPI REST API server
│
├── mcp/
│   └── server.py                  # Model Context Protocol (MCP) JSON-RPC stdio server
│
├── scripts/
│   ├── setup_wsl.sh               # WSL2 Ubuntu automated setup script
│   ├── download_algonauts.sh      # Algonauts & OpenNeuro dataset download pointers
│   ├── extract_features.py        # Feature pre-extraction script
│   └── run_script_inference.py    # Quick inference runner
│
├── tests/
│   └── test_pipeline.py           # Comprehensive unit test suite (9/9 passing)
│
└── TRIBE_original/                # Official cloned Meta TRIBE v2 repository
```

---

## 14. Troubleshooting FAQ

| Problem | Cause | Recommended Solution |
|:---|:---|:---|
| **`CUDA out of memory (OOM)`** | Batch size too large or activations un-cached | Ensure `--batch-size 1` and `--fp16` are used. Use `extract_features.py` to cache features to disk ahead of time. |
| **`No module named 'tribe_local'`** | PYTHONPATH does not include the project root | Run commands from the `TRibev2` directory or set `PYTHONPATH=.` in your terminal. |
| **`DeepSeek API 404 / Rate Limit`** | Endpoint difference or API quota | The text encoder automatically falls back to local SentenceTransformers `all-MiniLM-L6-v2`. |
| **`nvidia-smi not detected in WSL2`** | Missing Windows NVIDIA display driver | Install the latest NVIDIA GeForce Game Ready Driver on Windows. Do not install Linux drivers inside WSL2. |
| **`Port 8501 or 8000 already in use`** | A previously running server process is open | Pass a custom port: `streamlit run ui/app.py --server.port 8502` or `uvicorn api.server:app --port 8001`. |

---

## 15. Citations

```bibtex
@article{michael2026tribe,
  title={A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience},
  author={Michael, et al.},
  journal={arXiv preprint arXiv:2605.04326},
  year={2026},
  publisher={Meta AI}
}
```
