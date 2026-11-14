# Architectural Deviations from Meta TRIBE v2

> **Document Status:** Authoritative Architectural Mapping & Deviation Record  
> **Reference Model:** Meta TRIBE v2 (*"A Foundation Model of Vision, Audition, and Language for In-Silico Neuroscience"*, arXiv:2605.04326)  
> **Repository:** `facebookresearch/tribev2`  
> **Target Environment:** Single GPU (NVIDIA RTX 5060 Ti, 8GB VRAM, 32GB RAM, Windows 11 + WSL2 Ubuntu 24.04)

---

## 1. Summary of Design Rationale & 8GB Budget Constraints

Meta's official TRIBE v2 foundation model was designed for multi-node cluster training on enterprise 80GB GPUs (NVIDIA H100 / A100) using Meta's internal Slurm/`exca` orchestration and FSDP (Fully Sharded Data Parallelism). In contrast, **TRIBE-local** is an independent, modular research implementation adapted specifically for a single consumer GPU with an **8GB VRAM envelope**.

To make end-to-end multimodal brain encoding runnable on consumer hardware, every heavyweight component was re-engineered while preserving the mathematical formulation, temporal frequency alignment (2 Hz), hemodynamic lag handling (5 s HRF offset), cortical projection mesh (fsaverage5, 20,484 vertices), and loss/evaluation objectives.

---

## 2. Component-by-Component Deviation Matrix

| Subsystem | Meta TRIBE v2 (Original) | TRIBE-local (Ours) | Rationale & Hardware Justification | Expected Impact / Caveats |
| :--- | :--- | :--- | :--- | :--- |
| **Video Encoder** | **V-JEPA 2 ViT-Giant** (`facebook/vjepa2-vitg-fpc64-256`, ~1B params). Outputs `(B, 2, 1408, T)` from layers [0.75, 1.0]. | **VideoMAE-Small** (`MCG-NJU/videomae-small`, ~22M params). Outputs `(B, 384, T)` from 16-frame tubelet patches. | ViT-Giant alone requires >16GB VRAM for forward inference in FP32/FP16. VideoMAE-Small uses <150MB VRAM, allowing local execution. | VideoMAE produces 384-dim representations rather than 1408-dim. High-level semantic video features may be less detailed than ViT-Giant. |
| **Audio Encoder** | **Wav2Vec-BERT 2.0** (~300M params, 24 layers, 1024 dim). Outputs `(B, 2, 1024, T)` from layers [0.75, 1.0]. | **AST (Audio Spectrogram Transformer)** (`MIT/ast-finetuned-audioset-10-10-0.4593`, ~86M params). | AST uses patch-based 2D spectrogram vision transformers. Highly optimized on consumer CUDA, outputs 768-dim features. | Operates on 128-bin Mel spectrograms chunked at 10.24s rather than raw audio waveforms with Conformer blocks. |
| **Text Encoder** | **LLaMA 3.2-3B** (~3.2B params, 28 layers, 3072 dim). Outputs `(B, 6, 3072, T)` from 6 sampled layers. | **DeepSeek API** (`deepseek-embedding`, 1536 dim) + local **MiniLM-L6-v2** fallback (384 dim). | A 3.2B local LLM requires 6.5GB VRAM just for weights, causing immediate OOM on an 8GB GPU alongside video and audio models. | Embeddings are computed via external API or compact sentence transformer; no token-by-token causal transformer activations in VRAM. |
| **Modality Fusion** | **TransformerEncoder** (depth=8, hidden=1152) preceded by simple concatenated linear projection. Combiner is `Identity`. | **Cross-Attention Transformer** (depth=4, hidden=512) with explicit pairwise cross-attention (V↔A, V↔T, A↔T) and gated fusion. | Hidden dimension reduced from 1152 to 512, depth from 8 to 4 to cut transformer activation memory by ~70%. Pairwise cross-attention compensates for smaller depth. | Parameter count of fusion reduced from ~130M to ~18M. |
| **Bottleneck Dimension** | `low_rank_head` Linear mapping `1152 → 2048`. | `low_rank_head` Linear mapping `512 → 1024`. | Reduces bottleneck dimension `r` by 50% to save memory in subsequent subject-specific projection layers. | Half the bottleneck representation rank compared to TRIBE v2. |
| **Subject Layers** | **Monolithic Parameter Tensor** `(n_subjects, 2048, 20484)`. For 25 subjects = 1.05 × 10^9 weights = **4.2 GB VRAM for weights alone!** | **Lightweight Decomposed Head**: One shared decoder `(1024, 20484)` + per-subject FiLM style embeddings `(n_subj, 128)` and scale/bias vectors. | Eliminates the single largest memory bottleneck in TRIBE v2. Cuts prediction head parameters from 4.2 GB to ~85 MB. | Forces cross-subject shared subspace representation with individual modulation, rather than fully independent vertex matrices. |
| **Alternative Head** | Cortical surface vertex-wise only (`20,484` vertices). | Dual-mode: **Voxel Mode** (`20,484` vertices) + **ROI Mode** (`360` HCP-MMP1 parcels). | Enables ultra-fast prototyping, debugging, and lower VRAM usage (<1GB) in ROI mode. | ROI mode outputs parcel-averaged signals rather than individual cortical vertex responses. |
| **Batch Size & Precision** | `batch_size = 8`, FP32 or bfloat16. | `batch_size = 1`, FP16 mixed precision (`16-mixed`) + gradient checkpointing. | Required to guarantee fit in 8GB VRAM during backward passes without out-of-memory errors. | Smaller batch size requires gradient clipping (`1.0`) and OneCycleLR with appropriate step scaling. |
| **Orchestration Infra** | Slurm grid engine + Meta internal `exca` library (`exca.MapInfra`, `exca.TaskInfra`). | Standard **PyTorch Lightning** (`pl.Trainer`) runnable on local Windows/WSL2. | Meta's `exca` infra is private/proprietary and only works on Meta's internal Slurm clusters. | Eliminates cluster dependency; enables single-command CLI execution. |
| **Feature Caching** | Handled dynamically via `exca.MapInfra(mode="cached")`. | Dedicated local cache in `~/.tribe_cache/{video,audio,text}/` keyed by MD5/SHA256 hashes. | Allows one-time stimulus feature extraction on CPU/GPU, eliminating encoder runtime during fusion training. | Stimulus files are pre-computed once; training loop only runs lightweight fusion and head. |

---

## 3. Explicit Disclaimer Regarding Model Accuracy

> [!WARNING]
> **No Claim of Original Accuracy:**
> This repository is an independent, lightweight implementation inspired by the architecture and mathematical principles of Meta TRIBE v2. Because all three foundation encoders (V-JEPA2, Wav2Vec-BERT, LLaMA) have been replaced with lighter alternatives, and the fusion/subject-head capacity has been scaled down to fit an 8GB VRAM budget, **no claim is made that this implementation matches the Pearson correlation benchmarks reported in the official TRIBE v2 paper (arXiv:2605.04326).**
>
> Quantitative performance will depend entirely on local fine-tuning, dataset scale, and empirical validation against the provided benchmarks.
