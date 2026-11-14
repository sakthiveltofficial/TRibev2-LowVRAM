"""
server.py — FastAPI REST API Server for TRIBE-local
===================================================
Exposes high-performance RESTful API endpoints for:
- Multimodal brain-encoding inference
- Second-by-second timeline engagement scoring
- DeepSeek viral script generation with instant retention analysis
- GPU hardware profiling & health monitoring
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "tribe_local"))
sys.path.insert(0, str(PROJECT_ROOT.parent))

import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from tribe_local.evaluation.inference import TribeLocalPredictor
    from tribe_local.evaluation.timeline_analyzer import TimelineEngagementAnalyzer
except ImportError:
    from evaluation.inference import TribeLocalPredictor
    from evaluation.timeline_analyzer import TimelineEngagementAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("tribe_local.api")

# Load environment
env_file = PROJECT_ROOT / ".env"
if not env_file.exists():
    env_file = PROJECT_ROOT / "tribe_local" / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.startswith("#") and "=" in line:
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)

# Initialize FastAPI App
app = FastAPI(
    title="TRIBE-local Brain Encoding & Retention API",
    description="REST API for in-silico multimodal brain encoding (RTX 5060 Ti 8GB Optimized)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
_PREDICTORS: Dict[str, TribeLocalPredictor] = {}
_ANALYZER: Optional[TimelineEngagementAnalyzer] = None


def get_models():
    global _PREDICTORS, _ANALYZER
    if not _PREDICTORS:
        roi_cfg_p = PROJECT_ROOT / "configs" / "roi_config.yaml"
        voxel_cfg_p = PROJECT_ROOT / "configs" / "local_config.yaml"
        if not roi_cfg_p.exists():
            roi_cfg_p = PROJECT_ROOT / "tribe_local" / "configs" / "roi_config.yaml"
        if not voxel_cfg_p.exists():
            voxel_cfg_p = PROJECT_ROOT / "tribe_local" / "configs" / "local_config.yaml"

        with open(roi_cfg_p, "r") as f:
            roi_cfg = yaml.safe_load(f)
        with open(voxel_cfg_p, "r") as f:
            voxel_cfg = yaml.safe_load(f)

        device = "cuda" if os.environ.get("USE_CUDA", "").lower() in ("1", "true") else "cpu"
        _PREDICTORS["roi"] = TribeLocalPredictor(config=roi_cfg, device=device)
        _PREDICTORS["voxel"] = TribeLocalPredictor(config=voxel_cfg, device=device)
        _ANALYZER = TimelineEngagementAnalyzer(_PREDICTORS["roi"])
    return _PREDICTORS, _ANALYZER


# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: Optional[str] = Field(None, description="Text prompt or speech transcript")
    video_path: Optional[str] = Field(None, description="Absolute path to video file (.mp4)")
    audio_path: Optional[str] = Field(None, description="Absolute path to audio file (.wav)")
    mode: str = Field("roi", description="'roi' (360 parcels) or 'voxel' (20484 vertices)")
    subject_id: int = Field(0, description="Subject ID index in [0, 24]")


class EngagementRequest(BaseModel):
    text: Optional[str] = Field(None, description="Video script voiceover or transcript")
    video_path: Optional[str] = Field(None, description="Optional path to video file")
    audio_path: Optional[str] = Field(None, description="Optional path to audio track")
    duration_secs: float = Field(55.0, description="Target timeline duration in seconds")
    subject_id: int = Field(0, description="Subject index")


class ScriptGenerationRequest(BaseModel):
    topic: str = Field(..., description="Topic of YouTube Short / TikTok video")
    duration_secs: int = Field(55, description="Target video duration in seconds")
    style: str = Field("Urgent, slightly controversial, fast-paced", description="Script tone and pacing")


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------
@app.get("/api/v1/health")
def health_check():
    """System health check and GPU hardware status."""
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Fallback"
    return {
        "status": "healthy",
        "service": "tribe_local_api",
        "target_hardware": "NVIDIA GeForce RTX 5060 Ti (8GB VRAM)",
        "cuda_available": cuda_avail,
        "device": gpu_name,
        "deepseek_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
    }


@app.post("/api/v1/predict")
def predict_brain_activity(req: PredictRequest):
    """Predict cortical fMRI BOLD activity from multimodal stimulus."""
    predictors, _ = get_models()
    mode = req.mode.lower()
    if mode not in predictors:
        raise HTTPException(status_code=400, detail=f"Invalid mode '{mode}'. Choose 'roi' or 'voxel'.")

    pred_engine = predictors[mode]
    res = pred_engine.predict(
        text_input=req.text,
        video_path=req.video_path,
        audio_path=req.audio_path,
        subject_id=req.subject_id,
    )

    bold = res["predicted_bold"]
    return {
        "mode": res["mode"],
        "shape": list(res["shape"]),
        "latency_ms": res["latency_ms"],
        "max_activation": float(np.max(bold)),
        "min_activation": float(np.min(bold)),
        "mean_activation": float(np.mean(bold)),
        # Return first 10 timepoints of first 5 regions for lightweight JSON
        "sample_activation_slice": bold[:5, :10].tolist(),
    }


@app.post("/api/v1/engagement")
def analyze_engagement_timeline(req: EngagementRequest):
    """Compute second-by-second timeline engagement and drop-off risks."""
    _, analyzer = get_models()
    if not req.text and not req.video_path and not req.audio_path:
        raise HTTPException(status_code=400, detail="Must provide at least one stimulus input (text, video, or audio).")

    report = analyzer.analyze_stimulus(
        text_input=req.text,
        video_path=req.video_path,
        audio_path=req.audio_path,
        duration_secs=req.duration_secs,
        subject_id=req.subject_id,
    )
    return report


@app.post("/api/v1/generate-script")
def generate_and_analyze_script(req: ScriptGenerationRequest):
    """Generate viral YouTube Short script via DeepSeek and instantly score engagement."""
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not configured in environment.")

    prompt = (
        f"You are a master viral YouTube Shorts scriptwriter. "
        f"Write an ultra-engaging, high-retention {req.duration_secs}-second script on: '{req.topic}'. "
        f"Tone and style: {req.style}. "
        f"Format with: Time stamps, Visual B-Roll, Voiceover audio, and On-screen captions. "
        f"Include a powerful 3-second hook and an infinite loop cliffhanger at the end."
    )

    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a viral YouTube Shorts scriptwriting and audience retention expert."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
    }

    try:
        req_obj = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
        with urllib.request.urlopen(req_obj, timeout=30) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            script_content = res_json["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"DeepSeek API error: {e}")

    # Immediately score engagement on the voiceover text
    _, analyzer = get_models()
    engagement_report = analyzer.analyze_stimulus(
        text_input=script_content,
        duration_secs=float(req.duration_secs),
    )

    return {
        "topic": req.topic,
        "duration_secs": req.duration_secs,
        "script": script_content,
        "engagement_analysis": engagement_report,
    }


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Run uvicorn server."""
    logger.info(f"Starting TRIBE-local REST API server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()
