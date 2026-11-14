"""
server.py — Model Context Protocol (MCP) Server for TRIBE-local
===============================================================
Exposes TRIBE-local brain-encoding and timeline retention tools via
the standard JSON-RPC 2.0 Model Context Protocol (stdio transport).

Can be registered with Claude Desktop, Antigravity, Cursor, or any MCP client:
{
  "mcpServers": {
    "tribe-local": {
      "command": "python",
      "args": ["-m", "tribe_local.mcp.server"]
    }
  }
}
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

# Setup logging to stderr only so stdout remains pure JSON-RPC
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tribe_local.mcp")

# Lazy-loaded predictor and analyzer
_PREDICTORS: Dict[str, Any] = {}
_ANALYZER = None


def get_analyzer():
    global _PREDICTORS, _ANALYZER
    if _ANALYZER is None:
        import yaml
        from tribe_local.evaluation.inference import TribeLocalPredictor
        from tribe_local.evaluation.timeline_analyzer import TimelineEngagementAnalyzer

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
# Tool Definitions
# ------------------------------------------------------------------------------
TOOLS = [
    {
        "name": "tribe_predict_brain",
        "description": "Predict human cortical fMRI BOLD activity from multimodal stimulus (video, audio, or text script). Returns activation statistics and predicted timecourses.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Stimulus sentence, transcript, or video narration."},
                "video_path": {"type": "string", "description": "Optional absolute path to stimulus video file (.mp4)."},
                "audio_path": {"type": "string", "description": "Optional absolute path to stimulus audio file (.wav)."},
                "mode": {"type": "string", "enum": ["roi", "voxel"], "default": "roi", "description": "'roi' for 360 cortical parcels or 'voxel' for 20,484 vertices."},
                "subject_id": {"type": "integer", "default": 0, "description": "Subject index in [0, 24]."},
            },
        },
    },
    {
        "name": "tribe_analyze_engagement",
        "description": "Compute continuous second-by-second cognitive engagement curves, hook strength (0-3s), peak engaging climaxes, and attention drop-off risks across a video timeline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Video script transcript or voiceover narration."},
                "video_path": {"type": "string", "description": "Optional path to video file."},
                "duration_secs": {"type": "number", "default": 55.0, "description": "Total duration of video in seconds."},
            },
            "required": ["text"],
        },
    },
    {
        "name": "tribe_generate_viral_script",
        "description": "Generate a high-retention viral YouTube Shorts / TikTok script using DeepSeek API and immediately score its engagement curve and drop-off risks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "Topic or concept for the short-form video."},
                "duration_secs": {"type": "integer", "default": 55, "description": "Target duration in seconds (15-60)."},
                "style": {"type": "string", "default": "Urgent, fast-paced, high curiosity", "description": "Desired tone."},
            },
            "required": ["topic"],
        },
    },
    {
        "name": "tribe_get_system_status",
        "description": "Check RTX 5060 Ti hardware allocation, VRAM budget status, and DeepSeek connection.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ------------------------------------------------------------------------------
# Tool Handlers
# ------------------------------------------------------------------------------
def handle_predict_brain(arguments: Dict[str, Any]) -> Dict[str, Any]:
    predictors, _ = get_analyzer()
    mode = arguments.get("mode", "roi").lower()
    engine = predictors.get(mode, predictors["roi"])

    res = engine.predict(
        text_input=arguments.get("text"),
        video_path=arguments.get("video_path"),
        audio_path=arguments.get("audio_path"),
        subject_id=arguments.get("subject_id", 0),
    )
    bold = res["predicted_bold"]
    return {
        "status": "success",
        "mode": res["mode"],
        "shape": list(res["shape"]),
        "latency_ms": res["latency_ms"],
        "max_activation": float(bold.max()),
        "min_activation": float(bold.min()),
        "mean_activation": float(bold.mean()),
    }


def handle_analyze_engagement(arguments: Dict[str, Any]) -> Dict[str, Any]:
    _, analyzer = get_analyzer()
    from tribe_local.evaluation.timeline_analyzer import render_ascii_timeline

    text = arguments.get("text")
    duration = float(arguments.get("duration_secs", 55.0))

    report = analyzer.analyze_stimulus(
        text_input=text,
        video_path=arguments.get("video_path"),
        duration_secs=duration,
    )
    chart = render_ascii_timeline(report)

    return {
        "duration_secs": report["duration_secs"],
        "mean_engagement_score": report["mean_engagement"],
        "hook_score_0_3s": report["hook_score_0_3s"],
        "climax_score": report["climax_score"],
        "loop_retention_score": report["loop_retention_score"],
        "top_peaks": report["top_peaks"],
        "drop_off_risks": report["drop_off_risks"],
        "ascii_timeline_chart": chart,
    }


def handle_generate_viral_script(arguments: Dict[str, Any]) -> Dict[str, Any]:
    import urllib.request
    from tribe_local.evaluation.timeline_analyzer import render_ascii_timeline

    topic = arguments.get("topic", "")
    duration = int(arguments.get("duration_secs", 55))
    style = arguments.get("style", "Urgent, fast-paced")

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return {"error": "DEEPSEEK_API_KEY is not set in environment or .env."}

    prompt = (
        f"Write an ultra-engaging, high-retention {duration}-second YouTube Shorts script on: '{topic}'. "
        f"Tone: {style}. Include timestamps, B-roll, voiceover text, on-screen captions, a 3s hook, and a loop cliffhanger."
    )

    url = "https://api.deepseek.com/chat/completions"
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a master viral video retention scriptwriter."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
    }

    req_obj = urllib.request.Request(url, data=json.dumps(data).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req_obj, timeout=30) as resp:
        res_json = json.loads(resp.read().decode("utf-8"))
        script_text = res_json["choices"][0]["message"]["content"]

    _, analyzer = get_analyzer()
    engagement = analyzer.analyze_stimulus(text_input=script_text, duration_secs=float(duration))
    chart = render_ascii_timeline(engagement)

    return {
        "topic": topic,
        "script": script_text,
        "mean_engagement": engagement["mean_engagement"],
        "hook_score": engagement["hook_score_0_3s"],
        "top_peaks": engagement["top_peaks"],
        "drop_off_risks": engagement["drop_off_risks"],
        "ascii_timeline": chart,
    }


def handle_system_status() -> Dict[str, Any]:
    import torch
    cuda_avail = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Fallback"
    return {
        "target_hardware": "NVIDIA GeForce RTX 5060 Ti (8GB VRAM)",
        "cuda_available": cuda_avail,
        "device_name": gpu_name,
        "deepseek_configured": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "vram_headroom_gb": 7.8,
        "status": "ready",
    }


# ------------------------------------------------------------------------------
# JSON-RPC 2.0 Dispatcher Loop
# ------------------------------------------------------------------------------
def send_response(response: Dict[str, Any]):
    body = json.dumps(response)
    sys.stdout.write(f"{body}\n")
    sys.stdout.flush()


def run_stdio_server():
    """Main JSON-RPC 2.0 loop reading from stdin."""
    logger.info("TRIBE-local MCP Server initialized. Listening on stdin...")

    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue

            # Handle Content-Length header or raw JSON
            if line.startswith("Content-Length:"):
                length = int(line.split(":", 1)[1].strip())
                sys.stdin.readline()  # empty separator line
                content = sys.stdin.read(length)
                req = json.loads(content)
            else:
                req = json.loads(line)

            req_id = req.get("id")
            method = req.get("method")
            params = req.get("params", {})

            # 1. Initialize
            if method == "initialize":
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "tribe-local-mcp", "version": "1.0.0"},
                    },
                })

            # 2. List tools
            elif method == "tools/list":
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": TOOLS},
                })

            # 3. Call tool
            elif method == "tools/call":
                tool_name = params.get("name")
                args = params.get("arguments", {})

                try:
                    if tool_name == "tribe_predict_brain":
                        res = handle_predict_brain(args)
                    elif tool_name == "tribe_analyze_engagement":
                        res = handle_analyze_engagement(args)
                    elif tool_name == "tribe_generate_viral_script":
                        res = handle_generate_viral_script(args)
                    elif tool_name == "tribe_get_system_status":
                        res = handle_system_status()
                    else:
                        raise ValueError(f"Unknown tool: {tool_name}")

                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(res, indent=2)}]
                        },
                    })
                except Exception as e:
                    logger.error(f"Error handling tool {tool_name}: {e}")
                    send_response({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32603, "message": str(e)},
                    })

            # 4. Ping / Notifications
            elif method == "ping":
                send_response({"jsonrpc": "2.0", "id": req_id, "result": {}})
            else:
                # Unsupported method
                send_response({
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method {method} not supported"},
                })

        except Exception as e:
            logger.error(f"Fatal error in MCP server: {e}")
            break


if __name__ == "__main__":
    run_stdio_server()
