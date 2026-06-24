# Pharmacy Surveillance AI — Audio-First Medical Insurance Fraud Detection

> An open-source, multi-modal pipeline for detecting medical-insurance fraud (medication substitution, cashback, multi-card swiping, over-dispensing) in pharmacy surveillance video + audio. Three iterative prototypes, each independently runnable, tracing a path from "can the model see?" to "can the model hear?" to "can it build an evidence chain?".

[中文文档](README.md) | English

---

## What This Project Does

Medical-insurance regulators need to audit hours of pharmacy surveillance footage to catch fraud patterns: swapping covered drugs for health supplements, cashing out insurance balances, swiping multiple cards, over-dispensing. Watching every minute by hand is impossible.

This project builds three progressively stronger detection prototypes:

| Version | Codename | Core Capability | Key Tech |
|---------|----------|----------------|----------|
| v1 | Smoke Test | "Can the model understand a pharmacy video?" — upload a short clip, ask a question, get a structured risk assessment | Qwen3-VL-Plus multimodal LLM |
| v2 | Behavior Detect | "Can we proactively find suspicious moments in long video?" — YOLO-World scans at 5x+ speed, triggers FFmpeg slicing on crowd / carry / loiter | YOLO-World + Supervision ByteTrack |
| v3 | Audio Scan | "Can audio lead the way?" — extract the audio track, VAD-filter silence, ASR transcribe, two-stage risk funnel (keyword + LLM), slice video at keyword timestamps | Fun-ASR + pypinyin + Qwen LLM |

All three versions share the same frontend pattern (single-file HTML, zero build step) and a FastAPI backend. Each is a standalone web app you can run with one command.

---

## Architecture Overview

```
                 v1 Smoke Test                v2 Behavior Detect           v3 Audio Scan
              (model capability)             (visual triggers)            (audio-first)
  ┌─────────────────────────────┐  ┌─────────────────────────────┐  ┌─────────────────────────────┐
  │  Upload video + question    │  │  Upload / path to long video│  │  Upload / path to long video│
  │          │                  │  │          │                  │  │          │                  │
  │          ▼                  │  │          ▼                  │  │          ▼                  │
  │  Qwen3-VL-Plus API          │  │  YOLO-World (5x speed scan) │  │  FFmpeg extract audio track │
  │  (video + prompt → JSON)    │  │  + Supervision ByteTrack    │  │          │                  │
  │          │                  │  │          │                  │  │          ▼                  │
  │          ▼                  │  │  3 triggers: crowd/carry/   │  │  Silero VAD (skip silence)  │
  │  Structured risk report     │  │  loiter → FFmpeg slice      │  │          │                  │
  │  (scene_summary, answer,    │  │          │                  │  │          ▼                  │
  │  risk_hint, observations)   │  │          ▼                  │  │  Fun-ASR (word timestamps)  │
  │                             │  │  Annotated clips + events   │  │          │                  │
  │                             │  │  live preview stream        │  │          ▼                  │
  │                             │  │                             │  │  Stage 1: pypinyin keywords │
  │                             │  │                             │  │  Stage 2: Qwen LLM context  │
  │                             │  │                             │  │          │                  │
  │                             │  │                             │  │          ▼                  │
  │                             │  │                             │  │  FFmpeg boundary-clamped    │
  │                             │  │                             │  │  video slice at keyword ts  │
  └─────────────────────────────┘  └─────────────────────────────┘  └─────────────────────────────┘
```

Full design docs (Chinese) are in the `docs/` folder: original business requirements, technical architecture, and development log.

---

## Project Structure

```
pharmacy-surveillance-ai/
├── README.md                        # This file (Chinese)
├── README_EN.md                     # English README
├── LICENSE                          # MIT License
├── .gitignore
├── start_all.bat                    # One-click launcher (Windows)
├── docs/
│   ├── 原始业务需求.txt               # Original business requirements
│   ├── 技术架构设计.txt               # Technical architecture design
│   └── 开发过程.txt                   # Development log
├── v1_smoke_test/                   # Version 1: Qwen3-VL video Q&A
│   ├── .env.example
│   ├── .gitignore
│   ├── backend/
│   │   ├── main.py                  # FastAPI: /api/health, /api/analyze
│   │   ├── qwen_client.py           # Qwen3-VL-Plus client + prompt design
│   │   ├── config.py                # Env-based settings
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html               # Single-page UI (zero dependencies)
├── v2_behavior_detect/              # Version 2: YOLO-World behavior triggers
│   ├── .gitignore
│   ├── start_server.bat
│   ├── backend/
│   │   ├── main.py                  # FastAPI + threaded task manager
│   │   ├── scanner.py               # YOLO-World + Supervision scan engine
│   │   ├── slicer.py                # FFmpeg clip slicer
│   │   ├── config.py                # Trigger thresholds, model paths
│   │   └── requirements.txt
│   └── frontend/
│       └── index.html               # Upload + live preview + clip gallery
└── v3_audio_scan/                   # Version 3: Audio-first keyword scan
    ├── .env.example
    ├── .gitignore
    ├── start_server.bat
    ├── backend/
    │   ├── main.py                  # FastAPI + AudioScanTask orchestrator
    │   ├── audio_pipeline.py        # FFmpeg audio extract + Silero VAD
    │   ├── asr_client.py            # Fun-ASR WebSocket client + mock
    │   ├── keyword_matcher.py       # pypinyin keyword matching (Stage 1)
    │   ├── risk_llm.py              # Qwen LLM context risk control (Stage 2)
    │   ├── scanner.py               # Pipeline orchestrator + task state
    │   ├── slicer.py                # FFmpeg boundary-clamped slicer
    │   ├── config.py                # Env-based settings
    │   └── requirements.txt
    └── frontend/
        └── index.html               # Pipeline stages + live transcript + clips
```

---

## Prerequisites

1. **Python 3.10+** (developed and tested on Python 3.14)
2. **FFmpeg + FFprobe** installed and on your system `PATH` (required by v2 and v3)
3. **A DashScope (Aliyun Qwen) API key** — get one at https://dashscope.console.aliyun.com/
4. **YOLO-World weights** (only for v2): `yolov8s-world.pt` — download from the Ultralytics releases and place it in a `weights/` folder at the repo root (or set the `WEIGHTS_PATH` env var)

### Create a virtual environment (recommended)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## Quick Start

### Version 1 — Smoke Test (Qwen3-VL video understanding)

```powershell
cd v1_smoke_test\backend
pip install -r requirements.txt

# Configure your API key
copy ..\.env.example ..\.env
# Edit .env and set DASHSCOPE_API_KEY=sk-your-key

# Launch
python -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Open http://127.0.0.1:8001/ in your browser. Upload a short pharmacy video (MP4, ≤50MB), optionally type a question, and click "Start Analysis". The backend sends the video to Qwen3-VL-Plus and returns a structured JSON report: `scene_summary`, `answer`, `risk_hint` (none/suspect/violation), and `raw_observations`.

### Version 2 — Behavior Detection (YOLO-World visual triggers)

```powershell
cd v2_behavior_detect\backend
pip install -r requirements.txt

# Make sure yolov8s-world.pt is available (default: ../../weights/yolov8s-world.pt)
# Or set WEIGHTS_PATH env var to its location

# Launch
python -m uvicorn main:app --host 0.0.0.0 --port 8001
```

Open http://127.0.0.1:8001/. Upload a long video (or enter its absolute path). The scanner samples 1 frame per 15 (≈2 fps on 30 fps video), runs YOLO-World detection, and tracks persons with ByteTrack. Three triggers fire:

- **Crowd**: ≥12 persons for ≥3 consecutive sampled frames
- **Carry**: a target object (phone/backpack/hat) inside an expanded person box for ≥2 frames
- **Loiter**: a tracked person sitting on/near a chair for ≥2 seconds

On each trigger (with a 15s cooldown), FFmpeg slices a clip centered on the trigger timestamp and saves it to `clips/`. The frontend shows a live annotated-frame preview and a gallery of clipped videos.

### Version 3 — Audio Scan (audio-first keyword detection)

```powershell
cd v3_audio_scan\backend
pip install -r requirements.txt

# Configure your API key
copy ..\.env.example ..\.env
# Edit .env: set DASHSCOPE_API_KEY, ASR_MODEL, LLM_MODEL

# Launch
python -m uvicorn main:app --host 0.0.0.0 --port 8002
```

Open http://127.0.0.1:8002/. Upload a long video. The pipeline:

1. **Extract audio** — FFmpeg extracts the full soundtrack to 16kHz mono WAV
2. **VAD pre-filter** — Silero VAD scans 5-minute windows; windows with <5s of speech are skipped (saves ASR tokens)
3. **ASR transcription** — Fun-ASR (realtime WebSocket) transcribes each speech window with word-level timestamps (ms)
4. **Stage 1 keyword match** — pypinyin syllable matching against a keyword lexicon (exact pinyin match, tolerant of ASR homophone errors)
5. **Stage 2 LLM risk control** — Qwen LLM reviews the keyword context and classifies as "confirmed" / "coded language" / "false positive"
6. **Video slicing** — FFmpeg cuts a boundary-clamped clip around each confirmed keyword's absolute timestamp

The frontend shows pipeline stage progress, a live streaming transcript, and a gallery of clips with matched keywords and risk verdicts.

> **Demo mode** (`DEMO_MODE=1`, default): in addition to the production keyword lexicon, any VAD-positive chunk triggers a slice — useful for testing the timestamp alignment pipeline on non-pharmacy videos. Set `MOCK_ASR=1` to inject a canned transcript with fraud keywords (no ASR API quota needed).

---

## API Reference

All three versions expose a consistent REST API:

### v1 Smoke Test

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (API key configured? max video size? fps?) |
| `POST` | `/api/analyze` | Upload video + optional question → structured risk report |

### v2 Behavior Detect

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (weights loaded? trigger thresholds?) |
| `POST` | `/api/analyze` | Upload video file → starts scan, returns `task_id` |
| `POST` | `/api/analyze/path` | Submit local video path → starts scan, returns `task_id` |
| `GET` | `/api/task/{task_id}` | Poll task status (progress, triggers, clips, events) |
| `GET` | `/api/task/{task_id}/frame/{idx}` | Get annotated frame JPEG (use `latest` for most recent) |
| `GET` | `/api/clips/{filename}` | Download a clipped video |

### v3 Audio Scan

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check (API key? ASR/LLM models? demo mode?) |
| `POST` | `/api/analyze` | Upload video file → starts pipeline, returns `task_id` |
| `POST` | `/api/analyze/path` | Submit local video path → starts pipeline, returns `task_id` |
| `GET` | `/api/task/{task_id}` | Poll task status (stage, progress, clips, events) |
| `GET` | `/api/task/{task_id}/transcript` | Get live streaming transcript snapshot |
| `GET` | `/api/clips/{filename}` | Download a clipped video |

---

## Key Design Decisions

### Why three separate versions?

This project follows a "small-step, fast-iteration" philosophy: each version validates one core hypothesis before building the next. v1 proves the model can understand pharmacy video. v2 proves local vision can find suspicious moments cheaply. v3 proves audio can lead detection at lower cost than full-video vision. Together they form a complete evidence-chain pipeline.

### Why YOLO-World (not YOLO11)?

Standard COCO YOLO only detects "person" (1 of 5 pharmacy-relevant targets). YOLO-World supports open-vocabulary classes — you can set custom categories like `card`, `box`, `bottle`, `phone` at runtime. The trade-off: it works well for behavior-level detection (counting, grouping, carrying) but is unreliable for fine object recognition (hands, insurance cards, POS terminals) — those need fine-tuning in a future version.

### Why pypinyin for keyword matching?

ASR can mishear "串换" (swap) as "川换". By converting both the transcript and keywords to pinyin syllables and matching on syllable sequences, we tolerate homophone errors without fuzzy string matching. This is Stage 1 — fast, local, high-recall. Stage 2 sends the surrounding context to an LLM for precision.

### Why VAD before ASR?

A 24-hour pharmacy recording has long silent stretches. Sending silence to ASR wastes tokens and time. Silero VAD runs locally, skips windows with <5s of speech, and only speech-active windows go to the paid ASR API — cutting cost by ~80% on typical recordings.

### Boundary clamping for video slicing

If a keyword fires at second 30, `start = max(0, 30 - 60)` would be negative. The slicer clamps: `Start = max(0, T - half)`, `End = min(duration, T + half)`, and caps total length at `SLICE_MAX`. Verified on a 964s video: a 30s hit → [0, 90]; a 950s hit → [890, 964.5].

---

## Configuration

Each version loads settings from a `.env` file (see `.env.example`). Key variables:

### v1 (`v1_smoke_test/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | — | Your DashScope API key (required) |
| `QWEN_VL_MODEL` | `qwen3-vl-plus` | Vision-language model name |
| `QWEN_VL_FPS` | `2` | Sampling frames per second |
| `MAX_VIDEO_MB` | `50` | Max upload size |

### v2 (`v2_behavior_detect/backend/config.py`)
Configured via environment variables or code defaults. Key knobs:
| Variable | Default | Description |
|----------|---------|-------------|
| `WEIGHTS_PATH` | `../../weights/yolov8s-world.pt` | YOLO-World weights location |
| `CROWD_THRESHOLD` | `12` | Min persons for crowd trigger |
| `CARRY_SUSTAIN_FRAMES` | `2` | Frames a carry must persist |
| `LOITER_SECONDS` | `2.0` | Sitting duration for loiter trigger |
| `CLIP_HALF` | `5` | Seconds before/after trigger for slicing |
| `COOLDOWN` | `15` | Min seconds between same-trigger slices |

### v3 (`v3_audio_scan/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `DASHSCOPE_API_KEY` | — | Your DashScope API key (required) |
| `ASR_MODEL` | `fun-asr-realtime-2026-02-28` | Fun-ASR model |
| `LLM_MODEL` | `qwen3.5-flash` | Stage-2 risk LLM (falls back to `qwen-plus`) |
| `CHUNK_SECONDS` | `300` | Audio chunk window size |
| `VAD_MIN_SPEECH` | `5` | Min speech seconds to keep a chunk |
| `SLICE_HALF` | `10` | Seconds before/after keyword for video slice |
| `DEMO_MODE` | `1` | Enable any-voice triggers for testing |
| `MOCK_ASR` | `0` | Inject canned transcript (no API needed) |

---

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **Vision (v2)**: Ultralytics YOLO-World, Supervision (ByteTrack), OpenCV
- **Audio (v3)**: Silero VAD, Fun-ASR (DashScope WebSocket), pypinyin
- **LLM (v1, v3)**: Qwen3-VL-Plus, Qwen3.5-Flash / Qwen-Plus (via DashScope OpenAI-compatible API)
- **Media**: FFmpeg / FFprobe (audio extraction, video slicing)
- **Frontend**: Vanilla HTML/CSS/JS (single file, no build step)

---

## Known Limitations

- v1: single video call per request, ~45s latency, no ASR / YOLO / billing-data integration
- v2: YOLO-World is unreliable for fine objects (cards, POS terminals); CPU inference is slow on long videos
- v3: ASR is streaming WebSocket-based; the Fun-ASR model name may change across DashScope versions
- All versions: no authentication, no persistent storage, no PDF report generation — these are demo prototypes, not production systems
- The keyword lexicon in v3 is currently tuned for a market-scene test video (`买菜交易` category); for real pharmacy use, uncomment the 6-category fraud lexicon in `keyword_matcher.py`

---

## License

MIT License — see [LICENSE](LICENSE). Free to use, modify, and distribute.

## Contributing

This is a research/demo project. Issues and pull requests are welcome, especially for:
- Expanding the fraud keyword lexicon for real pharmacy scenarios
- Fine-tuning YOLO-World for pharmacy-specific objects (insurance cards, drug boxes)
- Adding PDF evidence-report generation
- Production hardening (auth, persistence, async task queues)