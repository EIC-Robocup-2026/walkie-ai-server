# 🤖 Walkie AI Server

> 🚀 A multimodal AI inference server powered by Flask, serving vision, speech, and language models through a clean REST API.

---

## 📖 Overview

**Walkie AI Server** is a self-hosted Flask application that bundles multiple AI capabilities into a single service. It uses a **provider-based architecture** so you can swap between local and cloud backends for each feature — run everything on your own GPU or delegate to cloud APIs. ☁️⚡

---

## ✨ Features

| Feature | Description | Providers |
|---------|-------------|-----------|
| **Speech-to-Text** | Transcribe audio to text | `whisper` (local), `google` (Cloud Speech) |
| **Text-to-Speech** | Synthesize natural speech from text | `piper` (local ONNX), `elevenlabs` (cloud) |
| **Object Detection** | Detect & classify objects in images | `yolo` (Ultralytics / Objects365), `sam3` (open-vocab concept segmentation + masks), `yoloe` (open-vocab detect + segmentation, text prompts or prompt-free) |
| **Pose Estimation** | Detect human body keypoints (17 COCO) | `yolo_pose` (Ultralytics) |
| **Image Captioning** | Generate captions / answer visual questions | `florence2`, `paligemma`, `google` (Gemini) |
| **Face Recognition** | Detect faces & return L2-normalized embeddings for re-ID | `insightface` (RetinaFace + ArcFace `buffalo_l`) |
| **LLM Serving** | Optional vLLM / Ollama sidecar | Qwen 3.5-9B (quantized) |


## 🚀 Getting Started

### 📋 Prerequisites

- 🐍 Python **3.12+**
- 📦 [uv](https://docs.astral.sh/uv/) package manager
- 🎮 NVIDIA GPU recommended (CUDA-capable) for local inference

### ⚙️ Installation

```bash
# 1️⃣ Clone the repository
git clone https://github.com/your-org/walkie-ai-server.git
cd walkie-ai-server

# 2️⃣ Install dependencies with uv
uv sync

# 3️⃣ Set up environment variables
cp .env.example .env  # edit with your API keys
```

### 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ELEVENLABS_API_KEY` | 🟡 Optional | ElevenLabs TTS API key |
| `GOOGLE_API_KEY` | 🟡 Optional | Google Gemini API key |
| `GOOGLE_CLOUD_PROJECT` | 🟡 Optional | GCP project for Cloud Speech |
| `GOOGLE_APPLICATION_CREDENTIALS` | 🟡 Optional | Path to GCP service account JSON |
| `HF_TOKEN` | 🟡 Optional | Hugging Face token for gated models |
| `OBJECT_DETECTION_PROVIDER` | 🟡 Optional | Object-detection backend: `yolo` (default), `sam3`, or `yoloe` |
| `SAM3_MODEL` | 🟡 Optional | Path to `sam3.pt` weights (required when provider is `sam3`) |
| `YOLOE_MODEL` | 🟡 Optional | YOLOE text-prompt checkpoint (default `yoloe-11m-seg.pt`, auto-downloads) |
| `YOLOE_PF_MODEL` | 🟡 Optional | YOLOE prompt-free checkpoint (default `yoloe-11m-seg-pf.pt`, auto-downloads) |

> 💡 Only needed if you use the corresponding cloud providers. Local-only setups require no API keys!

### ▶️ Running the Server

```bash
# 🟢 Start the Flask API server on port 5000
./scripts/run_app.sh
```

```bash
# 🧩 (Optional) Start with SAM3 open-vocab object detection
SAM3_MODEL=/path/to/sam3.pt ./scripts/run_sam3.sh
```

```bash
# 🧠 (Optional) Start the LLM sidecar on port 8000
./scripts/serve_llm.sh          # vLLM (default)
./scripts/serve_llm.sh ollama   # or Ollama
```


## 📡 API Reference

All endpoints return JSON in the format:

```json
{ "success": true, "data": { ... } }
```

Or on error:

```json
{ "success": false, "error": "description" }
```

### 🏠 Index

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | 📋 List available models & endpoints |

### 🎤 Speech-to-Text (`/stt`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/stt/providers` | 📋 List STT providers |
| `POST` | `/stt/transcribe` | 🎙️ Transcribe audio file (multipart `audio`) |

### 🔊 Text-to-Speech (`/tts`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/tts/providers` | 📋 List TTS providers |
| `POST` | `/tts/synthesize` | 🗣️ Synthesize speech (returns audio bytes) |
| `POST` | `/tts/synthesize-stream` | 🌊 Streamed synthesis |

### 🔍 Object Detection (`/object-detection`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/object-detection/providers` | 📋 List detection providers |
| `POST` | `/object-detection/detect` | 📦 Detect objects in image (multipart `image`; optional `prompts`, optional `return_mask`) |

> 🧩 **SAM3 (open-vocabulary):** when running with `OBJECT_DETECTION_PROVIDER=sam3`, pass text concepts via a `prompts` form field (comma-separated or repeated) to find arbitrary objects, e.g. `-F prompts="red mug,cereal box"`. YOLO ignores `prompts`.
>
> 🧠 **YOLOE (open-vocabulary):** with `OBJECT_DETECTION_PROVIDER=yoloe`, `prompts` steer a text-prompt model; when no prompts are given it falls back to a prompt-free open-vocabulary model. Both checkpoints auto-download.
>
> 🎭 **Masks:** add `-F return_mask=true` to include a base64 PNG segmentation `mask_b64` per detection (default `false` → bbox only). SAM3 and YOLOE produce masks; YOLO only does so with a `-seg` checkpoint (otherwise it warns and returns a null mask). `bbox` is always returned.

### 🏃 Pose Estimation (`/pose-estimation`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/pose-estimation/providers` | 📋 List pose providers |
| `POST` | `/pose-estimation/estimate` | 🦴 Estimate body keypoints (multipart `image`) |

### 🖼️ Image Captioning (`/image-caption`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/image-caption/providers` | 📋 List caption providers |
| `POST` | `/image-caption/caption` | 💬 Caption a single image |
| `POST` | `/image-caption/caption-batch` | 📚 Caption multiple images |

### 🧑 Face Recognition (`/face-recognition`)

Stateless face detection + embedding for person re-identification (RoboCup @Home
Receptionist). Image in → per face: an `xyxy` bbox, an **L2-normalized** embedding
(constant dim, e.g. 512), and a detection score. No names, no database, no matching
on the server — the agent owns enrollment and cosine-distance matching.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/face-recognition/providers` | 📋 List face providers |
| `GET` | `/face-recognition/info` | 🪪 Model name + embedding dim (vector provenance) |
| `POST` | `/face-recognition/embed` | 🧑 Detect faces & embed each (multipart `image`) |

`/embed` returns `data: [{ "bbox_xyxy": [x1,y1,x2,y2], "embedding": [...], "det_score": 0.99 }, ...]`
— `[]` (with `success: true`) when no face is present.

> **GPU note:** InsightFace runs on the CPU `onnxruntime` already pulled in
> transitively. For real-time speed on the GPU box, replace it with
> `onnxruntime-gpu` (`uv pip install onnxruntime-gpu`, after removing `onnxruntime`)
> — both expose the same `onnxruntime` module and must not be installed together.
> The provider auto-selects GPU (`ctx_id=0`) when a CUDA execution provider is available.

---

## 🧪 Testing

Tests are integration-style and require a **running server**:

```bash
# 1️⃣ Start the server in one terminal
./scripts/run_app.sh

# 2️⃣ Run tests in another terminal
uv run pytest tests/

# 🎯 Run a specific test module
uv run pytest tests/test_stt.py

# 🌐 Test against a different host
uv run pytest tests/ --base-url http://your-server:5000
```

## 🏗️ Architecture

```
walkie-ai-server/
├── app.py                  # Flask entrypoint (port 5000)
├── pyproject.toml          # Dependencies & project metadata
├── uv.lock                 # Locked dependency versions
│
├── 📂 api/
│   ├── __init__.py            # App factory & blueprint registration
│   ├── utils.py               # JSON helpers, image decoding, base64
│   └── 📂 routes/             # One blueprint per feature
│       ├── stt.py             # 🎤 /stt/*
│       ├── tts.py             # 🔊 /tts/*
│       ├── object_detection.py# 🔍 /object-detection/*
│       ├── pose_estimation.py # 🏃 /pose-estimation/*
│       ├── image_caption.py   # 🖼️ /image-caption/*
│       ├── image_embed.py     # 🔗 /image-embed/* (disabled)
│       └── face_recognition.py# 🧑 /face-recognition/*
│
├── 📂 services/               # Provider pattern — abstract base + implementations
│   ├── stt/
│   │   ├── base.py
│   │   └── providers/         # whisper.py, google.py
│   ├── tts/
│   │   ├── base.py
│   │   └── providers/         # piper_tts.py, elevenlabs.py
│   ├── object_detection/
│   │   ├── base.py
│   │   └── providers/         # yolo.py, sam3.py, yoloe.py
│   ├── pose_estimation/
│   │   ├── base.py
│   │   └── providers/         # yolo_pose.py
│   ├── image_caption/
│   │   ├── base.py
│   │   └── providers/         # florence2_large.py, paligemma.py, google_caption.py
│   ├── image_embed/
│   │   ├── base.py
│   │   └── providers/         # clip.py
│   └── face_recognition/
│       ├── base.py
│       └── providers/         # insightface_provider.py
│
├── 📂 scripts/
│   ├── run_app.sh             # 🟢 Start the Flask server
│   ├── run_sam3.sh            # 🧩 Start with SAM3 object detection
│   └── serve_llm.sh           # 🧠 Start vLLM / Ollama sidecar
│
├── 📂 tests/                  # 🧪 Integration tests (pytest + requests)
│   ├── conftest.py
│   ├── test_stt.py
│   ├── test_tts.py
│   ├── test_object_detection.py
│   ├── test_pose_estimation.py
│   ├── test_image_caption.py
│   └── test_face_recognition.py
│
└── 📂 voices/                 # 🗣️ Piper TTS voice assets (.onnx)
```

---

<div align="center">

🛠️ Built with ❤️ by **Your Mom**

</div>
