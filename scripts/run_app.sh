#!/bin/bash
# scripts/run_app.sh — run the Flask app with uv (project venv / deps from pyproject.toml)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Ensure the venv + deps exist before we probe it for CUDA libs below.
# uv sync

# onnxruntime-gpu (InsightFace face recognition) dlopens its CUDA execution
# provider, which needs CUDA 12 / cuDNN 9. Those libs ship inside torch's
# bundled nvidia-*-cu12 wheels in the venv but aren't on the default loader
# path, so onnxruntime would silently fall back to CPU (~550ms/req). Put them on
# LD_LIBRARY_PATH so the CUDA provider loads (~36ms/req).
NV_LIBS="$(find "$ROOT"/.venv/lib/python*/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: || true)"
if [ -n "$NV_LIBS" ]; then
    export LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Run without internet by default. Every default provider's weights are already
# on disk (HF cache, ~/.insightface/buffalo_l, repo-root *.pt / *.ts, graspnet
# checkpoint), so the only offline risk is libraries phoning home to revalidate
# the cache. These switches keep them local:
#   HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE -> clip, florence2, osnet, faster-whisper
#   YOLO_OFFLINE                          -> ultralytics skips its DNS probe + telemetry
# (insightface needs none — it checks its local model dir first.) The cloud
# providers (google STT, elevenlabs TTS, google_caption) can't run offline and
# aren't defaults. Set WALKIE_OFFLINE=0 to allow downloads (e.g. add a model).
if [ "${WALKIE_OFFLINE:-1}" != "0" ]; then
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_DISABLE_TELEMETRY=1
    export YOLO_OFFLINE=true
fi

exec uv run python app.py "$@"
