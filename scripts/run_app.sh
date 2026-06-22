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

exec uv run python app.py "$@"
