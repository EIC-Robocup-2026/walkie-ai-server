#!/bin/bash
# scripts/install_grasp.sh — install the GraspNet stack for the /grasp endpoint.
#
# The /grasp endpoint needs three things on top of the core server deps. None of
# them are in pyproject.toml — grasp is an opt-in capability, its pinned deps are
# broken on py3.12, and its CUDA ops can't be built here (see below) — so they all
# live in this script instead of the core dependency set:
#   1. open3d            — point-cloud filtering (plain wheel).
#   2. graspnetAPI       — the GraspGroup container + nms (pure python; its pinned
#                          deps are broken on py3.12 so we install it --no-deps and
#                          add the handful it actually imports at runtime).
#   3. graspnet-baseline pointnet2 / knn / grasp_nms CUDA+cython ops — compiled
#                          extensions. They CANNOT be built here (no nvcc on PATH;
#                          torch is cu12.8 and the only toolkit is CUDA 13). So we
#                          reuse the already-built .so files from the perception
#                          venv, which load cleanly under this venv's torch (the
#                          extension ABI is compatible across torch 2.10 <-> 2.11).
#
# IMPORTANT: `uv sync` (run by run_app.sh) is exact by default and will REMOVE the
# packages this script installs, because they aren't (and mostly can't be) declared
# in pyproject.toml. Re-run this script after any `uv sync`, or run the app without
# re-syncing. (open3d is declared, so only the graspnet bits get dropped.)
#
# To rebuild the CUDA ops properly instead of reusing prebuilt .so (e.g. on a host
# with a matching CUDA 12.x toolkit / nvcc):
#   CUDA_HOME=/usr/local/cuda-12.x uv pip install --no-build-isolation \
#       -e ~/graspnet-baseline/pointnet2 -e ~/graspnet-baseline/knn
#   uv pip install -e ~/graspnetAPI/grasp_nms   # or wherever grasp_nms's setup lives

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRV="$ROOT/.venv/bin/python"
SP="$(echo "$ROOT"/.venv/lib/python*/site-packages)"

# Where the prebuilt extensions live (the perception venv that built them).
PERC_SP="${WALKIE_PERCEPTION_SP:-$HOME/test_ros_ws/src/perception/.venv/lib/python3.12/site-packages}"

echo "== server venv: $SRV =="

echo "== 1/3 open3d (declared dep; ensure present) =="
uv pip install --python "$SRV" "open3d>=0.18"

echo "== 2/3 graspnetAPI (--no-deps) + its real runtime imports =="
uv pip install --python "$SRV" --no-deps -e "$HOME/graspnetAPI"
uv pip install --python "$SRV" \
    "transforms3d>=0.4" trimesh h5py pywavefront cvxopt autolab_core

echo "== 3/3 reuse prebuilt CUDA/cython ops from the perception venv =="
for pkg in pointnet2 knn_pytorch; do
    if [ -d "$PERC_SP/$pkg" ]; then
        rm -rf "${SP:?}/$pkg"
        cp -r "$PERC_SP/$pkg" "$SP/$pkg"
        echo "  copied $pkg"
    else
        echo "  WARNING: $PERC_SP/$pkg not found — build it or set WALKIE_PERCEPTION_SP"
    fi
done
GRASP_NMS="$(find "$PERC_SP" -maxdepth 1 -name 'grasp_nms*.so' | head -1)"
if [ -n "$GRASP_NMS" ]; then
    cp "$GRASP_NMS" "$SP/"
    echo "  copied $(basename "$GRASP_NMS")"
else
    echo "  WARNING: grasp_nms .so not found in $PERC_SP"
fi

echo "== verify import (with torch's bundled CUDA libs on the loader path) =="
NV_LIBS="$(find "$ROOT"/.venv/lib/python*/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | paste -sd: || true)"
LD_LIBRARY_PATH="${NV_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" "$SRV" - <<'PY'
import os, sys
root = os.path.expanduser("~/graspnet-baseline")
for sub in ("", "models", "pointnet2", "utils", "knn", "dataset"):
    p = os.path.join(root, sub) if sub else root
    if os.path.isdir(p):
        sys.path.insert(0, p)
import open3d  # noqa
from models.graspnet import GraspNet, pred_decode  # noqa
from graspnetAPI.grasp import GraspGroup  # noqa
from grasp_nms import nms_grasp  # noqa
print("grasp stack import OK")
PY
echo "== done =="
