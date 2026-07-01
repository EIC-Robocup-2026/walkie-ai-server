#!/usr/bin/env python
"""Pre-fetch everything the "detecty" object-detection provider needs, so the
server can then boot fully offline (mirrors the README "run online once" step,
as one command).

Fetches (reading IDs from the vendored detecty config so they never drift):
  1. Grounding DINO   — HF `model.id` from detecty/data/config.yaml (transformers)
  2. DINOv3           — timm `DEFAULT_MODEL` from detecty.embedder
  3. EasyOCR          — languages from detecty/data/ensemble.yaml (~/.EasyOCR)
  4. Prototype bank   — builds weights/detecty_prototypes.npz from the bundled
                        reference images (skipped if it already exists)

Run online once on a fresh box:

    uv run python scripts/prefetch_detecty.py

Then `./scripts/run_app.sh` serves detecty with the network cut. Safe to re-run:
every step no-ops when its artifact is already cached.
"""

from __future__ import annotations

# --- Force online BEFORE importing transformers/timm/huggingface_hub. Those
# libraries read the offline switches once at import; if the caller's shell (or a
# sourced run_app.sh) left them set, the fetch would be silently blocked. This is
# the download tool, so clear them unconditionally.
import os

for _k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "YOLO_OFFLINE"):
    os.environ.pop(_k, None)

import argparse
import sys
from pathlib import Path

import yaml

# Make the repo importable when run as a plain script (scripts/ is on sys.path,
# not the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Reuse the provider's repo-relative defaults + path resolver (single source of
# truth for where the prototype bank and reference images live).
from services.object_detection.providers.detecty import (  # noqa: E402
    DEFAULT_CATALOG_DIR,
    DEFAULT_PROTOS,
    DEFAULT_PROTOTYPES_DIR,
    _resolve,
)
from detecty._resources import default_config, default_ensemble  # noqa: E402
from detecty.embedder import DEFAULT_MODEL as DINO_MODEL  # noqa: E402


def _auto_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def fetch_grounding_dino() -> None:
    """Cache the Grounding DINO localizer (processor + weights) via transformers."""
    gd_id = yaml.safe_load(open(default_config()))["model"]["id"]
    print(f"↓ Grounding DINO: {gd_id}")
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    AutoProcessor.from_pretrained(gd_id)
    AutoModelForZeroShotObjectDetection.from_pretrained(gd_id)
    print(f"  ✓ cached {gd_id}")


def fetch_dinov3() -> None:
    """Cache the DINOv3 backbone weights via timm (same call as Embedder)."""
    print(f"↓ DINOv3 backbone: {DINO_MODEL}")
    import timm

    timm.create_model(DINO_MODEL, pretrained=True, num_classes=0)
    print(f"  ✓ cached {DINO_MODEL}")


def fetch_easyocr() -> None:
    """Cache the EasyOCR detector + recognizers for detecty's languages."""
    langs = yaml.safe_load(open(default_ensemble()))["ocr"]["langs"]
    print(f"↓ EasyOCR models: langs={langs}")
    try:
        import easyocr

        easyocr.Reader(langs, gpu=False, verbose=False)
        print(f"  ✓ cached EasyOCR ({langs}) under ~/.EasyOCR")
    except Exception as e:  # noqa: BLE001 — never abort the rest of the prefetch
        print(f"  ⚠ EasyOCR prefetch skipped ({e}); OCR will fetch on first use")


def build_prototype_bank(protos: str, proto_dir: str, cat_dir: str, device: str) -> None:
    """Build weights/detecty_prototypes.npz from the bundled reference images."""
    if os.path.isfile(protos):
        print(f"• prototype bank already present: {protos} (skip)")
        return
    print(f"↻ building prototype bank -> {protos}")
    os.makedirs(os.path.dirname(protos) or ".", exist_ok=True)
    from detecty.build_prototypes import main as build_protos

    build_protos(
        [
            "--prototypes-dir", proto_dir,
            "--catalog-dir", cat_dir,
            "--out", protos,
            "--device", device,
            "--model", DINO_MODEL,
        ]
    )
    print(f"  ✓ built {protos}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Pre-fetch detecty's models + prototype bank for offline use."
    )
    ap.add_argument(
        "--device", default=None,
        help="cpu/cuda for the prototype-bank build (default: auto).",
    )
    ap.add_argument("--no-ocr", action="store_true", help="Skip EasyOCR models.")
    ap.add_argument(
        "--no-build", action="store_true", help="Skip building the prototype bank."
    )
    ap.add_argument("--protos", default=DEFAULT_PROTOS)
    ap.add_argument("--prototypes-dir", default=DEFAULT_PROTOTYPES_DIR)
    ap.add_argument("--catalog-dir", default=DEFAULT_CATALOG_DIR)
    args = ap.parse_args(argv)

    device = args.device or _auto_device()
    protos = _resolve(args.protos)
    proto_dir = _resolve(args.prototypes_dir)
    cat_dir = _resolve(args.catalog_dir)

    print("Prefetching detecty models (network required) ...\n")
    fetch_grounding_dino()
    fetch_dinov3()
    if not args.no_ocr:
        fetch_easyocr()
    if not args.no_build:
        build_prototype_bank(protos, proto_dir, cat_dir, device)

    hf_cache = os.environ.get("HF_HOME") or "~/.cache/huggingface"
    print("\nDone. Cached artifacts:")
    print(f"  • Grounding DINO + DINOv3  -> {hf_cache}")
    if not args.no_ocr:
        print("  • EasyOCR                  -> ~/.EasyOCR")
    if not args.no_build:
        print(f"  • prototype bank           -> {protos}")
    print("\nThe server can now run offline: ./scripts/run_app.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
