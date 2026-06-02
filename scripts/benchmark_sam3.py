#!/usr/bin/env python3
"""Benchmark SAM3 detection latency across inference image sizes (``imgsz``).

For each ``imgsz`` it builds a fresh ``SAM3ObjectDetectionProvider``, warms it
up, then times ``detect()`` over N runs and reports latency stats + throughput.
Use it to pick the speed/recall trade-off before deploying.

Examples:
    # default sizes (1024/768/640/512) on a sample image
    uv run python scripts/benchmark_sam3.py --image path/to/photo.jpg

    # custom sizes, more runs, with torch.compile
    uv run python scripts/benchmark_sam3.py --imgsz 768 512 384 --runs 30 --compile default

    # quick smoke test with a synthetic image (no file needed)
    uv run python scripts/benchmark_sam3.py --runs 10

SAM3 weights (sam3.pt) are gated and do NOT auto-download. Point --model (or the
SAM3_MODEL env var) at the local checkpoint; defaults to <repo>/sam3.pt.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# Make the repo importable when run as a plain script.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from services.object_detection.providers.sam3 import (  # noqa: E402
    SAM3ObjectDetectionProvider,
    _DEFAULT_PROMPTS,
)


def _cuda_sync(device: str) -> None:
    """Block until pending CUDA work finishes so timings are accurate."""
    if device != "cpu":
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass


def _load_image(path: str | None, size: int) -> Image.Image:
    """Load the benchmark image, or synthesize noise if no path is given."""
    if path:
        return Image.open(path).convert("RGB")
    rng = np.random.default_rng(0)  # fixed seed → comparable across sizes
    arr = rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _fmt(ms: float) -> str:
    return f"{ms:7.1f}"


def benchmark_size(
    imgsz: int | None,
    image: Image.Image,
    prompts: list[str],
    *,
    model: str,
    device: str | None,
    compile_mode: object,
    runs: int,
    warmup_runs: int,
) -> dict[str, float]:
    """Build a provider at ``imgsz`` and time ``detect()`` over ``runs`` calls."""
    config: dict[str, object] = {
        "model": model,
        "imgsz": imgsz,
        "compile": compile_mode,
        "warmup": False,  # we do explicit, timed warm-up below
        "prompts": prompts,
    }
    if device:
        config["device"] = device

    provider = SAM3ObjectDetectionProvider(config)
    provider.load_model()
    dev = provider._device  # resolved device string for sync

    # Warm-up (build/compile/autotune) — not counted in the reported stats.
    t_warm = time.perf_counter()
    for _ in range(max(1, warmup_runs)):
        provider.detect(image, prompts)
    _cuda_sync(dev)
    warm_ms = (time.perf_counter() - t_warm) * 1000.0 / max(1, warmup_runs)

    # Timed runs.
    samples: list[float] = []
    n_dets = 0
    for _ in range(runs):
        _cuda_sync(dev)
        t0 = time.perf_counter()
        dets = provider.detect(image, prompts)
        _cuda_sync(dev)
        samples.append((time.perf_counter() - t0) * 1000.0)
        n_dets = len(dets)

    samples.sort()
    mean = statistics.fmean(samples)
    p90 = samples[min(len(samples) - 1, int(round(0.9 * (len(samples) - 1))))]
    return {
        "imgsz": float(imgsz) if imgsz else 1024.0,
        "warmup_ms": warm_ms,
        "mean_ms": mean,
        "median_ms": statistics.median(samples),
        "p90_ms": p90,
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "fps": 1000.0 / mean if mean > 0 else 0.0,
        "n_dets": float(n_dets),
    }


def main() -> int:
    # Stream progress live even when stdout is piped/redirected (Python
    # block-buffers a non-tty by default, which hides per-size progress).
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.environ.get("SAM3_MODEL", str(_REPO_ROOT / "sam3.pt")),
        help="Path to sam3.pt (default: $SAM3_MODEL or <repo>/sam3.pt).",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Image to benchmark. Omit to use a synthetic noise image.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[1024, 768, 640, 512],
        help="Inference sizes to sweep (use 1024 for the model native size).",
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="Timed runs per size (default: 20)."
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=3,
        help="Warm-up runs per size, not timed (default: 3).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='"cuda" or "cpu" (default: auto-detect).',
    )
    parser.add_argument(
        "--compile",
        dest="compile_mode",
        default=False,
        help='torch.compile mode: "default"/"reduce-overhead"/'
        '"max-autotune-no-cudagraphs", or omit to disable.',
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=None,
        help="Concepts to detect (default: the provider's RoboCup default set).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        print(f"⚠️  SAM3 weights not found at: {args.model}", file=sys.stderr)
        print("    Download sam3.pt (gated) and pass --model / set SAM3_MODEL.", file=sys.stderr)
        return 2

    prompts = args.prompts if args.prompts else list(_DEFAULT_PROMPTS)
    compile_mode = args.compile_mode
    if isinstance(compile_mode, str) and compile_mode.lower() in {"false", "off", "none", ""}:
        compile_mode = False

    print("SAM3 latency benchmark")
    print(f"  model   : {args.model}")
    print(f"  image   : {args.image or 'synthetic noise'}")
    print(f"  prompts : {len(prompts)} concept(s)")
    print(f"  runs    : {args.runs} (+{args.warmup_runs} warm-up) per size")
    print(f"  compile : {compile_mode}")
    print(f"  sizes   : {args.imgsz}")
    print()

    results = []
    for imgsz in args.imgsz:
        size = imgsz if imgsz else 1024
        image = _load_image(args.image, size)
        print(f"→ imgsz={imgsz} ...", flush=True)
        try:
            results.append(
                benchmark_size(
                    imgsz,
                    image,
                    prompts,
                    model=args.model,
                    device=args.device,
                    compile_mode=compile_mode,
                    runs=args.runs,
                    warmup_runs=args.warmup_runs,
                )
            )
        except Exception as e:  # keep going so one bad size doesn't kill the sweep
            print(f"   failed: {e}")

    if not results:
        print("No successful runs.")
        return 1

    # Results table, with speedup relative to the largest size benchmarked.
    base = max(results, key=lambda r: r["imgsz"])["mean_ms"]
    print()
    header = (
        f"{'imgsz':>6} | {'mean':>8} {'median':>8} {'p90':>8} {'min':>8} "
        f"{'max':>8} | {'fps':>6} | {'speedup':>7} | {'dets':>4} | {'warmup':>8}"
    )
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda r: -r["imgsz"]):
        speedup = base / r["mean_ms"] if r["mean_ms"] > 0 else 0.0
        print(
            f"{int(r['imgsz']):>6} | {_fmt(r['mean_ms'])} {_fmt(r['median_ms'])} "
            f"{_fmt(r['p90_ms'])} {_fmt(r['min_ms'])} {_fmt(r['max_ms'])} | "
            f"{r['fps']:6.1f} | {speedup:6.2f}x | {int(r['n_dets']):>4} | "
            f"{_fmt(r['warmup_ms'])}"
        )
    print("\n(all latencies in ms; speedup vs largest size; 'dets' = objects found)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
