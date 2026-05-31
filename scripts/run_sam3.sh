#!/bin/bash
# scripts/run_sam3.sh — run the Flask app with SAM3 as the object-detection
# provider (open-vocabulary concept segmentation + masks).
#
# SAM3 weights (sam3.pt) are gated and do NOT auto-download. Download them
# manually and point SAM3_MODEL at the file (defaults to <repo>/sam3.pt).
#
# Usage:
#   SAM3_MODEL=/path/to/sam3.pt ./scripts/run_sam3.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OBJECT_DETECTION_PROVIDER=sam3
export SAM3_MODEL="${SAM3_MODEL:-$ROOT/sam3.pt}"

if [[ ! -f "$SAM3_MODEL" ]]; then
  echo "⚠️  SAM3 weights not found at: $SAM3_MODEL" >&2
  echo "    Download sam3.pt (gated) and set SAM3_MODEL to its path." >&2
fi

exec uv run python app.py "$@"
