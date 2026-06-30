"""Flask application entrypoint."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Offline-by-default (mirrors scripts/run_app.sh). Serve every model from the
# local HF cache / ~/.insightface pack / repo-root weights and never let a
# library phone home to revalidate or download. This MUST run before the
# create_app import below: that import eagerly loads whisper/clip/yoloe at
# import time, and transformers/huggingface_hub read these vars once at their
# own import. setdefault so an explicit shell value (incl. WALKIE_OFFLINE=0
# clearing them) always wins. Set WALKIE_OFFLINE=0 to allow downloads.
# ---------------------------------------------------------------------------
if os.getenv("WALKIE_OFFLINE", "1") != "0":
    for _k, _v in (
        ("HF_HUB_OFFLINE", "1"),
        ("TRANSFORMERS_OFFLINE", "1"),
        ("HF_HUB_DISABLE_TELEMETRY", "1"),
        ("YOLO_OFFLINE", "true"),
    ):
        os.environ.setdefault(_k, _v)

# ---------------------------------------------------------------------------
# Deferred import: create_app registers blueprints which instantiate
# and load all model singletons.
# ---------------------------------------------------------------------------
from api import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    # threaded=False is deliberate. Flask defaults threaded=True, which hands
    # each request to a brand-new thread; the first CUDA kernel launch in a
    # fresh thread pays a ~800ms per-thread init cost, so every request ate it
    # (~820ms even for a 64x64 image). Serving single-threaded drops that to
    # ~25ms. GPU inference serializes on one CUDA stream anyway, so a thread
    # pool buys throughput only with reused (warmed) threads — use waitress for
    # that, not Werkzeug's new-thread-per-request dev server.
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=False)
