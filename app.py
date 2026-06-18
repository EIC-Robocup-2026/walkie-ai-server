"""Flask application entrypoint."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

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
