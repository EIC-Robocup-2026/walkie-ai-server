"""Grasp-pose generation blueprint — ``POST /grasp``.

A pure **cloud → grasp poses** endpoint. The request is multipart:

  - ``cloud``: a file of raw ``.npy`` bytes for an ``(N, 3)`` float32 array — the
    segmented object's points in **one** frame. The camera-optical frame is
    recommended (it keeps GraspNet in-distribution), but the server is
    frame-agnostic: grasps come back in whatever frame the cloud was sent in.
  - ``spec``: a JSON form field of options (all optional)::

        {"score_threshold": 0.0, "max_grasps": 20, "antipodal": true,
         "voxel_size": 0.005, "num_point": 10000,
         "outlier_removal": true, "cluster_filter": false,
         "approach_preference": "side", "up": [0, 0, 1],
         "approach_weight": 1.0, "max_approach_up": 0.2, "center_weight": 0.5,
         "upright_x": true}

    ``approach_preference`` (``"side"`` / ``"top"`` / ``"none"``) softly re-ranks
    grasps by how their approach aligns with the ``up`` vector — given **in the
    cloud frame** (gravity = ``-up``): ``side`` favours horizontal approaches,
    ``top`` favours approaches pointing down. ``approach_weight`` scales the
    bonus; both are ignored without a usable ``up``. With a ``side``/``top``
    preference, "bottom-up" grasps whose approach points upward past
    ``max_approach_up`` (max allowed approach·up; 0 = horizontal-or-below,
    1 = disabled) are dropped outright. For ``side`` only, ``center_weight``
    additionally favours grasps whose centre is near the cloud centroid (grab the
    middle of the object, not an edge), and ``upright_x`` (default true) rolls the
    gripper 180 about its approach when needed so its X axis points up.

The response is ``{"grasps": [...], "count": n}`` with each grasp shaped
``{"translation": [x,y,z], "rotation": [[..3x3..]], "width": float, "score": float,
"antipodal_score": float|null}``. All robot-frame work (planning-frame transform, the
end-effector alignment, pre-grasp approach poses) is the caller's job.
"""

from __future__ import annotations

import io
import json

import numpy as np
from flask import Blueprint, request

from api.models import get_grasp
from api.utils import error, success
from services import debug_viewer

bp = Blueprint("grasp", __name__)

# Per-request options forwarded straight to the provider's infer(); anything else
# in the spec is ignored so the wire contract can grow without breaking old clients.
_OPT_KEYS = (
    "score_threshold",
    "max_grasps",
    "antipodal",
    "voxel_size",
    "num_point",
    "outlier_removal",
    "cluster_filter",
    "approach_preference",
    "up",
    "approach_weight",
    "max_approach_up",
    "center_weight",
    "upright_x",
)


@bp.post("/grasp")
def grasp():
    if "cloud" not in request.files:
        return error("Missing 'cloud' file (a .npy of an (N, 3) float32 array)")
    try:
        cloud = np.load(io.BytesIO(request.files["cloud"].read()), allow_pickle=False)
    except Exception as exc:
        return error(f"Invalid cloud (.npy): {exc}")
    cloud = np.asarray(cloud)
    if cloud.ndim != 2 or cloud.shape[1] != 3:
        return error(f"cloud must be an (N, 3) array; got shape {tuple(cloud.shape)}")

    raw_spec = request.form.get("spec")
    try:
        spec: dict = json.loads(raw_spec) if raw_spec else {}
    except Exception as exc:
        return error(f"Invalid spec JSON: {exc}")
    if not isinstance(spec, dict):
        return error("spec must be a JSON object")
    opts = {k: spec[k] for k in _OPT_KEYS if k in spec}

    try:
        grasps = get_grasp().infer(cloud, **opts)
    except ValueError as exc:
        # Bad/insufficient input (e.g. too few points after filtering) — client error.
        return error(str(exc))
    except Exception as exc:
        return error(str(exc), 500)
    debug_viewer.show_grasp(cloud, grasps, rotation_offset=get_grasp().rotation_offset)
    return success({"grasps": grasps, "count": len(grasps)})
