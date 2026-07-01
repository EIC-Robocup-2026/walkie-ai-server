"""Tests for the grasp endpoint — POST /grasp.

Live integration tests: they POST to a running walkie-ai-server (see ``--base-url``).
The cloud is uploaded as raw ``.npy`` bytes (an ``(N, 3)`` float32 array); the JSON
``spec`` form field selects options. The happy-path tests load GraspNet on the
server's first ``/grasp`` call, so they are slow (and need the model + its CUDA ops).
"""

import io
import json

import numpy as np
import requests

from services.grasp.providers.graspnet_provider import GraspNetProvider


def _grasp(base_url, cloud_npy, spec, return_status=False):
    resp = requests.post(
        f"{base_url}/grasp",
        files={"cloud": ("cloud.npy", cloud_npy, "application/octet-stream")},
        data={"spec": json.dumps(spec)},
    )
    if return_status:
        return resp
    body = resp.json()
    assert resp.status_code == 200, body
    assert body["success"] is True, body
    return body["data"]


def _npy(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.asarray(arr, dtype=np.float32), allow_pickle=False)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Approach-bias scoring (pure NumPy — no model, no server)
# ---------------------------------------------------------------------------

def test_approach_scores_side_vs_top():
    up = np.array([0.0, 0.0, 1.0])              # world up in the cloud frame
    down = np.array([[0.0, 0.0, -1.0]])         # top-down grasp: approach points down
    horiz = np.array([[1.0, 0.0, 0.0]])         # side grasp: horizontal approach
    upward = np.array([[0.0, 0.0, 1.0]])        # approach pointing up

    # "top" rewards down-pointing approaches, penalizes horizontal, and never
    # prefers an upward approach (clipped to 0).
    assert np.isclose(GraspNetProvider._approach_scores(down, up, "top")[0], 1.0)
    assert np.isclose(GraspNetProvider._approach_scores(horiz, up, "top")[0], 0.0)
    assert np.isclose(GraspNetProvider._approach_scores(upward, up, "top")[0], 0.0)

    # "side" rewards horizontal approaches, penalizes vertical (either sign).
    assert np.isclose(GraspNetProvider._approach_scores(horiz, up, "side")[0], 1.0)
    assert np.isclose(GraspNetProvider._approach_scores(down, up, "side")[0], 0.0)
    assert np.isclose(GraspNetProvider._approach_scores(upward, up, "side")[0], 0.0)

    # Unnormalised input is handled, and "none" → zeros over a batch.
    batch = np.vstack([down * 5.0, horiz, upward])
    assert np.isclose(GraspNetProvider._approach_scores(batch, up, "top")[0], 1.0)
    assert np.allclose(GraspNetProvider._approach_scores(batch, up, "none"), 0.0)


def test_approach_up_mask_drops_bottom_up():
    up = np.array([0.0, 0.0, 1.0])              # world up in the cloud frame
    down = np.array([0.0, 0.0, -1.0])           # top-down: approach points down
    horiz = np.array([1.0, 0.0, 0.0])           # side: horizontal approach
    upward = np.array([0.0, 0.0, 1.0])          # bottom-up: approach points up
    tilt_up = np.array([1.0, 0.0, 0.1])         # slight upward tilt (dot ≈ 0.1)
    batch = np.vstack([down * 5.0, horiz, upward, tilt_up])  # unnormalised ok

    # Default-ish tolerance: keep down/horizontal, drop straight-up; the slight
    # tilt (~5.7° above horizontal) survives a 0.2 tolerance but not 0.0.
    keep = GraspNetProvider._approach_up_mask(batch, up, 0.2)
    assert keep.tolist() == [True, True, False, True]
    assert GraspNetProvider._approach_up_mask(batch, up, 0.0).tolist() == [
        True, True, False, False
    ]
    # max_up_cos = 1.0 disables the filter (nothing dropped).
    assert GraspNetProvider._approach_up_mask(batch, up, 1.0).all()


def test_center_scores_favour_centroid():
    # A box-ish cloud; _cloud_center returns the centroid + its max point radius.
    obj = np.array(
        [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    centroid, scale = GraspNetProvider._cloud_center(obj)
    assert np.allclose(centroid, [0.0, 0.0, 0.0])
    assert np.isclose(scale, 1.0)

    at_center = np.array([[0.0, 0.0, 0.0]])
    half = np.array([[0.5, 0.0, 0.0]])          # half a radius out
    edge = np.array([[1.0, 0.0, 0.0]])          # a full radius out
    beyond = np.array([[3.0, 0.0, 0.0]])        # past the radius → clamps at 0

    assert np.isclose(GraspNetProvider._center_scores(at_center, centroid, scale)[0], 1.0)
    assert np.isclose(GraspNetProvider._center_scores(half, centroid, scale)[0], 0.5)
    assert np.isclose(GraspNetProvider._center_scores(edge, centroid, scale)[0], 0.0)
    assert np.isclose(GraspNetProvider._center_scores(beyond, centroid, scale)[0], 0.0)

    # Batch + a non-origin centroid: nearer the centroid scores higher.
    c = np.array([1.0, 1.0, 1.0])
    out = GraspNetProvider._center_scores(
        np.array([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]]), c, 2.0
    )
    assert np.isclose(out[0], 1.0) and np.isclose(out[1], 0.0)


def test_emit_rotation_upright_x_flip():
    # roll/pitch/yaw offsets default to 0 → R_offset = I, so the emitted X axis
    # is the raw approach (col-0) and Z (col-2) is the raw col-2. No model load.
    prov = GraspNetProvider({})
    up = np.array([0.0, 0.0, 1.0])              # world up in the cloud frame
    # A "wrong-way-up" side grasp: X (col-0) points down, Z (col-2) horizontal.
    R = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]])
    assert R[:, 0] @ up < 0                      # sanity: X starts pointing down

    # x_target = +up + X below horizontal → roll 180 about Z: X flips up, Z held.
    flipped = np.array(prov._emit_rotation(R, up))
    assert flipped[:, 0] @ up > 0
    assert np.allclose(flipped[:, 2], R[:, 2])

    # x_target=None passes the rotation through untouched (the offset is identity).
    assert np.allclose(np.array(prov._emit_rotation(R, None)), R)

    # X already toward the target → no flip even when x_target is supplied.
    assert np.allclose(np.array(prov._emit_rotation(flipped, up)), flipped)

    # Inverse target (-up): X is forced into the lower hemisphere instead. R
    # already has X down, so it passes through; the upright "flipped" rotation
    # (X up) gets rolled back down, with Z held either way.
    assert np.allclose(np.array(prov._emit_rotation(R, -up)), R)
    inv = np.array(prov._emit_rotation(flipped, -up))
    assert inv[:, 0] @ up < 0
    assert np.allclose(inv[:, 2], flipped[:, 2])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_grasp_missing_cloud(base_url):
    resp = requests.post(f"{base_url}/grasp", data={"spec": "{}"})
    assert resp.status_code == 400
    body = resp.json()
    assert body["success"] is False
    assert "cloud" in body["error"].lower()


def test_grasp_invalid_npy(base_url):
    resp = requests.post(
        f"{base_url}/grasp",
        files={"cloud": ("cloud.npy", b"not-an-npy", "application/octet-stream")},
        data={"spec": "{}"},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_grasp_wrong_shape(base_url):
    resp = requests.post(
        f"{base_url}/grasp",
        files={"cloud": ("cloud.npy", _npy(np.zeros((10, 4))), "application/octet-stream")},
        data={"spec": "{}"},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


def test_grasp_invalid_spec_json(base_url, sample_cloud_npy):
    resp = requests.post(
        f"{base_url}/grasp",
        files={"cloud": ("cloud.npy", sample_cloud_npy, "application/octet-stream")},
        data={"spec": "{not json"},
    )
    assert resp.status_code == 400
    assert resp.json()["success"] is False


# ---------------------------------------------------------------------------
# Happy path (loads GraspNet — slow, needs GPU + CUDA ops)
# ---------------------------------------------------------------------------

def test_grasp_shape(base_url, sample_cloud_npy):
    data = _grasp(base_url, sample_cloud_npy, {"max_grasps": 10})
    assert isinstance(data["grasps"], list)
    assert data["count"] == len(data["grasps"])
    for g in data["grasps"]:
        assert len(g["translation"]) == 3
        assert len(g["rotation"]) == 3 and all(len(r) == 3 for r in g["rotation"])
        assert isinstance(g["width"], (int, float))
        assert isinstance(g["score"], (int, float))
        assert g["antipodal_score"] is None  # not requested


def test_grasp_respects_max_grasps(base_url, sample_cloud_npy):
    data = _grasp(base_url, sample_cloud_npy, {"max_grasps": 3})
    assert len(data["grasps"]) <= 3


def test_grasp_antipodal_attaches_score(base_url, sample_cloud_npy):
    data = _grasp(base_url, sample_cloud_npy, {"max_grasps": 5, "antipodal": True})
    for g in data["grasps"]:
        assert isinstance(g["antipodal_score"], (int, float))
        assert 0.0 <= g["antipodal_score"] <= 1.0


def test_grasp_approach_preference_biases_direction(base_url, sample_can_npy):
    """'top' favours downward approaches; 'side' favours horizontal ones.

    Recovers each grasp's *raw* approach (rotation col-0) by undoing the server's
    output rotation offset, exactly as the debug viewer does, then compares it to
    gravity. Averaged over a few requests since GraspNet re-samples per call.
    """
    from api.routes.config import compact, section

    cfg = section("grasp")
    provider = cfg.get("provider", "graspnet")
    r_off = GraspNetProvider(compact(cfg.get(provider, {}))).rotation_offset
    up = np.array([0.0, -1.0, 0.0])      # optical frame: up = -Y, gravity = +Y
    gravity = -up

    def aligns(preference):
        # Isolate the *approach* preference from the confounders that otherwise
        # muddy the raw col-0 direction on this thin upright can:
        #   - max_approach_up=1.0 disables the bottom-up hard filter. At the
        #     production default (0.2) it strips the upward tail of "side"'s
        #     (near-horizontal) approaches, biasing them ~20° downward and
        #     collapsing the side/top separation this test measures.
        #   - center_weight/closing_weight=0 (+ max_closing_up=1.0) drop the
        #     "side"-only centre/closing biases, which re-rank on translation and
        #     col-1 without regard to approach horizontality.
        # A larger approach_weight makes the preference dominate GraspNet's own
        # score spread. Production keeps all these on by design; here we want a
        # clean read of "does approach_preference bias the approach direction".
        data = _grasp(base_url, sample_can_npy,
                      {"max_grasps": 12, "approach_preference": preference,
                       "up": up.tolist(), "approach_weight": 4.0,
                       "max_approach_up": 1.0, "center_weight": 0.0,
                       "closing_weight": 0.0, "max_closing_up": 1.0})
        out = []
        for g in data["grasps"]:
            r = np.asarray(g["rotation"], float)
            a = (r @ r_off.T)[:, 0]          # undo offset → raw approach
            a /= np.linalg.norm(a) + 1e-9
            out.append(float(a @ gravity))
        assert out, "expected grasps"
        return np.array(out)

    reps = 4
    top_down = np.mean([aligns("top").mean() for _ in range(reps)])
    side_down = np.mean([aligns("side").mean() for _ in range(reps)])
    side_horiz = np.mean([np.abs(aligns("side")).mean() for _ in range(reps)])
    top_horiz = np.mean([np.abs(aligns("top")).mean() for _ in range(reps)])

    # 'top' approaches point more along gravity (downward) than 'side' approaches.
    assert top_down > side_down + 0.1, (top_down, side_down)
    # 'side' approaches are markedly more horizontal (smaller |approach·gravity|).
    assert side_horiz < top_horiz - 0.1, (side_horiz, top_horiz)
