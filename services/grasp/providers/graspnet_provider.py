"""GraspNet-1Billion grasp provider (wraps the ``graspnet-baseline`` repo).

Ported from the on-robot ROS ``grasp_node`` (``/grasp/from_cloud`` + the antipodal
validation from ``/grasp/pos``), with every ROS/TF concern removed. The provider is a
pure function of the input cloud:

    object cloud (N, 3) in some frame
      → voxel + statistical-outlier (+ optional largest-cluster DBSCAN) filtering
      → random-sample to ``num_point``
      → GraspNet forward + ``pred_decode`` → GraspGroup
      → sort-by-score → NMS → score-threshold → trim to a candidate pool
      → [optional] antipodal surface-normal validation (containment + refine)
      → grasp dicts (translation / rotation / width / score), in the input frame

The heavy bits — ``models.graspnet.GraspNet`` (with its compiled ``pointnet2`` / ``knn``
CUDA ops), ``graspnetAPI.GraspGroup``, and ``open3d`` — live outside this repo. The
model is loaded **lazily** on the first ``infer`` call (and on an explicit
``load_model()``); ``graspnet_root`` is put on ``sys.path`` the same way the ROS node
and ``demo.py`` do it.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np


def _unit(v: np.ndarray):
    """Return *v* normalised, or ``None`` if it is (near) zero length."""
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


class GraspNetProvider:
    """6-DOF grasp generation via GraspNet-1Billion on a supplied point cloud."""

    DEFAULT_CHECKPOINT = "~/graspnet-baseline/logs/log_rs/checkpoint-rs.tar"
    DEFAULT_ROOT = "~/graspnet-baseline"

    # 180 about the gripper's own Z, post-multiplied: flips the X (and Y) axes
    # while leaving Z (the approach for a side grasp) fixed. See _emit_rotation.
    _RZ_180 = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialise the GraspNet provider.

        Args:
            config: Optional keys (defaults match the ROS ``grasp_node`` params):
                - ``checkpoint_path`` / ``graspnet_root``: model + repo locations.
                - ``device``: ``"cuda"`` / ``"cpu"`` (``""`` → auto, CUDA when available).
                - inference: ``num_point`` (10000), ``num_view`` (300),
                  ``voxel_size`` (0.005), ``min_points`` (200), ``rerank_pool_size`` (200).
                - outlier removal: ``outlier_nb_neighbors`` (20), ``outlier_std_ratio`` (2.0).
                - clustering: ``cluster_eps`` (0.02), ``cluster_min_samples`` (10).
                - antipodal: ``select_margin_m`` (0.02), ``normal_radius_m`` (0.02),
                  ``normal_max_nn`` (30), ``antipodal_r_tol_m`` (0.01),
                  ``antipodal_min_pts`` (4), ``width_clearance_m`` (0.01),
                  ``max_gripper_width_m`` (0.10).
        """
        c = config

        def f(key: str, default: float) -> float:
            v = c.get(key)
            return float(v) if v not in (None, "") else default

        def i(key: str, default: int) -> int:
            v = c.get(key)
            return int(v) if v not in (None, "") else default

        def b(key: str, default: bool) -> bool:
            v = c.get(key)
            if v in (None, ""):
                return default
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "on")
            return bool(v)

        self._checkpoint_path: str = c.get("checkpoint_path") or self.DEFAULT_CHECKPOINT
        self._graspnet_root: str = c.get("graspnet_root") or self.DEFAULT_ROOT
        self._device_cfg: str | None = c.get("device") or None

        # Inference / sampling.
        self._num_point = i("num_point", 10000)
        self._num_view = i("num_view", 300)
        self._voxel_size = f("voxel_size", 0.005)
        self._min_points = i("min_points", 200)
        self._rerank_pool_size = i("rerank_pool_size", 200)

        # Outlier removal + clustering (filtering the supplied object cloud).
        self._outlier_nb_neighbors = i("outlier_nb_neighbors", 20)
        self._outlier_std_ratio = f("outlier_std_ratio", 2.0)
        self._cluster_eps = f("cluster_eps", 0.02)
        self._cluster_min_samples = i("cluster_min_samples", 10)

        # Antipodal validation (optional, frame-agnostic).
        self._select_margin_m = f("select_margin_m", 0.02)
        self._normal_radius_m = f("normal_radius_m", 0.02)
        self._normal_max_nn = i("normal_max_nn", 30)
        self._antipodal_r_tol_m = f("antipodal_r_tol_m", 0.01)
        self._antipodal_min_pts = i("antipodal_min_pts", 4)
        self._width_clearance_m = f("width_clearance_m", 0.01)
        self._max_gripper_width_m = f("max_gripper_width_m", 0.10)

        # Approach-bias re-rank (optional, per request). Blend weight for the
        # client-supplied "up" preference; the "up" vector + preference mode
        # themselves arrive per request. See _approach_scores / infer().
        self._approach_weight = f("approach_weight", 1.0)
        # Hard reject "bottom-up" grasps when a side/top preference is active:
        # the max allowed approach·up (col-0 dotted with the cloud-frame "up").
        # 0 keeps only at/below-horizontal approaches, a small positive value
        # tolerates a slight upward tilt, 1.0 disables the filter. See
        # _approach_up_mask / infer().
        self._max_approach_up = f("max_approach_up", 0.2)

        # Centre-bias re-rank, "side" preference only. Among the (already
        # horizontal) side approaches, also favour grasps whose centre sits near
        # the object cloud's centroid — i.e. grab the middle of the object rather
        # than an edge. Blend weight for the added [0,1] centrality bonus; 0
        # disables it. See _center_scores / infer().
        self._center_weight = f("center_weight", 0.5)

        # Upright-X flip, "side" preference only. After the output rotation offset
        # is applied, roll the gripper 180 about its own Z (the approach for a
        # side grasp) whenever its X axis points into the lower ("-up")
        # hemisphere, so X always ends up pointing up. Keeps the wrist from
        # hanging upside-down on otherwise-equivalent side grasps. A per-request
        # upright_x wins. See _emit_rotation / infer().
        self._upright_x = b("upright_x", True)
        # Invert the upright-X target: point X into the lower ("-up") hemisphere
        # (X down) instead of up. Only has an effect when upright_x is on (side
        # grasps). A per-request upright_x_inverse wins. See _emit_rotation / infer().
        self._upright_x_inverse = b("upright_x_inverse", False)

        # End-effector convention. GraspNet emits rotations with the approach
        # along +X (col-0). roll/pitch/yaw_offset_deg post-rotate each *output*
        # rotation about the gripper's own X/Y/Z axes, re-aligning it to a robot
        # whose tool frame differs. Pitch (local Y) re-points "forward": +90 maps
        # the approach onto local +Z (toward the object), -90 onto -Z. The
        # internal antipodal math always uses the raw (unrotated) rotation.
        self.roll_offset_deg = f("roll_offset_deg", 0.0)
        self.pitch_offset_deg = f("pitch_offset_deg", 0.0)
        self.yaw_offset_deg = f("yaw_offset_deg", 0.0)
        self._R_offset = self._rpy_matrix(
            self.roll_offset_deg, self.pitch_offset_deg, self.yaw_offset_deg
        )

        self._model_name = f"graspnet-1billion ({os.path.basename(self._checkpoint_path)})"

        # Lazily populated in _ensure_loaded().
        self._net: Any | None = None
        self._device: str = "cpu"
        self._torch: Any | None = None
        self._pred_decode: Any | None = None
        self._GraspGroup: Any | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Pre-load the GraspNet network + checkpoint into memory."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._net is not None:
            return
        import torch

        root = os.path.expanduser(self._graspnet_root)
        if not os.path.isdir(root):
            raise RuntimeError(
                f"graspnet_root '{root}' not found. Set [grasp.graspnet].graspnet_root "
                "to the graspnet-baseline checkout."
            )
        # graspnet-baseline uses flat, demo-style absolute imports: graspnet.py does
        # `from backbone import ...` (a sibling in models/) and `import pointnet2_utils`
        # (in pointnet2/). It only puts pointnet2/ + utils/ on the path itself, so we
        # add the rest — matching the perception venv's graspnet_baseline.pth.
        for sub in ("", "models", "pointnet2", "utils", "knn", "dataset"):
            p = os.path.join(root, sub) if sub else root
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        try:
            from models.graspnet import GraspNet, pred_decode
            from graspnetAPI.grasp import GraspGroup
        except Exception as exc:  # pragma: no cover - import/env failure
            raise RuntimeError(
                "Could not import GraspNet / graspnetAPI. Ensure 'open3d' and "
                "'graspnetAPI' are installed in this venv and the pointnet2/knn CUDA "
                f"ops under '{root}' are built against this torch. Error: {exc}"
            ) from exc

        device = self._device_cfg or ("cuda" if torch.cuda.is_available() else "cpu")
        net = GraspNet(
            input_feature_dim=0,
            num_view=self._num_view,
            num_angle=12,
            num_depth=4,
            cylinder_radius=0.05,
            hmin=-0.02,
            hmax_list=[0.01, 0.02, 0.03, 0.04],
            is_training=False,
        )
        ckpt_path = os.path.expanduser(self._checkpoint_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ckpt["model_state_dict"])
        net = net.to(device).eval()

        self._torch = torch
        self._pred_decode = pred_decode
        self._GraspGroup = GraspGroup
        self._net = net
        self._device = device
        self._warmup()

    def _warmup(self) -> None:
        torch = self._torch
        dummy = torch.randn(1, self._num_point, 3).to(self._device)
        with torch.no_grad():
            self._pred_decode(self._net({"point_clouds": dummy}))
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def infer(
        self,
        cloud: np.ndarray,
        *,
        score_threshold: float = 0.0,
        max_grasps: int = 20,
        antipodal: bool = False,
        voxel_size: float | None = None,
        num_point: int | None = None,
        outlier_removal: bool = True,
        cluster_filter: bool = False,
        approach_preference: str = "none",
        up: list | None = None,
        approach_weight: float | None = None,
        max_approach_up: float | None = None,
        center_weight: float | None = None,
        upright_x: bool | None = None,
        upright_x_inverse: bool | None = None,
        **_ignored: Any,
    ) -> list[dict]:
        """Generate grasps for an ``(N, 3)`` cloud, in the cloud's own frame.

        ``approach_preference`` (``"side"`` / ``"top"`` / ``"none"``) softly
        re-ranks candidates by how their approach (raw rotation col-0) aligns
        with a client-supplied ``up`` vector **in the cloud frame** (gravity =
        ``-up``): ``top`` favours approaches pointing down, ``side`` favours
        horizontal approaches. ``approach_weight`` (default config value) scales
        the added bonus. A ``side``/``top`` preference with no usable ``up`` is
        ignored.

        With a ``side``/``top`` preference active, "bottom-up" grasps — whose
        raw approach points upward (against gravity) by more than
        ``max_approach_up`` (the max allowed approach·up; default config value)
        — are **hard-dropped** before re-ranking, since the soft bias alone can
        still surface them when the candidate pool is thin.

        For ``side`` only, the re-rank also folds in a **centrality** bonus
        (``center_weight``, default config value) that favours grasps whose
        centre sits near the object cloud's centroid — preferring a grip on the
        middle of the object over one near an edge. It is scored on the refined
        centre on the antipodal path.

        Also for ``side`` only, ``upright_x`` (default config value, default
        True) keeps the gripper's X axis in the **up** hemisphere: after the
        output rotation offset is applied, any grasp whose X axis points below
        horizontal (``X·up < 0``) is rolled 180 about its own Z (the approach),
        so X always points up. This only re-rolls the wrist — the approach and
        contact geometry are unchanged. ``upright_x_inverse`` (default config
        value, default False) flips the target so X is forced **down** (into the
        ``-up`` hemisphere) instead; it only matters when ``upright_x`` is on.
        """
        try:
            self._ensure_loaded()
            pts_in = np.asarray(cloud, dtype=np.float32)
            if pts_in.ndim != 2 or pts_in.shape[1] != 3:
                raise ValueError(f"cloud must be (N, 3); got shape {pts_in.shape}")
            pts_in = pts_in[np.isfinite(pts_in).all(axis=1)]
            if len(pts_in) == 0:
                raise ValueError("cloud has no finite points")

            voxel = self._voxel_size if voxel_size is None else float(voxel_size)
            npoint = self._num_point if num_point is None else int(num_point)
            n_out = int(max_grasps) if max_grasps and max_grasps > 0 else 20

            # Approach-bias preference (optional). Needs a usable "up" vector.
            preference = str(approach_preference or "none").lower()
            up_unit = _unit(np.asarray(up, dtype=np.float64)) if up is not None else None
            if preference in ("side", "top") and up_unit is None:
                print(
                    f"[grasp] approach_preference='{preference}' ignored: "
                    "no/zero 'up' vector"
                )
                preference = "none"
            pref_active = preference in ("side", "top")
            weight = self._approach_weight if approach_weight is None else float(approach_weight)
            max_up = (
                self._max_approach_up if max_approach_up is None else float(max_approach_up)
            )
            # Centre bias is "side"-only; per-request override wins over config.
            cen_w = self._center_weight if center_weight is None else float(center_weight)
            cen_w = cen_w if preference == "side" else 0.0
            # Upright-X flip is "side"-only; per-request overrides win over config.
            # x_target is the direction the gripper's X axis should point toward
            # ("up" normally, "-up"/down when inverted), passed to _emit_rotation;
            # None disables the flip.
            up_x = self._upright_x if upright_x is None else bool(upright_x)
            up_x_inv = (
                self._upright_x_inverse if upright_x_inverse is None else bool(upright_x_inverse)
            )
            x_target = None
            if up_x and preference == "side":
                x_target = -up_unit if up_x_inv else up_unit

            # 1. Filter the object cloud (voxel + statistical-outlier + optional cluster).
            obj = self._filter_world(pts_in, voxel, bool(outlier_removal), bool(cluster_filter))
            if len(obj) < self._min_points:
                raise ValueError(
                    f"insufficient points after filtering: {len(obj)} "
                    f"(need {self._min_points})"
                )

            # 2. GraspNet on a fixed-size sample of the object cloud.
            pts = self._sample_points(obj, npoint)
            t0 = time.perf_counter()
            raw = self._run_graspnet(pts)
            infer_ms = (time.perf_counter() - t0) * 1e3

            # 3. Sort + NMS + score-threshold + trim to a candidate pool. A re-rank
            #    (antipodal and/or approach-bias) needs a pool larger than n_out.
            pool_size = self._rerank_pool_size if (antipodal or pref_active) else n_out
            gg = self._select_pool(raw, float(score_threshold), max(1, pool_size))
            if len(gg) == 0:
                return []

            # 3b. With a side/top preference, hard-drop "bottom-up" grasps whose
            #     raw approach (col-0) points upward past max_up — the soft
            #     re-rank alone can still surface these when the pool is thin.
            if pref_active:
                approaches = np.asarray(gg.rotation_matrices)[:, :, 0]
                keep = self._approach_up_mask(approaches, up_unit, max_up)
                gg = gg[keep]
                if len(gg) == 0:
                    print(
                        f"[grasp] no grasp left after dropping bottom-up approaches "
                        f"(preference={preference}, max_approach_up={max_up})"
                    )
                    return []

            # 4. Optional antipodal validation; else take the top quality-sorted poses.
            #    Both paths fold in the approach-bias bonus when pref_active, plus
            #    the "side"-only centrality bonus when cen_w > 0.
            if antipodal:
                grasps = self._antipodal_select(
                    gg, obj, n_out, up_unit, preference, weight, cen_w, x_target
                )
            else:
                if pref_active:
                    approaches = np.asarray(gg.rotation_matrices)[:, :, 0]
                    blended = gg.scores + weight * self._approach_scores(
                        approaches, up_unit, preference
                    )
                    if cen_w > 0:
                        centroid, scale = self._cloud_center(obj)
                        blended = blended + cen_w * self._center_scores(
                            np.asarray(gg.translations), centroid, scale
                        )
                    gg = gg[np.argsort(-blended)]
                grasps = self._as_dicts(gg[:n_out], x_target)

            print(
                f"[grasp] {len(grasps)} grasp(s) | {len(pts_in)} pts in | "
                f"{len(obj)} after filter | {len(pts)} fed | "
                f"antipodal={antipodal} | preference={preference} | infer {infer_ms:.0f} ms"
            )
        except Exception as exc:
            print(f"[grasp] inference failed: {exc}")
            grasps = []
        return grasps

    # ------------------------------------------------------------------
    # Pipeline helpers (ported from grasp_node.py, ROS stripped)
    # ------------------------------------------------------------------

    def _run_graspnet(self, pts: np.ndarray) -> np.ndarray:
        """GraspNet forward + decode → ``(num_grasps, 17)`` GraspGroup array."""
        torch = self._torch
        cloud_t = torch.from_numpy(np.ascontiguousarray(pts, dtype=np.float32))
        cloud_t = cloud_t.unsqueeze(0).to(self._device)
        with torch.no_grad():
            ep = self._net({"point_clouds": cloud_t})
            preds = self._pred_decode(ep)
        if self._device.startswith("cuda"):
            torch.cuda.synchronize()
        return preds[0].detach().cpu().numpy()

    def _sample_points(self, pts: np.ndarray, num_point: int) -> np.ndarray:
        """Random-sample (with replacement if short) to exactly ``num_point``."""
        replace = len(pts) < num_point
        idx = np.random.choice(len(pts), num_point, replace=replace)
        return pts[idx]

    def _filter_world(
        self, pts: np.ndarray, voxel: float, outlier_removal: bool, cluster_filter: bool
    ) -> np.ndarray:
        """Voxel-downsample, then optional statistical outlier removal + largest cluster."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        if voxel and voxel > 0:
            pcd = pcd.voxel_down_sample(voxel)
        pts = np.asarray(pcd.points, dtype=np.float32)

        if outlier_removal and len(pts) > self._outlier_nb_neighbors:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
            _, idx = pcd.remove_statistical_outlier(
                nb_neighbors=self._outlier_nb_neighbors,
                std_ratio=self._outlier_std_ratio,
            )
            pts = pts[idx]

        if cluster_filter and len(pts) >= self._cluster_min_samples:
            pts = self._keep_largest_cluster(pts)
        return pts

    def _keep_largest_cluster(self, pts: np.ndarray) -> np.ndarray:
        """DBSCAN — keep the cluster with the most points (count only)."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
        labels = np.array(
            pcd.cluster_dbscan(
                eps=self._cluster_eps,
                min_points=self._cluster_min_samples,
                print_progress=False,
            )
        )
        unique = set(labels) - {-1}
        if not unique:
            return pts
        best = max(unique, key=lambda lbl: int((labels == lbl).sum()))
        return pts[labels == best]

    def _select_pool(self, raw: np.ndarray, score_threshold: float, pool_size: int):
        """Sort + NMS + score-threshold, keep up to ``pool_size`` candidates."""
        gg = self._GraspGroup(raw)
        gg = gg.sort_by_score()[:2000].nms()
        if score_threshold > 0:
            gg = gg[gg.scores >= score_threshold]
        if len(gg) == 0:
            return gg
        return gg[:pool_size]

    @staticmethod
    def _approach_scores(
        approaches: np.ndarray, up_unit: np.ndarray, preference: str
    ) -> np.ndarray:
        """Approach-bias score in ``[0, 1]`` per grasp (frame-agnostic).

        ``approaches`` is ``(N, 3)`` of raw approach vectors (rotation col-0);
        ``up_unit`` is the unit "up" direction in the cloud frame (gravity =
        ``-up``). ``top`` rewards approaches pointing down (along gravity);
        ``side`` rewards horizontal approaches. Any other preference → zeros.
        """
        a = approaches / np.clip(
            np.linalg.norm(approaches, axis=1, keepdims=True), 1e-9, None
        )
        d = a @ np.asarray(up_unit, dtype=np.float64)
        if preference == "top":
            return np.clip(-d, 0.0, 1.0)
        if preference == "side":
            return 1.0 - np.abs(d)
        return np.zeros(len(approaches))

    @staticmethod
    def _cloud_center(obj_pts: np.ndarray) -> tuple[np.ndarray, float]:
        """Centroid + characteristic radius (max point distance) of a cloud.

        The radius is the scale ``_center_scores`` normalises against, so a grasp
        at the far surface scores ~0 and one at the centroid ~1.
        """
        pts = np.asarray(obj_pts, dtype=np.float64)
        centroid = pts.mean(axis=0)
        scale = float(np.linalg.norm(pts - centroid, axis=1).max()) if len(pts) else 0.0
        return centroid, scale

    @staticmethod
    def _center_scores(
        translations: np.ndarray, centroid: np.ndarray, scale: float
    ) -> np.ndarray:
        """Centrality score in ``[0, 1]`` per grasp — higher near the centroid.

        ``translations`` is ``(N, 3)`` of grasp centres; ``centroid`` and
        ``scale`` come from ``_cloud_center``. A grasp at the centroid scores 1,
        one a full radius (or more) away 0. Used by the "side" preference to
        favour mid-object grips over ones near an edge.
        """
        d = np.linalg.norm(
            np.asarray(translations, dtype=np.float64)
            - np.asarray(centroid, dtype=np.float64),
            axis=1,
        )
        return np.clip(1.0 - d / max(float(scale), 1e-9), 0.0, 1.0)

    @staticmethod
    def _approach_up_mask(
        approaches: np.ndarray, up_unit: np.ndarray, max_up_cos: float
    ) -> np.ndarray:
        """Boolean keep-mask dropping "bottom-up" grasps (frame-agnostic).

        ``approaches`` is ``(N, 3)`` of raw approach vectors (rotation col-0);
        ``up_unit`` is the unit "up" in the cloud frame (gravity = ``-up``). A
        grasp is dropped when its approach points upward (along ``up``) by more
        than ``max_up_cos`` — i.e. keep where ``approach·up <= max_up_cos``. A
        horizontal approach scores 0, a straight-up (bottom-up) approach +1, so
        ``max_up_cos = 0`` keeps only at/below-horizontal approaches and
        ``1.0`` disables the filter.
        """
        a = approaches / np.clip(
            np.linalg.norm(approaches, axis=1, keepdims=True), 1e-9, None
        )
        return (a @ np.asarray(up_unit, dtype=np.float64)) <= float(max_up_cos)

    @staticmethod
    def _rpy_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        """Local roll/pitch/yaw offset about the gripper X/Y/Z axes (post-multiply).

        Composed ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` — the URDF/ROS RPY order —
        which reduces to the pitch-only ``Ry(pitch)`` when roll = yaw = 0.
        """
        rx, ry, rz = (np.radians(float(d)) for d in (roll_deg, pitch_deg, yaw_deg))
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
        Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
        Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        return Rz @ Ry @ Rx

    @property
    def rotation_offset(self) -> np.ndarray:
        """The composed roll/pitch/yaw offset applied to every output rotation."""
        return self._R_offset

    def _emit_rotation(
        self, rot: np.ndarray, x_target: np.ndarray | None = None
    ) -> list[list[float]]:
        """Apply the configured rotation offset and return a JSON-friendly 3x3.

        When ``x_target`` (a unit direction in the cloud frame) is given and the
        offset rotation's X axis points away from it (``X·x_target < 0``), roll
        the gripper 180 about its own Z so X swings toward ``x_target``. ``None``
        (the default) leaves the rotation as-is. The upright-X behaviour passes
        ``+up`` (X ends up pointing up); its inverse passes ``-up`` (X down).
        """
        r = np.asarray(rot, dtype=np.float64) @ self._R_offset
        if x_target is not None and float(r[:, 0] @ np.asarray(x_target, dtype=np.float64)) < 0.0:
            r = r @ self._RZ_180
        return [[float(v) for v in row] for row in r]

    def _as_dicts(self, gg, x_target: np.ndarray | None = None) -> list[dict]:
        """Serialize a GraspGroup to grasp dicts (no antipodal refinement)."""
        out = []
        for k in range(len(gg)):
            out.append(
                {
                    "translation": [float(x) for x in gg.translations[k]],
                    "rotation": self._emit_rotation(gg.rotation_matrices[k], x_target),
                    "width": float(gg.widths[k]),
                    "score": float(gg.scores[k]),
                    "antipodal_score": None,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Antipodal validation (optional) — ported from /grasp/pos
    # ------------------------------------------------------------------

    def _antipodal_select(
        self,
        gg,
        obj_pts: np.ndarray,
        n_out: int,
        up_unit: np.ndarray | None = None,
        preference: str = "none",
        weight: float = 0.0,
        center_weight: float = 0.0,
        x_target: np.ndarray | None = None,
    ) -> list[dict]:
        """Keep grasps that lie on the object surface; refine width/centre + score."""
        obj_normals = self._estimate_normals(obj_pts)
        lo = obj_pts.min(axis=0) - self._select_margin_m
        hi = obj_pts.max(axis=0) + self._select_margin_m
        pref_active = preference in ("side", "top") and up_unit is not None
        # Centre bias is "side"-only; precompute the cloud centroid + scale once.
        side_center = preference == "side" and center_weight > 0
        centroid, scale = self._cloud_center(obj_pts) if side_center else (None, 0.0)

        cand: list[dict] = []
        for k in range(len(gg)):
            c = np.asarray(gg.translations[k], dtype=np.float64)
            if not np.all((c >= lo) & (c <= hi)):
                continue  # grasp centre not on this object
            rot = np.asarray(gg.rotation_matrices[k], dtype=np.float64)
            closing = rot[:, 1]  # GraspNet col-1 = closing/spread direction
            anti, width, c_ref = self._antipodal(
                c, closing, obj_pts.astype(np.float64), obj_normals, float(gg.widths[k])
            )
            # Approach-bias bonus on the raw approach (col-0); 0 when inactive.
            pref = (
                float(self._approach_scores(rot[:, 0][None, :], up_unit, preference)[0])
                if pref_active
                else 0.0
            )
            # Centrality bonus on the *refined* centre; 0 unless side-center.
            cen = (
                float(self._center_scores(c_ref[None, :], centroid, scale)[0])
                if side_center
                else 0.0
            )
            cand.append(
                {"rot": rot, "center": c_ref, "width": width,
                 "gn": float(gg.scores[k]), "anti": anti, "pref": pref, "cen": cen}
            )
        if not cand:
            return []

        # Rank by quality + antipodal (the ROS /grasp/pos default blend), plus the
        # weighted approach-bias bonus when a preference is active and the
        # "side"-only centrality bonus.
        cand.sort(
            key=lambda d: d["gn"] + d["anti"] + weight * d["pref"] + center_weight * d["cen"],
            reverse=True,
        )
        out = []
        for d in cand[:n_out]:
            out.append(
                {
                    "translation": [float(x) for x in d["center"]],
                    "rotation": self._emit_rotation(d["rot"], x_target),
                    "width": float(d["width"]),
                    "score": float(d["gn"]),
                    "antipodal_score": float(d["anti"]),
                }
            )
        return out

    def _estimate_normals(self, obj_pts: np.ndarray) -> np.ndarray:
        """Per-point normals oriented outward from the object centroid."""
        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(obj_pts.astype(np.float64))
        pcd.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self._normal_radius_m, max_nn=self._normal_max_nn
            )
        )
        n = np.asarray(pcd.normals)
        d = obj_pts - obj_pts.mean(axis=0)
        flip = (n * d).sum(axis=1) < 0
        n[flip] *= -1.0
        return n

    def _antipodal(self, center, closing, obj_pts, obj_normals, gn_width):
        """Antipodal quality + refined (width, centre) for a grasp on the surface.

        Returns ``(score in [0,1], width, centre)``. Neutral (0.5, gn_width, center)
        when too few surface points fall in the closing tube to evaluate.
        """
        u = _unit(np.asarray(closing, dtype=np.float64))
        if u is None:
            return 0.5, gn_width, center
        rel = obj_pts - center
        s = rel @ u
        perp = np.linalg.norm(rel - np.outer(s, u), axis=1)
        near = perp < self._antipodal_r_tol_m
        if int(near.sum()) < self._antipodal_min_pts:
            return 0.5, gn_width, center  # neutral, no refine
        s_n = s[near]
        n_n = obj_normals[near]
        ip, im = int(np.argmax(s_n)), int(np.argmin(s_n))
        a_plus = float(np.dot(_unit(n_n[ip]) if _unit(n_n[ip]) is not None else n_n[ip], u))
        a_minus = float(np.dot(_unit(n_n[im]) if _unit(n_n[im]) is not None else n_n[im], -u))
        score = float(np.clip(a_plus, 0, 1) * np.clip(a_minus, 0, 1))
        contact_w = float(s_n[ip] - s_n[im])
        width = float(min(contact_w + self._width_clearance_m, self._max_gripper_width_m))
        centre = center + u * ((s_n[ip] + s_n[im]) / 2.0)
        return score, width, centre

    def get_model_name(self) -> str:
        return self._model_name
