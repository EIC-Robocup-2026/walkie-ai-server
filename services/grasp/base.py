"""Base classes for the grasp-pose generation service.

The server-side grasp service is **stateless and frame-agnostic**: one segmented
object point cloud in → a ranked list of 6-DOF grasp poses out, expressed in the
*same* frame as the input cloud. The provider does **no** robot transforms (no TF,
no planning frame, no end-effector alignment, no pre-grasp back-off). The one
exception is the optional ``approach_preference`` re-rank, which stays
frame-agnostic because the client supplies its reference ``up`` vector **in the
cloud frame** (gravity = ``-up``) per request; everything else frame-specific
lives in the agent/client.

This mirrors the on-robot ROS ``grasp_node``'s ``/grasp/from_cloud`` path (cloud →
GraspNet → filtered/NMS'd, quality-sorted poses), with the optional antipodal
surface-normal validation from its ``/grasp/pos`` path folded in as a flag (it only
needs the supplied cloud, so it is frame-agnostic too). All the ROS/TF machinery is
stripped away — the network only ever sees points in one frame and returns grasps in
that same frame.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class GraspProvider(ABC):
    """Abstract base class for grasp-pose generation providers."""

    @abstractmethod
    def infer(self, cloud: np.ndarray, **opts: Any) -> list[dict]:
        """Generate grasp poses for an ``(N, 3)`` point cloud in its own frame.

        The provider grasps **whatever cloud it is given** — no detection,
        segmentation, or framing happens here; the agent lifts the object's
        mask/bbox to points and (recommended) sends them in the camera-optical
        frame, then maps the returned grasps back itself.

        Args:
            cloud: ``(N, 3)`` float32 array of XYZ points in a single frame.
            **opts: Per-request overrides (``score_threshold``, ``max_grasps``,
                ``voxel_size``, ``num_point``, ``outlier_removal``,
                ``cluster_filter``, ``antipodal`` + its params, and the
                approach-bias re-rank ``approach_preference`` /``up`` /
                ``approach_weight`` /``center_weight`` /``closing_weight`` — the
                last two ``"side"``-only nudges toward grasps near the cloud
                centroid and with a horizontal gripper-width axis).
                Anything unset falls back to the provider's configured default.

        Returns:
            A list of grasp dicts sorted best-first, each shaped::

                {"translation": [x, y, z],
                 "rotation": [[r00, r01, r02], ...],   # 3x3, col-0 = approach
                 "width": float,                        # gripper opening, metres
                 "score": float,                        # GraspNet quality
                 "antipodal_score": float | None}       # None unless antipodal=True

            All geometry is in the **input cloud's frame**. ``col-0 = approach``
            holds unless a provider applies an output rotation offset to match a
            robot's tool convention (e.g. GraspNet's roll/pitch/yaw ``_offset_deg``).
        """

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op. Override in providers that use lazy
        loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short model name for logging / provenance."""
