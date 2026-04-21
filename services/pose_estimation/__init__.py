"""Pose estimation module with pluggable providers."""

from .base import (
    COCO_KEYPOINT_NAMES,
    SKELETON_CONNECTIONS,
    PersonDetectionWithPose,
    PersonPose,
    PoseEstimationProvider,
    PoseKeypoint,
)
from .pose_estimation import PoseEstimation

__all__ = [
    "COCO_KEYPOINT_NAMES",
    "SKELETON_CONNECTIONS",
    "PersonDetectionWithPose",
    "PersonPose",
    "PoseEstimation",
    "PoseEstimationProvider",
    "PoseKeypoint",
]
