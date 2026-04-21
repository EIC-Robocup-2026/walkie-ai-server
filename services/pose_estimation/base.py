"""Base classes and data structures for pose estimation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from PIL import Image

from ..object_detection.base import DetectedObject

# 17 COCO keypoint names in index order (0-16).
COCO_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

# Skeleton connections for visualisation (pairs of keypoint indices).
SKELETON_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # Face
    (0, 1), (0, 2), (1, 3), (2, 4),
    # Shoulders & arms
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    # Torso
    (5, 11), (6, 12), (11, 12),
    # Legs
    (11, 13), (13, 15), (12, 14), (14, 16),
)


@dataclass
class PoseKeypoint:
    """A single detected keypoint on a person."""

    x: float
    """Pixel x-coordinate."""
    y: float
    """Pixel y-coordinate."""
    confidence: float
    """Detection confidence in [0, 1]."""
    name: str
    """Human-readable name (e.g. ``'nose'``, ``'left_shoulder'``)."""
    index: int
    """COCO keypoint index (0-16)."""


@dataclass
class PersonPose:
    """A detected person together with their pose keypoints."""

    bbox: tuple[int, int, int, int]
    """Bounding box in ``(cx, cy, w, h)`` format (matching the existing YOLO
    object-detection convention used elsewhere in this project)."""
    confidence: float
    """Person detection confidence."""
    keypoints: list[PoseKeypoint] = field(default_factory=list)
    """List of 17 COCO keypoints."""
    cropped_image: Image.Image | None = None
    """Cropped region of the person with optional padding."""


@dataclass
class PersonDetectionWithPose:
    """Groups a person from object detection with a matched pose estimation.

    Produced by :pymeth:`WalkieVision.detect_persons_with_pose`.
    """

    detection: DetectedObject
    """Person ``DetectedObject`` from the object-detection provider."""
    pose: PersonPose | None
    """Matched ``PersonPose`` from the pose-estimation provider, or ``None``
    if no sufficiently close pose was found."""
    distance: float
    """Matching distance (Euclidean between bbox centres).  ``0.0`` when
    *pose* is ``None``."""


class PoseEstimationProvider(ABC):
    """Abstract base class for pose estimation providers."""

    @abstractmethod
    def estimate(self, image: Image.Image) -> list[PersonPose]:
        """Estimate poses for all persons detected in *image*.

        Args:
            image: PIL Image (RGB) to process.

        Returns:
            List of ``PersonPose`` instances.
        """

    def load_model(self) -> None:
        """Pre-load model weights into memory.

        Default implementation is a no-op.  Override in providers that use
        lazy loading so that ``load_model()`` can be called eagerly.
        """

    @abstractmethod
    def get_model_name(self) -> str:
        """Return a short model name for logging."""
