"""Debug viewer for vision services.

When the ``VISION_DEBUG`` environment variable is truthy, the most recent
image and inference output for ``image_caption``, ``object_detection``, and
``pose_estimation`` are rendered to dedicated Tkinter windows.

We use Tkinter rather than ``cv2.imshow`` because the OpenCV wheel that
actually wins the import in this venv is ``opencv-python-headless`` (pulled
in by ``vllm``), which has no GUI support. Tkinter ships with the stdlib
and works on any X11/Wayland session.

All Tk objects are owned by a single background thread; route handlers push
``(window, PIL.Image)`` updates onto a queue.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .face_recognition.base import FaceEmbedding
    from .object_detection.base import DetectedObject
    from .pose_estimation.base import PersonPose

_DEBUG_ENV_VAR = "VISION_DEBUG"

WINDOW_IMAGE_CAPTION = "vision-debug: image_caption"
WINDOW_OBJECT_DETECTION = "vision-debug: object_detection"
WINDOW_POSE_ESTIMATION = "vision-debug: pose_estimation"
WINDOW_FACE_RECOGNITION = "vision-debug: face_recognition"


def is_enabled() -> bool:
    return os.getenv(_DEBUG_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class _Frame:
    window: str
    image: Image.Image  # already-annotated RGB image


class _DebugViewer:
    """Owns a Tk root and one Toplevel per window name, on a worker thread."""

    def __init__(self) -> None:
        self._queue: queue.Queue[_Frame] = queue.Queue(maxsize=16)
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._failed = False

    def start(self) -> None:
        with self._lock:
            if self._thread is not None or self._failed:
                return
            self._thread = threading.Thread(
                target=self._run, name="vision-debug-viewer", daemon=True
            )
            self._thread.start()

    def push(self, window: str, image: Image.Image) -> None:
        if self._failed:
            return
        try:
            self._queue.put_nowait(_Frame(window, image))
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(_Frame(window, image))
            except queue.Empty:
                pass

    def _run(self) -> None:
        try:
            import tkinter as tk
            from PIL import ImageTk
        except Exception as exc:  # pragma: no cover - missing tk
            self._failed = True
            print(f"[VISION_DEBUG] viewer disabled (tkinter unavailable): {exc}")
            return

        try:
            root = tk.Tk()
        except Exception as exc:
            self._failed = True
            print(f"[VISION_DEBUG] viewer disabled (no display: {exc})")
            return

        root.withdraw()  # hide the implicit root; we use Toplevels per window
        windows: dict[str, tuple[tk.Toplevel, tk.Label]] = {}
        # Keep PhotoImage refs alive — Tk will garbage-collect them otherwise.
        photo_refs: dict[str, "ImageTk.PhotoImage"] = {}

        def pump() -> None:
            drained = 0
            while drained < 8:  # avoid starving the Tk main loop
                try:
                    frame = self._queue.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                try:
                    if frame.window not in windows:
                        top = tk.Toplevel(root)
                        top.title(frame.window)
                        top.protocol(
                            "WM_DELETE_WINDOW",
                            lambda w=frame.window: _hide_window(w),
                        )
                        label = tk.Label(top)
                        label.pack()
                        windows[frame.window] = (top, label)
                    top, label = windows[frame.window]
                    if not top.winfo_viewable():
                        top.deiconify()
                    photo = ImageTk.PhotoImage(frame.image)
                    label.configure(image=photo)
                    photo_refs[frame.window] = photo
                    top.geometry(f"{frame.image.width}x{frame.image.height}")
                except Exception as exc:
                    print(f"[VISION_DEBUG] failed to render {frame.window}: {exc}")
            root.after(40, pump)

        def _hide_window(name: str) -> None:
            entry = windows.get(name)
            if entry is not None:
                entry[0].withdraw()

        root.after(40, pump)
        try:
            root.mainloop()
        except Exception as exc:
            self._failed = True
            print(f"[VISION_DEBUG] viewer thread crashed: {exc}")


_viewer: _DebugViewer | None = None
_viewer_lock = threading.Lock()


def _get_viewer() -> _DebugViewer | None:
    global _viewer
    if not is_enabled():
        return None
    with _viewer_lock:
        if _viewer is None:
            _viewer = _DebugViewer()
            _viewer.start()
    return _viewer


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _wrap(text: str, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_text_block(
    img: np.ndarray,
    lines: list[str],
    origin: tuple[int, int] = (8, 22),
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(img, line, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 1, cv2.LINE_AA)
        y += 22


def show_image_caption(image: Image.Image, caption: str) -> None:
    viewer = _get_viewer()
    if viewer is None:
        return
    bgr = _pil_to_bgr(image)
    lines = ["caption:"] + _wrap(caption or "<empty>", width=70)
    _draw_text_block(bgr, lines)
    print(f"[VISION_DEBUG image_caption] {caption!r}")
    viewer.push(WINDOW_IMAGE_CAPTION, _bgr_to_pil(bgr))


def show_object_detection(
    image: Image.Image, detections: "list[DetectedObject]"
) -> None:
    viewer = _get_viewer()
    if viewer is None:
        return
    bgr = _pil_to_bgr(image)
    for d in detections:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        parts = []
        if d.class_name:
            parts.append(d.class_name)
        if d.confidence is not None:
            parts.append(f"{d.confidence:.2f}")
        label = " ".join(parts) or "obj"
        cv2.putText(bgr, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    summary_lines = [f"detections: {len(detections)}"]
    for i, d in enumerate(detections[:8]):
        conf = f"{d.confidence:.2f}" if d.confidence is not None else "?"
        name = d.class_name or "?"
        summary_lines.append(f"[{i}] {name} conf={conf} area={d.area_ratio:.3f}")
    _draw_text_block(bgr, summary_lines)
    print(f"[VISION_DEBUG object_detection] count={len(detections)}")
    for i, d in enumerate(detections):
        print(f"  [{i}] class={d.class_name} conf={d.confidence} bbox={d.bbox}")
    viewer.push(WINDOW_OBJECT_DETECTION, _bgr_to_pil(bgr))


def show_pose_estimation(image: Image.Image, poses: "list[PersonPose]") -> None:
    viewer = _get_viewer()
    if viewer is None:
        return
    from .pose_estimation.base import SKELETON_CONNECTIONS

    bgr = _pil_to_bgr(image)
    for p in poses:
        cx, cy, w, h = p.bbox
        x1, y1 = int(cx - w / 2), int(cy - h / 2)
        x2, y2 = int(cx + w / 2), int(cy + h / 2)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 200, 255), 2)
        for a, b in SKELETON_CONNECTIONS:
            if a < len(p.keypoints) and b < len(p.keypoints):
                ka, kb = p.keypoints[a], p.keypoints[b]
                if ka.confidence > 0.3 and kb.confidence > 0.3:
                    cv2.line(bgr, (int(ka.x), int(ka.y)),
                             (int(kb.x), int(kb.y)), (0, 255, 0), 2)
        for kp in p.keypoints:
            if kp.confidence > 0.3:
                cv2.circle(bgr, (int(kp.x), int(kp.y)), 4, (0, 0, 255), -1)
    summary_lines = [f"persons: {len(poses)}"]
    for i, p in enumerate(poses[:8]):
        visible = sum(1 for k in p.keypoints if k.confidence > 0.3)
        summary_lines.append(
            f"[{i}] conf={p.confidence:.2f} visible_kpts={visible}/{len(p.keypoints)}"
        )
    _draw_text_block(bgr, summary_lines)
    print(f"[VISION_DEBUG pose_estimation] persons={len(poses)}")
    viewer.push(WINDOW_POSE_ESTIMATION, _bgr_to_pil(bgr))


def show_face_recognition(image: Image.Image, faces: "list[FaceEmbedding]") -> None:
    viewer = _get_viewer()
    if viewer is None:
        return
    bgr = _pil_to_bgr(image)
    for f in faces:
        x1, y1, x2, y2 = (int(v) for v in f.bbox_xyxy)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 200, 0), 2)
        label = f"{f.det_score:.2f}"
        cv2.putText(bgr, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1, cv2.LINE_AA)
    summary_lines = [f"faces: {len(faces)}"]
    for i, f in enumerate(faces[:8]):
        dim = len(f.embedding)
        summary_lines.append(f"[{i}] det={f.det_score:.2f} dim={dim}")
    _draw_text_block(bgr, summary_lines)
    print(f"[VISION_DEBUG face_recognition] faces={len(faces)}")
    viewer.push(WINDOW_FACE_RECOGNITION, _bgr_to_pil(bgr))
