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
WINDOW_APPEARANCE = "vision-debug: appearance"
WINDOW_GRASP = "vision-debug: grasp"


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
        # Grasp updates ride a separate queue — they carry a (cloud, grasps)
        # 3D scene rendered with matplotlib rather than a ready-made image.
        self._grasp_queue: queue.Queue[tuple] = queue.Queue(maxsize=4)
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

    def push_grasp(
        self, cloud: np.ndarray, grasps: list, rotation_offset=None
    ) -> None:
        if self._failed:
            return
        item = (cloud, grasps, rotation_offset)
        try:
            self._grasp_queue.put_nowait(item)
        except queue.Full:
            try:
                self._grasp_queue.get_nowait()
                self._grasp_queue.put_nowait(item)
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
        # Lazily-built matplotlib 3D window for grasps: {top, fig, ax, canvas}.
        grasp_ui: dict = {}

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

            # Grasp scene (only the latest matters — drain to the newest).
            latest = None
            while True:
                try:
                    latest = self._grasp_queue.get_nowait()
                except queue.Empty:
                    break
            if latest is not None:
                try:
                    _render_grasp_window(tk, root, grasp_ui, *latest)
                except Exception as exc:
                    print(f"[VISION_DEBUG] failed to render grasp: {exc}")

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


def show_appearance(image: Image.Image, embedding: "list[float]") -> None:
    """Show the person crop the agent sent — there is no bbox to draw; the
    useful signal is what the crop actually contains."""
    viewer = _get_viewer()
    if viewer is None:
        return
    bgr = _pil_to_bgr(image)
    norm = float(np.sqrt(np.dot(embedding, embedding))) if embedding else 0.0
    _draw_text_block(bgr, [f"dim={len(embedding)} norm={norm:.3f}"])
    print(f"[VISION_DEBUG appearance] dim={len(embedding)}")
    viewer.push(WINDOW_APPEARANCE, _bgr_to_pil(bgr))


# ----------------------------------------------------------------------------
# Grasp viewer — a 3D scene (matplotlib), not a 2D image
# ----------------------------------------------------------------------------
#
# The grasp output is a point cloud plus 6-DOF poses, so it can't be drawn into
# the Tk image pipeline above. We render it with a matplotlib 3D axes embedded
# in the *same* Tk worker thread/root the image windows use — that route already
# works over X11/XWayland here, unlike Open3D's GLFW (Wayland) backend. The
# window is interactive: mouse-drag to orbit, and the toolbar zooms/pans/saves.


def _box_faces(lo: tuple, hi: tuple) -> list:
    """The six quad faces of an axis-aligned box from corner ``lo`` to ``hi``."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],  # 0-3 bottom
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],  # 4-7 top
        ]
    )
    idx = [
        [0, 1, 2, 3], [4, 5, 6, 7],  # bottom, top (z)
        [0, 1, 5, 4], [3, 2, 6, 7],  # front, back (y)
        [1, 2, 6, 5], [0, 3, 7, 4],  # right, left (x)
    ]
    return [corners[i] for i in idx]


def _gripper_mesh(grasp: dict, rotation_offset: np.ndarray | None = None):
    """A parallel-jaw gripper as world-space quad faces + color.

    Built from four boxes — two fingers, a base plate joining them, and a rear
    handle — in the gripper's local frame (x = approach / R col-0, y = closing /
    R col-1), then rotated by R and shifted to ``translation`` (the grasp centre
    between the fingertips). Color goes red (low score) → green (high score).

    The provider may have post-rotated the output rotation by ``rotation_offset``
    (a roll/pitch/yaw matrix) to match a robot's tool convention; we undo it with
    its transpose so the drawn gripper shows GraspNet's true approach onto the cloud.
    """
    c = np.asarray(grasp["translation"], dtype=float)
    R = np.asarray(grasp["rotation"], dtype=float)
    if rotation_offset is not None:
        R = R @ np.asarray(rotation_offset, dtype=float).T  # undo offset (R is orthonormal)
    half = min(max(float(grasp.get("width") or 0.0), 0.0), 0.15) / 2.0

    fw = 0.006      # finger / plate thickness (closing axis)
    hh = 0.006 / 2  # half gripper height (third axis)
    depth = 0.045   # finger length along approach
    tail = 0.04     # rear handle length

    boxes = [
        ((0.0, -half - fw, -hh), (depth, -half, hh)),    # left finger
        ((0.0, half, -hh), (depth, half + fw, hh)),      # right finger
        ((-fw, -half - fw, -hh), (0.0, half + fw, hh)),  # base plate
        ((-tail - fw, -fw / 2, -hh), (-fw, fw / 2, hh)),  # rear handle
    ]
    faces_local = [f for lo, hi in boxes for f in _box_faces(lo, hi)]
    world = (R @ np.array(faces_local).reshape(-1, 3).T).T + c
    faces = world.reshape(-1, 4, 3)  # (n_faces, 4 corners, xyz)

    s = min(max(float(grasp.get("score") or 0.0), 0.0), 1.0)
    return faces, (1.0 - s, s, 0.1)


def _draw_grasp_axes(
    ax, cloud: np.ndarray, grasps: "list[dict]",
    rotation_offset: np.ndarray | None = None,
) -> None:
    """(Re)draw the gray cloud + one solid gripper per grasp on ``ax``."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.clear()
    pts = np.asarray(cloud, dtype=float).reshape(-1, 3)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) > 4000:  # keep the scatter responsive to orbit
        pts = pts[np.random.choice(len(pts), 4000, replace=False)]
    if len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=2, c="0.6", depthshade=False)

    bounds = [pts] if len(pts) else []
    for g in grasps:
        faces, color = _gripper_mesh(g, rotation_offset)
        bounds.append(faces.reshape(-1, 3))
        ax.add_collection3d(
            Poly3DCollection(
                list(faces), facecolor=color, edgecolor=(0, 0, 0, 0.4),
                linewidths=0.3, alpha=0.9,
            )
        )

    scores = [float(g.get("score") or 0.0) for g in grasps]
    top = f"{max(scores):.3f}" if scores else "n/a"
    ax.set_title(f"grasps={len(grasps)}  top_score={top}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    # Equal aspect so grippers aren't distorted: cube limits around the data.
    if bounds:
        P = np.vstack(bounds)
        ctr = (P.max(0) + P.min(0)) / 2.0
        r = float((P.max(0) - P.min(0)).max()) / 2.0 or 0.05
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass


def _render_grasp_window(
    tk, root, ui: dict, cloud: np.ndarray, grasps: list,
    rotation_offset: np.ndarray | None = None,
) -> None:
    """Lazily build the embedded matplotlib window, then redraw the scene.

    Runs on the Tk worker thread (called from the viewer's pump), so all Tk and
    matplotlib widget access stays single-threaded. ``ui`` holds the persistent
    ``{top, fig, ax, canvas}`` state across calls.
    """
    if "canvas" not in ui:
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        top = tk.Toplevel(root)
        top.title(WINDOW_GRASP)
        top.protocol("WM_DELETE_WINDOW", top.withdraw)
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(projection="3d")
        canvas = FigureCanvasTkAgg(fig, master=top)
        canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(canvas, top).update()
        ui.update(top=top, fig=fig, ax=ax, canvas=canvas)

    if not ui["top"].winfo_viewable():
        ui["top"].deiconify()
    _draw_grasp_axes(ui["ax"], cloud, grasps, rotation_offset)
    ui["canvas"].draw_idle()


def show_grasp(
    cloud: np.ndarray, grasps: "list[dict]", rotation_offset: np.ndarray | None = None
) -> None:
    """Render the input cloud and generated grasps in an interactive 3D window."""
    viewer = _get_viewer()
    if viewer is None:
        return
    scores = [float(g.get("score") or 0.0) for g in grasps]
    top = f"{max(scores):.3f}" if scores else "n/a"
    print(
        f"[VISION_DEBUG grasp] grasps={len(grasps)} points={len(cloud)} top_score={top}"
    )
    viewer.push_grasp(np.asarray(cloud), list(grasps), rotation_offset)
