"""Shared pytest fixtures for walkie-agent-v2 API tests."""

import io
import struct

import numpy as np
import pytest
from PIL import Image


BASE_URL = "http://localhost:5000"


def pytest_addoption(parser):
    parser.addoption(
        "--base-url",
        default=BASE_URL,
        help="Base URL of the running walkie-agent-v2 server",
    )


@pytest.fixture(scope="session")
def base_url(request):
    return request.config.getoption("--base-url")


@pytest.fixture(scope="session")
def sample_image_bytes() -> bytes:
    """A minimal 100x100 RGB PNG image."""
    img = Image.new("RGB", (100, 100), color=(100, 149, 237))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_cloud_npy() -> bytes:
    """A synthetic graspable object — points on a small (4 cm) box ~0.4 m in front
    of the camera, in the optical frame (X-right, Y-down, Z-forward). Serialized as
    ``.npy`` bytes, the shape the /grasp endpoint expects."""
    rng = np.random.default_rng(0)
    half = 0.02  # 4 cm box → 0.04 m gripper-scale object
    center = np.array([0.0, 0.0, 0.4], dtype=np.float32)
    faces = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            pts = rng.uniform(-half, half, size=(500, 3)).astype(np.float32)
            pts[:, axis] = sign * half
            faces.append(pts)
    box = np.concatenate(faces, axis=0) + center
    buf = io.BytesIO()
    np.save(buf, box.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_can_npy() -> bytes:
    """A synthetic upright can — a vertical cylinder (r=3 cm, h=12 cm) ~0.4 m in
    front of the camera, in the optical frame (X-right, Y-down, Z-forward), so its
    axis runs along Y. Used to exercise the side/top approach preference: world-up
    is ``-Y`` (gravity = ``+Y``). Serialized as ``.npy`` bytes."""
    rng = np.random.default_rng(0)
    r, h, zc = 0.03, 0.12, 0.40
    th = rng.uniform(0, 2 * np.pi, 2500)
    yy = rng.uniform(-h / 2, h / 2, 2500)
    side = np.stack([r * np.cos(th), yy, zc + r * np.sin(th)], axis=1)
    th2 = rng.uniform(0, 2 * np.pi, 500)
    rr = r * np.sqrt(rng.uniform(0, 1, 500))
    cap = np.stack([rr * np.cos(th2), np.full(500, -h / 2), zc + rr * np.sin(th2)], axis=1)
    can = np.vstack([side, cap]).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, can, allow_pickle=False)
    return buf.getvalue()


@pytest.fixture(scope="session")
def sample_audio_bytes() -> bytes:
    """A minimal silent WAV file: 1 second, 16 kHz, mono, 16-bit PCM."""
    sample_rate = 16_000
    num_channels = 1
    bits_per_sample = 16
    num_samples = sample_rate  # 1 second of silence
    data_size = num_samples * num_channels * (bits_per_sample // 8)

    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))                                          # chunk size
    buf.write(struct.pack("<H", 1))                                           # PCM
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * num_channels * bits_per_sample // 8))
    buf.write(struct.pack("<H", num_channels * bits_per_sample // 8))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    buf.seek(0)
    return buf.getvalue()
