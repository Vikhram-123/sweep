"""Synthetic videos, garments and fake models for tests (no model downloads)."""

from __future__ import annotations

import hashlib
import itertools
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from sweep.detect import RawDetection

WALL = (205, 200, 190)
# Synthetic closet: (label, RGB, x0, y0, x1, y1) on a 2400x600 panorama.
GARMENTS = [
    ("t-shirt", (200, 30, 40), 150, 80, 450, 360),     # red tee
    ("jeans", (45, 70, 120), 650, 120, 870, 560),      # denim
    ("jacket", (35, 80, 45), 1080, 60, 1420, 470),     # forest green jacket
    ("sneakers", (245, 245, 242), 1650, 420, 1900, 540),  # white sneakers
    ("dress", (240, 160, 185), 2050, 60, 2300, 520),   # pink dress
]


def panorama(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.empty((600, 2400, 3), np.uint8)
    img[:] = WALL
    img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)  # wall texture
    for x in range(0, 2400, 40):
        cv2.line(img, (x, 0), (x, 600), (175, 170, 160), 1)  # paneling = sharp edges
    for _, rgb, x0, y0, x1, y1 in GARMENTS:
        patch = np.clip(np.array(rgb, float) + rng.normal(0, 5, (y1 - y0, x1 - x0, 3)), 0, 255)
        img[y0:y1, x0:x1] = patch.astype(np.uint8)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), tuple(int(c * 0.7) for c in rgb), 3)  # seams
    return img


def write_mp4(path: Path, frames_rgb: list[np.ndarray], fps: int = 30) -> Path:
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    return path


def display_matrix(angle_ccw: float) -> list[int]:
    """FFmpeg av_display_rotation_set: 16.16 fixed point rotation, 2.30 for w."""
    s, c = math.sin(math.radians(angle_ccw)), math.cos(math.radians(angle_ccw))
    fx = lambda x: int(round(x * 65536))
    return [fx(c), fx(-s), 0, fx(s), fx(c), 0, 0, 0, 1 << 30]


def write_hevc_mov(path: Path, frames_rgb, fps: int = 30, rotation_ccw: float = -90) -> Path:
    """iPhone-style HEVC .mov with a display matrix (portrait iPhone = -90).

    frames_rgb may be any iterable (e.g. a generator, for long videos).
    """
    av = pytest.importorskip("av")
    if "libx265" not in av.codecs_available:
        pytest.skip("PyAV build lacks libx265")
    it = iter(frames_rgb)
    first = next(it)
    h, w = first.shape[:2]
    with av.open(str(path), "w") as out:
        s = out.add_stream("libx265", rate=fps, options={"x265-params": "log-level=error"})
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.set_display_matrix(display_matrix(rotation_ccw))
        for f in itertools.chain([first], it):
            for p in s.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return path


def closet_video_frames(seconds: float = 6.0, fps: int = 30, blur_window=(2.0, 2.6)) -> list[np.ndarray]:
    """Camera pans left-to-right over the panorama; motion blur in blur_window."""
    pano = panorama()
    W, view = pano.shape[1], 640
    n = int(seconds * fps)
    out = []
    for i in range(n):
        t = i / fps
        x = int((W - view) * i / (n - 1))
        f = pano[60:540, x:x + view].copy()
        if blur_window[0] <= t < blur_window[1]:
            f = cv2.blur(f, (41, 1))
        out.append(f)
    return out


@pytest.fixture(scope="session")
def closet_video(tmp_path_factory) -> Path:
    return write_mp4(tmp_path_factory.mktemp("vid") / "closet.mp4", closet_video_frames())


class FakeDetector:
    """Finds garments by their known colors; labels from GARMENTS."""

    def __init__(self, tol: float = 30):
        self.tol = tol
        self.calls = 0

    def detect(self, images):
        self.calls += 1
        out = []
        for img in images:
            dets = []
            for label, rgb, *_ in GARMENTS:
                m = (np.abs(img.astype(int) - rgb).max(-1) < self.tol).astype(np.uint8)
                n, _, stats, _ = cv2.connectedComponentsWithStats(m)
                for j in range(1, n):
                    x, y, w, h, area = stats[j]
                    if area > 400:
                        dets.append(RawDetection((float(x), float(y), float(x + w), float(y + h)), 0.8, label))
            out.append(dets)
        return out


class FakeSegmenter:
    def segment(self, image, boxes):
        not_wall = np.abs(image.astype(int) - WALL).max(-1) > 35
        return [not_wall.copy() for _ in boxes]


class FakeEmbedder:
    """Color-histogram 'embedding'; texts map to fixed pseudo-random vectors."""

    dim = 64

    def embed_images(self, images):
        vecs = []
        for im in images:
            px = im.reshape(-1, 3)
            px = px[(px < 250).any(1)]  # ignore white padding from on_white()
            if len(px) == 0:
                px = im.reshape(-1, 3)
            h, _ = np.histogramdd(px // 64, bins=(4, 4, 4), range=((0, 4),) * 3)
            v = h.ravel().astype(np.float32)
            vecs.append(v / (np.linalg.norm(v) + 1e-9))
        return np.stack(vecs)

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dim).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def make_item(id, category, subtype, color_name, pattern="solid", formality=None, warmth=None):
    from sweep.colors import describe_name
    from sweep.taxonomy import subtype_attrs

    f, w = subtype_attrs(category, subtype)
    return {"id": id, "category": category, "subtype": subtype, "pattern": pattern,
            "color": describe_name(color_name),
            "formality": formality if formality is not None else f,
            "warmth": warmth if warmth is not None else w}
