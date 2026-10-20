"""Frame sampling from closet videos.

Samples ~N fps by timestamp, keeps the sharpest fraction (variance of the
Laplacian), and resizes to a max side length. Decoding goes through OpenCV
with auto-orientation enabled (iPhone videos store portrait rotation as
metadata); if OpenCV can't decode the file (e.g. an HEVC build without the
codec) we fall back to PyAV and apply the display-matrix rotation ourselves.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    index: int          # position among sampled frames (stable id within a scan)
    time_s: float       # timestamp in the source video
    image: np.ndarray   # RGB uint8, HxWx3
    sharpness: float = 0.0


def sharpness(image: np.ndarray, max_side: int = 512) -> float:
    """Variance of the Laplacian on a downscaled grayscale copy.

    Downscaling first makes scores comparable across resolutions and fast.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def relative_sharpness(scores: np.ndarray, window: int = 7) -> np.ndarray:
    """Sharpness divided by the rolling median of its neighbours.

    Raw Laplacian variance depends on content: a frame filled by a plain
    sweater scores lower than one showing a busy shelf even when both are in
    focus. Comparing each frame with nearby frames (similar content) isolates
    motion blur, so whole stretches of the video aren't dropped.
    """
    half = window // 2
    padded = np.pad(scores, half, mode="edge")
    med = np.array([np.median(padded[i:i + window]) for i in range(len(scores))])
    return scores / (med + 1e-9)


def resize_max(image: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def rotate_image(image: np.ndarray, degrees_cw: int) -> np.ndarray:
    """Rotate by a multiple of 90 degrees clockwise."""
    k = (round(degrees_cw / 90)) % 4
    return np.ascontiguousarray(np.rot90(image, k=-k)) if k else image


def _iter_opencv(path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open {path}")
    try:
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        orientation = cap.get(cv2.CAP_PROP_ORIENTATION_META)
        if orientation:
            log.info("video orientation metadata: %s deg (auto-applied)", orientation)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = 1.0 / fps
        next_t = 0.0
        i = 0
        while True:
            if not cap.grab():
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if t <= 0 and i > 0:  # some containers don't report POS_MSEC
                t = i / src_fps
            i += 1
            if t + 1e-6 < next_t:
                continue
            ok, bgr = cap.retrieve()
            if not ok:
                continue
            next_t += step * max(1, math.floor((t - next_t) / step) + 1)
            yield t, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _pyav_rotation(stream, frame) -> int:
    """Clockwise rotation (degrees) needed to display a PyAV frame upright."""
    # PyAV >= 13 exposes the display-matrix angle (counter-clockwise, ffmpeg
    # convention) as frame.rotation; older containers use a 'rotate' tag.
    ccw = getattr(frame, "rotation", None)
    if ccw:
        return int(-ccw) % 360
    tag = stream.metadata.get("rotate") if stream.metadata else None
    return int(tag) % 360 if tag else 0


def _iter_pyav(path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        step = 1.0 / fps
        next_t = 0.0
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            t = float(frame.time)
            if t + 1e-6 < next_t:
                continue
            next_t += step * max(1, math.floor((t - next_t) / step) + 1)
            rgb = frame.to_ndarray(format="rgb24")
            yield t, rotate_image(rgb, _pyav_rotation(stream, frame))


def iter_video(path: str | Path, fps: float = 3.0, backend: str = "auto") -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp, RGB image) pairs sampled at ~fps."""
    path = Path(path)
    if backend in ("auto", "opencv"):
        try:
            got_any = False
            for item in _iter_opencv(path, fps):
                got_any = True
                yield item
            if got_any or backend == "opencv":
                return
            log.warning("OpenCV decoded no frames from %s; falling back to PyAV", path)
        except IOError:
            if backend == "opencv":
                raise
            log.warning("OpenCV could not open %s; falling back to PyAV", path)
    yield from _iter_pyav(path, fps)


def extract_frames(
    path: str | Path,
    fps: float = 3.0,
    keep_fraction: float = 0.7,
    max_side: int = 1024,
    backend: str = "auto",
    min_relative: float = 0.85,
) -> list[Frame]:
    """Sample frames, drop up to (1 - keep_fraction) of the blurriest, resize.

    Blur is judged relative to neighbouring frames, and a frame at least
    `min_relative` as sharp as its neighbours is never dropped, so a steady
    video keeps everything while a shaky one keeps the sharpest keep_fraction.

    Returned frames are in time order and re-indexed 0..n-1.
    """
    sampled: list[Frame] = []
    for t, rgb in iter_video(path, fps=fps, backend=backend):
        img = resize_max(rgb, max_side)
        sampled.append(Frame(index=len(sampled), time_s=t, image=img, sharpness=sharpness(img)))
    if not sampled:
        raise ValueError(f"No frames decoded from {path}")

    n_keep = max(1, round(len(sampled) * keep_fraction))
    rel = relative_sharpness(np.array([f.sharpness for f in sampled]), window=max(3, round(2 * fps) | 1))
    order = np.argsort(-rel, kind="stable")
    chosen = set(order[:n_keep].tolist()) | set(np.flatnonzero(rel >= min_relative).tolist())
    keep = [sampled[i] for i in sorted(chosen)]
    for i, f in enumerate(keep):
        f.index = i
    log.info("sampled %d frames, kept %d sharpest", len(sampled), len(keep))
    return keep
