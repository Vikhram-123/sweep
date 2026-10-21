"""Clothing detection: Detection records, box filtering, Grounding DINO wrapper."""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .taxonomy import DETECTION_PROMPT, category_distance, label_to_category, normalize_label

log = logging.getLogger(__name__)


@dataclass
class RawDetection:
    box: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    score: float
    label: str


@dataclass
class Detection:
    id: int
    frame: int
    box: tuple[float, float, float, float]
    score: float
    label: str
    category: str
    mask: np.ndarray | None = field(default=None, repr=False)   # bool HxW (frame size)
    cutout: np.ndarray | None = field(default=None, repr=False)  # RGBA crop
    sharpness: float = 0.0
    area_frac: float = 0.0
    touches_edge: bool = False
    sliver: bool = False
    mask_fill: float = 1.0


class Detector(Protocol):
    def detect(self, images: list[np.ndarray]) -> list[list[RawDetection]]: ...


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix between boxes a [n,4] and b [m,4]."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda r: (r[:, 2] - r[:, 0]) * (r[:, 3] - r[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def filter_detections(
    raw: list[RawDetection],
    image_shape: tuple[int, int],
    min_area_frac: float = 0.01,
    max_area_frac: float = 0.7,
    nms_iou: float = 0.7,
    contain_frac: float = 0.85,
) -> list[tuple[RawDetection, str]]:
    """Drop unmapped labels, tiny boxes, near-full-frame boxes; class-agnostic NMS.

    Grounding DINO often fires twice on one garment ("jacket" and "coat"), so
    NMS ignores labels and keeps the higher-scoring box. It also boxes parts of
    a garment (a hoodie's body without the hood), so a box mostly inside a
    compatible, at-least-as-confident box is dropped too. Returns (det, category).
    """
    h, w = image_shape[:2]
    kept = []
    for d in raw:
        cat = label_to_category(d.label)
        if cat is None:
            continue
        x0, y0, x1, y1 = d.box
        frac = max(0.0, x1 - x0) * max(0.0, y1 - y0) / (w * h)
        if frac < min_area_frac or frac > max_area_frac:
            continue
        kept.append((d, cat))
    kept.sort(key=lambda dc: dc[0].score, reverse=True)
    if len(kept) <= 1:
        return kept
    boxes = np.array([d.box for d, _ in kept], dtype=float)
    iou = box_iou(boxes, boxes)
    contained = box_containment(boxes)
    alive = np.ones(len(kept), dtype=bool)
    for i in range(len(kept)):
        if not alive[i]:
            continue
        alive[i + 1:] &= iou[i, i + 1:] < nms_iou
        for j in range(i + 1, len(kept)):  # j has a lower (or equal) score than i
            if alive[j] and contained[j, i] >= contain_frac and category_distance(kept[i][1], kept[j][1]) < 1:
                alive[j] = False
    return [kc for kc, a in zip(kept, alive) if a]


def box_containment(boxes: np.ndarray) -> np.ndarray:
    """c[i, j] = fraction of box i's area that lies inside box j."""
    x0 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y0 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x1 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y1 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area[:, None] + 1e-9)


def touches_edge(box, image_shape, margin: int = 3) -> bool:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = box
    return x0 <= margin or y0 <= margin or x1 >= w - margin or y1 >= h - margin


def is_sliver(box, image_shape, max_frac: float = 0.35, margin: int = 3) -> bool:
    """Cut by the left/right frame edge AND narrow.

    Closets are filmed with a sideways pan, so garments entering or leaving
    the shot show up as narrow strips at the left/right edge. Boxes touching
    the top/bottom are usually whole items (shoes on the floor, a long coat).
    """
    w = image_shape[1]
    x0, _, x1, _ = box
    return (x0 <= margin or x1 >= w - margin) and (x1 - x0) < max_frac * w


class GroundingDinoDetector:
    """Zero-shot clothing detector (IDEA-Research/grounding-dino-tiny)."""

    def __init__(self, model_id: str = "IDEA-Research/grounding-dino-tiny", device: str | None = None,
                 box_threshold: float = 0.3, text_threshold: float = 0.25, prompt: str = DETECTION_PROMPT):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from .device import align_weights, get_device

        self.torch = torch
        self.device = device or get_device()
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = align_weights(AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device),
                                   self.device).eval()
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.prompt = prompt
        # transformers renamed box_threshold -> threshold in post-processing.
        params = inspect.signature(self.processor.post_process_grounded_object_detection).parameters
        self._thr_kw = "threshold" if "threshold" in params else "box_threshold"

    def detect(self, images: list[np.ndarray], batch_size: int | None = None) -> list[list[RawDetection]]:
        from PIL import Image

        # Batching doesn't speed up MPS and multiplies peak memory (~2 GB per
        # 1024px frame), so only batch on CUDA.
        batch_size = batch_size or (4 if self.device == "cuda" else 1)
        out: list[list[RawDetection]] = []
        for i in range(0, len(images), batch_size):
            batch = [Image.fromarray(im) for im in images[i:i + batch_size]]
            inputs = self.processor(images=batch, text=[self.prompt] * len(batch),
                                    return_tensors="pt", padding=True).to(self.device)
            with self.torch.inference_mode():
                outputs = self.model(**inputs)
            results = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, text_threshold=self.text_threshold,
                target_sizes=[im.size[::-1] for im in batch], **{self._thr_kw: self.box_threshold},
            )
            for r in results:
                labels = r["text_labels"] if "text_labels" in r else r["labels"]
                out.append([
                    RawDetection(tuple(float(v) for v in box.tolist()), float(score), normalize_label(str(lab)))
                    for box, score, lab in zip(r["boxes"].cpu(), r["scores"].cpu(), labels)
                ])
        return out
