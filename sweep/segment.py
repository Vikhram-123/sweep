"""Box-prompted segmentation (SAM) and RGBA cutout helpers."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class Segmenter(Protocol):
    def segment(self, image: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> list[np.ndarray]: ...


def clean_mask(mask: np.ndarray, box) -> np.ndarray:
    """Clip to the box, keep the largest connected component, fill small holes."""
    h, w = mask.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    clipped = np.zeros_like(mask, dtype=bool)
    clipped[max(0, y0):min(h, y1), max(0, x0):min(w, x1)] = mask[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    m = clipped.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n > 2:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (labels == largest).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m.astype(bool)


def make_cutout(image: np.ndarray, mask: np.ndarray, box, pad: int = 4) -> np.ndarray:
    """RGBA crop of the box (plus padding) with alpha = mask."""
    h, w = mask.shape
    x0, y0, x1, y1 = box
    x0, y0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
    x1, y1 = min(w, int(np.ceil(x1)) + pad), min(h, int(np.ceil(y1)) + pad)
    rgb = image[y0:y1, x0:x1]
    alpha = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
    return np.dstack([rgb, alpha])


def on_white(rgba: np.ndarray) -> np.ndarray:
    """Composite an RGBA cutout onto white (FashionCLIP saw product shots on white)."""
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return (rgba[..., :3] * a + 255 * (1 - a)).astype(np.uint8)


class SamSegmenter:
    def __init__(self, model_id: str = "facebook/sam-vit-base", device: str | None = None):
        import torch
        from transformers import SamModel, SamProcessor

        from .device import align_weights, get_device

        self.torch = torch
        self.device = device or get_device()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = align_weights(SamModel.from_pretrained(model_id).to(self.device), self.device).eval()

    def segment(self, image: np.ndarray, boxes) -> list[np.ndarray]:
        if not boxes:
            return []
        from PIL import Image

        inputs = self.processor(Image.fromarray(image), input_boxes=[[list(b) for b in boxes]],
                                return_tensors="pt")
        # The processor emits float64 boxes; MPS has no float64.
        model_inputs = {k: v.to(self.device, dtype=self.torch.float32) if v.is_floating_point() else v.to(self.device)
                        for k, v in inputs.items() if k not in ("original_sizes", "reshaped_input_sizes")}
        with self.torch.inference_mode():
            outputs = self.model(**model_inputs, multimask_output=False)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )[0]  # [n_boxes, 1, H, W]
        return [m[0].numpy().astype(bool) for m in masks]
