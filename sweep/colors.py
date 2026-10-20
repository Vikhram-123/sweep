"""Color utilities in pure numpy: sRGB<->CIELAB, masked k-means, named colors.

No OpenCV/torch here so outfits.py and the app can use it cheaply.
"""

from __future__ import annotations

import colorsys

import numpy as np

# name -> (sRGB, is_neutral). Neutrals pair with anything in outfits.py.
NAMED_COLORS: dict[str, tuple[tuple[int, int, int], bool]] = {
    "black": ((20, 20, 22), True),
    "charcoal": ((60, 62, 66), True),
    "gray": ((128, 128, 128), True),
    "light gray": ((195, 195, 195), True),
    "white": ((245, 245, 242), True),
    "cream": ((240, 232, 210), True),
    "beige": ((214, 196, 166), True),
    "khaki": ((189, 174, 130), True),
    "tan": ((176, 141, 100), True),
    "brown": ((105, 70, 45), True),
    "dark brown": ((62, 42, 30), True),
    "navy": ((30, 38, 72), True),
    "denim": ((70, 95, 135), True),
    "olive": ((95, 100, 55), True),
    "burgundy": ((120, 25, 45), False),
    "red": ((200, 30, 40), False),
    "coral": ((240, 120, 100), False),
    "orange": ((235, 120, 30), False),
    "mustard": ((205, 165, 40), False),
    "yellow": ((245, 220, 60), False),
    "green": ((40, 140, 60), False),
    "forest green": ((35, 75, 45), False),
    "mint": ((170, 225, 195), False),
    "teal": ((20, 125, 130), False),
    "light blue": ((160, 200, 235), False),
    "blue": ((40, 90, 200), False),
    "purple": ((110, 50, 140), False),
    "lavender": ((190, 170, 220), False),
    "pink": ((240, 160, 185), False),
    "hot pink": ((225, 50, 135), False),
}

# Chroma (LAB) above which a non-neutral color counts as "bold".
BOLD_CHROMA = 35.0


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB uint8/float (..., 3) in [0,255] -> CIELAB (D65), same leading shape."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.where(xyz > eps, np.cbrt(xyz), (kappa * xyz + 16) / 116)
    L = 116 * f[..., 1] - 16
    a = 500 * (f[..., 0] - f[..., 1])
    b = 200 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


_NAMES = list(NAMED_COLORS)
_PALETTE_LAB = srgb_to_lab(np.array([NAMED_COLORS[n][0] for n in _NAMES]))


def delta_e(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE76 color difference (Euclidean in LAB)."""
    return np.linalg.norm(np.asarray(lab1, float) - np.asarray(lab2, float), axis=-1)


def chroma(lab) -> float:
    return float(np.hypot(lab[1], lab[2]))


def hue_deg(rgb) -> float:
    r, g, b = (np.asarray(rgb, float) / 255.0).tolist()
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360.0


def nearest_named(lab) -> str:
    return _NAMES[int(np.argmin(delta_e(_PALETTE_LAB, lab)))]


def kmeans(x: np.ndarray, k: int, iters: int = 25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Plain k-means (k-means++ init). Returns (centers [k,d], labels [n])."""
    rng = np.random.default_rng(seed)
    n = len(x)
    k = min(k, n)
    centers = [x[rng.integers(n)]]
    for _ in range(1, k):
        d2 = np.min(((x[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() == 0:
            break
        centers.append(x[rng.choice(n, p=d2 / d2.sum())])
    centers = np.array(centers)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        labels = ((x[:, None, :] - centers[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, labels


def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, float)
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    eps, kappa = 216 / 24389, 24389 / 27
    f = np.stack([fx, fy, fz], -1)
    xyz = np.where(f ** 3 > eps, f ** 3, (116 * f - 16) / kappa) * np.array([0.95047, 1.0, 1.08883])
    m_inv = np.array([[3.2404542, -1.5371385, -0.4985314],
                      [-0.9692660, 1.8760108, 0.0415560],
                      [0.0556434, -0.2040259, 1.0572252]])
    c = np.clip(xyz @ m_inv.T, 0, 1)
    c = np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)
    return np.clip(np.round(c * 255), 0, 255).astype(int)


def describe_lab(lab) -> dict:
    """Full color record for a LAB value: name, hex, neutral/bold flags."""
    lab = np.asarray(lab, float)
    rgb = lab_to_srgb(lab)
    name = nearest_named(lab)
    neutral = NAMED_COLORS[name][1]
    return {
        "name": name,
        "rgb": [int(v) for v in rgb],
        "hex": "#%02x%02x%02x" % tuple(int(v) for v in rgb),
        "lab": [round(float(v), 2) for v in lab],
        "neutral": bool(neutral),
        "bold": bool(not neutral and chroma(lab) >= BOLD_CHROMA),
    }


def describe_name(name: str) -> dict:
    """Color record for a palette name (used when the user edits a tag)."""
    rgb, _ = NAMED_COLORS[name]
    rec = describe_lab(srgb_to_lab(np.array(rgb)))
    rec["name"] = name
    rec["neutral"] = NAMED_COLORS[name][1]
    rec["bold"] = bool(not rec["neutral"] and chroma(rec["lab"]) >= BOLD_CHROMA)
    return rec


def mean_lab(rgba: np.ndarray) -> np.ndarray:
    """Mean LAB of masked pixels: a stable color signature for dedupe.

    Dominant k-means colors flip between views of multicolor items (purple
    checks vs white sole), the mean does not.
    """
    px = rgba[..., :3][rgba[..., 3] > 127] if rgba.shape[-1] == 4 else rgba.reshape(-1, 3)
    if len(px) == 0:
        px = rgba[..., :3].reshape(-1, 3)
    return srgb_to_lab(px[:: max(1, len(px) // 4000)]).mean(0)


def dominant_colors(rgba: np.ndarray, k: int = 3, max_pixels: int = 4000, seed: int = 0) -> list[dict]:
    """Cluster masked pixels (alpha > 127) in LAB; return colors by share, largest first.

    Each record is describe_lab(...) plus "share" (fraction of masked pixels).
    """
    px = rgba[..., :3][rgba[..., 3] > 127] if rgba.shape[-1] == 4 else rgba.reshape(-1, 3)
    if len(px) == 0:
        px = rgba[..., :3].reshape(-1, 3)
    rng = np.random.default_rng(seed)
    if len(px) > max_pixels:
        px = px[rng.choice(len(px), max_pixels, replace=False)]
    lab = srgb_to_lab(px)
    centers, labels = kmeans(lab, k, seed=seed)
    shares = np.bincount(labels, minlength=len(centers)) / len(labels)
    out = []
    for j in np.argsort(-shares):
        if shares[j] == 0:
            continue
        rec = describe_lab(centers[j])
        rec["share"] = round(float(shares[j]), 3)
        out.append(rec)
    return out
