import numpy as np
import pytest

from sweep.colors import describe_name, dominant_colors, lab_to_srgb, nearest_named, srgb_to_lab
from sweep.tagging import tag_color


@pytest.mark.parametrize("rgb,lab", [
    ((255, 255, 255), (100.0, 0.0, 0.0)),
    ((0, 0, 0), (0.0, 0.0, 0.0)),
    ((255, 0, 0), (53.24, 80.09, 67.20)),
    ((0, 0, 255), (32.30, 79.19, -107.86)),
])
def test_srgb_to_lab_reference_values(rgb, lab):
    assert np.allclose(srgb_to_lab(np.array(rgb)), lab, atol=0.05)


def test_lab_roundtrip():
    rgb = np.random.default_rng(0).integers(0, 256, (200, 3))
    assert np.abs(lab_to_srgb(srgb_to_lab(rgb)) - rgb).max() <= 1


def garment_cutout(rgb, bg=(250, 250, 250), size=(120, 100), noise=8, seed=0):
    """RGBA: garment pixels inside an ellipse mask, bright background outside (alpha 0)."""
    rng = np.random.default_rng(seed)
    h, w = size
    yy, xx = np.mgrid[:h, :w]
    mask = ((yy - h / 2) / (h * 0.3)) ** 2 + ((xx - w / 2) / (w * 0.3)) ** 2 < 1  # ~28% of pixels
    img = np.empty((h, w, 3), float)
    img[:] = bg
    shade = rng.normal(0, noise, (h, w, 3)) + np.linspace(-15, 15, h)[:, None, None]  # folds/lighting
    img[mask] = np.array(rgb) + shade[mask]
    return np.dstack([np.clip(img, 0, 255).astype(np.uint8), (mask * 255).astype(np.uint8)])


@pytest.mark.parametrize("rgb,name,neutral,bold", [
    ((195, 35, 45), "red", False, True),
    ((32, 40, 75), "navy", True, False),
    ((22, 22, 24), "black", True, False),
    ((72, 96, 132), "denim", True, False),
    ((190, 172, 222), "lavender", False, False),
    ((215, 196, 165), "beige", True, False),
])
def test_dominant_color_ignores_masked_background(rgb, name, neutral, bold):
    # Background is most of the image but alpha=0, so it must not win.
    c = dominant_colors(garment_cutout(rgb))[0]
    assert c["name"] == name
    assert c["neutral"] is neutral and c["bold"] is bold
    assert c["share"] > 0.4


def test_secondary_color_for_two_tone_garment():
    rgba = garment_cutout((245, 245, 242))
    for k in range(4):  # navy stripes: 4 of every 8 rows
        rgba[k::8, :, :3] = (30, 38, 72)
    c = tag_color(rgba)
    assert {c["name"], c["secondary"]} == {"white", "navy"}


def test_nearest_named_and_describe_name():
    assert nearest_named(srgb_to_lab(np.array((240, 120, 30)))) == "orange"
    rec = describe_name("olive")
    assert rec["neutral"] and rec["hex"].startswith("#") and len(rec["lab"]) == 3
