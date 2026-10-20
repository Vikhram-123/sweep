"""Build an iPhone-style demo closet video from public product photos.

    python scripts/make_demo_video.py            # -> scripts/demo_data/IMG_DEMO.MOV

Downloads ~12 labelled garment photos from the Hugging Face dataset
benitomartin/fashion-product-images-small-900x1200, hangs them on a
synthetic closet wall (shoes on a shelf), and films a slow handheld pan
with exposure drift and two whip-pan blurs. The output is a portrait HEVC
.MOV with the same rotation metadata an iPhone writes, plus truth.json
with the dataset's labels so you can compare against what Sweep tags.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

DATASET = "benitomartin/fashion-product-images-small-900x1200"
WANT = [("Tshirts", "Red"), ("Tshirts", None), ("Shirts", None), ("Sweatshirts", None), ("Jackets", None),
        ("Jeans", None), ("Trousers", None), ("Shorts", None), ("Dresses", None), ("Casual Shoes", None),
        ("Formal Shoes", None), ("Tops", None)]


def curl(url: str, dest: Path) -> None:
    # curl rather than urllib: python.org builds often lack root certificates.
    subprocess.run(["curl", "-sSfL", "--max-time", "60", url, "-o", str(dest)], check=True)


def fetch(data: Path) -> list[dict]:
    picked_path = data / "picked.json"
    if picked_path.exists():
        return json.loads(picked_path.read_text())
    (data / "imgs").mkdir(parents=True, exist_ok=True)
    rows = []
    for offset in range(0, 600, 100):
        page = data / f"rows_{offset}.json"
        curl(f"https://datasets-server.huggingface.co/rows?dataset={DATASET}&config=default&split=train"
             f"&offset={offset}&length=100", page)
        rows += [r["row"] for r in json.loads(page.read_text()).get("rows", [])]
        page.unlink()
    picked, used = [], set()
    for kind, colour in WANT:
        cands = [r for r in rows if r["articleType"] == kind and r["id"] not in used
                 and (colour is None or r["baseColour"] == colour)]
        cands.sort(key=lambda r: r["baseColour"] in {p["baseColour"] for p in picked})  # prefer new colours
        if cands:
            picked.append(cands[0])
            used.add(cands[0]["id"])
    for r in picked:
        curl(r["image"]["src"], data / "imgs" / f"{r['id']}.jpg")
        r.pop("image")
    picked_path.write_text(json.dumps(picked, indent=1))
    return picked


def cut(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alpha = everything not flood-filled from the corners (studio backdrop of any shade)."""
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for x, y in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if mask[y + 1, x + 1] == 0:
            cv2.floodFill(img.copy(), mask, (x, y), 0, (10, 10, 10), (10, 10, 10), flags)
    alpha = cv2.erode(np.where(mask[1:-1, 1:-1] == 255, 0, 255).astype(np.uint8), np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(alpha)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def paste(wall, rgb, a, x, y, shadow=True):
    h, w = rgb.shape[:2]
    a = a[..., None] / 255.0
    if shadow:
        s = cv2.GaussianBlur(a[..., 0], (41, 41), 15)[..., None]
        region = wall[y + 12:y + 12 + h, x + 12:x + 12 + w]
        region *= 1 - 0.35 * s[:region.shape[0], :region.shape[1]]
    wall[y:y + h, x:x + w] = wall[y:y + h, x:x + w] * (1 - a) + rgb * a


def build_wall(data: Path, picked: list[dict], rng) -> tuple[np.ndarray, list[dict]]:
    H, W = 1760, 5200
    grain = cv2.resize(rng.normal(0, 1, (H // 6, W // 60)).astype(np.float32), (W, H))
    wall = np.empty((H, W, 3), np.float32)
    wall[:] = (182, 160, 130)
    wall += grain[..., None] * 14 + rng.normal(0, 4, (H, W, 1))
    for x in range(0, W, 260):
        cv2.line(wall, (x, 0), (x, H), (150, 128, 100), 3)
    cv2.rectangle(wall, (0, 95), (W, 115), (120, 120, 125), -1)      # hanging rod
    cv2.rectangle(wall, (0, 1470), (W, 1500), (120, 92, 66), -1)     # shoe shelf
    load = lambda r: cut(cv2.cvtColor(cv2.imread(str(data / "imgs" / f"{r['id']}.jpg")), cv2.COLOR_BGR2RGB))
    truth, x = [], 150
    for r in [r for r in picked if "Shoes" not in r["articleType"]]:
        rgb, a = load(r)
        th = 520 if r["articleType"] == "Shorts" else int(rng.uniform(700, 950))
        s = th / rgb.shape[0]
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), th), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (rgb.shape[1], th))
        if x + rgb.shape[1] > W - 50:
            break
        cv2.line(wall, (x + rgb.shape[1] // 2, 100), (x + rgb.shape[1] // 2, 150), (60, 60, 60), 4)
        paste(wall, rgb, a, x, 140)
        truth.append({"id": r["id"], "type": r["articleType"], "colour": r["baseColour"],
                      "name": r["productDisplayName"], "x": [x, x + rgb.shape[1]]})
        x += rgb.shape[1] + int(rng.uniform(-60, 40))  # slight overlap, like a real rail
    for i, r in enumerate(r for r in picked if "Shoes" in r["articleType"]):
        rgb, a = load(r)
        s = 260 / rgb.shape[0]
        rgb = cv2.resize(rgb, (int(rgb.shape[1] * s), 260), interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, (rgb.shape[1], 260))
        sx = 900 + i * 1700
        paste(wall, rgb, a, sx, 1470 - 260, shadow=False)
        truth.append({"id": r["id"], "type": r["articleType"], "colour": r["baseColour"],
                      "name": r["productDisplayName"], "x": [sx, sx + rgb.shape[1]]})
    return np.clip(wall, 0, 255).astype(np.uint8), truth


def film(wall: np.ndarray, out: Path, rng, secs: int = 14, fps: int = 30) -> None:
    import av

    H, W = wall.shape[:2]
    view_w = round(H * 9 / 16)
    n = secs * fps
    with av.open(str(out), "w") as container:
        st = container.add_stream("libx265", rate=fps, options={"x265-params": "log-level=error"})
        st.width, st.height, st.pix_fmt = 1920, 1080, "yuv420p"  # landscape sensor frames...
        s, c = math.sin(math.radians(-90)), math.cos(math.radians(-90))
        fx = lambda v: int(round(v * 65536))
        st.set_display_matrix([fx(c), fx(-s), 0, fx(s), fx(c), 0, 0, 0, 1 << 30])  # ...shown portrait
        for i in range(n):
            t = i / fps
            cx = int((W - view_w) * (0.5 - 0.5 * math.cos(math.pi * i / (n - 1))))  # ease in/out pan
            jy = max(0, int(8 * math.sin(t * 3.1)))
            f = cv2.resize(wall[jy:jy + H - 8, cx:cx + view_w], (1080, 1920)).astype(np.float32)
            f *= 1 + 0.08 * math.sin(t * 0.9)  # auto-exposure drift
            if 4.0 <= t < 4.4 or 9.0 <= t < 9.3:  # whip-pan motion blur
                f = cv2.blur(f, (61, 1))
            f = np.clip(f + rng.normal(0, 2.5, f.shape), 0, 255).astype(np.uint8)
            frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.rot90(f, k=1)), format="rgb24")
            for p in st.encode(frame):
                container.mux(p)
        for p in st.encode():
            container.mux(p)


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "demo_data"
    data.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    picked = fetch(data)
    wall, truth = build_wall(data, picked, rng)
    cv2.imwrite(str(data / "wall.jpg"), cv2.cvtColor(cv2.resize(wall, (wall.shape[1] // 4, wall.shape[0] // 4)),
                                                     cv2.COLOR_RGB2BGR))
    film(wall, data / "IMG_DEMO.MOV", rng)
    (data / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {data / 'IMG_DEMO.MOV'} ({len(truth)} garments; labels in truth.json)")


if __name__ == "__main__":
    main()
