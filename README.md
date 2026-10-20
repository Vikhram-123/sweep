# Sweep

Film your closet, get a tagged digital wardrobe, and get outfit suggestions from your iPhone.

```
video ─► frames ─► detect ─► segment ─► embed ─► dedupe ─► best crop ─► tag ─► closet/
         OpenCV    Grounding  SAM       Fashion-  cluster   sharpness    color, type,
         3 fps,    DINO       (box      CLIP      views of  × area ×     pattern,
         sharpest  (tiny)     prompts)            the same  score        formality,
                                                  garment                warmth
```

The models run on your Mac (Apple Silicon GPU via MPS). You use the app from your phone's browser over Wi-Fi.

## Setup (Mac, once)

```bash
cd Sweep
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

* The first scan downloads ~1.5 GB of models into `~/.cache/huggingface`. After that, everything runs offline.
* If you installed Python from python.org and see `CERTIFICATE_VERIFY_FAILED` anywhere, run
  `/Applications/Python\ 3.*/Install\ Certificates.command` once.

## Use it from your iPhone

```bash
./run.sh
```

This prints the exact URL (your Mac's local IP, e.g. `http://192.168.1.23:8501`) and a QR code. Point the iPhone camera at the QR code. The phone has to be on the same Wi-Fi as the Mac.

* If macOS asks whether Python may accept incoming connections, click **Allow**.
* **Campus / office Wi-Fi** (e.g. eduroam, DukeBlue) usually blocks devices from reaching each other. Turn on
  **Personal Hotspot** on the iPhone, join it from the Mac, and run `./run.sh` again. It will print the
  hotspot address.
* `run.sh` raises the upload limit to 1 GB and keeps the Mac awake (`caffeinate`) while the server runs.
* Tip: Safari → Share → **Add to Home Screen** makes it open like an app.

### Tabs

* **Scan**: upload a video from Photos and tap *Scan video*. The scan runs on the Mac in the background, so you can lock your phone and check back. New items are **added** to the closet; items it already has are skipped.
* **Closet**: tap *Edit* to fix category, type, color, pattern, dressiness or warmth, or to delete an item. Turn on *Select to merge or delete* to merge duplicates (the first one you select keeps its photo).
* **Outfits**: pick an occasion and weather, optionally an item that must be included. 👍/👎 votes go to `closet/feedback.json` and nudge future suggestions toward or away from those item combinations.

### Filming tips

Pan slowly, left to right, about one hanger-width per second, and keep garments as unobstructed as you can. Several short videos (one per section) work better than one long one. Each one is appended to the closet.

## Command line

```bash
.venv/bin/python -m sweep.scan IMG_1234.MOV --out closet/
```

| flag | default | what it does |
|---|---|---|
| `--threshold` | 0.25 | dedupe distance; **raise** if one garment shows up as several items, **lower** if different garments get merged |
| `--min-detections` | 2 | drop "items" seen in fewer frames (one-off false detections) |
| `--box-threshold` | 0.3 | detector confidence; lower finds more, with more junk |
| `--fps` / `--keep` | 3 / 0.7 | sampling rate, and the minimum fraction of frames kept after blur filtering |
| `--device` | auto | force `mps`, `cuda` or `cpu` |
| `--no-debug` | | skip writing debug images |

### Seeing where things go wrong

Every scan writes `closet/debug/<scan_id>/`:

* `frames/`: kept frames; each file name includes the timestamp and sharpness score
* `detections/`: each frame with boxes, labels, scores and masks drawn on it
* `masks/`: every individual cutout
* `clusters/`: one contact sheet per cluster (= candidate item); the green border marks the crop chosen as the item photo. File names ending in `_dropped-few` / `_dropped-sliver` were filtered out.
* `summary.json`: counts and the config used

Typical fixes: a garment split across several `clusters/` sheets → raise `--threshold`. Two garments on one sheet → lower it. Missing garment with no box in `detections/` → lower `--box-threshold`.

## Colab (GPU backup)

Open `sweep_colab.ipynb` in Google Colab and follow the cells. You'll need the code as a zip:

```bash
zip -r sweep_code.zip sweep requirements.txt
zip -r closet.zip closet          # optional: so the Colab scan appends to your closet
```

The notebook gives you `closet.zip` back. Run `unzip -o closet.zip` in the Sweep folder to replace `closet/`.

## Try it without filming

```bash
.venv/bin/python scripts/make_demo_video.py     # downloads 12 product photos, renders an iPhone-style pan
.venv/bin/python -m sweep.scan scripts/demo_data/IMG_DEMO.MOV --out scripts/demo_data/closet
SWEEP_CLOSET=scripts/demo_data/closet ./run.sh
```

`scripts/demo_data/truth.json` has the dataset's real labels to compare against.

## How it works

* **Frames** (`sweep/frames.py`): samples by timestamp; OpenCV applies iPhone rotation metadata, with PyAV as a fallback decoder for HEVC. Blur is judged against neighboring frames (a plain sweater is less "sharp" than a busy shelf even when in focus), so only frames blurrier than their neighbors are dropped.
* **Detect** (`sweep/detect.py`): Grounding DINO with a clothing prompt; drops tiny or near-full-frame boxes, duplicate boxes, and boxes that cover only part of another garment.
* **Dedupe** (`sweep/dedupe.py`): agglomerative clustering on `cosine(FashionCLIP) + color + category penalty`, with a hard penalty for two detections in the same frame (they must be different garments). Detector labels are noisy (a coat is "dress" in one frame, "jacket" in the next), so related categories get only a small penalty. A final pass merges look-alike clusters that never share a frame, and drops clusters made only of thin strips at the frame edge.
* **Tag** (`sweep/tagging.py`, `sweep/colors.py`): the category is a vote between the detector and FashionCLIP. Color comes from k-means in LAB over masked pixels, mapped to 30 named colors with neutral/bold flags. Type and pattern come from FashionCLIP zero-shot; formality and warmth from a lookup table (`sweep/taxonomy.py`).
* **Outfits** (`sweep/outfits.py`, pure numpy): color harmony (neutrals go with anything, one bold color is a statement, clashing bold hues are penalized), one pattern max, light/dark contrast, formality vs. occasion, warmth vs. weather (outerwear adds warmth), plus feedback. Picks are diversified so the top N don't all share the same items.

### Memory note (Apple Silicon)

By default, PyTorch lets the GPU allocate *more* than physical RAM on Apple Silicon, so a memory spike swaps the whole Mac to a crawl. `sweep/device.py` caps MPS at the system's recommended working set, and the detector runs one frame at a time on MPS. On a 16 GB M1 Pro, a 14 s video scans in about 75 s with no swapping.

## Tests

```bash
.venv/bin/python -m pytest
```

The tests use synthetic videos (including an HEVC `.MOV` with iPhone rotation metadata), synthetic garments, and a fake detector/segmenter/embedder, so they need no model downloads.

## Known limitations

* Shiny black shoes can come out "gray" (specular highlights dominate the pixels). Fix it in the Edit dialog.
* Garments that are only partly visible in every frame get tagged from that partial view (e.g. cropped jeans → "shorts").
* Only the item photo's dominant color and pattern are tagged; multicolor items also get a `secondary` color.
