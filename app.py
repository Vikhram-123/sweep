"""Sweep: phone-first Streamlit app. Run with ./run.sh (serves on your LAN).

Only lightweight modules are imported at the top; the vision pipeline
(torch, transformers, OpenCV) is imported inside the scan thread the first
time you scan, so browsing the closet and outfits stays fast.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import streamlit as st

from sweep.closet import Closet
from sweep.colors import NAMED_COLORS, describe_name
from sweep.compat import load_default as load_compat_model
from sweep.outfits import OCCASIONS, WEATHER, generate_outfits
from sweep.taxonomy import CATEGORIES, PATTERNS, SUBTYPES, subtype_attrs

CLOSET = Closet(os.environ.get("SWEEP_CLOSET", Path(__file__).parent / "closet"))
CAT_ICON = {"top": "👕", "bottom": "👖", "outerwear": "🧥", "dress": "👗", "shoes": "👟"}

st.set_page_config(page_title="Sweep", page_icon="🧺", layout="centered", initial_sidebar_state="collapsed")
st.markdown(
    """
<style>
.block-container {padding: 2.2rem .75rem 4rem; max-width: 760px;}
/* Keep grids as grids on a phone instead of stacking columns vertically. */
div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .45rem !important;}
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {min-width: 0 !important; flex: 1 1 0 !important; width: auto !important;}
.stButton button, .stDownloadButton button {min-height: 2.6rem; width: 100%;}
[data-testid="stImage"] img {border-radius: 12px; background: #fff;}
.item-cap {font-size: .78rem; line-height: 1.15; margin: -.2rem 0 .1rem; color: var(--text-color); opacity: .8;
           white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.reason {font-size: .9rem; margin: 0;}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Background scan job (one at a time, shared by all browser sessions)
# ---------------------------------------------------------------------------
@dataclass
class ScanJob:
    video: str
    progress: float = 0.0
    message: str = "Starting"
    started: float = field(default_factory=time.time)
    done: bool = False
    result: object = None
    error: str | None = None
    acknowledged: bool = False

    def update(self, frac: float, msg: str) -> None:
        self.progress, self.message = max(0.0, min(1.0, frac)), msg


@st.cache_resource
def job_slot() -> dict:
    return {"job": None}


def start_scan(video_path: Path, cfg_kwargs: dict) -> ScanJob:
    job = ScanJob(video=video_path.name)

    def run():
        try:
            from sweep.scan import ScanConfig, run_scan  # heavy import, first scan only

            job.result = run_scan(video_path, CLOSET.root, ScanConfig(**cfg_kwargs), progress=job.update)
        except Exception:
            job.error = traceback.format_exc()
        finally:
            job.done = True

    threading.Thread(target=run, daemon=True, name="sweep-scan").start()
    job_slot()["job"] = job
    return job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_items() -> list[dict]:
    return CLOSET.load_items()


def item_label(it: dict) -> str:
    return f"{CAT_ICON.get(it['category'], '')} {it['color']['name']} {it['subtype']}"


def thumb(it: dict, caption: bool = True) -> None:
    p = CLOSET.image_path(it, thumb=True)
    if p.exists():
        st.image(str(p), width="stretch")
    else:
        st.markdown("🖼️ missing image")
    if caption:
        st.markdown(f"<p class='item-cap'>{it['color']['name']} {it['subtype']}</p>", unsafe_allow_html=True)


def grid(items: list[dict], cols: int = 3):
    for i in range(0, len(items), cols):
        row = st.columns(cols)
        for c, it in zip(row, items[i:i + cols]):
            with c:
                yield it


# ---------------------------------------------------------------------------
# Scan tab
# ---------------------------------------------------------------------------
def scan_tab():
    job: ScanJob | None = job_slot()["job"]
    running = job is not None and not job.done

    if not running:
        st.write("Film your closet slowly, left to right, one section at a time. "
                 "New items are **added** to your closet; things it already has are skipped.")
        video = st.file_uploader("Closet video", type=["mov", "mp4", "m4v", "hevc"], key="upload")
        with st.expander("Advanced"):
            threshold = st.slider("Duplicate threshold (higher merges more)", 0.05, 0.6, 0.25, 0.01)
            fps = st.select_slider("Frames per second", [1.0, 2.0, 3.0, 4.0, 5.0], 3.0)
            min_det = st.slider("Ignore items seen in fewer than N frames", 1, 5, 2)
            box_thr = st.slider("Detector confidence", 0.15, 0.6, 0.3, 0.05)
        if video is not None and st.button("Scan video", type="primary", icon="🔍"):
            dest = CLOSET.root / "uploads" / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(video.name).name}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as f:
                shutil.copyfileobj(video, f, length=16 << 20)
            start_scan(dest, dict(threshold=threshold, fps=fps, min_detections=min_det, box_threshold=box_thr))
            st.rerun()

    if job is not None:
        scan_status()


@st.fragment(run_every=1.5)
def scan_status():
    job: ScanJob | None = job_slot()["job"]
    if job is None:
        return
    elapsed = time.time() - job.started
    if not job.done:
        st.progress(job.progress, text=f"{job.message} · {elapsed:.0f}s")
        st.caption(f"Scanning **{job.video}** on your Mac. You can lock your phone; come back any time.")
        return
    if job.error:
        st.error("Scan failed")
        st.code(job.error[-2000:])
    else:
        r = job.result
        st.success(f"Added **{len(r.added)}** new items · {len(set(r.matched))} already in your closet · "
                   f"{r.n_frames} frames, {r.n_detections} detections · {r.seconds:.0f}s")
        new = [it for it in load_items() if it["id"] in set(r.added)]
        for it in grid(new, 4):
            thumb(it)
        if r.debug_dir:
            st.caption(f"Debug images on your Mac: `closet/{os.path.relpath(r.debug_dir, CLOSET.root)}`")
    if not job.acknowledged:
        job.acknowledged = True
        st.rerun(scope="app")  # refresh the Closet tab with the new items
    if st.button("Clear", key="clear_job"):
        job_slot()["job"] = None
        st.rerun(scope="app")


# ---------------------------------------------------------------------------
# Closet tab
# ---------------------------------------------------------------------------
@st.dialog("Edit item")
def edit_dialog(it: dict):
    st.image(str(CLOSET.image_path(it, thumb=False)), width="stretch")
    k = it["id"]
    cat = st.selectbox("Category", CATEGORIES, index=CATEGORIES.index(it["category"]), key=f"cat_{k}")
    subs = list(SUBTYPES[cat]) + ([it["subtype"]] if cat == it["category"] and it["subtype"] not in SUBTYPES[cat] else [])
    sub = st.selectbox("Type", subs, index=subs.index(it["subtype"]) if it["subtype"] in subs else 0, key=f"sub_{k}")
    names = list(NAMED_COLORS)
    color = st.selectbox("Color", names, index=names.index(it["color"]["name"]) if it["color"]["name"] in names else 0,
                         key=f"col_{k}")
    pattern = st.selectbox("Pattern", PATTERNS, index=PATTERNS.index(it.get("pattern", "solid")), key=f"pat_{k}")
    # Changing the type resets formality/warmth to that type's defaults.
    f0, w0 = (it["formality"], it["warmth"]) if sub == it["subtype"] else subtype_attrs(cat, sub)
    c1, c2 = st.columns(2)
    formality = c1.slider("Dressy", 1, 5, int(f0), key=f"f_{k}_{sub}")
    warmth = c2.slider("Warm", 1, 5, int(w0), key=f"w_{k}_{sub}")
    b1, b2 = st.columns(2)
    if b1.button("Save", type="primary", key=f"save_{k}"):
        new_color = it["color"] if color == it["color"]["name"] else describe_name(color)
        CLOSET.update_item(k, category=cat, subtype=sub, color=new_color, pattern=pattern,
                           formality=formality, warmth=warmth, user_edited=True)
        st.rerun()
    if b2.button("Delete", key=f"del_{k}"):
        CLOSET.delete_items([k])
        st.rerun()


def closet_tab():
    items = load_items()
    if not items:
        st.info("Your closet is empty. Scan a video in the **Scan** tab.")
        return
    counts = {c: sum(it["category"] == c for it in items) for c in CATEGORIES}
    options = ["all"] + [c for c in CATEGORIES if counts[c]]
    cat = st.segmented_control("Category", options, default="all", label_visibility="collapsed",
                               format_func=lambda c: f"All {len(items)}" if c == "all" else f"{CAT_ICON[c]} {counts[c]}",
                               key="closet_cat") or "all"
    shown = [it for it in items if cat == "all" or it["category"] == cat]
    shown.sort(key=lambda it: (CATEGORIES.index(it["category"]), it["color"]["lab"][0]))

    select = st.toggle("Select to merge or delete", key="select_mode")
    selected = [it["id"] for it in shown if select and st.session_state.get(f"sel_{it['id']}")]
    if select:
        st.caption("Merging keeps the photo of the first-selected item.")
        b1, b2 = st.columns(2)
        if b1.button(f"Merge {len(selected)}", disabled=len(selected) < 2):
            CLOSET.merge_items(selected[0], selected[1:])
            for i in selected:
                st.session_state.pop(f"sel_{i}", None)
            st.toast(f"Merged {len(selected)} items")
            st.rerun()
        if b2.button(f"Delete {len(selected)}", disabled=not selected):
            CLOSET.delete_items(selected)
            st.toast(f"Deleted {len(selected)} items")
            st.rerun()

    for it in grid(shown, 3):
        thumb(it)
        if select:
            st.checkbox("select", key=f"sel_{it['id']}", label_visibility="collapsed")
        elif st.button("Edit", key=f"edit_{it['id']}", type="tertiary"):
            edit_dialog(it)


# ---------------------------------------------------------------------------
# Outfits tab
# ---------------------------------------------------------------------------
def outfits_tab():
    items = load_items()
    if not items:
        st.info("Scan your closet first.")
        return
    occasion = st.segmented_control("Occasion", list(OCCASIONS), default="casual", key="occasion") or "casual"
    weather = st.segmented_control("Weather", list(WEATHER), default="mild", key="weather") or "mild"
    ids_e, vecs = CLOSET.load_embeddings()
    embeddings = dict(zip(ids_e, vecs))
    model_ok = load_compat_model() is not None and any(it["id"] in embeddings for it in items)
    scorer = st.segmented_control(
        "Style scoring", ["model", "rules"] if model_ok else ["rules"], key="scorer",
        default="model" if model_ok else "rules",
        format_func=lambda s: {"model": "Learned model", "rules": "Color rules"}[s],
        help="Learned model: trained on 17k Polyvore outfits to judge which pieces go together. "
             "Color rules: the hand-written color/pattern/contrast rules. Occasion and weather apply to both.",
    ) or "rules"
    by_id = {it["id"]: it for it in items}
    must = st.selectbox("Must include", [None] + [it["id"] for it in items], key="must",
                        format_func=lambda i: "Anything" if i is None else item_label(by_id[i]))
    n = st.slider("How many", 3, 10, 5, key="n_outfits")

    feedback = CLOSET.load_feedback()
    params = (occasion, weather, scorer, must, n, len(items), CLOSET.items_path.stat().st_mtime)
    if st.session_state.get("outfit_params") != params or st.button("Shuffle again", icon="🔄"):
        # Votes from this batch count immediately; later regenerations use them.
        st.session_state.outfit_params = params
        st.session_state.outfits = generate_outfits(items, occasion, weather, n=n, must_include=must,
                                                    feedback=feedback, scorer=scorer, embeddings=embeddings)
        st.session_state.voted = {}
    outs = st.session_state.get("outfits", [])
    if not outs:
        st.warning("No complete outfits possible yet. You need tops + bottoms, or a dress.")
        return

    for idx, o in enumerate(outs):
        with st.container(border=True):
            for it in grid(o.items, max(3, len(o.items))):
                thumb(it)
            st.markdown("".join(f"<p class='reason'>• {r}</p>" for r in o.reasons), unsafe_allow_html=True)
            key = ",".join(o.ids)
            voted = st.session_state.voted.get(key)
            c1, c2, c3 = st.columns([1, 1, 2])
            c3.caption(f"score {o.score:+.2f}" + ("  ·  saved 👍" if voted == 1 else "  ·  saved 👎" if voted == -1 else ""))
            if c1.button("👍", key=f"up_{idx}", disabled=voted is not None):
                CLOSET.add_feedback(o.ids, +1, {"occasion": occasion, "weather": weather, "scorer": scorer})
                st.session_state.voted[key] = 1
                st.rerun()
            if c2.button("👎", key=f"down_{idx}", disabled=voted is not None):
                CLOSET.add_feedback(o.ids, -1, {"occasion": occasion, "weather": weather, "scorer": scorer})
                st.session_state.voted[key] = -1
                st.rerun()


# ---------------------------------------------------------------------------
st.markdown("### 🧺 Sweep")
scan, closet, outfits = st.tabs(["Scan", "Closet", "Outfits"])
with scan:
    scan_tab()
with closet:
    closet_tab()
with outfits:
    outfits_tab()
