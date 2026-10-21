"""FashionCLIP text embeddings of each item's name (title if present, else url_name).

    python -m compat.embed_text

Writes data/polyvore/emb/text/<split>.npz (ids, emb) with the same ids as the image embeddings.
"""
from __future__ import annotations

import numpy as np

from .data import SPLITS, item_metadata, parquet_ids
from .embed_items import out_dir


def item_text(item_id: str) -> str:
    m = item_metadata()[item_id]
    return (m.get("title") or "").strip() or m["url_name"].strip()


def main() -> None:
    from sweep.embed import FashionClipEmbedder
    from sweep.device import free_memory

    emb = FashionClipEmbedder()
    tok = emb.processor.tokenizer
    root = out_dir("text")
    root.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        ids = parquet_ids(split)
        texts = [item_text(i) for i in ids]
        out = []
        for s in range(0, len(texts), 256):
            batch = texts[s:s + 256]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=77).to(emb.device)
            with emb.torch.inference_mode():
                f = emb.model.get_text_features(**inputs)
            f = (f.pooler_output if not hasattr(f, "shape") else f).float().cpu().numpy()
            out.append(f / np.linalg.norm(f, axis=1, keepdims=True))
        e = np.concatenate(out).astype(np.float32)
        assert np.isfinite(e).all() and len(e) == len(ids)
        np.savez(root / f"{split}.npz", ids=np.array(ids), emb=e)
        free_memory()
        print(f"[{split}] {e.shape} -> {root / split}.npz", flush=True)


if __name__ == "__main__":
    main()
