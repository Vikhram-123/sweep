"""Download the Polyvore Outfits *disjoint* split from the authors' Hugging Face repo and verify it.

    python -m compat.download --out data/polyvore

Every file is checked against the hash Hugging Face records for it: SHA-256 for LFS files
(the parquet shards with embedded images), git blob SHA-1 for small files. Re-running skips
files that already verify, so an interrupted download just resumes.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = "mvasil/polyvore-outfits"
PATTERNS = (
    "data/disjoint/",          # train/validation/test parquet (item images + metadata)
    "disjoint/",               # outfit splits, compatibility + FITB questions, typespaces
    "polyvore_item_metadata.json",
    "polyvore_outfit_titles.json",
    "categories.csv",
    "README.md",
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def verify(path: Path, entry) -> bool:
    if not path.exists() or path.stat().st_size != entry.size:
        return False
    if entry.lfs is not None:
        return _sha256(path) == entry.lfs.sha256
    return _git_blob_sha1(path) == entry.blob_id


def main(argv=None) -> int:
    from huggingface_hub import HfApi, hf_hub_download

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=Path("data/polyvore"))
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    entries = [e for e in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True)
               if hasattr(e, "size") and e.path.startswith(PATTERNS)]
    total = sum(e.size for e in entries)
    free = shutil.disk_usage(args.out).free
    print(f"{len(entries)} files, {total / 1e9:.2f} GB to fetch; {free / 1e9:.1f} GB free")
    if total * 1.2 > free:
        print("Not enough free disk space.", file=sys.stderr)
        return 1

    bad = []
    for e in sorted(entries, key=lambda e: e.size):
        dest = args.out / e.path
        if verify(dest, e):
            print(f"  ok (cached)  {e.path}")
            continue
        hf_hub_download(REPO, e.path, repo_type="dataset", local_dir=args.out)
        status = "ok" if verify(dest, e) else "HASH MISMATCH"
        print(f"  {status:<12} {e.path}  ({e.size / 1e6:.1f} MB)")
        if status != "ok":
            bad.append(e.path)
    if bad:
        print(f"{len(bad)} files failed verification: {bad}", file=sys.stderr)
        return 1
    print("All files verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
