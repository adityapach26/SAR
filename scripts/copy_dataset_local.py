"""Copy the agri dataset (s1/ + s2/) from Drive to a local Colab path.

Reads source from configs/config.yaml 'dataset.path' (the mounted Drive
folder) and mirrors it to a local path (default configs/config.yaml
'dataset.local_dataset_path' = /content/local_dataset/agri) so training can
read fast from local disk instead of streaming through the Drive mount.

Only the *read* path switches; train scripts keep saving checkpoints to
WorkPaths.checkpoint_dir (Drive), since those must persist.

Usage:
    python scripts/copy_dataset_local.py
    python scripts/copy_dataset_local.py --local-dir /content/agri_copy
    python scripts/copy_dataset_local.py --source /path/to/agri --local-dir /content/agri
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402

# Fallback when dataset.local_dataset_path is absent from the config.
DEFAULT_LOCAL_DIR = "/content/local_dataset/agri"
PROGRESS_EVERY = 100


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{int(n)} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def local_dataset_dir(cfg):
    """The local (Colab) directory holding the dataset mirror."""
    p = getattr(cfg.dataset, "local_dataset_path", None)
    return str(p) if p else DEFAULT_LOCAL_DIR


def copy_tree(src, dst):
    """Copy every file under ``src`` into ``dst``, printing progress.

    Uses shutil.copy2 per file (dirs created as needed), reporting a progress
    line every PROGRESS_EVERY files plus a final summary. Returns
    ``(n_files, n_bytes, total_sec)``.
    """
    src = Path(src)
    dst = Path(dst)
    files = [p for p in src.rglob("*") if p.is_file()]
    total = len(files)
    if total == 0:
        raise SystemExit(f"ERROR: no files found under {src!r}")

    t0 = time.time()
    copied = 0
    n_bytes = 0
    dst.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = dst / f.relative_to(src)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)
        copied += 1
        n_bytes += target.stat().st_size
        if copied % PROGRESS_EVERY == 0 or copied == total:
            elapsed = time.time() - t0
            rate = (n_bytes / elapsed) if elapsed else 0.0
            print(f"  [{copied:>5}/{total:>5}] {elapsed:>6.1f}s  "
                  f"({human_bytes(n_bytes)} @ {human_bytes(rate)}/s)")
    total_sec = time.time() - t0
    return copied, n_bytes, total_sec


def main() -> int:
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    ap = argparse.ArgumentParser(
        description="Copy the agri dataset from Drive to a local Colab path.")
    ap.add_argument("--source", default=str(cfg.dataset.path),
                    help="Dataset folder to copy (default: configs/config.yaml 'dataset.path').")
    ap.add_argument("--local-dir", default=local_dataset_dir(cfg),
                    help="Local destination (default: configs/config.yaml "
                         "dataset.local_dataset_path).")
    args = ap.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: dataset folder not found: {source!r} "
              "(run where the Drive folder is mounted, or pass --source).")
        return 1
    local = Path(args.local_dir)
    print(f"copying dataset\n  from: {source}\n  to  : {local}")
    n_files, n_bytes, total_sec = copy_tree(source, local)
    print(f"done: {n_files} files, {human_bytes(n_bytes)} copied in {total_sec:.1f}s")
    print(f"local dataset ready at: {local}")
    return 0


if __name__ == "__main__":
    sys.exit(main())