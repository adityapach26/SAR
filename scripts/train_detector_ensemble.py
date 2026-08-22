"""Detector ensemble training: run :func:`train_detector` once per seed (Phase 5 Step 5.3).

Reads ``ensemble.seeds`` from config (currently ``[1, 2, 3]``, the same seeds
as the generator ensemble). For each seed it calls the existing
:func:`train_detector` from :mod:`scripts.train_detector`, saving the fine-tuned
detector to ``<paths.checkpoint_dir>/detector_seed{seed}.pt`` (Drive, via config).

Resume / skip: if ``<paths.checkpoint_dir>/detector_seed{seed}.pt`` already exists on disk
(``Path.exists()``), that seed is skipped instead of retrained — the same
existence-check style the engine uses for its checkpoint resume, so a crashed or
partially-finished run can be re-launched idempotently.

Epoch count: ``detection.epochs`` from config if present, else a default of 15.
The detector fine-tunes from COCO-pretrained weights and converges much faster
than the generator, so it deliberately does NOT share ``train.num_epochs``.
``--epochs N`` overrides the value for a quick multi-seed sanity run.

This module only orchestrates calls to the existing :func:`train_detector`; it
contains no model-building or training-loop code of its own.

Run:
    python scripts/train_detector_ensemble.py [--epochs N] [--dataset-path PATH]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config                 # noqa: E402
from scripts.train_detector import train_detector            # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensemble detection training (train_detector per seed).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override epochs for all seeds (default: detection.epochs or 15)")
    ap.add_argument("--dataset-path", default=None,
                    help="override detection.dataset_path (e.g. a tiny local dummy set for a sanity run)")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    if args.dataset_path:
        cfg.detection.dataset_path = args.dataset_path

    # Epoch count: detection.epochs if set, else a detector-appropriate default
    # of 15 (NOT train.num_epochs, which is the much larger generator setting).
    det_epochs = getattr(cfg.detection, "epochs", None)
    epochs = det_epochs if det_epochs is not None else 15
    if args.epochs is not None:
        epochs = args.epochs
    cfg.train.num_epochs = epochs  # train_detector reads its epoch count from config.train

    seeds = list(cfg.ensemble.seeds)
    checkpoint_dir = Path(cfg.paths.checkpoint_dir)
    t_start = time.time()

    print(f"ensemble: detector  members={len(seeds)}  seeds={seeds}  "
          f"epochs={epochs}  dataset={cfg.detection.dataset_path!r}")
    print(f"          checkpoints -> {checkpoint_dir}")

    checkpoints = []
    n_skipped = 0
    for i, seed in enumerate(seeds, 1):
        out = checkpoint_dir / f"detector_seed{seed}.pt"
        print(f"\n{'='*60}\n[member {i}/{len(seeds)}] training seed {seed} ...")

        if out.exists():
            print(f"  skipping seed {seed} — checkpoint already exists: {out}")
            checkpoints.append(out)
            n_skipped += 1
            continue

        t0 = time.time()
        train_detector(seed, cfg, str(out))
        dt = time.time() - t0
        print(f"[member {i}/{len(seeds)}] seed {seed} complete")
        print(f"  elapsed {dt:.1f}s -> {out}")
        checkpoints.append(out)

    total = time.time() - t_start
    n_trained = len(seeds) - n_skipped
    print(f"\n{'='*60}\nEnsemble complete: {len(seeds)} members "
          f"({n_trained} trained, {n_skipped} skipped). total wall-clock {total:.1f}s")
    print("checkpoints:")
    for c in checkpoints:
        status = "exists" if c.exists() else "MISSING"
        print(f"  {c}  [{status}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())