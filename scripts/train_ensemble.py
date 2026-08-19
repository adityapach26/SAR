"""Ensemble training: run :func:`train_generator` once per member.

Reads ``ensemble.seeds`` and ``ensemble.diversity_mode`` from config.

  * ``diversity_mode == "seed"``:
      loop over seeds calling
      ``train_generator(seed, cfg, f"checkpoints/generator_seed{seed}.pt")``.
      Every member trains on the SAME split (the config default); seed is the
      only source of diversity.

  * ``diversity_mode == "seed_and_split"``:
      additionally derive a distinct held-out train/val split per member, using
      the member's seed to drive the split (deterministic, reproducible, and
      different for every seed). That training subset is passed as ``train_split``
      to ``train_generator``, so members now differ by BOTH seed and data.

Progress is printed between runs.

Local sanity check (tiny dummy data):
    python scripts/train_ensemble.py --dataset-path data/_tiny_dummy --epochs 1

Run:
    python scripts/train_ensemble.py [--dataset-path PATH] [--epochs N] [--mode seed|seed_and_split]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config            # noqa: E402
from data.dataset import build_pairs                    # noqa: E402
from scripts.train import train_generator, count_scenes  # noqa: E402


def make_seed_split(seed, config):
    """Distinct train/val split derived deterministically from ``seed``.

    Returns ``(train_pairs, val_pairs)``. ``config.dataset.val_split`` fraction
    of the pairs is held out as validation, but *which* pairs is decided by a
    PRNG seeded with ``seed`` — so every member holds out a different subset.
    """
    pairs, _ = build_pairs(config.dataset.path,
                           num_scenes=count_scenes(config.dataset.path))
    if not pairs:
        raise SystemExit(f"no matching image pairs under {config.dataset.path!r}")

    # Deterministic per-seed ordering (python's random.Random, not global RNG).
    rng = random.Random(seed)
    order = list(range(len(pairs)))
    rng.shuffle(order)

    val_n = max(1, int(round(len(pairs) * config.dataset.val_split)))
    val_idx = set(order[:val_n])
    train_pairs = [p for i, p in enumerate(pairs) if i not in val_idx]
    val_pairs = [p for i, p in enumerate(pairs) if i in val_idx]

    # Log which positions are held out, so run-to-run (and member-to-member)
    # split diversity is visible on screen, not just implicit.
    print(f"  split indices: train={sorted(set(range(len(pairs))) - val_idx)}  "
          f"held-out val={sorted(val_idx)}  (from seed {seed})")
    return train_pairs, val_pairs


def main() -> int:
    ap = argparse.ArgumentParser(description="Ensemble training (train_generator per member).")
    ap.add_argument("--dataset-path", default=None,
                    help="override dataset path (e.g. a tiny local dummy set for a sanity run)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.num_epochs (for a quick local sanity check)")
    ap.add_argument("--mode", default=None,
                    help="override ensemble.diversity_mode: seed | seed_and_split")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    if args.dataset_path:
        cfg.dataset.path = args.dataset_path
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs

    mode = args.mode or getattr(cfg.ensemble, "diversity_mode", "seed")
    if mode not in ("seed", "seed_and_split"):
        raise SystemExit(f"unknown diversity_mode: {mode!r} (expected 'seed' or 'seed_and_split')")
    seeds = list(cfg.ensemble.seeds)
    print(f"ensemble: mode={mode}  members={len(seeds)}  seeds={seeds}  "
          f"epochs={cfg.train.num_epochs}  dataset={cfg.dataset.path!r}")

    for i, seed in enumerate(seeds, 1):
        out = str(ROOT / "checkpoints" / f"generator_seed{seed}.pt")
        print(f"\n{'='*60}\n[member {i}/{len(seeds)}] training seed {seed} ...")
        if mode == "seed_and_split":
            train_pairs, val_pairs = make_seed_split(seed, cfg)
            print(f"  split: train={len(train_pairs)}  val (held out)={len(val_pairs)}  "
                  f"(derived deterministically from seed {seed})")
            train_generator(seed, cfg, out, train_split=train_pairs)
        else:  # "seed"
            print("  split: config default (identical for every member); seed = only diversity")
            train_generator(seed, cfg, out)
        print(f"[member {i}/{len(seeds)}] done -> {out}")

    print(f"\n{'='*60}\nEnsemble complete: {len(seeds)} members trained ({mode}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())