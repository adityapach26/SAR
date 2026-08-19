"""Quick end-to-end sanity run on the real dataset (or a tiny subset).

Loads the first 24 SAR/RGB pairs from the real dataset path (configs/config.yaml
'dataset.path'), builds the SEN12Dataset from just those pairs, wraps it in a
DataLoader (batch_size=8), and runs exactly 1 epoch of the Phase 3 training
loop (Generator + Discriminator forward/backward, all loss terms enabled per
config) while timing it. Prints throughput numbers plus an extrapolated total
for the full 4000-pair dataset over train.num_epochs (50) epochs, and whether
CUDA is available.

Usage:
    python scripts/sanity_test_real_data.py
    python scripts/sanity_test_real_data.py --dataset-path /path/to/agri
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.dataset import build_pairs, SEN12Dataset          # noqa: E402
from models.generator import Generator                      # noqa: E402
from models.discriminator import PatchGANDiscriminator       # noqa: E402
from losses.perceptual_loss import PerceptualLoss            # noqa: E402
from losses.semantic_loss import SemanticLoss                # noqa: E402
from utils.config_loader import load_config                  # noqa: E402

FIRST_N_PAIRS = 24
BATCH_SIZE = 8
FULL_DATASET_SIZE = 4000


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity test on real (subset) data.")
    ap.add_argument("--dataset-path", default=None,
                    help="Path to the agri folder. Defaults to configs/config.yaml 'dataset.path'.")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    path = args.dataset_path or cfg.dataset.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}  (device: {device})")
    print(f"dataset path: {path}\n")

    # --- Real pairs, first 24 only. ---
    all_pairs, mismatches = build_pairs(path)
    print(f"Total pairs found (full scan): {len(all_pairs)}   "
          f"mismatches: {len(mismatches)}")
    pairs = all_pairs[: min(FIRST_N_PAIRS, len(all_pairs))]
    if not pairs:
        print("WARNING: 0 pairs — nothing to run. "
              "Is the dataset path mounted in this environment?")
        return 1

    num_channels = cfg.input_channels.num_channels
    ds = SEN12Dataset(pairs, num_channels=num_channels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"Using {len(ds)} pairs, batch_size={BATCH_SIZE} -> {len(loader)} batch(es)\n")

    # --- Models + optimizers + losses (all per config). ---
    gen = Generator(sar_channels=num_channels, rgb_channels=3,
                    base_channels=cfg.model.generator.base_channels).to(device)
    disc = PatchGANDiscriminator(sar_channels=num_channels, rgb_channels=3,
                                 base_channels=cfg.model.discriminator.base_channels).to(device)
    lr = float(cfg.train.learning_rate)
    betas = (cfg.train.beta1, cfg.train.beta2)
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    lp = cfg.loss.lambda_perceptual
    ls = cfg.loss.lambda_semantic
    perceptual = PerceptualLoss().to(device) if lp > 0 else None
    semantic = SemanticLoss().to(device) if ls > 0 else None
    print(f"lambdas: gan={cfg.loss.lambda_gan} l1={cfg.loss.lambda_l1} "
          f"perc={lp} sem={ls}  (perceptual={'on' if perceptual else 'off'}, "
          f"semantic={'on' if semantic else 'off'})")

    # --- Exactly 1 epoch, timed. ---
    gen.train()
    disc.train()
    epoch_t0 = time.time()
    batch_times = []
    for i, (sar, rgb) in enumerate(loader, 1):
        t0 = time.time()
        sar, rgb = sar.to(device), rgb.to(device)

        disc.zero_grad()
        fake = gen(sar).detach()
        d_real = disc(sar, rgb)
        d_fake = disc(sar, fake)
        loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
        loss_d.backward()
        opt_d.step()

        gen.zero_grad()
        fake = gen(sar)
        g_adv = bce(disc(sar, fake), torch.ones_like(d_real))
        g_l1 = l1(fake, rgb)
        g_loss = cfg.loss.lambda_gan * g_adv + cfg.loss.lambda_l1 * g_l1
        if perceptual is not None:
            g_loss = g_loss + lp * perceptual(fake, rgb)
        if semantic is not None:
            g_loss = g_loss + ls * semantic(fake, rgb)
        g_loss.backward()
        opt_g.step()

        batch_times.append(time.time() - t0)

    epoch_sec = time.time() - epoch_t0
    n_batches = len(loader)

    # --- Report + extrapolate. ---
    sec_batch = epoch_sec / n_batches if n_batches else 0.0
    full_epoch_batches = (FULL_DATASET_SIZE + BATCH_SIZE - 1) // BATCH_SIZE
    full_epoch_sec = sec_batch * full_epoch_batches
    total_sec = full_epoch_sec * cfg.train.num_epochs

    print("\n================ SANITY REPORT ================")
    print(f"total pairs found (full scan) : {len(all_pairs)}")
    print(f"total mismatches              : {len(mismatches)}")
    print(f"batches run (1 epoch)         : {n_batches}")
    print(f"seconds per batch             : {sec_batch:.3f}")
    print(f"seconds for full epoch (sub)  : {epoch_sec:.3f}")
    print(f"est. sec/epoch on {FULL_DATASET_SIZE} pairs : {full_epoch_sec:.1f}")
    print(f"est. total for {FULL_DATASET_SIZE} pairs x {cfg.train.num_epochs} epochs : "
          f"{total_sec:.2f}s (~{total_sec / 60:.2f} min)")
    print(f"torch.cuda.is_available()     : {torch.cuda.is_available()}")
    print("================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())