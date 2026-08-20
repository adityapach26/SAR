"""Quick end-to-end sanity run on the real dataset (or a tiny subset).

Selects a random subset (--num-pairs, default 160) of SAR/RGB pairs from the real
dataset path (configs/config.yaml 'dataset.path'), seeded via
config.dataset.random_seed (mirroring full-dataset shuffling in real training),
builds the SEN12Dataset from just those pairs, wraps it in a
DataLoader (--batch-size, default 8 -> 20 batches), and runs exactly 1 epoch of
the Phase 3 training loop (Generator + Discriminator forward/backward, all loss
terms enabled per config) while timing it.

Breakdown focused on *where* the time goes:
  * a 2-batch untimed warmup flushes one-time CUDA/cuDNN init before the timer
  * then the remaining batches are timed and split into stages:
        data-load  : DataLoader iteration + device copy
        gen-fwd    : generator forward
        disc-fwd   : discriminator forward (both real + fake)
        loss       : loss computation (incl. aux loss-network forwards)
        backward   : .backward() + optimizer .step() (both D and G)

Also explicitly prints the .device of the generator, discriminator, and the
perceptual / semantic loss networks so you can confirm everything is actually
on CUDA.

Usage:
    python scripts/sanity_test_real_data.py
    python scripts/sanity_test_real_data.py --dataset-path /path/to/agri
    python scripts/sanity_test_real_data.py --batch-size 16 --num-pairs 320
"""

from __future__ import annotations

import argparse
import random
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

BATCH_SIZE = 8           # default; overridable via --batch-size
TARGET_PAIRS = 160       # default timed sample target; overridable via --num-pairs
WARMUP_BATCHES = 2       # untimed throwaway batches before the timer
FULL_DATASET_SIZE = 4000
PIN_MEMORY = True        # page-lock host tensors for faster GPU copies
PROFILE_SAMPLES = 4      # how many samples to use for the loading-stage profile


def param_device(module) -> str:
    """'cuda:0' / 'cpu' of the module's first parameter (or 'n/a' if empty)."""
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "n/a"


def train_step(gen, disc, opt_g, opt_d, bce, l1, perceptual, semantic, cfg,
               sar, rgb, times: dict):
    """One full D+G step; adds stage wall-times into the ``times`` dict.
    Assumes ``sar``/``rgb`` are already on ``cfg_device`` (call sites copy them)."""
    # ---- Discriminator: real pair vs. fake (generated) pair ----
    t0 = time.time()
    fake = gen(sar).detach()
    times["gen_fwd"] += time.time() - t0

    t0 = time.time()
    d_real = disc(sar, rgb)
    d_fake = disc(sar, fake)
    times["disc_fwd"] += time.time() - t0

    t0 = time.time()
    loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
    times["loss"] += time.time() - t0

    t0 = time.time()
    loss_d.backward()
    opt_d.step()
    times["backward"] += time.time() - t0

    # ---- Generator: adversarial + L1 (+ perceptual/semantic if enabled) ----
    t0 = time.time()
    fake = gen(sar)
    times["gen_fwd"] += time.time() - t0

    t0 = time.time()
    g_adv = bce(disc(sar, fake), torch.ones_like(d_real))
    g_l1 = l1(fake, rgb)
    g_loss = cfg.loss.lambda_gan * g_adv + cfg.loss.lambda_l1 * g_l1
    if perceptual is not None:
        g_loss = g_loss + cfg.loss.lambda_perceptual * perceptual(fake, rgb)
    if semantic is not None:
        g_loss = g_loss + cfg.loss.lambda_semantic * semantic(fake, rgb)
    times["loss"] += time.time() - t0

    t0 = time.time()
    g_loss.backward()
    opt_g.step()
    times["backward"] += time.time() - t0

    return g_loss


def main() -> int:
    ap = argparse.ArgumentParser(description="Sanity test on real (subset) data.")
    ap.add_argument("--dataset-path", default=None,
                    help="Path to the agri folder. Defaults to configs/config.yaml 'dataset.path'.")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                    help="Batch size for the DataLoader (default: %(default)s).")
    ap.add_argument("--num-pairs", type=int, default=TARGET_PAIRS,
                    help="Number of SAR/RGB pairs to randomly sample for the run "
                         "(default: %(default)s).")
    args = ap.parse_args()

    batch_size = args.batch_size
    num_pairs = args.num_pairs

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    path = args.dataset_path or cfg.dataset.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg_device = device
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}   selected device: {device}")
    print(f"dataset path: {path}\n")

    # --- Real pairs: num_pairs randomly sampled from the full scan. ---
    all_pairs, mismatches = build_pairs(path)
    print(f"Total pairs found (full scan): {len(all_pairs)}   mismatches: {len(mismatches)}")
    rng = random.Random(cfg.dataset.random_seed)
    pairs = rng.sample(all_pairs, min(num_pairs, len(all_pairs)))
    if not pairs:
        print("WARNING: 0 pairs — nothing to run. "
              "Is the dataset path mounted in this environment?")
        return 1

    num_channels = cfg.input_channels.num_channels
    num_workers = cfg.train.num_workers
    ds = SEN12Dataset(pairs, num_channels=num_channels)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=PIN_MEMORY)
    n_batches = len(loader)
    # Warmup uses its own (throwaway) iterator passes, so the timed epoch below
    # still covers all n_batches of the real loader -> maximal timed sample.
    print(f"Using {len(ds)} pairs, batch_size={batch_size} num_workers={num_workers} "
          f"pin_memory={PIN_MEMORY} -> {n_batches} batches "
          f"(plus {WARMUP_BATCHES} warmup passes, untimed; all {n_batches} timed)\n")

    # --- Image-loading stage profile: PIL open vs. multichannel (uniform_filter).
    # Run in-process (num_workers=0 path) so timings land in one process.
    prof_ds = SEN12Dataset(pairs[:PROFILE_SAMPLES], num_channels=num_channels, profile=True)
    prof_ds.reset_profile()
    for i in range(len(prof_ds)):
        prof_ds[i]  # force load + multichannel
    p = prof_ds.profile_times
    n_p = float(len(prof_ds))
    print("-- loading stage profile (per-sample avg, in-process) --")
    print(f"  PIL Image.open (SAR)      : {p['pil_sar'] / n_p:.4f} s")
    print(f"  multichannel (uniform_ft) : {p['multichannel'] / n_p:.4f} s")
    print(f"  PIL Image.open (RGB)      : {p['pil_rgb'] / n_p:.4f} s")
    print(f"  loading total (per-sample): {(p['pil_sar'] + p['multichannel'] + p['pil_rgb']) / n_p:.4f} s")
    if p['multichannel'] > p['pil_sar']:
        print("  -> build_multichannel_input (uniform_filter) is the bottleneck in loading")
    else:
        print("  -> PIL Image.open (disk/decoding) is the bottleneck in loading")
    print()

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
    lp, ls = cfg.loss.lambda_perceptual, cfg.loss.lambda_semantic
    perceptual = PerceptualLoss().to(device) if lp > 0 else None
    semantic = SemanticLoss().to(device) if ls > 0 else None
    print(f"lambdas: gan={cfg.loss.lambda_gan} l1={cfg.loss.lambda_l1} "
          f"perc={lp} sem={ls}  (perceptual={'on' if perceptual else 'off'}, "
          f"semantic={'on' if semantic else 'off'})")

    # --- (1) Prove every network is actually on the selected device. ---
    print("\n-- device check --")
    print(f"  generator          : {param_device(gen)}")
    print(f"  discriminator      : {param_device(disc)}")
    print(f"  perceptual (VGG16) : {param_device(perceptual) if perceptual else 'disabled'}")
    print(f"  semantic  (ResNet) : {param_device(semantic) if semantic else 'disabled'}")

    gen.train()
    disc.train()

    # --- (2) Warmup: WARMUP_BATCHES untimed throwaway batches. ---
    print(f"\n-- warmup: {WARMUP_BATCHES} untimed batch(es) to flush CUDA/cuDNN init --")
    for _ in range(WARMUP_BATCHES):
        sar, rgb = next(iter(loader))
        sar, rgb = sar.to(device), rgb.to(device)
        train_step(gen, disc, opt_g, opt_d, bce, l1, perceptual, semantic, cfg,
                   sar, rgb, times := {k: 0.0 for k in
                                       ("data_load", "gen_fwd", "disc_fwd", "loss", "backward")})

    # --- (3)+(4) Timed epoch with per-stage breakdown. ---
    times = {"data_load": 0.0, "gen_fwd": 0.0, "disc_fwd": 0.0,
             "loss": 0.0, "backward": 0.0}
    epoch_t0 = time.time()
    timed_batches = 0
    it = iter(loader)
    while True:
        t_data = time.time()
        try:
            sar, rgb = next(it)
        except StopIteration:
            break
        sar, rgb = sar.to(device), rgb.to(device)  # device copy counted under data_load
        times["data_load"] += time.time() - t_data
        train_step(gen, disc, opt_g, opt_d, bce, l1, perceptual, semantic, cfg,
                   sar, rgb, times)
        timed_batches += 1
    epoch_sec = time.time() - epoch_t0

    # --- Report + extrapolate. ---
    per = {k: v / timed_batches for k, v in times.items()} if timed_batches else times
    total_stage = per["data_load"] + per["gen_fwd"] + per["disc_fwd"] + per["loss"] + per["backward"]
    full_epoch_batches = (FULL_DATASET_SIZE + batch_size - 1) // batch_size
    full_epoch_sec = total_stage * full_epoch_batches
    total_sec = full_epoch_sec * cfg.train.num_epochs

    print("\n================ SANITY REPORT ================")
    print(f"total pairs found (full scan) : {len(all_pairs)}")
    print(f"total mismatches              : {len(mismatches)}")
    print(f"batches run (1 epoch)         : {n_batches}  (timed: {timed_batches})")
    print(f"seconds per batch (total)     : {total_stage:.4f}")
    print(f"seconds for full epoch (sub)  : {epoch_sec:.3f}")
    print(f"-- per-batch stage breakdown (avg over {timed_batches} timed batches) --")
    print(f"  data load    : {per['data_load']:.4f} s  ({100 * per['data_load'] / total_stage:.1f}%)")
    print(f"  gen forward  : {per['gen_fwd']:.4f} s  ({100 * per['gen_fwd'] / total_stage:.1f}%)")
    print(f"  disc forward : {per['disc_fwd']:.4f} s  ({100 * per['disc_fwd'] / total_stage:.1f}%)")
    print(f"  loss compute : {per['loss']:.4f} s  ({100 * per['loss'] / total_stage:.1f}%)")
    print(f"  backward     : {per['backward']:.4f} s  ({100 * per['backward'] / total_stage:.1f}%)")
    print(f"-- device front --")
    print(f"  generator  : {param_device(gen)}   discriminator: {param_device(disc)}")
    print(f"  perceptual : {param_device(perceptual) if perceptual else 'disabled'}   "
          f"semantic: {param_device(semantic) if semantic else 'disabled'}")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"-- extrapolation ({FULL_DATASET_SIZE} pairs, batch {batch_size}, "
          f"{cfg.train.num_epochs} epochs) --")
    print(f"  est. sec/epoch on {FULL_DATASET_SIZE} pairs : {full_epoch_sec:.1f}")
    print(f"  est. total: {total_sec:.2f}s (~{total_sec / 60:.2f} min)")
    print("================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())