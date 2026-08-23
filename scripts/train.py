"""Pix2Pix training loop for the dual-branch SAR-to-RGB generator.

The training logic lives in :func:`train_generator`, which is self-contained:
it seeds RNG, builds *fresh* Generator/Discriminator + optimizers, trains for
``config.train.num_epochs`` epochs, and saves the final generator weights to
``output_checkpoint_path``. It returns the trained generator so callers can
e.g. train an ensemble (Phase 4).

``train_split`` is optional. When omitted, training uses the default split
derived from ``config`` (the whole dataset). When provided, it *replaces* that
default with a caller-supplied split (a DataLoader, or an explicit list of
``(sar_path, rgb_path)`` pairs) — which enables split-based diversity across
runs, not just seed-based diversity.

Standard GAN objective:
  * Discriminator is trained to tell real vs. fake (SAR, RGB) pairs apart via
    BCELogitsLoss over the PatchGAN map.
  * Generator is trained to fool the discriminator (adversarial term) and to
    match the ground-truth RGB with an L1 loss weighted by ``lambda_l1``:
      G_loss = lambda_gan*BCE(D(G(sar)), real) + lambda_l1*|G(sar) - rgb|_1

All hyperparameters are read from configs/config.yaml. The adversarial (real)
label is a ones-tensor matching the discriminator's patch-map shape.

Run:  python scripts/train.py [--seed N] [--epochs N] [--output PATH] [--local-copy]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import Generator            # noqa: E402
from models.discriminator import PatchGANDiscriminator  # noqa: E402
from utils.config_loader import load_config        # noqa: E402
from data.dataset import build_pairs, SEN12Dataset, split_pairs  # noqa: E402
from losses.perceptual_loss import PerceptualLoss  # noqa: E402
from losses.semantic_loss import SemanticLoss  # noqa: E402
from scripts.copy_dataset_local import local_dataset_dir  # noqa: E402


def count_scenes(dataset_path: Path) -> int:
    """Number of legacy SEN1-2 scenes (max s1_i index + 1). 0 for the agri layout."""
    indices = [int(d.name[3:]) for d in Path(dataset_path).iterdir()
               if d.is_dir() and d.name.startswith("s1_")]
    return (max(indices) + 1) if indices else 0


def _build_loader(cfg, pairs, shuffle=True, batch_size=None):
    """Wrap an explicit pair list in a parallel, pinned DataLoader.

    ``batch_size`` overrides ``cfg.train.batch_size`` (used by fine-tuning).
    """
    num_channels = cfg.input_channels.num_channels
    ds = SEN12Dataset(pairs, num_channels=num_channels,
                      texture_kernel_size=cfg.input_channels.texture_kernel_size)
    # Parallel background workers keep image loading off the GPU/math critical
    # path; pin_memory makes host->device copies faster on CUDA systems.
    loader = DataLoader(ds, batch_size=batch_size or cfg.train.batch_size, shuffle=shuffle,
                        num_workers=cfg.train.num_workers, pin_memory=True,
                        drop_last=False)
    return loader, len(ds)


def make_loaders(cfg):
    """Default split from config: train/val/test from all matching pairs.

    Returns ``(train_loader, n_train, val_pairs, test_pairs)``. The train loader
    is built from the train portion only; val/test pair lists are returned for
    later evaluation.
    """
    pairs, _mismatches = build_pairs(cfg.dataset.path, num_scenes=count_scenes(cfg.dataset.path))
    if not pairs:
        raise SystemExit(f"no matching image pairs under {cfg.dataset.path!r}")
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs, cfg.dataset.val_split, cfg.dataset.test_split, cfg.dataset.random_seed)
    loader, n_train = _build_loader(cfg, train_pairs, shuffle=True)
    print(f"  split: train={n_train}  val={len(val_pairs)}  "
          f"test={len(test_pairs)}  (of {len(pairs)} total)")
    return loader, n_train, val_pairs, test_pairs


def save_checkpoint(cfg, epoch, gen, disc, opt_g, opt_d, path):
    torch.save({
        "epoch": epoch,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
    }, path)
    print(f"[save] checkpoint -> {path}")


def train_generator(seed, config, output_checkpoint_path, train_split=None,
                    init_checkpoint=None, lr=None, checkpoint_name=None,
                    batch_size=None, num_epochs=None, eval_split=None,
                    best_checkpoint_path=None):
    """Train a SAR-to-RGB generator and save its final weights.

    The extra parameters are all optional (None = current behavior) and let the
    water fine-tuning entry point (scripts/finetune_water.py) reuse this exact
    training loop for a domain-adaptation run:

    init_checkpoint : Path | None
        A full checkpoint dict (``seed1_latest.pt``) whose Generator +
        Discriminator weights are restored as a *starting point only*. Unlike
        normal resume, the optimizers are NOT loaded — they are built fresh (at
        the effective LR below), so fine-tuning starts from the trained weights
        with a new, lower learning rate. This file is read-only, never written.
    lr : float | None
        Overrides ``config.train.learning_rate`` for the fresh optimizers.
    checkpoint_name : str | None
        Base name for this run's checkpoints. Defaults to ``seed{seed}``, giving
        ``seed{seed}_latest.pt`` (resume) / ``seed{seed}_epoch_XXXX.pt``
        (snapshots). Fine-tuning passes ``water_finetune_seed{seed}`` so it
        writes ``water_finetune_seed{seed}_latest.pt`` and NEVER touches the
        agriculture ``seed{seed}_latest.pt``. Resume also reads this name, so
        fine-tuning resumes its OWN latest, not agriculture's.
    batch_size : int | None
        Overrides ``config.train.batch_size`` for the DataLoader.
    num_epochs : int | None
        Overrides ``config.train.num_epochs``.
    eval_split : list[tuple[str, str]] | None
        Held-out pairs (e.g. the water 10% test split) evaluated after every
        epoch — forward pass only, no gradient/optimizer updates. ``None`` (the
        agriculture default) disables evaluation entirely.
    best_checkpoint_path : str | Path | None
        Optional Generator-only checkpoint path updated when held-out test L1
        improves. Used by water fine-tuning only; ``None`` preserves agriculture
        behavior.

    Parameters
    ----------
    seed : int
        RNG seed. ``torch.manual_seed(seed)`` is called at the very start.
    config : Config
        Loaded config object (attribute access): train/loss/model/paths keys.
    output_checkpoint_path : str | Path
        Where the final generator weights will be saved (state_dict only).
    train_split : DataLoader | list[tuple[str, str]] | None, optional
        Optional *replacement* for the default config-derived split:
          * None              -> use config's default split (whole dataset)
          * list of pairs     -> train on exactly those (sar_path, rgb_path) pairs
          * DataLoader        -> train on that loader directly
        This supports split-based diversity across runs (not just seed-based).

    Resume
    ------
    If ``<checkpoint_dir>/seed{seed}_latest.pt`` exists, this run RESUMES from
    ``saved_epoch + 1``: the saved generator/discriminator/optimizer state dicts
    are loaded into freshly-constructed modules and the loop restarts after the
    saved epoch. Otherwise it starts FRESH at epoch 1. The latest checkpoint is
    overwritten every ``train.save_every_n_epochs`` (and an epoch-numbered copy
    ``seed{seed}_epoch_XXXX.pt`` is also kept per save), so a stopped run always
    has something to resume from.

    Returns
    -------
    nn.Module
        The trained generator.
    """
    # PyYAML resolves the bare exponent "2e-4" to a string; coerce to float.
    # ``lr``/``num_epochs``/``batch_size`` override the config (fine-tuning).
    lr = float(lr if lr is not None else config.train.learning_rate)
    betas = (config.train.beta1, config.train.beta2)
    lambda_gan, lambda_l1 = config.loss.lambda_gan, config.loss.lambda_l1
    lambda_perc = config.loss.lambda_perceptual
    lambda_sem = config.loss.lambda_semantic
    n_epochs = int(num_epochs if num_epochs is not None else config.train.num_epochs)
    log_every = config.train.log_every_n_batches
    save_every = config.train.save_every_n_epochs

    # (1) deterministic RNG for this run — torch, numpy, and python all seeded
    # so data loaders / splits / weight init are reproducible from the seed.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # (2) fresh Generator / Discriminator
    sar_channels = config.input_channels.num_channels
    gen = Generator(sar_channels=sar_channels, rgb_channels=3,
                    base_channels=config.model.generator.base_channels).to(device)
    disc = PatchGANDiscriminator(sar_channels=sar_channels, rgb_channels=3,
                                 base_channels=config.model.discriminator.base_channels).to(device)

    # (3) resolve the data split (train_split overrides the config default)
    if train_split is None:
        loader, n_samples, val_pairs, test_pairs = make_loaders(config)
        split_desc = "config default"
        print(f"  split: {split_desc}  train={n_samples}  val={len(val_pairs)}  "
              f"test={len(test_pairs)}  batches/epoch={len(loader)}")
    elif isinstance(train_split, DataLoader):
        loader = train_split
        n_samples = len(loader.dataset)
        split_desc = "provided DataLoader"
        print(f"  split: {split_desc}  samples={n_samples}  batches/epoch={len(loader)}")
    else:
        pairs = list(train_split)
        loader, n_samples = _build_loader(config, pairs, shuffle=True,
                                          batch_size=batch_size)
        split_desc = f"provided split ({len(pairs)} pairs)"
        print(f"  split: {split_desc}  samples={n_samples}  batches/epoch={len(loader)}")

    print(f"seed={seed}  device={device}  epochs={n_epochs}  "
          f"batch_size={loader.batch_size}  batches_per_epoch={len(loader)}")

    # Held-out evaluation loader (water fine-tuning only; None elsewhere). The
    # test set is used ONLY for forward-pass evaluation, never for gradients.
    eval_loader = None
    if eval_split is not None:
        eval_loader, _ = _build_loader(config, list(eval_split), shuffle=False,
                                       batch_size=batch_size)
        print(f"  eval: {len(eval_split)} held-out pairs, "
              f"{len(eval_loader)} batches (forward-only)")

    # Auxiliary losses are built only when enabled (heavy + frozen).
    perceptual = PerceptualLoss().to(device) if lambda_perc > 0 else None
    semantic = SemanticLoss().to(device) if lambda_sem > 0 else None

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    ckpt_dir = Path(config.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    best_test_l1 = float("inf")
    best_epoch = None

    # ---- resume / bootstrap support ----
    # A seed-specific "latest" checkpoint is overwritten at every save interval,
    # so a stopped/crashed run can pick up from the last saved epoch. Uses the
    # freshly-constructed models/optimizers, then overwrites them with the loaded
    # state dicts (correct arch + device via map_location).
    #
    # checkpoint_name lets a fine-tuning run (scripts/finetune_water.py) use its
    # OWN latest/snapshot prefix ("water_finetune_seed{seed}") instead of the
    # agriculture "seed{seed}" — so it resumes its own latest and writes only
    # water-named files, never touching seed{seed}_latest.pt.
    base_name = checkpoint_name if checkpoint_name is not None else f"seed{seed}"
    latest_path = ckpt_dir / f"{base_name}_latest.pt"
    start_epoch = 1
    if latest_path.exists():
        try:
            _latest = torch.load(latest_path, map_location=device)
            start_epoch = int(_latest["epoch"]) + 1
            gen.load_state_dict(_latest["generator_state_dict"])
            disc.load_state_dict(_latest["discriminator_state_dict"])
            opt_g.load_state_dict(_latest["optimizer_g_state_dict"])
            opt_d.load_state_dict(_latest["optimizer_d_state_dict"])
            print(f"[resume] found {latest_path} (epoch {_latest['epoch']}) -> "
                  f"RESUMING from epoch {start_epoch}")
        except Exception as e:  # noqa: BLE001
            print(f"[resume] WARNING could not load {latest_path}: {e} -> starting fresh")
            start_epoch = 1
    elif init_checkpoint is not None:
        # Fine-tuning bootstrap: restore trained Generator + Discriminator
        # weights as a starting point, but keep the FRESH optimizers (already
        # built at the effective LR above) so fine-tuning runs with a new, lower
        # learning rate. The init checkpoint is loaded READ-ONLY — it is never
        # written back, and this run's checkpoints go to the water-named paths.
        try:
            _init = torch.load(init_checkpoint, map_location=device)
            gen.load_state_dict(_init["generator_state_dict"])
            disc.load_state_dict(_init["discriminator_state_dict"])
            print(f"[init] bootstrapped G+D from {init_checkpoint} "
                  f"(fresh optimizers @ lr={lr:g})")
        except Exception as e:  # noqa: BLE001
            print(f"[init] WARNING could not load {init_checkpoint}: {e} -> starting fresh")
    else:
        print(f"[resume] no checkpoint at {latest_path} -> FRESH start from epoch 1")

    for epoch in range(start_epoch, n_epochs + 1):
        gen.train()
        disc.train()
        acc = {"d": 0.0, "g_adv": 0.0, "g_l1": 0.0, "g_perc": 0.0, "g_sem": 0.0,
              "g": 0.0, "n": 0}

        t0 = time.time()
        for i, (sar, rgb) in enumerate(loader, 1):
            sar, rgb = sar.to(device), rgb.to(device)
            batch = sar.size(0)

            # ---- Discriminator: real pair vs. fake (generated) pair ----
            # Labels take the discriminator's own patch-map shape (not hardcoded),
            # so they stay correct for any input size / final partial batch.
            disc.zero_grad()
            fake = gen(sar).detach()
            d_real = disc(sar, rgb)
            d_fake = disc(sar, fake)
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            loss_d.backward()
            opt_d.step()

            # ---- Generator: adversarial + L1 (+ perceptual if enabled) ----
            gen.zero_grad()
            fake = gen(sar)
            g_adv = bce(disc(sar, fake), torch.ones_like(d_real))
            g_l1 = l1(fake, rgb)
            loss_g = lambda_gan * g_adv + lambda_l1 * g_l1
            if perceptual is not None:
                g_perc = perceptual(fake, rgb)
                loss_g = loss_g + lambda_perc * g_perc
            else:
                g_perc = torch.zeros_like(loss_g)
            if semantic is not None:
                g_sem = semantic(fake, rgb)
                loss_g = loss_g + lambda_sem * g_sem
            else:
                g_sem = torch.zeros_like(loss_g)
            loss_g.backward()
            opt_g.step()

            acc["d"] += loss_d.item() * batch
            acc["g_adv"] += g_adv.item() * batch
            acc["g_l1"] += g_l1.item() * batch
            acc["g_perc"] += g_perc.item() * batch
            acc["g_sem"] += g_sem.item() * batch
            acc["g"] += loss_g.item() * batch
            acc["n"] += batch

            if i % log_every == 0 or i == len(loader):
                print(f"  epoch {epoch}  batch {i}/{len(loader)} "
                      f"D {acc['d'] / acc['n']:.4f} | "
                      f"G {acc['g'] / acc['n']:.4f} "
                      f"(adv {acc['g_adv'] / acc['n']:.4f} + "
                      f"L1 {acc['g_l1'] / acc['n']:.4f} + "
                      f"perc {acc['g_perc'] / acc['n']:.4f} + "
                      f"sem {acc['g_sem'] / acc['n']:.4f})")

        print(f"[epoch {epoch}] D {acc['d'] / acc['n']:.4f} | "
              f"G {acc['g'] / acc['n']:.4f} "
              f"(adv {acc['g_adv'] / acc['n']:.4f} + L1 {acc['g_l1'] / acc['n']:.4f} + "
              f"perc {acc['g_perc'] / acc['n']:.4f} + sem {acc['g_sem'] / acc['n']:.4f}) "
              f"{time.time() - t0:.1f}s")

        # Held-out test evaluation (forward-only, no weight updates) — reports the
        # same L1 translation-loss convention used in training.
        if eval_loader is not None:
            gen.eval()
            test_l1, nt = 0.0, 0
            with torch.no_grad():
                for sar, rgb in eval_loader:
                    sar, rgb = sar.to(device), rgb.to(device)
                    test_l1 += l1(gen(sar), rgb).item() * sar.size(0)
                    nt += sar.size(0)
            gen.train()
            current_test_l1 = test_l1 / max(nt, 1)
            print(f"[eval] epoch {epoch} held-out test L1 = "
                  f"{current_test_l1:.4f}  (n={nt})")
            if best_path is not None and current_test_l1 < best_test_l1:
                best_test_l1 = current_test_l1
                best_epoch = epoch
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(gen.state_dict(), best_path)
                print(f"[best] epoch {epoch} test L1 = {current_test_l1:.4f} "
                      f"-> saved {best_path.name}")

        # Always write the seed-specific "latest" checkpoint (resumable), and
        # additionally snap an epoch-numbered copy when saving past a boundary.
        if epoch % save_every == 0 or epoch == n_epochs:
            save_checkpoint(config, epoch, gen, disc, opt_g, opt_d, latest_path)
            snapped = ckpt_dir / f"{base_name}_epoch_{epoch:04d}.pt"
            save_checkpoint(config, epoch, gen, disc, opt_g, opt_d, snapped)

    # Save the final generator weights to the requested path.
    out = Path(output_checkpoint_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), out)
    print(f"[save] final generator weights -> {out}")
    if best_path is not None:
        print(f"Best epoch: {best_epoch}")
        print(f"Best held-out test L1: {best_test_l1:.4f}")
        print(f"Best checkpoint: {best_path}")

    return gen


def main() -> int:
    ap = argparse.ArgumentParser(description="Pix2Pix training (dual-branch).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (default: dataset.random_seed from config)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.num_epochs from config (for testing)")
    ap.add_argument("--output", default=None,
                    help="final generator weights path (default: <checkpoint_dir>/generator_final.pt)")
    ap.add_argument("--local-copy", action="store_true",
                    help="read training data from the local dataset copy "
                         "(dataset.local_dataset_path) instead of dataset.path (Drive). "
                         "Checkpoints still save to checkpoint_dir (Drive).")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    if args.seed is not None:
        cfg.dataset.random_seed = args.seed
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    output = args.output or str(Path(cfg.paths.checkpoint_dir) / "generator_final.pt")
    if args.local_copy:
        cfg.dataset.path = local_dataset_dir(cfg)
        print(f"  reading data from local copy: {cfg.dataset.path!r} "
              f"(checkpoints still -> {cfg.paths.checkpoint_dir})")

    train_generator(seed=cfg.dataset.random_seed, config=cfg,
                    output_checkpoint_path=output)
    return 0


if __name__ == "__main__":
    sys.exit(main())