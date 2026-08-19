"""Pix2Pix training loop for the dual-branch SAR-to-RGB generator.

Standard GAN objective:
  * Discriminator is trained to tell real vs. fake (fake) (SAR, RGB) pairs apart
    via BCELogitsLoss over the PatchGAN map.
  * Generator is trained to fool the discriminator (adversarial term) and to
    match the ground-truth RGB with an L1 loss weighted by ``lambda_l1``.
      G_loss = lambda_gan * BCE(D(G(sar)), real) + lambda_l1 * |G(sar) - rgb|_1

All hyperparameters are read from configs/config.yaml. The adversarial (real)
label is a ones-tensor matching the discriminator's patch-map shape.

Run:  python scripts/train.py [--epochs N] [--device cpu|cuda]
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

from models.generator import Generator            # noqa: E402
from models.discriminator import PatchGANDiscriminator  # noqa: E402
from utils.config_loader import load_config        # noqa: E402
from data.dataset import build_pairs, SEN12Dataset  # noqa: E402
from losses.perceptual_loss import PerceptualLoss  # noqa: E402
from losses.semantic_loss import SemanticLoss  # noqa: E402


def count_scenes(dataset_path: Path) -> int:
    """Number of SEN1-2 scenes (max s1_i index + 1)."""
    indices = [int(d.name[3:]) for d in Path(dataset_path).iterdir()
               if d.is_dir() and d.name.startswith("s1_")]
    return (max(indices) + 1) if indices else 0


def make_loaders(cfg):
    num_scenes = count_scenes(cfg.dataset.path)
    if num_scenes == 0:
        raise SystemExit(f"no s1_* scenes found under {cfg.dataset.path!r}")
    pairs, _mismatches = build_pairs(cfg.dataset.path, num_scenes=num_scenes)
    if not pairs:
        raise SystemExit(f"no matching image pairs under {cfg.dataset.path!r}")
    num_channels = cfg.input_channels.num_channels
    ds = SEN12Dataset(pairs, num_channels=num_channels)
    # Parallel background workers keep image loading off the GPU/math critical
    # path; pin_memory makes host->device copies faster on CUDA systems.
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=True,
                        num_workers=cfg.train.num_workers, pin_memory=True,
                        drop_last=False)
    return loader, len(ds)


def save_checkpoint(cfg, epoch, gen, disc, opt_g, opt_d, path):
    torch.save({
        "epoch": epoch,
        "generator_state_dict": gen.state_dict(),
        "discriminator_state_dict": disc.state_dict(),
        "optimizer_g_state_dict": opt_g.state_dict(),
        "optimizer_d_state_dict": opt_d.state_dict(),
    }, path)
    print(f"[save] checkpoint -> {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pix2Pix training (dual-branch).")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.num_epochs from config (for testing)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg.dataset.random_seed)

    n_epochs = args.epochs or cfg.train.num_epochs
    base_g = cfg.model.generator.base_channels
    base_d = cfg.model.discriminator.base_channels
    sar_channels = cfg.input_channels.num_channels
    # PyYAML resolves the bare exponent "2e-4" to a string; coerce to float.
    lr = float(cfg.train.learning_rate)
    betas = (cfg.train.beta1, cfg.train.beta2)
    lambda_gan, lambda_l1 = cfg.loss.lambda_gan, cfg.loss.lambda_l1
    lambda_perc = cfg.loss.lambda_perceptual
    lambda_sem = cfg.loss.lambda_semantic
    log_every = cfg.train.log_every_n_batches
    save_every = cfg.train.save_every_n_epochs
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    gen = Generator(sar_channels=sar_channels, rgb_channels=3, base_channels=base_g).to(device)
    disc = PatchGANDiscriminator(sar_channels=sar_channels, rgb_channels=3,
                                 base_channels=base_d).to(device)

    # Only build the (heavy, frozen) auxiliary losses when they are enabled.
    perceptual = PerceptualLoss().to(device) if lambda_perc > 0 else None
    semantic = SemanticLoss().to(device) if lambda_sem > 0 else None

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=betas)
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=betas)

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    loader, n_samples = make_loaders(cfg)
    print(f"device={device}  samples={n_samples}  scenes={count_scenes(cfg.dataset.path)}  "
          f"epochs={n_epochs}  batch_size={cfg.train.batch_size}\n")

    for epoch in range(1, n_epochs + 1):
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

        if epoch % save_every == 0 or epoch == n_epochs:
            path = ckpt_dir / f"epoch_{epoch:04d}.pt"
            save_checkpoint(cfg, epoch, gen, disc, opt_g, opt_d, path)

    print("\nTraining complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())