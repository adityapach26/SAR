"""Pipeline: run the trained generator (or an ensemble of them) on SAR input (Phase 6, Step 6.1).

``run_generator_ensemble`` loads the M generator checkpoints saved by
:mod:`scripts.train_ensemble` (``generator_seed{seed}.pt``), runs the SAME SAR
input through each member in eval/no_grad, and returns:

  * ``mean_rgb``     -- (B, C, H, W) ensemble average RGB prediction.
  * ``variance_map`` -- (B, H, W) per-pixel variance across the M outputs:

        var(x, y) = mean_m ( pred_m(x, y) - mean_pred(x, y) )^2

    Each member's prediction is an RGB image; channels are collapsed first so
    every spatial location has one value, exactly as the guide describes.

Checkpoints are raw ``state_dict``s (``torch.save(gen.state_dict(), out)`` in
:func:`scripts.train.train_generator`). ``num_members`` controls how many
``generator_seed{seed}.pt`` files are loaded, using ``ensemble.seeds`` (from
config) as the seed order.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402
from models.generator import Generator  # noqa: E402


def generator_from_config(config, device=None):
    """Build a fresh Generator matching how the ensemble trained them."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Generator(
        sar_channels=int(config.input_channels.num_channels),
        rgb_channels=3,
        base_channels=int(config.model.generator.base_channels),
    ).to(device)


@torch.no_grad()
def _run_members(sar_input: torch.Tensor, ckpt_paths: list[Path], config):
    """Forward ``sar_input`` through each checkpoint; return stats tensors.

    Returns (stacked, collapsed, variance_map):
        stacked       : (M, B, C, H, W)
        collapsed     : (M, B, H, W)  — channels averaged per member
        variance_map  : (B, H, W)     — per-pixel variance across members
    """
    outs = []
    for ck in ckpt_paths:
        gen = generator_from_config(config).eval()
        device = next(gen.parameters()).device
        gen.load_state_dict(torch.load(ck, map_location=device))
        outs.append(gen(sar_input))
    stacked = torch.stack(outs, dim=0)          # (M, B, C, H, W)
    collapsed = stacked.mean(dim=2)             # (M, B, H, W)
    variance_map = torch.mean(
        (collapsed - collapsed.mean(dim=0, keepdim=True)) ** 2, dim=0
    )
    return stacked, collapsed, variance_map


@torch.no_grad()
def run_generator_ensemble(sar_input: torch.Tensor, checkpoint_dir, num_members: int,
                           seeds=None, config=None):
    """Run ``sar_input`` through ``num_members`` generator checkpoints.

    Returns
    -------
    (mean_rgb, variance_map)
        mean_rgb      : Tensor (B, C, H, W) — ensemble average RGB prediction.
        variance_map  : Tensor (B, H, W) — one value per pixel.
    """
    if config is None:
        config = load_config(str(ROOT / "configs" / "config.yaml"))
    if seeds is None:
        seeds = list(config.ensemble.seeds)
    ckpt_dir = Path(checkpoint_dir)

    member_seeds = seeds[:num_members]
    if len(member_seeds) < num_members:
        raise ValueError(
            f"num_members={num_members} but only {len(seeds)} seeds available"
        )

    paths = [ckpt_dir / f"generator_seed{s}.pt" for s in member_seeds]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"generator checkpoint(s) not found: {missing}")

    stacked, _, variance_map = _run_members(sar_input, paths, config)
    mean_rgb = stacked.mean(dim=0)  # (B, C, H, W)
    return mean_rgb, variance_map


def _make_dummy_sar(config, batch=1, size=256):
    c = int(config.input_channels.num_channels)
    return torch.randn(batch, c, size, size)


def _find_checkpoint_dir(config) -> Path:
    """Prefer config's Drive checkpoint_dir; fall back to local checkpoints/."""
    for cand in (Path(config.paths.checkpoint_dir), ROOT / "checkpoints"):
        if (cand / "generator_seed1.pt").exists():
            return cand
    return Path(config.paths.checkpoint_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description="Generator ensemble: mean RGB + per-pixel variance.")
    ap.add_argument("--checkpoint-dir", default=None,
                    help="directory with generator_seed{seed}.pt (default: paths.checkpoint_dir)")
    ap.add_argument("--members", type=int, default=None,
                    help="number of members (default: ensemble.num_members)")
    ap.add_argument("--size", type=int, default=256, help="dummy SAR image side (squared)")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else _find_checkpoint_dir(cfg)
    num_members = args.members or int(cfg.ensemble.num_members)
    seeds = list(cfg.ensemble.seeds)
    print(f"checkpoint_dir={ckpt_dir}  members={num_members}  seeds={seeds}")

    sar = _make_dummy_sar(cfg, size=args.size)

    # (a) run the full function with the real checkpoint dir.
    real_paths = [ckpt_dir / f"generator_seed{s}.pt" for s in seeds[:num_members]]
    have_real = all(p.exists() for p in real_paths)
    if have_real:
        mean_rgb, var_map = run_generator_ensemble(sar, ckpt_dir, num_members,
                                                   seeds=seeds, config=cfg)
        g = generator_from_config(cfg)
        single_shape = g(sar[:1]).shape  # (B, C, H, W) from a single member
        print(f"\n[a] run_generator_ensemble -> mean {tuple(mean_rgb.shape)}  "
              f"var {tuple(var_map.shape)}")
        # (b) mean_rgb matches a single generator's output shape; var is per-pixel.
        ok_shape = tuple(mean_rgb.shape) == tuple(single_shape)
        ok_pixel = tuple(var_map.shape) == torch.Size((1, args.size, args.size))
        print("    (b) mean_rgb shape matches single output: ", "PASS" if ok_shape else
              f"FAIL ({tuple(mean_rgb.shape)} vs {tuple(single_shape)})")
        print(f"    (b) variance_map is per-pixel (1,{args.size},{args.size}): ",
              "PASS" if ok_pixel else f"FAIL ({tuple(var_map.shape)})")
    else:
        missing = [str(p) for p in real_paths if not p.exists()]
        print(f"\n[a/b] SKIPPED — need checkpoints in {ckpt_dir}; missing {missing}")
        return 0

    # (c) CRITICAL: same checkpoint used 3x => variance EXACTLY zero everywhere.
    print("\n[c] CRITICAL zero-variance check (same checkpoint repeated 3x):")
    same = [real_paths[0]] * 3
    _, _, var_same = _run_members(sar, same, cfg)
    exact = bool((var_same == 0).all())
    print(f"    max variance = {var_same.max().item():.6g}")
    print("    PASS: variance is EXACTLY zero (identical deterministic weights)"
          if exact else "    FAIL: expected EXACTLY zero variance, got nonzero")

    # (e) real distinct checkpoints => nonzero, spatially varying variance.
    print("\n[e] real distinct generator_seed1/2/3 variance:")
    _, _, var_real = _run_members(sar, real_paths, cfg)
    nz = int((var_real != 0).sum())
    print(f"    variance_map: min={var_real.min().item():.6g}  "
          f"max={var_real.max().item():.6g}  mean={var_real.mean().item():.6g}  "
          f"non-zero={nz}/{var_real.numel()}")
    ok = bool(nz > 0)
    print("    PASS: variance non-zero and spatially varying" if ok
          else "    FAIL: variance everywhere zero on distinct checkpoints")
    return 0


if __name__ == "__main__":
    sys.exit(main())