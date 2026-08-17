"""Phase 2 smoke test: wires the two-branch Generator + PatchGAN Discriminator
end to end and prints every intermediate shape (including each encoder branch
and the fused maps at every scale), not just the final outputs.

Run from the repository root:

    python scripts/smoke_test_phase2.py

Exits PASS/FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow `models` / `utils` imports when run as a script.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.generator import Generator          # noqa: E402
from models.discriminator import PatchGANDiscriminator  # noqa: E402
from utils.config_loader import load_config      # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))

    # Channels from config (fall back to 3 for SAR/RGB if not specified).
    sar_channels = getattr(cfg.input_channels, "num_channels", 3)
    rgb_channels = 3
    base_channels = cfg.model.generator.base_channels
    image_size = cfg.dataset.image_size
    batch = cfg.train.batch_size

    print(f"config: sar={sar_channels} rgb={rgb_channels} base={base_channels} "
          f"image={image_size}x{image_size} batch={batch}\n")

    g = Generator(sar_channels=sar_channels, rgb_channels=rgb_channels,
                  base_channels=base_channels)
    d = PatchGANDiscriminator(sar_channels=sar_channels, rgb_channels=rgb_channels,
                              base_channels=base_channels)

    # Dummy SAR batch.
    sar = torch.randn(batch, sar_channels, image_size, image_size)
    print(f"{'input SAR':<22}: {tuple(sar.shape)}")

    # --- Generator forward, exposing the two branches + fused maps per scale.
    general, physics, fused = g.encode(sar)
    fake_rgb = g(sar)
    print(f"{'fake RGB':<22}: {tuple(fake_rgb.shape)}")
    print("  per-scale shapes (finest -> coarsest):")
    for i in range(len(fused)):
        print(f"    level {i}: general {str(tuple(general[i].shape)):<22} "
              f"physics {str(tuple(physics[i].shape)):<22} fused {tuple(fused[i].shape)}")
        ok = (general[i].shape[-2:] == physics[i].shape[-2:] == fused[i].shape[-2:])
        if not ok:
            print(f"    !! spatial mismatch at level {i}")
            return 1

    # --- Discriminator on (real SAR geometry, generated RGB).
    d_sar = torch.randn(batch, sar_channels, image_size, image_size)
    d_out = d(d_sar, fake_rgb)
    print(f"{'D(SAR, fakeRGB)':<22}: {tuple(d_out.shape)}  (patch map, not scalar)")

    # --- Final checks.
    expect_rgb = (batch, rgb_channels, image_size, image_size)
    if fake_rgb.shape != expect_rgb:
        print(f"\nFAIL: fake RGB shape {tuple(fake_rgb.shape)} != {expect_rgb}")
        return 1
    if len(fused) != len(general) or len(fused) != len(physics):
        print("\nFAIL: fused / branch scale counts disagree")
        return 1
    if d_out.dim() != 4 or d_out.shape[1] != 1 or d_out.shape[0] != batch:
        print(f"\nFAIL: discriminator output malformed: {tuple(d_out.shape)}")
        return 1

    print("\nPASS: encoders -> fusion -> generator -> discriminator all wired and sized correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())