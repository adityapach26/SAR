"""Generator for SAR-to-RGB translation.

The generator is the two-branch encoder + attention-fusion network from the
Book. ``Generator`` internally builds:

  * ``SARFeatureEncoder``         -- general-purpose branch
  * ``SARPhysicsTextureEncoder``  -- physics/texture branch (same input, narrower)
  * ``MultiScaleAttentionFusion`` -- learns a per-pixel blend at each scale

Both encoders read the *same* SAR input; the fusion learns how much each
scale/location should trust the general vs. texture representation. The fused
maps are then decoded back to a full-resolution RGB image. The decoder here is
a straightforward upsampling U-Net-style path (skips = fused maps) and is
intentionally lightweight -- it can be swapped for a richer decoder later.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoders import SARFeatureEncoder, SARPhysicsTextureEncoder
from .fusion import MultiScaleAttentionFusion

_NUM_LEVELS = 5


class _Up(nn.Module):
    """Bilinear 2x upsampling + conv (no skip)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class _UpCat(nn.Module):
    """Conv block applied after concatenating a fused skip connection."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Generator(nn.Module):
    """Two-branch encoder -> attention fusion -> decoder (SAR in, RGB out)."""

    def __init__(self, sar_channels: int = 3, rgb_channels: int = 3,
                 base_channels: int = 64, fuse_channels: int = 256) -> None:
        super().__init__()
        self.sar_channels = sar_channels
        self.rgb_channels = rgb_channels

        # The two branches read the same SAR input.
        self.enc_general = SARFeatureEncoder(sar_channels, base_channels)
        self.enc_physics = SARPhysicsTextureEncoder(sar_channels, base_channels)

        # Per-scale channel counts (must match the encoders' doubling schedule).
        general_channels = [base_channels * (2 ** i) for i in range(_NUM_LEVELS)]
        physics_channels = [(base_channels // 2) * (2 ** i) for i in range(_NUM_LEVELS)]
        self.fusion = MultiScaleAttentionFusion(general_channels, physics_channels,
                                                fuse_channels=fuse_channels)

        # Decoder: from coarsest fused map (8x8) back up to full resolution.
        up_blocks, cat_blocks = nn.ModuleList(), nn.ModuleList()
        cur = fuse_channels
        for _ in range(_NUM_LEVELS - 1):  # 8->16->32->64->128
            up_blocks.append(_Up(cur, cur // 2))
            cat_blocks.append(_UpCat(cur // 2 + fuse_channels, cur // 2))
            cur //= 2
        self.up_blocks = up_blocks
        self.cat_blocks = cat_blocks

        # 128 -> 256, then to RGB channels.
        self.final = nn.Sequential(
            _Up(cur, cur),
            nn.Conv2d(cur, rgb_channels, kernel_size=3, padding=1),
        )

    def encode(self, sar: torch.Tensor) -> tuple[list[torch.Tensor],
                                                 list[torch.Tensor],
                                                 list[torch.Tensor]]:
        """Run the two branches + fusion, exposing every intermediate list."""
        general = self.enc_general(sar)
        physics = self.enc_physics(sar)
        fused = self.fusion(general, physics)
        return general, physics, fused

    def forward(self, sar: torch.Tensor) -> torch.Tensor:
        _, _, fused = self.encode(sar)
        rev = fused[::-1]          # coarsest -> finest
        x = rev[0]                 # coarsest fused map (8x8)
        for up, cat, skip in zip(self.up_blocks, self.cat_blocks, rev[1:]):
            x = up(x)
            x = cat(torch.cat([x, skip], dim=1))
        return self.final(x)


if __name__ == "__main__":
    torch.manual_seed(0)
    g = Generator(sar_channels=3, rgb_channels=3)
    sar = torch.randn(2, 3, 256, 256)
    general, physics, fused = g.encode(sar)
    rgb = g(sar)
    for i in range(_NUM_LEVELS):
        print(f"level {i}: general {tuple(general[i].shape)} | "
              f"physics {tuple(physics[i].shape)} | fused {tuple(fused[i].shape)}")
    print("fake RGB :", tuple(rgb.shape))
    assert rgb.shape == (2, 3, 256, 256), rgb.shape
    print("OK: generator produces full-res RGB.")