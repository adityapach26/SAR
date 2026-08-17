"""Multi-scale attention fusion for SAR encoder branches.

Given the per-scale feature maps from the two encoders defined in
``models.encoders`` (general purpose + physics/texture), this module learns
a *per-pixel blend* at every resolution:

    fused = general * (1 - attn) + physics_texture * attn

The attention map is a learned per-pixel weight in (0, 1): where it is near 1
the texture/physics branch wins, where it is near 0 the general branch wins.
This replaces a fixed concatenation with a trainable combination, so the
decoder receives a single fused feature map per scale while the network
decides, per location and per scale, which representation to trust.

The fused maps are returned as a single list (one per scale) ready to feed a
U-Net-style decoder as skip connections.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .encoders import SARFeatureEncoder, SARPhysicsTextureEncoder
except ImportError:  # direct execution as a script
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.encoders import SARFeatureEncoder, SARPhysicsTextureEncoder


class _ScaleFusion(nn.Module):
    """Fuses one pair of feature maps at a single resolution scale."""

    def __init__(self, general_in: int, physics_in: int, fuse_channels: int) -> None:
        super().__init__()
        # 1. Project both branches to a shared channel count via 1x1 convs.
        self.proj_general = nn.Conv2d(general_in, fuse_channels, kernel_size=1)
        self.proj_physics = nn.Conv2d(physics_in, fuse_channels, kernel_size=1)

        # 2. Spatial attention from the concatenated projection.
        #    Small conv block -> single channel -> sigmoid -> [0, 1] per pixel.
        self.attn = nn.Sequential(
            nn.Conv2d(fuse_channels * 2, fuse_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fuse_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        # Zero-initialize the final conv so the logit starts at 0 -> attn = 0.5.
        nn.init.zeros_(self.attn[-2].weight)
        nn.init.zeros_(self.attn[-2].bias)

    def forward(self, general: torch.Tensor, physics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.proj_general(general)
        p = self.proj_physics(physics)
        attn = self.attn(torch.cat([g, p], dim=1))  # (B, 1, H, W)
        # 3. Learned per-pixel blend.
        fused = g * (1 - attn) + p * attn
        return fused, attn


class MultiScaleAttentionFusion(nn.Module):
    """Fuses the two encoders' feature lists at every scale and returns one
    fused feature map per scale."""

    def __init__(self, general_channels: list[int], physics_channels: list[int],
                 fuse_channels: int = 256) -> None:
        super().__init__()
        assert len(general_channels) == len(physics_channels) > 0, (
            "expected equal, non-empty per-scale channel lists"
        )
        self.scales = nn.ModuleList()
        for gc, pc in zip(general_channels, physics_channels):
            self.scales.append(_ScaleFusion(gc, pc, fuse_channels))
        self.fuse_channels = fuse_channels
        # Attention maps from the most recent forward pass, exposed for
        # debugging / sanity checks (the forward output is fused maps only).
        self.last_attention: list[torch.Tensor] | None = None

    def forward(self, general: list[torch.Tensor],
                physics: list[torch.Tensor]) -> list[torch.Tensor]:
        assert len(general) == len(physics) == len(self.scales), (
            "feature lists must have one entry per configured scale"
        )
        fused: list[torch.Tensor] = []
        attns: list[torch.Tensor] = []
        for g, p, scale in zip(general, physics, self.scales):
            out, attn = scale(g, p)
            fused.append(out)
            attns.append(attn)
        self.last_attention = attns
        return fused


if __name__ == "__main__":
    torch.manual_seed(0)

    for in_channels in (1, 3):
        print(f"=== in_channels={in_channels} ===")
        enc_feat = SARFeatureEncoder(in_channels=in_channels)
        enc_tex = SARPhysicsTextureEncoder(in_channels=in_channels)

        dummy = torch.randn(2, in_channels, 256, 256)  # (B, in_channels, H, W)
        general = enc_feat(dummy)
        physics = enc_tex(dummy)

        general_ch = [f.shape[1] for f in general]
        physics_ch = [f.shape[1] for f in physics]

        fusion = MultiScaleAttentionFusion(general_ch, physics_ch, fuse_channels=128)
        fused = fusion(general, physics)

        assert len(fused) == len(general) == len(physics)
        # Spatial size preserved; only channels change (to fuse_channels).
        assert all(f.shape[-2:] == g.shape[-2:] for f, g in zip(fused, general)), (
            "spatial size changed"
        )
        assert all(f.shape[1] == fusion.fuse_channels for f in fused), "channel mismatch"
        for i, f in enumerate(fused):
            print(f"  level {i}: fused {tuple(f.shape)} (inputs {tuple(general[i].shape)}"
                  f" + {tuple(physics[i].shape)})")

        means = [a.mean().item() for a in fusion.last_attention]
        print("  mean attention:", "  ".join(f"{m:.4f}" for m in means))
        # Attention should start near 0.5 (sigmoid + zero-biased conv), not at extremes.
        assert all(0.2 < m < 0.8 for m in means), f"attention stuck at extremes: {means}"
        print()
