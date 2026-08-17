"""Encoder definitions for SAR imagery.

Two convolutional encoders that share an identical downsampling schedule so
their feature maps can be fused scale-by-scale:

  * SARFeatureEncoder          -- standard channel width, general-purpose features
  * SARPhysicsTextureEncoder   -- half channel width, specialized on the
                                  log-intensity / texture channels

Each returns a list of feature maps, one per resolution, ordered from the
first (finest) downsampling block to the last (coarsest). Because both use
the same stride-2 schedule, the spatial resolutions match at every scale
(only channel counts differ), which is what makes per-scale fusion possible.
"""

from __future__ import annotations

import torch
import torch.nn as nn

_NUM_LEVELS = 5


class SARFeatureEncoder(nn.Module):
    """Standard convolutional encoder producing multi-resolution feature maps.

    Five downsampling blocks halve the spatial resolution at every stage:
    Conv2d(stride=2) -> BatchNorm2d (skipped on the first layer) ->
    LeakyReLU(0.2). Channel width doubles per block, starting at
    ``base_channels``.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(_NUM_LEVELS):
            out_ch = base_channels * (2 ** i)
            self.blocks.append(self._make_block(in_ch, out_ch, kernel_size=3,
                                                use_bn=(i > 0)))
            in_ch = out_ch

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, kernel_size: int, use_bn: bool) -> nn.Sequential:
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=2,
                      padding=kernel_size // 2, bias=not use_bn),
        ]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


class SARPhysicsTextureEncoder(nn.Module):
    """Narrower encoder specialized on the texture / log-intensity channels.

    Identical 5-block downsampling schedule to :class:`SARFeatureEncoder` so
    the two can be fused at every scale, but the channel width is halved
    (``base_channels // 2``). Its first convolution uses a 5x5 kernel for a
    receptive field better suited to capturing local texture; the remaining
    blocks use 3x3 kernels.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        half = base_channels // 2
        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for i in range(_NUM_LEVELS):
            out_ch = half * (2 ** i)
            kernel_size = 5 if i == 0 else 3
            self.blocks.append(SARFeatureEncoder._make_block(
                in_ch, out_ch, kernel_size=kernel_size, use_bn=(i > 0)))
            in_ch = out_ch

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            features.append(x)
        return features


if __name__ == "__main__":
    enc_feat = SARFeatureEncoder()
    enc_tex = SARPhysicsTextureEncoder()

    dummy = torch.randn(2, 3, 256, 256)  # (batch, in_channels, H, W)
    feat_maps = enc_feat(dummy)
    tex_maps = enc_tex(dummy)

    assert len(feat_maps) == _NUM_LEVELS, "SARFeatureEncoder returned wrong number of scales"
    assert len(tex_maps) == _NUM_LEVELS, "SARPhysicsTextureEncoder returned wrong number of scales"

    for i, (fm, tm) in enumerate(zip(feat_maps, tex_maps)):
        print(f"level {i}: SARFeatureEncoder {tuple(fm.shape)} | "
              f"SARPhysicsTextureEncoder {tuple(tm.shape)}")
        assert fm.shape[-2:] == tm.shape[-2:], (
            f"spatial mismatch at level {i}: {tuple(fm.shape)} vs {tuple(tm.shape)}"
        )

    print("\nOK: feature maps match spatially at all 5 resolutions (channels may differ).")
