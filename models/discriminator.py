"""PatchGAN discriminator for SAR-to-RGB mapping.

The discriminator takes a paired (SAR, RGB) image, concatenates them along
the channel dimension, and downsamples through 4 conv blocks to produce a
patch-level output map: every receptive field in the output scores whether
that local patch is real or fake. A patch-based judgement (rather than a
single scalar) keeps the discriminator focused on local texture / high
frequency detail, which complements the encoder-fusion generator.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _init_weights(m: nn.Module) -> None:
    """Normal weight init (mean 0, std 0.02) + zero biases, as per DCGAN-style
    discriminators. Applied recursively to the whole module."""
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class PatchGANDiscriminator(nn.Module):
    """Concatenates SAR + RGB along the channel axis and judges local patches."""

    def __init__(self, sar_channels: int = 3, rgb_channels: int = 3,
                 base_channels: int = 64) -> None:
        super().__init__()

        # IMPORTANT: the first conv takes the concatenated pair on the channel
        # axis. For our 3-channel SAR + 3-channel RGB setup that is
        #   sar_channels + rgb_channels = 3 + 3 = 6,
        # NOT 4. Keep these in sync with the actual channel counts of the two
        # inputs so the first conv's in_channels matches what forward() feeds it.
        first_in = sar_channels + rgb_channels

        blocks = [nn.Conv2d(first_in, base_channels, kernel_size=4, stride=2,
                            padding=1)]
        # First block: no BatchNorm (standard PatchGAN practice).
        blocks.append(nn.LeakyReLU(0.2, inplace=True))

        in_ch = base_channels
        for i in range(1, 4):  # 3 more downsampling blocks, all with BatchNorm
            out_ch = base_channels * (2 ** i)
            blocks += [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            in_ch = out_ch

        # Final conv -> single-channel patch-level output map (no BatchNorm,
        # no activation; raw logits go to the GAN loss).
        blocks.append(nn.Conv2d(in_ch, 1, kernel_size=4, stride=1, padding=1))

        self.model = nn.Sequential(*blocks)
        self.apply(_init_weights)

    def forward(self, sar: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
        """Concatenate (sar, rgb) along the channel dim and produce the patch map."""
        x = torch.cat([sar, rgb], dim=1)  # (B, sar_channels + rgb_channels, H, W)
        return self.model(x)


if __name__ == "__main__":
    torch.manual_seed(0)

    d = PatchGANDiscriminator()
    sar = torch.randn(2, 3, 256, 256)
    rgb = torch.randn(2, 3, 256, 256)
    out = d(sar, rgb)

    # Sanity: first conv must have taken 6 (not 4) input channels.
    assert d.model[0].in_channels == 6, (
        f"first conv expects {d.model[0].in_channels} channels; should be "
        "sar_channels(3) + rgb_channels(3) = 6"
    )

    # Assertions per spec: batch size 2, single output channel.
    assert out.shape[0] == 2, f"batch size = {out.shape[0]}, expected 2"
    assert out.shape[1] == 1, f"output channels = {out.shape[1]}, expected 1"

    print(f"input sar  : {tuple(sar.shape)}")
    print(f"input rgb  : {tuple(rgb.shape)}")
    print(f"patch map  : {tuple(out.shape)}")
    print("first conv in_channels =", d.model[0].in_channels)
    print("\nOK: patch-level output map, batch 2, 1 channel.")