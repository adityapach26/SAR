"""Perceptual loss on a frozen pretrained VGG16.

Compares generated vs. target activations (L1) at a few mid-level layers of a
pretrained VGG16. The network is frozen (no gradients to its weights). Because
the generator's output is in [-1, 1], inputs are first brought into the
ImageNet-normalized range that VGG expects before forward passes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

# Mid-level VGG16.features indices whose activations form the perceptual loss:
#   relu2_2, relu3_3, relu4_2  (spread across low/mid-level texture + structure).
DEFAULT_LAYERS = (8, 15, 20)
DEFAULT_WEIGHTS = (1.0, 1.0, 1.0)


class PerceptualLoss(nn.Module):
    def __init__(self, layer_indices=DEFAULT_LAYERS, weights=DEFAULT_WEIGHTS) -> None:
        super().__init__()
        features = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features
        features.eval()
        for p in features.parameters():
            p.requires_grad_(False)  # freeze; never updated during G training
        self.features = features
        self.layer_indices = layer_indices
        self.weights = weights

        # ImageNet mean/std (per channel), broadcast over (B, C, H, W).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def to_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> ImageNet-normalized range: ((x+1)/2 - mean) / std."""
        return ((x + 1.0) / 2.0 - self.mean) / self.std

    def _extract(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run x through VGG features, returning activations at target layers."""
        acts: list[torch.Tensor] = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                acts.append(x)
        return acts

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """L1 between pred/target VGG activations, weighted per layer."""
        pred_a, tgt_a = self._extract(self.to_imagenet(pred)), self._extract(self.to_imagenet(target))
        total = pred_a[0].new_zeros(())
        for w, pa, ta in zip(self.weights, pred_a, tgt_a):
            total = total + w * F.l1_loss(pa, ta)
        return total


if __name__ == "__main__":
    torch.manual_seed(0)
    ploss = PerceptualLoss()
    pred = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    target = torch.randn(2, 3, 256, 256).clamp(-1, 1)
    val = ploss(pred, target).item()
    print(f"perceptual loss = {val:.6f}")
    assert torch.isfinite(torch.tensor(val)), "loss is not finite"
    assert val > 0, "loss should be positive (distinct random inputs)"
    print("OK: positive finite perceptual loss.")