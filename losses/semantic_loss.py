"""Semantic / detection-aware loss for the SAR-to-RGB generator.

PLACEHOLDER — this is not yet detection-aware. It will be replaced with the
real Faster R-CNN backbone's features in Phase 5, Step 5.4
(losses/semantic_loss_v2.py), after the detector is trained. Until that step
runs, this term behaves as a second perceptual loss and does not connect the
translation stage to the detection stage the way the Book's Section 3.3
describes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class SemanticLoss(nn.Module):
    """L1 (or cosine) distance on a late ResNet18 conv-layer activation.

    ResNet18 serves as a frozen detector-proxy: it is pretrained on ImageNet
    and its late features carry high-level scene/semantic content. Comparing
    generated vs. real RGB activations there encourages the translation stage
    to preserve semantics. See the module docstring for why this is a
    placeholder.
    """

    def __init__(self, distance: str = "l1") -> None:
        super().__init__()
        if distance not in ("l1", "cosine"):
            raise ValueError("distance must be 'l1' or 'cosine'")
        self.distance = distance

        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Keep conv1..layer4 (late conv features); drop avgpool + fc.
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)  # freeze; never updated during G training
        self.backbone = backbone

        # ImageNet mean/std (per channel), broadcast over (B, C, H, W).
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    @torch.no_grad()
    def to_imagenet(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> ImageNet-normalized range: ((x+1)/2 - mean) / std."""
        return ((x + 1.0) / 2.0 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        a = self.backbone(self.to_imagenet(pred))
        b = self.backbone(self.to_imagenet(target))
        if self.distance == "cosine":
            return 1.0 - F.cosine_similarity(a, b, dim=1).mean()
        return F.l1_loss(a, b)


if __name__ == "__main__":
    torch.manual_seed(0)
    for dist in ("l1", "cosine"):
        sloss = SemanticLoss(distance=dist)
        pred = torch.randn(2, 3, 256, 256).clamp(-1, 1)
        target = torch.randn(2, 3, 256, 256).clamp(-1, 1)
        val = sloss(pred, target).item()
        print(f"semantic loss ({dist}) = {val:.6f}")
        assert torch.isfinite(torch.tensor(val)), f"{dist}: loss not finite"
        assert val > 0, f"{dist}: loss should be positive (distinct random inputs)"
    print("OK: positive finite semantic loss (both distances).")