"""Detection-aware semantic loss (Phase 5, Step 5.4).

Replaces the ``SemanticLoss`` placeholder (``losses/semantic_loss.py``), which
used a frozen ImageNet ResNet18 as a detector-proxy, with the *actual* trained
ship detector's feature extractor: a FreeRCNN-ResNet50-FPN whose weights come
from a detector checkpoint trained by :mod:`scripts.train_detector`.

The ResNet-50-FPN feature maps that feed the detection head are extracted
(rather than the final classification / box-RoI outputs): those multi-scale
``C2..C5``-derived FPN maps carry the geometry + presence of ships, which is
precisely the semantic signal the generator should preserve.

Construction replicates :func:`scripts.train_detector._build_faster_rcnn`
(num_classes from ``detection.num_classes`` in config, box predictor head
replaced), then loads the checkpoint ``state_dict``, sets eval mode, and
freezes every parameter so only the input gradients flow during G training.

Inputs: generated/real RGB in ``[-1, 1]`` (the generator's output range). They
are shifted to ``[0, 1]`` before the detector backbone, which normalizes to
ImageNet internally — matching how :class:`data.detection_dataset.SSDDataset`
fed the detector at train time.
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402


def _build_faster_rcnn(num_classes: int):
    """Clone of :func:`scripts.train_detector._build_faster_rcnn` (CPU host)."""
    try:  # torchvision >= 0.13
        model = tvm.detection.fasterrcnn_resnet50_fpn(
            weights=tvm.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
    except (TypeError, AttributeError):  # torchvision < 0.13
        model = tvm.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = tvm.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model


class SemanticLossV2(nn.Module):
    """L1/cosine distance between detector FPN features of generated vs. real RGB.

    Parameters
    ----------
    detector_checkpoint_path : str | Path
        Path to a trained detector ``state_dict`` (e.g. ``checkpoints/
        detector_seed1.pt``). Weights are loaded and frozen.
    num_classes : int | None
        Detection head class count. Defaults to ``config.detection.num_classes``
        (2 = background + ship), matching how the detector was trained / built.
    distance : str
        ``"l1"`` or ``"cosine"``. Per-FPN-level distance, summed across levels.
    """

    def __init__(self, detector_checkpoint_path, num_classes: int | None = None,
                 distance: str = "l1") -> None:
        super().__init__()
        if distance not in ("l1", "cosine"):
            raise ValueError("distance must be 'l1' or 'cosine'")
        self.distance = distance

        if num_classes is None:
            cfg = load_config(str(ROOT / "configs" / "config.yaml"))
            num_classes = int(cfg.detection.num_classes)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        detector = _build_faster_rcnn(num_classes).to(device)
        detector.load_state_dict(torch.load(detector_checkpoint_path, map_location=device))
        detector.eval()
        for p in detector.parameters():
            p.requires_grad_(False)  # freeze; never updated during G training
        self.detector = detector

    @torch.no_grad()
    def _to_01(self, x: torch.Tensor) -> torch.Tensor:
        """[-1, 1] -> [0, 1] (the range the detector saw at train time)."""
        return ((x + 1.0) / 2.0).to(self.device)

    def forward(self, generated_rgb: torch.Tensor, real_rgb: torch.Tensor) -> torch.Tensor:
        """Distance between detector FPN features of the two images (finite scalar)."""
        gf = self.detector.backbone(self._to_01(generated_rgb))
        rf = self.detector.backbone(self._to_01(real_rgb))
        total = gf["0"].new_zeros(())
        for key in sorted(gf):
            a, b = gf[key], rf[key]
            if self.distance == "cosine":
                total = total + (1.0 - F.cosine_similarity(a, b, dim=1)).mean()
            else:
                total = total + F.l1_loss(a, b)
        return total


def _find_checkpoint() -> Path:
    """Locate a trained detector checkpoint, preferring the ensemble's Drive path."""
    candidates = [
        ROOT / "checkpoints" / "detector_seed1.pt",
        Path("/content/drive/MyDrive/SAR_Project/checkpoints/detector_seed1.pt"),
    ]
    try:
        cfg = load_config(str(ROOT / "configs" / "config.yaml"))
        candidates.insert(0, Path(cfg.paths.checkpoint_dir) / "detector_seed1.pt")
    except Exception:
        pass
    for c in candidates:
        if c.exists():
            return c
    return None  # not found


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="SemanticLossV2 smoke test (detector FPN features).")
    ap.add_argument("--checkpoint", default=None, help="detector state_dict path")
    ap.add_argument("--size", type=int, default=256, help="image side (squared)")
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not ckpt:
        found = _find_checkpoint()
        ckpt = str(found) if found else None

    if not ckpt or not Path(ckpt).exists():
        print("[!] no trained detector checkpoint found locally (it lives on Drive).")
        print("    Building an UNTRAINED Faster R-CNN of the same architecture to "
              "validate the code path — the loss VALUES will not be the trained "
              "detector's, but finite-and-different is still verifiable.")
        untrained = _build_faster_rcnn(num_classes=2)
        ckpt = str(ROOT / "_tmp_detector.pt")
        torch.save(untrained.state_dict(), ckpt)
        print(f"    wrote placeholder checkpoint -> {ckpt}")

    torch.manual_seed(0)
    size = args.size
    gen = torch.randn(1, 3, size, size).clamp(-1, 1)
    real = torch.randn(1, 3, size, size).clamp(-1, 1)

    v2 = SemanticLossV2(ckpt, distance="l1")
    from losses.semantic_loss import SemanticLoss
    placeholder = SemanticLoss(distance="l1")

    v2_val = v2(gen, real).item()
    ph_val = placeholder(gen, real).item()

    print(f"SemanticLossV2 (detector FPN) = {v2_val:.6f}")
    print(f"SemanticLoss   (placeholder)  = {ph_val:.6f}")
    print(f"[note] checkpoints for the placeholder run: {ckpt}")
    assert torch.isfinite(torch.tensor(v2_val)), "SemanticLossV2 not finite"
    assert v2_val > 0, "SemanticLossV2 should be positive (distinct random inputs)"
    assert torch.isfinite(torch.tensor(ph_val)), "placeholder loss not finite"
    assert ph_val > 0, "placeholder loss should be positive"
    assert v2_val != ph_val, "V2 and placeholder losses should differ"
    print("OK: SemanticLossV2 finite, positive, and different from the placeholder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())