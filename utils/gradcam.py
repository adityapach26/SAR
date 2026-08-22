"""Grad-CAM visualizations (Phase 7).

Hand-rolled Grad-CAM via PyTorch hooks (no ``pytorch-grad-cam`` dependency, so
it runs anywhere torch does, including Colab).

* ``detector_gradcam``   — heatmap of one detector *confidence* w.r.t. the last
  conv layer of a Faster R-CNN. The gradient target is the score of the
  highest-confidence detection, so the heatmap shows which image region most
  drove that detection.
* ``generator_gradcam``  — heatmap of the generator's *output* activations in a
  chosen region w.r.t. a conv layer; the target is the sum/L2-norm of the output
  in that region. Highlights the structured SAR regions the decoder leaned on.

Both return a ``(H, W)`` float heatmap in ``[0, 1]`` and can save an overlay to
``outputs/`` (``paths.output_dir`` from config if available, else local
``outputs/``).

NOTE: the visual checks below are only meaningful with *trained* checkpoints
(the real ones live on Drive at ``paths.checkpoint_dir``); an untrained model
still exercises the code path but its heatmap won't focus on a real object.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _resolve_layer(model, target_layer):
    """Accept a Module or a dot-path string like 'backbone.body.layer4[-1]'."""
    if isinstance(target_layer, torch.nn.Module):
        return target_layer
    obj = model
    for piece in target_layer.split("."):
        if piece.startswith("[") and piece.endswith("]"):
            obj = obj[int(piece[1:-1])]
        else:
            obj = getattr(obj, piece)
    return obj


class _GradCam:
    """Capture a conv layer's forward activations and its gradients."""

    def __init__(self, layer: torch.nn.Module) -> None:
        self.acts = None
        self.grads = None
        self._fh = layer.register_forward_hook(self._forward)
        self._bh = layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inp, out):
        self.acts = out  # (B, C, H', W')

    def _backward(self, module, grad_in, grad_out):
        self.grads = grad_out[0]  # gradient w.r.t. the conv output

    def remove(self):
        self._fh.remove()
        self._bh.remove()


def _heatmap(cam: _GradCam, spatial_size: tuple[int, int]) -> torch.Tensor:
    """Grad-CAM: channel-weighted average of activations x gradients, then ReLU+norm.

    ``spatial_size`` is the (H, W) of the input the caller provided, so the
    coarse conv-level map is bilinearly upsampled to it before normalizing.
    """
    acts = cam.acts.detach()
    grads = cam.grads.detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)       # (C, 1, 1) per-channel importance
    cam_map = (weights * acts).sum(dim=1, keepdim=True)  # (B, 1, H', W')
    up = F.interpolate(F.relu(cam_map), size=spatial_size,
                       mode="bilinear", align_corners=False)[0, 0]  # (H, W)
    n = up - up.min()
    denom = up.max() - up.min() + 1e-8
    return n / denom


def _default_detector_layer(model):
    """Last conv of the ResNet-50 body (C5), the standard Grad-CAM target."""
    for m in reversed(list(model.backbone.body.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise AttributeError("no conv layer found in detector backbone.body")


def _default_generator_layer(model):
    """Falling target: the last Conv2d in the generator."""
    for m in reversed(list(model.modules())):
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise AttributeError("no conv layer found in generator")


def _output_dir(config=None) -> Path:
    """Prefer ``paths.output_dir`` from config, else local ``outputs/``."""
    try:
        if config is not None:
            return Path(config.paths.output_dir)
    except Exception:
        pass
    return ROOT / "outputs"


def _save_overlay(img_01, heat, out_path, title="grad-cam"):
    """Save a matplotlib overlay of the heatmap on the image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if img_01.shape[0] == 1:
        show = np.repeat(img_01[0], 3, axis=0).transpose(1, 2, 0)
    else:
        show = np.transpose(img_01[:3] if img_01.shape[0] >= 3 else img_01, (1, 2, 0))
    show = np.clip(show, 0, 1)

    # Up-sample the heatmap to image resolution if needed.
    hmap = heat
    if hmap.shape != show.shape[:2]:
        hmap = F.interpolate(torch.from_numpy(hmap)[None, None].float(),
                             size=show.shape[:2], mode="bilinear",
                             align_corners=False)[0, 0].numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    ax1.imshow(show)
    ax1.axis("off")
    ax1.set_title("input")
    ax2.imshow(show, alpha=0.6)
    im = ax2.imshow(hmap, cmap="jet", alpha=0.5)
    ax2.axis("off")
    ax2.set_title(title)
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def detector_gradcam(detector_model, image_tensor, target_layer=None, save=True,
                     output_name="detector_gradcam", config=None, device=None):
    """Grad-CAM heatmap of the highest-confidence detection.

    Returns
    -------
    (heatmap, best_score)
        heatmap   : (H, W) float tensor in [0,1].
        best_score: the confidence whose gradient drove the heatmap.
    """
    if device is None:
        device = next(detector_model.parameters()).device
    if target_layer is None:
        target_layer = _default_detector_layer(detector_model)
    layer = _resolve_layer(detector_model, target_layer)

    detector_model.eval()
    img = image_tensor.detach().to(device).requires_grad_(True) if not image_tensor.requires_grad \
        else image_tensor.to(device)

    # First, find the highest-confidence detection (no grad).
    with torch.no_grad():
        out = detector_model([img])[0]
        scores = out["scores"]
        if scores.numel() == 0:
            raise RuntimeError("detector found no detections — cannot run Grad-CAM")
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])

    cam = _GradCam(layer)
    # Second forward with grad enabled, backprop through that score only.
    restarted = detector_model([img])[0]
    detector_model.zero_grad()
    restarted["scores"][best_idx].backward()
    heat = _heatmap(cam, tuple(img.shape[-2:])).to("cpu")
    cam.remove()

    if save:
        out_dir = _output_dir(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        img01 = (img.detach().cpu() + 1.0) / 2.0 if img.detach().cpu().min() < 0 else img.detach().cpu()
        _save_overlay(img01, heat.numpy(), out_dir / f"{output_name}.png",
                      title=f"detector Grad-CAM (score {best_score:.3f})")

    return heat, best_score


def generator_gradcam(generator_model, sar_tensor, target_layer=None, region=None,
                      norm: str = "sum", save=True, output_name="generator_gradcam",
                      config=None, device=None):
    """Grad-CAM for the generator: target = sum/L2 of output activations in ``region``.

    Parameters
    ----------
    region : tuple (y0, y1, x0, x1) | None
        Spatial box of the output to target. `None` = whole image.
    norm : "sum" | "l2"
        Aggregation over the region's output activations.

    Returns the (H, W) heatmap in [0, 1].
    """
    if device is None:
        device = next(generator_model.parameters()).device
    if target_layer is None:
        target_layer = _default_generator_layer(generator_model)
    layer = _resolve_layer(generator_model, target_layer)

    generator_model.eval()
    sar = sar_tensor.detach().to(device)
    sar_ = sar.unsqueeze(0) if sar.dim() == 3 else sar
    sar_.requires_grad_(True)

    cam = _GradCam(layer)
    out = generator_model(sar_)          # (1, C, H, W)
    if region is None:
        target = out
    else:
        y0, y1, x0, x1 = region
        target = out[:, :, y0:y1, x0:x1]
    if norm == "l2":
        loss = target.pow(2).sum()
    else:  # "sum"
        loss = target.sum()
    generator_model.zero_grad()
    loss.backward()
    heat = _heatmap(cam, tuple(sar_.shape[-2:])).to("cpu")
    cam.remove()

    if save:
        out_dir = _output_dir(config)
        out_dir.mkdir(parents=True, exist_ok=True)
        img01 = (sar_.detach().cpu().squeeze(0) + 1.0) / 2.0
        _save_overlay(img01, heat.numpy(), out_dir / f"{output_name}.png",
                      title=f"generator Grad-CAM ({norm})")

    return heat


if __name__ == "__main__":
    torch.manual_seed(0)

    # ---- 7.2 generator Grad-CAM first (fast) ----
    from models.generator import Generator
    g = Generator(sar_channels=3, rgb_channels=3, base_channels=32).eval()
    sar_test = torch.randn(1, 3, 128, 128)
    hg = generator_gradcam(g, sar_test, region=(32, 96, 32, 96), save=True,
                           output_name="generator_gradcam_smoke")
    print(f"generator_gradcam heatmap: shape={tuple(hg.shape)} "
          f"min={hg.min():.4f} max={hg.max():.4f} mean={hg.mean():.4f}")

    # ---- 7.1 detector Grad-CAM (untrained model to validate the path) ----
    print("\nNote: using an UNTRAINED detector; real checkpoints are on Drive.")
    num_classes = 2
    import torchvision
    try:
        det = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1)
    except (TypeError, AttributeError):
        det = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_f = det.roi_heads.box_predictor.cls_score.in_features
    det.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_f, num_classes)
    det.eval()

    # synthetic image with a strong bright object to give the detector something to see
    img = torch.rand(1, 3, 256, 256) * 0.3
    img[:, :, 100:170, 110:180] = 0.9
    try:
        hd, sc = detector_gradcam(det, img[0], save=True, output_name="detector_gradcam_smoke")
        print(f"detector_gradcam: shape={tuple(hd.shape)} max={hd.max():.4f} "
              f"score={sc:.4f}")
        # crude concentration check: heatmap peak should be in the upper half (near blob)
        hd_np = hd.numpy()
        peak_y, peak_x = np.unravel_index(hd_np.argmax(), hd_np.shape)
        inside_blob = (100 <= peak_y <= 180) and (110 <= peak_x <= 190)
        print(f"  heatmap peak at ({peak_y},{peak_x}) inside bright object: {inside_blob}")
    except RuntimeError as e:
        print("detector_gradcam skipped:", e)

    print("OK: saved overlays to outputs/ (visual CHECK via colab with trained weights).")