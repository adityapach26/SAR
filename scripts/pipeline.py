"""Pipeline: run the trained generator (or an ensemble of them) on SAR input.

Phase 6, Step 6.1 -- generator ensemble::

  ``run_generator_ensemble`` loads the M generator checkpoints saved by
  :mod:`scripts.train_ensemble` (``generator_seed{seed}.pt``), runs the SAME SAR
  input through each member in eval/no_grad, and returns:

    * ``mean_rgb``     -- (B, C, H, W) ensemble average RGB prediction.
    * ``variance_map`` -- (B, H, W) per-pixel variance across the M outputs:

          var(x, y) = mean_m ( pred_m(x, y) - mean_pred(x, y) )^2

      Each member's prediction is an RGB image; channels are collapsed first so
      every spatial location has one value, exactly as the guide describes.

Phase 6, Step 6.2 -- detector ensemble::

  ``run_detector_ensemble`` loads the M detector checkpoints saved by
  :mod:`scripts.train_detector_ensemble` (``detector_seed{seed}.pt``), runs the
  same RGB input through each Faster-RCNN member in eval/no_grad, then merges
  the detections. Boxes matched across models by IoU get their confidence
  averaged (the merged score) and the *variance of the member confidences* is
  reported as a detection-level uncertainty. Detections seen by only one model
  are explicitly flagged high-uncertainty (not silently treated as agreed).

Checkpoints are raw ``state_dict``s. ``num_members`` controls how many
``*_seed{seed}.pt`` files are loaded, using ``ensemble.seeds`` (from config).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torchvision

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402
from models.generator import Generator  # noqa: E402


def _default_device() -> torch.device:
    """The device the generator models land on: CUDA if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generator_from_config(config, device=None):
    """Build a fresh Generator matching how the ensemble trained them."""
    if device is None:
        device = _default_device()
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


def detector_from_config(config, device=None):
    """Build a fresh Faster R-CNN matching :func:`scripts.train_detector._build_faster_rcnn`.

    Same COCO-pretrained ``fasterrcnn_resnet50_fpn``, same head replacement for
    ``config.detection.num_classes`` (background + ship).
    """
    if device is None:
        device = _default_device()
    num_classes = int(config.detection.num_classes)
    try:  # torchvision >= 0.13
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1
        )
    except (TypeError, AttributeError):  # torchvision < 0.13
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )
    return model.to(device)


def _iou(box_a, box_b) -> float:
    """Intersection-over-union of two (xmin, ymin, xmax, ymax) boxes."""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(all_dets, iou_thresh: float = 0.5):
    """Greedily cluster per-model detections that refer to the same object.

    ``all_dets`` is a list of ``(model_idx, box, score)``. Greedy, score-descending:
    a box joins an existing cluster only if (a) it overlaps the cluster's
    representative box above ``iou_thresh`` and (b) that model is not already in
    the cluster (at most one vote per model per object). Returns a list of
    ``{box, score, uncertainty, count}`` dicts.
    """
    ordered = sorted(all_dets, key=lambda t: t[2], reverse=True)
    clusters, rep = [], []
    for model_idx, box, score in ordered:
        placed = False
        for ci in range(len(clusters)):
            if any(m[0] == model_idx for m in clusters[ci]):
                continue  # this model already voted for that object
            if _iou(box, rep[ci]) >= iou_thresh:
                clusters[ci].append((model_idx, box, score))
                n = len(clusters[ci])
                rep[ci] = [sum(m[1][d] for m in clusters[ci]) / n for d in range(4)]
                placed = True
                break
        if not placed:
            clusters.append([(model_idx, box, score)])
            rep.append(list(box))

    merged = []
    for cl in clusters:
        scores = [m[2] for m in cl]
        boxes = [m[1] for m in cl]
        n = len(cl)
        merged_box = [sum(b[d] for b in boxes) / n for d in range(4)]
        mean_score = sum(scores) / n
        if n >= 2:
            uncertainty = sum((s - mean_score) ** 2 for s in scores) / n
        else:
            uncertainty = 1.0  # seen by only one model -> explicitly high-uncertainty
        merged.append({
            "box": merged_box, "score": mean_score,
            "uncertainty": uncertainty, "count": n,
        })
    return merged


@torch.no_grad()
def run_detector_ensemble(rgb_input, checkpoint_dir, num_members: int,
                          seeds=None, config=None, score_threshold: float = 0.5,
                          iou_thresh: float = 0.5):
    """Run the SAME ``rgb_input`` through ``num_members`` detector checkpoints and merge detections.

    Parameters
    ----------
    rgb_input : torch.Tensor
        A single image ``(C, H, W)`` in ``[0, 1]`` (the range the detectors saw
        at train time). Passed to each detector as a one-element list.
    checkpoint_dir : str | Path
        Directory containing ``detector_seed{seed}.pt``.
    num_members : int
        How many member checkpoints to load (first ``num_members`` of ``seeds``).
    seeds, config, score_threshold, iou_thresh
        seeds/config default from config; ``score_threshold`` drops raw
        low-confidence detections; ``iou_thresh`` is the merge-overlap cutoff.

    Returns
    -------
    dict with:
        merged : list of ``{box, score, uncertainty, count}``
        n_dets : total merged detections
        n_all  : number found by every member (low uncertainty)
        n_single : number found by exactly one member (high uncertainty)
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

    device = _default_device()
    img = rgb_input.to(device)
    all_dets = []
    for seed in member_seeds:
        ckpt = ckpt_dir / f"detector_seed{seed}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"detector checkpoint not found: {ckpt}")
        det = detector_from_config(config, device).eval()
        det.load_state_dict(torch.load(ckpt, map_location=device))
        out = det([img])[0]  # {boxes, labels, scores}
        boxes = out["boxes"].cpu().tolist()
        scores = out["scores"].cpu().tolist()
        labels = out["labels"].cpu().tolist()
        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score >= score_threshold and int(label) != 0:
                all_dets.append((seed, box, score))

    merged = _merge_detections(all_dets, iou_thresh=iou_thresh)
    n_all = sum(1 for m in merged if m["count"] == num_members)
    n_single = sum(1 for m in merged if m["count"] == 1)
    return {"merged": merged, "n_dets": len(merged), "n_all": n_all, "n_single": n_single}


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

    # Dummy SAR is built on CPU; move it to the same device the generator
    # models load onto (CUDA if available, else CPU) to avoid a device mismatch.
    sar = _make_dummy_sar(cfg, size=args.size).to(_default_device())

    # (a) run the full function with the real checkpoint dir.
    real_paths = [ckpt_dir / f"generator_seed{s}.pt" for s in seeds[:num_members]]
    have_real = all(p.exists() for p in real_paths)
    mean_rgb = None
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

    # (c) CRITICAL: same checkpoint used 3x => variance ~zero everywhere.
    # Identical deterministic weights on the same input must produce no real
    # discrepancy, but GPU convolutions are not bit-reproducible, so a tolerance
    # of 1e-6 separates true signal (~1.0 for distinct members) from this noise
    # (observed ~2e-13 on GPU).
    print("\n[c] CRITICAL zero-variance check (same checkpoint repeated 3x):")
    same = [real_paths[0]] * 3
    _, _, var_same = _run_members(sar, same, cfg)
    exact = bool(torch.allclose(var_same, torch.zeros_like(var_same), atol=1e-6))
    print(f"    max variance = {var_same.max().item():.6g}")
    print("    PASS: variance within 1e-6 of zero (identical deterministic weights)"
          if exact else "    FAIL: variance exceeded 1e-6 for identical weights")

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

    # ---- Phase 6, Step 6.2: detector ensemble ----
    det_paths = [ckpt_dir / f"detector_seed{s}.pt" for s in seeds[:num_members]]
    have_det = all(p.exists() for p in det_paths)
    if not have_det:
        missing_det = [str(p) for p in det_paths if not p.exists()]
        print(f"\n[f] SKIPPED — need detector checkpoints in {ckpt_dir}; missing {missing_det}")
        return 0

    print("\n[f] detector ensemble (Phase 6.2):")
    if mean_rgb is not None:
        # mean_rgb is in the generator's output range; the detector expects [0, 1].
        rgb = (mean_rgb[0] + 1.0).clamp(0, 1) / 2.0  # (C, H, W)
        print("    input: ensemble mean RGB from step (a)  (normalized to [0,1])")
    else:
        rgb = torch.rand(1, args.size, args.size).repeat(3, 1, 1)  # placeholder RGB
        print("    input: dummy random RGB (no generator mean available)")

    det_res = run_detector_ensemble(rgb, ckpt_dir, num_members, seeds=seeds, config=cfg)
    print(f"    total detections         : {det_res['n_dets']}")
    print(f"    found by all {num_members} models (low uncertainty)  : {det_res['n_all']}")
    print(f"    found by only 1 model (high uncertainty): {det_res['n_single']}")
    print("    example detections (merged):")
    for m in sorted(det_res["merged"], key=lambda d: d["score"], reverse=True)[:5]:
        box = m["box"]
        tag = "low-unc" if m["count"] == num_members else ("HIGH-UNC" if m["count"] == 1 else "med")
        print(f"      score={m['score']:.3f}  uncertainty={m['uncertainty']:.4f}  "
              f"models={m['count']}/{num_members}  [{tag}]  "
              f"box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())