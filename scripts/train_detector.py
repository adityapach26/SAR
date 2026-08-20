"""Faster R-CNN detection training loop for RBox-SSDD (ships).

The training logic lives in :func:`train_detector`, which is self-contained:
it seeds RNG, builds a fresh ``fasterrcnn_resnet50_fpn`` loaded with COCO-
pretrained weights, replaces the box predictor head for ``config.detection.
num_classes`` (background + ship), and fine-tunes the whole model on the
``SSDDataset`` train split with a standard torchvision SGD setup. It saves
the final model weights to ``output_path`` and returns the trained model —
same shape as :func:`train_generator`, so an ensemble could loop over members.

Torchvision Faster R-CNN expects raw image tensors as a stack ``(N, C, H, W)``
in ``[0, 1]`` and targets as a list of dicts: ``boxes`` (FloatTensor Nx4 in
``(xmin, ymin, xmax, ymax)``) + ``labels``. Both conventions are exactly what
``SSDDataset`` already yields, so the DataLoader only needs a collate that
keeps each image+target pair separate (torchvision's detection training takes a
stacked tensor + a list of targets, not a default-collated batch).

Fine-tuning ("fine-tune"): the COCO-pretrained backbone/FPN weights stay in the
model and every parameter is trained onward with SGD — no layers frozen, so the
head's random-initialized ``FastRCNNPredictor`` learns the ship/background
classes alongside the finetuned features.

Run:  python scripts/train_detector.py [--seed N] [--epochs N] [--output PATH] [--dataset-path PATH]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402
from data.detection_dataset import SSDDataset  # noqa: E402


def _build_faster_rcnn(num_classes: int, device: torch.device) -> nn.Module:
    """``fasterrcnn_resnet50_fpn`` with COCO weights and a ship head.

    Works across torchvision versions: newer builds take ``weights=``, older
    ones only ``pretrained=``. The classification head is replaced with a fresh
    ``FastRCNNPredictor`` sized for ``num_classes`` (background + ships), so
    only the box/detection head changes while backbone + RPN + RoI features
    stay; backbones come pretrained from COCO.
    """
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


def _collate(batch):
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets


def train_detector(seed, config, output_path):
    """Train a COCO-pretrained Faster R-CNN on ``SSDDataset`` train.

    Parameters
    ----------
    seed : int
        RNG seed. ``torch.manual_seed(seed)`` (plus numpy/python) is called at
        the very start so weight/loader shuffling is reproducible.
    config : Config
        Loaded config object; reads ``detection`` and ``train`` keys.
    output_path : str | Path
        Where the final model weights (state_dict) will be saved.

    Returns
    -------
    nn.Module
        The trained Faster R-CNN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_classes = int(config.detection.num_classes)
    lr = float(getattr(config.detection, "learning_rate", config.train.learning_rate))
    batch_size = getattr(config.detection, "batch_size", None) or config.train.batch_size
    momentum = float(getattr(config.detection, "momentum", 0.9))
    weight_decay = float(getattr(config.detection, "weight_decay", 0.0005))
    n_epochs = config.train.num_epochs
    log_every = getattr(config.train, "log_every_n_batches", 20)

    model = _build_faster_rcnn(num_classes, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    print(f"seed={seed}  device={device}  epochs={n_epochs}  "
          f"num_classes={num_classes}  lr={lr}")

    ds = SSDDataset(config.detection.dataset_path, split="train")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                        num_workers=config.train.num_workers, pin_memory=True,
                        collate_fn=_collate, drop_last=False)
    print(f"  dataset: {ds.root}")
    print(f"  train   : {len(ds)} images  ({len(loader)} batches/epoch)")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, n_epochs + 1):
        model.train()
        acc_loss, n_batches = 0.0, 0
        t0 = time.time()
        for i, (images, targets) in enumerate(loader, 1):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(v for v in loss_dict.values())
            opt.zero_grad()
            loss.backward()
            opt.step()

            acc_loss += loss.item()
            n_batches += 1
            if i % log_every == 0 or i == len(loader):
                print(f"  epoch {epoch}  batch {i}/{len(loader)}  "
                      f"total {loss.item():.4f}  "
                      f"({', '.join(f'{k} {v.item():.4f}' for k, v in loss_dict.items())})")

        print(f"[epoch {epoch}] mean loss {acc_loss / max(n_batches, 1):.4f} "
              f"({time.time() - t0:.1f}s)")

    torch.save(model.state_dict(), out)
    print(f"[save] final detection model -> {out}")
    return model


def main() -> int:
    ap = argparse.ArgumentParser(description="Faster R-CNN fine-tuning (ships).")
    ap.add_argument("--seed", type=int, default=None,
                    help="RNG seed (default: dataset.random_seed from config)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override train.num_epochs from config (for testing)")
    ap.add_argument("--output", default=None,
                    help="final model weights path "
                         "(default: <checkpoint_dir>/detector_final.pt)")
    ap.add_argument("--dataset-path", default=None,
                    help="override detection.dataset_path (voc_style root)")
    args = ap.parse_args()

    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    if args.seed is not None:
        cfg.dataset.random_seed = args.seed
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.dataset_path is not None:
        cfg.detection.dataset_path = args.dataset_path
    output = args.output or str(Path(cfg.paths.checkpoint_dir) / "detector_final.pt")

    train_detector(seed=cfg.dataset.random_seed, config=cfg, output_path=output)
    return 0


if __name__ == "__main__":
    sys.exit(main())