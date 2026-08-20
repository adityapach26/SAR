"""Detection dataset for RBox-SSDD (rotated bounding box, PASCAL-VOC XML).

The real dataset lives at RBox_SSDD/voc_style/:
    voc_style/
        JPEGImages/          000001.jpg ... (SAR/RGB images)
        Annotations/         000001.xml ... (PASCAL-VOC XML per image)
        ImageSets/Main/      train.txt / test.txt (one image id per line)

Each <object> carries a <rotated_bndbox> with corner points x1,y1..x4,y4 (NOT a
plain axis-aligned <bndbox>). The rotated quad is reduced to an axis-aligned
bounding box by taking the min/max of the four corners:
    [min(x1..x4), min(y1..y4), max(x1..x4), max(y1..y4)]

``__getitem__`` returns ``(image_tensor, target_dict)``:
    image_tensor : FloatTensor (3, H, W) in [0, 1]
    target       : dict
        boxes  -> FloatTensor (N, 4) as (xmin, ymin, xmax, ymax)
        labels -> LongTensor (N,) with all 1 (class "ship"; background is 0)

Run:  python data/detection_dataset.py [--dataset PATH] [--split train|test] [--index N]
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402

# Fallback used only when cfg.detection.dataset_path is empty.
DEFAULT_DATASET_ROOT = "RBox_SSDD/voc_style"


def localname(tag: str) -> str:
    """Strip any XML namespace prefix (``{ns}tag`` -> ``tag``)."""
    return tag.rsplit("}", 1)[-1]


def _child(el, name: str):
    """First direct child of ``el`` whose local name matches, else None."""
    for c in el:
        if localname(c.tag) == name:
            return c
    return None


def _children(el, name: str):
    """All direct children of ``el`` whose local name matches."""
    return [c for c in el if localname(c.tag) == name]


def parse_annotation(xml_path: Path):
    """Parse a PASCAL-VOC XML into an AABB-based detection target.

    Each <object> is reduced to an axis-aligned box ``(xmin, ymin, xmax, ymax)``
    derived from its <rotated_bndbox> corner points (or a plain <bndbox> if that
    is all the annotation carries). Every object gets label 1 ("ship"); the 0
    class is reserved for background. Images with no objects yield empty tensors (boxes ``(0, 4)``, labels ``(0,)``).

    Returns
    -------
    (boxes, labels) : (FloatTensor (N, 4), LongTensor (N,))
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes, labels = [], []
    for obj in _children(root, "object"):
        rotated = _child(obj, "rotated_bndbox")
        if rotated is not None:
            xs = _corners_of(rotated, "x")
            ys = _corners_of(rotated, "y")
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
        else:
            plain = _child(obj, "bndbox")
            if plain is None:
                continue
            xmin = float(_child(plain, "xmin").text)
            ymin = float(_child(plain, "ymin").text)
            xmax = float(_child(plain, "xmax").text)
            ymax = float(_child(plain, "ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # "ship"
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)
    return (torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.long))


def _corners_of(rotated: ET.Element, coord: str) -> list[float]:
    """The four corner values (``coord`` = 'x' or 'y') of a <rotated_bndbox>."""
    return [float(_child(rotated, f"{coord}{i}").text) for i in range(1, 5)]


class SSDDataset(Dataset):
    """Load RBox-SSDD image + annotation pairs for one official split.

    Parameters
    ----------
    dataset_root : str | Path
        The voc_style folder (JPEGImages/ + Annotations/ + ImageSets/Main/).
    split : str
        One of "train" or "test"; selects the id list from
        ``ImageSets/Main/<split>.txt``.
    """

    def __init__(self, dataset_root: str | Path, split: str = "train") -> None:
        if split not in ("train", "test"):
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")
        self.root = Path(dataset_root)
        self.split = split
        self.image_dir = self.root / "JPEGImages"
        self.annot_dir = self.root / "Annotations"
        ids_file = self.root / "ImageSets" / "Main" / f"{split}.txt"
        if not ids_file.exists():
            raise FileNotFoundError(f"split list not found: {ids_file}")
        self.ids = [ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()]
        if not self.ids:
            raise ValueError(f"no image ids in {ids_file}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img = np.array(Image.open(self.image_dir / f"{img_id}.jpg").convert("RGB"),
                       dtype=np.float32)  # (H, W, 3) in [0, 255]
        img_h, img_w = img.shape[:2]
        image = torch.from_numpy(img).permute(2, 0, 1) / 255.0  # -> (3, H, W) in [0, 1]

        xml_path = self.annot_dir / f"{img_id}.xml"
        boxes, labels = parse_annotation(xml_path)
        target = {"boxes": boxes, "labels": labels}
        return image, target


def _resolve_root(cfg) -> str:
    """Config'detection.dataset_path, or the hardcoded local fallback."""
    p = cfg.detection.dataset_path
    return str(p) if p else DEFAULT_DATASET_ROOT


def main() -> int:
    try:
        cfg = load_config(str(ROOT / "configs" / "config.yaml"))
        default_root = _resolve_root(cfg)
    except Exception:
        default_root = DEFAULT_DATASET_ROOT

    ap = argparse.ArgumentParser(description="RBox-SSDD detection dataset demo.")
    ap.add_argument("--dataset", default=default_root,
                    help="voc_style dataset root (default: cfg.detection.dataset_path)")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--index", type=int, default=0, help="annotation index to visualize")
    args = ap.parse_args()

    ds = SSDDataset(args.dataset, split=args.split)
    if not 0 <= args.index < len(ds):
        raise SystemExit(f"--index {args.index} out of range [0, {len(ds)}) for split {args.split!r}")
    print(f"dataset root : {ds.root}")
    print(f"split        : {args.split}  ({len(ds)} images)")
    print(f"index        : {args.index}  (image id {ds.ids[args.index]})")

    image, target = ds[args.index]
    img_id = ds.ids[args.index]
    img_h, img_w = image.shape[1], image.shape[2]
    boxes = target["boxes"]
    labels = target["labels"]
    print(f"image shape  : {tuple(image.shape)}  (H={img_h}, W={img_w})")
    print(f"n objects    : {len(boxes)}")
    for i, (box, lbl) in enumerate(zip(boxes.tolist(), labels.tolist())):
        x0, y0, x1, y1 = box
        assert 0.0 <= x0 <= x1 <= img_w, f"box {i} x-range {x0}..{x1} outside [0, {img_w}]"
        assert 0.0 <= y0 <= y1 <= img_h, f"box {i} y-range {y0}..{y1} outside [0, {img_h}]"
        print(f"  box {i}: [{x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}]  label={lbl} (ship)")
    print("OK: every axis-aligned box is within the image <size>.")

    # --- visualization: draw the derived boxes; they should land on real ships. ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rgb = image.permute(1, 2, 0).numpy()  # (H, W, 3) [0,1]
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(rgb)
    for b in boxes.tolist():
        x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, edgecolor="lime", lw=2))
    ax.set_title(f"RBox-SSDD {args.split} {img_id} — derived AABB boxes ({len(boxes)})")
    ax.axis("off")
    out = ROOT / f"detection_{args.split}_{img_id}_boxes.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"saved visualization -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())