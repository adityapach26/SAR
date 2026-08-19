import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
try:
    # When imported as part of the package
    from .channel_stack import build_multichannel_input
except ImportError:
    # When run as a script directly
    from channel_stack import build_multichannel_input


_IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')


def _is_image(fname: str) -> bool:
    return fname.lower().endswith(_IMAGE_EXTS)


def build_pairs(dataset_path: str, num_scenes: int | None = None):
    """
    Return matching SAR-RGB pairs as (pairs, mismatches).

    Supports two folder layouts:

    Real "agri" layout (detected automatically — the two subfolders are
    exactly ``s1/`` and ``s2/``):
        dataset_path/
            s1/ROIs1868_summer_s1_59_p2.png
            s2/ROIs1868_summer_s2_59_p2.png
      Filenames correspond by replacing ``_s1_`` with ``_s2_``. Every file in
      ``s1/`` is matched to its expected ``s2/`` counterpart; if that file is
      missing the SAR filename is collected as a mismatch (reported, not fatal).

    Legacy SEN1-2 layout (only scanned when ``num_scenes`` is given, ignored
    for the new layout):
        dataset_path/
            s1_0/image001.png
            s2_0/image001.png
            s1_1/...  s2_1/...

    Parameters
    ----------
    dataset_path : str
        Path to the root folder.
    num_scenes : int | None
        Scene-pair count for the legacy layout. Ignored for the s1/s2 layout.

    Returns
    -------
    pairs : list of (sar_path, rgb_path)
    mismatches : list of SAR filenames whose expected s2 counterpart was missing
    """
    s1_dir = os.path.join(dataset_path, "s1")
    s2_dir = os.path.join(dataset_path, "s2")
    if os.path.isdir(s1_dir) and os.path.isdir(s2_dir):
        pairs, mismatches = [], []
        s2_files = {f for f in os.listdir(s2_dir) if _is_image(f)}
        for fname in sorted(os.listdir(s1_dir)):
            if not _is_image(fname):
                continue
            expected = fname.replace("_s1_", "_s2_")
            if expected in s2_files:
                pairs.append((os.path.join(s1_dir, fname), os.path.join(s2_dir, expected)))
            else:
                mismatches.append(fname)
        return pairs, mismatches

    # Legacy fallback
    pairs = []
    for i in range(num_scenes or 0):
        s1_i = os.path.join(dataset_path, f"s1_{i}")
        s2_i = os.path.join(dataset_path, f"s2_{i}")
        if not (os.path.isdir(s1_i) and os.path.isdir(s2_i)):
            continue
        s1_files = {f.lower(): f for f in os.listdir(s1_i) if _is_image(f)}
        s2_files = {f.lower(): f for f in os.listdir(s2_i) if _is_image(f)}
        for base_name_lower, s1_fname in s1_files.items():
            if base_name_lower in s2_files:
                pairs.append((os.path.join(s1_i, s1_fname),
                              os.path.join(s2_i, s2_files[base_name_lower])))
    return pairs, []


class SEN12Dataset(Dataset):
    """
    Dataset for SEN1-2 SAR-RGB pairs.

    Returns:
        sar_tensor: Tensor of shape (C, H, W) where C=1 if num_channels=1 else C=3
        rgb_tensor: Tensor of shape (3, H, W)
    Both tensors are normalized to [-1, 1] to match Tanh output range.
    """
    def __init__(self, pairs, num_channels=1):
        """
        Parameters
        ----------
        pairs : list of tuple
            List of (sar_path, rgb_path) from build_pairs.
        num_channels : int, optional
            Number of channels for SAR input (1 or 3). Default: 1.
        """
        self.pairs = pairs
        self.num_channels = num_channels
        if self.num_channels not in (1, 3):
            raise ValueError("num_channels must be 1 or 3")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sar_path, rgb_path = self.pairs[idx]

        # Load SAR image (grayscale)
        sar_img = Image.open(sar_path).convert('L')  # Ensure single channel
        sar_np = np.array(sar_img, dtype=np.uint8)  # shape (H, W)

        if self.num_channels == 1:
            # Convert to tensor and normalize to [-1, 1]
            sar_tensor = torch.from_numpy(sar_np).float() / 255.0  # [0, 1]
            sar_tensor = sar_tensor * 2.0 - 1.0  # [-1, 1]
            sar_tensor = sar_tensor.unsqueeze(0)  # shape (1, H, W)
        else:  # num_channels == 3
            # Use build_multichannel_input to get (3, H, W) in [0, 1]
            sar_np_float = build_multichannel_input(sar_np, kernel_size=7)  # shape (3, H, W), float32 in [0,1]
            sar_tensor = torch.from_numpy(sar_np_float)  # shape (3, H, W)
            sar_tensor = sar_tensor * 2.0 - 1.0  # [-1, 1]

        # Load RGB image
        rgb_img = Image.open(rgb_path).convert('RGB')  # Ensure 3 channels
        rgb_np = np.array(rgb_img, dtype=np.uint8)  # shape (H, W, 3)
        # Convert to tensor (C, H, W) and normalize to [-1, 1]
        rgb_tensor = torch.from_numpy(rgb_np).permute(2, 0, 1).float() / 255.0  # [0, 1]
        rgb_tensor = rgb_tensor * 2.0 - 1.0  # [-1, 1]

        return sar_tensor, rgb_tensor


if __name__ == "__main__":
    # Inspect SAR/RGB pair matching for the real "agri" layout (s1/ + s2/).
    import argparse
    import sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).resolve().parents[1]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from utils.config_loader import load_config

    ap = argparse.ArgumentParser(description="Match SAR/RGB pairs (agri layout: s1/ + s2/).")
    ap.add_argument("--dataset-path", default=None,
                    help="Path to the agri folder. Defaults to configs/config.yaml 'dataset.path'.")
    args = ap.parse_args()

    path = args.dataset_path or load_config(str(_ROOT / "configs" / "config.yaml")).dataset.path
    print(f"Scanning dataset path: {path}")

    if not os.path.isdir(path):
        sys.exit(f"ERROR: dataset path not found: {path!r} — "
                 "run in the environment where the Drive folder is mounted.")

    pairs, mismatches = build_pairs(path)
    print(f"Total matching pairs found : {len(pairs)}")
    print(f"Mismatched SAR files       : {len(mismatches)}")
    for m in mismatches:
        print(f"  mismatch: {m}")

    if pairs:
        sar, rgb = pairs[0]
        print("Example pair:")
        print(f"  SAR: {sar}")
        print(f"  RGB: {rgb}")