"""Generate a small folder of dummy SAR/RGB pairs for dev / smoke testing.

Writes images into the SEN1-2 layout that ``data.dataset.build_pairs``
expects:
    data/dummy_data/
        s1_0/img_000.png   (SAR, grayscale)
        s2_0/img_000.png   (RGB)
        s1_1/...
        s2_1/...

Each scene is one directory pair; each scene holds ``images_per_scene``
matching image pairs. The images are synthetic patches (SAR = gradient +
noise, RGB = colored gradients) large enough to run the Phase 2 network
end to end.

Run:  python scripts/make_dummy_data.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_scene(mode: str, side: int, rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray]:
    """Return (sar, rgb) as uint8 arrays: (H, W) and (H, W, 3)."""
    if mode == "gradient":
        ramp = np.linspace(40, 220, side, dtype=np.float32)          # (side,)
        base = np.broadcast_to(ramp[None, :], (side, side))          # vertical gradient
        sar = base + rng.normal(0, 25, (side, side)).astype(np.float32)
        r = np.broadcast_to(ramp[None, :], (side, side))
        g = np.broadcast_to(ramp[::-1][None, :], (side, side))
        b = np.full((side, side), 128.0, dtype=np.float32)
        rgb = np.stack([r, g, b], axis=-1)                            # (H, W, 3)
    else:
        sar = rng.integers(0, 256, (side, side), dtype=np.uint8)
        rgb = rng.integers(0, 256, (side, side, 3), dtype=np.uint8)

    sar = np.clip(sar, 0, 255).astype(np.uint8)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return sar, rgb


def main(dataset_path: Path, num_scenes: int, images_per_scene: int,
         side: int = 256, mode: str = "gradient", seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    dataset_path.mkdir(parents=True, exist_ok=True)
    for s in range(num_scenes):
        (dataset_path / f"s1_{s}").mkdir(parents=True, exist_ok=True)
        (dataset_path / f"s2_{s}").mkdir(parents=True, exist_ok=True)
        for k in range(images_per_scene):
            sar, rgb = make_scene(mode, side, rng)
            name = f"img_{k:03d}.png"
            Image.fromarray(sar, mode="L").save(dataset_path / f"s1_{s}" / name)
            Image.fromarray(rgb, mode="RGB").save(dataset_path / f"s2_{s}" / name)
    print(f"wrote {num_scenes} scenes x {images_per_scene} pairs to {dataset_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate dummy SAR/RGB pairs.")
    ap.add_argument("--dataset-path", default="data/dummy_data")
    ap.add_argument("--num-scenes", type=int, default=3)
    ap.add_argument("--images-per-scene", type=int, default=6)
    ap.add_argument("--side", type=int, default=256)
    ap.add_argument("--mode", choices=["gradient", "noise"], default="gradient")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(Path(args.dataset_path),
         args.num_scenes, args.images_per_scene,
         args.side, args.mode, args.seed)