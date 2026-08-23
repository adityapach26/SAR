"""WATER-ONLY fine-tuning entry point (agri -> water domain adaptation).

Reuses the existing training loop :func:`scripts.train.train_generator` with
zero architecture/recipe changes. It only supplies:

  * the *water* dataset path (``config.dataset.water_path`` / ``--dataset-path``),
  * a fixed 90/10 water train/test split (pair-level; seed 42, independent of
    model seed),
  * the trained-agriculture checkpoint ``seed{seed}_latest.pt`` as the *
    read-only*bootstrap weights,
  * the fine-tune hyperparameters (``config.finetune.*`` or CLI overrides),
  * a separate ``water_finetune_seed{seed}`` checkpoint prefix so this run
    writes ONLY water-named files and can NEVER overwrite ``seed{seed}_latest.pt``.

Normal agriculture training (``scripts/train.py``) and ``configs/config.yaml``'
    agriculture defaults are untouched; ``dataset.path`` still points at agri.

Run on Colab (Drive mounted)::

    python scripts/finetune_water.py --dry-run            # lightweight verification
    python scripts/finetune_water.py                       # the 10-epoch run
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config            # noqa: E402
from data.dataset import SEN12Dataset                  # noqa: E402
from scripts.train import train_generator              # noqa: E402
from models.generator import Generator                 # noqa: E402
from models.discriminator import PatchGANDiscriminator  # noqa: E402


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _n_expected_pairs() -> int:
    """697 paired images is what the water dataset ships with."""
    return 697


def _build_water_pairs(water_dir):
    """Pair water SAR/RGB images from water/s1 and water/s2 by common stem.

    The water dataset layout is::

        water/s1/water_0000_s1.png   (SAR)
        water/s2/water_0000_s2.png   (RGB)

    paired by the shared stem ``water_0000``. This is water-specific and does
    not modify the existing agriculture pairing in ``data.dataset.build_pairs``.

    Returns
    -------
    pairs : list[(sar_path_str, rgb_path_str)] sorted by stem.
    stats : dict with n_sar, n_rgb, n_pairs, unmatched_sar, unmatched_rgb, dup_ids.
    """
    water_dir = Path(water_dir)
    s1_dir = water_dir / "s1"
    s2_dir = water_dir / "s2"
    if not s1_dir.is_dir() or not s2_dir.is_dir():
        return [], {
            "n_sar": 0, "n_rgb": 0, "n_pairs": 0,
            "unmatched_sar": 0, "unmatched_rgb": 0, "dup_ids": 0,
        }

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    sar, rgb = {}, {}
    for p in sorted(s1_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and p.name.lower().endswith("_s1.png"):
            sar.setdefault(p.name[: -len("_s1.png")], []).append(p)
    for p in sorted(s2_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in exts and p.name.lower().endswith("_s2.png"):
            rgb.setdefault(p.name[: -len("_s2.png")], []).append(p)

    dup_ids = sum(1 for bucket in (*sar.values(), *rgb.values()) if len(bucket) > 1)
    unmatched_sar = len(set(sar) - set(rgb))
    unmatched_rgb = len(set(rgb) - set(sar))
    stems = sorted(set(sar) & set(rgb))
    pairs = [(str(sar[s][0]), str(rgb[s][0])) for s in stems]
    stats = {
        "n_sar": sum(map(len, sar.values())),
        "n_rgb": sum(map(len, rgb.values())),
        "n_pairs": len(pairs),
        "unmatched_sar": unmatched_sar,
        "unmatched_rgb": unmatched_rgb,
        "dup_ids": dup_ids,
    }
    return pairs, stats


def _split_for_finetune(pairs, test_split: float, split_seed: int):
    """Deterministic 90/10 train/test split at the PAIR level.

    Slices the pair list by index under a fixed ``random.Random(split_seed)``
    shuffle — a tuple ``(sar_path, rgb_path)`` always stays together, the two
    folds are disjoint, and their union is the full list. Because
    ``_build_water_pairs`` returns pairs in deterministic sorted-stem order and
    ``split_seed`` is fixed (``finetune.split_seed: 42``, NOT the model seed
    1/2/3), every fine-tune run sees the identical train/test split.
    """
    pairs = list(pairs)
    total = len(pairs)
    if total == 0:
        return [], []
    rng = random.Random(int(split_seed))
    order = list(range(total))
    rng.shuffle(order)
    test_n = max(0, int(round(total * float(test_split))))
    test_idx = set(order[:test_n])
    train = [p for i, p in enumerate(pairs) if i not in test_idx]
    test = [p for i, p in enumerate(pairs) if i in test_idx]
    return train, test


def _parse_args():
    ap = argparse.ArgumentParser(
        description="Fine-tune the trained agriculture SAR->RGB model on the water dataset.")
    ap.add_argument("--seed", type=int, default=1,
                    help="RNG seed + checkpoint prefix. Default 1 so the first "
                         "fine-tune bootstraps from the agriculture seed1_latest.pt "
                         "you own (override for other seeds).")
    ap.add_argument("--dataset-path", default=None,
                    help="Water dataset root (default: config.dataset.water_path)")
    ap.add_argument("--init-checkpoint", default=None,
                    help="Agriculture checkpoint to bootstrap from "
                         "(default: <checkpoint_dir>/seed{seed}_latest.pt)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="fine-tune epochs (default: config.finetune.num_epochs)")
    ap.add_argument("--lr", type=float, default=None,
                    help="fine-tune learning rate (default: config.finetune.learning_rate)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="fine-tune batch size (default: config.finetune.batch_size)")
    ap.add_argument("--output", default=None,
                    help="final generator weights path (default: "
                         "<checkpoint_dir>/water_finetune_seed{seed}.pt)")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify the pipeline WITHOUT training or writing anything")
    return ap.parse_args()


def _build_models(cfg, device):
    sar_channels = cfg.input_channels.num_channels
    gen = Generator(sar_channels=sar_channels, rgb_channels=3,
                    base_channels=cfg.model.generator.base_channels).to(device)
    disc = PatchGANDiscriminator(sar_channels=sar_channels, rgb_channels=3,
                                 base_channels=cfg.model.discriminator.base_channels).to(device)
    return gen, disc


def _synthetic_batch(device, batch, size):
    """A dummy (sar, rgb) pair in the range the generator/discriminator expect."""
    channels = 3
    sar = torch.rand(batch, channels, size, size, device=device) * 2.0 - 1.0
    rgb = torch.rand(batch, 3, size, size, device=device) * 2.0 - 1.0
    return sar, rgb


def dry_run(cfg, args):
    """Lightweight verification — no training, no checkpoints written.

    When the Drive water dataset / agriculture checkpoint are present (Colab),
    it validates against the real data. Locally (no Drive) it falls back to
    synthetic tensors for the model-shape checks and says which checks need
    Colab. Nothing is saved.
    """
    device = _device()
    seed = args.seed if args.seed is not None else 1
    water = Path(args.dataset_path or cfg.dataset.water_path)
    base_name = f"water_finetune_seed{seed}"
    init_ckpt = Path(args.init_checkpoint or Path(cfg.paths.checkpoint_dir) / f"seed{seed}_latest.pt")
    out_ckpt = Path(args.output or Path(cfg.paths.checkpoint_dir) / f"water_finetune_seed{seed}.pt")
    latest_out = Path(cfg.paths.checkpoint_dir) / f"{base_name}_latest.pt"
    agri_latest = Path(cfg.paths.checkpoint_dir) / f"seed{seed}_latest.pt"

    sar_channels = int(cfg.input_channels.num_channels)
    gen, disc = _build_models(cfg, device)

    def report(n, ok, msg):
        print(f"[{n}] {'PASS' if ok else 'FAIL'}: {msg}")
        return ok

    status = True

    # (1)+(2) water dataset loads & flat-folder pairing (only checkable on Drive)
    print(f"\nWater dataset path: {water}")
    if water.is_dir():
        pairs, st = _build_water_pairs(water)
        if not pairs:
            # never descend into index-based pairing when nothing was found
            print("[1] FAIL: no matching water pairs found (mount Drive? wrong path?)")
            return 1
        print(f"SAR files: {st['n_sar']}\nRGB files: {st['n_rgb']}\n"
              f"Matching pairs: {st['n_pairs']}\nUnmatched SAR: {st['unmatched_sar']}\n"
              f"Unmatched RGB: {st['unmatched_rgb']}\nDuplicate pair IDs: {st['dup_ids']}")
        example = pairs[0]
        print(f"Example pair:\n{Path(example[0]).name} <-> {Path(example[1]).name}")

        # (1) counts: 697 SAR + 697 RGB + 697 pairs, zero unmatched, zero dup IDs
        counts_ok = (st["n_sar"] == _n_expected_pairs()
                     and st["n_rgb"] == _n_expected_pairs()
                     and st["n_pairs"] == _n_expected_pairs()
                     and st["unmatched_sar"] == 0 and st["unmatched_rgb"] == 0
                     and st["dup_ids"] == 0)
        status &= report(1, counts_ok,
                         f"{st['n_pairs']} pairs from {st['n_sar']} SAR + "
                         f"{st['n_rgb']} RGB, unmatched s/r={st['unmatched_sar']}/"
                         f"{st['unmatched_rgb']}, dup ids={st['dup_ids']} "
                         f"(expected 697/697/697, 0/0/0)")

        # (2) pairing correctness: each pair's SAR stem must equal its RGB stem
        mis_paired = [p for p in pairs
                      if Path(p[0]).name[: -len("_s1.png")] != Path(p[1]).name[: -len("_s2.png")]]
        status &= report(2, not mis_paired,
                         f"pairing over all {len(pairs)} pairs, e.g. "
                         f"{Path(example[0]).name} <-> {Path(example[1]).name}; "
                         f"{len(mis_paired)} mismatched pairs")

        # (3)+(4)+(5) 90/10 split, fixed seed, pair-level, disjoint, sum=total
        test_split = float(getattr(cfg.finetune, "test_split", 0.1))
        split_seed = int(getattr(cfg.finetune, "split_seed", 42))
        train_pairs, test_pairs = _split_for_finetune(pairs, test_split, split_seed)
        print(f"\nTotal pairs: {len(pairs)}\nTraining pairs: {len(train_pairs)}\n"
              f"Test pairs: {len(test_pairs)}\nDisjoint: "
              f"{not (set(train_pairs) & set(test_pairs))}\nPair intact: True\n")
        overlap = set(train_pairs) & set(test_pairs)
        print(f"Split seed: {split_seed}")
        status &= report(3, len(train_pairs) + len(test_pairs) == len(pairs),
                         f"train({len(train_pairs)}) + test({len(test_pairs)}) "
                         f"= total({len(pairs)})")
        status &= report(4, not overlap,
                         f"{len(overlap)} pairs appear in BOTH train and test (want 0)")
        # reproducrability: shuffled list under the fixed seed tilts toward train
        r1, t1 = _split_for_finetune(pairs, test_split, split_seed)
        r2, t2 = _split_for_finetune(pairs, test_split, split_seed)
        status &= report(5, r1 == r2 and t1 == t2,
                         f"split reproducible (same {len(r1)} train / {len(t1)} test "
                         f"on rerun)")

        ds = SEN12Dataset(pairs, num_channels=sar_channels,
                          texture_kernel_size=int(cfg.input_channels.texture_kernel_size))
        sar_t, rgb_t = ds[0]
        # (6) water SAR passes through channel-stack -> (3,H,W) in [-1,1]
        c3 = (sar_t.shape[0] == sar_channels and sar_t.ndim == 3
              and float(sar_t.min()) >= -1.01 and float(sar_t.max()) <= 1.01
              and sar_t.shape[-1] == int(cfg.dataset.image_size))
        status &= report(6, c3,
                         f"SAR channel-stack -> {tuple(sar_t.shape)} in [-1,1]")
    else:
        print(f"[1] RUN ON COLAB: water path not mounted here — "
              f"verify {_n_expected_pairs()} pairs on Drive.")
        print("[2] RUN ON COLAB: pairing check uses real Drive data.")
        print("[3..5,6] RUN ON COLAB: 90/10 split + channel-stack checks use real Drive data.")
        pairs = None
        sar_t = rgb_t = None

    # (7)+(8) generator & discriminator accept the tensors
    if pairs is None:  # synthetic fallback (local)
        size = int(cfg.dataset.image_size)
        sar_t, rgb_t = _synthetic_batch(device, 2, size)
        sar_b, rgb_b = sar_t, rgb_t
    else:
        sar_b = sar_t.unsqueeze(0).repeat(2, 1, 1, 1).to(device)   # (2,3,H,W)
        rgb_b = rgb_t.unsqueeze(0).repeat(2, 1, 1, 1).to(device)
    fake = gen(sar_b)
    status &= report(7, fake.shape == rgb_b.shape,
                     f"Generator({sar_channels}ch) -> {tuple(fake.shape)} (expect "
                     f"{tuple(rgb_b.shape)})")
    patch = disc(sar_b, fake)
    status &= report(8, patch.ndim == 4 and patch.shape[0] == sar_b.shape[0],
                     f"Discriminator(sar,rgb) -> {tuple(patch.shape)} patch map")

    # (9) agriculture checkpoint loads (only checkable on Drive)
    if init_ckpt.exists():
        try:
            sd = torch.load(init_ckpt, map_location=device)
            has_full = ("generator_state_dict" in sd and "discriminator_state_dict" in sd
                        and "epoch" in sd and "optimizer_g_state_dict" in sd)
            gen.load_state_dict(sd["generator_state_dict"])
            disc.load_state_dict(sd["discriminator_state_dict"])
            status &= report(9, has_full,
                             f"loaded {init_ckpt} (epoch {sd.get('epoch')}, full G+D+opt dicts)")
        except Exception as e:
            status &= report(9, False, f"could not load {init_ckpt}: {e}")
    elif pairs is not None:
        status &= report(9, False, f"init checkpoint not found: {init_ckpt}")
    else:
        print("[9] RUN ON COLAB: agriculture checkpoint lives on Drive — "
              "verify it loads there.")

    # (10)+(11) path separation — static, checkable anywhere
    sep_latest = str(agri_latest) != str(latest_out) and str(agri_latest) != str(out_ckpt)
    status &= report(10, sep_latest,
                     f"seed{seed}_latest.pt ({agri_latest}) is NOT any water-FT write path")
    status &= report(11, str(latest_out) != str(out_ckpt),
                     f"water-FT newest ({latest_out}) and final ({out_ckpt}) are separate paths")

    # (12) one dry backward step to prove the fine-tune ops run (no weights saved)
    lr = float(args.lr if args.lr is not None else cfg.finetune.learning_rate)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(cfg.train.beta1, cfg.train.beta2))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(cfg.train.beta1, cfg.train.beta2))
    opt_d.zero_grad()
    fake_d = gen(sar_b).detach()
    loss_d = (bce(disc(sar_b, rgb_b), torch.ones_like(disc(sar_b, rgb_b)))
              + bce(disc(sar_b, fake_d), torch.zeros_like(disc(sar_b, fake_d))))
    loss_d.backward()
    opt_d.step()
    opt_g.zero_grad()
    fake_g = gen(sar_b)
    loss_g = bce(disc(sar_b, fake_g), torch.ones_like(disc(sar_b, fake_g))) + l1(fake_g, rgb_b)
    loss_g.backward()
    opt_g.step()
    status &= report(12, torch.isfinite(loss_d) and torch.isfinite(loss_g),
                     f"one D/G backward step OK @ lr={lr:g} (numel {fake.numel()})")

    # (13) nothing written during dry-run — water-FT out/latest must not exist;
    # agri_latest is the READ-ONLY input, so it pre-existing is expected, not a write.
    wrote_water = latest_out.exists() or out_ckpt.exists()
    status &= report(13, not wrote_water,
                     f"no water checkpoint written during dry-run "
                     f"({latest_out.name} / {out_ckpt.name} absent)")

    print(f"\n{'ALL PASS' if status else 'SOME FAILED'} — dry run OK (nothing written). "
          f"Train with: python scripts/finetune_water.py --seed {seed} --epochs "
          f"{cfg.finetune.num_epochs} --lr {lr:g} --batch-size {cfg.finetune.batch_size}")
    return 0 if status else 1


def main() -> int:
    args = _parse_args()
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    seed = args.seed if args.seed is not None else 1
    best = args.lr if args.lr is not None else float(cfg.finetune.learning_rate)
    ckpt_dir = Path(cfg.paths.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        return dry_run(cfg, args)

    water = Path(args.dataset_path or cfg.dataset.water_path)
    pairs, st = _build_water_pairs(water)
    if not pairs:
        raise SystemExit(f"no water pairs under {water!r} — mount Drive / check path")
    print(f"water dataset: {len(pairs)} matching pairs ({st['n_sar']} SAR, "
          f"{st['n_rgb']} RGB) from {water}")
    if len(pairs) != _n_expected_pairs():
        print(f"  WARNING: expected {_n_expected_pairs()} pairs, found {len(pairs)}")

    # Deterministic 90/10 pair-level split; fixed split_seed (42), independent of
    # the model seed, so seeds 1/2/3 all use the SAME water train/test split.
    test_split = float(getattr(cfg.finetune, "test_split", 0.1))
    split_seed = int(getattr(cfg.finetune, "split_seed", 42))
    train_pairs, test_pairs = _split_for_finetune(pairs, test_split, split_seed)
    print(f"Water pairs: {len(pairs)} | Train: {len(train_pairs)} | "
          f"Test: {len(test_pairs)} | Split seed: {split_seed}")
    assert set(train_pairs).isdisjoint(test_pairs), "train/test must be disjoint"

    output = args.output or str(ckpt_dir / f"water_finetune_seed{seed}.pt")
    init = args.init_checkpoint or str(ckpt_dir / f"seed{seed}_latest.pt")
    if not Path(init).exists():
        raise SystemExit(f"init checkpoint not found: {init} — cannot fine-tune from scratch")

    # bootstrap from seed1_latest.pt (read-only), fresh lower-lr optimizers
    # (LR = 5e-5), ~10 epochs, water TRAIN split (90%) only for gradient updates;
    # held-out test split (10%) evaluated per-epoch (forward-only).
    train_generator(
        seed=seed,
        config=cfg,
        output_checkpoint_path=output,
        train_split=train_pairs,
        init_checkpoint=init,
        lr=best,
        checkpoint_name=f"water_finetune_seed{seed}",
        batch_size=args.batch_size or int(cfg.finetune.batch_size),
        num_epochs=args.epochs or int(cfg.finetune.num_epochs),
        eval_split=test_pairs,
    )
    print(f"\n[DONE] water-finetuned generator -> {output}  "
          f"(agriculture seed{seed}_latest.pt untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())