"""Pre-flight gate: run this BEFORE a real (Colab) training run.

Verifies, with a printed PASS/FAIL per item and an overall verdict:

  [1] Drive paths — configs/config.yaml ``dataset.path``, ``paths.checkpoint_dir``,
      ``paths.output_dir`` must all point at ``/content/drive/...`` (not local
      / relative paths). Fails loudly otherwise.
  [2] Loss weights — ``loss.lambda_gan/l1/perceptual/semantic`` must all be non-zero.
  [3] Seeding — ``scripts/train.py`` defines ``train_generator`` with a ``seed``
      parameter that seeds torch (and ideally numpy + python) inside it.
  [4] Ensemble — ``scripts/train_ensemble.py`` exists, loops over ``ensemble.seeds``,
      and gives each member a distinct checkpoint path (no overwrite risk).
  [5] Output dirs — ``paths.checkpoint_dir`` and ``paths.output_dir`` exist as real
      folders (created with ``os.makedirs(..., exist_ok=True)`` if missing).

Ends with "SAFE TO LAUNCH" (all PASS, exit 0) or "DO NOT LAUNCH" (exit 1).

For testing the [5] folder logic off-Colab, set PREFLIGHT_MOUNT to a writable local
directory: the drive prefix is remapped ONLY for the existence/creation check, so
you can exercise a full green run locally. On a real Colab runtime, leave it unset.

Usage:
    python scripts/preflight_check.py [--config PATH]
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402

DRIVE_PREFIX = "/content/drive"
GREEN, RED, RESET = "\033[92m", "\033[91m", "\033[0m"


def banner(text: str) -> str:
    bar = "=" * (len(text) + 4)
    return f"\n{bar}\n  {text}\n{bar}"


def report(ok: bool, label: str, detail: str = "") -> bool:
    mark = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{mark}] {label}{suffix}")
    return ok


def check_drive_paths(cfg) -> bool:
    """[1] dataset.path, checkpoint_dir, output_dir must be Drive paths."""
    values = {
        "dataset.path": cfg.dataset.path,
        "paths.checkpoint_dir": cfg.paths.checkpoint_dir,
        "paths.output_dir": cfg.paths.output_dir,
    }
    bad = {k: v for k, v in values.items() if not str(v).startswith(DRIVE_PREFIX)}
    for k, v in values.items():
        fmt = v if str(v).startswith(DRIVE_PREFIX) else f"{v!r} (NOT a Drive path)"
        print(f"      {k:<20} = {fmt}")
    if bad:
        return report(False, "1. Drive paths",
                      "expected all under /content/drive/; got non-Drive: "
                      + ", ".join(bad))
    return report(True, "1. Drive paths", "all under /content/drive/")


def check_loss_weights(cfg) -> bool:
    """[2] all four lambdas non-zero."""
    vals = {k: getattr(cfg.loss, k) for k in
            ("lambda_gan", "lambda_l1", "lambda_perceptual", "lambda_semantic")}
    for k, v in vals.items():
        print(f"      {k:<20} = {v}")
    zero = [k for k, v in vals.items() if v == 0]
    if zero:
        return report(False, "2. Loss weights non-zero", "zero weights: " + ", ".join(zero))
    return report(True, "2. Loss weights non-zero", "all four lambdas != 0")


def check_train_generator() -> bool:
    """[3] train_generator exists, has seed param, seeds torch (ideally np+py)."""
    try:
        from scripts.train import train_generator
    except Exception as e:  # noqa: BLE001
        return report(False, "3. train_generator seeding", f"import failed: {e}")

    sig = inspect.signature(train_generator)
    src = inspect.getsource(train_generator)
    has_seed_param = "seed" in sig.parameters

    has_torch_seed = re.search(r"torch\.manual_seed\(\s*seed\s*\)", src) is not None
    has_np_seed = re.search(r"np\.random\.seed\(\s*seed\s*\)", src) is not None
    has_py_seed = re.search(r"random\.seed\(\s*seed\s*\)", src) is not None

    print(f"      signature: {sig}")
    print(f"      torch.manual_seed(seed): {has_torch_seed}   "
          f"np.random.seed(seed): {has_np_seed}   random.seed(seed): {has_py_seed}")

    if has_seed_param and has_torch_seed:
        extra = "full (torch+numpy+python)" if (has_np_seed and has_py_seed) \
            else "torch only (np/python not seeded)"
        return report(True, "3. train_generator seeding", f"seed param + {extra}")
    return report(False, "3. train_generator seeding",
                  "need 'seed' param and torch.manual_seed(seed) inside")


def check_train_ensemble(cfg) -> bool:
    """[4] train_ensemble.py exists, loops seeds, distinct per-seed output path."""
    te = ROOT / "scripts" / "train_ensemble.py"
    if not te.exists():
        return report(False, "4. train_ensemble", "scripts/train_ensemble.py missing")

    src = te.read_text()
    loops_seeds = ("cfg.ensemble.seeds" in src) and ("seed" in src and "for " in src)
    # distinct per-seed checkpoint: generator_seed{seed}.pt has no overwrite risk
    distinct = re.search(r"generator_seed\{?seed\}?\.pt", src) is not None
    seeds = list(cfg.ensemble.seeds)
    outputs = {f"checkpoints/generator_seed{s}.pt" for s in seeds}
    print(f"      seeds from config: {seeds}")
    print(f"      distinct output paths: {sorted(outputs)} (n={len(outputs)})")

    if loops_seeds and distinct and len(outputs) == len(seeds):
        return report(True, "4. train_ensemble distinct outputs",
                      f"{len(seeds)} members, distinct paths, no overwrite")
    return report(False, "4. train_ensemble distinct outputs",
                  "check loops over ensemble.seeds and distinct generator_seed{seed}.pt")


def check_output_dirs(cfg) -> bool:
    """[5] checkpoint_dir & output_dir exist as folders (create if missing).

    PREFLIGHT_MOUNT (test-only) remaps the Drive prefix for existence checks so a
    full green run can be exercised on a non-Colab machine.
    """
    mount = os.environ.get("PREFLIGHT_MOUNT", "")
    results = []
    for label, p in (("paths.checkpoint_dir", cfg.paths.checkpoint_dir),
                     ("paths.output_dir", cfg.paths.output_dir)):
        target = p
        if mount and str(p).startswith(DRIVE_PREFIX):
            target = os.path.join(mount, str(p).lstrip(DRIVE_PREFIX).lstrip("/"))
        # Guard: on Windows, os.makedirs("/content/...") would silently create a
        # folder at the current DRIVE ROOT (e.g. E:\content\...) — a false green
        # and a stray directory. Fail loudly instead unless PREFLIGHT_MOUNT remaps
        # to a writable local path (test/CI only).
        if (not mount) and os.name == "nt" and str(p).startswith(DRIVE_PREFIX):
            ok = False
            print(f"      {label:<20} = {p}  -> NOT checked "
                  f"(Drive path can't be verified on Windows; run on Colab "
                  f"or set PREFLIGHT_MOUNT)")
        else:
            try:
                os.makedirs(target, exist_ok=True)
                ok = os.path.isdir(target)
                print(f"      {label:<20} = {target}  "
                      f"-> {'exists' if ok else 'NOT a dir'}")
            except Exception as e:  # noqa: BLE001
                ok = False
                print(f"      {label:<20} = {target}  -> error: {e}")
        results.append(ok)
    if mount:
        print(f"      (PREFLIGHT_MOUNT={mount!r}: Drive prefix remapped for this check only)")
    if all(results):
        return report(True, "5. Output dirs exist", "checkpoint_dir + output_dir present")
    return report(False, "5. Output dirs exist",
                  "could not ensure checkpoint_dir/output_dir exist (Drive mounted?)")


def main() -> int:
    ap = argparse.ArgumentParser(description="Pre-flight gate before real training.")
    ap.add_argument("--config", default=str(ROOT / "configs" / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    print(banner("PRE-FLIGHT CHECK"))
    print(f"  config      : {args.config}")
    print(f"  repo root   : {ROOT}\n")

    results = [
        check_drive_paths(cfg),
        check_loss_weights(cfg),
        check_train_generator(),
        check_train_ensemble(cfg),
        check_output_dirs(cfg),
    ]
    n_pass, n = sum(results), len(results)
    print(banner(f"RESULT {n_pass}/{n} PASSED"))

    if all(results):
        print(f"{GREEN}SAFE TO LAUNCH{RESET}")
        return 0
    print(f"{RED}DO NOT LAUNCH{RESET} — fix the failing check(s) above first.")
    return 1


if __name__ == "__main__":
    sys.exit(main())