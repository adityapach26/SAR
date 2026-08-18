"""Pre-flight gate: warn loudly if auxiliary losses are disabled.

Loads configs/config.yaml and checks lambda_perceptual / lambda_semantic.
If either is 0, training would run as GAN+L1 only (the perceptual and
semantic stages of the Book would be silently skipped), so this prints a
visually distinct warning banner and exits non-zero. Use it as a gate before
any real training run:

    python scripts/verify_loss_config.py && python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.config_loader import load_config  # noqa: E402


def _banner(text: str) -> str:
    """A visually distinct warning block, clearly not a routine log line."""
    width = max(len(line) for line in text.splitlines()) + 4
    bar = "!" * width
    return f"\n{bar}\n! {text}\n{bar}"


def main() -> int:
    cfg = load_config(str(ROOT / "configs" / "config.yaml"))
    lp = cfg.loss.lambda_perceptual
    ls = cfg.loss.lambda_semantic
    print("verify_loss_config: "
          f"lambda_perceptual={lp}  lambda_semantic={ls}")

    missing = [name for name, val in (("lambda_perceptual", lp),
                                      ("lambda_semantic", ls)) if val == 0]

    if not missing:
        print("OK: both auxiliary losses are enabled.")
        return 0

    lines = "\n".join(
        f"  !! {name} == 0"
        for name in missing
    )
    print(_banner(
        "WARNING: auxiliary losses are DISABLED.\n"
        f"{lines}\n\n"
        "Training will proceed as GAN + L1 ONLY.\n"
        "The perceptual and/or semantic stages of the Book will be skipped.\n"
        "Set the corresponding lambda to > 0 to enable them."
    ))
    return 1


if __name__ == "__main__":
    sys.exit(main())