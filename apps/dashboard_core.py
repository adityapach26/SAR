"""Pure decision logic for the Phase 8 dashboard.

Kept free of Streamlit (only PIL + numpy) so the risk-color / failure-mode /
drawing rules can be unit-tested locally, even though the ``streamlit`` UI in
``dashboard.py`` only runs where Streamlit is installed (Colab).

The risk model combines the two signals the pipeline already produces per
merged detection:

    score      -- averaged confidence across ensemble members.
    uncertainty-- variance of member confidences; ``1.0`` when found by only one
                  member (never agreed upon -> explicitly high uncertainty).

A "green" detection is both confident AND agreed-upon. A "red" one is either
improbably confident or strongly disagreed-about. Everything in between is
"amber". For amber/red we attach the failure-mode explanation from
``utils.failure_analyzer``.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from utils.failure_analyzer import analyze_failure_mode  # noqa: E402

# Detection risk thresholds (score in [0,1], uncertainty in [0,1]).
RED_SCORE_MAX = 0.5      # below this confidence -> red
RED_UNC_MIN = 0.4        # above this uncertainty -> red
GREEN_SCORE_MIN = 0.6    # need at least this confidence for green
GREEN_UNC_MAX = 0.15     # and at most this uncertainty for green


def risk_color(score: float, uncertainty: float) -> str:
    """Return ``"green"`` / ``"amber"`` / ``"red"`` for a merged detection."""
    if score is None or uncertainty is None:
        return "amber"
    if score < RED_SCORE_MAX or uncertainty > RED_UNC_MIN:
        return "red"
    if score >= GREEN_SCORE_MIN and uncertainty <= GREEN_UNC_MAX:
        return "green"
    return "amber"


def image_metrics(rgb_01: np.ndarray):
    """Speckle-variance proxy + contrast from a ``[0,1]`` RGB image ``(H,W,3)``.

    Speckle is the dominant SAR noise; the variance of the normalized intensity
    estimates it. Contrast = normalized (max - min), so a washed-out image reads
    as near-zero. Mirrors the proxy names ``utils.failure_analyzer`` expects.
    """
    gray = np.asarray(rgb_01, dtype=np.float64).mean(axis=2)  # (H, W) in [0,1]
    speckle_variance = float(gray.var())
    contrast = float(gray.max() - gray.min())
    return speckle_variance, contrast


def failure_text(rgb_01: np.ndarray, boxes) -> str:
    """Failure-mode explanation for amber/red detections on ``rgb_01``."""
    speckle_var, contrast = image_metrics(rgb_01)
    return analyze_failure_mode(
        speckle_variance=speckle_var, contrast=contrast, boxes=boxes
    )


def _numpy_rgb(rgb_01) -> np.ndarray:
    """Coerce a translated-RGB input to a ``(H, W, 3)`` uint8 float ``[0,1]`` array."""
    if hasattr(rgb_01, "detach"):  # torch tensor
        rgb_01 = rgb_01.detach().cpu().numpy()
    a = np.asarray(rgb_01)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3 and a.shape[0] in (1, 3):  # (C, H, W) -> (H, W, C)
        if a.shape[0] == 1:
            a = np.repeat(a[0], 3, axis=0)
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:  # single grayscale channel -> RGB
        a = np.stack([a, a, a], axis=-1)
    return np.clip(a, 0.0, 1.0)


def draw_detections(rgb_01, dets, risk_fn=risk_color, box_label="ship"):
    """Return a PIL RGB image of ``rgb_01`` with risk-colored detection boxes.

    ``dets`` is the pipeline's merged list ``{box, score, uncertainty, count}``.
    Each box is drawn its risk color with a caption ``ship s=0.xx u=0.xx`` and a
    small filled risk badge dot in the top-left corner of the image.
    """
    img = Image.fromarray((_numpy_rgb(rgb_01) * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img, "RGBA")

    colors = {
        "green": (34, 197, 94, 255),
        "amber": (245, 158, 11, 255),
        "red": (239, 68, 68, 255),
    }
    # risk badge legend at top-left
    bw, bh = 12, 12
    for i, (label, (r, g, b, _)) in enumerate(colors.items()):
        x, y = 4, 4 + i * (bh + 4)
        draw.rectangle([x, y, x + bw, y + bh], fill=(r, g, b, 255))
        draw.text((x + bw + 4, y), label, fill=(255, 255, 255, 255))
    legend = (4, 4 + 3 * (bh + 4) + 6)
    draw.text((4, legend[1]), f"{len(dets)} detection(s)", fill=(255, 255, 255, 255))

    for d in sorted(dets, key=lambda x: x["score"], reverse=True):
        color = risk_fn(d["score"], d["uncertainty"])
        r, g, b, a = colors[color]
        x0, y0, x1, y1 = [float(v) for v in d["box"]]
        width = 3 if color == "green" else 4
        draw.rectangle([x0, y0, x1, y1], outline=(r, g, b, a), width=width)
        caption = f"{box_label} s={d['score']:.2f} u={d['uncertainty']:.2f}"
        draw.text((x0, max(y0 - 16, 0)), caption, fill=(r, g, b, a))

    return img