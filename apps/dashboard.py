"""Phase 8 — Streamlit dashboard.

Bare upload -> translation -> detections -> uncertainty/Grad-CAM toggles ->
risk color-coding with failure-mode text. Pipline ensemble functions live in
``scripts/pipeline.py``; Risk / failure / drawing logic in ``apps/dashboard_core.py``
(stay free of Streamlit so it is unit-testable).

RUN (needs Streamlit, e.g. on Colab with the Drive checkpoints mounted)::

    pip install streamlit
    streamlit run apps/dashboard.py

The config's ``paths.checkpoint_dir`` may be overridden from the sidebar for a
local run against ``./checkpoints``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import streamlit as st

from PIL import Image

from utils.config_loader import load_config
from scripts.pipeline import (
    _load_sar,
    _default_device,
    generator_from_config,
    detector_from_config,
    run_generator_ensemble,
    run_detector_ensemble,
)
from utils.gradcam import detector_gradcam, generator_gradcam
from apps.dashboard_core import (
    risk_color,
    failure_text,
    draw_detections,
    _numpy_rgb,
)

COLORS = {"green": (34, 197, 94), "amber": (245, 158, 11), "red": (239, 68, 68)}


@st.cache_resource
def _load_config():
    return load_config(str(ROOT / "configs" / "config.yaml"))


def _device() -> torch.device:
    return _default_device()


def _to_heat_pil(heat: np.ndarray) -> Image.Image:
    """Map a [0,1] heatmap to a jet-colored PIL image (for overlaying)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    norm_ = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    colored = cm.jet(norm_)[..., :3]  # (H, W, 3) in [0,1]
    return Image.fromarray((colored * 255).astype(np.uint8))


def main() -> None:
    st.set_page_config(page_title="SAR Translation Dashboard", layout="wide")
    cfg = _load_config()
    dev = _device()

    st.title("SAR → RGB translation + ship detection (Phase 8)")
    st.caption(
        "Upload a SAR image; runs the generator ensemble, then the detector "
        "ensemble, and colors each detection by confidence + uncertainty."
    )

    # ---- sidebar: single ground-truth config + checkpoint path override ----
    with st.sidebar:
        st.header("Config")
        default_members = int(cfg.ensemble.num_members)
        num_members = st.number_input("Ensemble members", 1, 5, default_members)
        ckpt_dir = st.text_input(
            "Checkpoint dir", value=str(cfg.paths.checkpoint_dir)
        )
        st.caption("Real checkpoints live on Drive; over-ride to ./checkpoints for local runs.")

    ckpt_dir = Path(ckpt_dir)

    # ---- 8.1 upload ----
    uploaded = st.file_uploader("Upload a SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if uploaded is None:
        st.info("Upload a SAR image to begin.")
        return

    # ---------------------------------------------------------------- 8.1 ---
    # Display the raw uploaded bytes before any preprocessing.
    st.subheader("8.1 — Raw upload")
    raw = uploaded.getvalue()
    st.image(raw, caption="uploaded file (raw bytes)", use_container_width=True)

    # ---------------------------------------------------------------- 8.2 ---
    st.subheader("8.2 — Translation")
    # write bytes to a temp file the pipeline's _load_sar can open
    suffix = Path(uploaded.name).suffix or ".png"
    tmp = Path(f"/tmp/sar_dash{suffix}")
    tmp.write_bytes(raw)

    with st.spinner("Running generator ensemble…"):
        sar = _load_sar(str(tmp), cfg).to(dev)  # (C, H, W) [-1,1]
        sar_b = sar.unsqueeze(0)               # (1, C, H, W)
        mean_rgb, var_map = run_generator_ensemble(
            sar_b, ckpt_dir, num_members, config=cfg
        )  # mean_rgb (1,C,H,W)[-1,1]; var_map (1,H,W)
    rgb = mean_rgb[0]                       # (C,H,W) [-1,1]
    rgb_01 = ((rgb + 1.0).clamp(0, 1)) / 2.0  # -> [0,1] for the detector

    sar_01 = ((sar + 1.0) / 2.0).clamp(0, 1)
    if sar_01.shape[0] == 1:
        sar_01 = sar_01.repeat(3, 1, 1)
    sar_pil = Image.fromarray((_numpy_rgb(sar_01) * 255).astype(np.uint8))
    rgb_pil = Image.fromarray((_numpy_rgb(rgb_01) * 255).astype(np.uint8))

    c1, c2 = st.columns(2)
    c1.image(sar_pil, caption="Input SAR", use_container_width=True)
    c2.image(rgb_pil, caption="Translated RGB (ensemble mean)", use_container_width=True)

    # ---------------------------------------------------------------- 8.3 ---
    st.subheader("8.3 — Detections")
    det_res = run_detector_ensemble(rgb_01, ckpt_dir, num_members, config=cfg)
    dets = det_res["merged"]

    if not dets:
        st.warning("No detections above the score threshold.")
    else:
        drawn = draw_detections(_numpy_rgb(rgb_01), dets)
        c1, c2 = st.columns(2)
        c1.image(drawn, caption="Detections (risk-colored)", use_container_width=True)
        # summarize ensemble agreement
        c2.metric("Detections found", det_res["n_dets"])
        c2.metric("Agreed by all members", det_res["n_all"])
        c2.metric("Single-model (high unc.)", det_res["n_single"])
        with st.expander("Detail table"):
            st.dataframe(
                [
                    {
                        "score": round(d["score"], 3),
                        "uncertainty": round(d["uncertainty"], 4),
                        "models": d["count"],
                        "risk": risk_color(d["score"], d["uncertainty"]),
                    }
                    for d in dets
                ]
            )

    # ---------------------------------------------------------------- 8.4 ---
    st.subheader("8.4 — Uncertainty heatmap + Grad-CAM toggles")
    with st.expander("Toggle overlays", expanded=True):
        show_unc = st.checkbox("Show generator-ensemble uncertainty heatmap")
        show_grad_gen = st.checkbox("Show generator Grad-CAM")
        show_grad_det = st.checkbox("Show detector Grad-CAM")

    def _blend_heat(heat: np.ndarray, base: Image.Image) -> Image.Image:
        """Mix a jet-colored heatmap under ``base`` (RGB) at ~45% opacity."""
        heat_rgba = _to_heat_pil(heat).convert("RGBA")
        base_rgba = base.convert("RGBA")
        under = Image.blend(base_rgba, heat_rgba, 0.45)
        # put the image back on top so structure stays visible behind the heat
        return Image.alpha_composite(under, base_rgba).convert("RGB")

    if show_unc:
        hv = var_map[0].float().cpu().numpy()
        st.image(_blend_heat(hv, rgb_pil),
                 caption="Uncertainty heatmap overlaid (jet = higher uncertainty)",
                 use_container_width=True)

    # Grad-CAM overlays need a loaded member model; load on demand.
    st.markdown("**Grad-CAM overlays**")
    gc1, gc2 = st.columns(2)

    gen_gc = None
    if show_grad_gen:
        with st.spinner("Building generator Grad-CAM…"):
            gen = generator_from_config(cfg, dev).eval()
            gen.load_state_dict(torch.load(
                ckpt_dir / f"generator_seed{cfg.ensemble.seeds[0]}.pt",
                map_location=dev))
            heat = generator_gradcam(gen, sar, save=False).numpy()
            gen_gc = _blend_heat(heat, rgb_pil)
    if gen_gc is not None:
        gc1.image(gen_gc, caption="Generator Grad-CAM overlay", use_container_width=True)
    else:
        gc1.info("Toggle on to render.")

    det_gc = None
    if show_grad_det:
        with st.spinner("Building detector Grad-CAM…"):
            det = detector_from_config(cfg, dev).eval()
            det.load_state_dict(torch.load(
                ckpt_dir / f"detector_seed{cfg.ensemble.seeds[0]}.pt",
                map_location=dev))
            heat, score = detector_gradcam(det, rgb_01, save=False)
            det_gc = _blend_heat(heat.numpy(), rgb_pil)
    if det_gc is not None:
        gc2.image(det_gc, caption="Detector Grad-CAM overlay", use_container_width=True)
    else:
        gc2.info("Toggle on to render.")

    # ---------------------------------------------------------------- 8.5 ---
    st.subheader("8.5 — Risk color-coding + failure mode")

    if not dets:
        st.info("No detections to color.")
        return

    boxes = [d["box"] for d in dets]
    metric_boxes = boxes
    txt = failure_text(_numpy_rgb(rgb_01), metric_boxes)
    st.markdown(f"**Failure-mode explanation:** {txt}")

    # list every detection with a color badge
    for d in dets:
        r = risk_color(d["score"], d["uncertainty"])
        col = COLORS[r]
        st.markdown(
            f"<span style='background:{'#' + '%02x%02x%02x' % col}; color:white;"
            f"padding:2px 8px;border-radius:8px'>{r.upper()}</span> "
            f"score={d['score']:.2f}  uncertainty={d['uncertainty']:.2f}  "
            f"(seen by {d['count']}/{num_members} members)",
            unsafe_allow_html=True,
        )

    # sanity: does the current image produce at least one amber/red?
    amber_red = [d for d in dets if risk_color(d["score"], d["uncertainty"]) != "green"]
    if amber_red:
        st.success(
            "CHECK: at least one amber/red detection present with a sensible "
            "explanation (above)."
        )
    else:
        st.info("All detections green — no amber/red case to explain here.")


if __name__ == "__main__":  # Colab/streamlit run only.
    main()