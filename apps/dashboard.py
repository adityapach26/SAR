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


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0B1117;
            --panel: #111923;
            --card: #17212B;
            --card-2: #101821;
            --blue: #4F83B5;
            --gold: #C8A85B;
            --text: #E6EDF3;
            --muted: #8B98A5;
            --green: #22C55E;
            --amber: #F59E0B;
            --red: #EF4444;
            --border: rgba(200, 168, 91, 0.24);
            --blue-border: rgba(79, 131, 181, 0.34);
        }
        .stApp {
            background:
                radial-gradient(circle at 20% 0%, rgba(79, 131, 181, 0.16), transparent 32rem),
                radial-gradient(circle at 92% 8%, rgba(200, 168, 91, 0.10), transparent 30rem),
                var(--bg);
            color: var(--text);
        }
        [data-testid="stHeader"] { background: rgba(11, 17, 23, 0.72); }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #091017 0%, #111923 100%);
            border-right: 1px solid rgba(79, 131, 181, 0.26);
        }
        [data-testid="stSidebar"] * { color: var(--text); }
        .block-container { padding-top: 1.1rem; max-width: 1500px; }
        .command-header {
            border: 1px solid var(--blue-border);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(23, 33, 43, 0.98), rgba(13, 21, 30, 0.98));
            box-shadow: 0 14px 40px rgba(0,0,0,0.28);
            padding: 1.05rem 1.25rem;
            margin-bottom: 1rem;
        }
        .brand {
            color: var(--gold);
            font-size: 0.8rem;
            letter-spacing: 0.22em;
            font-weight: 800;
            text-transform: uppercase;
        }
        .title {
            color: var(--text);
            font-size: clamp(1.45rem, 3vw, 2.25rem);
            font-weight: 800;
            letter-spacing: 0.03em;
            line-height: 1.1;
            margin-top: 0.18rem;
        }
        .subtitle {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.4rem;
        }
        .status-pill, .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border: 1px solid var(--blue-border);
            border-radius: 999px;
            padding: 0.34rem 0.68rem;
            color: var(--text);
            background: rgba(79, 131, 181, 0.10);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        .badge.gold { border-color: var(--border); background: rgba(200,168,91,0.12); color: #F2D98B; }
        .badge.green { border-color: rgba(34,197,94,0.32); background: rgba(34,197,94,0.12); color: #86EFAC; }
        .section-card {
            border: 1px solid rgba(79, 131, 181, 0.24);
            border-radius: 18px;
            background: linear-gradient(180deg, rgba(23, 33, 43, 0.98), rgba(16, 24, 33, 0.98));
            box-shadow: 0 12px 30px rgba(0,0,0,0.23);
            padding: 1rem;
            margin: 0.45rem 0 1rem 0;
        }
        .section-card.gold { border-color: var(--border); }
        .section-title {
            color: var(--text);
            font-weight: 800;
            letter-spacing: 0.08em;
            font-size: 0.94rem;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .section-subtitle { color: var(--muted); font-size: 0.82rem; margin-bottom: 0.75rem; }
        .empty-state, .warning-state, .info-state {
            border: 1px dashed rgba(200,168,91,0.40);
            border-radius: 16px;
            background: rgba(200,168,91,0.08);
            color: var(--text);
            padding: 1rem;
            font-weight: 650;
        }
        .warning-state { border-color: rgba(245,158,11,0.50); background: rgba(245,158,11,0.10); }
        .info-state { border-color: rgba(79,131,181,0.40); background: rgba(79,131,181,0.10); }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.45rem 0 0.75rem;
        }
        .metric-card {
            border: 1px solid rgba(79, 131, 181, 0.26);
            background: rgba(11, 17, 23, 0.50);
            border-radius: 14px;
            padding: 0.82rem;
        }
        .metric-label {
            color: var(--muted);
            font-size: 0.72rem;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            font-weight: 800;
        }
        .metric-value { color: var(--text); font-size: 1.55rem; font-weight: 900; line-height: 1.1; }
        .risk-row {
            display: flex;
            gap: 0.6rem;
            align-items: center;
            border: 1px solid rgba(79,131,181,0.18);
            border-radius: 12px;
            padding: 0.5rem 0.65rem;
            margin: 0.34rem 0;
            background: rgba(11,17,23,0.34);
        }
        .risk-pill {
            color: white;
            padding: 0.16rem 0.55rem;
            border-radius: 999px;
            font-size: 0.72rem;
            font-weight: 900;
            min-width: 4.2rem;
            text-align: center;
        }
        .nav-title {
            color: var(--gold);
            font-weight: 900;
            letter-spacing: 0.16em;
            font-size: 0.86rem;
            margin: 0.25rem 0 0.8rem;
        }
        .nav-item {
            border: 1px solid rgba(79,131,181,0.18);
            background: rgba(23,33,43,0.68);
            border-radius: 12px;
            padding: 0.55rem 0.65rem;
            margin: 0.32rem 0;
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            color: var(--muted);
        }
        .nav-item.active {
            color: var(--text);
            border-color: var(--border);
            background: linear-gradient(90deg, rgba(200,168,91,0.18), rgba(79,131,181,0.08));
        }
        div[data-testid="stFileUploader"] section {
            background: rgba(11,17,23,0.42);
            border: 1px dashed rgba(200,168,91,0.42);
            border-radius: 16px;
        }
        div[data-testid="stMetric"] {
            background: rgba(11,17,23,0.38);
            border: 1px solid rgba(79,131,181,0.18);
            border-radius: 14px;
            padding: 0.65rem;
        }
        .stDataFrame { border: 1px solid rgba(79,131,181,0.22); border-radius: 14px; overflow: hidden; }
        h1, h2, h3 { color: var(--text); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card_start(title: str, subtitle: str = "", gold: bool = False) -> None:
    cls = "section-card gold" if gold else "section-card"
    st.markdown(
        f"""
        <div class=\"{cls}\">
            <div class=\"section-title\">{title}</div>
            <div class=\"section-subtitle\">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def _card_end() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def _metric_cards(det_res, num_members: int) -> None:
    st.markdown(
        f"""
        <div class=\"metric-grid\">
            <div class=\"metric-card\"><div class=\"metric-label\">Detected Objects</div><div class=\"metric-value\">{det_res['n_dets']}</div></div>
            <div class=\"metric-card\"><div class=\"metric-label\">All-Member Agreement</div><div class=\"metric-value\">{det_res['n_all']}/{num_members}</div></div>
            <div class=\"metric-card\"><div class=\"metric-label\">High Uncertainty</div><div class=\"metric-value\">{det_res['n_single']}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="SAR.AI Command Center", layout="wide")
    _inject_css()
    cfg = _load_config()
    dev = _device()

    with st.sidebar:
        st.markdown('<div class="nav-title">SAR.AI NAVIGATION</div>', unsafe_allow_html=True)
        nav = st.radio(
            "Section",
            ["DASHBOARD", "DATA", "PRE-PROCESSING", "OBJECT DETECTION", "ANALYSIS", "SETTINGS"],
            label_visibility="collapsed",
        )
        for item in ["DASHBOARD", "DATA", "PRE-PROCESSING", "OBJECT DETECTION", "ANALYSIS", "SETTINGS"]:
            active = " active" if item == nav else ""
            st.markdown(f'<div class="nav-item{active}">{item}</div>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### SETTINGS")
        model_set = st.selectbox("Model set", ["Agriculture", "Water Best"])
        default_members = min(int(cfg.ensemble.num_members), 3)
        num_members = st.number_input("Ensemble members", 1, 3, default_members)
        ckpt_dir = st.text_input("Checkpoint dir", value=str(cfg.paths.checkpoint_dir))
        st.caption("Real checkpoints live on Drive; override to ./checkpoints for local runs.")

        st.markdown("#### Detection")
        score_threshold = st.slider("Detection score threshold", 0.0, 1.0, 0.5, 0.05)
        iou_thresh = st.slider("IoU merge threshold", 0.0, 1.0, 0.5, 0.05)

        st.markdown("#### Visualization")
        show_unc = st.checkbox("Show generator-ensemble uncertainty heatmap")
        show_grad_gen = st.checkbox("Show generator Grad-CAM")
        show_grad_det = st.checkbox("Show detector Grad-CAM")
        heatmap_opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.45, 0.05)

    ckpt_dir = Path(ckpt_dir)
    num_members = int(num_members)
    water_best_names = [f"water_finetune_seed{s}_best.pt" for s in cfg.ensemble.seeds[:num_members]]
    generator_checkpoint_names = water_best_names if model_set == "Water Best" else None
    first_generator_checkpoint = (
        ckpt_dir / water_best_names[0]
        if model_set == "Water Best"
        else ckpt_dir / f"generator_seed{cfg.ensemble.seeds[0]}.pt"
    )

    st.markdown(
        f"""
        <div class=\"command-header\">
            <div style=\"display:flex; justify-content:space-between; gap:1rem; align-items:flex-start; flex-wrap:wrap;\">
                <div>
                    <div class=\"brand\">SAR.AI</div>
                    <div class=\"title\">ADVANCED SYNTHETIC APERTURE RADAR ANALYTICS PLATFORM</div>
                    <div class=\"subtitle\">Maritime Intelligence • SAR Translation • Vessel Detection • Uncertainty Analysis</div>
                </div>
                <div class=\"status-pill\">DEVICE: {str(dev).upper()}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_left, top_right = st.columns([0.92, 1.08], gap="large")

    with top_left:
        _card_start("1. INPUT: SAR IMAGE LOAD", "Upload a SAR image for translation and detection.", gold=True)
        uploaded = st.file_uploader("Upload SAR image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        st.caption("Accepted file types: PNG, JPG/JPEG, TIFF")
        if uploaded is None:
            st.markdown(
                '<div class="empty-state">Awaiting SAR image upload. The inference pipeline will remain idle until a file is provided.</div>',
                unsafe_allow_html=True,
            )
            _card_end()
            return
        raw = uploaded.getvalue()
        try:
            raw_img = Image.open(uploaded)
            raw_dims = raw_img.size
            uploaded.seek(0)
        except Exception:
            raw_dims = None
        st.markdown(f'<span class="badge gold">FILE: {uploaded.name}</span>', unsafe_allow_html=True)
        if raw_dims is not None:
            st.markdown(f'<span class="badge">DIMENSIONS: {raw_dims[0]} × {raw_dims[1]}</span>', unsafe_allow_html=True)
        st.image(raw, caption="Uploaded SAR preview", use_container_width=True)
        _card_end()

    suffix = Path(uploaded.name).suffix or ".png"
    tmp = Path(f"/tmp/sar_dash{suffix}")
    tmp.write_bytes(raw)

    if model_set == "Water Best":
        missing = [name for name in water_best_names if not (ckpt_dir / name).exists()]
        if missing:
            st.markdown(
                '<div class="warning-state">Missing Water Best generator checkpoint(s): '
                + ", ".join(missing)
                + '</div>',
                unsafe_allow_html=True,
            )
            return

    with st.spinner("Running generator ensemble…"):
        sar = _load_sar(str(tmp), cfg).to(dev)  # (C, H, W) [-1,1]
        sar_b = sar.unsqueeze(0)               # (1, C, H, W)
        mean_rgb, var_map = run_generator_ensemble(
            sar_b,
            ckpt_dir,
            num_members,
            config=cfg,
            checkpoint_filenames=generator_checkpoint_names,
        )  # mean_rgb (1,C,H,W)[-1,1]; var_map (1,H,W)
    rgb = mean_rgb[0]                       # (C,H,W) [-1,1]
    rgb_01 = ((rgb + 1.0).clamp(0, 1)) / 2.0  # -> [0,1] for the detector

    sar_01 = ((sar + 1.0) / 2.0).clamp(0, 1)
    if sar_01.shape[0] == 1:
        sar_01 = sar_01.repeat(3, 1, 1)
    sar_pil = Image.fromarray((_numpy_rgb(sar_01) * 255).astype(np.uint8))
    rgb_pil = Image.fromarray((_numpy_rgb(rgb_01) * 255).astype(np.uint8))

    with top_right:
        _card_start("2. PRE-PROCESSING: SAR → RGB", "Generator ensemble translation with selected checkpoint set.")
        st.markdown(
            f'<span class="badge gold">MODEL SET: {model_set}</span> '
            f'<span class="badge">ENSEMBLE MEMBERS: {num_members}</span>',
            unsafe_allow_html=True,
        )
        img_l, img_r = st.columns(2)
        img_l.image(sar_pil, caption="RAW SAR INPUT", use_container_width=True)
        img_r.image(rgb_pil, caption="TRANSLATED RGB", use_container_width=True)
        _card_end()

    _card_start("3. CORE: OBJECT DETECTION", "Existing detector ensemble over the translated RGB output.")
    st.markdown(
        f'<span class="badge">SCORE THRESHOLD: {score_threshold:.2f}</span> '
        f'<span class="badge">IOU MERGE: {iou_thresh:.2f}</span> '
        f'<span class="badge green">CHECKPOINT DIR ACTIVE</span>',
        unsafe_allow_html=True,
    )
    det_res = run_detector_ensemble(
        rgb_01,
        ckpt_dir,
        num_members,
        config=cfg,
        score_threshold=score_threshold,
        iou_thresh=iou_thresh,
    )
    dets = det_res["merged"]
    _metric_cards(det_res, num_members)

    det_img_col, det_table_col = st.columns([1.42, 0.58], gap="large")
    with det_img_col:
        if not dets:
            st.markdown(
                '<div class="warning-state">No detections above the score threshold.</div>',
                unsafe_allow_html=True,
            )
            st.image(rgb_pil, caption="Detection panel idle — no boxes to render", use_container_width=True)
        else:
            drawn = draw_detections(_numpy_rgb(rgb_01), dets)
            st.image(drawn, caption="Detections (risk-colored)", use_container_width=True)
    with det_table_col:
        st.markdown('<div class="section-subtitle">Detection summary and compact detail table.</div>', unsafe_allow_html=True)
        if dets:
            with st.expander("Detection detail table", expanded=True):
                st.dataframe(
                    [
                        {
                            "score": round(d["score"], 3),
                            "uncertainty": round(d["uncertainty"], 4),
                            "models": d["count"],
                            "risk": risk_color(d["score"], d["uncertainty"]),
                        }
                        for d in dets
                    ],
                    use_container_width=True,
                )
        else:
            st.markdown('<div class="info-state">No detection rows to display.</div>', unsafe_allow_html=True)
    _card_end()

    def _blend_heat(heat: np.ndarray, base: Image.Image) -> Image.Image:
        """Mix a jet-colored heatmap under ``base`` (RGB)."""
        heat_rgba = _to_heat_pil(heat).convert("RGBA")
        base_rgba = base.convert("RGBA")
        return Image.blend(base_rgba, heat_rgba, heatmap_opacity).convert("RGB")

    analysis_l, analysis_c, analysis_r = st.columns([1.0, 1.0, 1.0], gap="large")

    with analysis_l:
        _card_start("4. OUTPUT & ANALYSIS", "Risk color-coding and failure-mode status.", gold=True)
        if not dets:
            st.markdown('<div class="info-state">No detections to color.</div>', unsafe_allow_html=True)
        else:
            boxes = [d["box"] for d in dets]
            metric_boxes = boxes
            txt = failure_text(_numpy_rgb(rgb_01), metric_boxes)
            st.markdown(f'<div class="info-state"><strong>Failure-mode explanation:</strong><br>{txt}</div>', unsafe_allow_html=True)
            for d in dets:
                r = risk_color(d["score"], d["uncertainty"])
                col = COLORS[r]
                st.markdown(
                    f"<div class='risk-row'>"
                    f"<span class='risk-pill' style='background:{'#' + '%02x%02x%02x' % col}'>{r.upper()}</span>"
                    f"<span>score={d['score']:.2f} · uncertainty={d['uncertainty']:.2f} · "
                    f"seen by {d['count']}/{num_members} members</span></div>",
                    unsafe_allow_html=True,
                )
            amber_red = [d for d in dets if risk_color(d["score"], d["uncertainty"]) != "green"]
            if amber_red:
                st.markdown(
                    '<div class="warning-state">CHECK: at least one amber/red detection present with a sensible explanation.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown('<div class="info-state">All detections green — no amber/red case to explain here.</div>', unsafe_allow_html=True)
        _card_end()

    with analysis_c:
        _card_start("UNCERTAINTY", "Generator ensemble variance visualization.")
        if show_unc:
            hv = var_map[0].float().cpu().numpy()
            st.image(_blend_heat(hv, rgb_pil), caption="Uncertainty heatmap overlaid (jet = higher uncertainty)", use_container_width=True)
        else:
            st.markdown('<div class="info-state">Enable the uncertainty heatmap in Settings.</div>', unsafe_allow_html=True)
        _card_end()

    with analysis_r:
        _card_start("EXPLAINABILITY", "Grad-CAM overlays from existing utilities.")
        gen_gc = None
        if show_grad_gen:
            with st.spinner("Building generator Grad-CAM…"):
                gen = generator_from_config(cfg, dev).eval()
                gen.load_state_dict(torch.load(first_generator_checkpoint, map_location=dev))
                heat = generator_gradcam(gen, sar, save=False).numpy()
                gen_gc = _blend_heat(heat, rgb_pil)
        if gen_gc is not None:
            st.image(gen_gc, caption="Generator Grad-CAM overlay", use_container_width=True)
        else:
            st.markdown('<div class="info-state">Generator Grad-CAM is disabled in Settings.</div>', unsafe_allow_html=True)

        det_gc = None
        det_gc_msg = "Detector Grad-CAM is disabled in Settings."
        if show_grad_det:
            if not dets:
                det_gc_msg = "No detections available for Detector Grad-CAM."
            else:
                with st.spinner("Building detector Grad-CAM…"):
                    det = detector_from_config(cfg, dev).eval()
                    det.load_state_dict(torch.load(
                        ckpt_dir / f"detector_seed{cfg.ensemble.seeds[0]}.pt",
                        map_location=dev))
                    try:
                        heat, score = detector_gradcam(det, rgb_01, save=False)
                        det_gc = _blend_heat(heat.numpy(), rgb_pil)
                    except RuntimeError as e:
                        if "detector found no detections" not in str(e):
                            raise
                        det_gc_msg = "No detections available for Detector Grad-CAM."
        if det_gc is not None:
            st.image(det_gc, caption="Detector Grad-CAM overlay", use_container_width=True)
        else:
            st.markdown(f'<div class="info-state">{det_gc_msg}</div>', unsafe_allow_html=True)
        _card_end()


if __name__ == "__main__":  # Colab/streamlit run only.
    main()
