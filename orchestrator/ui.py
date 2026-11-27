"""
ModelShield Adaptive Red-Team Mission Lab - Live Neural Network Mission Experience
Compact, interactive cybersecurity investigation environment inspired by TryHackMe.
"""
import json
import os
import time
from html import escape
from typing import Any, Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

from orchestrator.copilot import InvestigationCopilot
from orchestrator.engine import RedTeamOrchestrator

# RAM++ is the only target model for this workspace.
from model.ram_adapter import get_ram_model


class ModelConfigurationError(Exception):
    pass

@st.cache_resource
def get_cached_model():
    """Reuse the RAM++ model and device from the shared adapter."""
    try:
        model, device = get_ram_model()
        model.eval()
        return model, device
    except Exception as exc:
        raise ModelConfigurationError(f"RAM++ model loading failed: {exc}") from exc


def inject_mission_css():
    """Inject sleek, dark command-centre cybersecurity aesthetic with polished visual hierarchy."""
    st.markdown(
        """
        <style>
        /* Top Mission Header */
        .ms-header-bar {
            background: linear-gradient(90deg, #090d14 0%, #161b22 100%);
            border: 1px solid #30363d;
            border-left: 4px solid #00ff66;
            border-radius: 6px;
            padding: 12px 18px;
            margin-bottom: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .ms-header-title {
            font-size: 1.25rem;
            font-weight: 800;
            color: #f0f6fc;
            letter-spacing: -0.5px;
        }
        .ms-header-meta {
            color: #8b949e;
            font-size: 0.82rem;
            margin-top: 3px;
        }

        /* KPI Metric Cards */
        .ms-kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 16px;
        }
        .ms-kpi-card {
            background-color: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 14px;
        }
        .ms-kpi-label {
            font-size: 0.75rem;
            color: #8b949e;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        .ms-kpi-val {
            font-size: 1.35rem;
            font-weight: 800;
            color: #f0f6fc;
            margin-top: 2px;
        }
        .ms-kpi-delta {
            font-size: 0.74rem;
            color: #f85149;
            font-weight: 600;
        }

        /* Neural Pipeline Visualizer */
        .ms-pipeline-container {
            background: #080b10;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 18px;
        }
        .ms-pipeline-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 12px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 8px;
        }
        .ms-pipeline-nodes {
            display: flex;
            align-items: stretch;
            justify-content: space-between;
            gap: 8px;
            overflow-x: auto;
            padding: 4px 0;
        }
        .ms-node-card {
            background: #12161f;
            border: 1px solid #30363d;
            border-radius: 6px;
            padding: 10px;
            text-align: center;
            min-width: 120px;
            flex: 1 1 0;
            min-height: 178px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: flex-start;
        }
        .ms-pipeline-nodes > .ms-node-arrow {
            align-self: center;
            flex: 0 0 auto;
        }
        [class*="st-key-path_btn_"] button {
            height: auto !important;
            min-height: 92px;
            padding: 10px 8px !important;
            white-space: normal !important;
        }
        [class*="st-key-path_btn_"] button p {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.35;
        }
        .ms-node-card.active {
            border-color: #58a6ff;
            background: #161e2e;
        }
        .ms-node-card.alert {
            border-color: #f85149;
            background: #251214;
        }
        .ms-node-title {
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            color: #8b949e;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }
        .ms-node-arrow {
            color: #388bfd;
            font-size: 1.1rem;
            font-weight: bold;
            padding: 0 4px;
        }

        /* Attack Path Progression Map */
        .ms-path-container {
            background: #0d1117;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px 14px;
            margin-bottom: 16px;
        }
        .ms-path-step {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.74rem;
            font-weight: 600;
            margin-right: 6px;
            margin-bottom: 6px;
            border: 1px solid #30363d;
            background: #161b22;
            color: #c9d1d9;
        }
        .ms-path-step.completed { border-color: #2ea043; background: rgba(46, 160, 67, 0.15); color: #3fb950; }
        .ms-path-step.escalated { border-color: #da3633; background: rgba(248, 81, 73, 0.15); color: #f85149; }
        .ms-path-step.skipped { border-color: #6e7681; background: rgba(110, 118, 129, 0.1); color: #8b949e; opacity: 0.7; }
        .ms-path-step.current { border-color: #58a6ff; background: rgba(88, 166, 255, 0.2); color: #58a6ff; }

        /* Discovery & Milestone Cards */
        .ms-disc-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .ms-disc-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-left: 3px solid #00ff66;
            border-radius: 6px;
            padding: 10px 12px;
        }
        .ms-disc-card.locked {
            background: #0d1117;
            border: 1px dashed #30363d;
            opacity: 0.6;
        }

        /* Badges */
        .ms-badge {
            display: inline-block;
            padding: 2px 7px;
            border-radius: 4px;
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .ms-badge-green { background: rgba(46, 160, 67, 0.2); color: #3fb950; border: 1px solid #2ea043; }
        .ms-badge-red { background: rgba(248, 81, 73, 0.2); color: #f85149; border: 1px solid #da3633; }
        .ms-badge-amber { background: rgba(210, 153, 34, 0.2); color: #d29922; border: 1px solid #bb8009; }
        .ms-badge-cyan { background: rgba(88, 166, 255, 0.2); color: #58a6ff; border: 1px solid #388bfd; }
        .ms-badge-purple { background: rgba(187, 128, 255, 0.2); color: #d2a8ff; border: 1px solid #bc8cff; }

        /* Raw Terminal Log */
        .ms-terminal-box {
            background-color: #080b0f;
            border: 1px solid #21262d;
            border-radius: 6px;
            padding: 12px;
            font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
            font-size: 0.8rem;
            color: #c9d1d9;
            max-height: 260px;
            overflow-y: auto;
            line-height: 1.45;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_persisted_runs(results_dir: str = "results/runs") -> List[Dict[str, Any]]:
    """Scan and load all historical persisted mission audit runs."""
    if not os.path.exists(results_dir):
        return []
    runs = []
    for fname in sorted(os.listdir(results_dir), reverse=True):
        if fname.endswith(".json"):
            fpath = os.path.join(results_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    runs.append(json.load(f))
            except Exception:
                continue
    return runs


_PLOT_BASE = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font=dict(color="#c9d1d9", size=11))


def plot_confidence_distribution(rows: List[Dict[str, Any]], show_mode: str = "PGD", image_id: int = 1) -> go.Figure:
    """Horizontal bars: measured RAM++ sigmoid tag scores, clean vs attacked (top baseline tags + introduced tags)."""
    key = "pgd" if show_mode == "PGD" else "fgsm"
    rows = [r for r in rows if r.get(key) is not None]
    labels = [f"{r['tag']} (new)" if r.get("is_new") else r["tag"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=[r["clean"] for r in rows], name="Clean Tag Score (%)", orientation="h",
        marker=dict(color="#58a6ff", line=dict(color="#388bfd", width=1)),
    ))
    fig.add_trace(go.Bar(
        y=labels, x=[r[key] for r in rows], name=f"{show_mode} Tag Score (%)", orientation="h",
        marker=dict(color="#f85149", line=dict(color="#da3633", width=1)),
    ))
    fig.update_layout(
        title=dict(text=f"Image #{image_id}: RAM++ Tag Score Distribution ({show_mode})", font=dict(size=13, color="#f0f6fc")),
        barmode="group", **_PLOT_BASE,
        margin=dict(l=110, r=25, t=75, b=45),
        height=max(340, 26 * len(labels) + 125),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(title="Instance Tag Score — sigmoid (%)", gridcolor="#21262d", zerolinecolor="#30363d", range=[0, 100]),
        yaxis=dict(autorange="reversed", gridcolor="#21262d"),
    )
    return fig


def plot_interactive_heatmap(diff_matrix: List[List[float]], epsilon: float, pool: int = 8) -> go.Figure:
    """Interactive Plotly heatmap of the pixel-space perturbation magnitude (pooled for payload size)."""
    n = len(diff_matrix)
    fig = go.Figure(data=go.Heatmap(
        z=diff_matrix,
        colorscale="Inferno",
        colorbar=dict(title=dict(text="Δ", side="right"), thickness=16, len=0.9, x=1.01),
        zmin=0.0,
        zmax=epsilon,
        hovertemplate="Block (X: %{x}, Y: %{y})<br>Mean |Δ|: %{z:.5f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"{n}×{n} Spatial Perturbation Heatmap (|Adversarial − Clean|, raw pixels, {pool}×{pool} pooled)",
                   font=dict(size=13, color="#f0f6fc")),
        **_PLOT_BASE,
        margin=dict(l=45, r=55, t=55, b=45),
        height=500, width=560,
        xaxis=dict(title="Block X", showgrid=False, zeroline=False, range=[-0.5, len(diff_matrix[0])-0.5], constrain="domain"),
        yaxis=dict(title="Block Y", showgrid=False, zeroline=False, range=[n-0.5, -0.5], scaleanchor="x", scaleratio=1, constrain="domain"),
    )
    return fig


def plot_trajectory_chart(trajectory: List[Dict[str, Any]]) -> go.Figure:
    """Baseline-tag preservation across the evaluation checkpoints (clean = 100% by definition)."""
    stages = [t.get("stage", f"Step {i}") for i, t in enumerate(trajectory)]
    pres = [t.get("true_class_prob", t.get("confidence", 0.0)) for t in trajectory]
    labels = [
        f"{t.get('tags_kept')}/{t.get('tags_total')} tags ({v:.1f}%)" if t.get("tags_total") is not None else f"{v:.1f}%"
        for t, v in zip(trajectory, pres)
    ]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=stages, y=pres, mode="lines+markers+text",
        text=[(f"{t.get('tags_kept')}/{t.get('tags_total')}" if t.get("tags_total") is not None else f"{v:.0f}%") for t,v in zip(trajectory,pres)],
        customdata=labels, textposition=["bottom center" if v >= 90 else "top center" for v in pres], cliponaxis=False,
        hovertemplate="Stage: %{x}<br>%{customdata}<extra></extra>",
        line=dict(color="#39d353", width=2),
        marker=dict(color="#58a6ff", size=8, line=dict(color="#f0f6fc", width=1.5)),
        name="Baseline tags preserved (%)",
    ))
    mean_scores = [(t.get("stage"), t.get("mean_baseline_score")) for t in trajectory if t.get("mean_baseline_score") is not None]
    if len(mean_scores) > 1:
        fig.add_trace(go.Scatter(
            x=[m[0] for m in mean_scores], y=[m[1] for m in mean_scores], mode="lines+markers",
            line=dict(color="#d29922", width=1.5, dash="dot"), marker=dict(size=6),
            name="Mean baseline-tag score (%)",
            hovertemplate="Stage: %{x}<br>Mean score of baseline tags: %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(
        title=dict(text="Single-Image Baseline-Tag Preservation Progression", font=dict(size=13, color="#f0f6fc")),
        **_PLOT_BASE,
        margin=dict(l=55, r=45, t=55, b=100),
        height=390,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="left", x=0, font=dict(size=10)),
        yaxis=dict(title="Baseline Tags Preserved (%)", range=[0, 100], gridcolor="#21262d", tickvals=[0,20,40,60,80,100]),
        xaxis=dict(gridcolor="#21262d"),
    )
    return fig


_STAGE_COLS = [("clean", "Clean"), ("fgsm", "FGSM"), ("pgd", "PGD"), ("def_fgsm", "Def. FGSM"), ("def_pgd", "Def. PGD")]


def plot_class_weakness_chart(class_metrics: Dict[str, Any]) -> go.Figure:
    """Grouped bars: measured RAM++ score of each baseline tag under Clean / FGSM / PGD."""
    tags = list(class_metrics.keys())
    fig = go.Figure()
    for stage, name, color in (("clean", "Clean Reference Score", "#3fb950"), ("fgsm", "FGSM Score", "#d29922"), ("pgd", "PGD Score", "#f85149")):
        vals = [class_metrics[t].get(f"{stage}_acc") for t in tags]
        if any(v is not None for v in vals):
            fig.add_trace(go.Bar(x=tags, y=vals, name=name, marker=dict(color=color)))
    fig.update_layout(
        title=dict(text="Baseline-Tag Score Comparison (Clean vs FGSM vs PGD)", font=dict(size=13, color="#f0f6fc")),
        barmode="group", **_PLOT_BASE,
        margin=dict(l=55, r=25, t=85, b=100),
        height=390,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="RAM++ tag score (%)", range=[0, 100], gridcolor="#21262d"),
        xaxis=dict(gridcolor="#21262d", tickangle=-40),
    )
    return fig


def plot_tag_survival_matrix(class_metrics: Dict[str, Any]) -> go.Figure:
    """Heatmap: score of each baseline tag at every measured stage (hover shows whether it stayed active)."""
    tags = list(class_metrics.keys())
    cols = [(k, n) for k, n in _STAGE_COLS if any(class_metrics[t].get(f"{k}_acc") is not None for t in tags)]
    z = [[class_metrics[t].get(f"{k}_acc") for k, _ in cols] for t in tags]
    status = [["active" if class_metrics[t].get(f"{k}_active") else "lost (below RAM++ threshold)" for k, _ in cols] for t in tags]
    fig = go.Figure(data=go.Heatmap(
        z=z, x=[n for _, n in cols], y=tags, customdata=status, colorscale="Viridis", zmin=0, zmax=100,
        colorbar=dict(title=dict(text="Score (%)", side="right")),
        text=[[("—" if v is None else f"{v:.0f}") for v in row] for row in z], texttemplate="%{text}",
        hovertemplate="Tag: %{y}<br>Stage: %{x}<br>Score: %{z:.2f}%<br>Status: %{customdata}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Tag-Score Matrix (Baseline Tags × Evaluation Stage)", font=dict(size=13, color="#f0f6fc")),
        **_PLOT_BASE,
        margin=dict(l=110, r=25, t=55, b=65),
        height=390,
        xaxis=dict(title="Evaluation Stage", showgrid=False),
        yaxis=dict(title="Baseline Tag", autorange="reversed", showgrid=False),
    )
    return fig


# ---------------------------------------------------------------------------
# HTML card helpers (styling reuses the existing .ms-* classes)
# ---------------------------------------------------------------------------
_PALETTE = {
    "green": ("#2ea043", "rgba(46,160,67,0.15)", "#3fb950"),
    "red": ("#da3633", "rgba(248,81,73,0.15)", "#f85149"),
    "amber": ("#bb8009", "rgba(210,153,34,0.15)", "#d29922"),
    "blue": ("#388bfd", "rgba(88,166,255,0.15)", "#58a6ff"),
    "purple": ("#bc8cff", "rgba(187,128,255,0.15)", "#d2a8ff"),
    "gray": ("#30363d", "#161b22", "#8b949e"),
}


def _thumb(b64: str, size: int = 76, missing: str = "Not recorded") -> str:
    if b64:
        return (f'<img src="{b64}" style="width:{size}px;height:{size}px;border-radius:6px;object-fit:cover;'
                f'display:block;margin:2px auto;border:1px solid #30363d;">')
    return (f'<div style="width:{size}px;height:{size}px;border-radius:6px;border:1px dashed #30363d;margin:2px auto;'
            f'display:flex;align-items:center;justify-content:center;color:#6e7681;font-size:0.62rem;text-align:center;">{escape(missing)}</div>')


def _pil_to_data_uri(img, size: int = 160) -> str:
    """Thumbnail of the uploaded investigation image (real pixels, no processing beyond resize)."""
    try:
        import base64, io
        buf = io.BytesIO()
        img.convert("RGB").resize((size, size)).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        return ""


def _pill(text: str, color: str) -> str:
    border, bg, fg = _PALETTE[color]
    return (f'<div style="display:inline-block;border:1px solid {border};background:{bg};color:{fg};border-radius:4px;'
            f'padding:2px 8px;font-size:0.66rem;font-weight:700;letter-spacing:0.4px;text-transform:uppercase;margin:3px 0;">{escape(text)}</div>')


def _node(title: str, inner: str, cls: str = "", title_color: Optional[str] = None) -> str:
    tc = f' style="color:{title_color};"' if title_color else ""
    return f'<div class="ms-node-card {cls}"><div class="ms-node-title"{tc}>{escape(title)}</div>{inner}</div>'


def _line(text: str, color: str = "#8b949e", size: str = "0.68rem", bold: bool = False) -> str:
    return f'<div style="color:{color};font-size:{size};{"font-weight:700;" if bold else ""}margin-top:2px;">{escape(text)}</div>'


def _compare_card(chip: str, color: str, img_b64: str, headline: str, line1: str, line2: str, line2_color: str = "#8b949e",
                  missing: str = "Not executed") -> str:
    border, bg, fg = _PALETTE[color]
    return (
        '<div style="background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:10px;text-align:center;">'
        f'<div style="display:inline-block;border:1px solid {border};background:{bg};color:{fg};border-radius:4px;padding:2px 8px;'
        f'font-size:0.66rem;font-weight:700;letter-spacing:0.4px;margin-bottom:8px;">{escape(chip)}</div>'
        + _thumb(img_b64, 160, missing)
        + f'<div style="font-weight:700;color:#f0f6fc;font-size:0.92rem;margin-top:8px;">{escape(headline)}</div>'
        + _line(line1, size="0.75rem") + _line(line2, line2_color, "0.75rem")
        + '</div>'
    )


def _kpi_grid(cards: List[Dict[str, str]]) -> str:
    out = ['<div class="ms-kpi-grid">']
    for c in cards:
        delta = f'<div class="ms-kpi-delta">{escape(c["delta"])}</div>' if c.get("delta") else ""
        out.append(
            f'<div class="ms-kpi-card"><div class="ms-kpi-label">{escape(c["label"])}</div>'
            f'<div class="ms-kpi-val" style="color:{c["color"]};">{escape(c["value"])}</div>'
            f'<div style="font-size:0.72rem;color:{c.get("note_color", "#8b949e")};">{escape(c["note"])}</div>{delta}</div>'
        )
    out.append("</div>")
    return "".join(out)


def render_red_team_mission_lab():
    """Main entrypoint for the Adaptive Red-Team Mission Lab workspace."""
    inject_mission_css()

    # Session State Tracking
    if "current_mission_run" not in st.session_state:
        st.session_state["current_mission_run"] = None
    if "active_sample_idx" not in st.session_state:
        st.session_state["active_sample_idx"] = 0
    if "active_path_node" not in st.session_state or not st.session_state.get("active_path_node"):
        st.session_state["active_path_node"] = "DECISION"

    # ------------------ SIDEBAR SCOPE CONTROLS ------------------
    replay_selected = st.session_state.get("ms_mission_mode") == "📜 Replay Audit"
    budget_limit, epsilon, pgd_steps, quant_levels = 5, 8 / 255, 10, 16
    if not replay_selected:
        with st.sidebar:
            st.markdown("### ⚙️ Mission Scope & Budget")
            st.metric("Target Sample Size", "1 Image")
            budget_limit = st.slider("Operations Budget (Probes)", min_value=3, max_value=8, value=5, step=1, help="Max testing actions permitted.")
            epsilon = st.slider("L-infinity Epsilon (Budget)", min_value=0.01, max_value=0.10, value=8 / 255, step=0.005, format="%.4f", help="Maximum allowable pixel perturbation.")
            pgd_steps = st.slider("PGD Ascent Iterations", min_value=5, max_value=25, value=10, step=5, help="Number of projected gradient ascent steps.")
            quant_levels = st.selectbox("Defense Quantization Bins", options=[8, 16, 32], index=1)
            st.caption(f"Budget &epsilon; = {epsilon:.4f} corresponds to ~{int(round(epsilon * 255))}/255 max pixel shift.")

    # ------------------ CONCISE MISSION HEADER ------------------
    mode_col1, mode_col2 = st.columns([3, 1])
    with mode_col1:
        st.markdown(
            f"""
            <div class="ms-header-bar">
                <div>
                    <div class="ms-header-title">⚔️ ADAPTIVE RED-TEAM MISSION LAB</div>
                    <div class="ms-header-meta">
                        Target Asset: <b>RAM++ (Swin-L)</b> &nbsp;|&nbsp;
                        {("Recorded configuration in sidebar" if replay_selected else f"Threat Envelope: L∞ Norm (ε = {epsilon:.4f}) &nbsp;|&nbsp; Budget: {budget_limit} probes")}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with mode_col2:
        mission_mode = st.radio(
            "Mode",
            ["🚀 Active Mission", "📜 Replay Audit"],
            horizontal=True,
            label_visibility="collapsed", key="ms_mission_mode",
        )

    # ------------------ REPLAY MODE SELECTION ------------------
    if mission_mode == "📜 Replay Audit":
        persisted_runs = [r for r in load_persisted_runs() if "RAM++" in str(r.get("model_name", ""))]
        if not persisted_runs:
            st.info("No persisted mission runs found in `results/runs/`. Execute an active mission first.")
            return

        run_options = {
            f"{r.get('run_id')} · Recorded audit": r
            for r in persisted_runs
        }
        selected_run_label = st.selectbox("Select Mission Run to Replay & Audit:", list(run_options.keys()))
        run_data = run_options[selected_run_label]
        is_replay = True
        with st.sidebar:
            st.markdown("### Recorded investigation")
            st.caption(str(run_data.get("run_id", "Unknown run")))
            st.write(str(run_data.get("dataset_name", "Unknown image")))
            sample = next(iter(run_data.get("detailed_samples") or []), {})
            uri = sample.get("clean_img_b64", "")
            if isinstance(uri, str) and uri.startswith("data:image/png;base64,"):
                try:
                    import base64
                    st.image(base64.b64decode(uri.split(",", 1)[1], validate=True), caption="Saved audit preview", use_container_width=True)
                except Exception:
                    st.caption("Saved preview unavailable.")
            st.markdown("**Recorded configuration**")
            for key, label in [("max_budget", "Probe budget"), ("epsilon", "Pixel budget"), ("pgd_steps", "Recorded iterations"), ("quantization_levels", "Quantization levels")]:
                value = (run_data.get("config") or {}).get(key)
                st.write(f"{label}: {value:.4f}" if isinstance(value, float) else f"{label}: {value if value is not None else 'Not recorded'}")
            tags = (run_data.get("sample_inspection") or {}).get("baseline_tags") or []
            if tags:
                st.caption("Recorded baseline tags")
                st.write(", ".join(map(str, tags)))
    else:
        is_replay = False
        # Launch Mission Action
        launch_col, _ = st.columns([2, 5])
        with launch_col:
            run_btn = st.button("🚀 Launch Red-Team Mission", type="primary", use_container_width=True)

        if run_btn:
            inv = st.session_state.get("investigation", {})
            if inv.get("pixel_tensor") is None or inv.get("ram_baseline", {}).get("scores_tensor") is None:
                st.warning("Upload an investigation image and obtain a RAM++ baseline first.")
                return
            with st.spinner("Acquiring target model and verifying data..."):
                try:
                    model, device = get_cached_model()
                except Exception as err:
                    st.error(f"❌ **Target Asset Configuration Notice**\n\n```\n{str(err)}\n```")
                    st.info(
                        "**To resolve:** Ensure the RAM++ checkpoint is at `ram-plus/pretrained/ram_plus_swin_large_14m.pth` "
                        "and review the model-loading error above."
                    )
                    return

                # Reuse the shared investigation; never fall back to a dataset.
                ram_mode = True
                images = inv["pixel_tensor"]
                ram_baseline = inv["ram_baseline"]
                baseline_tags = ram_baseline.get("tags", [])
                baseline_scores = ram_baseline["scores_tensor"]
                tag_list = ram_baseline.get("tag_list", [])
                labels = baseline_scores
                dataset_name_ui = f"Investigation: {inv.get('filename', 'image')}"
                model_name_ui = "RAM++ Swin-L"

                config = {
                    "attacks": {
                        "fgsm": {"epsilon": epsilon},
                        "pgd": {"epsilon": epsilon, "steps": pgd_steps, "step_size": 2 / 255},
                    },
                    "quantization_levels": quant_levels,
                }

                orchestrator = RedTeamOrchestrator(
                    model=model,
                    device=device,
                    images=images,
                    labels=labels,
                    config=config,
                    max_budget=budget_limit,
                    model_name=model_name_ui,
                    dataset_name=dataset_name_ui,
                    ram_mode=ram_mode,
                    tag_list=tag_list,
                    baseline_tags=baseline_tags,
                    baseline_scores=baseline_scores,
                )

                prog_bar = st.progress(0.0)
                status_msg = st.empty()

                def on_progress(desc, completed, total):
                    status_msg.markdown(f"**Executing Operation:** `{desc}` (`{completed}/{total}` completed)")
                    prog_bar.progress(min(1.0, max(0.0, completed / max(1, total))))

                if not hasattr(orchestrator, "_scoring_fn"):
                    st.warning("Stale `orchestrator/engine.py` detected (older than `ui.py`). Replace it with the "
                               "version from the ZIP and fully restart Streamlit (Ctrl+C, then run again).")
                assessment = orchestrator.run(progress_callback=on_progress)
                prog_bar.progress(1.0)
                status_msg.markdown(
                    f"**Mission Status:** `{'EXECUTION ABORTED' if assessment.overall_status == 'INTEGRITY_FAILED' else 'EXECUTION COMPLETE'}`"
                )

                st.session_state["current_mission_run"] = {
                    "run_id": orchestrator.run_id,
                    "model_name": orchestrator.model_name,
                    "dataset_name": orchestrator.dataset_name,
                    "ram_mode": orchestrator.ram_mode,
                    "config": {
                        "epsilon": epsilon,
                        "pgd_steps": pgd_steps,
                        "quantization_levels": quant_levels,
                        "max_budget": budget_limit,
                    },
                    "completed_actions": orchestrator.completed_actions,
                    "decisions": [d.to_dict() for d in orchestrator.decision_history],
                    "step_history": [s.to_dict() for s in orchestrator.step_history],
                    "final_assessment": assessment.to_dict(),
                    "anomalies_detected": orchestrator.anomalies_detected,
                    "preflight_result": orchestrator.preflight_result.to_dict() if orchestrator.preflight_result else None,
                    "sample_inspection": orchestrator.sample_inspection,
                    "detailed_samples": orchestrator.detailed_samples,
                    "class_metrics": orchestrator.class_metrics,
                    "flip_matrix": orchestrator.flip_matrix,
                    "scoring_report": getattr(orchestrator, "scoring_report", None),
                    "discoveries": orchestrator.discoveries,
                }
                if assessment.overall_status == "INTEGRITY_FAILED":
                    st.error(f"Mission `{orchestrator.run_id}` aborted; see audit report.")
                else:
                    st.success(f"Mission `{orchestrator.run_id}` completed and sealed into audit history.")

        run_data = st.session_state.get("current_mission_run")

    inv = st.session_state.get("investigation", {}) if not is_replay else {}
    baseline = inv.get("ram_baseline", {})
    scores = baseline.get("scores_dict", {})
    baseline_count = len(baseline.get("tags", [])) if baseline else None

    # ------------------ EMPTY STATE (IDLE) ------------------
    if not run_data:
        st.markdown(
            """
            <div style="background-color: #0d1117; border: 1px dashed #30363d; border-radius: 6px; padding: 28px 18px; text-align: center; margin-top: 10px;">
                <span style="font-size: 2rem;">🎯</span>
                <h4 style="color: #f0f6fc; margin-top: 6px; margin-bottom: 4px;">Awaiting Red-Team Mission Deployment</h4>
                <p style="color: #8b949e; max-width: 520px; margin: 0 auto; font-size: 0.88rem;">
                    Configure parameters in the left sidebar and click <b>Launch Red-Team Mission</b>.
                    The autonomous orchestrator will execute preflight checks, measure baseline tag predictions,
                    probe first-order gradients, and adaptively escalate based on empirical evidence.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Extract Run Data
    assessment = run_data.get("final_assessment", {})
    decisions = run_data.get("decisions", [])
    steps = run_data.get("step_history", [])
    discoveries = run_data.get("discoveries", [])
    detailed_samples = run_data.get("detailed_samples", [])
    class_metrics = run_data.get("class_metrics", {})
    config = run_data.get("config", {})
    eps = config.get("epsilon", 8 / 255)
    budget_used = assessment.get("budget_consumed", 0)
    budget_total = assessment.get("budget_allocated", 0)
    overall_status = str(assessment.get("overall_status") or "UNKNOWN")
    worst_pres = assessment.get("worst_adversarial_accuracy")      # % of baseline tags preserved (worst case)
    peak_disruption = assessment.get("max_attack_success_rate")    # % of baseline tags lost (peak)
    cur_sample = detailed_samples[0] if detailed_samples else {}
    aborted = overall_status == "INTEGRITY_FAILED"

    def _step_ok(action: str) -> bool:
        return any(x.get("action_type") == action and x.get("status") == "SUCCESS" for x in steps)

    has_fgsm, has_pgd = _step_ok("RUN_FGSM"), _step_ok("ESCALATE_PGD")
    atk_kind = "pgd" if has_pgd else ("fgsm" if has_fgsm else None)
    atk_label = {"pgd": f"PGD-{config.get('pgd_steps', '')}".rstrip("-"), "fgsm": "FGSM", None: ""}[atk_kind]

    baseline_count = cur_sample.get("clean_tag_count")
    if baseline_count is None:
        baseline_step = next((x for x in steps if x.get("action_type") == "RUN_BASELINE" and x.get("status") == "SUCCESS"), {})
        baseline_count = baseline_step.get("metrics", {}).get("tag_count")
    fname = str(run_data.get("dataset_name", "image")).replace("Investigation: ", "")
    fname_short = fname if len(fname) <= 22 else fname[:19] + "…"
    top5 = cur_sample.get("clean_top5") or []
    top_tag = top5[0] if top5 else None

    # Clean-side facts that exist independently of the mission (shared baseline); never attack results.
    cur_sample = dict(cur_sample)
    if not cur_sample.get("clean_img_b64") and inv.get("image_pil") is not None:
        cur_sample["clean_img_b64"] = _pil_to_data_uri(inv["image_pil"])
    if baseline_count is None and baseline:
        baseline_count = len(baseline.get("tags", []))
    if not top5 and scores:
        top5 = [{"class": t, "prob": round(v * 100.0, 2)} for t, v in sorted(scores.items(), key=lambda kv: -kv[1])[:5]]
        top_tag = top5[0]

    if aborted:
        _pf = run_data.get("preflight_result") or {}
        _failed = [(n, c) for n, c in (_pf.get("checks") or {}).items() if not c.get("passed", True)]
        with st.expander("⛔ Why this mission was aborted (failed integrity checks)", expanded=True):
            if _failed:
                for _n, _c in _failed:
                    st.markdown(f"- **{_n}** — {_c.get('message', '')}")
            for _e in _pf.get("errors", []):
                st.code(str(_e), language="text")
            if not _failed and not _pf.get("errors"):
                st.write("No preflight record was stored; see the Debrief & Audit tab.")
            st.caption("No attack was run and no robustness result is claimed. The clean image and tags shown below are the shared baseline.")

    # =========================================================================
    # 1. KPI CARDS (single-image RAM++ tag metrics)
    # =========================================================================
    st.markdown("##### 📊 SINGLE-IMAGE TAG-ROBUSTNESS METRICS (RAM++ MULTI-LABEL)")
    verified_steps = [x for x in steps if x.get("status") == "SUCCESS" and x.get("trust_flag", {}).get("status") == "PASS"]
    all_verified = bool(steps) and len(verified_steps) == len(steps) and not aborted and not assessment.get("trust_warnings") and not run_data.get("anomalies_detected")
    budget_note = "Budget limit reached" if overall_status == "INCONCLUSIVE" and budget_total and budget_used >= budget_total and all_verified else overall_status
    status_color = "#f85149" if ("FAILED" in overall_status or "CRITICAL" in overall_status) else ("#d29922" if overall_status in ("MODERATE_VULNERABILITY", "INCONCLUSIVE") else "#3fb950")
    st.markdown(_kpi_grid([
        {"label": "Baseline Tags (Clean)", "value": str(baseline_count) if baseline_count is not None else "—",
         "color": "#3fb950", "note": "Active RAM++ tags on the image" if baseline_count is not None else "Not recorded"},
        {"label": "Min Tag Retention", "value": f"{worst_pres:.1f}%" if worst_pres is not None else "—",
         "color": "#f85149" if (worst_pres is not None and worst_pres < 100) else "#3fb950",
         "note": "Worst of FGSM / PGD" if worst_pres is not None else "No verified attack result",
         "delta": f"Retention loss: {100.0 - worst_pres:.1f} pp" if worst_pres is not None else ""},
        {"label": "Peak Tag Disruption", "value": f"{peak_disruption:.1f}%" if peak_disruption is not None else "—",
         "color": "#d29922", "note": "Baseline tags lost (peak)" if peak_disruption is not None else "No verified attack result"},
        {"label": "Evaluation Probes Spent", "value": f"{budget_used} / {budget_total}", "color": "#f0f6fc",
         "note": budget_note, "note_color": status_color},
    ]), unsafe_allow_html=True)

    st.caption(
        "ℹ️ **Note on Metrics:** KPI cards summarise tag-set retention for this single investigation image "
        "(clean baseline = 100% by definition; every other value is measured). The pipeline below traces the selected "
        "image through RAM++: measured patch-embedding activations, tag scores and the attack outcome."
    )

    # =========================================================================
    # 2. NEURAL PIPELINE FORENSICS
    # =========================================================================
    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;margin:10px 0 6px 0;">'
        '<h3 style="margin:0;">🧬 Neural Network Pipeline — Verified Execution Forensics</h3>'
        '<span class="ms-badge ms-badge-cyan">INTERACTIVE FORENSIC TRACE</span></div>',
        unsafe_allow_html=True,
    )
    if atk_kind:
        kept = cur_sample.get(f"{atk_kind}_tags_kept")
        lost = cur_sample.get(f"{atk_kind}_tags_lost")
        pres_pct = cur_sample.get(f"{atk_kind}_confidence")
        disrupted = bool(lost)
        sel_text = (f"Image 1: {fname_short} · Clean: {baseline_count} tags · {atk_label}: {kept}/{baseline_count} kept "
                    f"({'💥 TAGS DISRUPTED' if disrupted else 'TAGS PRESERVED'})")
    else:
        kept = lost = pres_pct = None
        disrupted = False
        sel_text = f"Image 1: {fname_short} · Clean: {baseline_count if baseline_count is not None else '—'} tags · attack not evaluated"
    sel_col, info_col = st.columns([4, 1.4])
    with sel_col:
        st.selectbox("Inspected image", [sel_text], label_visibility="collapsed", key="pipeline_image_select")
    with info_col:
        st.markdown(
            f"<div style='color:#8b949e;font-size:0.8rem;padding-top:8px;'>Inspecting Image #1 · Top tag: "
            f"<code>{escape(top_tag['class']) if top_tag else 'n/a'}</code></div>",
            unsafe_allow_html=True,
        )

    pm = cur_sample.get("feature_meta", {}) or {}
    ch = (pm.get("patch_embed_shape") or [None])[0]
    n1 = _node("01 INPUT TENSOR", _thumb(cur_sample.get("clean_img_b64", "")) +
               _line(fname_short, "#f0f6fc", "0.78rem", True) + _line("(3, 384, 384) · inverse-norm ∈ [0, 1]"))
    n2 = _node("02 FEATURE EXTRACTOR", _thumb(cur_sample.get("clean_patch_feature_b64", cur_sample.get("clean_feature_b64", ""))) +
               _line("Patch-Embed Conv", "#58a6ff", "0.78rem", True) + _line(f"{ch}ch Spatial Map" if ch else "Activation not recorded"), cls="active")
    n3 = _node("03 RAM++ SWIN-L", '<div style="font-size:1.7rem;margin:6px 0;">🧠</div>' +
               _line("Swin-L Encoder", "#3fb950", "0.7rem", True) + _line("RAM++", "#f0f6fc", "0.82rem", True) + _line("Multi-label Tag Head"))
    if top_tag and baseline_count is not None:
        n4 = _node("04 CLEAN OUTPUT",
                   _line(f"{baseline_count} TAGS", "#f0f6fc", "1.05rem", True) + _pill(f"TOP: {top_tag['class']} · {top_tag['prob']:.2f}%", "green") +
                   _line("BASELINE LOCKED", "#3fb950", "0.7rem", True) + _line("Instance Tag Score"))
    else:
        n4 = _node("04 CLEAN OUTPUT", _line("Not recorded", "#8b949e", "0.85rem", True))
    if atk_kind:
        n5 = _node("05 ∇ₓL ATTACK", _thumb(cur_sample.get(f"{atk_kind}_heatmap_b64", "")) +
                   _line(atk_label, "#f85149", "0.8rem", True) + _line(f"Δ ≤ {eps:.4f}"), cls="alert", title_color="#f85149")
        n6 = _node("06 ADV OUTPUT",
                   _line(f"{kept}/{baseline_count} KEPT", "#f85149" if disrupted else "#3fb950", "1.05rem", True) +
                   _pill(f"{pres_pct:.2f}% RETAINED", "red" if disrupted else "green") +
                   _line("💥 TAGS DISRUPTED" if disrupted else "TAGS PRESERVED", "#f85149" if disrupted else "#3fb950", "0.7rem", True) +
                   _line("Tag Preservation"), cls="alert" if disrupted else "")
    else:
        why = "Mission aborted" if aborted else "No attack evaluated"
        n5 = _node("05 ∇ₓL ATTACK", _thumb("", 76, "Not executed") + _line(why))
        n6 = _node("06 ADV OUTPUT", _line("Not evaluated", "#8b949e", "0.85rem", True) + _line("No verified result"))
    arrow = '<div class="ms-node-arrow">→</div>'
    bolt = '<div class="ms-node-arrow" style="color:#f0883e;">⚡</div>'
    st.markdown('<div class="ms-pipeline-nodes">' + n1 + arrow + n2 + arrow + n3 + arrow + n4 + bolt + n5 + arrow + n6 + '</div>',
                unsafe_allow_html=True)

    # =========================================================================
    # 3. COMPACT INTERACTIVE INVESTIGATION TABS
    # =========================================================================
    tab_inspect, tab_policy, tab_weakness, tab_debrief, tab_copilot = st.tabs([
        "🔬 Visual Attack Explorer",
        "🧠 Adaptive Policy Path",
        "📊 Model Weakness Explorer",
        "📑 Debrief & Audit Report",
        "🤖 Investigation Copilot",
    ])

    # ------------------ TAB 1: VISUAL ATTACK EXPLORER ------------------
    with tab_inspect:
        image_tab, activation_tab, score_tab = st.tabs([
            "🖼️ Side-by-Side Images & Perturbations",
            "🧬 Internal Neural Activations (Swin)",
            "📈 Tag Score Shift & Trajectory",
        ])

        with image_tab:
            def _attack_card(kind: str, chip: str, color: str):
                if not _step_ok("RUN_FGSM" if kind == "fgsm" else "ESCALATE_PGD"):
                    return _compare_card(chip, color, "", "Not executed", "No verified result", "")
                lost_n = cur_sample.get(f"{kind}_tags_lost", 0)
                return _compare_card(
                    chip, color, cur_sample.get(f"{kind}_img_b64", ""),
                    f"{cur_sample.get(f'{kind}_tags_kept')}/{baseline_count} tags kept",
                    f"Measured L∞: {cur_sample.get(f'{kind}_l_inf', 0.0):.4f}",
                    f"Tag preservation: {cur_sample.get(f'{kind}_confidence', 0.0):.2f}%",
                    "#f85149" if lost_n else "#3fb950",
                )

            cards = [
                _compare_card("1. ORIGINAL CLEAN", "blue", cur_sample.get("clean_img_b64", ""),
                              f"{baseline_count if baseline_count is not None else '—'} baseline tags",
                              (f"Top: {top_tag['class']}" if top_tag else "Top tag: n/a"),
                              (f"Score: {top_tag['prob']:.2f}%" if top_tag else ""), "#3fb950"),
                _attack_card("fgsm", "2. FGSM 1-STEP", "amber"),
                _attack_card("pgd", f"3. {atk_label or 'PGD'} ITERATIVE".upper() if has_pgd else "3. PGD ITERATIVE", "red"),
            ]
            heat_kind = atk_kind
            if heat_kind:
                cards.append(_compare_card("4. PERTURBATION HEATMAP", "purple", cur_sample.get(f"{heat_kind}_heatmap_b64", ""),
                                           "Perturbation Magnitude", "Colormap: Inferno",
                                           f"Budget: ≤ {eps:.4f} (raw pixels)", "#58a6ff"))
            else:
                cards.append(_compare_card("4. PERTURBATION HEATMAP", "purple", "", "Not executed", "No verified result", ""))
            for col, html_card in zip(st.columns(4), cards):
                with col:
                    st.markdown(html_card, unsafe_allow_html=True)

            with st.expander("🔍 Interactive Perturbation Inspector (Hover over grid)"):
                avail = [k for k, ok in (("fgsm", has_fgsm), ("pgd", has_pgd)) if ok and cur_sample.get(f"{k}_diff_matrix")]
                if avail:
                    pick = st.radio("Attack", [k.upper() for k in avail], index=len(avail) - 1, horizontal=True, key="inspector_attack")
                    st.plotly_chart(plot_interactive_heatmap(cur_sample[f"{pick.lower()}_diff_matrix"], eps), use_container_width=False)
                else:
                    st.info("No verified perturbation was recorded for this run.")

        with activation_tab:
            meta = cur_sample.get("feature_meta", {}) or {}

            def _map_row(title: str, kind: str, note: str):
                st.markdown(f"**{title}**")
                dist_key = next((f"{k}_{kind}_distortion_b64" for k in ("pgd", "fgsm") if cur_sample.get(f"{k}_{kind}_distortion_b64")), None)
                dist_from = "PGD" if (dist_key or "").startswith("pgd") else "FGSM"
                clean_b64 = cur_sample.get(f"clean_{kind}_feature_b64", "")
                row = [
                    _compare_card("CLEAN", "blue", clean_b64, "Clean activations", note, "", missing="Not recorded"),
                    _compare_card("FGSM", "amber", cur_sample.get(f"fgsm_{kind}_feature_b64", ""), "FGSM activations",
                                  f"Mean rel. shift: {meta.get(f'fgsm_{kind}_mean_rel_shift')}" if meta.get(f"fgsm_{kind}_mean_rel_shift") is not None else "No verified result", "",
                                  missing="Not executed"),
                    _compare_card("PGD", "red", cur_sample.get(f"pgd_{kind}_feature_b64", ""), "PGD activations",
                                  f"Mean rel. shift: {meta.get(f'pgd_{kind}_mean_rel_shift')}" if meta.get(f"pgd_{kind}_mean_rel_shift") is not None else "No verified result", "",
                                  missing="Not executed"),
                    _compare_card("DISTORTION", "purple", cur_sample.get(dist_key, "") if dist_key else "", "Feature distortion",
                                  f"|{dist_from} − clean| (magma)" if dist_key else "No measurable change / not run", "", missing="Not available"),
                ]
                for col, html_card in zip(st.columns(4), row):
                    with col:
                        st.markdown(html_card, unsafe_allow_html=True)

            if not cur_sample.get("clean_patch_feature_b64") and not cur_sample.get("clean_stage4_feature_b64"):
                st.info("No measured Swin feature activations were recorded for this run.")
            else:
                ps = meta.get("patch_embed_shape")
                gs = meta.get("stage4_grid")
                _map_row("Patch-embedding convolution — mean |activation| over channels",
                         "patch", f"{ps[0]}ch · {ps[1]}×{ps[2]} map" if ps else "Spatial map")
                _map_row("Final encoder tokens — L2 norm per spatial token",
                         "stage4", f"{gs[0]}×{gs[1]} token grid" if gs else "Token map")
                st.caption("Maps are captured by forward hooks during the same RAM++ pass that produced the tags. "
                           "Attacked maps share the clean map's colour scale; distortion = |attacked − clean|.")

        with score_tab:
            rows = run_data.get("sample_inspection", {}).get("tag_comparison", [])
            show_mode = "PGD" if has_pgd else ("FGSM" if has_fgsm else None)
            sc1, sc2 = st.columns(2)
            with sc1:
                if show_mode and rows:
                    st.plotly_chart(plot_confidence_distribution(rows, show_mode), use_container_width=True)
                else:
                    st.info("No verified attack scores: run an attack step to compare tag scores.")
            with sc2:
                traj = cur_sample.get("trajectory", [])
                if len(traj) > 1:
                    st.plotly_chart(plot_trajectory_chart(traj), use_container_width=True)
                else:
                    st.info("Trajectory unavailable until verified comparison results exist.")
            st.caption("Scores are RAM++ sigmoid outputs measured on each evaluated image. Clean preservation is 100% by definition; "
                       "tags marked “(new)” appeared only after the attack.")

    # ------------------ TAB 2: ADAPTIVE POLICY PATH ------------------
    with tab_policy:
        st.markdown("#### 🗺️ Visible Attack-Path & Decision Engine")
        st.caption("Click any protocol checkpoint to inspect its triggering threshold, budget accounting, and policy rationale:")

        # Budget Accounting Ledger
        st.markdown(
            f"""
            <div style="background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:10px 14px; margin-bottom:14px; font-size:0.82rem; color:#8b949e; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                <div>
                    <span style="font-weight:700; color:#f0f6fc;">💡 Budget Accounting Ledger:</span>
                    <span style="color:#c9d1d9;">5 Model Evaluation Probes</span> (Cost: 1 probe each: Preflight, Baseline, FGSM, PGD, Defense) +
                    <span style="color:#58a6ff;">2 Zero-Cost Control Stages</span> (04 Policy Decision Engine, 07 Debrief & Audit Synthesis).
                </div>
                <div>
                    <span class="ms-badge ms-badge-cyan">EVALUATION PROBES: {budget_used}/{budget_total}</span>
                    <span class="ms-badge ms-badge-green">BUDGET DISCIPLINE: 100% BOUNDED</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Horizontal Attack Path Map (labels reflect what actually executed in this run)
        probe_nodes = {"PREFLIGHT": ("RUN_PREFLIGHT", 1), "BASELINE": ("RUN_BASELINE", 2), "FGSM": ("RUN_FGSM", 3),
                       "PGD": ("ESCALATE_PGD", 4), "DEFENSE": ("TEST_DEFENSE", 5)}

        def _probe_line(node_id: str) -> str:
            act, n = probe_nodes[node_id]
            rec = next((x for x in steps if x.get("action_type") == act), None)
            if rec:
                return f"[PROBE {n} · COST: 1]" + ("" if rec.get("status") == "SUCCESS" else " · FAILED")
            return "[ABORTED · COST: 0]" if aborted else "[BYPASSED · COST: 0]"

        path_nodes = [
            ("PREFLIGHT", f"01 Preflight\n{_probe_line('PREFLIGHT')}"),
            ("BASELINE", f"02 Baseline\n{_probe_line('BASELINE')}"),
            ("FGSM", f"03 FGSM Probe\n{_probe_line('FGSM')}"),
            ("DECISION", "04 Policy Engine\n[LOGIC · COST: 0]"),
            ("PGD", f"05 PGD Escalation\n{_probe_line('PGD')}"),
            ("DEFENSE", f"06 Defense Verify\n{_probe_line('DEFENSE')}"),
            ("DEBRIEF", "07 Tech Debrief\n[SYNTHESIS · COST: 0]"),
        ]

        node_cols = st.columns(len(path_nodes))
        for idx, (node_id, label) in enumerate(path_nodes):
            with node_cols[idx]:
                if st.button(label, key=f"path_btn_{node_id}", use_container_width=True):
                    st.session_state["active_path_node"] = node_id

        # Detail Panel for Selected Path Node
        sel_node = str(st.session_state.get("active_path_node") or "DECISION")

        target_action_map = {
            "PREFLIGHT": ["RUN_PREFLIGHT"],
            "BASELINE": ["RUN_BASELINE"],
            "FGSM": ["RUN_FGSM"],
            "DECISION": ["ESCALATE_PGD", "TEST_DEFENSE", "STOP_SUFFICIENT_EVIDENCE"],
            "PGD": ["ESCALATE_PGD"],
            "DEFENSE": ["TEST_DEFENSE"],
            "DEBRIEF": ["STOP_SUFFICIENT_EVIDENCE", "STOP_BUDGET_EXHAUSTED", "HALT_INTEGRITY_FAILURE"],
        }
        target_actions = target_action_map.get(sel_node, [sel_node])
        matching_dec = next((d for d in decisions if d.get("action") in target_actions), None)
        if not matching_dec and decisions:
            matching_dec = next((d for d in decisions if sel_node in str(d.get("action") or "") or sel_node in str(d.get("reason") or "")), None)

        def _display_policy_text(value):
            text = str(value)
            for old, new in [("Clean model classification accuracy", "Clean reference tag scores"), ("Clean baseline accuracy", "Clean reference tag retention"), ("ground-truth prediction alignment", "baseline tag-set alignment"), ("clean accuracy", "clean reference retention"), ("FGSM accuracy", "FGSM tag retention"), ("adversarial accuracy", "baseline-tag retention"), ("accuracy drop", "retention loss"), ("Acc drop", "Retention loss"), ("clean_acc", "clean reference retention"), ("fgsm_asr", "FGSM tag disruption"), ("peak_asr", "peak tag disruption"), ("ASR", "tag disruption"), ("attack success rate", "tag disruption rate")]:
                text = text.replace(old, new)
            return escape(text)

        if matching_dec:
            cost_str = f"{matching_dec.get('budget_cost', 0)} Probe(s)" if sel_node != "DECISION" else "0 (Decision Logic Check)"
            st.markdown(
                f"""
                <div style="background:#12161f; border:1px solid #30363d; border-left:4px solid #58a6ff; border-radius:6px; padding:12px 16px; margin:12px 0;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-weight:700; color:#f0f6fc; font-size:0.95rem;">Checkpoint: {sel_node} &nbsp;&rarr;&nbsp; Action: {matching_dec.get('action')}</span>
                        <div>
                            <span class="ms-badge ms-badge-cyan">BUDGET COST: {cost_str}</span>
                            <span class="ms-badge ms-badge-amber">REMAINING BUDGET: {matching_dec.get('remaining_budget')}</span>
                        </div>
                    </div>
                    <div style="margin-top:8px; font-size:0.86rem; color:#c9d1d9;"><b>Triggering Rule:</b> {_display_policy_text(matching_dec.get('reason'))}</div>
                    <div style="margin-top:6px; font-size:0.8rem; color:#8b949e;">
                        <code>Condition: {_display_policy_text(matching_dec.get('condition_evaluated', 'Evaluated'))}</code> &nbsp;|&nbsp;
                        <code>Threshold: {_display_policy_text(matching_dec.get('threshold_used', 'Policy Rule'))}</code> &nbsp;|&nbsp;
                        <code>Measured: {_display_policy_text(matching_dec.get('measured_value', 'Recorded'))}</code>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info(f"Checkpoint `{sel_node}`: No recorded decision for this checkpoint.")

        # Compact Mission Milestones & Discoveries
        if discoveries:
            st.markdown("##### 🏆 Adversarial Discoveries Unlocked")
            st.markdown("<div class='ms-disc-grid'>", unsafe_allow_html=True)
            for disc in discoveries:
                unlocked = disc.get("unlocked", False)
                card_cls = "ms-disc-card" if unlocked else "ms-disc-card locked"
                badge_cls = "ms-badge-green" if unlocked else "ms-badge-amber"
                st.markdown(
                    f"""
                    <div class="{card_cls}">
                        <span class="ms-badge {badge_cls}">{disc.get('badge')}</span>
                        <div style="font-weight:700; color:#f0f6fc; margin:4px 0; font-size:0.82rem;">{disc.get('title')}</div>
                        <div style="font-size:0.75rem; color:#8b949e;">{disc.get('summary')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)

    # ------------------ TAB 3: MODEL WEAKNESS EXPLORER ------------------
    with tab_weakness:
        w1, w2 = st.columns(2)
        measured = bool(class_metrics) and (has_fgsm or has_pgd)
        with w1:
            if measured:
                st.plotly_chart(plot_class_weakness_chart(class_metrics), use_container_width=True)
            else:
                st.info("No verified tag comparison: no attack step completed successfully.")
        with w2:
            if measured:
                st.plotly_chart(plot_tag_survival_matrix(class_metrics), use_container_width=True)
            else:
                st.info("No verified tag-survival data for this run.")
        if measured:
            st.caption("Top baseline tags by clean score (max 20). Scores are RAM++ sigmoid outputs measured on each evaluated image; "
                       "a tag is 'lost' when it falls below RAM++'s own decision threshold.")

    # ------------------ TAB 4: DEBRIEF & AUDIT REPORT ------------------
    with tab_debrief:
        if not is_replay and baseline.get("scores_tensor") is not None:
            from orchestrator.preflight import validate_ram_baseline
            with st.expander("Clean recognition integrity checks", expanded=False):
                try:
                    model, device = get_cached_model()
                    result = validate_ram_baseline(model, inv.get("pixel_tensor"), baseline, device)
                    for name, check in result.checks.items():
                        st.write(f"{'PASS' if check['passed'] else 'FAIL'} — {name}: {check['message']}")
                except Exception as exc:
                    st.error(str(exc))
        with st.expander("Recorded mission preflight checks", expanded=False):
            preflight = run_data.get("preflight_result") or {}
            for name, check in preflight.get("checks", {}).items():
                st.write(f"{'PASS' if check.get('passed') else 'FAIL'} — {name}: {check.get('message')}")
            for error in preflight.get("errors", []):
                st.error(error)
        # Separate execution integrity from evidence coverage; preserve raw audit verdict.
        verified_steps = [x for x in steps if x.get("status") == "SUCCESS" and x.get("trust_flag", {}).get("status") == "PASS"]
        all_verified = bool(steps) and len(verified_steps) == len(steps) and not aborted and not assessment.get("trust_warnings") and not run_data.get("anomalies_detected")
        st.markdown(_kpi_grid([
            {"label": "Execution", "value": "COMPLETED" if all_verified else "REVIEW", "color": "#3fb950" if all_verified else "#d29922", "note": f"{len(verified_steps)}/{len(steps)} recorded stages passed"},
            {"label": "Recorded Integrity", "value": "PASS" if all_verified else "REVIEW", "color": "#3fb950" if all_verified else "#d29922", "note": "Configured checks only"},
            {"label": "Observed Stability", "value": ("TAG LOSS" if worst_pres < 100 else "RETAINED") if all_verified and worst_pres is not None else "UNKNOWN", "color": "#f85149" if all_verified and worst_pres is not None and worst_pres < 100 else "#8b949e", "note": "Under recorded test conditions"},
            {"label": "Evidence Coverage", "value": "LIMITED", "color": "#d29922", "note": "Single-image evaluation"},
        ]), unsafe_allow_html=True)
        if all_verified and worst_pres is not None:
            st.write(f"Observed minimum baseline-tag retention: **{worst_pres:.1f}%**. This is a measured comparison, not dataset-wide accuracy or a robustness certificate.")
        if overall_status == "INCONCLUSIVE" and budget_total and budget_used >= budget_total and all_verified:
            st.caption("Budget limit reached. The recorded stages passed their checks; broader robustness remains unassessed.")
        with st.expander("Original audit verdict", expanded=not all_verified):
            st.write(f"**{overall_status}:** {assessment.get('overall_summary')}")
        defense = next((x.get("metrics", {}) for x in verified_steps if x.get("action_type") == "TEST_DEFENSE"), {})
        if defense:
            before, after, stages = [], [], []
            for action, name, key in [("RUN_FGSM", "FGSM", "defended_fgsm_tag_pres"), ("ESCALATE_PGD", "PGD", "defended_pgd_tag_pres")]:
                source = next((x.get("metrics", {}) for x in verified_steps if x.get("action_type") == action), {})
                if source.get("tag_preservation_rate") is not None and defense.get(key) is not None:
                    stages.append(name)
                    before.append(100 * source["tag_preservation_rate"])
                    after.append(100 * defense[key])
            if stages:
                with st.expander("Recorded defense comparison", expanded=False):
                    fig = go.Figure([go.Bar(name="Before", x=stages, y=before, text=[f"{v:.1f}%" for v in before], textposition="outside", marker_color="#d29922", cliponaxis=False), go.Bar(name="After", x=stages, y=after, text=[f"{v:.1f}%" for v in after], textposition="outside", marker_color="#58a6ff", cliponaxis=False)])
                    fig.update_layout(barmode="group", yaxis=dict(title="Baseline tags retained (%)", range=[0, 115]), height=320, margin=dict(l=55,r=25,t=45,b=45), legend=dict(orientation="h", y=1.12), **_PLOT_BASE)
                    st.plotly_chart(fig, use_container_width=True)

        deb_col1, deb_col2 = st.columns(2)
        with deb_col1:
            st.markdown("**Key Verified Findings:**")
            for f in assessment.get("key_findings", []):
                st.markdown(f"- {f}")
            st.markdown("**Untested Attack Surfaces:**")
            for u in assessment.get("untested_vectors", []):
                st.markdown(f"- `{u}`")

        with deb_col2:
            st.markdown("**Scientific Caveats:**")
            for l in assessment.get("scientific_limitations", []):
                st.markdown(f"- ⚠️ {l}")
            st.markdown("**Recommendations:**")
            for r in ([] if overall_status == "INTEGRITY_FAILED" else assessment.get("recommendations", [])):
                st.markdown(f"- 💡 {r}")

        # Defense Verification Lab Handoff Card
        st.markdown(
            """
            <div style="background-color: #161b22; border: 1px solid #388bfd; border-radius: 6px; padding: 12px 16px; margin: 12px 0;">
                <span class="ms-badge ms-badge-cyan">DEFENSE HANDOFF PROTOCOL</span>
                <h5 style="color: #f0f6fc; margin: 4px 0;">Investigate Mitigations in Defense Verification Lab</h5>
                <p style="color: #8b949e; margin: 0; font-size: 0.82rem;">
                    Experimental defenses may be explored separately. An aborted mission establishes no model vulnerability.
                    Switch to the <b>🛡️ Defense Verification Lab</b> workspace in the left sidebar to experiment with quantization levels,
                    inspect clean-accuracy trade-offs, and verify empirical mitigation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # JSON Report Download
        rep_json = json.dumps(run_data, indent=2)
        st.download_button(
            label="📥 Export Mission Audit Report (JSON)",
            data=rep_json,
            file_name=f"modelshield_mission_{run_data.get('run_id')}.json",
            mime="application/json",
        )

        with st.expander("Inspect Raw Terminal Audit Log"):
            terminal_lines = []
            for d in decisions:
                act = d.get("action")
                rsn = d.get("reason")
                ts = d.get("timestamp", "")
                terminal_lines.append(f"<span style='color:#58a6ff;'>[{ts[11:19]}]</span> <span style='color:#39d353; font-weight:bold;'>DECISION: {act}</span>")
                terminal_lines.append(f"<span style='color:#8b949e;'>Policy Rule:</span> {rsn}")
                terminal_lines.append("<span style='color:#21262d;'>--------------------------------------------------</span>")
            terminal_html = "<div class='ms-terminal-box'>" + "<br>".join(terminal_lines) + "</div>"
            st.markdown(terminal_html, unsafe_allow_html=True)

    # ------------------ TAB 5: INVESTIGATION COPILOT ------------------
    with tab_copilot:
        copilot = InvestigationCopilot(run_data)

        c_q1, c_q2, c_q3, c_q4 = st.columns(4)
        quick_q = None
        with c_q1:
            if st.button("Why choose PGD?", key="cop_q1", use_container_width=True):
                quick_q = "Why did the orchestrator choose PGD?"
        with c_q2:
            if st.button("Explain Active Sample", key="cop_q2", use_container_width=True):
                quick_q = f"Explain Sample {st.session_state.get('active_sample_idx', 0) + 1}"
        with c_q3:
            if st.button("Most Vulnerable Tag?", key="cop_q3", use_container_width=True):
                quick_q = "Which tag was most vulnerable to adversarial attacks?"
        with c_q4:
            if st.button("Presentation Limits", key="cop_q4", use_container_width=True):
                quick_q = "What limitations should I mention when presenting this result?"

        cop_query = st.text_input("Ask Copilot a question about this mission:", value=quick_q or "")
        if cop_query:
            answer = copilot.answer_inquiry(cop_query)
            answer = str(answer).replace(
                "ModelShield follows rigorous AI security standards by",
                "ModelShield documents its evaluation methodology by"
            )
            with st.container(border=True):
                st.markdown("**🤖 Copilot Technical Analysis**")
                st.markdown(answer, unsafe_allow_html=False)
