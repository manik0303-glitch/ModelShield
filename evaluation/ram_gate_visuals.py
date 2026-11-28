"""
Cybersecurity Investigation Lab Visual Components for the ModelShield
RAM / RAM++ Security Regression Gate.

Provides dark-themed, high-clarity visual components:
  1. Model Identity Panels (Baseline vs Candidate)
  2. Horizontal Execution Pipeline Diagram
  3. Prominent Security Verdict Panel
  4. Interactive Altair Comparative Analytics
  5. Tag Impact Explorer (Shared / Lost / New tags)
  6. Per-Image Forensic Inspection with Adversarial Perturbation Heatmap
"""

from __future__ import annotations

import html
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from evaluation.ram_eval import perturb_image_tensor
from evaluation.regression_gate import (
    EvaluationSnapshot,
    GateVerdict,
    MetricStatus,
    RegressionGateResult,
)


# ── Cyber-Lab Theme Palette ─────────────────────────────────────────────────
_BG_CARD = "#0f172a"
_BG_SUB = "#1e293b"
_BORDER = "#334155"
_TEXT_PRIMARY = "#f8fafc"
_TEXT_MUTED = "#94a3b8"
_ACCENT_CYAN = "#38bdf8"
_PASS_GREEN = "#22c55e"
_WARN_AMBER = "#f59e0b"
_FAIL_RED = "#ef4444"


def inject_cyber_lab_css() -> None:
    """Injects scoped CSS for the dark cybersecurity investigation lab panels."""
    st.markdown(
        f"""
        <style>
        .ms-id-card {{
            background: linear-gradient(145deg, {_BG_CARD}, #131c31);
            border: 1px solid {_BORDER};
            border-radius: 10px;
            padding: 14px 16px;
            margin-bottom: 10px;
            color: {_TEXT_PRIMARY};
        }}
        .ms-id-card.baseline {{ border-left: 4px solid {_ACCENT_CYAN}; }}
        .ms-id-card.candidate {{ border-left: 4px solid #a855f7; }}
        .ms-badge {{
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 2px 8px;
            border-radius: 999px;
            margin-bottom: 6px;
        }}
        .ms-badge-base {{ background: rgba(56, 189, 248, 0.16); color: {_ACCENT_CYAN}; border: 1px solid rgba(56, 189, 248, 0.4); }}
        .ms-badge-cand {{ background: rgba(168, 85, 247, 0.16); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.4); }}
        .ms-id-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 6px 14px;
            margin-top: 8px;
            font-size: 0.84rem;
        }}
        .ms-id-label {{ color: {_TEXT_MUTED}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em; }}
        .ms-id-val {{ color: {_TEXT_PRIMARY}; font-weight: 600; word-break: break-word; }}
        .ms-pipeline {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 6px;
            background: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 10px;
            padding: 12px 14px;
            margin: 10px 0 14px 0;
            overflow-x: auto;
        }}
        .ms-step {{
            flex: 1;
            min-width: 118px;
            background: {_BG_SUB};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 8px 10px;
            text-align: center;
            font-size: 0.78rem;
            color: {_TEXT_MUTED};
        }}
        .ms-step.done {{ border-color: rgba(34, 197, 94, 0.55); color: {_TEXT_PRIMARY}; background: rgba(34, 197, 94, 0.08); }}
        .ms-step.active {{ border-color: {_ACCENT_CYAN}; color: {_TEXT_PRIMARY}; background: rgba(56, 189, 248, 0.12); }}
        .ms-step.warn {{ border-color: {_WARN_AMBER}; color: {_TEXT_PRIMARY}; background: rgba(245, 158, 11, 0.12); }}
        .ms-step.fail {{ border-color: {_FAIL_RED}; color: {_TEXT_PRIMARY}; background: rgba(239, 68, 68, 0.12); }}
        .ms-arrow {{ color: {_TEXT_MUTED}; font-weight: 700; font-size: 1rem; }}
        .ms-verdict-hero {{
            border-radius: 12px;
            padding: 16px 20px;
            margin: 10px 0 14px 0;
            border: 1px solid {_BORDER};
            background: {_BG_CARD};
            color: {_TEXT_PRIMARY};
        }}
        .ms-verdict-hero.PASS {{ border-left: 6px solid {_PASS_GREEN}; background: linear-gradient(90deg, rgba(34,197,94,0.12), {_BG_CARD} 45%); }}
        .ms-verdict-hero.WARN {{ border-left: 6px solid {_WARN_AMBER}; background: linear-gradient(90deg, rgba(245,158,11,0.14), {_BG_CARD} 45%); }}
        .ms-verdict-hero.FAIL {{ border-left: 6px solid {_FAIL_RED}; background: linear-gradient(90deg, rgba(239,68,68,0.15), {_BG_CARD} 45%); }}
        .ms-tag-pill {{
            display: inline-block;
            padding: 3px 9px;
            margin: 3px 4px 3px 0;
            border-radius: 6px;
            font-size: 0.80rem;
            font-weight: 500;
        }}
        .ms-tag-shared {{ background: rgba(20, 184, 166, 0.16); color: #5eead4; border: 1px solid rgba(20, 184, 166, 0.4); }}
        .ms-tag-lost {{ background: rgba(239, 68, 68, 0.16); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .ms-tag-new {{ background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .ms-tag-neutral {{ background: rgba(148, 163, 184, 0.14); color: #e2e8f0; border: 1px solid rgba(148, 163, 184, 0.35); }}
        .ms-thumb-meta {{
            background: {_BG_CARD};
            border: 1px solid {_BORDER};
            border-radius: 8px;
            padding: 8px 10px;
            margin-top: 4px;
            margin-bottom: 6px;
            font-size: 0.78rem;
            color: {_TEXT_PRIMARY};
        }}
        .ms-status-pill {{
            display: inline-block;
            font-size: 0.68rem;
            font-weight: 700;
            padding: 1px 7px;
            border-radius: 999px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        .ms-status-OK {{ background: rgba(34, 197, 94, 0.16); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.4); }}
        .ms-status-WARN {{ background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .ms-status-FAIL {{ background: rgba(239, 68, 68, 0.16); color: #fca5a5; border: 1px solid rgba(239, 68, 68, 0.4); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _format_params(total_params: Any) -> str:
    if isinstance(total_params, (int, float)) and total_params > 0:
        if total_params >= 1_000_000:
            return f"{total_params / 1_000_000:.1f}M ({int(total_params):,})"
        return f"{int(total_params):,}"
    return "Unavailable"


def render_model_identity_panels(
    base_meta: Dict[str, Any],
    cand_meta: Dict[str, Any],
) -> None:
    """Renders side-by-side Model Identity Panels for Baseline and Candidate."""
    col_b, col_c = st.columns(2)
    for col, role, meta in [
        (col_b, "baseline", base_meta),
        (col_c, "candidate", cand_meta),
    ]:
        extra = meta.get("extra", {}) if isinstance(meta.get("extra"), dict) else {}
        model_name = html.escape(str(meta.get("model_name", "Unknown")))
        arch_raw = str(meta.get("architecture") or extra.get("architecture_display") or "Unknown")
        if arch_raw == "ram_swin_large":
            arch_disp = "RAM (ram)"
        elif arch_raw == "ram_plus_swin_large":
            arch_disp = "RAM++ (ram_plus)"
        else:
            arch_disp = str(extra.get("architecture_display") or arch_raw)
        backbone = html.escape(str(extra.get("backbone") or "Swin-Large"))
        ckpt = html.escape(str(extra.get("checkpoint_name") or "In-memory / Snapshot"))
        dev = html.escape(str(extra.get("device") or "auto"))
        params = html.escape(_format_params(extra.get("total_parameters")))
        badge_cls = "ms-badge-base" if role == "baseline" else "ms-badge-cand"
        role_label = "BASELINE MODEL" if role == "baseline" else "CANDIDATE MODEL"

        with col:
            st.markdown(
                f"""
                <div class="ms-id-card {role}">
                    <span class="ms-badge {badge_cls}">{role_label}</span>
                    <div style="font-size:1.05rem;font-weight:700;margin-bottom:2px;">{model_name}</div>
                    <div class="ms-id-grid">
                        <div><div class="ms-id-label">Architecture</div><div class="ms-id-val">{html.escape(arch_disp)}</div></div>
                        <div><div class="ms-id-label">Backbone</div><div class="ms-id-val">{backbone}</div></div>
                        <div><div class="ms-id-label">Checkpoint</div><div class="ms-id-val">{ckpt}</div></div>
                        <div><div class="ms-id-label">Device</div><div class="ms-id-val">{dev}</div></div>
                        <div><div class="ms-id-label">Parameters</div><div class="ms-id-val">{params}</div></div>
                        <div><div class="ms-id-label">Vocabulary</div><div class="ms-id-val">{meta.get('num_tags', 4585)} tags</div></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_execution_pipeline(
    gate_result: Optional[RegressionGateResult] = None,
    has_baseline: bool = False,
    has_candidate: bool = False,
    is_running: bool = False,
    active_stage: int = 0,
) -> None:
    """
    Renders the horizontal execution pipeline:
    Input Dataset → Baseline Model → Candidate Model → Adversarial Evaluation → Regression Analysis → Verdict
    """
    stages = [
        ("1. Input Dataset", "Benchmark / Upload"),
        ("2. Baseline Model", "Reference Inference"),
        ("3. Candidate Model", "Target Inference"),
        ("4. Adversarial Eval", "L∞ Perturbations"),
        ("5. Regression Analysis", "Delta & Trust Audit"),
        ("6. Gate Verdict", gate_result.verdict.value if gate_result else "Pending"),
    ]

    html_parts = ['<div class="ms-pipeline">']
    for idx, (title, sub) in enumerate(stages):
        cls = "ms-step"
        icon = "○"
        if gate_result is not None:
            if idx < 5:
                cls += " done"
                icon = "✓"
            else:
                v = gate_result.verdict
                cls += " done" if v == GateVerdict.PASS else (" warn" if v == GateVerdict.WARN else " fail")
                icon = "✅" if v == GateVerdict.PASS else ("⚠️" if v == GateVerdict.WARN else "❌")
        elif is_running:
            if idx < active_stage:
                cls += " done"
                icon = "✓"
            elif idx == active_stage:
                cls += " active"
                icon = "⟳"
        else:
            if idx == 0 or (idx == 1 and has_baseline) or (idx == 2 and has_candidate):
                cls += " done"
                icon = "✓"

        html_parts.append(
            f'<div class="{cls}"><div style="font-weight:700;">{icon} {html.escape(title)}</div>'
            f'<div style="font-size:0.70rem;opacity:0.85;">{html.escape(sub)}</div></div>'
        )
        if idx < len(stages) - 1:
            html_parts.append('<div class="ms-arrow">→</div>')
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def render_security_verdict_panel(res: RegressionGateResult) -> None:
    """Renders the prominent PASS/WARN/FAIL Security Verdict Panel."""
    v = res.verdict.value
    icon = "✅" if res.verdict == GateVerdict.PASS else ("⚠️" if res.verdict == GateVerdict.WARN else "❌")
    n_samples = res.baseline_metadata.get("num_samples", 0)

    # Identify main regression detected (largest negative impact FAIL or WARN metric)
    main_reg = "None detected"
    worst_deltas = [
        d for d in res.metric_deltas.values() if d.status in (MetricStatus.FAIL, MetricStatus.WARN)
    ]
    if worst_deltas:
        worst = sorted(worst_deltas, key=lambda x: (0 if x.status == MetricStatus.FAIL else 1, -abs(x.delta)))[0]
        sign = "+" if worst.delta > 0 else ""
        main_reg = f"{worst.metric_name} ({sign}{worst.delta:.2f})"
    elif res.failures:
        main_reg = res.failures[0][:90]

    st.markdown(
        f"""
        <div class="ms-verdict-hero {v}">
            <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;">
                <div>
                    <div style="font-size:0.76rem;letter-spacing:0.08em;text-transform:uppercase;color:{_TEXT_MUTED};">
                        SECURITY REGRESSION GATE VERDICT
                    </div>
                    <div style="font-size:1.6rem;font-weight:800;margin-top:2px;">{icon} {v}</div>
                </div>
                <div style="font-size:0.88rem;color:{_TEXT_PRIMARY};max-width:620px;">
                    {html.escape(res.summary)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Critical Failures", str(len(res.failures)))
    c2.metric("Warnings", str(len(res.warnings)))
    c3.metric("Passed Checks", str(len(res.passed_checks)))
    c4.metric("Benchmark Size", f"{n_samples} images")
    c5.metric("Main Regression", main_reg if len(main_reg) <= 26 else main_reg[:24] + "…", help=main_reg)


def render_comparative_analytics(
    res: RegressionGateResult,
    base_snap: EvaluationSnapshot,
    cand_snap: EvaluationSnapshot,
) -> None:
    """Renders interactive Altair comparative analytics charts."""
    if not res.metric_deltas:
        return

    st.subheader("📊 Comparative Security Analytics")

    rate_keys = [
        "mean_jaccard_similarity",
        "mean_tag_retention_rate",
        "tag_agreement_rate",
        "semantic_consistency",
    ]
    rate_rows = []
    for k in rate_keys:
        if k in res.metric_deltas:
            d = res.metric_deltas[k]
            label = k.replace("_", " ").title()
            rate_rows.append({"Metric": label, "Model": "Baseline", "Value (%)": d.baseline_value})
            rate_rows.append({"Metric": label, "Model": "Candidate", "Value (%)": d.candidate_value})

    delta_rows = []
    for k, d in res.metric_deltas.items():
        delta_rows.append({
            "Metric": k,
            "Delta": round(d.delta, 2),
            "Status": d.status.value,
            "Direction": "Higher is better" if d.direction.value == "HIGHER_IS_BETTER" else "Lower is better",
        })

    col_l, col_r = st.columns(2)
    with col_l:
        if rate_rows:
            st.markdown("**Clean vs Adversarial Robustness (%)**")
            df_rates = pd.DataFrame(rate_rows)
            chart_rates = (
                alt.Chart(df_rates)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Metric:N", axis=alt.Axis(labelAngle=-20, title=None)),
                    y=alt.Y("Value (%):Q", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color(
                        "Model:N",
                        scale=alt.Scale(domain=["Baseline", "Candidate"], range=[_ACCENT_CYAN, "#a855f7"]),
                    ),
                    xOffset="Model:N",
                    tooltip=["Metric", "Model", alt.Tooltip("Value (%):Q", format=".2f")],
                )
                .properties(height=280)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_rates, use_container_width=True)

    with col_r:
        if delta_rows:
            st.markdown("**Metric Delta (Candidate − Baseline)**")
            df_delta = pd.DataFrame(delta_rows)
            chart_delta = (
                alt.Chart(df_delta)
                .mark_bar(cornerRadius=3)
                .encode(
                    y=alt.Y("Metric:N", sort=None, title=None),
                    x=alt.X("Delta:Q", title="Delta (Candidate − Baseline)"),
                    color=alt.Color(
                        "Status:N",
                        scale=alt.Scale(
                            domain=["OK", "WARN", "FAIL"],
                            range=[_PASS_GREEN, _WARN_AMBER, _FAIL_RED],
                        ),
                    ),
                    tooltip=["Metric", alt.Tooltip("Delta:Q", format="+.2f"), "Status", "Direction"],
                )
                .properties(height=280)
                .configure_view(strokeWidth=0)
            )
            st.altair_chart(chart_delta, use_container_width=True)

    # Latency & Tag Count Comparison
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        b_lat = base_snap.metrics.get("mean_latency_ms", 0.0)
        c_lat = cand_snap.metrics.get("mean_latency_ms", 0.0)
        df_lat = pd.DataFrame([
            {"Model": "Baseline", "Median Latency (ms)": b_lat},
            {"Model": "Candidate", "Median Latency (ms)": c_lat},
        ])
        st.markdown("**Inference Latency Comparison (Lower is Better)**")
        chart_lat = (
            alt.Chart(df_lat)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=50)
            .encode(
                x=alt.X("Model:N", title=None),
                y=alt.Y("Median Latency (ms):Q"),
                color=alt.Color("Model:N", scale=alt.Scale(domain=["Baseline", "Candidate"], range=[_ACCENT_CYAN, "#a855f7"]), legend=None),
                tooltip=["Model", alt.Tooltip("Median Latency (ms):Q", format=".2f")],
            )
            .properties(height=220)
        )
        st.altair_chart(chart_lat, use_container_width=True)

    with col_b2:
        tag_metrics = [
            ("Avg Clean Tags", "avg_tags_detected"),
            ("Tags Lost / Img", "tags_lost_per_image"),
            ("New Tags / Img", "new_tags_per_image"),
        ]
        tag_rows = []
        for label, key in tag_metrics:
            if key in base_snap.metrics or key in cand_snap.metrics:
                tag_rows.append({"Metric": label, "Model": "Baseline", "Tags": base_snap.metrics.get(key, 0.0)})
                tag_rows.append({"Metric": label, "Model": "Candidate", "Tags": cand_snap.metrics.get(key, 0.0)})
        if tag_rows:
            st.markdown("**Tag Volume & Perturbation Churn**")
            df_tags = pd.DataFrame(tag_rows)
            chart_tags = (
                alt.Chart(df_tags)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Metric:N", title=None),
                    y=alt.Y("Tags:Q"),
                    color=alt.Color("Model:N", scale=alt.Scale(domain=["Baseline", "Candidate"], range=[_ACCENT_CYAN, "#a855f7"])),
                    xOffset="Model:N",
                    tooltip=["Metric", "Model", alt.Tooltip("Tags:Q", format=".2f")],
                )
                .properties(height=220)
            )
            st.altair_chart(chart_tags, use_container_width=True)


def _render_tag_pills(tags: List[str], pill_cls: str) -> str:
    if not tags:
        return '<span style="color:#64748b;font-size:0.82rem;">(none)</span>'
    return "".join(f'<span class="ms-tag-pill {pill_cls}">{html.escape(t)}</span>' for t in tags)


def compute_forensic_gallery_summary(forensic_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Computes dataset-level summary statistics and per-image gallery rows from
    live RAM/RAM++ forensic records. Pure helper for deterministic UI and testing.
    """
    if not forensic_data:
        return None
    images: List[Tuple[str, Image.Image]] = forensic_data.get("images") or []
    base_records: List[Dict[str, Any]] = forensic_data.get("baseline_per_image") or []
    cand_records: List[Dict[str, Any]] = forensic_data.get("candidate_per_image") or []

    if not images or not cand_records:
        return None

    rows: List[Dict[str, Any]] = []
    n_total = min(len(images), len(cand_records))

    for idx in range(n_total):
        img_name, _ = images[idx]
        c_rec = cand_records[idx]
        b_rec = base_records[idx] if idx < len(base_records) else {}

        c_clean = c_rec.get("clean_tags", [])
        b_clean = b_rec.get("clean_tags", []) if b_rec else []
        c_set = set(c_clean)
        b_set = set(b_clean)

        shared_vs_base = sorted(b_set & c_set) if b_rec else sorted(c_set)
        lost_vs_base = sorted(b_set - c_set) if b_rec else []
        new_vs_base = sorted(c_set - b_set) if b_rec else []

        adv_lost = c_rec.get("lost_tags", [])
        adv_new = c_rec.get("new_tags", [])
        retention = float(c_rec.get("retention", 100.0))
        jaccard = float(c_rec.get("jaccard", 100.0))

        # Regression status: consider both adversarial tag loss and cross-model loss vs baseline
        effective_lost = len(lost_vs_base) if b_rec else len(adv_lost)
        effective_new = len(new_vs_base) if b_rec else len(adv_new)

        if retention < 60.0 or jaccard < 50.0 or (b_rec and len(b_set) > 0 and len(lost_vs_base) >= max(2, len(b_set) // 2)):
            status = "FAIL"
        elif retention < 90.0 or jaccard < 85.0 or effective_lost > 0:
            status = "WARN"
        else:
            status = "OK"

        rows.append({
            "idx": idx,
            "image_name": img_name,
            "status": status,
            "retention": round(retention, 2),
            "jaccard": round(jaccard, 2),
            "adv_lost_count": len(adv_lost),
            "adv_new_count": len(adv_new),
            "lost_vs_base_count": len(lost_vs_base),
            "new_vs_base_count": len(new_vs_base),
            "display_lost_count": effective_lost,
            "display_new_count": effective_new,
            "clean_tags": c_clean,
            "adv_tags": c_rec.get("adv_tags", []),
            "overlap_tags": c_rec.get("overlap_tags", []),
            "adv_lost_tags": adv_lost,
            "adv_new_tags": adv_new,
            "base_clean_tags": b_clean,
            "shared_vs_base": shared_vs_base,
            "lost_vs_base": lost_vs_base,
            "new_vs_base": new_vs_base,
            "adversarial_evaluated": bool(c_rec.get("adversarial_evaluated", forensic_data.get("evaluate_adversarial", True))),
            "perturbation_linf": c_rec.get("perturbation_linf"),
            "perturbation_l2": c_rec.get("perturbation_l2"),
            "perturbation_mean_abs": c_rec.get("perturbation_mean_abs"),
        })

    if not rows:
        return None

    n = float(len(rows))
    images_with_regressions = sum(1 for r in rows if r["status"] != "OK")
    avg_retention = round(sum(r["retention"] for r in rows) / n, 2)
    avg_jaccard = round(sum(r["jaccard"] for r in rows) / n, 2)
    avg_lost = round(sum(r["display_lost_count"] for r in rows) / n, 2)
    avg_new = round(sum(r["display_new_count"] for r in rows) / n, 2)

    return {
        "total_images": len(rows),
        "images_with_regressions": images_with_regressions,
        "avg_retention": avg_retention,
        "avg_jaccard": avg_jaccard,
        "avg_lost_tags": avg_lost,
        "avg_new_tags": avg_new,
        "has_baseline_comparison": bool(base_records),
        "rows": rows,
    }


def render_tag_impact_and_forensics(forensic_data: Optional[Dict[str, Any]]) -> None:
    """
    Renders the compact Multi-Image Forensic Report:
      - Dataset-level summary metrics
      - Aggregate adversarial & tag-churn visualizations
      - 4-column responsive thumbnail evidence gallery
      - Click-to-expand single-image forensic detail panel (avoids stacking full-size images)
    """
    summary = compute_forensic_gallery_summary(forensic_data)
    if not summary or not forensic_data:
        return

    images: List[Tuple[str, Image.Image]] = forensic_data.get("images") or []
    epsilon: float = float(forensic_data.get("epsilon", 8.0 / 255.0))
    rows: List[Dict[str, Any]] = summary["rows"]

    st.subheader("🔬 Multi-Image Forensic Report & Tag Impact Explorer")
    st.caption(
        "Compact dataset-wide forensic overview and responsive thumbnail gallery. "
        "Click **Inspect** on any thumbnail to expand its detailed tag comparison. "
        "(`inference_ram` returns discrete tag sets without confidence scores, so scores are not fabricated.)"
    )

    # ── B. Dataset-Level Summary ─────────────────────────────────────────────
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("Images Evaluated", str(summary["total_images"]))
    s2.metric("With Regressions", str(summary["images_with_regressions"]))
    s3.metric("Avg Tag Retention", f"{summary['avg_retention']:.1f}%")
    s4.metric("Avg Jaccard Sim.", f"{summary['avg_jaccard']:.1f}%")
    s5.metric("Avg Lost Tags / Img", f"{summary['avg_lost_tags']:.2f}")
    s6.metric("Avg New Tags / Img", f"{summary['avg_new_tags']:.2f}")

    # ── C. Aggregate Adversarial Visualization ───────────────────────────────
    with st.expander("📊 Aggregate Robustness & Tag Churn Charts", expanded=True):
        agg_c1, agg_c2 = st.columns(2)
        with agg_c1:
            rob_rows = []
            for r in rows:
                rob_rows.append({"Image": r["image_name"], "Metric": "Retention (%)", "Value": r["retention"]})
                rob_rows.append({"Image": r["image_name"], "Metric": "Jaccard (%)", "Value": r["jaccard"]})
            df_rob = pd.DataFrame(rob_rows)
            chart_rob = (
                alt.Chart(df_rob)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Image:N", sort=None, axis=alt.Axis(labelAngle=-25, title=None)),
                    y=alt.Y("Value:Q", scale=alt.Scale(domain=[0, 100]), title="Score (%)"),
                    color=alt.Color(
                        "Metric:N",
                        scale=alt.Scale(domain=["Retention (%)", "Jaccard (%)"], range=[_ACCENT_CYAN, "#a855f7"]),
                    ),
                    xOffset="Metric:N",
                    tooltip=["Image", "Metric", alt.Tooltip("Value:Q", format=".1f")],
                )
                .properties(height=220, title="Per-Image Robustness Distribution")
            )
            st.altair_chart(chart_rob, use_container_width=True)

        with agg_c2:
            churn_rows = []
            for r in rows:
                churn_rows.append({"Image": r["image_name"], "Type": "Lost Tags", "Count": r["display_lost_count"]})
                churn_rows.append({"Image": r["image_name"], "Type": "New Tags", "Count": r["display_new_count"]})
            df_churn = pd.DataFrame(churn_rows)
            chart_churn = (
                alt.Chart(df_churn)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(
                    x=alt.X("Image:N", sort=None, axis=alt.Axis(labelAngle=-25, title=None)),
                    y=alt.Y("Count:Q", title="Tag Count"),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Lost Tags", "New Tags"], range=[_FAIL_RED, _WARN_AMBER]),
                    ),
                    xOffset="Type:N",
                    tooltip=["Image", "Type", "Count"],
                )
                .properties(height=220, title="Lost vs. New Tags per Image")
            )
            st.altair_chart(chart_churn, use_container_width=True)

        # Real tensor perturbation intensity chart (only when actual stats exist)
        pert_rows = [
            {
                "Image": r["image_name"],
                "L∞ Norm": r["perturbation_linf"],
                "Mean |Δ|": r["perturbation_mean_abs"],
            }
            for r in rows
            if r.get("adversarial_evaluated") and r.get("perturbation_linf") is not None
        ]
        if pert_rows:
            df_pert = pd.DataFrame(pert_rows)
            chart_pert = (
                alt.Chart(df_pert)
                .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#38bdf8")
                .encode(
                    x=alt.X("Image:N", sort=None, axis=alt.Axis(labelAngle=-25, title=None)),
                    y=alt.Y("L∞ Norm:Q", title="Actual L∞ Perturbation"),
                    tooltip=["Image", alt.Tooltip("L∞ Norm:Q", format=".6f"), alt.Tooltip("Mean |Δ|:Q", format=".6f")],
                )
                .properties(height=160, title=f"Actual Tensor Perturbation Intensity (Budget ε = {epsilon:.6f})")
            )
            st.altair_chart(chart_pert, use_container_width=True)

    # ── A. Multi-Image Evidence Gallery (4 columns on wide screens) ──────────
    st.markdown("#### 🖼️ Multi-Image Evidence Gallery")
    if "ram_forensic_selected_idx" not in st.session_state or st.session_state["ram_forensic_selected_idx"] >= len(rows):
        st.session_state["ram_forensic_selected_idx"] = 0

    for row_start in range(0, len(rows), 4):
        cols = st.columns(4)
        for col_offset in range(4):
            idx = row_start + col_offset
            if idx >= len(rows):
                break
            r = rows[idx]
            img_name, pil_img = images[idx]
            thumb = pil_img.convert("RGB").copy()
            thumb.thumbnail((160, 160))

            with cols[col_offset]:
                st.image(thumb, use_container_width=True)
                st.markdown(
                    f"""
                    <div class="ms-thumb-meta">
                        <div style="display:flex;justify-content:space-between;align-items:center;gap:4px;">
                            <strong style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="{html.escape(img_name)}">{html.escape(img_name)}</strong>
                            <span class="ms-status-pill ms-status-{r['status']}">{r['status']}</span>
                        </div>
                        <div style="color:{_TEXT_MUTED};margin-top:3px;">
                            Retention: <strong>{r['retention']:.0f}%</strong> · Lost: <strong>{r['display_lost_count']}</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if st.button("🔍 Inspect", key=f"ram_gallery_btn_{idx}", use_container_width=True):
                    st.session_state["ram_forensic_selected_idx"] = idx

    # ── Click-to-Expand Single-Image Forensic Detail Panel ───────────────────
    sel_idx = int(st.session_state.get("ram_forensic_selected_idx", 0))
    sel_row = rows[sel_idx]
    sel_name, sel_pil = images[sel_idx]

    with st.expander(f"🔎 Expanded Forensic Inspection — {sel_name} ({sel_row['status']})", expanded=True):
        det_c1, det_c2 = st.columns([1, 2])
        with det_c1:
            rgb_384 = sel_pil.convert("RGB").resize((384, 384))
            st.image(rgb_384, caption=f"Original: {sel_name}", use_container_width=True)
            if sel_row["adversarial_evaluated"] and sel_row.get("perturbation_linf") is not None:
                st.caption(
                    f"Actual Tensor Perturbation — L∞: `{sel_row['perturbation_linf']:.6f}` · "
                    f"L2: `{sel_row['perturbation_l2']:.4f}` · "
                    f"Mean |Δ|: `{sel_row['perturbation_mean_abs']:.6f}`"
                )

        with det_c2:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Tag Retention", f"{sel_row['retention']:.1f}%")
            m2.metric("Jaccard Sim.", f"{sel_row['jaccard']:.1f}%")
            m3.metric("Lost Tags", str(sel_row["display_lost_count"]))
            m4.metric("New Tags", str(sel_row["display_new_count"]))

            if summary["has_baseline_comparison"]:
                st.markdown("**Baseline vs. Candidate Clean Tags**")
                bc1, bc2 = st.columns(2)
                with bc1:
                    st.caption(f"Baseline Tags ({len(sel_row['base_clean_tags'])})")
                    st.markdown(_render_tag_pills(sel_row["base_clean_tags"], "ms-tag-neutral"), unsafe_allow_html=True)
                with bc2:
                    st.caption(f"Candidate Tags ({len(sel_row['clean_tags'])})")
                    st.markdown(_render_tag_pills(sel_row["clean_tags"], "ms-tag-neutral"), unsafe_allow_html=True)

                sc1, sc2, sc3 = st.columns(3)
                with sc1:
                    st.caption(f"Shared ({len(sel_row['shared_vs_base'])})")
                    st.markdown(_render_tag_pills(sel_row["shared_vs_base"], "ms-tag-shared"), unsafe_allow_html=True)
                with sc2:
                    st.caption(f"Lost vs Baseline ({len(sel_row['lost_vs_base'])})")
                    st.markdown(_render_tag_pills(sel_row["lost_vs_base"], "ms-tag-lost"), unsafe_allow_html=True)
                with sc3:
                    st.caption(f"New vs Baseline ({len(sel_row['new_vs_base'])})")
                    st.markdown(_render_tag_pills(sel_row["new_vs_base"], "ms-tag-new"), unsafe_allow_html=True)

            if sel_row["adversarial_evaluated"]:
                st.markdown("**Adversarial Comparison (Clean ↔ Perturbed)**")
                ac1, ac2 = st.columns(2)
                with ac1:
                    st.caption(f"Adversarial Predictions ({len(sel_row['adv_tags'])})")
                    st.markdown(_render_tag_pills(sel_row["adv_tags"], "ms-tag-neutral"), unsafe_allow_html=True)
                with ac2:
                    st.caption(f"Adv Lost ({len(sel_row['adv_lost_tags'])}) / Injected ({len(sel_row['adv_new_tags'])})")
                    st.markdown(
                        _render_tag_pills(sel_row["adv_lost_tags"], "ms-tag-lost")
                        + _render_tag_pills(sel_row["adv_new_tags"], "ms-tag-new"),
                        unsafe_allow_html=True,
                    )
