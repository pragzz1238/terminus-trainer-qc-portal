"""Terminus QC Portal — shared UI styles and HTML components."""

from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "favicon.png"
if not FAVICON_PATH.is_file():
    FAVICON_PATH = APP_DIR / "favicon.ico"
APP_VERSION = "1.0"
PRODUCT_NAME = "Terminus Edition-2"
PRODUCT_TAGLINE = "Task Quality Checker"


def inject_global_css() -> None:
    st.markdown(
        """
<style>
    :root {
        --t-bg: #f4f6f9;
        --t-surface: #ffffff;
        --t-border: #dde3ea;
        --t-text: #0f172a;
        --t-muted: #64748b;
        --t-accent: #0f766e;
        --t-accent-soft: #ecfdf5;
        --t-pass: #047857;
        --t-fail: #b91c1c;
        --t-warn: #b45309;
        --t-shadow: 0 1px 2px rgba(15, 23, 42, 0.05), 0 8px 24px rgba(15, 23, 42, 0.06);
    }
    .stApp {
        background: linear-gradient(180deg, #f8fafc 0%, var(--t-bg) 120px, var(--t-bg) 100%);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .block-container {
        padding-top: 1.25rem;
        padding-bottom: 3rem;
        max-width: 1080px;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
        font-family: inherit !important;
    }
    code, .stCaption code {
        font-family: ui-monospace, SFMono-Regular, Menlo, monospace !important;
        font-size: 0.85em;
    }
    #MainMenu, footer, header[data-testid="stHeader"] {
        visibility: hidden;
        height: 0;
    }
    div[data-testid="stSidebar"] { display: none; }

    .t-topbar {
        display: flex; align-items: center; justify-content: space-between;
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 16px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
        box-shadow: var(--t-shadow);
    }
    .t-brand { display: flex; align-items: center; gap: 0.9rem; }
    .t-logo {
        width: 44px; height: 44px; border-radius: 12px;
        background: linear-gradient(145deg, #115e59 0%, #0f766e 100%);
        color: #fff; font-weight: 700; font-size: 0.95rem;
        display: flex; align-items: center; justify-content: center;
        letter-spacing: -0.02em;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.15);
    }
    .t-logo-img {
        width: 44px; height: 44px; border-radius: 12px;
        object-fit: contain; display: block;
        box-shadow: var(--t-shadow);
    }
    .t-title { margin: 0; font-size: 1.35rem; font-weight: 700; color: var(--t-text); line-height: 1.2; }
    .t-subtitle { margin: 0.15rem 0 0 0; color: var(--t-muted); font-size: 0.92rem; }
    .t-badge-row { display: flex; flex-wrap: wrap; gap: 0.45rem; justify-content: flex-end; }
    .t-badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.3rem 0.65rem; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid var(--t-border); background: #f8fafc; color: #334155;
    }
    .t-badge.ok { background: var(--t-accent-soft); border-color: #99f6e4; color: #115e59; }
    .t-badge.warn { background: #fffbeb; border-color: #fde68a; color: #92400e; }

    .t-hero {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-left: 4px solid var(--t-accent);
        border-radius: 16px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--t-shadow);
    }
    .t-hero h2 { margin: 0 0 0.45rem 0; font-size: 1.05rem; color: var(--t-text); }
    .t-hero p { margin: 0; color: var(--t-muted); font-size: 0.95rem; line-height: 1.55; }

    .t-stepper {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.65rem;
        margin-bottom: 1.25rem;
    }
    .t-step {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 12px;
        padding: 0.75rem 0.9rem;
        text-align: left;
    }
    .t-step.active { border-color: #5eead4; background: #f0fdfa; box-shadow: inset 0 0 0 1px #99f6e4; }
    .t-step.done { border-color: #bbf7d0; background: #f0fdf4; }
    .t-step-num {
        display: inline-block; font-size: 0.72rem; font-weight: 700;
        color: var(--t-accent); letter-spacing: 0.06em; text-transform: uppercase;
    }
    .t-step-label { display: block; margin-top: 0.2rem; font-size: 0.88rem; font-weight: 600; color: var(--t-text); }

    .t-panel {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 16px;
        padding: 1.25rem 1.35rem 1.1rem;
        margin-bottom: 1.1rem;
        box-shadow: var(--t-shadow);
    }
    .t-panel-head {
        display: flex; align-items: flex-start; justify-content: space-between;
        gap: 1rem; margin-bottom: 0.85rem;
    }
    .t-panel-title { margin: 0; font-size: 1.08rem; font-weight: 700; color: var(--t-text); }
    .t-panel-desc { margin: 0.25rem 0 0 0; color: var(--t-muted); font-size: 0.88rem; line-height: 1.45; }
    .t-step-pill {
        flex-shrink: 0;
        background: #f1f5f9; color: #475569;
        border: 1px solid var(--t-border);
        border-radius: 999px; padding: 0.28rem 0.7rem;
        font-size: 0.74rem; font-weight: 700; letter-spacing: 0.04em;
    }

    .t-metrics {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.75rem;
        margin: 1rem 0 1.1rem 0;
    }
    @media (max-width: 900px) { .t-metrics { grid-template-columns: repeat(2, 1fr); } }
    .t-metric {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 12px;
        padding: 0.95rem 1rem;
        box-shadow: var(--t-shadow);
    }
    .t-metric.pass { border-left: 4px solid var(--t-pass); }
    .t-metric.fail { border-left: 4px solid var(--t-fail); }
    .t-metric.warn { border-left: 4px solid var(--t-warn); }
    .t-metric.neutral { border-left: 4px solid #94a3b8; }
    .t-metric .label {
        color: var(--t-muted); font-size: 0.72rem; text-transform: uppercase;
        letter-spacing: 0.07em; font-weight: 600;
    }
    .t-metric .value {
        margin-top: 0.35rem; font-size: 1.2rem; font-weight: 700; color: var(--t-text);
        word-break: break-word;
    }
    .t-metric .value.small { font-size: 0.98rem; font-weight: 600; }

    .t-verdict {
        border-radius: 14px; padding: 1rem 1.2rem; margin: 0.75rem 0 1rem 0;
        border: 1px solid var(--t-border);
        display: flex; align-items: center; justify-content: space-between; gap: 1rem;
    }
    .t-verdict.pass { background: #ecfdf5; border-color: #6ee7b7; }
    .t-verdict.fail { background: #fef2f2; border-color: #fca5a5; }
    .t-verdict .title { margin: 0; font-size: 1.15rem; font-weight: 700; }
    .t-verdict.pass .title { color: var(--t-pass); }
    .t-verdict.fail .title { color: var(--t-fail); }
    .t-verdict .hint { margin: 0.2rem 0 0 0; color: var(--t-muted); font-size: 0.88rem; }

    .t-download {
        background: var(--t-surface);
        border: 1px solid #bfdbfe;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin: 1rem 0 1.25rem 0;
        box-shadow: var(--t-shadow);
    }
    .t-download h3 { margin: 0 0 0.3rem 0; color: #1e3a8a; font-size: 1rem; font-weight: 700; }
    .t-download p { margin: 0; color: var(--t-muted); font-size: 0.88rem; }

    .t-footer {
        margin-top: 2rem; padding-top: 1rem;
        border-top: 1px solid var(--t-border);
        color: var(--t-muted); font-size: 0.8rem; text-align: center;
    }

    .stTabs [data-baseweb="tab-list"] { gap: 0.35rem; }
    .stTabs [data-baseweb="tab"] {
        height: 2.6rem; border-radius: 10px 10px 0 0;
        padding-left: 1rem; padding-right: 1rem;
        font-weight: 600; font-size: 0.88rem;
    }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed #cbd5e1; border-radius: 12px; background: #f8fafc;
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--t-border) !important;
        border-radius: 16px !important;
        box-shadow: var(--t-shadow);
        padding: 0.35rem 0.15rem 0.15rem;
        margin-bottom: 0.75rem;
        background: var(--t-surface);
    }
    .stButton > button[kind="primary"] {
        border-radius: 10px; font-weight: 600;
        background: linear-gradient(180deg, #0f766e 0%, #115e59 100%);
        border: 1px solid #0f766e;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def _favicon_data_uri() -> str:
    if not FAVICON_PATH.is_file():
        return ""
    encoded = base64.b64encode(FAVICON_PATH.read_bytes()).decode("ascii")
    mime = "image/png" if FAVICON_PATH.suffix.lower() == ".png" else "image/x-icon"
    return f"data:{mime};base64,{encoded}"


def inject_page_favicon() -> None:
    """Force browser tab icon — Streamlit only reliably serves PNG for page_icon."""
    uri = _favicon_data_uri()
    if not uri:
        return
    mime = "image/png" if FAVICON_PATH.suffix.lower() == ".png" else "image/x-icon"
    # Markdown <link> lands in the body; inject into document.head for browser tabs.
    components.html(
        f"""
<script>
(function () {{
  var doc = window.parent.document;
  var href = {uri!r};
  doc.querySelectorAll("link[rel*='icon']").forEach(function (el) {{ el.remove(); }});
  var link = doc.createElement("link");
  link.rel = "icon";
  link.type = {mime!r};
  link.href = href;
  doc.head.appendChild(link);
}})();
</script>
""",
        height=0,
        width=0,
    )


def render_topbar(llm_ready: bool, provider: str, model: str) -> None:
    llm_badge = (
        f'<span class="t-badge ok">LLM · {provider} · {model}</span>'
        if llm_ready
        else '<span class="t-badge warn">LLM not configured</span>'
    )
    favicon_uri = _favicon_data_uri()
    logo_html = (
        f'<img src="{favicon_uri}" alt="Terminal Bench" class="t-logo-img" />'
        if favicon_uri
        else '<div class="t-logo">T2</div>'
    )
    st.markdown(
        f"""
<div class="t-topbar">
  <div class="t-brand">
    {logo_html}
    <div>
      <p class="t-title">{PRODUCT_NAME}</p>
      <p class="t-subtitle">{PRODUCT_TAGLINE}</p>
    </div>
  </div>
  <div class="t-badge-row">
    <span class="t-badge">Terminal-Bench</span>
    <span class="t-badge">3-layer QC</span>
    {llm_badge}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
<div class="t-hero">
  <h2>Pre-submission quality gate for trainer tasks</h2>
  <p>Run <strong>instruction similarity</strong> or <strong>full task QC</strong> independently.
  Full assessment checks folder structure first, then static rules, similarity, and LLM alignment
  against <strong>8 accepted reference tasks</strong>.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_workflow_stepper(active_step: int) -> None:
    steps = [
        (0, "Instruction check"),
        (1, "Upload task zip"),
        (2, "QC results"),
    ]
    cells = []
    for num, label in steps:
        cls = "t-step"
        if num == active_step:
            cls += " active"
        elif num < active_step:
            cls += " done"
        cells.append(
            f'<div class="{cls}"><span class="t-step-num">Step {num}</span>'
            f'<span class="t-step-label">{label}</span></div>'
        )
    st.markdown(f'<div class="t-stepper">{"".join(cells)}</div>', unsafe_allow_html=True)


def render_panel_header(step: int, title: str, description: str) -> None:
    st.markdown(
        f"""
<div class="t-panel-head">
  <div>
    <p class="t-panel-title">{title}</p>
    <p class="t-panel-desc">{description}</p>
  </div>
  <span class="t-step-pill">STEP {step}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def render_section_header(step: int, title: str, description: str) -> None:
    """Header for sections where Streamlit widgets cannot sit inside a custom HTML wrapper."""
    st.markdown(
        f"""
<div class="t-panel" style="padding-bottom:0.85rem;margin-bottom:0.75rem;">
  <div class="t-panel-head" style="margin-bottom:0;">
    <div>
      <p class="t-panel-title">{title}</p>
      <p class="t-panel-desc">{description}</p>
    </div>
    <span class="t-step-pill">STEP {step}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_metric_grid(cards: list[tuple[str, str, str]]) -> None:
    """cards: list of (label, value, tone) where tone is pass|fail|warn|neutral."""
    html = '<div class="t-metrics">'
    for label, value, tone in cards:
        size_class = " small" if len(value) > 18 else ""
        html += (
            f'<div class="t-metric {tone}"><div class="label">{label}</div>'
            f'<div class="value{size_class}">{value}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def render_verdict_banner(passed: bool, hint: str = "") -> None:
    title = "Ready to submit" if passed else "Needs fixes before submission"
    cls = "pass" if passed else "fail"
    hint_html = f'<p class="hint">{hint}</p>' if hint else ""
    st.markdown(
        f"""
<div class="t-verdict {cls}">
  <div>
    <p class="title">{title}</p>
    {hint_html}
  </div>
  <span class="t-badge {"ok" if passed else "warn"}">{"PASS" if passed else "REVIEW"}</span>
</div>
""",
        unsafe_allow_html=True,
    )


def severity_badge(severity: str) -> str:
    colors = {
        "CRITICAL": ("#fef2f2", "#b91c1c", "#fecaca"),
        "HIGH": ("#fff7ed", "#c2410c", "#fed7aa"),
        "MEDIUM": ("#fffbeb", "#b45309", "#fde68a"),
        "LOW": ("#f8fafc", "#475569", "#e2e8f0"),
    }
    bg, fg, border = colors.get(severity.upper(), colors["LOW"])
    return (
        f'<span style="display:inline-block;padding:0.15rem 0.5rem;border-radius:6px;'
        f'font-size:0.72rem;font-weight:700;letter-spacing:0.04em;'
        f'background:{bg};color:{fg};border:1px solid {border}">{severity}</span>'
    )


def llm_verdict_label(verdict: str) -> str:
    styles = {
        "PASS": ("#ecfdf5", "#047857", "#6ee7b7"),
        "NEEDS_WORK": ("#fffbeb", "#b45309", "#fde68a"),
        "FAIL": ("#fef2f2", "#b91c1c", "#fca5a5"),
        "REJECT": ("#fef2f2", "#b91c1c", "#fca5a5"),
        "SKIPPED": ("#f1f5f9", "#475569", "#cbd5e1"),
    }
    bg, fg, border = styles.get(verdict, ("#f8fafc", "#475569", "#e2e8f0"))
    return (
        f'<span style="display:inline-block;padding:0.2rem 0.55rem;border-radius:6px;'
        f'font-size:0.78rem;font-weight:700;background:{bg};color:{fg};'
        f'border:1px solid {border}">{verdict}</span>'
    )


def render_footer() -> None:
    st.markdown(
        f"""
<div class="t-footer">
  Cognyzer · Snorkel Terminal-Bench trainer tool · {PRODUCT_NAME} QC v{APP_VERSION}
  · Reports generated in-session · API keys admin-only
</div>
""",
        unsafe_allow_html=True,
    )


def kb(size_bytes: int) -> str:
    return f"{max(size_bytes / 1024, 0.1):.1f} KB"


def render_download_panel(
    html_data: str,
    json_data: str,
    html_filename: str,
    json_filename: str,
    *,
    title: str = "Download your reports",
    subtitle: str = "HTML for reviewers · JSON for detailed debugging.",
    key_prefix: str = "dl",
) -> None:
    html_size = kb(len(html_data.encode("utf-8")))
    json_size = kb(len(json_data.encode("utf-8")))
    st.markdown(
        f"""
<div class="t-download">
  <h3>{title}</h3>
  <p>{subtitle}</p>
</div>
""",
        unsafe_allow_html=True,
    )
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            f"Download HTML report ({html_size})",
            data=html_data,
            file_name=html_filename,
            mime="text/html",
            use_container_width=True,
            type="primary",
            key=f"{key_prefix}_html",
            help="Formatted report — open in any browser",
        )
        st.caption(f"`{html_filename}`")
    with d2:
        st.download_button(
            f"Download JSON data ({json_size})",
            data=json_data,
            file_name=json_filename,
            mime="application/json",
            use_container_width=True,
            type="primary",
            key=f"{key_prefix}_json",
            help="Full structured output — LLM gaps, scores, diagnostics",
        )
        st.caption(f"`{json_filename}`")
