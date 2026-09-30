"""Terminus QC Portal — shared UI styles and HTML components."""

from __future__ import annotations

import base64
import html as html_module
from pathlib import Path
from typing import Any

import streamlit as st

APP_DIR = Path(__file__).resolve().parent
FAVICON_PATH = APP_DIR / "favicon.png"
if not FAVICON_PATH.is_file():
    FAVICON_PATH = APP_DIR / "favicon.ico"
APP_VERSION = "3.5"
PRODUCT_NAME = "Cognyzer"
PRODUCT_TAGLINE = "Terminal Bench · check instructions before you submit on Tela"


def inject_global_css() -> None:
    st.markdown(
        """
<style>
    :root {
        --t-bg: #000000;
        --t-bg-elevated: #18181b;
        --t-surface: #18181b;
        --t-border: #27272a;
        --t-text: #fafafa;
        --t-muted: #a1a1aa;
        --t-accent: #ff5c33;
        --t-brand-grad: linear-gradient(90deg, #ffb800 0%, #ff5c33 40%, #d6249f 75%, #8224e3 100%);
        --t-pass: #34d399;
        --t-fail: #f87171;
        --t-warn: #fbbf24;
        --t-shadow: 0 4px 24px rgba(0, 0, 0, 0.35);
    }
    .stApp,
    [data-testid="stAppViewContainer"],
    .main .block-container {
        background-color: var(--t-bg) !important;
        color: var(--t-text);
    }
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stCaption {
        color: var(--t-text) !important;
    }
    .block-container span:not([data-testid="stMarkdownContainer"] *) {
        color: inherit;
    }
    .stCaption, small, [data-testid="stMarkdownContainer"] p {
        color: var(--t-muted) !important;
    }
    textarea, input, select {
        background-color: var(--t-bg-elevated) !important;
        color: var(--t-text) !important;
        border-color: var(--t-border) !important;
    }
    div[data-baseweb="select"] > div {
        background-color: var(--t-bg-elevated) !important;
    }
    .stButton > button[kind="secondary"],
    div[data-testid="stButton"] > button:not([kind="primary"]) {
        background: var(--t-surface) !important;
        color: var(--t-text) !important;
        border: 1px solid var(--t-border) !important;
    }
    [data-testid="stExpander"] details {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 12px;
    }
    [data-testid="stAlert"] {
        background-color: var(--t-surface) !important;
    }
    /* Streamlit primary actions — solid Cognyzer orange (overrides theme teal below) */
    .stButton > button[kind="primary"],
    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="stBaseButton-primary"] {
        background: #ff5c33 !important;
        background-color: #ff5c33 !important;
        border: 1px solid #e54e2b !important;
        color: #fff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
    .stButton > button[kind="primary"]:hover,
    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover {
        background: #ff704d !important;
        background-color: #ff704d !important;
        border-color: #ff5c33 !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"] *,
    div[data-testid="stButton"] > button[kind="primary"] *,
    button[data-testid="stBaseButton-primary"],
    button[data-testid="stBaseButton-primary"] * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        fill: #ffffff !important;
    }
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stDownloadButton"] > button * {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
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
        background: var(--t-brand-grad);
        color: #fff; font-weight: 700; font-size: 0.72rem;
        display: flex; align-items: center; justify-content: center;
        letter-spacing: -0.02em;
        box-shadow: var(--t-shadow);
    }
    .t-logo-img {
        width: 44px; height: 44px; border-radius: 12px;
        object-fit: contain; display: block;
        box-shadow: var(--t-shadow);
    }
    .t-title { margin: 0; font-size: 1.35rem; font-weight: 700; color: #fafafa; line-height: 1.2; }
    .t-subtitle { margin: 0.15rem 0 0 0; color: #a1a1aa; font-size: 0.92rem; }
    .t-badge-row { display: flex; flex-wrap: wrap; gap: 0.45rem; justify-content: flex-end; }
    .t-badge {
        display: inline-flex; align-items: center; gap: 0.35rem;
        padding: 0.3rem 0.65rem; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600; letter-spacing: 0.02em;
        border: 1px solid #3f3f46; background: rgba(255,255,255,0.06); color: #e4e4e7;
    }
    .t-badge.ok { background: rgba(255, 92, 51, 0.15); border-color: rgba(255, 92, 51, 0.45); color: #fdba74; }
    .t-badge.warn { background: rgba(251, 191, 36, 0.12); border-color: rgba(251, 191, 36, 0.35); color: #fde68a; }

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
    .t-step.active { border-color: #ff5c33; background: rgba(255, 92, 51, 0.08); }
    .t-step.done { border-color: #34d399; background: rgba(52, 211, 153, 0.08); }
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
        background: var(--t-bg-elevated); color: var(--t-muted);
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
    .t-verdict.pass { background: rgba(52, 211, 153, 0.1); border-color: #34d399; }
    .t-verdict.fail { background: rgba(248, 113, 113, 0.1); border-color: #f87171; }
    .t-verdict .title { margin: 0; font-size: 1.15rem; font-weight: 700; }
    .t-verdict.pass .title { color: var(--t-pass); }
    .t-verdict.fail .title { color: var(--t-fail); }
    .t-verdict .hint { margin: 0.2rem 0 0 0; color: var(--t-muted); font-size: 0.88rem; }

    .t-download {
        background: var(--t-surface);
        border: 1px solid var(--t-border);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin: 1rem 0 1.25rem 0;
        box-shadow: var(--t-shadow);
    }
    .t-download h3 { margin: 0 0 0.3rem 0; color: var(--t-text); font-size: 1rem; font-weight: 700; }
    .t-download p { margin: 0; color: var(--t-muted); font-size: 0.88rem; }

    .t-instr-scroll {
        max-height: min(72vh, 720px);
        overflow: auto;
        background: var(--t-bg-elevated);
        border: 1px solid var(--t-border);
        border-radius: 10px;
        padding: 14px 16px;
        margin: 0.35rem 0 0.75rem 0;
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        font-size: 0.82rem;
        line-height: 1.55;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--t-text);
    }
    .t-instr-meta { color: var(--t-muted); font-size: 0.8rem; margin: 0 0 0.25rem 0; }

    .t-footer {
        margin-top: 2rem; padding-top: 1rem;
        border-top: 1px solid var(--t-border);
        color: var(--t-muted); font-size: 0.8rem; text-align: center;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 2.6rem; border-radius: 10px 10px 0 0;
        padding-left: 1rem; padding-right: 1rem;
        font-weight: 600; font-size: 0.88rem;
        color: var(--t-muted) !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: var(--t-text) !important;
        border-bottom-color: #ff5c33 !important;
    }
    div[data-testid="stFileUploader"] section {
        border: 1px dashed var(--t-border);
        border-radius: 12px;
        background: var(--t-bg-elevated);
    }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--t-border) !important;
        border-radius: 16px !important;
        box-shadow: var(--t-shadow);
        padding: 0.35rem 0.15rem 0.15rem;
        margin-bottom: 0.75rem;
        background: var(--t-surface) !important;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--t-border);
        border-radius: 12px;
        overflow: hidden;
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
    """Optional favicon hook — disabled on Streamlit Cloud (iframe sandbox errors)."""
    return


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
        else '<div class="t-logo">CZ</div>'
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
    <span class="t-badge">tela.cognyzer.com</span>
    {llm_badge}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    """Intentionally minimal — intro lives in the dark top bar."""
    return


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
  <a href="https://www.cognyzer.com" style="color:#a1a1aa;text-decoration:none;">Cognyzer</a>
  · Trainer QC v{APP_VERSION} · Reports stay in your browser session
</div>
""",
        unsafe_allow_html=True,
    )


def kb(size_bytes: int) -> str:
    return f"{max(size_bytes / 1024, 0.1):.1f} KB"


def _similarity_flag_label(match: Any) -> str:
    dual = getattr(match, "dual_block", False)
    reason = getattr(match, "block_reason", "") or ""
    if not dual:
        return "No"
    if reason == "meaning":
        return "YES · meaning ≥85%"
    if reason == "cosine" or reason == "words":
        return "YES · cosine ≥85%"
    return "YES"


def _render_full_instruction(text: str) -> None:
    safe = html_module.escape(text or "")
    if not safe.strip():
        st.warning("No instruction text available — re-run the check to reload from Tela.")
        return
    st.markdown(f'<div class="t-instr-scroll">{safe}</div>', unsafe_allow_html=True)


def _match_task_id(match: Any) -> str:
    if isinstance(match, dict):
        return str(match.get("task_id") or match.get("task") or "").strip()
    return str(getattr(match, "task_id", "") or "").strip()


def _match_matched_instruction(match: Any) -> str:
    if isinstance(match, dict):
        return str(match.get("matched_instruction") or "").strip()
    return str(getattr(match, "matched_instruction", "") or "").strip()


def _resolve_tracker_instruction(
    match: Any,
    tracker_instructions: dict[str, str] | None = None,
    corpus_instructions: dict[str, str] | None = None,
) -> str:
    direct = _match_matched_instruction(match)
    if direct:
        return direct
    task_id = _match_task_id(match)
    if tracker_instructions and task_id:
        text = (tracker_instructions.get(task_id) or "").strip()
        if text:
            return text
    if corpus_instructions and task_id:
        return (corpus_instructions.get(task_id) or "").strip()
    return ""


def render_similarity_match_table(matches: list[Any]) -> None:
    if not matches:
        return
    st.dataframe(
        [
            {
                "Task": m.task_id,
                "Trainer": m.trainer or "—",
                "Cosine similarity %": round((m.lexical_score or 0) * 100, 1),
                "Embedding cosine %": (
                    round(m.semantic_score * 100, 1)
                    if m.semantic_score is not None else "—"
                ),
                "Flagged?": _similarity_flag_label(m),
            }
            for m in matches
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_similarity_instruction_reviews(
    your_instruction: str,
    matches: list[Any],
    *,
    key_prefix: str = "sim",
    max_reviews: int = 10,
    tracker_instructions: dict[str, str] | None = None,
    corpus_instructions: dict[str, str] | None = None,
) -> None:
    if not matches or not (your_instruction or "").strip():
        return

    st.markdown("**👁 Compare full instructions**")
    st.caption(
        "Select a Tela match below — both prompts show in full (scroll inside each panel)."
    )

    options = list(range(min(len(matches), max_reviews)))
    labels = []
    for i in options:
        m = matches[i]
        sem = (
            round(m.semantic_score * 100, 1)
            if m.semantic_score is not None else None
        )
        sem_s = f"{sem}%" if sem is not None else "—"
        lex = round((m.lexical_score or 0) * 100, 1)
        labels.append(f"👁 {m.task_id} · embed {sem_s} · cosine {lex}% · {_similarity_flag_label(m)}")

    pick = st.selectbox(
        "Tracker match to compare",
        options,
        format_func=lambda i: labels[i],
        key=f"{key_prefix}_pick",
    )
    m = matches[pick]
    tracker_text = _resolve_tracker_instruction(
        m,
        tracker_instructions,
        corpus_instructions,
    )

    left, right = st.columns(2)
    with left:
        st.markdown(
            f'<p class="t-instr-meta"><strong>Your instruction</strong> · '
            f"{len(your_instruction):,} characters</p>",
            unsafe_allow_html=True,
        )
        _render_full_instruction(your_instruction)
    with right:
        st.markdown(
            f'<p class="t-instr-meta"><strong>Tracker: {html_module.escape(m.task_id)}</strong>'
            f" · {html_module.escape(m.trainer or 'unknown trainer')}"
            f" · {len(tracker_text):,} characters</p>",
            unsafe_allow_html=True,
        )
        _render_full_instruction(tracker_text)

    if len(matches) > 1:
        with st.expander(f"Other matches ({len(matches) - 1} more)", expanded=False):
            for i, other in enumerate(matches[:max_reviews]):
                if i == pick:
                    continue
                sem = (
                    round(other.semantic_score * 100, 1)
                    if other.semantic_score is not None else "—"
                )
                st.markdown(
                    f"- **{other.task_id}** · meaning {sem}% · "
                    f"cosine {round((other.lexical_score or 0) * 100, 1)}% · "
                    f"{_similarity_flag_label(other)}"
                )


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
