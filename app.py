"""Terminus Trainer QC Portal — professional UI, admin secrets, LLM-first QC."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from alignment_prompts import ALIGNMENT_LABELS
from config import llm_configured, resolve_llm_model, resolve_openai_api_key, resolve_sheet_defaults
from qc_engine import (
    CHANGE_TASK_MESSAGE,
    assess_task,
    check_instruction_similarity,
    render_html_report,
    report_to_dict,
)
from tracker_defaults import INSTRUCTION_SIM_THRESHOLD

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = APP_DIR / "terminus_task_corpus.json"

st.set_page_config(
    page_title="Terminus QC Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
    .block-container { padding-top: 1.5rem; max-width: 1100px; }
    .hero {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white; padding: 2rem 2.5rem; border-radius: 16px;
        margin-bottom: 1.5rem;
    }
    .hero h1 { color: white; font-size: 2rem; margin: 0 0 0.5rem 0; }
    .hero p { color: #c8d6e5; margin: 0; font-size: 1.05rem; }
    .step-card {
        background: #f8f9fb; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 1.25rem; margin-bottom: 1rem;
    }
    .verdict-pass { color: #16a34a; font-weight: 700; font-size: 1.4rem; }
    .verdict-fail { color: #dc2626; font-weight: 700; font-size: 1.4rem; }
    .metric-box {
        background: white; border: 1px solid #e2e8f0; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .metric-box .label { color: #64748b; font-size: 0.85rem; text-transform: uppercase; }
    .metric-box .value { font-size: 1.5rem; font-weight: 700; color: #1e293b; }
    div[data-testid="stSidebar"] { display: none; }
</style>
""",
    unsafe_allow_html=True,
)

sheet_defaults = resolve_sheet_defaults()
llm_model = resolve_llm_model()
llm_ready = llm_configured()
sheet_preconfigured = bool(sheet_defaults.get("url"))

if "instruction_pre_passed" not in st.session_state:
    st.session_state.instruction_pre_passed = False
if "instruction_pre_text" not in st.session_state:
    st.session_state.instruction_pre_text = ""

st.markdown(
    """
<div class="hero">
  <h1>🛡️ Terminus Task QC Portal</h1>
  <p>Check <strong>instruction.md first</strong> → upload zip → LLM alignment judge (primary).
  Compared against <strong>8 accepted reference tasks</strong> + team tracker sheet.</p>
</div>
""",
    unsafe_allow_html=True,
)

trainer_name = st.text_input("Your name", placeholder="Optional — appears on the QC report")

with st.expander("Settings (optional)", expanded=False):
    run_llm = st.checkbox("Run LLM alignment judge (primary)", value=True, disabled=not llm_ready)
    if sheet_preconfigured:
        st.caption(
            f"Similarity source: **Terminus Task Tracker** · tab "
            f'`{sheet_defaults.get("worksheet", "")}` · col '
            f'**{sheet_defaults.get("instruction_col", "Task Instruction")}** (column 16)'
        )
        sheet_url = sheet_defaults.get("url", "")
        worksheet = sheet_defaults.get("worksheet", "")
        task_col = sheet_defaults.get("task_col", "")
        instruction_col = sheet_defaults.get("instruction_col", "")
        trainer_col = sheet_defaults.get("trainer_col", "")
        instruction_col_index = int(sheet_defaults.get("instruction_col_index", "16") or "16")
        spec_col = sheet_defaults.get("spec_col", "")
        use_local_corpus = False
    else:
        st.markdown("**Similarity sheet** — ask admin to configure in secrets")
        s1, s2 = st.columns(2)
        with s1:
            sheet_url = st.text_input("Google Sheet URL", value=sheet_defaults.get("url", ""))
            worksheet = st.text_input("Worksheet tab", value=sheet_defaults.get("worksheet", ""))
        with s2:
            task_col = st.text_input("Task Name column", value=sheet_defaults.get("task_col", ""))
            instruction_col = st.text_input(
                "Task Instruction column", value=sheet_defaults.get("instruction_col", "")
            )
            trainer_col = sheet_defaults.get("trainer_col", "")
            instruction_col_index = int(sheet_defaults.get("instruction_col_index", "16") or "16")
            spec_col = st.text_input("SPEC column", value=sheet_defaults.get("spec_col", ""))
        use_local_corpus = st.checkbox("Fallback to bundled corpus", value=True)

corpus_path = str(DEFAULT_CORPUS) if use_local_corpus and DEFAULT_CORPUS.exists() else ""

if llm_ready:
    st.caption(f"LLM judge: **primary layer** · model `{llm_model}` · key configured by admin")
else:
    st.warning(
        "LLM judge is **not configured** on this deployment. "
        "Static + similarity checks will still run. Contact your team lead."
    )

# ── Step 0: Instruction pre-check (before zip) ─────────────────────────────
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown("### Step 0 — Check instruction.md first")
st.caption(
    f"Lexical and embedding run **in parallel**. "
    f"If **both** are ≥ **{int(INSTRUCTION_SIM_THRESHOLD * 100)}%** on the same task → change the task."
)

inst_file = st.file_uploader(
    "Upload instruction.md (optional)",
    type=["md"],
    key="instruction_only",
)
inst_text_area = st.text_area(
    "Or paste your instruction text",
    value=st.session_state.instruction_pre_text,
    height=160,
    placeholder="Paste the full instruction.md content here…",
)

instruction_text = inst_text_area.strip()
if inst_file is not None:
    instruction_text = inst_file.getvalue().decode("utf-8", errors="replace").strip()

check_inst_btn = st.button("Check Instruction First", type="primary", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

if check_inst_btn:
    if not instruction_text:
        st.error("Paste or upload your instruction before checking.")
    else:
        with st.spinner("Running parallel lexical + embedding check against tracker sheet…"):
            pre_result = check_instruction_similarity(
                instruction_text=instruction_text,
                sheet_url=sheet_url,
                worksheet=worksheet,
                task_col=task_col,
                instruction_col=instruction_col,
                trainer_col=trainer_col,
                instruction_col_index=instruction_col_index,
                corpus_json_path=corpus_path if not sheet_url.strip() else "",
                api_key=resolve_openai_api_key(),
            )
        st.session_state.instruction_pre_text = instruction_text

        if pre_result.get("blocked"):
            st.session_state.instruction_pre_passed = False
            st.error(pre_result.get("message") or CHANGE_TASK_MESSAGE)
        else:
            st.session_state.instruction_pre_passed = True
            st.success(pre_result.get("message", "Instruction pre-check passed."))

        if pre_result.get("matches"):
            st.dataframe(
                [
                    {
                        "Task": m.task_id,
                        "Trainer": m.trainer or "—",
                        "Lexical %": round((m.lexical_score or 0) * 100, 1),
                        "Embedding %": (
                            round(m.semantic_score * 100, 1)
                            if m.semantic_score is not None else "—"
                        ),
                        "Change task?": "YES" if m.dual_block else "No",
                        "Method": m.method,
                    }
                    for m in pre_result["matches"]
                ],
                use_container_width=True,
                hide_index=True,
            )

if st.session_state.instruction_pre_passed:
    st.success("✓ Instruction pre-check passed — you can upload your task zip below.")
elif instruction_text and not check_inst_btn:
    st.info("Run **Check Instruction First** before uploading your zip.")

# ── Step 1: Zip upload ────────────────────────────────────────────────────
st.markdown('<div class="step-card">', unsafe_allow_html=True)
st.markdown("### Step 1 — Upload your task zip")
uploaded = st.file_uploader(
    "Drop your `.zip` here",
    type=["zip"],
    label_visibility="collapsed",
    disabled=not st.session_state.instruction_pre_passed,
)
st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.instruction_pre_passed:
    st.stop()

if uploaded is None:
    st.info("Upload a task zip to begin full assessment.")
    st.stop()

fcol1, fcol2, fcol3 = st.columns(3)
fcol1.markdown(
    f'<div class="metric-box"><div class="label">File</div><div class="value" style="font-size:1rem">'
    f"{uploaded.name}</div></div>",
    unsafe_allow_html=True,
)
fcol2.markdown(
    f'<div class="metric-box"><div class="label">Size</div>'
    f'<div class="value">{uploaded.size / 1024:.0f} KB</div></div>',
    unsafe_allow_html=True,
)
fcol3.markdown(
    f'<div class="metric-box"><div class="label">References</div>'
    f'<div class="value">8 accepted</div></div>',
    unsafe_allow_html=True,
)

run_qc = st.button("Run Full QC Assessment", type="primary", use_container_width=True)

if not run_qc:
    st.stop()

progress = st.progress(0, text="Starting assessment…")
status = st.empty()


def llm_progress(current: int, total: int, label: str, model: str) -> None:
    pct = int((current - 1) / total * 80) + 15
    progress.progress(min(pct, 95), text=f"LLM check {current}/{total}: {label} ({model})…")


progress.progress(3, text="Checking instruction.md from zip (parallel lexical + embedding)…")

with status.container():
    with st.spinner("Running assessment — instruction first, then static, then LLM judge…"):
        with tempfile.TemporaryDirectory(prefix="terminus_qc_") as tmp:
            report, _ = assess_task(
                zip_bytes=uploaded.getvalue(),
                zip_name=uploaded.name,
                trainer_name=trainer_name,
                sheet_url=sheet_url,
                worksheet=worksheet,
                task_col=task_col,
                instruction_col=instruction_col,
                spec_col=spec_col,
                trainer_col=trainer_col,
                instruction_col_index=instruction_col_index,
                corpus_json_path=corpus_path if not sheet_url.strip() else "",
                openai_api_key=resolve_openai_api_key(),
                run_llm=run_llm and llm_ready,
                work_dir=Path(tmp),
                on_llm_progress=llm_progress,
            )

progress.progress(100, text="Done!")
status.empty()

report_json = json.dumps(report_to_dict(report), indent=2)
report_html = render_html_report(report)
safe_name = report.task_name.replace(" ", "-")

st.markdown("---")
st.markdown("### Step 2 — Results")

if report.instruction_blocked:
    st.error(report.instruction_block_message or CHANGE_TASK_MESSAGE)

verdict_class = "verdict-pass" if report.overall_pass else "verdict-fail"
verdict_text = "READY TO SUBMIT" if report.overall_pass else "NEEDS FIXES"
verdict_icon = "✅" if report.overall_pass else "❌"
st.markdown(
    f'<p class="{verdict_class}">{verdict_icon} {verdict_text}</p>',
    unsafe_allow_html=True,
)
if report.llm_results and not report.overall_pass and report.llm_pass is False:
    st.caption("LLM alignment judge is the **primary** gate — fix alignment issues first.")

m1, m2, m3, m4 = st.columns(4)
m1.markdown(
    f'<div class="metric-box"><div class="label">Task</div>'
    f'<div class="value" style="font-size:1.1rem">{report.task_name}</div></div>',
    unsafe_allow_html=True,
)
m2.markdown(
    f'<div class="metric-box"><div class="label">Instruction</div>'
    f'<div class="value">{"BLOCK" if report.instruction_blocked else "OK"}</div></div>',
    unsafe_allow_html=True,
)
m3.markdown(
    f'<div class="metric-box"><div class="label">LLM Align</div>'
    f'<div class="value">'
    f'{"SKIPPED" if not report.llm_results else ("PASS" if report.llm_pass else "FAIL")}'
    f'</div></div>',
    unsafe_allow_html=True,
)
m4.markdown(
    f'<div class="metric-box"><div class="label">Static</div>'
    f'<div class="value">{"PASS" if report.static_pass else "FAIL"}</div></div>',
    unsafe_allow_html=True,
)

tab_llm, tab_similarity, tab_static, tab_download = st.tabs(
    ["LLM Alignment (primary)", "Instruction Similarity", "Static Checks", "Download Report"]
)

with tab_llm:
    if not report.llm_results:
        st.info("LLM judge was not run. Ask your admin to configure the API key.")
    else:
        for key, item in report.llm_results.items():
            if not isinstance(item, dict):
                continue
            verdict = item.get("verdict", "UNKNOWN")
            icon = "✅" if verdict == "PASS" else "⚠️" if verdict == "NEEDS_WORK" else "❌"
            label = item.get("label", ALIGNMENT_LABELS.get(key, key))
            score = item.get("alignment_score", "—")

            with st.container(border=True):
                st.markdown(f"#### {icon} {label}")
                st.markdown(f"**{verdict}** · score {score}/100")
                st.markdown(item.get("reasoning", ""))

                ref_comps = item.get("reference_comparisons", [])
                if isinstance(ref_comps, list) and ref_comps:
                    st.markdown("**Per-reference comparison (all 8 accepted tasks):**")
                    st.dataframe(
                        [
                            {
                                "Reference": r.get("reference_task", ""),
                                "Structure": r.get("structural_alignment", "—"),
                                "Differentiation": r.get("differentiation_score", "—"),
                                "Too similar": r.get("too_similar", False),
                                "Issues": "; ".join(r.get("issues", [])[:2]),
                            }
                            for r in ref_comps
                            if isinstance(r, dict)
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

                gaps = item.get("gaps", [])
                if isinstance(gaps, list) and gaps:
                    gap_data = []
                    for gap in gaps:
                        if isinstance(gap, dict):
                            gap_data.append({
                                "File": gap.get("file", "—"),
                                "Issue": gap.get("issue", ""),
                                "Fix": gap.get("fix", ""),
                            })
                        elif isinstance(gap, str) and gap:
                            gap_data.append({"File": "—", "Issue": gap, "Fix": ""})
                    if gap_data:
                        st.dataframe(gap_data, use_container_width=True, hide_index=True)

                flags = [
                    (k, item[k])
                    for k in (
                        "hardcoded_solution", "instruction_leaks_rules", "unused_hash_checks",
                        "bypass_possible", "copy_paste_risk", "too_similar_to_existing",
                    )
                    if item.get(k) is True
                ]
                if flags:
                    st.warning("Flags: " + ", ".join(k for k, _ in flags))

with tab_similarity:
    st.caption(
        f"Check #1: pre-upload instruction · Check #2: instruction.md from zip. "
        f"Parallel lexical + embedding — **change task** only when **both** ≥ "
        f"**{int(INSTRUCTION_SIM_THRESHOLD * 100)}%**."
    )
    if report.instruction_matches:
        st.markdown("**instruction.md vs team tracker sheet**")
        st.dataframe(
            [
                {
                    "Task": m.task_id,
                    "Trainer": m.trainer or "—",
                    "Lexical %": round((m.lexical_score or 0) * 100, 1),
                    "Embedding %": (
                        round(m.semantic_score * 100, 1)
                        if m.semantic_score is not None else "—"
                    ),
                    "Change task?": "YES" if m.dual_block else "No",
                    "Method": m.method,
                }
                for m in report.instruction_matches
            ],
            use_container_width=True,
            hide_index=True,
        )
    if report.spec_matches:
        st.markdown("**SPEC.md matches (secondary)**")
        st.dataframe(
            [
                {"Task": m.task_id, "Score %": round(m.score * 100, 1)}
                for m in report.spec_matches
            ],
            use_container_width=True,
            hide_index=True,
        )
    if not report.instruction_matches and not report.spec_matches:
        st.info("No similarity results — configure a Google Sheet or enable local corpus.")

with tab_static:
    if not report.static_issues:
        st.success("All static checks passed.")
    else:
        for issue in report.static_issues:
            sev = issue.severity
            color = "#dc2626" if sev == "CRITICAL" else "#ea580c" if sev == "HIGH" else "#ca8a04"
            st.markdown(
                f'<span style="color:{color};font-weight:600">{sev}</span> — {issue.message}',
                unsafe_allow_html=True,
            )
            if issue.fix_hint:
                st.caption(f"Fix: {issue.fix_hint}")

with tab_download:
    st.markdown("Download your QC report — share the HTML with reviewers.")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            f"📄 {safe_name}_qc_report.html",
            data=report_html,
            file_name=f"{safe_name}_qc_report.html",
            mime="text/html",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            f"📋 {safe_name}_qc_report.json",
            data=report_json,
            file_name=f"{safe_name}_qc_report.json",
            mime="application/json",
            use_container_width=True,
        )

if report.notes:
    with st.expander("Technical notes"):
        for note in report.notes:
            st.write(f"- {note}")
