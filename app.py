"""Terminus Trainer QC Portal — professional UI, admin secrets, LLM-first QC."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from config import (
    api_provider_label,
    llm_configured,
    resolve_llm_model,
    resolve_openai_api_key,
    resolve_sheet_defaults,
)
from tracker_defaults import INSTRUCTION_SIM_THRESHOLD
from ui_components import (
    inject_global_css,
    llm_verdict_label,
    render_download_panel,
    render_panel_header,
    render_footer,
    render_hero,
    render_metric_grid,
    render_topbar,
    render_verdict_banner,
    render_section_header,
    render_workflow_stepper,
    severity_badge,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = APP_DIR / "terminus_task_corpus.json"
SIM_PCT = int(INSTRUCTION_SIM_THRESHOLD * 100)
INSTRUCTION_CHECK_HELP = (
    f"Compare your instruction against the team tracker. You'll be blocked only if "
    f"it's too similar to an existing task on both word overlap and meaning "
    f"(both ≥ {SIM_PCT}%)."
)
SIMILARITY_TAB_HELP = (
    f"Checked before upload and again from your zip. Change task only when word overlap "
    f"and meaning are both ≥ {SIM_PCT}%."
)
MAX_INSTRUCTION_MD_MB = 5
MAX_ZIP_MB = 200
MAX_INSTRUCTION_MD_BYTES = MAX_INSTRUCTION_MD_MB * 1024 * 1024
MAX_ZIP_BYTES = MAX_ZIP_MB * 1024 * 1024


def _format_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def _text_size_bytes(text: str) -> int:
    return len(text.encode("utf-8"))


@st.cache_resource(show_spinner=False)
def _qc_engine():
    import qc_engine

    return qc_engine


st.set_page_config(
    page_title="Terminus QC · Task Checker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_global_css()

sheet_defaults = resolve_sheet_defaults()
llm_model = resolve_llm_model()
llm_ready = llm_configured()
sheet_preconfigured = bool(sheet_defaults.get("url"))

if "instruction_pre_passed" not in st.session_state:
    st.session_state.instruction_pre_passed = False
if "instruction_pre_text" not in st.session_state:
    st.session_state.instruction_pre_text = ""
if "instruction_pre_result" not in st.session_state:
    st.session_state.instruction_pre_result = None
if "qc_cache" not in st.session_state:
    st.session_state.qc_cache = None
if "show_pre_downloads" not in st.session_state:
    st.session_state.show_pre_downloads = False

provider = api_provider_label() if llm_ready else "—"
render_topbar(llm_ready, provider, llm_model)
render_hero()
render_workflow_stepper(0 if not st.session_state.instruction_pre_passed else 1)

meta_left, meta_right = st.columns([2, 1])
with meta_left:
    trainer_name = st.text_input(
        "Trainer name",
        placeholder="Optional — included on exported reports",
        label_visibility="visible",
    )
with meta_right:
    if llm_ready:
        st.markdown(
            '<p style="margin:1.6rem 0 0 0;color:#64748b;font-size:0.85rem;">'
            f"LLM judge enabled · <code>{llm_model}</code></p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("LLM judge unavailable — configure API key in deployment secrets.")

with st.expander("Assessment settings", expanded=False):
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

# ── Step 0: Instruction pre-check (before zip) ─────────────────────────────
with st.container(border=True):
    render_panel_header(
        0,
        "Instruction similarity check",
        INSTRUCTION_CHECK_HELP,
    )

    inst_file = st.file_uploader(
        f"Upload instruction.md (optional, max {MAX_INSTRUCTION_MD_MB} MB)",
        type=["md"],
        key="instruction_only",
    )
    st.caption(f"Markdown only · maximum {MAX_INSTRUCTION_MD_MB} MB per file")
    inst_text_area = st.text_area(
        "Or paste your instruction text",
        value=st.session_state.instruction_pre_text,
        height=160,
        placeholder="Paste the full instruction.md content here…",
    )

    instruction_text = inst_text_area.strip()
    instruction_file_too_large = False
    if inst_file is not None:
        if inst_file.size > MAX_INSTRUCTION_MD_BYTES:
            instruction_file_too_large = True
            st.error(
                f"instruction.md is too large ({_format_size(inst_file.size)}). "
                f"Maximum is {MAX_INSTRUCTION_MD_MB} MB."
            )
        else:
            instruction_text = inst_file.getvalue().decode("utf-8", errors="replace").strip()

    check_inst_btn = st.button("Run instruction check", type="primary", use_container_width=True)

if check_inst_btn:
    if instruction_file_too_large:
        pass
    elif not instruction_text:
        st.error("Paste or upload your instruction before checking.")
    elif _text_size_bytes(instruction_text) > MAX_INSTRUCTION_MD_BYTES:
        st.error(
            f"Instruction text is too large ({_format_size(_text_size_bytes(instruction_text))}). "
            f"Maximum is {MAX_INSTRUCTION_MD_MB} MB."
        )
    else:
        qe = _qc_engine()
        with st.spinner("Comparing your instruction against the team tracker…"):
            pre_result = qe.check_instruction_similarity(
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
        st.session_state.instruction_pre_result = pre_result
        st.session_state.show_pre_downloads = False

        if pre_result.get("embedding_ran"):
            st.info(
                f"Meaning check completed ({pre_result.get('api_provider', 'OpenAI')}) "
                f"against **{pre_result.get('corpus_size', 0)}** tracker instructions."
            )
        elif pre_result.get("embedding_error"):
            st.warning(f"Meaning check did not run: {pre_result['embedding_error']}")
        elif not pre_result.get("api_key_present") and not llm_ready:
            st.warning(
                "Meaning check did not run — add `OPENAI_API_KEY` in Streamlit Cloud secrets. "
                "Only word-overlap was checked."
            )
        elif not pre_result.get("api_key_present"):
            st.warning("API key missing for meaning check (full zip QC may still work).")
        else:
            st.warning(
                "Meaning check did not complete — see **Sheet load details** and download the report below."
            )

        corpus_count = pre_result.get("corpus_count", 0) or pre_result.get("corpus_size", 0)
        if corpus_count:
            st.caption(f"Compared against **{corpus_count}** reference instructions.")

        if pre_result.get("notes"):
            with st.expander("Sheet load details", expanded=corpus_count == 0):
                for note in pre_result["notes"]:
                    st.write(f"- {note}")

        if pre_result.get("blocked"):
            st.session_state.instruction_pre_passed = False
            st.error(pre_result.get("message") or qe.CHANGE_TASK_MESSAGE)
        elif corpus_count == 0:
            st.session_state.instruction_pre_passed = False
            st.error(pre_result.get("message", "No reference corpus loaded."))
        else:
            st.session_state.instruction_pre_passed = True
            st.success(pre_result.get("message", "Instruction pre-check passed."))

        if pre_result.get("matches"):
            st.dataframe(
                [
                    {
                        "Task": m.task_id,
                        "Trainer": m.trainer or "—",
                        "Word overlap %": round((m.lexical_score or 0) * 100, 1),
                        "Meaning %": (
                            round(m.semantic_score * 100, 1)
                            if m.semantic_score is not None else "—"
                        ),
                        "Too similar?": "YES" if m.dual_block else "No",
                    }
                    for m in pre_result["matches"]
                ],
                use_container_width=True,
                hide_index=True,
            )

        pre_html = qe.render_instruction_precheck_html(pre_result, instruction_text, trainer_name)
        pre_json = json.dumps(
            qe.instruction_precheck_to_dict(pre_result, instruction_text, trainer_name),
            indent=2,
        )
        render_download_panel(
            pre_html,
            pre_json,
            "instruction_precheck_report.html",
            "instruction_precheck_report.json",
            title="Instruction pre-check report",
            subtitle="Top matches, meaning-check status, and tracker load details.",
            key_prefix="dl_pre",
        )

elif st.session_state.instruction_pre_result:
    pre_result = st.session_state.instruction_pre_result
    blocked = pre_result.get("blocked")
    status = "blocked" if blocked else "passed"
    st.caption(f"Last instruction check: **{status}** — run again to refresh or download reports.")
    if pre_result.get("matches"):
        with st.expander("Previous match table", expanded=False):
            st.dataframe(
                [
                    {
                        "Task": m.task_id,
                        "Trainer": m.trainer or "—",
                        "Word overlap %": round((m.lexical_score or 0) * 100, 1),
                        "Meaning %": (
                            round(m.semantic_score * 100, 1)
                            if m.semantic_score is not None else "—"
                        ),
                        "Too similar?": "YES" if m.dual_block else "No",
                    }
                    for m in pre_result["matches"]
                ],
                use_container_width=True,
                hide_index=True,
            )
    if st.button("Prepare instruction report downloads", key="prep_pre_dl"):
        st.session_state.show_pre_downloads = True
    if st.session_state.get("show_pre_downloads"):
        qe = _qc_engine()
        pre_html = qe.render_instruction_precheck_html(
            pre_result, st.session_state.instruction_pre_text, trainer_name
        )
        pre_json = json.dumps(
            qe.instruction_precheck_to_dict(
                pre_result, st.session_state.instruction_pre_text, trainer_name
            ),
            indent=2,
        )
        render_download_panel(
            pre_html,
            pre_json,
            "instruction_precheck_report.html",
            "instruction_precheck_report.json",
            title="Instruction pre-check report",
            subtitle="From your most recent instruction check.",
            key_prefix="dl_pre_persist",
        )

if st.session_state.instruction_pre_passed:
    st.success("Instruction check passed — proceed to task zip upload.")
elif instruction_text and not check_inst_btn:
    st.info("Run the instruction check before uploading your task zip.")

# ── Step 1: Zip upload ────────────────────────────────────────────────────
render_workflow_stepper(1)
with st.container(border=True):
    render_panel_header(
        1,
        "Upload task archive",
        f"Submit your Terminal-Bench task as a .zip after the instruction check passes "
        f"(maximum {MAX_ZIP_MB} MB).",
    )
    uploaded = st.file_uploader(
        f"Task zip file (max {MAX_ZIP_MB} MB)",
        type=["zip"],
        disabled=not st.session_state.instruction_pre_passed,
    )

if not st.session_state.instruction_pre_passed:
    st.stop()

if uploaded is None:
    st.info("Upload a task zip to begin full assessment.")
    st.stop()

if uploaded.size > MAX_ZIP_BYTES:
    st.error(
        f"Zip file is too large ({_format_size(uploaded.size)}). "
        f"Maximum is {MAX_ZIP_MB} MB."
    )
    st.stop()

upload_sig = f"{uploaded.name}:{uploaded.size}"
if (
    st.session_state.qc_cache
    and st.session_state.qc_cache.get("upload_sig") != upload_sig
):
    st.session_state.qc_cache = None

with st.container(border=True):
    render_panel_header(
        1,
        "Run assessment",
        "Execute static checks, tracker similarity, and LLM alignment against 8 accepted references.",
    )
    render_metric_grid([
        ("Archive", uploaded.name, "neutral"),
        ("Size", f"{uploaded.size / 1024:.0f} KB", "neutral"),
        ("References", "8 accepted", "neutral"),
    ])
    run_qc = st.button("Run full QC assessment", type="primary", use_container_width=True)

if run_qc:
    qe = _qc_engine()
    progress = st.progress(0, text="Starting assessment…")
    status = st.empty()

    def llm_progress(current: int, total: int, label: str, model: str) -> None:
        pct = int((current - 1) / total * 80) + 15
        progress.progress(
            min(pct, 95),
            text=f"LLM alignment {current}/{total} done — {label} ({model})…",
        )

    progress.progress(3, text="Checking instruction.md from your zip against the team tracker…")

    with status.container():
        with st.spinner(
            "Running assessment — instruction check, static checks, "
            "then LLM alignment (9 checks in parallel)…"
        ):
            with tempfile.TemporaryDirectory(prefix="terminus_qc_") as tmp:
                report, _ = qe.assess_task(
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

    safe_name = report.task_name.replace(" ", "-")
    st.session_state.qc_cache = {
        "upload_sig": upload_sig,
        "report": report,
        "safe_name": safe_name,
    }

cache = st.session_state.qc_cache
if not run_qc:
    if cache and cache.get("upload_sig") == upload_sig:
        report = cache["report"]
        safe_name = cache["safe_name"]
    else:
        st.stop()
else:
    report = st.session_state.qc_cache["report"]
    safe_name = st.session_state.qc_cache["safe_name"]

qe = _qc_engine()
report_json = json.dumps(qe.report_to_dict(report), indent=2)
report_html = qe.render_html_report(report)

render_workflow_stepper(2)
render_section_header(
    2,
    "Assessment results",
    "Summary of instruction similarity, static checks, and LLM alignment against accepted references.",
)

if report.instruction_blocked:
    st.error(report.instruction_block_message or qe.CHANGE_TASK_MESSAGE)

hint = ""
if report.llm_results and not report.overall_pass and report.llm_pass is False:
    hint = "LLM alignment is the primary gate — address alignment gaps before resubmitting."
render_verdict_banner(report.overall_pass, hint)

inst_tone = "fail" if report.instruction_blocked else "pass"
llm_val = "SKIPPED" if not report.llm_results else ("PASS" if report.llm_pass else "FAIL")
llm_tone = "neutral" if not report.llm_results else ("pass" if report.llm_pass else "fail")
static_tone = "pass" if report.static_pass else "fail"

render_metric_grid([
    ("Task", report.task_name, "neutral"),
    ("Instruction", "BLOCK" if report.instruction_blocked else "OK", inst_tone),
    ("LLM alignment", llm_val, llm_tone),
    ("Static checks", "PASS" if report.static_pass else "FAIL", static_tone),
])

render_download_panel(
    report_html,
    report_json,
    f"{safe_name}_qc_report.html",
    f"{safe_name}_qc_report.json",
    title="Full QC report",
    subtitle=(
        "HTML for human review · JSON with LLM gaps, similarity scores, and static diagnostics."
    ),
    key_prefix="dl_qc_main",
)

tab_llm, tab_similarity, tab_static, tab_details = st.tabs(
    ["LLM Alignment (primary)", "Instruction Similarity", "Static Checks", "Report summary"]
)

with tab_llm:
    if not report.llm_results:
        st.info("LLM judge was not run. Ask your admin to configure the API key.")
    else:
        from alignment_prompts import ALIGNMENT_LABELS

        for key, item in report.llm_results.items():
            if not isinstance(item, dict):
                continue
            verdict = item.get("verdict", "UNKNOWN")
            label = item.get("label", ALIGNMENT_LABELS.get(key, key))
            score = item.get("alignment_score", "—")

            with st.container(border=True):
                st.markdown(f"#### {label}")
                st.markdown(
                    f"{llm_verdict_label(verdict)} &nbsp; "
                    f'<span style="color:#64748b;font-size:0.9rem;">Score {score}/100</span>',
                    unsafe_allow_html=True,
                )
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
    st.caption(SIMILARITY_TAB_HELP)
    if report.instruction_matches:
        st.markdown("**instruction.md vs team tracker sheet**")
        st.dataframe(
            [
                {
                    "Task": m.task_id,
                    "Trainer": m.trainer or "—",
                    "Word overlap %": round((m.lexical_score or 0) * 100, 1),
                    "Meaning %": (
                        round(m.semantic_score * 100, 1)
                        if m.semantic_score is not None else "—"
                    ),
                    "Too similar?": "YES" if m.dual_block else "No",
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
            st.markdown(
                f"{severity_badge(issue.severity)} &nbsp; {issue.message}",
                unsafe_allow_html=True,
            )
            if issue.fix_hint:
                st.caption(f"Fix: {issue.fix_hint}")

with tab_details:
    st.markdown("#### Report at a glance")
    st.caption(
        "Accepted reference tasks may still fail LLM alignment or show high tracker similarity "
        "(e.g. matching their own row). Full downloads are in the panel above the tabs."
    )
    summary = qe.report_to_dict(report)
    st.json({
        "overall_pass": summary["overall_pass"],
        "task_name": summary["task_name"],
        "static_pass": summary["static_checks"]["pass"],
        "similarity_pass": summary["similarity"]["pass"],
        "llm_pass": summary["llm_judge"]["pass"],
        "issue_count": len(summary["static_checks"]["issues"]),
    })
    render_download_panel(
        report_html,
        report_json,
        f"{safe_name}_qc_report.html",
        f"{safe_name}_qc_report.json",
        title="Download again",
        subtitle="Same report files as the panel above.",
        key_prefix="dl_qc_tab",
    )

if report.notes:
    with st.expander("Technical notes"):
        for note in report.notes:
            st.write(f"- {note}")

render_footer()
