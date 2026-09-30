"""Terminus 3 trainer QC portal: instruction similarity, full task QC, rubric check."""

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
from tracker_defaults import (
    INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD,
    INSTRUCTION_SIM_THRESHOLD,
)
from ui_components import (
    inject_global_css,
    inject_page_favicon,
    render_download_panel,
    render_panel_header,
    render_footer,
    render_hero,
    render_metric_grid,
    render_similarity_instruction_reviews,
    render_similarity_match_table,
    render_topbar,
    render_section_header,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = APP_DIR / "terminus_task_corpus.json"
SIM_PCT = int(INSTRUCTION_SIM_THRESHOLD * 100)
MEANING_BLOCK_PCT = int(INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD * 100)
INSTRUCTION_CHECK_HELP = (
    f"Compare your instruction against the team tracker. Flagged when "
    f"word overlap and meaning are both ≥ {SIM_PCT}%, or meaning alone ≥ {MEANING_BLOCK_PCT}%. "
    f"Use 👁 review to read both prompts side-by-side."
)
SIMILARITY_TAB_HELP = (
    f"Compared from your zip's instruction.md against the team tracker. "
    f"Flagged when both ≥ {SIM_PCT}% or meaning ≥ {MEANING_BLOCK_PCT}%. "
    f"Open 👁 review to compare full instructions."
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


def _instruction_review_tracker_maps(
    pre_result: dict,
    *,
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = 7,
    corpus_json_path: str = "",
) -> dict[str, str]:
    """Tracker instruction text for side-by-side review, with live corpus fallback."""
    qe = _qc_engine()
    tracker = dict(
        pre_result.get("tracker_instructions")
        or st.session_state.get("tracker_instruction_cache")
        or {}
    )
    matches = pre_result.get("matches") or []
    missing_ids = [
        m.task_id
        for m in matches
        if not (
            (getattr(m, "matched_instruction", "") or "").strip()
            or (tracker.get(m.task_id) or "").strip()
        )
    ]
    if not missing_ids:
        return tracker

    try:
        instructions, _, _, _ = qe.fetch_similarity_corpus(
            sheet_url=sheet_url,
            worksheet=worksheet,
            task_col=task_col,
            instruction_col=instruction_col,
            trainer_col=trainer_col,
            instruction_col_index=instruction_col_index,
            corpus_json_path=corpus_json_path,
        )
    except Exception:
        return tracker

    for match in matches:
        task_id = match.task_id
        if (tracker.get(task_id) or "").strip():
            continue
        text = (instructions.get(task_id) or "").strip()
        if not text:
            continue
        tracker[task_id] = text
        if not (getattr(match, "matched_instruction", "") or "").strip():
            match.matched_instruction = text
    return tracker


@st.cache_resource(show_spinner=False)
def _qc_engine():
    import similarity_service

    return similarity_service


st.set_page_config(
    page_title="Terminus QC · Task Checker",
    page_icon="favicon.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_page_favicon()

inject_global_css()

sheet_defaults = resolve_sheet_defaults()
llm_model = resolve_llm_model()
llm_ready = llm_configured()
sheet_preconfigured = bool(sheet_defaults.get("url"))

if "instruction_pre_result" not in st.session_state:
    st.session_state.instruction_pre_result = None
if "instruction_pre_text" not in st.session_state:
    st.session_state.instruction_pre_text = ""
if "qc_cache" not in st.session_state:
    st.session_state.qc_cache = None
if "show_pre_downloads" not in st.session_state:
    st.session_state.show_pre_downloads = False
if "rubric_pre_text" not in st.session_state:
    st.session_state.rubric_pre_text = ""

provider = api_provider_label() if llm_ready else "—"
render_topbar(llm_ready, provider, llm_model)
render_hero()

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
    run_llm = st.checkbox("Run the LLM review (quality gate, LLMaJ, quality panel)", value=True, disabled=not llm_ready)
    if sheet_preconfigured:
        st.caption(
            f"Similarity source: **Terminus Task Instructions** · tab "
            f'`{sheet_defaults.get("worksheet", "")}` · col '
            f'**{sheet_defaults.get("instruction_col", "Task Instruction")}** (column 7)'
        )
        sheet_url = sheet_defaults.get("url", "")
        worksheet = sheet_defaults.get("worksheet", "")
        task_col = sheet_defaults.get("task_col", "")
        instruction_col = sheet_defaults.get("instruction_col", "")
        trainer_col = sheet_defaults.get("trainer_col", "")
        instruction_col_index = int(sheet_defaults.get("instruction_col_index", "7") or "7")
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
            instruction_col_index = int(sheet_defaults.get("instruction_col_index", "7") or "7")
            spec_col = st.text_input("SPEC column", value=sheet_defaults.get("spec_col", ""))
        use_local_corpus = st.checkbox("Fallback to bundled corpus", value=True)

corpus_path = str(DEFAULT_CORPUS) if use_local_corpus and DEFAULT_CORPUS.exists() else ""

tab_instruction, tab_full_qc, tab_rubric = st.tabs(["Instruction similarity", "Full task QC", "Rubric check"])

with tab_instruction:
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
        max_upload_size=MAX_INSTRUCTION_MD_MB,
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
            try:
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
            except Exception as exc:
                st.error("Instruction check failed — see details below.")
                st.exception(exc)
            else:
                st.session_state.instruction_pre_text = instruction_text
                st.session_state.instruction_pre_result = pre_result
                st.session_state.show_pre_downloads = False
                if pre_result.get("tracker_instructions"):
                    st.session_state.tracker_instruction_cache = pre_result["tracker_instructions"]

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
                    st.error(pre_result.get("message") or qe.CHANGE_TASK_MESSAGE)
                elif corpus_count == 0:
                    st.error(pre_result.get("message", "No reference corpus loaded."))
                else:
                    st.success(pre_result.get("message", "Instruction check passed."))

                if pre_result.get("matches"):
                    tracker_maps = _instruction_review_tracker_maps(
                        pre_result,
                        sheet_url=sheet_url,
                        worksheet=worksheet,
                        task_col=task_col,
                        instruction_col=instruction_col,
                        trainer_col=trainer_col,
                        instruction_col_index=instruction_col_index,
                        corpus_json_path=corpus_path if not sheet_url.strip() else "",
                    )
                    st.session_state.tracker_instruction_cache = tracker_maps
                    render_similarity_match_table(pre_result["matches"])
                    render_similarity_instruction_reviews(
                        instruction_text,
                        pre_result["matches"],
                        key_prefix="pre_inst",
                        tracker_instructions=tracker_maps,
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
                    title="Instruction similarity report",
                    subtitle="Top matches, 👁 side-by-side instruction review, and tracker load details.",
                    key_prefix="dl_pre",
                )

    elif st.session_state.instruction_pre_result:
        pre_result = st.session_state.instruction_pre_result
        blocked = pre_result.get("blocked")
        status = "blocked" if blocked else "passed"
        st.caption(f"Last instruction check: **{status}** — run again to refresh or download reports.")
        if pre_result.get("matches"):
            tracker_maps = _instruction_review_tracker_maps(
                pre_result,
                sheet_url=sheet_url,
                worksheet=worksheet,
                task_col=task_col,
                instruction_col=instruction_col,
                trainer_col=trainer_col,
                instruction_col_index=instruction_col_index,
                corpus_json_path=corpus_path if not sheet_url.strip() else "",
            )
            st.session_state.tracker_instruction_cache = tracker_maps
            render_similarity_match_table(pre_result["matches"])
            render_similarity_instruction_reviews(
                st.session_state.instruction_pre_text,
                pre_result["matches"],
                key_prefix="pre_persist",
                tracker_instructions=tracker_maps,
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
                title="Instruction similarity report",
                subtitle="From your most recent instruction check.",
                key_prefix="dl_pre_persist",
            )

with tab_full_qc:
    with st.container(border=True):
        render_panel_header(
            1,
            "Full task QC (Terminus 3)",
            "Upload the submission zip: task.toml, instruction.md, environment/, solution/ and tests/ at the "
            f"top level, no rubrics.txt or README.md. Maximum {MAX_ZIP_MB} MB.",
        )
        uploaded = st.file_uploader(
            f"Task zip file (max {MAX_ZIP_MB} MB)",
            type=["zip"],
            max_upload_size=MAX_ZIP_MB,
        )
        qc_rubric = st.text_area(
            "Rubric (optional): the text you will paste in the platform or pass with `stb ... -r rubric.txt`",
            height=110,
            placeholder="Agent reads the evidence before writing code, +2\nAgent edits files under /tests, -5",
            key="qc_rubric",
        )

    if uploaded is None:
        st.info("Upload a task zip to run the full Terminus 3 review.")
    elif uploaded.size > MAX_ZIP_BYTES:
        st.error(f"Zip file is too large ({_format_size(uploaded.size)}). Maximum is {MAX_ZIP_MB} MB.")
    else:
        upload_sig = f"{uploaded.name}:{uploaded.size}:{hash(qc_rubric)}:{run_llm}"
        if st.session_state.qc_cache and st.session_state.qc_cache.get("upload_sig") != upload_sig:
            st.session_state.qc_cache = None
        render_metric_grid([
            ("Archive", uploaded.name, "neutral"),
            ("Size", _format_size(uploaded.size), "neutral"),
            ("LLM review", "on" if (run_llm and llm_ready) else "off", "neutral"),
        ])
        run_qc = st.button("Run full QC", type="primary", use_container_width=True)

        if run_qc:
            import t3_review
            from config import build_openai_client, resolve_llm_parallel_workers

            qe = _qc_engine()
            bar = st.progress(0, text="Starting…")

            def _progress(label: str, frac: float) -> None:
                bar.progress(min(max(frac, 0.0), 1.0), text=label)

            def _similarity(text: str) -> dict:
                return qe.check_instruction_similarity(
                    instruction_text=text,
                    sheet_url=sheet_url,
                    worksheet=worksheet,
                    task_col=task_col,
                    instruction_col=instruction_col,
                    trainer_col=trainer_col,
                    instruction_col_index=instruction_col_index,
                    corpus_json_path=corpus_path if not sheet_url.strip() else "",
                    api_key=resolve_openai_api_key(),
                )

            client = build_openai_client(resolve_openai_api_key()) if (run_llm and llm_ready) else None
            try:
                with tempfile.TemporaryDirectory(prefix="t3qc_") as tmp:
                    review = t3_review.run_review(
                        uploaded.getvalue(),
                        uploaded.name,
                        Path(tmp),
                        rubric_text=qc_rubric,
                        similarity_fn=_similarity,
                        client=client,
                        model=llm_model,
                        run_llm=bool(client),
                        parallel=resolve_llm_parallel_workers(10),
                        on_progress=_progress,
                    )
            except Exception as exc:
                bar.empty()
                st.error("The review could not run on this archive.")
                st.exception(exc)
            else:
                bar.empty()
                st.session_state.qc_cache = {"upload_sig": upload_sig, "review": review}

        cache = st.session_state.qc_cache
        if cache and cache.get("upload_sig") == upload_sig and cache.get("review") is not None:
            import t3_prompts
            import t3_review

            review = cache["review"]
            verdict, reasons = review.verdict()
            if verdict == "READY TO UPLOAD":
                st.success(f"**{verdict}** · " + " ".join(reasons))
            elif verdict in ("STATIC ONLY", "INCOMPLETE"):
                st.warning(f"**{verdict}** · " + " ".join(reasons))
            else:
                st.error(f"**{verdict}**")
                for r in reasons:
                    st.markdown(f"- {r}")

            panel_block = review.axis_blocking()
            render_metric_grid([
                ("Task", review.task_name, "neutral"),
                ("Static / CI", f"{review.static.get('errors', 0)} err · {review.static.get('warnings', 0)} warn",
                 "fail" if review.static.get("errors") else ("warn" if review.static.get("warnings") else "pass")),
                ("Similarity", "BLOCK" if review.similarity_blocked() else ("OK" if review.similarity else "n/a"),
                 "fail" if review.similarity_blocked() else "pass"),
                ("LLMaJ", "—" if not review.llm_ran else f"{len(review.llmaj_failures())} fail",
                 "neutral" if not review.llm_ran else ("fail" if review.llmaj_failures() else "pass")),
                ("Quality gate", "—" if not review.llm_ran else f"{len(review.gate_failures())} fail",
                 "neutral" if not review.llm_ran else ("fail" if review.gate_failures() else "pass")),
                ("Quality panel", "—" if not review.llm_ran else ("BLOCK" if panel_block else "clear"),
                 "neutral" if not review.llm_ran else ("fail" if panel_block else "pass")),
            ])

            report_html = t3_review.render_html(review)
            report_json = json.dumps(t3_review.to_dict(review), indent=2, default=str)
            safe = review.task_name.replace(" ", "-")
            render_download_panel(
                report_html, report_json, f"{safe}_t3_qc.html", f"{safe}_t3_qc.json",
                title="Full QC report",
                subtitle="Every finding with file:line citations, the obligation list and the gate results.",
                key_prefix="dl_t3",
            )

            t_static, t_sim, t_panel, t_gate, t_llmaj, t_inst, t_obl = st.tabs([
                "Static / CI", "Similarity", "Quality panel", "Quality gate", "LLMaJ", "Instruction & difficulty", "Obligations",
            ])
            with t_static:
                st.caption("The CI and preflight rules that can be decided by reading files (vendored from the "
                           "team's taskkit static checker). ERROR blocks at CI; WARN must be fixed or justified.")
                for n in review.layout_notes:
                    st.error(n)
                rows = [f for f in review.static.get("findings", []) if f["severity"] != "INFO"]
                if rows:
                    st.dataframe(
                        [{"Severity": f["severity"], "Code": f["code"], "Where": f["where"],
                          "Finding": f["message"], "Rule": f["source"]} for f in rows],
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.success("No static errors or warnings.")
                with st.expander("Info-level notes"):
                    for f in review.static.get("findings", []):
                        if f["severity"] == "INFO":
                            st.write(f"- `{f['code']}` {f['where']}: {f['message']}")
            with t_sim:
                sim = review.similarity
                if not sim:
                    st.info("Similarity did not run (no instruction.md or no corpus).")
                else:
                    (st.error if sim.get("blocked") else st.success)(sim.get("message", ""))
                    if sim.get("matches"):
                        render_similarity_match_table(sim["matches"])
                        render_similarity_instruction_reviews(
                            review.instruction_text, sim["matches"], key_prefix="t3_sim",
                            tracker_instructions={m.task_id: m.matched_instruction for m in sim["matches"]},
                        )
            with t_panel:
                if not review.llm_ran:
                    st.info("The LLM review did not run.")
                st.caption("Five axes, each reviewed on the split view the platform uses. Minor or Major blocks on "
                           "every axis except protected_ground_truth, where only Major blocks. Advisory never blocks.")
                for key, axis in t3_prompts.AXES.items():
                    res = review.axes.get(key)
                    if not res:
                        continue
                    blocks = res["verdict"] in axis["blocks_on"]
                    with st.container(border=True):
                        st.markdown(f"#### `{key}` · **{res['verdict']}**{' · blocks' if blocks else ''}")
                        st.caption(f"View: {axis['view']}")
                        st.write(res.get("summary", ""))
                        for i, f in enumerate(res.get("findings", []), 1):
                            st.markdown(f"**{i}. ({f.get('severity')}) {f.get('title', '')}**")
                            st.write(f.get("mechanism", ""))
                            st.caption(f"Cited: {', '.join(f.get('citations') or [])} · Fix: {f.get('fix', '')}")
            with t_gate:
                st.caption("The 29 public Terminal-Bench quality criteria the platform runs before the panel. "
                           "`difficult` is advisory in returned payloads; treat every other failure as blocking.")
                if review.gate:
                    st.dataframe(
                        [{"Criterion": c.get("name"), "Result": c.get("result"),
                          "Reasoning": c.get("reasoning")} for c in sorted(review.gate, key=lambda c: c.get("result") != "fail")],
                        use_container_width=True, hide_index=True,
                    )
            with t_llmaj:
                st.caption("The seven LLM-as-judge checks in `stb harbor check`.")
                if review.llmaj:
                    st.dataframe(
                        [{"Check": c.get("name"), "Result": c.get("result"), "Reasoning": c.get("reasoning")}
                         for c in review.llmaj],
                        use_container_width=True, hide_index=True,
                    )
            with t_inst:
                if review.instruction:
                    ins = review.instruction
                    st.markdown(f"**Instruction:** {ins.get('verdict')} · AI-text risk **{ins.get('ai_text_risk')}**")
                    st.write(ins.get("summary", ""))
                    for i in ins.get("issues", []):
                        st.markdown(f"- [{i.get('severity')}] {i.get('issue')} ({i.get('where')}). Fix: {i.get('fix')}")
                if review.difficulty:
                    d = review.difficulty
                    st.markdown(f"**Expertise floor:** {d.get('expertise_floor')} · **Difficulty risk:** "
                                f"{d.get('difficulty_risk')} · **difficulty_explanation:** {d.get('explanation_quality')}")
                    st.write(f"Crux: {d.get('crux')}")
                    st.write(d.get("reasoning", ""))
                    for s in d.get("suggestions", []):
                        st.markdown(f"- {s}")
            with t_obl:
                ob = (review.obligations or {}).get("obligations", [])
                st.caption("Every normative sentence becomes an obligation the reference must satisfy and a visible "
                           "test must exercise. Anything here with no test is a sound_verifier finding waiting to happen.")
                if ob:
                    st.dataframe(
                        [{"Id": o.get("id"), "Kind": o.get("kind"), "Core": o.get("core"),
                          "Obligation": o.get("text"), "Source": o.get("citation")} for o in ob],
                        use_container_width=True, hide_index=True,
                    )
            if review.errors:
                with st.expander("Run notes"):
                    for e in review.errors:
                        st.write(f"- {e}")

with tab_rubric:
    with st.container(border=True):
        render_panel_header(
            2,
            "Rubric check",
            "Rules as of 23 Sep 2026: one line per criterion, `Agent …, ±N`, N any integer from 1 to 5 (never 0), "
            "an explicit + on positives, at least one negative, positives summing to 10 to 40, and no mention of "
            "tests, task.toml or instruction.md.",
        )
        rubric_text = st.text_area("Paste the rubric", height=200, key="rubric_tab_text")
        if st.button("Check rubric", type="primary", use_container_width=True, key="rubric_btn"):
            import t3_static

            if not rubric_text.strip():
                st.error("Paste a rubric first.")
            else:
                with tempfile.TemporaryDirectory() as d:
                    rp = Path(d) / "rubric.txt"
                    rp.write_text(rubric_text)
                    rep = t3_static.Report()
                    t3_static.check_rubric(rp, rep)
                errs = [f for f in rep.findings if f["severity"] == "ERROR"]
                warns = [f for f in rep.findings if f["severity"] == "WARN"]
                (st.error if errs else (st.warning if warns else st.success))(
                    f"{len(errs)} error(s), {len(warns)} warning(s)")
                for f in rep.findings:
                    where = f["where"].replace(str(rp), "rubric")
                    st.markdown(f"- **{f['severity']}** `{f['code']}` {where}: {f['message']}")

render_footer()
