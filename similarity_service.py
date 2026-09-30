"""Instruction similarity against the team corpus (Task Instructions tab, or bundled JSON).

Extracted from the Edition-2 qc_engine: only the similarity path is kept. Full task QC now lives in
t3_review.py (Terminus 3 rules).
"""

from __future__ import annotations

import json
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO, StringIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from config import (
    api_provider_label,
    build_openai_client,
    resolve_embed_model,
    resolve_openai_api_key,
)

from tracker_defaults import (
    INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD,
    INSTRUCTION_SIM_THRESHOLD,
    INSTRUCTION_SIM_BLOCK,
    INSTRUCTION_SIM_WARN,
    TASK_INSTRUCTION_HEADER,
    TRACKER_COL_TASK_INSTRUCTION,
)
SIM_THRESHOLD_WARN = INSTRUCTION_SIM_WARN
SIM_THRESHOLD_BLOCK = INSTRUCTION_SIM_BLOCK
SEMANTIC_BLOCK_PCT = int(INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD * 100)
DUAL_BLOCK_PCT = int(INSTRUCTION_SIM_THRESHOLD * 100)
BUNDLED_CORPUS_PATH = Path(__file__).resolve().parent / "terminus_task_corpus.json"
CHANGE_TASK_MESSAGE = (
    f"CANNOT UPLOAD: this instruction is {SEMANTIC_BLOCK_PCT}% or more similar to a task already in Tela. "
    "Change the task before uploading."
)

TASK_REQUIRED_FILES: dict[str, str] = {
    "task.toml": "Task metadata file",
    "instruction.md": "Agent instructions",
    "environment/Dockerfile": "Docker build file",
    "solution/solve.sh": "Oracle solution",
    "tests/test.sh": "Test runner",
    "tests/test_outputs.py": "Test assertions",
}




@dataclass
class Issue:
    severity: str
    message: str
    fix_hint: str = ""


@dataclass
class SimilarityMatch:
    task_id: str
    score: float
    source: str
    trainer: str = ""
    method: str = "tfidf"
    lexical_score: float | None = None
    semantic_score: float | None = None
    dual_block: bool = False
    block_reason: str = ""
    matched_instruction: str = ""


def similarity_flag_label(match: SimilarityMatch) -> str:
    if not match.dual_block:
        return "No"
    if match.block_reason == "meaning":
        return f"YES (meaning ≥{SEMANTIC_BLOCK_PCT}%)"
    if match.block_reason == "words":
        return f"YES (cosine ≥{DUAL_BLOCK_PCT}%)"
    return "YES"


def _html_comparison_styles() -> str:
    return """
    pre.instr-body {
      background: #f6f8fa; padding: 14px 16px; border-radius: 8px;
      white-space: pre-wrap; word-break: break-word;
      font-size: 13px; line-height: 1.55; margin: 0;
      border: 1px solid #e2e8f0; max-height: none;
    }
    .match-card {
      border: 1px solid #cbd5e1; border-radius: 12px;
      padding: 14px 16px 18px; margin: 20px 0;
      background: #fff;
    }
    .match-card.flagged { border-color: #f87171; background: #fffbfb; }
    .match-scores {
      font-size: 15px; margin-bottom: 12px; padding-bottom: 10px;
      border-bottom: 1px solid #e2e8f0;
    }
    .compare-grid {
      display: grid; grid-template-columns: 1fr 1fr; gap: 18px;
    }
    .instr-label {
      font-size: 12px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.04em; color: #64748b; margin-bottom: 6px;
    }
    @media (max-width: 960px) { .compare-grid { grid-template-columns: 1fr; } }
"""


def _resolve_match_instruction_text(
    task_id: str,
    matched_instruction: str,
    tracker_instructions: dict[str, str] | None,
    corpus_instructions: dict[str, str] | None = None,
) -> str:
    text = (matched_instruction or "").strip()
    if text:
        return text
    if tracker_instructions:
        text = (tracker_instructions.get(task_id) or "").strip()
        if text:
            return text
    if corpus_instructions:
        return (corpus_instructions.get(task_id) or "").strip()
    return ""


def enrich_similarity_match_texts(
    matches: list[SimilarityMatch],
    instructions: dict[str, str],
    tracker_instructions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Ensure every match has full tracker text for UI and HTML export."""
    merged = dict(tracker_instructions or {})
    for match in matches:
        task_id = match.task_id
        text = _resolve_match_instruction_text(
            task_id,
            match.matched_instruction,
            merged,
            instructions,
        )
        if text and not (match.matched_instruction or "").strip():
            match.matched_instruction = text
        if text:
            merged[task_id] = text
        elif task_id not in merged:
            merged[task_id] = ""
    return merged


def _html_instruction_comparison_section(
    your_instruction: str,
    matches: list[Any],
    tracker_instructions: dict[str, str] | None = None,
) -> str:
    import html as html_module

    if not matches:
        return "<p><em>No tracker matches returned.</em></p>"

    safe_yours = html_module.escape(your_instruction or "")
    yours_len = len(your_instruction or "")
    blocks = ""

    for m in matches:
        if isinstance(m, SimilarityMatch):
            task_id = m.task_id
            trainer = m.trainer or "—"
            lex = round((m.lexical_score or 0) * 100, 1)
            emb_val = (
                round(m.semantic_score * 100, 1)
                if m.semantic_score is not None else None
            )
            flagged = m.dual_block
            flag = similarity_flag_label(m)
            raw_match = _resolve_match_instruction_text(
                task_id, m.matched_instruction, tracker_instructions,
            )
        else:
            task_id = str(m.get("task", ""))
            trainer = m.get("trainer") or "—"
            lex = m.get("lexical_percent", 0)
            emb_val = m.get("embedding_percent")
            flagged = bool(m.get("dual_block"))
            flag = m.get("flag_label") or ("YES" if flagged else "No")
            raw_match = _resolve_match_instruction_text(
                task_id,
                str(m.get("matched_instruction") or ""),
                tracker_instructions,
            )

        emb_s = f"{emb_val}%" if emb_val is not None else "—"
        safe_match = html_module.escape(raw_match) if raw_match else ""
        match_len = len(raw_match)
        card_cls = "match-card flagged" if flagged else "match-card"
        match_body = (
            safe_match
            if safe_match
            else "<em>Tracker instruction not loaded — re-run the check and download again.</em>"
        )

        blocks += f"""
<section class="{card_cls}" id="review-{html_module.escape(task_id)}">
  <div class="match-scores">
    👁 <strong>{html_module.escape(task_id)}</strong>
    · Trainer: {html_module.escape(trainer)}
    · Word overlap: <strong>{lex}%</strong>
    · Meaning: <strong>{emb_s}</strong>
    · Flagged: <strong>{html_module.escape(flag)}</strong>
  </div>
  <div class="compare-grid">
    <div class="instr-col">
      <div class="instr-label">Your instruction · {yours_len:,} characters</div>
      <pre class="instr-body">{safe_yours}</pre>
    </div>
    <div class="instr-col">
      <div class="instr-label">Tracker instruction · {match_len:,} characters</div>
      <pre class="instr-body">{match_body}</pre>
    </div>
  </div>
</section>"""

    return blocks



def _sheet_id_from_url(url: str) -> str:
    path = urlparse(url.strip()).path
    parts = [p for p in path.split("/") if p]
    if "d" in parts:
        idx = parts.index("d")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    raise ValueError("Could not parse Google Sheet ID from URL")


def _sheet_csv_url(sheet_url: str, worksheet: str = "") -> str:
    sheet_id = _sheet_id_from_url(sheet_url)
    base = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv"
    if worksheet:
        base += f"&sheet={quote(worksheet)}"
    return base


def _fetch_sheet_dataframe(sheet_url: str, worksheet: str = "") -> pd.DataFrame:
    """Fetch public Google Sheet as CSV (works on Streamlit Cloud)."""
    import requests

    csv_url = _sheet_csv_url(sheet_url, worksheet)
    resp = requests.get(csv_url, timeout=45)
    resp.raise_for_status()
    text = resp.text
    if not text.strip():
        raise ValueError("Sheet CSV export returned empty body.")
    if text.lstrip().startswith("<!DOCTYPE") or "<html" in text[:500].lower():
        raise ValueError(
            "Sheet CSV export returned HTML — share the sheet as "
            "'Anyone with the link can view' and verify the worksheet tab name."
        )
    df = pd.read_csv(StringIO(text))
    if df.empty:
        raise ValueError("Sheet CSV parsed to zero rows.")
    return df


def _pick_column(columns: list[str], candidates: list[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    for col in columns:
        norm = col.lower().strip()
        if any(token in norm for token in candidates):
            return col
    return None


def _resolve_instruction_column(
    df: pd.DataFrame,
    instruction_col: str,
    instruction_col_index: int,
) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    if instruction_col:
        picked = _pick_column(
            cols,
            [instruction_col.lower(), "task instruction", TASK_INSTRUCTION_HEADER.lower()],
        )
        if picked:
            return picked
    picked = _pick_column(
        cols,
        ["task instruction", "instruction", "instruction.md", "instruction text"],
    )
    if picked:
        return picked
    if 1 <= instruction_col_index <= len(cols):
        return cols[instruction_col_index - 1]
    return None


def load_reference_from_sheet(
    sheet_url: str,
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    spec_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = TRACKER_COL_TASK_INSTRUCTION,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    """Load corpus from Terminus Task Tracker sheet (CSV export)."""
    notes: list[str] = []
    df = _fetch_sheet_dataframe(sheet_url, worksheet)
    df.columns = [str(c).strip() if str(c).strip() else f"unnamed_{i}" for i, c in enumerate(df.columns)]

    task_column = task_col or _pick_column(
        list(df.columns), ["task name", "task_name", "taskname", "task title", "title"]
    )
    if not task_column:
        task_column = _pick_column(list(df.columns), ["task id", "taskid", "task"])

    instruction_column = _resolve_instruction_column(df, instruction_col, instruction_col_index)
    trainer_column = trainer_col or _pick_column(
        list(df.columns), ["trainer name", "trainer", "name"]
    )
    spec_column = spec_col or _pick_column(
        list(df.columns), ["spec", "spec.md", "specification", "spec text"]
    ) if spec_col else None

    if not instruction_column:
        raise ValueError(
            f'Could not find instruction column "{TASK_INSTRUCTION_HEADER}" '
            f"(expected column {instruction_col_index})."
        )

    instructions: dict[str, str] = {}
    specs: dict[str, str] = {}
    corpus_meta: dict[str, dict[str, str]] = {}

    for row_idx, row in df.iterrows():
        task_name = str(row.get(task_column, "")).strip() if task_column else ""
        instruction = str(row.get(instruction_column, "")).strip()
        trainer = str(row.get(trainer_column, "")).strip() if trainer_column else ""
        if not instruction or instruction.lower() == "nan":
            continue
        if not task_name or task_name.lower() == "nan":
            task_name = f"row-{row_idx + 2}"

        key = task_name
        if trainer:
            key = f"{task_name} · {trainer}"

        instructions[key] = instruction
        corpus_meta[key] = {"instruction": instruction, "trainer": trainer, "task_name": task_name}
        if spec_column:
            spec = str(row.get(spec_column, "")).strip()
            if spec and spec.lower() != "nan":
                specs[key] = spec

    notes.append(
        f"Loaded {len(instructions)} instructions from tracker sheet "
        f'("{worksheet or "default tab"}", col "{instruction_column}")'
    )
    if spec_column and specs:
        notes.append(f"Loaded {len(specs)} SPEC rows from sheet")
    return instructions, specs, corpus_meta, notes


def load_reference_from_json(corpus_path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    data = json.loads(corpus_path.read_text())
    instructions = {
        task_id: entry["instruction"]
        for task_id, entry in data.items()
        if isinstance(entry, dict) and entry.get("instruction")
    }
    specs = {
        task_id: entry["spec"]
        for task_id, entry in data.items()
        if isinstance(entry, dict) and entry.get("spec")
    }
    return instructions, specs, [f"Loaded {len(instructions)} tasks from local corpus JSON"]


def _should_exclude_task(task_id: str, exclude_name: str, query_text: str, corpus_text: str) -> bool:
    if not exclude_name:
        return False
    task_id_l = task_id.lower()
    exclude_l = exclude_name.lower()
    if task_id_l == exclude_l:
        return True
    if task_id_l.endswith(f"/{exclude_l}") or task_id_l.endswith(exclude_l):
        return True
    if exclude_l in task_id_l:
        return True
    if query_text.strip() and corpus_text.strip() and query_text.strip() == corpus_text.strip():
        return True
    return False


def _rank_similarity(
    query_text: str,
    corpus: dict[str, str],
    source: str,
    exclude_name: str = "",
    top_n: int = 5,
) -> list[SimilarityMatch]:
    if not query_text or not corpus:
        return []

    filtered = {
        task_id: text
        for task_id, text in corpus.items()
        if text.strip() and not _should_exclude_task(task_id, exclude_name, query_text, text)
    }
    if not filtered:
        return []

    task_ids = list(filtered.keys())
    docs = [filtered[tid] for tid in task_ids]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=5000,
        lowercase=True,
        strip_accents="unicode",
    )
    tb_vectors = vectorizer.fit_transform(docs)
    query_vector = vectorizer.transform([query_text])
    similarities = cosine_similarity(query_vector, tb_vectors).flatten()
    ranked = sorted(
        (
            SimilarityMatch(task_id=task_ids[idx], score=float(similarities[idx]), source=source)
            for idx in range(len(task_ids))
        ),
        key=lambda item: item.score,
        reverse=True,
    )
    return ranked[:top_n]


DEFAULT_TELA_INSTRUCTIONS_URL = "https://tela.cognyzer.com/api/sync/instructions"


def tela_settings() -> tuple[str, str]:
    """(url, token) for Tela's instruction corpus, from Streamlit secrets or the environment."""
    import os

    try:
        import streamlit as st

        secrets = dict(st.secrets)
    except Exception:
        secrets = {}
    token = str(secrets.get("TELA_SYNC_TOKEN", "") or os.environ.get("TELA_SYNC_TOKEN", "")).strip()
    url = str(secrets.get("TELA_INSTRUCTIONS_URL", "") or os.environ.get("TELA_INSTRUCTIONS_URL", "")
              or DEFAULT_TELA_INSTRUCTIONS_URL).strip()
    return url, token


def load_reference_from_tela() -> tuple[dict[str, str], dict[str, dict[str, str]], list[str]]:
    """Instructions trainers recorded in Tela (submitted, rework, approved, rejected), read live
    from GET /api/sync/instructions. Keyed by task name, like the sheet."""
    import requests

    url, token = tela_settings()
    if not token:
        return {}, {}, ["Tela not configured (no TELA_SYNC_TOKEN in secrets): Tela instructions were not loaded."]
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=30)
    if resp.status_code != 200:
        return {}, {}, [f"Tela instructions API returned {resp.status_code}: {resp.text[:200]}"]
    rows = resp.json().get("rows", [])
    instructions: dict[str, str] = {}
    meta: dict[str, dict[str, str]] = {}
    for r in rows:
        text = (r.get("instruction") or "").strip()
        key = (r.get("task_name") or r.get("external_id") or r.get("source_id") or "").strip()
        if not text or not key:
            continue
        instructions[key] = text
        meta[key] = {"instruction": text, "trainer": r.get("trainer_email") or "", "task_name": key,
                     "status": r.get("tela_status") or "", "source": "Tela"}
    return instructions, meta, [f"Tela: {len(instructions)} instructions loaded live "
                                f"(statuses submitted, rework, approved, rejected)."]


def load_similarity_corpus(
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    spec_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = TRACKER_COL_TASK_INSTRUCTION,
    corpus_json_path: str = "",
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    """The similarity corpus is Tela only: every task trainers have submitted there (statuses
    submitted, rework, approved, rejected), read live. The sheet and bundled JSON arguments are kept
    for call compatibility and ignored."""
    instructions, corpus_meta, notes = load_reference_from_tela()
    return instructions, {}, corpus_meta, notes

def fetch_similarity_corpus(
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    spec_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = TRACKER_COL_TASK_INSTRUCTION,
    corpus_json_path: str = "",
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    """Load tracker corpus with Streamlit cache when available."""
    try:
        from portal_cache import cached_load_similarity_corpus

        return cached_load_similarity_corpus(
            sheet_url=sheet_url,
            worksheet=worksheet,
            task_col=task_col,
            instruction_col=instruction_col,
            spec_col=spec_col,
            trainer_col=trainer_col,
            instruction_col_index=instruction_col_index,
            corpus_json_path=corpus_json_path,
        )
    except Exception:
        return load_similarity_corpus(
            sheet_url=sheet_url,
            worksheet=worksheet,
            task_col=task_col,
            instruction_col=instruction_col,
            spec_col=spec_col,
            trainer_col=trainer_col,
            instruction_col_index=instruction_col_index,
            corpus_json_path=corpus_json_path,
        )


def run_instruction_similarity(
    instruction_text: str,
    instructions_corpus: dict[str, str],
    corpus_meta: dict[str, dict[str, str]] | None = None,
    exclude_task_name: str = "",
    api_key: str = "",
    tracker_cache: dict[str, Any] | None = None,
) -> tuple[list[SimilarityMatch], bool, str, list[str], dict[str, Any]]:
    """Parallel lexical + embedding check. Block when dual ≥60% or meaning ≥70%."""
    from similarity_engine import compare_instruction_to_corpus_full

    notes: list[str] = []
    api_key = (api_key or resolve_openai_api_key()).strip()
    run_meta: dict[str, Any] = {
        "embedding_ran": False,
        "embed_model": resolve_embed_model(api_key),
        "embedding_error": None,
        "api_key_present": bool(api_key),
        "api_provider": api_provider_label(api_key),
        "corpus_size": 0,
    }
    inst_text = (instruction_text or "").strip()
    if not inst_text:
        return [], False, "Task Instruction is required.", notes, run_meta
    if not instructions_corpus:
        notes.append("Instruction similarity skipped — no reference corpus")
        return [], False, "", notes, run_meta

    meta = corpus_meta or {
        k: {"instruction": v, "trainer": "", "task_name": k}
        for k, v in instructions_corpus.items()
    }
    exclude = (
        {k for k in meta if exclude_task_name.lower() in k.lower()}
        if exclude_task_name.strip()
        else set()
    )

    result = compare_instruction_to_corpus_full(
        inst_text,
        meta,
        exclude_keys=exclude,
        api_key=api_key,
        top_n=10,
        tracker_cache=tracker_cache,
    )
    hits = result.hits
    sim_meta = result.meta
    run_meta = {
        "embedding_ran": sim_meta.embedding_ran,
        "embed_model": sim_meta.embed_model,
        "embedding_error": sim_meta.embedding_error,
        "api_key_present": sim_meta.api_key_present,
        "api_provider": api_provider_label(api_key),
        "corpus_size": sim_meta.corpus_size,
    }

    if sim_meta.embedding_ran:
        notes.append(
            f"Embedding check ran via {run_meta['api_provider']}: {sim_meta.embed_model} "
            f"({sim_meta.corpus_size} tracker instructions)"
        )
    elif sim_meta.embedding_error:
        notes.append(f"Embedding check did NOT run: {sim_meta.embedding_error}")
    else:
        notes.append("Embedding check did NOT run (no API key)")

    inst_matches = [
        SimilarityMatch(
            task_id=h.task_key,
            score=max(h.lexical_score, h.semantic_score or 0.0),
            source="instruction",
            trainer=h.trainer,
            method=h.method,
            lexical_score=h.lexical_score,
            semantic_score=h.semantic_score,
            dual_block=h.dual_block,
            block_reason=h.block_reason,
            matched_instruction=h.matched_instruction,
        )
        for h in hits
    ]

    blocked = any(m.dual_block for m in inst_matches)
    block_message = ""
    if blocked:
        top = next(m for m in inst_matches if m.dual_block)
        reason_note = (
            f"meaning ≥ {SEMANTIC_BLOCK_PCT}%"
            if top.block_reason == "meaning"
            else f"cosine similarity ≥ {DUAL_BLOCK_PCT}%"
        )
        block_message = (
            f"{CHANGE_TASK_MESSAGE} Closest match: {top.task_id}"
            f" (trainer: {top.trainer or 'unknown'}) — "
            f"cosine {round(top.lexical_score * 100)}%, "
            f"embedding {round((top.semantic_score or 0) * 100)}% "
            f"({reason_note}). Use 👁 review below to compare instructions."
        )
        notes.append(block_message)
    elif hits and sim_meta.embedding_ran:
        top = hits[0]
        notes.append(
            f"Top match — cosine {round(top.lexical_score * 100)}%, "
            f"embedding {round((top.semantic_score or 0) * 100)}% "
            f"(upload blocked at {SEMANTIC_BLOCK_PCT}%)"
        )
    elif hits:
        top = hits[0]
        notes.append(
            f"Top match — cosine {round(top.lexical_score * 100)}% only "
            f"(embedding check unavailable)"
        )

    return inst_matches, blocked, block_message, notes, run_meta


def check_instruction_similarity(
    instruction_text: str,
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = TRACKER_COL_TASK_INSTRUCTION,
    corpus_json_path: str = "",
    exclude_task_name: str = "",
    api_key: str = "",
) -> dict[str, Any]:
    """Instruction check against Tela, run before upload. The result message starts with CAN UPLOAD,
    CANNOT UPLOAD or CANNOT CONFIRM, for the screenshot trainers attach when they submit."""
    from portal_cache import tracker_cache_params

    tracker_cache = tracker_cache_params(
        sheet_url=sheet_url,
        worksheet=worksheet,
        task_col=task_col,
        instruction_col=instruction_col,
        spec_col="",
        trainer_col=trainer_col,
        instruction_col_index=instruction_col_index,
        corpus_json_path=corpus_json_path,
    )
    instructions, _, corpus_meta, load_notes = fetch_similarity_corpus(
        sheet_url=sheet_url,
        worksheet=worksheet,
        task_col=task_col,
        instruction_col=instruction_col,
        trainer_col=trainer_col,
        instruction_col_index=instruction_col_index,
        corpus_json_path=corpus_json_path,
    )
    api_key = api_key.strip() or resolve_openai_api_key()
    matches, blocked, block_message, notes, run_meta = run_instruction_similarity(
        instruction_text,
        instructions,
        corpus_meta=corpus_meta,
        exclude_task_name=exclude_task_name,
        api_key=api_key,
        tracker_cache=tracker_cache,
    )
    notes = load_notes + notes

    tela_ok = any(n.startswith("Tela: ") for n in load_notes)
    top = matches[0] if matches else None
    top_pct = 0
    if top is not None:
        top_pct = round(max(top.lexical_score or 0, top.semantic_score or 0) * 100)
    if blocked:
        pass_message = block_message
    elif not tela_ok:
        blocked = True
        pass_message = ("CANNOT CONFIRM: Tela's instructions could not be loaded, so this instruction was not "
                        "compared with anything. Do not upload on this result. " + " ".join(load_notes))
    elif not instructions:
        pass_message = (f"CAN UPLOAD: no tasks have been submitted in Tela yet, so there is nothing to be "
                        f"similar to (upload blocked at {SEMANTIC_BLOCK_PCT}%).")
    elif not run_meta.get("embedding_ran"):
        pass_message = (f"CAN UPLOAD on cosine similarity ({top_pct}% at most, limit {SEMANTIC_BLOCK_PCT}%), but the embedding "
                        "check did not run. Ask the admin to fix the API key and check again before uploading.")
        blocked = True
    else:
        pass_message = (f"CAN UPLOAD: highest similarity to any of the {len(instructions)} tasks in Tela is "
                        f"{top_pct}% (limit {SEMANTIC_BLOCK_PCT}%)"
                        + (f", closest: {top.task_id}." if top is not None else "."))

    tracker_instructions = enrich_similarity_match_texts(
        matches,
        instructions,
    )

    return {
        "blocked": blocked,
        "message": pass_message,
        "matches": matches,
        "query_instruction": instruction_text,
        "tracker_instructions": tracker_instructions,
        "notes": notes,
        "change_task": blocked,
        "corpus_count": len(instructions),
        "embedding_ran": run_meta.get("embedding_ran", False),
        "embed_model": run_meta.get("embed_model", "text-embedding-3-small"),
        "embedding_error": run_meta.get("embedding_error"),
        "corpus_size": run_meta.get("corpus_size", 0),
        "api_key_present": run_meta.get("api_key_present", False),
        "api_provider": run_meta.get("api_provider", ""),
    }


def instruction_precheck_to_dict(
    result: dict[str, Any],
    instruction_text: str,
    trainer_name: str = "",
) -> dict[str, Any]:
    matches = result.get("matches") or []
    tracker_map = result.get("tracker_instructions") or {}
    return {
        "type": "instruction_precheck",
        "report_format": "full_comparison_v2",
        "trainer_name": trainer_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "instruction_preview": instruction_text[:500],
        "instruction_full": instruction_text,
        "instruction_length": len(instruction_text),
        "blocked": result.get("blocked", False),
        "change_task": result.get("change_task", False),
        "message": result.get("message", ""),
        "corpus_count": result.get("corpus_count", 0),
        "embedding_ran": result.get("embedding_ran", False),
        "embed_model": result.get("embed_model", ""),
        "embedding_error": result.get("embedding_error"),
        "api_key_present": result.get("api_key_present", False),
        "api_provider": result.get("api_provider", ""),
        "dual_threshold_percent": DUAL_BLOCK_PCT,
        "semantic_block_threshold_percent": SEMANTIC_BLOCK_PCT,
        "tracker_instructions": result.get("tracker_instructions", {}),
        "matches": [
            {
                "task": m.task_id,
                "trainer": m.trainer,
                "lexical_percent": round((m.lexical_score or 0) * 100, 1),
                "embedding_percent": (
                    round(m.semantic_score * 100, 1)
                    if m.semantic_score is not None else None
                ),
                "dual_block": m.dual_block,
                "block_reason": m.block_reason,
                "flag_label": similarity_flag_label(m),
                "method": m.method,
                "matched_instruction": (
                    m.matched_instruction or tracker_map.get(m.task_id, "")
                ),
            }
            for m in matches
        ],
        "notes": result.get("notes", []),
    }


def render_instruction_precheck_html(
    result: dict[str, Any],
    instruction_text: str,
    trainer_name: str = "",
) -> str:
    import html as html_module

    data = instruction_precheck_to_dict(result, instruction_text, trainer_name)
    blocked = data["blocked"]
    color = "#e74c3c" if blocked else "#2ecc71"
    status = "CHANGE TASK" if blocked else "OK TO PROCEED"
    tracker_map = data.get("tracker_instructions") or {}

    rows = ""
    for m in data["matches"]:
        emb = m["embedding_percent"]
        emb_s = f"{emb}%" if emb is not None else "—"
        flag = m.get("flag_label") or ("YES" if m["dual_block"] else "No")
        task_id = html_module.escape(m["task"])
        rows += (
            f"<tr><td>{task_id}</td><td>{html_module.escape(m['trainer'] or '—')}</td>"
            f"<td>{m['lexical_percent']}%</td><td>{emb_s}</td>"
            f"<td>{html_module.escape(flag)}</td>"
            f"<td><a href=\"#review-{task_id}\">👁 Compare</a></td></tr>"
        )
    if not rows:
        rows = "<tr><td colspan='6'>No matches returned.</td></tr>"

    comparison_html = _html_instruction_comparison_section(
        instruction_text,
        data["matches"],
        tracker_map,
    )

    embed_status = "Ran" if data["embedding_ran"] else "Did not run"
    if data["embedding_error"]:
        embed_status += f" — {data['embedding_error']}"

    notes_html = "".join(f"<li>{html_module.escape(n)}</li>" for n in data["notes"])
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Instruction Similarity Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; max-width: 1400px; }}
    .summary {{ border: 2px solid {color}; border-radius: 12px; padding: 16px; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    {_html_comparison_styles()}
  </style>
</head>
<body>
  <div class="summary">
    <h1 style="color:{color}">{status}</h1>
    <p><strong>Trainer:</strong> {html_module.escape(trainer_name or "Not provided")}</p>
    <p><strong>Generated:</strong> {data["timestamp"]}</p>
    <p>{html_module.escape(data["message"])}</p>
    <p><strong>Corpus:</strong> {data["corpus_count"]} instructions ·
       <strong>Embedding:</strong> {html_module.escape(embed_status)} ({html_module.escape(data["embed_model"])}) ·
       <strong>Flag rules:</strong> both ≥ {data["dual_threshold_percent"]}% OR meaning ≥ {data["semantic_block_threshold_percent"]}%</p>
  </div>

  <h2>Similarity scores (top matches)</h2>
  <table>
    <tr><th>Task</th><th>Trainer</th><th>Word overlap</th><th>Meaning</th><th>Flagged?</th><th>Review</th></tr>
    {rows}
  </table>

  <h2>👁 Full instruction comparison</h2>
  <p>Each block shows <strong>your complete instruction</strong> next to the <strong>full tracker instruction</strong> for that row, with scores in the header. Use this to decide whether the task is truly too similar.</p>
  {comparison_html}

  <h2>Diagnostics</h2>
  <ul>{notes_html}</ul>
  <p><em>Re-download after re-running the check if tracker instructions appear empty.</em></p>
</body>
</html>"""


