#!/usr/bin/env python3
"""Smoke test: instruction similarity with a real task instruction.md."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tracker_defaults import TRACKER_SHEET_URL, TRACKER_WORKSHEET, TRACKER_COL_TASK_INSTRUCTION
from qc_engine import (
    check_instruction_similarity,
    enrich_similarity_match_texts,
    fetch_similarity_corpus,
    render_instruction_precheck_html,
)
from ui_components import _resolve_tracker_instruction


def main() -> int:
    task_path = Path(
        os.environ.get(
            "SMOKE_INSTRUCTION",
            ROOT.parent / "My_Accepted_Tasks/pkt-frag-audit/instruction.md",
        )
    )
    if not task_path.is_file():
        print(f"FAIL: instruction not found: {task_path}")
        return 1

    instruction = task_path.read_text(encoding="utf-8")
    print(f"Task instruction: {task_path.name} ({len(instruction)} chars)")

    api_key = os.environ.get("OPENAI_API_KEY", "")
    result = check_instruction_similarity(
        instruction_text=instruction,
        sheet_url=TRACKER_SHEET_URL,
        worksheet=TRACKER_WORKSHEET,
        instruction_col_index=TRACKER_COL_TASK_INSTRUCTION,
        api_key=api_key,
    )

    matches = result.get("matches") or []
    tracker = result.get("tracker_instructions") or {}
    print(f"Corpus: {result.get('corpus_count', 0)} instructions")
    print(f"Embedding ran: {result.get('embedding_ran')} blocked: {result.get('blocked')}")
    print(f"Matches: {len(matches)}")

    failures: list[str] = []
    for i, match in enumerate(matches):
        text = _resolve_tracker_instruction(match, tracker)
        if not text:
            failures.append(f"match[{i}] {match.task_id}: empty tracker instruction")
        elif len(text) < 20:
            failures.append(f"match[{i}] {match.task_id}: suspiciously short ({len(text)} chars)")

    instructions, _, _, _ = fetch_similarity_corpus(
        sheet_url=TRACKER_SHEET_URL,
        worksheet=TRACKER_WORKSHEET,
        instruction_col_index=TRACKER_COL_TASK_INSTRUCTION,
    )
    conntrack_key = next((k for k in instructions if "conntrack" in k.lower()), "")
    if conntrack_key:
        print(f"Corpus spot-check: {conntrack_key} -> {len(instructions[conntrack_key])} chars")
    else:
        print("Corpus spot-check: no conntrack task found (non-fatal)")

    # Simulate stale session objects missing matched_instruction.
    stale = pickle.loads(pickle.dumps(result))
    for match in stale["matches"]:
        match.matched_instruction = ""
    stale["tracker_instructions"] = {m.task_id: "" for m in stale["matches"]}
    recovered = enrich_similarity_match_texts(
        stale["matches"],
        instructions,
        stale["tracker_instructions"],
    )
    for match in stale["matches"]:
        if not _resolve_tracker_instruction(match, recovered, instructions):
            failures.append(f"stale recovery failed for {match.task_id}")

    html = render_instruction_precheck_html(result, instruction, "smoke-test")
    if "Full instruction comparison" not in html:
        failures.append("HTML report missing full comparison section")
    if html.count("match-card") < len(matches):
        failures.append("HTML report missing match comparison cards")

    out = ROOT / "smoke_instruction_precheck.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote sample report: {out}")

    if failures:
        print("\nFAILURES:")
        for item in failures:
            print(f"  - {item}")
        return 1

    print("\nOK — all smoke checks passed.")
    if matches:
        top = matches[0]
        print(
            f"Top match: {top.task_id} "
            f"(word {round((top.lexical_score or 0) * 100)}%, "
            f"meaning {round((top.semantic_score or 0) * 100) if top.semantic_score is not None else '—'}%)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
