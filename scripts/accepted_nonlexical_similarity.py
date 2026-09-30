#!/usr/bin/env python3
"""ACCEPTED-only similarity using non-lexical signals (Terminus-aligned).

Terminus QC portal rules (tracker_defaults.py):
  - Flag when instruction meaning >= 70% even if word overlap is low.
  - Dual block (word + meaning both >= 60%) also flags — we SKIP lexical-only.

This script checks all ACCEPTED task zips pairwise using ONLY:
  1. Instruction embedding cosine similarity (meaning)
  2. SPEC.md embedding similarity
  3. Combined instruction+SPEC embedding similarity
  4. Environment file-tree Jaccard (structure), when paired with moderate meaning

Industry practice (semantic dedup): 0.70–0.88 = review band; 0.88+ = near-duplicate.
We use Terminus 70% meaning threshold + 85% structure with meaning support.

Excludes copy-paste pairs (identical normalized text or lexical >= 97%).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reuse batch extraction / embedding helpers.
from scripts.batch_pairwise_similarity import (  # noqa: E402
    TaskEntry,
    build_combined_text,
    cosine_matrix_from_embeddings,
    embed_texts,
    extract_env_paths,
    extract_tasks,
    find_spec_in_zip,
    is_same_wording,
    lexical_similarity,
    load_env_file,
    parse_task_toml,
    structure_jaccard,
)
from scripts.enrich_similarity_reports import (  # noqa: E402
    COLUMN_SPEC,
    CSV_HEADERS,
    apply_tracker,
    build_similarity_pair,
    clean_flag_reason,
    load_tracker_lookup,
)
import zipfile  # noqa: E402
import re  # noqa: E402

# Terminus meaning-only block threshold (portal default).
SEMANTIC_FLAG_PCT = 70.0
SEMANTIC_HIGH_PCT = 88.0
STRUCTURE_FLAG_PCT = 85.0
STRUCTURE_MEANING_SUPPORT_PCT = 65.0

DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "All_Tasks_LIST/J26.0420"
DEFAULT_OUTPUT = DEFAULT_INPUT / "batch_similarity_reports"
DEFAULT_TRACKER = Path(__file__).resolve().parents[2] / "Main - Project Terminus Task Tracker - Cognyzer.xlsx"

FILENAME_PATTERN = re.compile(
    r"^ACCEPTED__Terminus-2nd-Edition__"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$"
)


@dataclass
class NonLexicalPair:
    task_a: TaskEntry
    task_b: TaskEntry
    instruction_semantic_pct: float
    spec_semantic_pct: float
    combined_semantic_pct: float
    structure_jaccard_pct: float
    instruction_lexical_pct: float
    non_lexical_max_pct: float
    tier: str
    flag_reason: str


def extract_accepted_only(input_dir: Path) -> list[TaskEntry]:
    """Extract only ACCEPTED zips from snorkel-ACCEPTED-* folder."""
    tasks: list[TaskEntry] = []
    errors: list[str] = []

    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("snorkel-ACCEPTED"):
            continue
        for zip_path in sorted(folder.iterdir()):
            if not zip_path.name.endswith(".zip"):
                continue
            m = FILENAME_PATTERN.match(zip_path.name)
            if not m:
                continue
            uuid = m.group(1)
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    names = zf.namelist()
                    if "instruction.md" not in names:
                        errors.append(f"No instruction.md: {zip_path.name}")
                        continue
                    instruction = zf.read("instruction.md").decode("utf-8", errors="replace").strip()
                    toml_raw = (
                        zf.read("task.toml").decode("utf-8", errors="replace")
                        if "task.toml" in names else ""
                    )
                    spec_path, spec_text = find_spec_in_zip(names, zf)
                    env_paths = extract_env_paths(names)
                meta = parse_task_toml(toml_raw)
                tasks.append(TaskEntry(
                    uuid=uuid,
                    status="ACCEPTED",
                    filename=zip_path.name,
                    instruction=instruction,
                    category=meta.get("category", ""),
                    languages=meta.get("languages", ""),
                    difficulty=meta.get("difficulty", ""),
                    instruction_preview=instruction[:200],
                    spec_text=spec_text,
                    spec_preview=spec_text[:200] if spec_text else "",
                    spec_path=spec_path,
                    env_paths=env_paths,
                    env_file_count=len(env_paths),
                    combined_text=build_combined_text(instruction, spec_text),
                ))
            except Exception as exc:
                errors.append(f"{zip_path.name}: {exc}")

    if errors:
        print(f"  Warnings ({len(errors)}): {errors[:5]}")
    return tasks


def evaluate_non_lexical_flag(
    inst_sem: float,
    spec_sem: float,
    combined_sem: float,
    struct_pct: float,
    inst_lex: float,
    inst_a: str,
    inst_b: str,
) -> tuple[bool, float, str, str]:
    """Return (flagged, non_lex_max, tier, reason). Lexical NOT used for flagging."""
    if is_same_wording(inst_a, inst_b, inst_lex):
        return False, 0.0, "", ""

    reasons: list[str] = []
    if inst_sem >= SEMANTIC_FLAG_PCT:
        reasons.append(f"instr_sem>={int(SEMANTIC_FLAG_PCT)}")
    if spec_sem >= SEMANTIC_FLAG_PCT:
        reasons.append(f"spec_sem>={int(SEMANTIC_FLAG_PCT)}")
    if combined_sem >= SEMANTIC_FLAG_PCT:
        reasons.append(f"combined_sem>={int(SEMANTIC_FLAG_PCT)}")
    if (
        struct_pct >= STRUCTURE_FLAG_PCT
        and (inst_sem >= STRUCTURE_MEANING_SUPPORT_PCT or spec_sem >= STRUCTURE_MEANING_SUPPORT_PCT)
    ):
        reasons.append(f"structure>={int(STRUCTURE_FLAG_PCT)}+meaning")

    if not reasons:
        return False, 0.0, "", ""

    # Reject lexical-only similarity: all non-lexical below flag thresholds.
    non_lex_scores = [inst_sem, spec_sem, combined_sem]
    if struct_pct >= STRUCTURE_FLAG_PCT and (
        inst_sem >= STRUCTURE_MEANING_SUPPORT_PCT or spec_sem >= STRUCTURE_MEANING_SUPPORT_PCT
    ):
        non_lex_scores.append(struct_pct)
    non_lex_max = max(non_lex_scores)

    # If ONLY lexical would have flagged (hypothetical 80% run) but no semantic signal:
    if non_lex_max < SEMANTIC_FLAG_PCT and struct_pct < STRUCTURE_FLAG_PCT:
        return False, 0.0, "", ""

    tier = "high" if non_lex_max >= SEMANTIC_HIGH_PCT else "review"
    return True, non_lex_max, tier, ", ".join(reasons)


def compute_accepted_pairs(
    tasks: list[TaskEntry],
    inst_emb,
    spec_emb,
    combined_emb,
) -> list[NonLexicalPair]:
    inst_cos = cosine_matrix_from_embeddings(inst_emb)
    spec_cos = cosine_matrix_from_embeddings(spec_emb)
    combined_cos = cosine_matrix_from_embeddings(combined_emb)

    flagged: list[NonLexicalPair] = []
    n = len(tasks)
    total = n * (n - 1) // 2
    print(f"  Pairwise: {total:,} ACCEPTED-only comparisons (non-lexical rules)")

    for i, j in combinations(range(n), 2):
        inst_lex = round(lexical_similarity(tasks[i].instruction, tasks[j].instruction) * 100, 1)
        inst_sem = round(float(inst_cos[i, j]) * 100, 1)
        spec_sem = 0.0
        if tasks[i].spec_text and tasks[j].spec_text:
            spec_sem = round(float(spec_cos[i, j]) * 100, 1)
        combined_sem = round(float(combined_cos[i, j]) * 100, 1)
        struct_pct = round(structure_jaccard(tasks[i].env_paths, tasks[j].env_paths) * 100, 1)

        ok, non_lex_max, tier, reason = evaluate_non_lexical_flag(
            inst_sem, spec_sem, combined_sem, struct_pct, inst_lex,
            tasks[i].instruction, tasks[j].instruction,
        )
        if ok:
            flagged.append(NonLexicalPair(
                task_a=tasks[i],
                task_b=tasks[j],
                instruction_semantic_pct=inst_sem,
                spec_semantic_pct=spec_sem,
                combined_semantic_pct=combined_sem,
                structure_jaccard_pct=struct_pct,
                instruction_lexical_pct=inst_lex,
                non_lexical_max_pct=non_lex_max,
                tier=tier,
                flag_reason=reason,
            ))

    print(f"  Flagged (non-lexical): {len(flagged)} pairs")
    return flagged


def pair_to_dict(pair: NonLexicalPair, lookup: dict) -> dict[str, str]:
    row = {
        "tier": pair.tier,
        "max_score_pct": round(pair.non_lexical_max_pct, 1),
        "flag_reason": pair.flag_reason,
        "similarity_pair": "",
        "task_a_status": "ACCEPTED",
        "task_b_status": "ACCEPTED",
        "task_a_uuid": pair.task_a.uuid,
        "task_b_uuid": pair.task_b.uuid,
        "task_a_trainer": "",
        "task_b_trainer": "",
        "task_a_name": "",
        "task_b_name": "",
        "task_a_category": pair.task_a.category,
        "task_b_category": pair.task_b.category,
        "task_a_languages": pair.task_a.languages,
        "task_b_languages": pair.task_b.languages,
        "task_a_difficulty": pair.task_a.difficulty,
        "task_b_difficulty": pair.task_b.difficulty,
        "instruction_lexical_pct": pair.instruction_lexical_pct,
        "instruction_semantic_pct": pair.instruction_semantic_pct,
        "spec_lexical_pct": 0.0,
        "spec_semantic_pct": pair.spec_semantic_pct,
        "structure_jaccard_pct": pair.structure_jaccard_pct,
        "combined_semantic_pct": pair.combined_semantic_pct,
    }
    apply_tracker(row, lookup, "task_a")
    apply_tracker(row, lookup, "task_b")
    row["similarity_pair"] = build_similarity_pair(row)
    row["flag_reason"] = clean_flag_reason(row["flag_reason"])
    return row


def build_summary_stats(tasks: list[TaskEntry], pairs: list[NonLexicalPair]) -> pd.DataFrame:
    involved: set[str] = set()
    for p in pairs:
        involved.add(p.task_a.uuid)
        involved.add(p.task_b.uuid)

    high = sum(1 for p in pairs if p.tier == "high")
    review = sum(1 for p in pairs if p.tier == "review")

    rows = [
        {"Metric": "ACCEPTED tasks analyzed", "Value": len(tasks)},
        {"Metric": "Non-lexical similar pairs (Terminus rules)", "Value": len(pairs)},
        {"Metric": "High tier (non-lexical max >= 88%)", "Value": high},
        {"Metric": "Review tier (70-88% meaning/structure)", "Value": review},
        {"Metric": "Unique ACCEPTED tasks in a similar pair", "Value": len(involved)},
        {"Metric": "ACCEPTED tasks with NO non-lexical match", "Value": len(tasks) - len(involved)},
        {"Metric": "", "Value": ""},
        {"Metric": "Rules used (no lexical flagging)", "Value": ""},
        {"Metric": "  Instruction meaning >= 70%", "Value": "Terminus portal default"},
        {"Metric": "  SPEC meaning >= 70%", "Value": "Same threshold"},
        {"Metric": "  Combined meaning >= 70%", "Value": "Instruction + SPEC embed"},
        {"Metric": "  Structure >= 85% + meaning >= 65%", "Value": "Fixture layout match"},
        {"Metric": "  Excluded copy-paste (lexical >= 97%)", "Value": "Yes"},
    ]
    return pd.DataFrame(rows)


def task_id_summary(pairs: list[NonLexicalPair], lookup: dict) -> pd.DataFrame:
    from collections import Counter
    counts: Counter[str] = Counter()
    meta: dict[str, dict] = {}
    for p in pairs:
        for t in (p.task_a, p.task_b):
            counts[t.uuid] += 1
            if t.uuid not in meta:
                m = lookup.get(t.uuid.lower(), {})
                meta[t.uuid] = {
                    "trainer": m.get("trainer", ""),
                    "name": m.get("name", ""),
                }
    rows = []
    for uuid, cnt in counts.most_common():
        m = meta.get(uuid, {})
        rows.append({
            "Task UUID": uuid,
            "Trainer": m.get("trainer", ""),
            "Task Name": m.get("name", ""),
            "Non-Lexical Similar Pair Count": cnt,
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="ACCEPTED-only non-lexical similarity check")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY required in .env")
        return 1

    print("ACCEPTED Non-Lexical Similarity Check (Terminus-aligned)")
    print(f"  Meaning threshold: {SEMANTIC_FLAG_PCT}% (portal default)")
    print(f"  High tier: >= {SEMANTIC_HIGH_PCT}%")
    print(f"  Structure rule: >= {STRUCTURE_FLAG_PCT}% + meaning >= {STRUCTURE_MEANING_SUPPORT_PCT}%")
    print("  Lexical: reference only — NOT used for flagging")
    print()

    print("[1] Extract ACCEPTED zips...")
    tasks = extract_accepted_only(args.input_dir)
    print(f"  Loaded {len(tasks)} ACCEPTED tasks")
    if len(tasks) < 2:
        return 1

    print("[2] Embed (instruction, SPEC, combined)...")
    inst_emb = embed_texts([t.instruction for t in tasks], api_key, "instructions")
    spec_emb = embed_texts([t.spec_text if t.spec_text else " " for t in tasks], api_key, "specs")
    combined_emb = embed_texts([t.combined_text for t in tasks], api_key, "combined")
    if inst_emb is None or combined_emb is None:
        print("ERROR: embedding failed")
        return 1

    print("[3] Pairwise non-lexical comparison...")
    pairs = compute_accepted_pairs(tasks, inst_emb, spec_emb, combined_emb)

    lookup = load_tracker_lookup(args.tracker.resolve())
    rows = [pair_to_dict(p, lookup) for p in sorted(pairs, key=lambda x: -x.non_lexical_max_pct)]

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ACCEPTED_nonlexical_similarity.csv"
    xlsx_path = out_dir / "ACCEPTED_nonlexical_similarity.xlsx"

    df = pd.DataFrame(rows)
    if not df.empty:
        df_out = df.rename(columns=dict(COLUMN_SPEC))
        df_out = df_out[[label for _, label in COLUMN_SPEC if label in df_out.columns]]
    else:
        df_out = pd.DataFrame(columns=CSV_HEADERS)
    df_out.to_csv(csv_path, index=False)

    summary_df = build_summary_stats(tasks, pairs)
    task_df = task_id_summary(pairs, lookup)
    pairs_df = df_out

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        task_df.to_excel(writer, sheet_name="Task IDs", index=False)
        pairs_df.to_excel(writer, sheet_name="Similar Pairs", index=False)

    involved = {p.task_a.uuid for p in pairs} | {p.task_b.uuid for p in pairs}
    print()
    print(f"Wrote: {csv_path.name}")
    print(f"Wrote: {xlsx_path.name}")
    print(f"  ACCEPTED tasks: {len(tasks)}")
    print(f"  Tasks similar (non-lexical): {len(involved)}")
    print(f"  Tasks clean (no match): {len(tasks) - len(involved)}")
    print(f"  Flagged pairs: {len(pairs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
