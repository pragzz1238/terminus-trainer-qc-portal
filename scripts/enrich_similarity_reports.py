#!/usr/bin/env python3
"""Enrich and reformat batch similarity CSVs with tracker trainer/task names."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

DEFAULT_TRACKER = (
    Path(__file__).resolve().parents[2]
    / "Main - Project Terminus Task Tracker - Cognyzer.xlsx"
)
DEFAULT_REPORTS = (
    Path(__file__).resolve().parents[2]
    / "All_Tasks_LIST/J26.0420/batch_similarity_reports"
)

# (internal_key, excel_header) — A/B fields interleaved for side-by-side comparison.
COLUMN_SPEC: list[tuple[str, str]] = [
    ("tier", "Similarity Tier"),
    ("max_score_pct", "Max Score %"),
    ("flag_reason", "Flag Reason"),
    ("similarity_pair", "Similar Pair"),
    ("task_a_status", "Task A Status"),
    ("task_b_status", "Task B Status"),
    ("task_a_uuid", "Task A UUID"),
    ("task_b_uuid", "Task B UUID"),
    ("task_a_trainer", "Task A Trainer"),
    ("task_b_trainer", "Task B Trainer"),
    ("task_a_name", "Task A Name"),
    ("task_b_name", "Task B Name"),
    ("task_a_category", "Task A Category"),
    ("task_b_category", "Task B Category"),
    ("task_a_languages", "Task A Languages"),
    ("task_b_languages", "Task B Languages"),
    ("task_a_difficulty", "Task A Difficulty"),
    ("task_b_difficulty", "Task B Difficulty"),
    ("instruction_lexical_pct", "Instruction Word Overlap %"),
    ("instruction_semantic_pct", "Instruction Meaning %"),
    ("spec_lexical_pct", "SPEC Word Overlap %"),
    ("spec_semantic_pct", "SPEC Meaning %"),
    ("structure_jaccard_pct", "Environment Structure %"),
    ("combined_semantic_pct", "Combined Meaning %"),
]

CANONICAL_COLUMNS = [key for key, _ in COLUMN_SPEC]
CSV_HEADERS = [label for _, label in COLUMN_SPEC]
HEADER_TO_KEY = {label: key for key, label in COLUMN_SPEC}

# Map legacy / duplicate headers -> canonical internal key.
COLUMN_ALIASES: dict[str, str] = {
    "similarity_tier": "tier",
    "Similarity Tier": "tier",
    "flagged_reason": "flag_reason",
    "Flag Reason": "flag_reason",
    "similarity_pair": "similarity_pair",
    "Similar Pair": "similarity_pair",
    "task_a_trainer_name": "task_a_trainer",
    "Task A Trainer": "task_a_trainer",
    "task_a_task_name": "task_a_name",
    "Task A Name": "task_a_name",
    "task_b_trainer_name": "task_b_trainer",
    "Task B Trainer": "task_b_trainer",
    "task_b_task_name": "task_b_name",
    "Task B Name": "task_b_name",
    "max_score_pct": "max_score_pct",
    "Max Score %": "max_score_pct",
    "task_a_status": "task_a_status",
    "Task A Status": "task_a_status",
    "task_b_status": "task_b_status",
    "Task B Status": "task_b_status",
    "task_a_uuid": "task_a_uuid",
    "Task A UUID": "task_a_uuid",
    "task_b_uuid": "task_b_uuid",
    "Task B UUID": "task_b_uuid",
    "instruction_lexical_pct": "instruction_lexical_pct",
    "Instruction Word Overlap %": "instruction_lexical_pct",
    "instruction_semantic_pct": "instruction_semantic_pct",
    "Instruction Meaning %": "instruction_semantic_pct",
    "spec_lexical_pct": "spec_lexical_pct",
    "SPEC Word Overlap %": "spec_lexical_pct",
    "spec_semantic_pct": "spec_semantic_pct",
    "SPEC Meaning %": "spec_semantic_pct",
    "structure_jaccard_pct": "structure_jaccard_pct",
    "Environment Structure %": "structure_jaccard_pct",
    "combined_semantic_pct": "combined_semantic_pct",
    "Combined Meaning %": "combined_semantic_pct",
    "task_a_category": "task_a_category",
    "Task A Category": "task_a_category",
    "task_b_category": "task_b_category",
    "Task B Category": "task_b_category",
    "task_a_languages": "task_a_languages",
    "Task A Languages": "task_a_languages",
    "task_b_languages": "task_b_languages",
    "Task B Languages": "task_b_languages",
    "task_a_difficulty": "task_a_difficulty",
    "Task A Difficulty": "task_a_difficulty",
    "task_b_difficulty": "task_b_difficulty",
    "Task B Difficulty": "task_b_difficulty",
}


def load_tracker_lookup(xlsx_path: Path, sheet: str = "May 1st - 31st") -> dict[str, dict[str, str]]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]

    lookup: dict[str, dict[str, str]] = {}
    for _, row in df.iterrows():
        raw_id = row.get("Task ID", "")
        if pd.isna(raw_id):
            continue
        task_id = str(raw_id).strip().lower().replace("{", "").replace("}", "")
        if not task_id or task_id == "nan":
            continue

        trainer = row.get("Trainer Name", "")
        task_name = row.get("Task name", "")
        lookup[task_id] = {
            "trainer": "" if pd.isna(trainer) else str(trainer).strip(),
            "name": "" if pd.isna(task_name) else str(task_name).strip(),
        }
    return lookup


def first_value(row: dict[str, str], *keys: str) -> str:
    """Return first non-empty value across duplicate column keys."""
    for key in keys:
        val = (row.get(key) or "").strip()
        if val:
            return val
    return ""


def normalize_row(raw: dict[str, str]) -> dict[str, str]:
    """Collapse duplicate columns and map to canonical internal keys."""
    merged: dict[str, str] = {}
    for key, val in raw.items():
        if key is None:
            continue
        canon = COLUMN_ALIASES.get(key, HEADER_TO_KEY.get(key, key))
        if canon not in CANONICAL_COLUMNS:
            canon = COLUMN_ALIASES.get(canon, canon)
        text = (val or "").strip()
        if canon not in merged or (text and not merged[canon]):
            merged[canon] = text

    out: dict[str, str] = {col: "" for col in CANONICAL_COLUMNS}

    for canon in CANONICAL_COLUMNS:
        if canon in merged:
            out[canon] = merged[canon]

    # Handle duplicate trainer/name from repeated headers in source CSV.
    out["task_a_trainer"] = first_value(
        raw, "task_a_trainer", "task_a_trainer_name",
    ) or out.get("task_a_trainer", "")
    out["task_a_name"] = first_value(
        raw, "task_a_name", "task_a_task_name",
    ) or out.get("task_a_name", "")
    out["task_b_trainer"] = first_value(
        raw, "task_b_trainer", "task_b_trainer_name",
    ) or out.get("task_b_trainer", "")
    out["task_b_name"] = first_value(
        raw, "task_b_name", "task_b_task_name",
    ) or out.get("task_b_name", "")

    if not out["tier"] and merged.get("similarity_tier"):
        out["tier"] = merged["similarity_tier"]
    if not out["flag_reason"] and merged.get("flagged_reason"):
        out["flag_reason"] = merged["flagged_reason"]

    return out


def apply_tracker(row: dict[str, str], lookup: dict[str, dict[str, str]], side: str) -> bool:
    uuid = (row.get(f"{side}_uuid") or "").strip().lower()
    meta = lookup.get(uuid)
    if not meta:
        return False
    if meta["trainer"]:
        row[f"{side}_trainer"] = meta["trainer"]
    if meta["name"]:
        row[f"{side}_name"] = meta["name"]
    return bool(meta["trainer"] or meta["name"])


def format_label(status: str, trainer: str, name: str, uuid: str) -> str:
    trainer = (trainer or "").strip()
    name = (name or "").strip()
    short_id = (uuid or "").strip()[:8]

    if trainer and name:
        body = f"{trainer} · {name}"
    elif trainer:
        body = f"{trainer} · {short_id}"
    elif name:
        body = name
    else:
        body = short_id or "unknown"

    status = (status or "").strip()
    return f"{status}: {body}" if status else body


def build_similarity_pair(row: dict[str, str]) -> str:
    a = format_label(
        row.get("task_a_status", ""),
        row.get("task_a_trainer", ""),
        row.get("task_a_name", ""),
        row.get("task_a_uuid", ""),
    )
    b = format_label(
        row.get("task_b_status", ""),
        row.get("task_b_trainer", ""),
        row.get("task_b_name", ""),
        row.get("task_b_uuid", ""),
    )
    return f"{a} ↔ {b}"


def clean_flag_reason(reason: str) -> str:
    """Shorten flag reason for readability."""
    if not reason:
        return ""
    parts = []
    for part in reason.split(","):
        part = part.strip()
        part = part.replace("instruction_lexical", "instr_lex")
        part = part.replace("instruction_semantic", "instr_sem")
        part = part.replace("spec_lexical", "spec_lex")
        part = part.replace("spec_semantic", "spec_sem")
        part = part.replace("structure_jaccard", "structure")
        part = part.replace("combined_semantic", "combined")
        parts.append(part)
    return ", ".join(parts)


def format_csv(csv_path: Path, lookup: dict[str, dict[str, str]]) -> tuple[int, int]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return 0, 0
        raw_rows = list(reader)

    matched = 0
    clean_rows: list[dict[str, str]] = []

    for raw in raw_rows:
        row = normalize_row(raw)
        for side in ("task_a", "task_b"):
            if apply_tracker(row, lookup, side):
                matched += 1
        row["similarity_pair"] = build_similarity_pair(row)
        row["flag_reason"] = clean_flag_reason(row.get("flag_reason", ""))
        clean_rows.append(row)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=CSV_HEADERS,
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in clean_rows:
            writer.writerow({label: row.get(key, "") for key, label in COLUMN_SPEC})

    return len(clean_rows), matched


def split_status_reports(reports_dir: Path) -> None:
    """Rebuild per-status CSVs from the cleaned combined report."""
    combined = reports_dir / "ALL_combined_similarity.csv"
    if not combined.is_file():
        return

    with open(combined, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    normalized = [normalize_row(r) for r in raw_rows]

    for status, filename in [
        ("ACCEPTED", "ACCEPTED_similarity.csv"),
        ("NEEDS-REVISION", "NEEDS_REVISION_similarity.csv"),
        ("REJECTED", "REJECTED_similarity.csv"),
    ]:
        filtered = [
            r for r in normalized
            if r.get("task_a_status") == status and r.get("task_b_status") == status
        ]
        path = reports_dir / filename
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
            writer.writeheader()
            for row in filtered:
                writer.writerow({label: row.get(key, "") for key, label in COLUMN_SPEC})
        print(f"  rebuilt {filename}: {len(filtered)} rows")


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean and enrich similarity CSV reports")
    parser.add_argument("--tracker", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--sheet", default="May 1st - 31st")
    args = parser.parse_args()

    if not args.tracker.is_file():
        print(f"ERROR: Tracker not found: {args.tracker}")
        return 1
    if not args.reports_dir.is_dir():
        print(f"ERROR: Reports dir not found: {args.reports_dir}")
        return 1

    lookup = load_tracker_lookup(args.tracker.resolve(), args.sheet)
    print(f"Tracker lookup: {len(lookup)} UUIDs")

    # Clean combined first, then rebuild per-status from it.
    combined = args.reports_dir / "ALL_combined_similarity.csv"
    if not combined.is_file():
        print("ERROR: ALL_combined_similarity.csv not found")
        return 1

    rows, matched = format_csv(combined, lookup)
    print(f"  ALL_combined_similarity.csv: {rows} rows, {matched} tracker hits")
    split_status_reports(args.reports_dir.resolve())
    print("Done — CSVs cleaned and reformatted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
