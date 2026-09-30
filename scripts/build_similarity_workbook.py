#!/usr/bin/env python3
"""Build Excel workbook with similarity data tabs + Summary stats sheet."""

from __future__ import annotations

import argparse
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_REPORTS = (
    Path(__file__).resolve().parents[2]
    / "All_Tasks_LIST/J26.0420/batch_similarity_reports"
)
DEFAULT_INPUT = Path(__file__).resolve().parents[2] / "All_Tasks_LIST/J26.0420"
DEFAULT_OUTPUT = DEFAULT_REPORTS / "batch_similarity_reports.xlsx"

STATUS_FOLDER = re.compile(r"snorkel-(ACCEPTED|NEEDS-REVISION|REJECTED)-\d{4}-\d{2}-\d{2}")
FNAME = re.compile(
    r"^(ACCEPTED|NEEDS-REVISION|REJECTED)__Terminus-2nd-Edition__"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$"
)


def count_batch_tasks(input_dir: Path) -> tuple[int, dict[str, int], set[str]]:
    """Count tasks in zip folders and collect all UUIDs."""
    status_counts: dict[str, int] = defaultdict(int)
    uuids: set[str] = set()
    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir() or not STATUS_FOLDER.match(folder.name):
            continue
        for z in folder.glob("*.zip"):
            m = FNAME.match(z.name)
            if not m:
                continue
            status_counts[m.group(1)] += 1
            with zipfile.ZipFile(z) as zf:
                if "instruction.md" in zf.namelist():
                    uuids.add(m.group(2).lower())
    return len(uuids), dict(status_counts), uuids


def load_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


def task_involvement(df: pd.DataFrame) -> pd.DataFrame:
    """Per-task stats: how many similar pairs each task appears in."""
    if df.empty:
        return pd.DataFrame()

    counts: Counter[str] = Counter()
    meta: dict[str, dict[str, str]] = {}

    for side in ("A", "B"):
        for _, row in df.iterrows():
            uuid = str(row.get(f"Task {side} UUID", "")).strip().lower()
            if not uuid or uuid == "nan":
                continue
            counts[uuid] += 1
            if uuid not in meta:
                meta[uuid] = {
                    "status": str(row.get(f"Task {side} Status", "")),
                    "trainer": str(row.get(f"Task {side} Trainer", "")).replace("nan", ""),
                    "name": str(row.get(f"Task {side} Name", "")).replace("nan", ""),
                    "category": str(row.get(f"Task {side} Category", "")).replace("nan", ""),
                }

    rows = []
    for uuid, pair_count in counts.most_common():
        m = meta.get(uuid, {})
        rows.append({
            "UUID": uuid,
            "Status": m.get("status", ""),
            "Trainer": m.get("trainer", ""),
            "Task Name": m.get("name", ""),
            "Category": m.get("category", ""),
            "Similar Pair Count": pair_count,
        })
    return pd.DataFrame(rows)


def build_summary_sections(
    all_df: pd.DataFrame,
    batch_total: int,
    batch_status: dict[str, int],
    batch_uuids: set[str],
) -> list[tuple[str, pd.DataFrame]]:
    """Return list of (section_title, dataframe) for summary sheet."""
    sections: list[tuple[str, pd.DataFrame]] = []

    if all_df.empty:
        sections.append(("Overview", pd.DataFrame({"Note": ["No similarity data"]})))
        return sections

    involved: set[str] = set()
    for side in ("A", "B"):
        col = f"Task {side} UUID"
        for val in all_df[col].dropna().astype(str):
            involved.add(val.strip().lower())

    not_involved = len(batch_uuids - involved)
    high = (all_df["Similarity Tier"] == "high").sum()
    near = (all_df["Similarity Tier"] == "near_similar").sum()

    within_accepted = (
        (all_df["Task A Status"] == "ACCEPTED") & (all_df["Task B Status"] == "ACCEPTED")
    ).sum()
    within_needs = (
        (all_df["Task A Status"] == "NEEDS-REVISION")
        & (all_df["Task B Status"] == "NEEDS-REVISION")
    ).sum()
    within_rejected = (
        (all_df["Task A Status"] == "REJECTED") & (all_df["Task B Status"] == "REJECTED")
    ).sum()
    cross_status = len(all_df) - within_accepted - within_needs - within_rejected

    overview = pd.DataFrame([
        {"Metric": "Total tasks in batch (with instruction.md)", "Value": batch_total},
        {"Metric": "Tasks by status — ACCEPTED", "Value": batch_status.get("ACCEPTED", 0)},
        {"Metric": "Tasks by status — NEEDS-REVISION", "Value": batch_status.get("NEEDS-REVISION", 0)},
        {"Metric": "Tasks by status — REJECTED", "Value": batch_status.get("REJECTED", 0)},
        {"Metric": "Zips skipped (no instruction.md)", "Value": sum(batch_status.values()) - batch_total},
        {"Metric": "", "Value": ""},
        {"Metric": "Flagged similar pairs (≥80% threshold)", "Value": len(all_df)},
        {"Metric": "High similarity pairs (≥90%)", "Value": int(high)},
        {"Metric": "Near-similar pairs (80–90%)", "Value": int(near)},
        {"Metric": "", "Value": ""},
        {"Metric": "Unique tasks in at least one similar pair", "Value": len(involved)},
        {"Metric": "Tasks with NO similar match in report", "Value": not_involved},
        {"Metric": "% of batch tasks flagged as similar", "Value": f"{100 * len(involved) / batch_total:.1f}%"},
        {"Metric": "", "Value": ""},
        {"Metric": "Pairs — both ACCEPTED", "Value": int(within_accepted)},
        {"Metric": "Pairs — both NEEDS-REVISION", "Value": int(within_needs)},
        {"Metric": "Pairs — both REJECTED", "Value": int(within_rejected)},
        {"Metric": "Pairs — cross-status (mixed)", "Value": int(cross_status)},
    ])
    sections.append(("Overview", overview))

    # Score distribution
    buckets = [
        ("95–100%", (all_df["Max Score %"] >= 95).sum()),
        ("90–95%", ((all_df["Max Score %"] >= 90) & (all_df["Max Score %"] < 95)).sum()),
        ("85–90%", ((all_df["Max Score %"] >= 85) & (all_df["Max Score %"] < 90)).sum()),
        ("80–85%", ((all_df["Max Score %"] >= 80) & (all_df["Max Score %"] < 85)).sum()),
    ]
    sections.append(("Score Distribution", pd.DataFrame(
        [{"Score Range": k, "Pair Count": int(v)} for k, v in buckets]
    )))

    # Flag reason breakdown (primary trigger)
    primary = []
    for reason in all_df["Flag Reason"].fillna(""):
        part = str(reason).split(",")[0].strip() if reason else "unknown"
        primary.append(part)
    reason_counts = Counter(primary)
    sections.append(("Primary Flag Trigger", pd.DataFrame(
        [{"Flag Reason": k, "Pair Count": v} for k, v in reason_counts.most_common()]
    )))

    # Top trainers by involvement
    trainer_pairs: Counter[str] = Counter()
    for side in ("A", "B"):
        for trainer in all_df[f"Task {side} Trainer"].fillna("").astype(str):
            t = trainer.strip()
            if t and t != "nan":
                trainer_pairs[t] += 1
    sections.append(("Trainer Involvement (pair appearances)", pd.DataFrame(
        [{"Trainer": k, "Appearances in Pairs": v} for k, v in trainer_pairs.most_common(25)]
    )))

    # Top tasks by similarity count
    per_task = task_involvement(all_df).head(30)
    sections.append(("Top 30 Tasks by Similar Pair Count", per_task))

    return sections


def write_summary_sheet(writer: pd.ExcelWriter, sections: list[tuple[str, pd.DataFrame]]) -> None:
    """Write all summary sections into one sheet with spacing."""
    workbook = writer.book
    worksheet = workbook.create_sheet("Summary", 0)
    writer.sheets["Summary"] = worksheet

    row = 1
    for title, df in sections:
        worksheet.cell(row=row, column=1, value=title)
        row += 1
        if df.empty:
            row += 2
            continue
        # Write header
        for col_idx, col_name in enumerate(df.columns, 1):
            worksheet.cell(row=row, column=col_idx, value=col_name)
        row += 1
        # Write data
        for _, data_row in df.iterrows():
            for col_idx, col_name in enumerate(df.columns, 1):
                val = data_row[col_name]
                worksheet.cell(row=row, column=col_idx, value=val)
            row += 1
        row += 2  # blank row between sections


def main() -> int:
    parser = argparse.ArgumentParser(description="Build similarity Excel workbook with Summary tab")
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    reports_dir = args.reports_dir.resolve()
    input_dir = args.input_dir.resolve()
    output_path = args.output.resolve()

    batch_total, batch_status, batch_uuids = count_batch_tasks(input_dir)

    files = {
        "ALL Combined": reports_dir / "ALL_combined_similarity.csv",
        "ACCEPTED": reports_dir / "ACCEPTED_similarity.csv",
        "NEEDS REVISION": reports_dir / "NEEDS_REVISION_similarity.csv",
        "REJECTED": reports_dir / "REJECTED_similarity.csv",
        "Tasks by Similarity": None,  # generated
    }

    all_df = load_csv(files["ALL Combined"])
    sections = build_summary_sections(all_df, batch_total, batch_status, batch_uuids)
    tasks_df = task_involvement(all_df)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        write_summary_sheet(writer, sections)
        tasks_df.to_excel(writer, sheet_name="Tasks by Similarity", index=False)

        for sheet_name, path in files.items():
            if path is None or sheet_name == "Tasks by Similarity":
                continue
            df = load_csv(path)
            if not df.empty:
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)

    print(f"Wrote: {output_path}")
    print(f"  Sheets: Summary, Tasks by Similarity, ALL Combined, ACCEPTED, NEEDS REVISION, REJECTED")
    if not all_df.empty:
        involved = set()
        for side in ("A", "B"):
            for v in all_df[f"Task {side} UUID"].dropna().astype(str):
                involved.add(v.strip().lower())
        print(f"  Batch tasks: {batch_total}")
        print(f"  Tasks in similar pairs: {len(involved)}")
        print(f"  Tasks with no match: {len(batch_uuids - involved)}")
        print(f"  Total flagged pairs: {len(all_df)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
