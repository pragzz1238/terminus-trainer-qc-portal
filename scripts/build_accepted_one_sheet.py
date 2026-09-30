#!/usr/bin/env python3
"""One simplified ACCEPTED similarity sheet: summary, domain pairs, task list."""

from __future__ import annotations

import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT.parent / "All_Tasks_LIST/J26.0420/batch_similarity_reports"
INPUT_DIR = ROOT.parent / "All_Tasks_LIST/J26.0420"
TRACKER = ROOT.parent / "Main - Project Terminus Task Tracker - Cognyzer.xlsx"
OUT_XLSX = REPORTS / "ACCEPTED_similarity_one_sheet.xlsx"

FILENAME_PATTERN = re.compile(
    r"^ACCEPTED__Terminus-2nd-Edition__"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$"
)

NL_LABEL = "Non-lexical 70%"
F80_LABEL = "Full 80%"


def load_all_accepted_tasks() -> dict[str, dict]:
    tasks: dict[str, dict] = {}
    for folder in sorted(INPUT_DIR.iterdir()):
        if not folder.is_dir() or not folder.name.startswith("snorkel-ACCEPTED"):
            continue
        for zip_path in sorted(folder.glob("*.zip")):
            m = FILENAME_PATTERN.match(zip_path.name)
            if not m:
                continue
            uuid = m.group(1)
            category = ""
            try:
                with zipfile.ZipFile(zip_path, "r") as zf:
                    if "task.toml" in zf.namelist():
                        raw = zf.read("task.toml").decode("utf-8", errors="replace")
                        for line in raw.splitlines():
                            line = line.strip()
                            if line.startswith("category"):
                                category = line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass
            tasks[uuid] = {"uuid": uuid, "category": category or "unknown"}
    return tasks


def load_tracker() -> dict[str, dict]:
    df = pd.read_excel(TRACKER, sheet_name="May 1st - 31st")
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        uid = str(row.iloc[1]).strip().lower() if pd.notna(row.iloc[1]) else ""
        if not uid or uid == "nan":
            continue
        lookup[uid] = {
            "trainer": str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else "",
            "name": str(row.iloc[3]).strip() if pd.notna(row.iloc[3]) else "",
        }
    return lookup


def parse_basis(reason: str) -> list[str]:
    if not isinstance(reason, str):
        return []
    parts = []
    for token in reason.split(","):
        t = token.strip()
        if "instr_sem" in t:
            parts.append("instruction meaning")
        elif "spec_sem" in t:
            parts.append("SPEC meaning")
        elif "combined_sem" in t:
            parts.append("combined meaning")
        elif "structure" in t:
            parts.append("environment structure")
        elif "instr_lex" in t or "spec_lex" in t:
            parts.append("word overlap")
    return parts


def primary_basis(reasons: list[str]) -> str:
    counts: Counter[str] = Counter()
    for r in reasons:
        for p in parse_basis(r):
            counts[p] += 1
    for key in ("SPEC meaning", "instruction meaning", "combined meaning", "environment structure", "word overlap"):
        if key in counts:
            return key
    return counts.most_common(1)[0][0] if counts else ""


def analyze_pairs(
    pairs_df: pd.DataFrame,
    all_tasks: dict[str, dict],
) -> tuple[dict, Counter, int]:
    involved: dict[str, dict] = defaultdict(lambda: {
        "count": 0,
        "max_score": 0.0,
        "tier": "",
        "reasons": [],
        "partners": [],
        "partner_domains": Counter(),
        "category": "",
    })
    domain_pair_counts: Counter[tuple[str, str]] = Counter()

    if pairs_df.empty:
        return involved, domain_pair_counts, 0

    for _, row in pairs_df.iterrows():
        ua, ub = str(row["Task A UUID"]), str(row["Task B UUID"])
        ca = str(row.get("Task A Category", all_tasks.get(ua, {}).get("category", "unknown")))
        cb = str(row.get("Task B Category", all_tasks.get(ub, {}).get("category", "unknown")))
        domain_pair_counts[tuple(sorted([ca, cb]))] += 1

        score = float(row.get("Max Score %", 0))
        tier = str(row.get("Similarity Tier", ""))
        reason = str(row.get("Flag Reason", ""))
        name_a = str(row.get("Task A Name", ""))
        name_b = str(row.get("Task B Name", ""))

        for uid, partner, pdom, cat in ((ua, name_b, cb, ca), (ub, name_a, ca, cb)):
            d = involved[uid]
            d["count"] += 1
            d["max_score"] = max(d["max_score"], score)
            if tier == "high" or not d["tier"]:
                d["tier"] = tier
            d["reasons"].append(reason)
            if partner and partner != "nan":
                d["partners"].append(partner)
            d["partner_domains"][pdom] += 1
            d["category"] = cat

    return involved, domain_pair_counts, len(involved)


def domain_stats(all_tasks: dict[str, dict], involved: dict, domain_pairs: Counter) -> list[dict]:
    by_cat_total: Counter[str] = Counter(t["category"] for t in all_tasks.values())
    rows = []
    for cat in sorted(by_cat_total):
        counts = [
            involved[uid]["count"]
            for uid in involved
            if all_tasks.get(uid, {}).get("category") == cat
        ]
        within = sum(c for (a, b), c in domain_pairs.items() if a == b == cat)
        cross = sum(c for (a, b), c in domain_pairs.items() if a != b and cat in (a, b))
        sim_n = len(counts)
        rows.append({
            "Category": cat,
            "Total": by_cat_total[cat],
            "Similar Tasks": sim_n,
            "Clean Tasks": by_cat_total[cat] - sim_n,
            "Within-Domain Pairs": within,
            "Cross-Domain Pairs": cross,
            "Avg Similar Partners": round(sum(counts) / len(counts), 1) if counts else 0,
            "Max Similar Partners": max(counts) if counts else 0,
        })
    return rows


def write_section_title(ws, row: int, title: str) -> int:
    ws.cell(row, 1, title).font = Font(bold=True, size=12)
    return row + 2


def write_table(ws, row: int, headers: list[str], data: list[list]) -> int:
    for col, h in enumerate(headers, 1):
        c = ws.cell(row, col, h)
        c.font = Font(bold=True)
    row += 1
    for record in data:
        for col, val in enumerate(record, 1):
            ws.cell(row, col, val)
        row += 1
    return row + 1


def build_one_sheet() -> Path:
    lookup = load_tracker()
    all_tasks = load_all_accepted_tasks()
    nonlex = pd.read_csv(REPORTS / "ACCEPTED_nonlexical_similarity.csv")
    full80 = pd.read_csv(REPORTS / "ACCEPTED_similarity.csv")

    nl_inv, nl_pairs, nl_n = analyze_pairs(nonlex, all_tasks)
    f80_inv, f80_pairs, f80_n = analyze_pairs(full80, all_tasks)

    wb = Workbook()
    ws = wb.active
    ws.title = "ACCEPTED Similarity"

    row = 1
    row = write_section_title(ws, row, "ACCEPTED TASK SIMILARITY — ONE SHEET SUMMARY")

    summary = [
        ["Total ACCEPTED tasks", len(all_tasks)],
        ["Similar tasks (non-lexical 70%)", nl_n],
        ["Similar tasks (full 80% check)", f80_n],
        ["Clean tasks (non-lexical 70%)", len(all_tasks) - nl_n],
        ["Flagged pairs (non-lexical 70%)", len(nonlex)],
        ["Flagged pairs (full 80%)", len(full80)],
        ["Is full 80% check less?", f"Yes — {f80_n} tasks vs {nl_n}; {len(full80)} pairs vs {len(nonlex)}"],
        ["Basis (non-lexical 70%)", "Embedding ≥70% on instruction / SPEC / combined; structure ≥85% + meaning ≥65%. No word overlap."],
        ["Basis (full 80%)", "Any of instruction word, instruction meaning, SPEC word, SPEC meaning, structure, combined ≥80%."],
    ]
    row = write_table(ws, row, ["Metric", "Value"], summary)

    row = write_section_title(ws, row, f"BY DOMAIN — {NL_LABEL} (tasks similar + pairwise pair counts)")
    nl_dom = domain_stats(all_tasks, nl_inv, nl_pairs)
    row = write_table(
        ws, row,
        ["Category", "Total", "Similar", "Clean", "Within-Domain Pairs", "Cross-Domain Pairs", "Avg Partners", "Max Partners"],
        [[d["Category"], d["Total"], d["Similar Tasks"], d["Clean Tasks"], d["Within-Domain Pairs"],
          d["Cross-Domain Pairs"], d["Avg Similar Partners"], d["Max Similar Partners"]] for d in nl_dom],
    )

    row = write_section_title(ws, row, f"BY DOMAIN — {F80_LABEL} (fewer matches for comparison)")
    f80_dom = domain_stats(all_tasks, f80_inv, f80_pairs)
    row = write_table(
        ws, row,
        ["Category", "Total", "Similar", "Clean", "Within-Domain Pairs", "Cross-Domain Pairs", "Avg Partners", "Max Partners"],
        [[d["Category"], d["Total"], d["Similar Tasks"], d["Clean Tasks"], d["Within-Domain Pairs"],
          d["Cross-Domain Pairs"], d["Avg Similar Partners"], d["Max Similar Partners"]] for d in f80_dom],
    )

    row = write_section_title(ws, row, f"DOMAIN PAIRWISE — {NL_LABEL} (similar pair count between domains)")
    pair_rows = []
    cats = sorted({c for pair in nl_pairs for c in pair})
    for a in cats:
        for b in cats:
            if a > b:
                continue
            cnt = nl_pairs.get((a, b), 0)
            if cnt:
                pair_rows.append([a, b, cnt, "same" if a == b else "cross"])
    pair_rows.sort(key=lambda x: -x[2])
    row = write_table(ws, row, ["Domain A", "Domain B", "Similar Pairs", "Type"], pair_rows)

    row = write_section_title(ws, row, f"ALL ACCEPTED TASKS — {NL_LABEL} (trainer + similarity details)")
    task_rows = []
    for uuid, meta in sorted(
        all_tasks.items(),
        key=lambda x: (x[1]["category"], lookup.get(x[0].lower(), {}).get("trainer", "")),
    ):
        tr = lookup.get(uuid.lower(), {})
        info = nl_inv.get(uuid)
        if info:
            basis = primary_basis(info["reasons"])
            top_dom = ", ".join(f"{d}({c})" for d, c in info["partner_domains"].most_common(2))
            task_rows.append([
                "Yes",
                uuid,
                tr.get("trainer", ""),
                tr.get("name", ""),
                meta["category"],
                info["count"],
                round(info["max_score"], 1),
                info["tier"] or "review",
                basis,
                top_dom,
            ])
        else:
            task_rows.append([
                "No",
                uuid,
                tr.get("trainer", ""),
                tr.get("name", ""),
                meta["category"],
                0,
                "",
                "",
                "",
                "",
            ])

    row = write_table(
        ws, row,
        ["Similar?", "Task UUID", "Trainer", "Task Name", "Category", "# Similar Tasks",
         "Max Score %", "Tier", "Flag Basis", "Similar Domains (top 2)"],
        task_rows,
    )

    widths = [10, 38, 16, 36, 28, 14, 12, 10, 22, 36]
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[chr(64 + i) if i <= 26 else "A"].width = w
    from openpyxl.utils import get_column_letter
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    wb.save(OUT_XLSX)
    # Also update the file reviewers already have open.
    alt = OUT_XLSX.parent / "ACCEPTED_domain_pairwise_summary.xlsx"
    wb.save(alt)
    return OUT_XLSX


def main() -> int:
    path = build_one_sheet()
    print(f"Wrote: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
