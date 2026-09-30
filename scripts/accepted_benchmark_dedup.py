#!/usr/bin/env python3
"""
ACCEPTED Task Similarity — Industry Benchmark Standard (LLM Decontaminator method)

Methodology (references):
  - "Rethinking Benchmark and Contamination for Language Models with Rephrased Samples"
    (lm-sys, arXiv:2311.04850) — THE gold standard for benchmark dedup, F1=0.92
  - "Soft Contamination Means Benchmarks Test Shallow Generalization" (arXiv:2602.12413)
    — cosine > 0.80 → 60-85% true semantic duplicate
  - Open-Platypus (Lee et al. 2023) — cosine threshold 0.80 for SFT decontamination

Two-step pipeline:
  Step 1: Embedding similarity (text-embedding-3-small) — screen candidates at cosine >= 0.80
           on FULL TASK (instruction + SPEC combined).
  Step 2: LLM judge (GPT-4o) — for each candidate pair, ask:
           "Are these two benchmark tasks testing the SAME underlying problem?"
           Only pairs confirmed by the LLM count as "similar".

This gives one clean number: out of 125, how many are similar.
Output: 2-sheet Excel (Summary + Task List).
"""

from __future__ import annotations

import argparse
import os
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "All_Tasks_LIST/J26.0420"
DEFAULT_TRACKER = REPO_ROOT / "Main - Project Terminus Task Tracker - Cognyzer.xlsx"
DEFAULT_OUTPUT = DEFAULT_INPUT / "batch_similarity_reports" / "ACCEPTED_benchmark_dedup.xlsx"

EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
COSINE_SCREEN_THRESHOLD = 0.80  # Open-Platypus standard
BATCH_SIZE = 80

FILENAME_PATTERN = re.compile(
    r"^ACCEPTED__Terminus-2nd-Edition__"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$"
)
UUID_PATTERN = re.compile(
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})"
)

LLM_JUDGE_PROMPT = """You are a benchmark task deduplication judge. You will be given two coding benchmark tasks (instruction + specification). 

Determine if these two tasks are testing the SAME underlying problem — meaning an agent solving one would essentially be solving the other, even if surface details (variable names, domain framing) differ.

Answer ONLY "SAME" or "DIFFERENT".

SAME means: same algorithmic challenge, same input/output structure, same core logic — just re-skinned.
DIFFERENT means: genuinely distinct problems requiring different algorithms or domain knowledge.

=== TASK A ===
{task_a}

=== TASK B ===
{task_b}

Your verdict (SAME or DIFFERENT):"""


@dataclass
class Task:
    uuid: str
    instruction: str
    spec: str
    combined: str
    category: str
    trainer: str
    name: str
    batch: str = ""
    source: str = "snorkel-ACCEPTED"


def load_env(env_path: Path | None = None) -> None:
    candidates = [env_path] if env_path else []
    candidates.extend([Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"])
    for path in candidates:
        if path and path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")
            return


def load_tracker(tracker_path: Path | None) -> dict[str, dict]:
    if not tracker_path or not tracker_path.exists():
        return {}
    df = pd.read_excel(tracker_path, sheet_name="May 1st - 31st")
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


def _load_task_from_zip(zip_path: Path, lookup: dict[str, dict], batch: str) -> Task | None:
    m = FILENAME_PATTERN.match(zip_path.name)
    if not m:
        return None
    uuid = m.group(1)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            if "instruction.md" not in names:
                return None
            instruction = zf.read("instruction.md").decode("utf-8", errors="replace").strip()
            spec = ""
            for candidate in names:
                if "spec" in candidate.lower() and candidate.endswith(".md"):
                    spec = zf.read(candidate).decode("utf-8", errors="replace").strip()
                    break
            category = ""
            if "task.toml" in names:
                raw = zf.read("task.toml").decode("utf-8", errors="replace")
                for line in raw.splitlines():
                    if line.strip().startswith("category"):
                        category = line.split("=", 1)[1].strip().strip('"').strip("'")
                        break
        tr = lookup.get(uuid.lower(), {})
        combined = instruction + "\n\n" + spec if spec else instruction
        return Task(
            uuid=uuid,
            instruction=instruction,
            spec=spec,
            combined=combined,
            category=category,
            trainer=tr.get("trainer", ""),
            name=tr.get("name", ""),
            batch=batch,
            source="snorkel-ACCEPTED",
        )
    except Exception:
        return None


def _load_task_from_dir(task_dir: Path, lookup: dict[str, dict], batch: str) -> Task | None:
    inst_path = task_dir / "instruction.md"
    if not inst_path.exists():
        return None
    m = UUID_PATTERN.search(task_dir.name)
    if not m:
        return None
    uuid = m.group(1)
    try:
        instruction = inst_path.read_text(encoding="utf-8", errors="replace").strip()
        spec = ""
        for spec_path in sorted(task_dir.rglob("SPEC.md")):
            spec = spec_path.read_text(encoding="utf-8", errors="replace").strip()
            break
        category = ""
        toml_path = task_dir / "task.toml"
        if toml_path.exists():
            for line in toml_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.strip().startswith("category"):
                    category = line.split("=", 1)[1].strip().strip('"').strip("'")
                    break
        tr = lookup.get(uuid.lower(), {})
        combined = instruction + "\n\n" + spec if spec else instruction
        return Task(
            uuid=uuid,
            instruction=instruction,
            spec=spec,
            combined=combined,
            category=category,
            trainer=tr.get("trainer", ""),
            name=tr.get("name", ""),
            batch=batch,
            source="completed_state",
        )
    except Exception:
        return None


def _collect_from_batch_dir(batch_dir: Path, lookup: dict[str, dict], seen_uuids: set[str]) -> list[Task]:
    tasks: list[Task] = []
    batch_name = batch_dir.name

    for folder in sorted(batch_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("snorkel-ACCEPTED"):
            for zip_path in sorted(folder.glob("*.zip")):
                task = _load_task_from_zip(zip_path, lookup, batch_name)
                if task and task.uuid not in seen_uuids:
                    seen_uuids.add(task.uuid)
                    tasks.append(task)

    for folder in sorted(batch_dir.iterdir()):
        if folder.is_dir() and folder.name.startswith("completed_state"):
            task = _load_task_from_dir(folder, lookup, batch_name)
            if task and task.uuid not in seen_uuids:
                seen_uuids.add(task.uuid)
                tasks.append(task)

    return tasks


def extract_tasks(input_dir: Path, lookup: dict[str, dict], cross_batch: bool = False) -> list[Task]:
    tasks: list[Task] = []
    seen_uuids: set[str] = set()

    if cross_batch:
        for batch_dir in sorted(input_dir.iterdir()):
            if not batch_dir.is_dir() or not batch_dir.name.startswith("J26."):
                continue
            tasks.extend(_collect_from_batch_dir(batch_dir, lookup, seen_uuids))
        return tasks

    batch_name = input_dir.name if input_dir.name.startswith("J26.") else input_dir.name
    tasks.extend(_collect_from_batch_dir(input_dir, lookup, seen_uuids))
    return tasks


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    MAX_CHARS = 8000
    truncated = [t[:MAX_CHARS] if len(t) > MAX_CHARS else t for t in texts]
    truncated = [t if t.strip() else "empty" for t in truncated]
    all_embs = []
    for i in range(0, len(truncated), BATCH_SIZE):
        batch = truncated[i:i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        all_embs.extend([item.embedding for item in resp.data])
        print(f"    Embedded batch {i//BATCH_SIZE + 1}")
    return np.array(all_embs, dtype=np.float64)


def cosine_matrix(embs: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_similarity
    mat = cosine_similarity(embs)
    np.fill_diagonal(mat, 0)
    return mat


def truncate_for_judge(text: str, max_chars: int = 2000) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated...]"


def llm_judge(client: OpenAI, task_a: Task, task_b: Task) -> str:
    """Ask GPT-4o if two tasks are the SAME or DIFFERENT."""
    text_a = truncate_for_judge(task_a.combined)
    text_b = truncate_for_judge(task_b.combined)
    prompt = LLM_JUDGE_PROMPT.format(task_a=text_a, task_b=text_b)
    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip().upper()
        if "SAME" in answer:
            return "SAME"
        return "DIFFERENT"
    except Exception as e:
        print(f"    LLM judge error: {e}")
        return "ERROR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ACCEPTED task similarity — LLM Decontaminator method (embedding + GPT judge)",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help="Folder containing snorkel-ACCEPTED-* subfolder(s) with task zips",
    )
    parser.add_argument(
        "--tracker",
        type=Path,
        default=DEFAULT_TRACKER,
        help="Terminus tracker xlsx for trainer/task names (optional)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output xlsx path",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Path to .env with OPENAI_API_KEY (optional)",
    )
    parser.add_argument(
        "--cross-batch",
        action="store_true",
        help="Scan all J26.*/snorkel-ACCEPTED-* subfolders under input-dir (full corpus)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_env(args.env_file)
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY required (set env var or pass --env-file)")
        return 1

    client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
    lookup = load_tracker(args.tracker.resolve() if args.tracker else None)

    print("=" * 60)
    print("ACCEPTED Task Similarity — Benchmark Standard (LLM Decontaminator)")
    print("=" * 60)
    print(f"Method: Two-step (embedding screen + LLM judge)")
    print(f"  Step 1: Cosine >= {COSINE_SCREEN_THRESHOLD} on full task (instruction+SPEC)")
    print(f"  Step 2: {LLM_MODEL} judges each candidate pair")
    print(f"  Reference: lm-sys/llm-decontaminator (arXiv:2311.04850, F1=0.92)")
    print()

    input_dir = args.input_dir.resolve()
    out_xlsx = args.output.resolve()
    if not input_dir.is_dir():
        print(f"ERROR: input dir not found: {input_dir}")
        return 1

    mode = "CROSS-BATCH (all J26.* folders)" if args.cross_batch else "SINGLE BATCH"
    print(f"Mode: {mode}")
    print()

    print("[1/4] Extracting ACCEPTED tasks...")
    tasks = extract_tasks(input_dir, lookup, cross_batch=args.cross_batch)
    if args.cross_batch:
        from collections import Counter as _Counter
        by_batch = _Counter(t.batch for t in tasks)
        for b, n in sorted(by_batch.items()):
            print(f"    {b}: {n} tasks")
    from collections import Counter as _Counter
    by_source = _Counter(t.source for t in tasks)
    for src, n in sorted(by_source.items()):
        print(f"    {src}: {n} tasks")
    print(f"  {len(tasks)} tasks loaded total")

    print("[2/4] Embedding full task text...")
    embs = embed_texts(client, [t.combined for t in tasks])
    print(f"  Embedded {len(tasks)} tasks")

    print("[3/4] Cosine screening (>= {:.0f}%)...".format(COSINE_SCREEN_THRESHOLD * 100))
    cos_mat = cosine_matrix(embs)
    candidates: list[tuple[int, int, float]] = []
    for i, j in combinations(range(len(tasks)), 2):
        score = float(cos_mat[i, j])
        if score >= COSINE_SCREEN_THRESHOLD:
            candidates.append((i, j, score))
    candidates.sort(key=lambda x: -x[2])
    print(f"  {len(candidates)} candidate pairs above {COSINE_SCREEN_THRESHOLD*100:.0f}%")

    print(f"[4/4] LLM judge ({LLM_MODEL}) on {len(candidates)} pairs...")
    confirmed: list[dict] = []
    different: list[dict] = []
    for idx, (i, j, score) in enumerate(candidates):
        verdict = llm_judge(client, tasks[i], tasks[j])
        pair_info = {
            "task_a_uuid": tasks[i].uuid,
            "task_b_uuid": tasks[j].uuid,
            "task_a_batch": tasks[i].batch,
            "task_b_batch": tasks[j].batch,
            "task_a_source": tasks[i].source,
            "task_b_source": tasks[j].source,
            "task_a_trainer": tasks[i].trainer,
            "task_b_trainer": tasks[j].trainer,
            "task_a_name": tasks[i].name,
            "task_b_name": tasks[j].name,
            "task_a_category": tasks[i].category,
            "task_b_category": tasks[j].category,
            "cross_batch": tasks[i].batch != tasks[j].batch,
            "cosine_pct": round(score * 100, 1),
            "llm_verdict": verdict,
        }
        if verdict == "SAME":
            confirmed.append(pair_info)
        else:
            different.append(pair_info)
        if (idx + 1) % 20 == 0:
            print(f"    Judged {idx+1}/{len(candidates)} — confirmed SAME so far: {len(confirmed)}")
        time.sleep(0.1)

    similar_uuids = set()
    for p in confirmed:
        similar_uuids.add(p["task_a_uuid"])
        similar_uuids.add(p["task_b_uuid"])

    clean_count = len(tasks) - len(similar_uuids)
    print()
    print("=" * 60)
    print(f"RESULT: {len(similar_uuids)} / {len(tasks)} ACCEPTED tasks are similar")
    print(f"  Clean (unique): {clean_count}")
    print(f"  Confirmed pairs: {len(confirmed)}")
    print(f"  Rejected by LLM: {len(different)}")
    print("=" * 60)

    # Build Excel — 2 sheets only
    # Sheet 1: Summary + all tasks
    task_rows = []
    for t in sorted(tasks, key=lambda x: (x.category, x.trainer)):
        is_similar = t.uuid in similar_uuids
        pair_count = sum(1 for p in confirmed if p["task_a_uuid"] == t.uuid or p["task_b_uuid"] == t.uuid)
        task_rows.append({
            "Similar?": "Yes" if is_similar else "No",
            "Batch": t.batch,
            "Source": t.source,
            "Task UUID": t.uuid,
            "Trainer": t.trainer,
            "Task Name": t.name,
            "Category": t.category,
            "# Confirmed Pairs": pair_count if is_similar else 0,
        })

    tasks_df = pd.DataFrame(task_rows)

    # Sheet 2: Confirmed pairs
    within_batch = sum(1 for p in confirmed if not p.get("cross_batch"))
    cross_batch_pairs = sum(1 for p in confirmed if p.get("cross_batch"))

    pairs_df = pd.DataFrame(confirmed) if confirmed else pd.DataFrame(
        columns=["task_a_uuid", "task_b_uuid", "task_a_batch", "task_b_batch",
                 "task_a_source", "task_b_source", "task_a_trainer", "task_b_trainer",
                 "task_a_name", "task_b_name", "task_a_category", "task_b_category",
                 "cross_batch", "cosine_pct", "llm_verdict"]
    )
    pairs_df = pairs_df.rename(columns={
        "task_a_uuid": "Task A UUID", "task_b_uuid": "Task B UUID",
        "task_a_batch": "Task A Batch", "task_b_batch": "Task B Batch",
        "task_a_source": "Task A Source", "task_b_source": "Task B Source",
        "task_a_trainer": "Task A Trainer", "task_b_trainer": "Task B Trainer",
        "task_a_name": "Task A Name", "task_b_name": "Task B Name",
        "task_a_category": "Task A Category", "task_b_category": "Task B Category",
        "cross_batch": "Cross-Batch?", "cosine_pct": "Cosine %", "llm_verdict": "LLM Verdict",
    })

    # Summary stats
    by_trainer = Counter(t.trainer for t in tasks if t.uuid in similar_uuids)
    summary_rows = [
        {"Metric": "Method", "Value": "LLM Decontaminator (embedding + GPT judge)"},
        {"Metric": "Reference", "Value": "lm-sys arXiv:2311.04850 (F1=0.92)"},
        {"Metric": "Embedding model", "Value": EMBED_MODEL},
        {"Metric": "Screen threshold", "Value": f"cosine >= {COSINE_SCREEN_THRESHOLD}"},
        {"Metric": "LLM judge", "Value": LLM_MODEL},
        {"Metric": "", "Value": ""},
        {"Metric": "Total ACCEPTED tasks", "Value": len(tasks)},
        {"Metric": "SIMILAR (confirmed by LLM)", "Value": len(similar_uuids)},
        {"Metric": "CLEAN (unique)", "Value": clean_count},
        {"Metric": "Confirmed same-task pairs", "Value": len(confirmed)},
        {"Metric": "  Within-batch pairs", "Value": within_batch},
        {"Metric": "  Cross-batch pairs", "Value": cross_batch_pairs},
        {"Metric": "Screened but rejected (different)", "Value": len(different)},
        {"Metric": "", "Value": ""},
        {"Metric": "By trainer (similar tasks):", "Value": ""},
    ]
    for trainer, cnt in by_trainer.most_common():
        summary_rows.append({"Metric": f"  {trainer}", "Value": cnt})

    summary_df = pd.DataFrame(summary_rows)

    out_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        # Sheet 1: Summary + Task List combined
        blank = pd.DataFrame([{"Metric": "", "Value": ""}])
        header2 = pd.DataFrame([{"Metric": "ALL TASKS:", "Value": ""}])
        combined = pd.concat([summary_df, blank, header2], ignore_index=True)
        combined.to_excel(writer, sheet_name="Summary & Tasks", index=False, startrow=0)
        tasks_df.to_excel(writer, sheet_name="Summary & Tasks", index=False,
                          startrow=len(combined) + 2)
        # Sheet 2: Confirmed pairs
        pairs_df.to_excel(writer, sheet_name="Similar Pairs", index=False)

    print(f"\nWrote: {out_xlsx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
