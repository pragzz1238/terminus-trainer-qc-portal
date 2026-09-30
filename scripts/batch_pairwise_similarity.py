#!/usr/bin/env python3
"""Batch pairwise similarity checker for Terminus task zips.

Modes:
  full (default) — instruction + SPEC.md + environment file-tree + combined embedding
  instruction      — instruction.md only (legacy)

Usage:
    python scripts/batch_pairwise_similarity.py \
        --input-dir ../All_Tasks_LIST/J26.0420 \
        --output-dir ../All_Tasks_LIST/J26.0420/batch_similarity_reports \
        --threshold 80
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from similarity_engine import lexical_similarity, normalize_instruction_text
from tracker_defaults import DEFAULT_EMBED_MODEL

SAME_WORDING_LEXICAL_CUTOFF = 97.0
EMBED_MODEL = DEFAULT_EMBED_MODEL
SPEC_TRUNCATE_CHARS = 15_000
COMBINED_SPEC_CHARS = 8_000

TIER_HIGH = 90.0
TIER_NEAR = 80.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TaskEntry:
    uuid: str
    status: str
    filename: str
    instruction: str
    category: str = ""
    languages: str = ""
    difficulty: str = ""
    instruction_preview: str = ""
    spec_text: str = ""
    spec_preview: str = ""
    spec_path: str = ""
    env_paths: frozenset[str] = field(default_factory=frozenset)
    env_file_count: int = 0
    env_tree_hash: str = ""
    combined_text: str = ""


@dataclass
class SimilarityPair:
    task_a: TaskEntry
    task_b: TaskEntry
    instruction_lexical_pct: float
    instruction_semantic_pct: float
    spec_lexical_pct: float
    spec_semantic_pct: float
    structure_jaccard_pct: float
    combined_semantic_pct: float
    max_score_pct: float
    similarity_tier: str
    flagged_reason: str


# ---------------------------------------------------------------------------
# Phase 1: Extract
# ---------------------------------------------------------------------------

STATUS_FOLDER_PATTERN = re.compile(r"snorkel-(ACCEPTED|NEEDS-REVISION|REJECTED)-\d{4}-\d{2}-\d{2}")
FILENAME_PATTERN = re.compile(
    r"^(ACCEPTED|NEEDS-REVISION|REJECTED)__Terminus-2nd-Edition__"
    r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.zip$"
)


def parse_task_toml(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("difficulty"):
            result["difficulty"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("category"):
            result["category"] = line.split("=", 1)[1].strip().strip('"')
        elif line.startswith("languages"):
            val = line.split("=", 1)[1].strip()
            result["languages"] = val.replace("[", "").replace("]", "").replace('"', "").strip()
    return result


def find_spec_in_zip(names: list[str], zf: zipfile.ZipFile) -> tuple[str, str]:
    """Return (path, text) for the first SPEC.md found in the archive."""
    candidates = sorted(n for n in names if n.endswith("SPEC.md") and not n.endswith("/"))
    if not candidates:
        return "", ""
    path = candidates[0]
    text = zf.read(path).decode("utf-8", errors="replace").strip()
    return path, text[:SPEC_TRUNCATE_CHARS]


def extract_env_paths(names: list[str]) -> frozenset[str]:
    """Normalized environment/ file paths (no directory-only entries)."""
    paths: set[str] = set()
    for name in names:
        if not name.startswith("environment/") or name.endswith("/"):
            continue
        rel = name[len("environment/"):]
        if rel:
            paths.add(rel)
    return frozenset(paths)


def build_combined_text(instruction: str, spec: str) -> str:
    parts = [instruction.strip()]
    if spec.strip():
        parts.append("---SPEC---")
        parts.append(spec.strip()[:COMBINED_SPEC_CHARS])
    return "\n\n".join(parts)


def extract_tasks(input_dir: Path, scope: str) -> list[TaskEntry]:
    tasks: list[TaskEntry] = []
    errors: list[str] = []

    for folder in sorted(input_dir.iterdir()):
        if not folder.is_dir():
            continue
        match = STATUS_FOLDER_PATTERN.match(folder.name)
        if not match:
            continue

        for zip_path in sorted(folder.iterdir()):
            if not zip_path.name.endswith(".zip"):
                continue
            fname_match = FILENAME_PATTERN.match(zip_path.name)
            if not fname_match:
                errors.append(f"Skipping (bad name): {zip_path.name}")
                continue

            file_status = fname_match.group(1)
            uuid = fname_match.group(2)

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
                    spec_path, spec_text = "", ""
                    env_paths = frozenset()
                    if scope == "full":
                        spec_path, spec_text = find_spec_in_zip(names, zf)
                        env_paths = extract_env_paths(names)

                meta = parse_task_toml(toml_raw)
                tree_hash = ""
                if env_paths:
                    tree_hash = hashlib.sha256(
                        "\n".join(sorted(env_paths)).encode()
                    ).hexdigest()[:12]

                tasks.append(TaskEntry(
                    uuid=uuid,
                    status=file_status,
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
                    env_tree_hash=tree_hash if env_paths else "",
                    combined_text=build_combined_text(instruction, spec_text) if scope == "full" else instruction,
                ))
            except Exception as exc:
                errors.append(f"Error reading {zip_path.name}: {exc}")

    if errors:
        print(f"  Extraction warnings ({len(errors)}):")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    return tasks


# ---------------------------------------------------------------------------
# Phase 2: Embeddings
# ---------------------------------------------------------------------------

def load_env_file(env_path: Path) -> None:
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key.startswith("sk-proj-"):
        os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"


def is_same_wording(text_a: str, text_b: str, lexical_pct: float) -> bool:
    if not text_a.strip() or not text_b.strip():
        return False
    if normalize_instruction_text(text_a) == normalize_instruction_text(text_b):
        return True
    return lexical_pct >= SAME_WORDING_LEXICAL_CUTOFF


def embed_texts(texts: list[str], api_key: str, label: str) -> np.ndarray | None:
    if not api_key:
        return None
    try:
        from config import build_openai_client

        client = build_openai_client(api_key)
        embeddings: list[list[float]] = []
        batch_size = 80

        print(f"  Embedding {len(texts)} {label} with {EMBED_MODEL}...")
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            embeddings.extend([item.embedding for item in resp.data])
            print(f"    {label} batch {i // batch_size + 1}: {min(i + batch_size, len(texts))}/{len(texts)}")

        return np.array(embeddings, dtype=np.float64)
    except Exception as exc:
        print(f"  Embedding failed ({label}): {exc}")
        return None


def cosine_matrix_from_embeddings(matrix: np.ndarray) -> np.ndarray:
    if matrix is None or len(matrix) == 0:
        return np.zeros((0, 0))
    return cosine_similarity(matrix)


def structure_jaccard(paths_a: frozenset[str], paths_b: frozenset[str]) -> float:
    if not paths_a and not paths_b:
        return 0.0
    if not paths_a or not paths_b:
        return 0.0
    inter = len(paths_a & paths_b)
    union = len(paths_a | paths_b)
    return inter / union if union else 0.0


def classify_tier(max_pct: float) -> str:
    if max_pct >= TIER_HIGH:
        return "high"
    if max_pct >= TIER_NEAR:
        return "near_similar"
    return "below_threshold"


def build_flag_reason(scores: dict[str, float], threshold: float) -> str:
    thresh = int(threshold)
    parts = []
    for key, pct in sorted(scores.items(), key=lambda x: -x[1]):
        if pct >= threshold:
            parts.append(f"{key}>={thresh}")
    return ", ".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Phase 3: Pairwise comparison
# ---------------------------------------------------------------------------

def compute_pairwise(
    tasks: list[TaskEntry],
    inst_embeddings: np.ndarray | None,
    spec_embeddings: np.ndarray | None,
    combined_embeddings: np.ndarray | None,
    threshold: float,
    scope: str,
) -> tuple[list[SimilarityPair], int]:
    n = len(tasks)
    total_pairs = n * (n - 1) // 2
    print(f"  Computing {total_pairs:,} pairwise comparisons (scope={scope})...")
    print(f"  Skipping copy-paste instruction pairs (lexical >= {SAME_WORDING_LEXICAL_CUTOFF:.0f}%)")

    inst_cos = cosine_matrix_from_embeddings(inst_embeddings) if inst_embeddings is not None else None
    spec_cos = cosine_matrix_from_embeddings(spec_embeddings) if spec_embeddings is not None else None
    combined_cos = cosine_matrix_from_embeddings(combined_embeddings) if combined_embeddings is not None else None

    flagged: list[SimilarityPair] = []
    skipped_copy_paste = 0
    progress_step = max(1, total_pairs // 20)
    count = 0

    for i, j in combinations(range(n), 2):
        count += 1
        if count % progress_step == 0:
            print(
                f"    Progress: {count:,}/{total_pairs:,} checked, "
                f"{len(flagged)} flagged, {skipped_copy_paste} copy-paste skipped"
            )

        inst_lex = round(lexical_similarity(tasks[i].instruction, tasks[j].instruction) * 100, 1)
        if is_same_wording(tasks[i].instruction, tasks[j].instruction, inst_lex):
            skipped_copy_paste += 1
            continue

        inst_sem = round(float(inst_cos[i, j]) * 100, 1) if inst_cos is not None else 0.0

        spec_lex = 0.0
        spec_sem = 0.0
        if scope == "full" and tasks[i].spec_text and tasks[j].spec_text:
            spec_lex = round(lexical_similarity(tasks[i].spec_text, tasks[j].spec_text) * 100, 1)
            if spec_cos is not None:
                spec_sem = round(float(spec_cos[i, j]) * 100, 1)

        struct_pct = 0.0
        if scope == "full":
            struct_pct = round(structure_jaccard(tasks[i].env_paths, tasks[j].env_paths) * 100, 1)

        combined_sem = 0.0
        if combined_cos is not None:
            combined_sem = round(float(combined_cos[i, j]) * 100, 1)

        score_map = {
            "instruction_lexical": inst_lex,
            "instruction_semantic": inst_sem,
            "combined_semantic": combined_sem,
        }
        if scope == "full":
            score_map["spec_lexical"] = spec_lex
            score_map["spec_semantic"] = spec_sem
            score_map["structure_jaccard"] = struct_pct

        max_pct = max(score_map.values())
        if max_pct < threshold:
            continue

        flagged.append(SimilarityPair(
            task_a=tasks[i],
            task_b=tasks[j],
            instruction_lexical_pct=inst_lex,
            instruction_semantic_pct=inst_sem,
            spec_lexical_pct=spec_lex,
            spec_semantic_pct=spec_sem,
            structure_jaccard_pct=struct_pct,
            combined_semantic_pct=combined_sem,
            max_score_pct=max_pct,
            similarity_tier=classify_tier(max_pct),
            flagged_reason=build_flag_reason(score_map, threshold),
        ))

    print(
        f"  Done: {len(flagged)} pairs flagged, {skipped_copy_paste} copy-paste skipped "
        f"out of {total_pairs:,} total"
    )
    return flagged, skipped_copy_paste


# ---------------------------------------------------------------------------
# Phase 4: Write CSVs
# ---------------------------------------------------------------------------

CSV_COLUMNS_FULL = [
    "similarity_tier",
    "max_score_pct",
    "flagged_reason",
    "task_a_status",
    "task_a_uuid",
    "task_a_category",
    "task_a_languages",
    "task_a_difficulty",
    "task_a_env_files",
    "task_a_has_spec",
    "task_a_instruction_preview",
    "task_b_status",
    "task_b_uuid",
    "task_b_category",
    "task_b_languages",
    "task_b_difficulty",
    "task_b_env_files",
    "task_b_has_spec",
    "task_b_instruction_preview",
    "instruction_lexical_pct",
    "instruction_semantic_pct",
    "spec_lexical_pct",
    "spec_semantic_pct",
    "structure_jaccard_pct",
    "combined_semantic_pct",
]

CSV_COLUMNS_INSTRUCTION = [
    "similarity_tier",
    "max_score_pct",
    "flagged_reason",
    "task_a_status",
    "task_a_uuid",
    "task_a_category",
    "task_a_languages",
    "task_a_difficulty",
    "task_a_instruction_preview",
    "task_b_status",
    "task_b_uuid",
    "task_b_category",
    "task_b_languages",
    "task_b_difficulty",
    "task_b_instruction_preview",
    "instruction_lexical_pct",
    "instruction_semantic_pct",
]


def pair_to_row(pair: SimilarityPair, scope: str) -> dict[str, Any]:
    row: dict[str, Any] = {
        "similarity_tier": pair.similarity_tier,
        "max_score_pct": pair.max_score_pct,
        "flagged_reason": pair.flagged_reason,
        "task_a_status": pair.task_a.status,
        "task_a_uuid": pair.task_a.uuid,
        "task_a_category": pair.task_a.category,
        "task_a_languages": pair.task_a.languages,
        "task_a_difficulty": pair.task_a.difficulty,
        "task_a_instruction_preview": pair.task_a.instruction_preview,
        "task_b_status": pair.task_b.status,
        "task_b_uuid": pair.task_b.uuid,
        "task_b_category": pair.task_b.category,
        "task_b_languages": pair.task_b.languages,
        "task_b_difficulty": pair.task_b.difficulty,
        "task_b_instruction_preview": pair.task_b.instruction_preview,
        "instruction_lexical_pct": pair.instruction_lexical_pct,
        "instruction_semantic_pct": pair.instruction_semantic_pct,
    }
    if scope == "full":
        row.update({
            "task_a_env_files": pair.task_a.env_file_count,
            "task_a_has_spec": "yes" if pair.task_a.spec_text else "no",
            "task_b_env_files": pair.task_b.env_file_count,
            "task_b_has_spec": "yes" if pair.task_b.spec_text else "no",
            "spec_lexical_pct": pair.spec_lexical_pct,
            "spec_semantic_pct": pair.spec_semantic_pct,
            "structure_jaccard_pct": pair.structure_jaccard_pct,
            "combined_semantic_pct": pair.combined_semantic_pct,
        })
    else:
        row["max_score_pct"] = max(pair.instruction_lexical_pct, pair.instruction_semantic_pct)
    return row


def write_csv(pairs: list[SimilarityPair], output_path: Path, scope: str) -> None:
    columns = CSV_COLUMNS_FULL if scope == "full" else CSV_COLUMNS_INSTRUCTION
    pairs_sorted = sorted(pairs, key=lambda p: p.max_score_pct, reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for pair in pairs_sorted:
            writer.writerow(pair_to_row(pair, scope))

    print(f"  Wrote {len(pairs_sorted)} rows -> {output_path.name}")


def print_summary(pairs: list[SimilarityPair], tasks: list[TaskEntry], scope: str) -> None:
    tiers = {"high": 0, "near_similar": 0}
    for p in pairs:
        tiers[p.similarity_tier] = tiers.get(p.similarity_tier, 0) + 1

    with_spec = sum(1 for t in tasks if t.spec_text)
    print()
    print("Summary")
    print(f"  Tasks analyzed: {len(tasks)}")
    if scope == "full":
        print(f"  Tasks with SPEC.md: {with_spec}")
        print(f"  Avg environment files: {sum(t.env_file_count for t in tasks) / len(tasks):.1f}")
    print(f"  Flagged pairs (>= {TIER_NEAR:.0f}%): {len(pairs)}")
    print(f"    high (>= {TIER_HIGH:.0f}%): {tiers.get('high', 0)}")
    print(f"    near_similar ({TIER_NEAR:.0f}-{TIER_HIGH:.0f}%): {tiers.get('near_similar', 0)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Batch pairwise similarity checker for Terminus tasks")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=80.0)
    parser.add_argument(
        "--scope",
        choices=["full", "instruction"],
        default="full",
        help="full = instruction + SPEC + env structure + combined embedding (default)",
    )
    args = parser.parse_args()

    input_dir: Path = args.input_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    threshold: float = args.threshold
    scope: str = args.scope

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}")
        return 1

    load_env_file(ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    print("Batch Pairwise Similarity Checker")
    print(f"  Input:       {input_dir}")
    print(f"  Output:      {output_dir}")
    print(f"  Scope:       {scope}")
    print(f"  Threshold:   {threshold}%")
    print(f"  Embed model: {EMBED_MODEL}")
    print(f"  Tiers:       high >= {TIER_HIGH:.0f}%, near_similar >= {TIER_NEAR:.0f}%")
    print(f"  API key:     {'present' if api_key else 'MISSING'}")
    print()

    print("[Phase 1] Extracting from zip archives...")
    tasks = extract_tasks(input_dir, scope)
    if not tasks:
        print("ERROR: No tasks extracted.")
        return 1

    status_counts: dict[str, int] = {}
    for t in tasks:
        status_counts[t.status] = status_counts.get(t.status, 0) + 1
    print(f"  Extracted {len(tasks)} tasks: {status_counts}")
    if scope == "full":
        with_spec = sum(1 for t in tasks if t.spec_text)
        print(f"  With SPEC.md: {with_spec}/{len(tasks)}")
    print()

    print("[Phase 2] Computing embeddings...")
    inst_emb = embed_texts([t.instruction for t in tasks], api_key, "instructions")
    spec_emb = None
    combined_emb = None
    if scope == "full":
        spec_emb = embed_texts(
            [t.spec_text if t.spec_text else " " for t in tasks],
            api_key,
            "specs",
        )
        combined_emb = embed_texts([t.combined_text for t in tasks], api_key, "combined")

    if inst_emb is None:
        print("ERROR: Embeddings required. Check OPENAI_API_KEY in .env")
        return 1
    print(f"  Instruction embeddings: {inst_emb.shape}")
    if combined_emb is not None:
        print(f"  Combined embeddings: {combined_emb.shape}")
    print()

    print("[Phase 3] Pairwise similarity...")
    all_pairs, _ = compute_pairwise(
        tasks, inst_emb, spec_emb, combined_emb, threshold, scope,
    )
    print()

    print("[Phase 4] Writing CSV reports...")
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "ALL_combined" if scope == "full" else "ALL_combined_instruction"

    write_csv(all_pairs, output_dir / f"{prefix}_similarity.csv", scope)

    for status_key in ["ACCEPTED", "NEEDS-REVISION", "REJECTED"]:
        status_pairs = [
            p for p in all_pairs
            if p.task_a.status == status_key and p.task_b.status == status_key
        ]
        if not status_pairs:
            continue
        csv_name = status_key.replace("-", "_") + "_similarity.csv"
        write_csv(status_pairs, output_dir / csv_name, scope)

    print_summary(all_pairs, tasks, scope)
    print()
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
