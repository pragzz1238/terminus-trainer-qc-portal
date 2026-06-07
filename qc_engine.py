"""Terminus Trainer QC engine — static checks, similarity, LLM judge, report export."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from alignment_prompts import (
    ALIGNMENT_CHECKS,
    ALIGNMENT_LABELS,
    ACCEPTED_TASK_NAMES,
)
from config import (
    api_provider_label,
    build_openai_client,
    resolve_embed_model,
    resolve_llm_model,
    resolve_openai_api_key,
)

VALID_SUBCATS = {
    "api_integration",
    "db_interaction",
    "long_context",
    "tool_specific",
    "ui_building",
}
LEAKAGE_PATTERN = re.compile(r"solve\.sh|solution/|/solution|/sol\b|oracle", re.IGNORECASE)
PLACEHOLDER_PATTERN = re.compile(r"SET_AFTER|PLACEHOLDER|TODO_HASH|FIXME", re.IGNORECASE)
LARGE_IMAGE_PATTERN = re.compile(
    r"^FROM\s+(gcc|golang|node|ruby|python(?!.*slim)):\S*-bookworm",
    re.MULTILINE | re.IGNORECASE,
)

from tracker_defaults import (
    INSTRUCTION_SIM_THRESHOLD,
    INSTRUCTION_SIM_BLOCK,
    INSTRUCTION_SIM_WARN,
    TASK_INSTRUCTION_HEADER,
    TRACKER_COL_TASK_INSTRUCTION,
)
SIM_THRESHOLD_WARN = INSTRUCTION_SIM_WARN
SIM_THRESHOLD_BLOCK = INSTRUCTION_SIM_BLOCK
CHANGE_TASK_MESSAGE = (
    "Change the task — instruction is too similar to an existing task "
    "(both lexical and embedding checks are at or above 60%)."
)


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


@dataclass
class QCReport:
    task_name: str
    trainer_name: str
    timestamp: str
    overall_pass: bool
    static_pass: bool
    similarity_pass: bool
    llm_pass: bool | None
    static_issues: list[Issue] = field(default_factory=list)
    instruction_matches: list[SimilarityMatch] = field(default_factory=list)
    spec_matches: list[SimilarityMatch] = field(default_factory=list)
    max_similarity: float = 0.0
    instruction_blocked: bool = False
    instruction_block_message: str = ""
    llm_results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def extract_task_dir(root: Path) -> Path:
    """Flatten single nested folder from zip extraction."""
    entries = [p for p in root.iterdir() if p.name not in {".DS_Store", "__MACOSX"}]
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            target = root / item.name
            if target.exists():
                if target.is_dir():
                    for sub in target.iterdir():
                        sub.rename(root / sub.name)
                    target.rmdir()
                else:
                    target.unlink()
            item.rename(target)
        nested.rmdir()
    return root


def unzip_task(zip_bytes: bytes, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
        zf.extractall(dest)
    return extract_task_dir(dest)


def detect_task_name(task_dir: Path, zip_name: str = "") -> str:
    if zip_name:
        stem = Path(zip_name).stem.strip()
        if stem:
            return stem
    return task_dir.name or "task"


def run_static_checks(task_dir: Path) -> list[Issue]:
    """Port of local-qc/run_qc.py checks."""
    issues: list[Issue] = []
    task_dir = Path(task_dir)

    required = {
        "task.toml": "Task metadata file",
        "instruction.md": "Agent instructions",
        "environment/Dockerfile": "Docker build file",
        "solution/solve.sh": "Oracle solution",
        "tests/test.sh": "Test runner",
        "tests/test_outputs.py": "Test assertions",
    }
    for relpath, desc in required.items():
        if not (task_dir / relpath).is_file():
            issues.append(Issue("CRITICAL", f"Missing {relpath} ({desc})"))

    toml_path = task_dir / "task.toml"
    toml = toml_path.read_text() if toml_path.is_file() else ""

    if toml:
        for i, line in enumerate(toml.splitlines(), 1):
            if line.strip().startswith("#"):
                issues.append(
                    Issue(
                        "HIGH",
                        f"task.toml has comment on line {i}",
                        "Remove all comments — task.toml must have ZERO comments",
                    )
                )
                break

        if re.search(r"^name\s*=", toml, re.MULTILINE):
            issues.append(
                Issue(
                    "HIGH",
                    "task.toml has non-standard 'name' field",
                    "Remove the name = line — it causes AutoEval BUILD FAILURE",
                )
            )

        if "allow_internet" not in toml:
            issues.append(
                Issue(
                    "CRITICAL",
                    "task.toml missing allow_internet",
                    "Add allow_internet = false under [environment]",
                )
            )
        elif "allow_internet = true" in toml:
            issues.append(Issue("CRITICAL", "allow_internet = true is blocked"))

        if 'codebase_size = "minimal"' in toml:
            issues.append(
                Issue(
                    "CRITICAL",
                    "codebase_size = 'minimal' is blocked",
                    "Use 'small' (>=20 files) or 'large'",
                )
            )

        subcat_match = re.search(r"subcategories\s*=\s*\[(.*?)\]", toml, re.DOTALL)
        if subcat_match and subcat_match.group(1).strip():
            subcats = re.findall(r'"([^"]+)"', subcat_match.group(1))
            for sc in subcats:
                if sc not in VALID_SUBCATS:
                    issues.append(
                        Issue(
                            "HIGH",
                            f"Invalid subcategory: '{sc}'",
                            f"Valid: {sorted(VALID_SUBCATS)}. Use subcategories = [] if none fit.",
                        )
                    )

        if '"python"' in toml and 'difficulty = "hard"' not in toml:
            issues.append(Issue("HIGH", "Python task must have difficulty = 'hard'"))

        if 'difficulty = "easy"' in toml:
            issues.append(Issue("HIGH", "difficulty = 'easy' is not accepted on the platform"))

        timeout_match = re.search(
            r"timeout_sec\s*=\s*([\d.]+)",
            toml.split("[agent]")[-1] if "[agent]" in toml else "",
        )
        if timeout_match:
            timeout_val = float(timeout_match.group(1))
            if timeout_val > 1800:
                issues.append(
                    Issue(
                        "CRITICAL",
                        f"agent.timeout_sec = {timeout_val} exceeds maximum (1800)",
                    )
                )
            elif timeout_val < 1:
                issues.append(
                    Issue(
                        "CRITICAL",
                        f"agent.timeout_sec = {timeout_val} below minimum (1)",
                    )
                )
            elif timeout_val != 1800:
                issues.append(
                    Issue(
                        "MEDIUM",
                        f"agent.timeout_sec = {timeout_val} (recommended: 1800)",
                    )
                )
        elif "[agent]" not in toml:
            issues.append(
                Issue(
                    "HIGH",
                    "task.toml missing [agent] section with timeout_sec",
                    "Add [agent] with timeout_sec = 1800.0",
                )
            )

        if re.search(r"^workdir\s*=", toml, re.MULTILINE):
            issues.append(
                Issue("MEDIUM", "task.toml has non-standard workdir field", "Remove it")
            )

        for bad_field in ("docker_flags", "gpus", "gpu_types"):
            if re.search(rf"^{bad_field}\s*=", toml, re.MULTILINE):
                issues.append(
                    Issue("MEDIUM", f"task.toml has non-standard {bad_field} field")
                )

    dockerfile_path = task_dir / "environment" / "Dockerfile"
    if dockerfile_path.is_file():
        dockerfile = dockerfile_path.read_text()
        for line in dockerfile.splitlines():
            stripped = line.strip()
            if stripped.startswith("FROM") and "@sha256:" not in stripped:
                issues.append(
                    Issue(
                        "CRITICAL",
                        f"Unpinned base image: {stripped}",
                        "Add @sha256:<digest> to pin the image",
                    )
                )

        if re.search(r"COPY\s+tests/", dockerfile):
            issues.append(
                Issue("CRITICAL", "Dockerfile has COPY tests/ (tests are mounted at runtime)")
            )
        if re.search(r"COPY\s+solution/", dockerfile):
            issues.append(
                Issue("CRITICAL", "Dockerfile has COPY solution/ (solution must never be in image)")
            )

        if LARGE_IMAGE_PATTERN.search(dockerfile):
            issues.append(
                Issue(
                    "HIGH",
                    "Dockerfile uses a large bookworm base image (risk of 7200s build timeout)",
                    "Use debian:bookworm-slim or python:*-slim and install tools explicitly",
                )
            )

        if toml and "allow_internet = false" in toml and "pytest" not in dockerfile:
            issues.append(
                Issue(
                    "HIGH",
                    "allow_internet=false but pytest not installed in Dockerfile",
                    "Bake pytest into the Dockerfile",
                )
            )

    for check_file in ("instruction.md", "tests/test_outputs.py"):
        fpath = task_dir / check_file
        if fpath.is_file():
            matches = LEAKAGE_PATTERN.findall(fpath.read_text())
            if matches:
                unique = list(dict.fromkeys(matches))[:5]
                issues.append(
                    Issue(
                        "CRITICAL",
                        f"{check_file} has solution leakage: {unique}",
                        "Remove references to solve.sh, solution/, oracle",
                    )
                )

    test_py = task_dir / "tests" / "test_outputs.py"
    if test_py.is_file():
        placeholders = PLACEHOLDER_PATTERN.findall(test_py.read_text())
        if placeholders:
            issues.append(
                Issue(
                    "CRITICAL",
                    f"test_outputs.py has placeholder values: {list(dict.fromkeys(placeholders))[:3]}",
                    "Run the oracle and fill in actual hash values",
                )
            )

    test_sh_path = task_dir / "tests" / "test.sh"
    if test_sh_path.is_file():
        test_sh = test_sh_path.read_text()
        if "reward.txt" not in test_sh:
            issues.append(Issue("CRITICAL", "test.sh doesn't write reward.txt"))
        if "ctrf" not in test_sh:
            issues.append(Issue("HIGH", "test.sh doesn't produce ctrf.json"))

        if toml and "allow_internet = false" in toml:
            internet_cmds = []
            if re.search(r"\bcurl\b", test_sh):
                internet_cmds.append("curl")
            if re.search(r"\bwget\b", test_sh):
                internet_cmds.append("wget")
            if "pip install" in test_sh:
                internet_cmds.append("pip install")
            if "apt-get install" in test_sh or "apt install" in test_sh:
                internet_cmds.append("apt install")
            if "install.sh" in test_sh and ("uv" in test_sh or "astral" in test_sh):
                internet_cmds.append("uv download")
            if internet_cmds:
                issues.append(
                    Issue(
                        "CRITICAL",
                        f"test.sh downloads packages ({', '.join(internet_cmds)}) but allow_internet=false",
                        "Move all test deps to Dockerfile",
                    )
                )

    env_dir = task_dir / "environment"
    if env_dir.is_dir() and toml:
        file_count = sum(
            1
            for f in env_dir.rglob("*")
            if f.is_file() and f.name not in ("Dockerfile", "docker-compose.yaml", "docker-compose.yml")
        )
        if 'codebase_size = "small"' in toml and file_count < 20:
            issues.append(
                Issue(
                    "HIGH",
                    f"codebase_size='small' but only {file_count} files in environment/ (need >=20)",
                    "Add more fixture files or split large JSONs",
                )
            )

    rubric_path = task_dir / "rubrics.txt"
    if rubric_path.is_file():
        lines = [line.strip() for line in rubric_path.read_text().splitlines() if line.strip()]
        pos_sum = 0
        neg_count = 0
        for i, line in enumerate(lines, 1):
            if not line.startswith("Agent"):
                issues.append(Issue("HIGH", f"rubrics.txt line {i}: must start with 'Agent'"))
            parts = line.rsplit(",", 1)
            if len(parts) != 2:
                issues.append(Issue("HIGH", f"rubrics.txt line {i}: missing ',+/-N' at end"))
                continue
            score_str = parts[1].strip()
            if not re.match(r"^[+-]\d+$", score_str):
                issues.append(Issue("HIGH", f"rubrics.txt line {i}: invalid score '{score_str}'"))
                continue
            score = int(score_str)
            if abs(score) == 4:
                issues.append(Issue("HIGH", f"rubrics.txt line {i}: ±4 is forbidden"))
            if score > 0:
                pos_sum += score
            else:
                neg_count += 1
        if pos_sum < 10 or pos_sum > 40:
            issues.append(Issue("HIGH", f"Rubric positive sum = {pos_sum} (must be 10-40)"))
        if neg_count < 3:
            issues.append(Issue("HIGH", f"Rubric has only {neg_count} negatives (need >=3)"))

    return issues


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
        base += f"&sheet={worksheet.replace(' ', '%20')}"
    return base


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
    csv_url = _sheet_csv_url(sheet_url, worksheet)
    df = pd.read_csv(csv_url)
    df.columns = [str(c).strip() for c in df.columns]

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
    """Load tracker sheet or local JSON for instruction similarity."""
    notes: list[str] = []
    instructions: dict[str, str] = {}
    specs: dict[str, str] = {}
    corpus_meta: dict[str, dict[str, str]] = {}

    if sheet_url.strip():
        try:
            instructions, specs, corpus_meta, sheet_notes = load_reference_from_sheet(
                sheet_url,
                worksheet=worksheet,
                task_col=task_col,
                instruction_col=instruction_col,
                spec_col=spec_col,
                trainer_col=trainer_col,
                instruction_col_index=instruction_col_index,
            )
            notes.extend(sheet_notes)
        except Exception as exc:
            notes.append(f"Google Sheet load failed: {exc}")
    elif corpus_json_path and Path(corpus_json_path).exists():
        instructions, specs, json_notes = load_reference_from_json(Path(corpus_json_path))
        corpus_meta = {
            k: {"instruction": v, "trainer": "", "task_name": k}
            for k, v in instructions.items()
        }
        notes.extend(json_notes)
    else:
        notes.append("No similarity reference — configure tracker sheet in admin secrets.")

    return instructions, specs, corpus_meta, notes


def run_instruction_similarity(
    instruction_text: str,
    instructions_corpus: dict[str, str],
    corpus_meta: dict[str, dict[str, str]] | None = None,
    exclude_task_name: str = "",
    api_key: str = "",
) -> tuple[list[SimilarityMatch], bool, str, list[str], dict[str, Any]]:
    """Parallel lexical + embedding check. Block only when both >= 60%."""
    from similarity_engine import compare_instruction_to_corpus_full

    notes: list[str] = []
    api_key = (api_key or "").strip()
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
    exclude = {k for k in meta if exclude_task_name.lower() in k.lower()}

    result = compare_instruction_to_corpus_full(
        inst_text, meta, exclude_keys=exclude, api_key=api_key, top_n=10,
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
        )
        for h in hits
    ]

    blocked = any(m.dual_block for m in inst_matches)
    block_message = ""
    if blocked:
        top = next(m for m in inst_matches if m.dual_block)
        block_message = (
            f"{CHANGE_TASK_MESSAGE} Closest match: {top.task_id}"
            f" (trainer: {top.trainer or 'unknown'}) — "
            f"lexical {round(top.lexical_score * 100)}%, "
            f"embedding {round((top.semantic_score or 0) * 100)}%."
        )
        notes.append(block_message)
    elif hits and sim_meta.embedding_ran:
        top = hits[0]
        notes.append(
            f"Top match — lexical {round(top.lexical_score * 100)}%, "
            f"embedding {round((top.semantic_score or 0) * 100)}% "
            f"(dual block needs both ≥ 60%)"
        )
    elif hits:
        top = hits[0]
        notes.append(
            f"Top match — lexical {round(top.lexical_score * 100)}% only "
            f"(embedding unavailable)"
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
    """Standalone instruction.md check — runs before zip upload."""
    instructions, _, corpus_meta, load_notes = load_similarity_corpus(
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
    )
    notes = load_notes + notes

    if blocked:
        pass_message = block_message
    elif run_meta.get("embedding_ran") and matches:
        top = matches[0]
        pass_message = (
            f"No change needed — top match lexical {round((top.lexical_score or 0) * 100)}%, "
            f"embedding {round((top.semantic_score or 0) * 100)}% "
            f"(via {run_meta.get('embed_model', 'text-embedding-3-small')}; "
            f"dual block requires both ≥ 60%)."
        )
    elif run_meta.get("embedding_ran"):
        pass_message = (
            f"Embedding ran ({run_meta.get('embed_model')}) but no tracker rows matched."
        )
    elif matches:
        pass_message = (
            "Lexical check only — embedding did NOT run. "
            "Configure OPENAI_API_KEY in Streamlit secrets for full dual check."
        )
    else:
        pass_message = "No tracker instructions loaded to compare against."

    return {
        "blocked": blocked,
        "message": pass_message,
        "matches": matches,
        "notes": notes,
        "change_task": blocked,
        "embedding_ran": run_meta.get("embedding_ran", False),
        "embed_model": run_meta.get("embed_model", "text-embedding-3-small"),
        "embedding_error": run_meta.get("embedding_error"),
        "corpus_size": run_meta.get("corpus_size", 0),
    }


def run_similarity_checks(
    task_dir: Path,
    instructions_corpus: dict[str, str],
    specs_corpus: dict[str, str],
    task_name: str,
    corpus_meta: dict[str, dict[str, str]] | None = None,
    api_key: str = "",
) -> tuple[list[SimilarityMatch], list[SimilarityMatch], float, bool, str, list[str]]:
    from similarity_engine import tfidf_similarity

    notes: list[str] = []
    inst_path = task_dir / "instruction.md"
    inst_text = inst_path.read_text().strip() if inst_path.exists() else ""
    spec_files = list(task_dir.rglob("SPEC.md"))
    spec_text = spec_files[0].read_text().strip() if spec_files else ""

    if not instructions_corpus and not specs_corpus:
        notes.append("Similarity skipped — no reference corpus provided")
        return [], [], 0.0, False, "", notes

    inst_matches, blocked, block_message, inst_notes, _ = run_instruction_similarity(
        inst_text,
        instructions_corpus,
        corpus_meta=corpus_meta,
        exclude_task_name=task_name,
        api_key=api_key,
    )
    notes.extend(inst_notes)

    spec_matches: list[SimilarityMatch] = []
    if spec_text and specs_corpus:
        for tid, score in tfidf_similarity(spec_text, specs_corpus, exclude_name=task_name):
            spec_matches.append(
                SimilarityMatch(task_id=tid, score=float(score), source="spec", method="tfidf")
            )

    max_score = 0.0
    if inst_matches:
        max_score = max(max_score, inst_matches[0].score)
    if spec_matches:
        max_score = max(max_score, spec_matches[0].score)

    return inst_matches, spec_matches, max_score, blocked, block_message, notes


def _read_limited(path: Path, limit: int = 15000) -> str:
    if not path.exists():
        return "[FILE NOT FOUND]"
    text = path.read_text()
    return text[:limit] if len(text) > limit else text


def _parse_llm_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)


def load_accepted_reference_digest(accepted_dir: Path, char_limit: int = 1200) -> str:
    """Build a compact digest of platform-accepted tasks for LLM comparison."""
    if not accepted_dir.is_dir():
        return "[No accepted reference directory found]"

    parts: list[str] = []
    for name in ACCEPTED_TASK_NAMES:
        task_path = accepted_dir / name
        if not task_path.is_dir():
            continue
        inst = task_path / "instruction.md"
        toml = task_path / "task.toml"
        snippet = ""
        if inst.is_file():
            snippet = inst.read_text().strip()[:char_limit]
        category = ""
        if toml.is_file():
            match = re.search(r'category\s*=\s*"([^"]+)"', toml.read_text())
            if match:
                category = match.group(1)
        parts.append(
            f"--- ACCEPTED: {name} (category={category}) ---\n{snippet}\n"
        )

    if not parts:
        return "[No accepted reference tasks loaded]"
    return "\n".join(parts)


def build_closest_match_context(
    inst_matches: list[SimilarityMatch],
    spec_matches: list[SimilarityMatch],
    instructions_corpus: dict[str, str],
    specs_corpus: dict[str, str],
) -> str:
    """Summarize the closest corpus matches for accepted-pattern comparison."""
    lines: list[str] = []
    if inst_matches:
        top = inst_matches[0]
        excerpt = instructions_corpus.get(top.task_id, "")[:1500]
        lines.append(
            f"Closest instruction match: {top.task_id} ({top.score:.1%})\n{excerpt}"
        )
    if spec_matches:
        top = spec_matches[0]
        excerpt = specs_corpus.get(top.task_id, "")[:1500]
        lines.append(
            f"Closest SPEC match: {top.task_id} ({top.score:.1%})\n{excerpt}"
        )
    return "\n\n".join(lines) if lines else ""


def run_llm_judge(
    task_dir: Path,
    api_key: str,
    accepted_dir: Path | None = None,
    inst_matches: list[SimilarityMatch] | None = None,
    spec_matches: list[SimilarityMatch] | None = None,
    instructions_corpus: dict[str, str] | None = None,
    specs_corpus: dict[str, str] | None = None,
    model: str | None = None,
    on_progress: Any = None,
) -> tuple[dict[str, Any], bool | None, list[str]]:
    notes: list[str] = []
    if not api_key.strip():
        notes.append("LLM judge skipped — no API key provided")
        return {}, None, notes

    try:
        build_openai_client(api_key)
    except ImportError as exc:
        notes.append(f"LLM judge skipped — openai package missing: {exc}")
        return {}, None, notes

    instruction = _read_limited(task_dir / "instruction.md")
    solve_sh = _read_limited(task_dir / "solution" / "solve.sh")
    test_py = _read_limited(task_dir / "tests" / "test_outputs.py")
    dockerfile = _read_limited(task_dir / "environment" / "Dockerfile")
    task_toml = _read_limited(task_dir / "task.toml", limit=5000)
    spec_files = list(task_dir.rglob("SPEC.md"))
    spec_md = _read_limited(spec_files[0]) if spec_files else "[NO SPEC.MD FOUND]"

    accepted_digest = load_accepted_reference_digest(accepted_dir or Path())
    closest_match = build_closest_match_context(
        inst_matches or [],
        spec_matches or [],
        instructions_corpus or {},
        specs_corpus or {},
    )

    file_kwargs = {
        "instruction": instruction,
        "spec_md": spec_md,
        "test_py": test_py,
        "solve_sh": solve_sh,
        "dockerfile": dockerfile,
        "task_toml": task_toml,
    }

    llm_model = model or resolve_llm_model(api_key)
    client = build_openai_client(api_key)
    provider = api_provider_label(api_key)
    results: dict[str, Any] = {}
    reject_count = 0
    needs_work_count = 0
    total_checks = len(ALIGNMENT_CHECKS)

    for idx, (key, label, prompt_fn) in enumerate(ALIGNMENT_CHECKS, start=1):
        if on_progress:
            on_progress(idx, total_checks, label, llm_model)
        if key == "accepted_pattern":
            prompt = prompt_fn(
                **file_kwargs,
                accepted_digest=accepted_digest,
                closest_match=closest_match,
            )
        else:
            prompt = prompt_fn(**file_kwargs)

        try:
            response = client.chat.completions.create(
                model=llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2500,
            )
            raw = response.choices[0].message.content or ""
            parsed = _parse_llm_json(raw)
            parsed["label"] = label
            results[key] = parsed
            verdict = parsed.get("verdict", "UNKNOWN")
            if verdict == "REJECT":
                reject_count += 1
            elif verdict == "NEEDS_WORK":
                needs_work_count += 1
        except Exception as exc:
            results[key] = {
                "label": label,
                "verdict": "ERROR",
                "alignment_score": 0,
                "reasoning": str(exc),
                "gaps": [],
            }
            notes.append(f"LLM alignment check failed for {key}: {exc}")

    notes.append(
        f"LLM alignment judge completed via {provider} ({llm_model}): "
        f"{len(ALIGNMENT_CHECKS)} checks, {reject_count} rejects, {needs_work_count} needs-work"
    )
    llm_pass = reject_count == 0 and needs_work_count == 0
    return results, llm_pass, notes


def static_pass(issues: list[Issue]) -> bool:
    return not any(issue.severity in {"CRITICAL", "HIGH"} for issue in issues)


def similarity_pass(matches: list[SimilarityMatch]) -> bool:
    """Fail when any row has BOTH lexical and embedding >= 60%."""
    return not any(m.dual_block for m in matches)


def overall_pass(
    static_ok: bool,
    sim_ok: bool,
    llm_ok: bool | None,
    instruction_blocked: bool = False,
) -> bool:
    """LLM alignment is primary; dual instruction block always fails."""
    if instruction_blocked:
        return False
    if llm_ok is False:
        return False
    if not sim_ok:
        return False
    if llm_ok is True:
        return static_ok
    return static_ok and sim_ok


def assess_task(
    zip_bytes: bytes,
    zip_name: str,
    trainer_name: str = "",
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    spec_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = TRACKER_COL_TASK_INSTRUCTION,
    corpus_json_path: str = "",
    openai_api_key: str = "",
    run_llm: bool = False,
    work_dir: Path | None = None,
    on_llm_progress: Any = None,
) -> tuple[QCReport, Path]:
    work_root = work_dir or Path("/tmp/terminus_qc")
    work_root.mkdir(parents=True, exist_ok=True)
    task_dir = work_root / "current_task"
    if task_dir.exists():
        for child in task_dir.iterdir():
            if child.is_dir():
                for sub in child.rglob("*"):
                    if sub.is_file():
                        sub.unlink()
                for sub in sorted(child.rglob("*"), reverse=True):
                    if sub.is_dir():
                        sub.rmdir()
                child.rmdir()
            elif child.is_file():
                child.unlink()
    task_dir.mkdir(parents=True, exist_ok=True)

    unzip_task(zip_bytes, task_dir)
    task_name = detect_task_name(task_dir, zip_name)

    notes: list[str] = []
    api_key = openai_api_key.strip() or resolve_openai_api_key()

    instructions_corpus, specs_corpus, corpus_meta, corpus_notes = load_similarity_corpus(
        sheet_url=sheet_url,
        worksheet=worksheet,
        task_col=task_col,
        instruction_col=instruction_col,
        spec_col=spec_col,
        trainer_col=trainer_col,
        instruction_col_index=instruction_col_index,
        corpus_json_path=corpus_json_path,
    )
    notes.extend(corpus_notes)

    # Instruction.md is checked FIRST — before static checks and LLM.
    inst_matches, spec_matches, max_sim, instruction_blocked, block_message, sim_notes = (
        run_similarity_checks(
            task_dir,
            instructions_corpus,
            specs_corpus,
            task_name,
            corpus_meta=corpus_meta,
            api_key=api_key,
        )
    )
    notes.extend(sim_notes)
    if instruction_blocked:
        notes.insert(0, f"[INSTRUCTION CHECK] {block_message}")

    static_issues = run_static_checks(task_dir)

    llm_results: dict[str, Any] = {}
    llm_ok: bool | None = None
    if run_llm:
        accepted_dir = Path(__file__).resolve().parent.parent / "My_Accepted_Tasks"
        llm_results, llm_ok, llm_notes = run_llm_judge(
            task_dir,
            api_key,
            accepted_dir=accepted_dir,
            inst_matches=inst_matches,
            spec_matches=spec_matches,
            instructions_corpus=instructions_corpus,
            specs_corpus=specs_corpus,
            on_progress=on_llm_progress,
        )
        notes.extend(llm_notes)

    sim_ok = similarity_pass(inst_matches)
    report = QCReport(
        task_name=task_name,
        trainer_name=trainer_name.strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_pass=overall_pass(
            static_pass(static_issues),
            sim_ok,
            llm_ok,
            instruction_blocked=instruction_blocked,
        ),
        static_pass=static_pass(static_issues),
        similarity_pass=sim_ok and not instruction_blocked,
        llm_pass=llm_ok,
        static_issues=static_issues,
        instruction_matches=inst_matches,
        spec_matches=spec_matches,
        max_similarity=max_sim,
        instruction_blocked=instruction_blocked,
        instruction_block_message=block_message,
        llm_results=llm_results,
        notes=notes,
    )
    return report, task_dir


def report_to_dict(report: QCReport) -> dict[str, Any]:
    return {
        "task_name": report.task_name,
        "trainer_name": report.trainer_name,
        "timestamp": report.timestamp,
        "overall_pass": report.overall_pass,
        "static_checks": {
            "pass": report.static_pass,
            "issues": [asdict(issue) for issue in report.static_issues],
        },
        "similarity": {
            "pass": report.similarity_pass,
            "instruction_blocked": report.instruction_blocked,
            "instruction_block_message": report.instruction_block_message,
            "dual_threshold_percent": int(INSTRUCTION_SIM_THRESHOLD * 100),
            "max_score_percent": round(report.max_similarity * 100, 1),
            "threshold_warn_percent": int(SIM_THRESHOLD_WARN * 100),
            "threshold_block_percent": int(SIM_THRESHOLD_BLOCK * 100),
            "instruction_top_matches": [
                {
                    "task": match.task_id,
                    "trainer": match.trainer,
                    "score_percent": round(match.score * 100, 1),
                    "lexical_percent": round(match.lexical_score * 100, 1) if match.lexical_score else None,
                    "semantic_percent": round(match.semantic_score * 100, 1) if match.semantic_score else None,
                    "dual_block": match.dual_block,
                    "method": match.method,
                    "source": match.source,
                }
                for match in report.instruction_matches
            ],
            "spec_top_matches": [
                {
                    "task": match.task_id,
                    "score_percent": round(match.score * 100, 1),
                    "source": match.source,
                }
                for match in report.spec_matches
            ],
        },
        "llm_judge": {
            "pass": report.llm_pass,
            "results": report.llm_results or None,
        },
        "notes": report.notes,
    }


def render_html_report(report: QCReport) -> str:
    data = report_to_dict(report)
    severity_colors = {
        "CRITICAL": "#e74c3c",
        "HIGH": "#e67e22",
        "MEDIUM": "#f1c40f",
        "INFO": "#3498db",
    }

    def badge(ok: bool | None) -> str:
        if ok is None:
            return '<span class="badge skip">SKIPPED</span>'
        if ok:
            return '<span class="badge pass">PASS</span>'
        return '<span class="badge fail">FAIL</span>'

    issue_rows = ""
    for issue in report.static_issues:
        color = severity_colors.get(issue.severity, "#666")
        issue_rows += (
            f"<tr><td style='color:{color};font-weight:700'>{issue.severity}</td>"
            f"<td>{issue.message}</td><td>{issue.fix_hint}</td></tr>"
        )
    if not issue_rows:
        issue_rows = "<tr><td colspan='3'>No static issues found.</td></tr>"

    def sim_rows(matches: list[SimilarityMatch]) -> str:
        if not matches:
            return "<tr><td colspan='5'>No matches.</td></tr>"
        rows = ""
        for match in matches:
            lex = round((match.lexical_score or 0) * 100, 1)
            sem = (
                round(match.semantic_score * 100, 1)
                if match.semantic_score is not None else "—"
            )
            status = "CHANGE TASK" if match.dual_block else "OK"
            rows += (
                f"<tr><td>{match.task_id}</td><td>{lex}%</td><td>{sem}%</td>"
                f"<td>{status}</td><td>{match.trainer or '—'}</td></tr>"
            )
        return rows

    llm_sections = ""
    if report.llm_results:
        for key, item in report.llm_results.items():
            if not isinstance(item, dict):
                continue
            verdict = item.get("verdict", "UNKNOWN")
            score = item.get("alignment_score", "—")
            label = item.get("label", ALIGNMENT_LABELS.get(key, key))
            v_color = "#2ecc71" if verdict == "PASS" else "#e67e22" if verdict == "NEEDS_WORK" else "#e74c3c"
            gaps = item.get("gaps", [])

            gap_rows = ""
            if isinstance(gaps, list):
                for gap in gaps:
                    if isinstance(gap, dict):
                        gap_rows += (
                            f"<tr><td><code>{gap.get('file', '—')}</code></td>"
                            f"<td>{gap.get('issue', '')}</td>"
                            f"<td>{gap.get('fix', '')}</td></tr>"
                        )
                    elif isinstance(gap, str) and gap:
                        gap_rows += f"<tr><td>—</td><td colspan='2'>{gap}</td></tr>"

            llm_sections += f"""
    <div style="border-left:4px solid {v_color};padding:8px 12px;margin:12px 0">
      <strong>{label}</strong>
      — <span style="color:{v_color};font-weight:700">{verdict}</span> (score: {score}/100)
      <br><em>{item.get('reasoning', '')}</em>
    </div>"""
            if gap_rows:
                llm_sections += f"""
    <table style="margin-left:16px">
      <tr><th>File</th><th>Issue</th><th>Fix</th></tr>
      {gap_rows}
    </table>"""
    else:
        llm_sections = "<p><em>LLM alignment judge not run.</em></p>"

    overall_color = "#2ecc71" if report.overall_pass else "#e74c3c"
    overall_text = "READY TO SUBMIT" if report.overall_pass else "NEEDS FIXES"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{report.task_name} QC Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .summary {{ border: 2px solid {overall_color}; border-radius: 12px; padding: 16px; margin-bottom: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 14px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f7f7f7; text-align: left; }}
    .badge {{ padding: 4px 10px; border-radius: 6px; color: white; font-weight: 700; }}
    .pass {{ background: #2ecc71; }}
    .fail {{ background: #e74c3c; }}
    .skip {{ background: #95a5a6; }}
    .meta {{ color: #666; }}
  </style>
</head>
<body>
  <div class="summary">
    <h1 style="color:{overall_color}">{overall_text}</h1>
    <p><strong>Task:</strong> {report.task_name}</p>
    <p><strong>Trainer:</strong> {report.trainer_name or "Not provided"}</p>
    <p class="meta"><strong>Generated:</strong> {report.timestamp}</p>
    <table>
      <tr><th>Layer</th><th>Result</th><th>Detail</th></tr>
      <tr><td>Static Checks</td><td>{badge(report.static_pass)}</td><td>{len(report.static_issues)} issues</td></tr>
      <tr><td>Similarity</td><td>{badge(report.similarity_pass)}</td><td>Max {round(report.max_similarity * 100, 1)}%</td></tr>
      <tr><td>LLM Judge</td><td>{badge(report.llm_pass)}</td><td>{'Completed' if report.llm_results else 'Skipped'}</td></tr>
    </table>
  </div>

  <h2>Static Issues</h2>
  <table>
    <tr><th>Severity</th><th>Issue</th><th>Fix</th></tr>
    {issue_rows}
  </table>

  <h2>Instruction Similarity (dual block: lexical &amp; embedding both &ge; 60%)</h2>
  {f'<p style="color:#e74c3c;font-weight:700">{report.instruction_block_message}</p>' if report.instruction_blocked else ''}
  <table>
    <tr><th>Task</th><th>Lexical</th><th>Embedding</th><th>Status</th><th>Trainer</th></tr>
    {sim_rows(report.instruction_matches)}
  </table>

  <h2>SPEC Similarity</h2>
  <table>
    <tr><th>Task</th><th>Score</th><th>Status</th></tr>
    {sim_rows(report.spec_matches)}
  </table>

  <h2>LLM Alignment Judge (vs 8 accepted reference tasks)</h2>
  {llm_sections}

  <h2>Notes</h2>
  <ul>
    {''.join(f'<li>{note}</li>' for note in report.notes)}
  </ul>
</body>
</html>"""
