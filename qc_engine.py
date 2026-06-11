"""Terminus Trainer QC engine — static checks, similarity, LLM judge, report export."""

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

from alignment_prompts import (
    ALIGNMENT_CHECKS,
    ALIGNMENT_LABELS,
    ACCEPTED_TASK_NAMES,
    LLM_CHECK_REQUIRES,
    LLM_CHECKS_NEED_SPEC,
    LLM_STRUCTURE_GUARD,
)
from config import (
    api_provider_label,
    build_openai_client,
    chat_completion_kwargs,
    resolve_embed_model,
    resolve_llm_model,
    resolve_llm_parallel_workers,
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
TEST_IMPORTS_SOLUTION = re.compile(
    r"\b(?:from|import)\s+solution\b|importlib\.import_module\s*\(\s*['\"]solution",
    re.IGNORECASE,
)
PLACEHOLDER_PATTERN = re.compile(r"SET_AFTER|PLACEHOLDER|TODO_HASH|FIXME", re.IGNORECASE)
LARGE_IMAGE_PATTERN = re.compile(
    r"^FROM\s+(gcc|golang|node|ruby|python(?!.*slim)):\S*-bookworm",
    re.MULTILINE | re.IGNORECASE,
)
GOLANG_BASE_PATTERN = re.compile(r"^FROM\s+golang:\S+", re.MULTILINE | re.IGNORECASE)
GOLANG_BOOKWORM_PATTERN = re.compile(r"^FROM\s+golang:\S*-bookworm", re.MULTILINE | re.IGNORECASE)

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
    "Change the task — your instruction is too similar to an existing one "
    f"(word overlap and meaning both ≥ {DUAL_BLOCK_PCT}%, or meaning ≥ {SEMANTIC_BLOCK_PCT}%)."
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
    if match.block_reason == "dual":
        return f"YES (both ≥{DUAL_BLOCK_PCT}%)"
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


@dataclass
class QCReport:
    task_name: str
    trainer_name: str
    timestamp: str
    overall_pass: bool
    static_pass: bool
    structure_pass: bool
    rubric_pass: bool
    similarity_pass: bool
    llm_pass: bool | None
    static_issues: list[Issue] = field(default_factory=list)
    instruction_matches: list[SimilarityMatch] = field(default_factory=list)
    spec_matches: list[SimilarityMatch] = field(default_factory=list)
    max_similarity: float = 0.0
    instruction_blocked: bool = False
    instruction_block_message: str = ""
    submitted_instruction: str = ""
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


def _has_spec_md(task_dir: Path) -> bool:
    return any(task_dir.rglob("SPEC.md"))


def check_task_structure(task_dir: Path) -> tuple[list[Issue], list[str]]:
    """Phase 1 — required submission layout (before content-level static/LLM checks)."""
    issues: list[Issue] = []
    missing: list[str] = []
    task_dir = Path(task_dir)

    for relpath, desc in TASK_REQUIRED_FILES.items():
        if not (task_dir / relpath).is_file():
            missing.append(relpath)
            issues.append(
                Issue(
                    "CRITICAL",
                    f"Missing {relpath} ({desc})",
                    "Fix zip layout — files must be at archive root (not nested in a folder).",
                )
            )

    if not missing:
        if not _has_spec_md(task_dir):
            issues.append(
                Issue(
                    "HIGH",
                    "Missing SPEC.md under environment/",
                    "Add environment/.../SPEC.md — required behavioral reference.",
                )
            )
    return issues, missing


def validate_rubric_text(rubric_text: str) -> list[Issue]:
    """Validate rubrics.txt content (uploaded separately — not in submission zip)."""
    issues: list[Issue] = []
    text = (rubric_text or "").strip()
    if not text:
        issues.append(
            Issue(
                "MEDIUM",
                "No rubrics uploaded",
                "Paste or upload rubrics.txt — entered in the Snorkel platform textbox at submission.",
            )
        )
        return issues

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        issues.append(Issue("HIGH", "rubrics.txt is empty"))
        return issues

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


def run_static_checks(task_dir: Path) -> list[Issue]:
    """Phase 2 — content checks (only on files that exist)."""
    issues: list[Issue] = []
    task_dir = Path(task_dir)

    structure_issues, missing = check_task_structure(task_dir)
    issues.extend(structure_issues)
    if missing:
        return issues
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

        if LARGE_IMAGE_PATTERN.search(dockerfile) or GOLANG_BOOKWORM_PATTERN.search(dockerfile):
            issues.append(
                Issue(
                    "CRITICAL",
                    "Dockerfile uses a large bookworm base image (gcc/golang/ruby/node/python — "
                    "risk of 7200s build timeout)",
                    "Use debian:bookworm-slim@sha256:… and install compilers/Go tarball explicitly",
                )
            )
        elif GOLANG_BASE_PATTERN.search(dockerfile) and not re.search(
            r"^FROM\s+golang:\S*-alpine", dockerfile, re.MULTILINE | re.IGNORECASE
        ):
            issues.append(
                Issue(
                    "MEDIUM",
                    "Dockerfile uses golang base image (prefer debian:bookworm-slim + Go tarball)",
                    "Install Go from official tarball on debian:bookworm-slim (~80MB base)",
                )
            )

        if toml and "allow_internet = false" in toml:
            if "pytest" not in dockerfile:
                issues.append(
                    Issue(
                        "HIGH",
                        "allow_internet=false but pytest not installed in Dockerfile",
                        "Bake pytest into the Dockerfile",
                    )
                )
            if "pytest-json-ctrf" not in dockerfile and "json-ctrf" not in dockerfile:
                issues.append(
                    Issue(
                        "MEDIUM",
                        "allow_internet=false but pytest-json-ctrf not in Dockerfile",
                        "Bake pytest-json-ctrf into the Dockerfile (test.sh needs ctrf.json)",
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
        test_src = test_py.read_text()
        if TEST_IMPORTS_SOLUTION.search(test_src):
            issues.append(
                Issue(
                    "CRITICAL",
                    "test_outputs.py imports or loads the solution package",
                    "Re-derive expected values from the spec — never import solution/",
                )
            )
        placeholders = PLACEHOLDER_PATTERN.findall(test_src)
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

    return issues


def structure_pass(issues: list[Issue]) -> bool:
    return not any(
        issue.severity == "CRITICAL" and issue.message.startswith("Missing ")
        for issue in issues
    )


def rubric_pass(issues: list[Issue]) -> bool:
    return not any(issue.severity in {"CRITICAL", "HIGH"} for issue in issues)


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
            if not instructions:
                notes.append(
                    "Tracker sheet loaded but column 'Task Instruction' (P) has no text "
                    f'on tab "{worksheet or "default"}".'
                )
        except Exception as exc:
            notes.append(f"Google Sheet load failed: {exc}")

    if not instructions:
        fallback_path = Path(corpus_json_path) if corpus_json_path else BUNDLED_CORPUS_PATH
        if fallback_path.exists():
            instructions, specs, json_notes = load_reference_from_json(fallback_path)
            corpus_meta = {
                k: {"instruction": v, "trainer": "", "task_name": k}
                for k, v in instructions.items()
            }
            notes.extend(json_notes)
            notes.append(
                f"Using bundled local corpus ({len(instructions)} tasks) — "
                "tracker sheet unavailable or empty."
            )
        elif not notes:
            notes.append("No similarity reference — configure tracker sheet in admin secrets.")

    return instructions, specs, corpus_meta, notes


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
            else f"both word overlap and meaning ≥ {DUAL_BLOCK_PCT}%"
        )
        block_message = (
            f"{CHANGE_TASK_MESSAGE} Closest match: {top.task_id}"
            f" (trainer: {top.trainer or 'unknown'}) — "
            f"word overlap {round(top.lexical_score * 100)}%, "
            f"meaning {round((top.semantic_score or 0) * 100)}% "
            f"({reason_note}). Use 👁 review below to compare instructions."
        )
        notes.append(block_message)
    elif hits and sim_meta.embedding_ran:
        top = hits[0]
        notes.append(
            f"Top match — word overlap {round(top.lexical_score * 100)}%, "
            f"meaning {round((top.semantic_score or 0) * 100)}% "
            f"(flagged when both ≥ {DUAL_BLOCK_PCT}% or meaning ≥ {SEMANTIC_BLOCK_PCT}%)"
        )
    elif hits:
        top = hits[0]
        notes.append(
            f"Top match — word overlap {round(top.lexical_score * 100)}% only "
            f"(meaning check unavailable)"
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

    if blocked:
        pass_message = block_message
    elif run_meta.get("embedding_ran") and matches:
        top = matches[0]
        pass_message = (
            f"Looks good — closest match has word overlap "
            f"{round((top.lexical_score or 0) * 100)}% and meaning "
            f"{round((top.semantic_score or 0) * 100)}% "
            f"(flagged when both ≥ {DUAL_BLOCK_PCT}% or meaning ≥ {SEMANTIC_BLOCK_PCT}%)."
        )
    elif run_meta.get("embedding_ran"):
        pass_message = (
            "Meaning check ran but no close matches on the tracker."
        )
    elif matches:
        pass_message = (
            "Word-overlap check only — meaning check did not run. "
            "Configure OPENAI_API_KEY in Streamlit secrets for the full comparison."
        )
    elif not instructions:
        sheet_errors = [n for n in load_notes if "failed" in n.lower() or "returned" in n.lower()]
        pass_message = (
            "No tracker instructions loaded to compare against. "
            + (sheet_errors[0] if sheet_errors else "Check sheet sharing and tab name in secrets.")
        )
    else:
        pass_message = "No similarity matches returned."

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


def run_similarity_checks(
    task_dir: Path,
    instructions_corpus: dict[str, str],
    specs_corpus: dict[str, str],
    task_name: str,
    corpus_meta: dict[str, dict[str, str]] | None = None,
    api_key: str = "",
    tracker_cache: dict[str, Any] | None = None,
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
        tracker_cache=tracker_cache,
    )
    enrich_similarity_match_texts(inst_matches, instructions_corpus)
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


def _build_alignment_prompt(
    key: str,
    prompt_fn: Any,
    file_kwargs: dict[str, str],
    accepted_digest: str,
    closest_match: str,
) -> str:
    if key == "accepted_pattern":
        body = prompt_fn(
            **file_kwargs,
            accepted_digest=accepted_digest,
            closest_match=closest_match,
        )
    else:
        body = prompt_fn(**file_kwargs)
    return f"{LLM_STRUCTURE_GUARD}\n\n{body}"


def _llm_check_applicable(task_dir: Path, check_key: str) -> bool:
    required = LLM_CHECK_REQUIRES.get(check_key, [])
    for relpath in required:
        if not (task_dir / relpath).is_file():
            return False
    if check_key in LLM_CHECKS_NEED_SPEC and not _has_spec_md(task_dir):
        return False
    return True


def _skipped_llm_result(label: str, reason: str) -> dict[str, Any]:
    return {
        "label": label,
        "verdict": "SKIPPED",
        "alignment_score": None,
        "reasoning": reason,
        "gaps": [],
    }


def _run_single_alignment_check(
    key: str,
    label: str,
    prompt_fn: Any,
    file_kwargs: dict[str, str],
    accepted_digest: str,
    closest_match: str,
    client: Any,
    llm_model: str,
) -> tuple[str, dict[str, Any], str | None]:
    """Run one alignment LLM call. Returns (key, result_dict, error_note)."""
    prompt = _build_alignment_prompt(
        key, prompt_fn, file_kwargs, accepted_digest, closest_match,
    )
    try:
        response = client.chat.completions.create(
            **chat_completion_kwargs(
                llm_model,
                [{"role": "user", "content": prompt}],
                max_output_tokens=2500,
                temperature=0.1,
            )
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_llm_json(raw)
        parsed["label"] = label
        return key, parsed, None
    except Exception as exc:
        return key, {
            "label": label,
            "verdict": "ERROR",
            "alignment_score": 0,
            "reasoning": str(exc),
            "gaps": [],
        }, f"LLM alignment check failed for {key}: {exc}"


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
    structure_ok: bool = True,
) -> tuple[dict[str, Any], bool | None, list[str]]:
    notes: list[str] = []
    if not api_key.strip():
        notes.append("LLM judge skipped — no API key provided")
        return {}, None, notes

    if not structure_ok:
        notes.append(
            "LLM alignment skipped — fix task folder structure first "
            "(missing required files at zip root)."
        )
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

    checks_to_run: list[tuple[str, str, Any]] = []
    for key, label, prompt_fn in ALIGNMENT_CHECKS:
        if _llm_check_applicable(task_dir, key):
            checks_to_run.append((key, label, prompt_fn))
        else:
            results[key] = _skipped_llm_result(
                label,
                "Skipped — required files not present (structure check flagged missing paths).",
            )

    total_checks = len(checks_to_run)
    if not checks_to_run:
        notes.append("LLM alignment skipped — no applicable checks (missing core files).")
        return results, None, notes

    max_workers = resolve_llm_parallel_workers(total_checks)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_single_alignment_check,
                key,
                label,
                prompt_fn,
                file_kwargs,
                accepted_digest,
                closest_match,
                client,
                llm_model,
            ): (key, label)
            for key, label, prompt_fn in checks_to_run
        }
        completed = 0
        for future in as_completed(futures):
            key, parsed, error_note = future.result()
            completed += 1
            if on_progress:
                on_progress(
                    completed,
                    total_checks,
                    parsed.get("label", futures[future][1]),
                    llm_model,
                )
            results[key] = parsed
            if error_note:
                notes.append(error_note)
            verdict = parsed.get("verdict", "UNKNOWN")
            if verdict == "REJECT":
                reject_count += 1
            elif verdict == "NEEDS_WORK":
                needs_work_count += 1

    notes.append(
        f"LLM alignment judge completed via {provider} ({llm_model}): "
        f"{len(checks_to_run)} checks run ({len(ALIGNMENT_CHECKS) - len(checks_to_run)} skipped), "
        f"parallel max {max_workers} workers, "
        f"{reject_count} rejects, {needs_work_count} needs-work"
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
    structure_ok: bool = True,
    rubric_ok: bool = True,
) -> bool:
    """LLM alignment is primary when structure is valid; dual instruction block always fails."""
    if instruction_blocked:
        return False
    if not structure_ok:
        return False
    if not rubric_ok:
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
    rubric_text: str = "",
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
    from portal_cache import tracker_cache_params

    tracker_cache = tracker_cache_params(
        sheet_url=sheet_url,
        worksheet=worksheet,
        task_col=task_col,
        instruction_col=instruction_col,
        spec_col=spec_col,
        trainer_col=trainer_col,
        instruction_col_index=instruction_col_index,
        corpus_json_path=corpus_json_path,
    )

    instructions_corpus, specs_corpus, corpus_meta, corpus_notes = fetch_similarity_corpus(
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

    # Phase 1 — folder structure + static content checks (before LLM).
    static_issues = run_static_checks(task_dir)
    rubric_issues = validate_rubric_text(rubric_text)
    struct_ok = structure_pass(static_issues)
    rubric_ok = rubric_pass(rubric_issues)
    if not struct_ok:
        notes.insert(0, "[STRUCTURE] Fix missing required files at zip root before LLM review.")

    inst_matches: list[SimilarityMatch] = []
    spec_matches: list[SimilarityMatch] = []
    max_sim = 0.0
    instruction_blocked = False
    block_message = ""

    if (task_dir / "instruction.md").is_file():
        inst_matches, spec_matches, max_sim, instruction_blocked, block_message, sim_notes = (
            run_similarity_checks(
                task_dir,
                instructions_corpus,
                specs_corpus,
                task_name,
                corpus_meta=corpus_meta,
                api_key=api_key,
                tracker_cache=tracker_cache,
            )
        )
        notes.extend(sim_notes)
        if instruction_blocked:
            notes.insert(0, f"[INSTRUCTION CHECK] {block_message}")
    else:
        notes.append("Instruction similarity skipped — instruction.md missing from zip.")

    llm_results: dict[str, Any] = {}
    llm_ok: bool | None = None
    if run_llm:
        accepted_root = Path(__file__).resolve().parent.parent / "My_Accepted_Tasks"
        accepted_dir = accepted_root if accepted_root.is_dir() else Path(__file__).resolve().parent
        llm_results, llm_ok, llm_notes = run_llm_judge(
            task_dir,
            api_key,
            accepted_dir=accepted_dir,
            inst_matches=inst_matches,
            spec_matches=spec_matches,
            instructions_corpus=instructions_corpus,
            specs_corpus=specs_corpus,
            on_progress=on_llm_progress,
            structure_ok=struct_ok,
        )
        notes.extend(llm_notes)

    sim_ok = similarity_pass(inst_matches)
    all_issues = static_issues + rubric_issues
    inst_path = task_dir / "instruction.md"
    submitted_instruction = inst_path.read_text().strip() if inst_path.is_file() else ""
    report = QCReport(
        task_name=task_name,
        trainer_name=trainer_name.strip(),
        timestamp=datetime.now(timezone.utc).isoformat(),
        overall_pass=overall_pass(
            static_pass(all_issues),
            sim_ok,
            llm_ok,
            instruction_blocked=instruction_blocked,
            structure_ok=struct_ok,
            rubric_ok=rubric_ok,
        ),
        static_pass=static_pass(all_issues),
        structure_pass=struct_ok,
        rubric_pass=rubric_ok,
        similarity_pass=sim_ok and not instruction_blocked,
        llm_pass=llm_ok,
        static_issues=all_issues,
        instruction_matches=inst_matches,
        spec_matches=spec_matches,
        max_similarity=max_sim,
        instruction_blocked=instruction_blocked,
        instruction_block_message=block_message,
        submitted_instruction=submitted_instruction,
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
        "structure_pass": report.structure_pass,
        "rubric_pass": report.rubric_pass,
        "static_checks": {
            "pass": report.static_pass,
            "issues": [asdict(issue) for issue in report.static_issues],
        },
        "similarity": {
            "pass": report.similarity_pass,
            "instruction_blocked": report.instruction_blocked,
            "instruction_block_message": report.instruction_block_message,
            "dual_threshold_percent": DUAL_BLOCK_PCT,
            "semantic_block_threshold_percent": SEMANTIC_BLOCK_PCT,
            "submitted_instruction": report.submitted_instruction,
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
                    "block_reason": match.block_reason,
                    "flag_label": similarity_flag_label(match),
                    "method": match.method,
                    "source": match.source,
                    "matched_instruction": match.matched_instruction,
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

    import html as html_module

    def sim_rows(matches: list[SimilarityMatch]) -> str:
        if not matches:
            return "<tr><td colspan='6'>No matches.</td></tr>"
        rows = ""
        for match in matches:
            lex = round((match.lexical_score or 0) * 100, 1)
            sem = (
                round(match.semantic_score * 100, 1)
                if match.semantic_score is not None else "—"
            )
            status = similarity_flag_label(match)
            rows += (
                f"<tr><td>{html_module.escape(match.task_id)}</td>"
                f"<td>{lex}%</td><td>{sem}%</td><td>{status}</td>"
                f"<td>{html_module.escape(match.trainer or '—')}</td>"
                f"<td><a href=\"#review-{html_module.escape(match.task_id)}\">👁</a></td></tr>"
            )
        return rows

    inst_review_html = _html_instruction_comparison_section(
        report.submitted_instruction,
        report.instruction_matches,
        {m.task_id: m.matched_instruction for m in report.instruction_matches},
    )

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
    {_html_comparison_styles()}
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

  <h2>Instruction Similarity (both ≥ {DUAL_BLOCK_PCT}% OR meaning ≥ {SEMANTIC_BLOCK_PCT}%)</h2>
  {f'<p style="color:#e74c3c;font-weight:700">{html_module.escape(report.instruction_block_message)}</p>' if report.instruction_blocked else ''}
  <table>
    <tr><th>Task</th><th>Word overlap</th><th>Meaning</th><th>Flagged?</th><th>Trainer</th><th>Review</th></tr>
    {sim_rows(report.instruction_matches)}
  </table>
  <h3>👁 Full instruction comparison</h3>
  <p>Complete instruction.md next to each tracker match — scores are in each block header.</p>
  {inst_review_html}

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
