"""Terminus Trainer QC — LLM alignment prompts.

8 alignment checks, each a dedicated LLM call comparing the submitted task
against accepted-task patterns and checking cross-file consistency.
Every prompt asks for exact issues with file names and line-level detail.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ACCEPTED_TASK_NAMES = [
    "cgroup-resource-cascade-auditor",
    "cargo-workspace-dependency-auditor",
    "core-dump-analyzer",
    "card-game-tournament-engine",
    "memory-leak-profiler",
    "api-compat-audit",
    "pkt-frag-audit",
    "api-deprecation-cascade-auditor",
]

REFERENCES_PATH = Path(__file__).resolve().parent / "accepted_references.json"


def load_references() -> dict[str, Any]:
    if REFERENCES_PATH.exists():
        return json.loads(REFERENCES_PATH.read_text())
    return {}


def _build_all_accepted_instructions(refs: dict[str, Any]) -> str:
    """Include every accepted instruction.md — all 8 references."""
    parts = []
    for name in ACCEPTED_TASK_NAMES:
        data = refs.get(name, {})
        inst = data.get("instruction_md", "")
        toml = data.get("task_toml", "")
        category = ""
        if toml:
            import re
            m = re.search(r'category\s*=\s*"([^"]+)"', toml)
            if m:
                category = m.group(1)
        if inst:
            parts.append(
                f"--- ACCEPTED REFERENCE {len(parts) + 1}/8: {name} "
                f"(category={category}) ---\n{inst}\n"
            )
    for name, data in refs.items():
        if name not in ACCEPTED_TASK_NAMES and data.get("instruction_md"):
            parts.append(f"--- ACCEPTED REFERENCE: {name} ---\n{data['instruction_md']}\n")
    return "\n".join(parts) if parts else "[No accepted references available]"


def _build_accepted_instruction_examples(refs: dict[str, Any], limit: int | None = None) -> str:
    """All accepted instructions by default; optional limit for legacy callers."""
    if limit is None:
        return _build_all_accepted_instructions(refs)
    parts = []
    for i, (name, data) in enumerate(refs.items()):
        if i >= limit:
            break
        inst = data.get("instruction_md", "")
        if inst:
            parts.append(f"--- ACCEPTED TASK: {name} ---\n{inst}\n")
    return "\n".join(parts) if parts else "[No accepted references available]"


def _build_accepted_test_patterns(refs: dict[str, Any]) -> str:
    """Summarize test patterns across all accepted tasks."""
    lines = []
    for name, data in refs.items():
        hash_keys = data.get("expected_field_hash_keys", [])
        classes = data.get("test_classes", [])
        asserted = data.get("hashes_asserted", False)
        lines.append(
            f"  {name}: {len(hash_keys)} hash-locked fields, "
            f"{len(classes)} test classes, hashes_asserted={asserted}"
        )
    return "\n".join(lines)


def _build_accepted_solution_patterns(refs: dict[str, Any]) -> str:
    lines = []
    for name, data in refs.items():
        solve = data.get("solution_solve_sh", "")
        lines.append(f"--- {name} solve.sh ---\n{solve}\n")
    return "\n".join(lines)


def _build_accepted_dockerfile_patterns(refs: dict[str, Any]) -> str:
    lines = []
    for name, data in refs.items():
        df = data.get("environment_Dockerfile", "")
        lines.append(f"--- {name} Dockerfile ---\n{df}\n")
    return "\n".join(lines)


def _build_accepted_toml_patterns(refs: dict[str, Any]) -> str:
    lines = []
    for name, data in refs.items():
        toml = data.get("task_toml", "")
        lines.append(f"--- {name} task.toml ---\n{toml}\n")
    return "\n".join(lines)


def _files_block(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    return f"""
===== SUBMITTED task.toml =====
{task_toml}

===== SUBMITTED instruction.md =====
{instruction}

===== SUBMITTED SPEC.md =====
{spec_md}

===== SUBMITTED tests/test_outputs.py =====
{test_py}

===== SUBMITTED solution/solve.sh =====
{solve_sh}

===== SUBMITTED environment/Dockerfile =====
{dockerfile}
"""


REVIEWER_PERSONA = """You are a senior Terminus Edition-2 task reviewer. You have reviewed
and approved the 8 accepted reference tasks shown below. You know what good looks like.
Your job is to compare a SUBMITTED task and find
every specific issue. Be precise: cite exact file names, line content, field names,
and what is wrong. Do not give vague feedback. Every gap you list must be actionable."""

LLM_STRUCTURE_GUARD = """
IMPORTANT REVIEW RULES:
- If any file block shows "[FILE NOT FOUND]" or "[NO SPEC.MD FOUND]", do NOT fail alignment
  for missing content in that file — static structure checks already flagged it.
- Evaluate ONLY files that are present. Never REJECT solely because a file is absent.
- tests/test.sh is NOT included in this review bundle — never flag missing test.sh here.
- Static checks already grep instruction.md and test_outputs.py for solve.sh/solution/oracle
  strings and flag Dockerfile COPY tests/ or COPY solution/ — do not REJECT for those
  if static checks would already catch them unless you see a genuine miss.
- Use NEEDS_WORK (not REJECT) for polish issues on otherwise sound accepted-style tasks.
- Flag language mismatch only when task.toml/instruction explicitly require a compiled
  language (C, Go, Rust) and solve.sh clearly ignores that requirement.
- Python/bash heredoc solutions are acceptable when the task does not require compilation.
"""


# ─── PROMPT 1: Accepted Pattern Alignment ───────────────────────────────

def prompt_accepted_pattern_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
    accepted_digest: str = "",
    closest_match: str = "",
) -> str:
    refs = load_references()
    examples = _build_accepted_instruction_examples(refs)
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether the submitted instruction.md follows the accepted 3-paragraph pattern.

ACCEPTED instruction.md EXAMPLES (these passed review):
{examples}

PATTERN RULES (from accepted tasks):
- Paragraph 1: Describe the input directory path, list each functional data file and its role,
  and explicitly list decorative/ignored files by name.
- Paragraph 2: One sentence stating the task goal ("Implement an auditor/engine that …
  and writes a JSON report to /app/<path>"). A second sentence saying SPEC.md is the
  authoritative and complete reference for ALL behavioral rules. Do NOT enumerate rules.
- Paragraph 3: List the exact top-level output keys. Specify formatting: 2-space indent,
  recursive sorted keys, no trailing newline.
- The instruction must NOT list numbered rules, algorithmic steps, or SPEC content.
- Absolute paths (/app/...) are required.
- For language-specific tasks (C, Go), paragraph 2 states the language requirement.

CLOSEST EXISTING TASK (for differentiation check):
{closest_match or "None identified"}

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON (no markdown fences):
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "instruction.md", "issue": "<exact problem>", "fix": "<what to do>"}},
    ...
  ],
  "paragraph_structure": {{
    "para1_input_files": "PASS|FAIL — <detail>",
    "para2_task_goal_spec_pointer": "PASS|FAIL — <detail>",
    "para3_output_schema": "PASS|FAIL — <detail>"
  }},
  "too_similar_to_existing": <true|false>,
  "lists_rules_or_algorithms": <true|false>,
  "per_reference_alignment": [
    {{"reference_task": "<name>", "pattern_match_score": <0-100>, "differentiation_ok": <true|false>, "note": "<one line>"}},
    ... (one entry for EACH of the 8 accepted references above)
  ]
}}"""


# ─── PROMPT 1b: Instruction vs ALL Accepted References ──────────────────

def prompt_instruction_vs_all_references(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    refs = load_references()
    all_instructions = _build_all_accepted_instructions(refs)
    ref_names = [n for n in ACCEPTED_TASK_NAMES if refs.get(n, {}).get("instruction_md")]
    return f"""{REVIEWER_PERSONA}

TASK: Compare the SUBMITTED instruction.md against EVERY accepted reference task below.
You MUST produce one comparison entry for each reference ({len(ref_names)} tasks).

For each reference, judge:
- Does the submitted task follow the same structural pattern (3 paragraphs, SPEC pointer, output keys)?
- Is the submitted task sufficiently DIFFERENT (not a copy/reskin of that reference)?
- Does it share the same domain/problem type in a way that would fail diversity review?

ALL ACCEPTED REFERENCE instruction.md FILES:
{all_instructions}

SUBMITTED instruction.md ONLY (compare this against each reference above):
{instruction}

Also consider submitted task.toml category vs each reference's category.

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100 overall>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "instruction.md", "issue": "<exact problem>", "fix": "<what to do>"}},
    ...
  ],
  "reference_comparisons": [
    {{
      "reference_task": "<exact name from list>",
      "structural_alignment": <0-100>,
      "differentiation_score": <0-100>,
      "too_similar": <true|false>,
      "issues": ["<specific issue vs this reference>", "..."]
    }}
  ],
  "references_checked": {len(ref_names)},
  "most_similar_reference": "<task name>",
  "copy_paste_risk": <true|false>
}}

RULE: reference_comparisons MUST have exactly one entry per accepted reference task listed above.
REJECT if any reference has too_similar=true AND structural_alignment > 70.
"""


# ─── PROMPT 2: Instruction ↔ SPEC Alignment ─────────────────────────────

def prompt_instruction_spec_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether instruction.md and SPEC.md are correctly separated.

RULE: instruction.md says WHAT to build. SPEC.md says HOW (all behavioral rules).
Instruction must point to SPEC as the authoritative reference and must NOT restate rules.

CHECK EACH OF THESE:
1. Does instruction.md contain the sentence "must conform to ... SPEC.md, which is the
   authoritative and complete reference"? (Exact or close wording required.)
2. Does instruction.md list ANY numbered rules (Rule 1, Rule 2, etc.)? → REJECT
3. Does instruction.md describe algorithms, enum values, sort orders, or tie-breaking
   logic that belong in SPEC? → REJECT
4. Are the top-level keys listed in instruction.md an exact match to what SPEC defines? List mismatches.
5. Does instruction.md reference the correct input directory path that SPEC also uses?
6. Are decorative files in instruction.md consistent with what SPEC ignores?
7. Is the output path in instruction.md the same as what SPEC references?

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "<file>", "issue": "<exact content that is wrong>", "fix": "<what to change>"}},
    ...
  ],
  "spec_pointer_present": <true|false>,
  "instruction_leaks_rules": <true|false>,
  "key_mismatches": ["<key in instruction but not SPEC>", "..."],
  "path_mismatches": ["<path inconsistency>", "..."]
}}"""


# ─── PROMPT 3: Instruction ↔ Tests Alignment ────────────────────────────

def prompt_instruction_test_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    refs = load_references()
    test_patterns = _build_accepted_test_patterns(refs)
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether tests cover exactly what instruction.md promises — no more, no less.

ACCEPTED TASK TEST PATTERNS (for comparison):
{test_patterns}

CHECK EACH OF THESE:
1. Does test_outputs.py verify the report exists at the exact path instruction.md states?
2. Does test_outputs.py check the exact number of top-level keys instruction.md claims?
3. Does test_outputs.py check every top-level key name listed in instruction.md?
4. Does test_outputs.py verify formatting (2-space indent, sorted keys, no trailing newline)
   as instruction.md requires?
5. Are there tests asserting behavior NOT mentioned in instruction.md or SPEC? List them.
6. Are there instruction.md claims with NO test coverage? List them.
7. Does test_outputs.py contain an EXPECTED_FIELD_HASHES dict? If yes, are ALL keys
   actually used in test assertions? (Defined-but-unused hashes are a known rejection reason.)
8. Do test class names follow the accepted pattern (TestInputIntegrity, TestReportStructure,
   TestCanonicalFieldHashes, then domain-specific test classes)?

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "tests/test_outputs.py", "issue": "<exact problem>", "fix": "<what to do>"}},
    ...
  ],
  "untested_instruction_claims": ["<specific claim>", "..."],
  "tests_beyond_spec": ["<test name or assertion>", "..."],
  "expected_field_hashes_used": <true|false|"not_present">,
  "hash_keys_defined": <number>,
  "hash_keys_asserted": <number>
}}"""


# ─── PROMPT 4: SPEC ↔ Tests Alignment ───────────────────────────────────

def prompt_spec_test_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether tests enforce SPEC.md tightly enough that broken implementations fail.

THIS IS THE MOST IMPORTANT CHECK. A task was rejected because:
- EXPECTED_FIELD_HASHES was defined with 8 keys but no test referenced the dict
- Tolerances accepted ±43% deviation, letting broken routing pass
- 10/10 agents passed tests despite producing wrong output

CHECK EACH OF THESE:
1. EXPECTED_FIELD_HASHES:
   - Is the dict defined? How many keys?
   - Is there a TestCanonicalFieldHashes class that actually asserts each key's hash?
   - Or is it defined and NEVER referenced in any assertion? (CRITICAL issue)
2. Tolerances:
   - For numerical assertions, are ranges tight (exact or ±5%)?
   - Any assertion with >=20% tolerance band? Flag each one with the test name.
3. Enum coverage:
   - Does SPEC define enum values (status types, categories, etc.)?
   - Is there at least one test per enum value, or are some untested?
4. Sort order verification:
   - Does SPEC define sort orders for arrays? Are they tested?
5. Anti-cheat / provenance:
   - For C/Go tasks: are there tests that verify a compiled binary exists?
   - Do tests re-run the binary to confirm it reproduces the report? (Not just "file exists")
   - Are interpreter scripts forbidden? (Check test_no_interpreter_solver_scripts pattern)
6. Edge cases:
   - Does SPEC have two-pass rules, cascades, tie-breakers?
   - Are these edge cases tested with specific fixture data?

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "tests/test_outputs.py", "issue": "<exact problem with line/test name>", "fix": "<specific fix>"}},
    ...
  ],
  "unused_hash_checks": <true|false>,
  "permissive_tolerances": [
    {{"test_name": "<name>", "tolerance": "<what it allows>", "expected": "<what it should be>"}},
    ...
  ],
  "untested_spec_rules": ["<SPEC rule with no test>", "..."],
  "anti_cheat_present": <true|false>,
  "anti_cheat_strength": "strong|weak|absent"
}}"""


# ─── PROMPT 5: Solution ↔ Instruction Alignment ─────────────────────────

def prompt_solution_instruction_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    refs = load_references()
    solve_patterns = _build_accepted_solution_patterns(refs)
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether solve.sh fulfills what instruction.md asks for.

ACCEPTED solve.sh PATTERNS (these passed review):
{solve_patterns}

CHECK EACH OF THESE:
1. Does solve.sh write to the EXACT output path instruction.md specifies?
   - instruction.md says "/app/<something>.json" — does solve.sh create that file?
2. Does solve.sh read from the input directory instruction.md describes?
3. Does solve.sh start with "#!/bin/bash" and "set -euo pipefail"?
4. For language-specific tasks:
   - If instruction says "Implement in C": does solve.sh compile .c file with gcc and run binary?
   - If instruction says "Implement in Go": does solve.sh compile .go file and run binary?
   - If instruction says nothing about language: Python heredoc is acceptable.
5. Is the solution deterministic (no randomness, no timestamps, no network calls)?
6. Does solve.sh match the accepted patterns shown above?
7. Language rule: REJECT language mismatch ONLY if instruction.md AND task.toml explicitly
   require C/Go/Rust compilation and solve.sh uses none of those. Otherwise NEEDS_WORK at most.

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "solution/solve.sh", "issue": "<exact problem>", "fix": "<what to do>"}},
    ...
  ],
  "output_path_match": <true|false>,
  "input_path_match": <true|false>,
  "language_match": <true|false>,
  "deterministic": <true|false>
}}"""


# ─── PROMPT 6: Solution ↔ SPEC Alignment ────────────────────────────────

def prompt_solution_spec_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether solve.sh implements SPEC.md rules — not a stub or hardcoded output.

A TASK WAS REJECTED FOR THIS EXACT REASON:
"The compiled C binary in the solution is 'int main(void) {{ return 0; }}' — a pure no-op.
The actual report generation was performed by a Python script in a bash heredoc."
This is called "hardcoded_solution" and is a blocking rejection.

CHECK EACH OF THESE:
1. STUB DETECTION — does solve.sh contain any of these patterns?
   - A C file with only "int main(){{return 0;}}" or "int main(void){{return 0;}}"
   - A Python heredoc that writes the full JSON output directly
   - An echo/cat command that prints pre-computed JSON
   - A script that copies a pre-existing JSON file to the output path
   If YES → REJECT with "hardcoded_solution"
2. Does the solution appear to READ input files (json.load, fopen, os.Open)?
3. Does the solution appear to COMPUTE results (loops, conditionals, data structures)?
4. For multi-phase SPEC rules (cascades, two-pass, recomputation):
   - Does the solution code show evidence of multiple phases?
   - Or is it a single pass that would miss re-evaluation?
5. task.toml says languages=[...] — does solve.sh use that language?
   - languages=["c","bash"] → solve.sh should compile and run a .c file
   - languages=["go","bash"] → solve.sh should compile and run a .go file
   - If languages are bash-only or instruction does not mandate compilation, do not REJECT
     for helper scripts / heredocs.
   - REJECT hardcoded_solution only for obvious stubs (empty main, pre-printed JSON).

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "<file>", "issue": "<exact stub or mismatch>", "fix": "<what to do>"}},
    ...
  ],
  "hardcoded_solution": <true|false>,
  "reads_input_files": <true|false>,
  "computes_results": <true|false>,
  "language_matches_toml": <true|false>
}}"""


# ─── PROMPT 7: Solution ↔ Tests Alignment ───────────────────────────────

def prompt_solution_test_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether tests and oracle form a closed loop — oracle passes tests, and
tests actually CATCH broken implementations (not just check file-exists).

A TASK WAS FLAGGED FOR THIS:
"The C-language requirement can be fully circumvented using inline Python execution
combined with a dummy compiled binary, while still passing every test."

CHECK EACH OF THESE:
1. BYPASS RISK — can an agent do this and STILL pass all tests?
   - Write a Python script that generates the correct JSON
   - Compile an empty C binary (int main(){{return 0;}})
   - Both the binary-exists test and the hash tests would pass
   → This means provenance tests are WEAK. Flag it.
2. Does test_outputs.py have a test that RE-RUNS the compiled binary and verifies
   the regenerated output matches? (test_binary_reproduces_report pattern)
   - This is the ONLY way to prevent the bypass above.
3. Does test_outputs.py check for forbidden interpreter scripts?
   - test_no_interpreter_solver_scripts: searches /app, /tmp, /root for .py, .sh, .rb, etc.
4. Would any test pass with EMPTY output (empty JSON {{}} or [])?
   - Check: are there tests that assert len(report["key"]) > 0?
5. Do the hash values in EXPECTED_FIELD_HASHES match what solve.sh would actually produce?
   (This is hard to verify without running, but check for obvious mismatches like
   hash dict referencing keys that solve.sh doesn't generate.)
6. Skip test.sh / reward.txt checks — test.sh is not in this review bundle.

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "<file>", "issue": "<exact bypass or weakness>", "fix": "<specific fix>"}},
    ...
  ],
  "bypass_possible": <true|false>,
  "bypass_method": "<how an agent could cheat>" | null,
  "binary_reproduction_test": <true|false>,
  "interpreter_script_check": <true|false>,
  "empty_output_passes": <true|false>
}}"""


# ─── PROMPT 8: Environment ↔ Task Alignment ─────────────────────────────

def prompt_environment_alignment(
    instruction: str,
    spec_md: str,
    test_py: str,
    solve_sh: str,
    dockerfile: str,
    task_toml: str,
) -> str:
    refs = load_references()
    df_patterns = _build_accepted_dockerfile_patterns(refs)
    toml_patterns = _build_accepted_toml_patterns(refs)
    return f"""{REVIEWER_PERSONA}

TASK: Judge whether Dockerfile, task.toml, and environment files are consistent
with each other and with what instruction/SPEC/tests expect.

ACCEPTED Dockerfile PATTERNS:
{df_patterns}

ACCEPTED task.toml PATTERNS:
{toml_patterns}

CHECK EACH OF THESE:
1. Dockerfile:
   - Base image pinned with @sha256:? (FROM ... @sha256:...)
   - Slim base? (debian:bookworm-slim or python:*-slim or golang:*-alpine)
   - NOT gcc:*-bookworm or golang:*-bookworm (these are >1GB, cause build timeouts)
   - COPY only environment/ data dirs? Never COPY tests/ or solution/?
   - pytest + pytest-json-ctrf baked in when allow_internet=false?
   - Compilers match task.toml languages? (gcc for C, go for Go, etc.)
2. task.toml:
   - No comments (lines starting with #)?
   - No "name" field?
   - codebase_size not "minimal"?
   - allow_internet = false present?
   - agent.timeout_sec = 1800.0?
   - subcategories either [] or from the valid list?
   - languages matches what solve.sh actually uses?
   - difficulty = "hard" if languages includes "python"?
3. File count:
   - codebase_size = "small" requires >=20 files in environment/ (excluding Dockerfile)
   - If less, flag it
4. Dead directories:
   - Are there directories in environment/ that Dockerfile doesn't COPY? Flag them.
5. Cross-consistency:
   - Does Dockerfile WORKDIR match task paths in instruction.md?
   - Does Dockerfile install the right compiler for the task language?

SUBMITTED TASK:
{_files_block(instruction, spec_md, test_py, solve_sh, dockerfile, task_toml)}

Return ONLY this JSON:
{{
  "verdict": "PASS" | "NEEDS_WORK" | "REJECT",
  "alignment_score": <0-100>,
  "reasoning": "<2-3 sentences>",
  "gaps": [
    {{"file": "<file>", "issue": "<exact problem>", "fix": "<what to do>"}},
    ...
  ],
  "dockerfile_risks": [
    {{"risk": "<problem>", "severity": "critical|high|medium"}},
    ...
  ],
  "toml_issues": ["<issue>", "..."],
  "language_consistent": <true|false>,
  "file_count_ok": <true|false>
}}"""


# ─── Registry ───────────────────────────────────────────────────────────

ALIGNMENT_CHECKS: list[tuple[str, str, Any]] = [
    ("instruction_vs_all_refs", "Instruction vs All 8 References", prompt_instruction_vs_all_references),
    ("accepted_pattern", "Accepted Task Pattern", prompt_accepted_pattern_alignment),
    ("instruction_spec", "Instruction ↔ SPEC", prompt_instruction_spec_alignment),
    ("instruction_test", "Instruction ↔ Tests", prompt_instruction_test_alignment),
    ("spec_test", "SPEC ↔ Tests", prompt_spec_test_alignment),
    ("solution_instruction", "Solution ↔ Instruction", prompt_solution_instruction_alignment),
    ("solution_spec", "Solution ↔ SPEC", prompt_solution_spec_alignment),
    ("solution_test", "Solution ↔ Tests", prompt_solution_test_alignment),
    ("environment", "Environment ↔ Task", prompt_environment_alignment),
]

ALIGNMENT_LABELS = {key: label for key, label, _ in ALIGNMENT_CHECKS}

# Minimum on-disk files before an LLM alignment check runs.
LLM_CHECK_REQUIRES: dict[str, list[str]] = {
    "instruction_vs_all_refs": ["instruction.md"],
    "accepted_pattern": ["instruction.md"],
    "instruction_spec": ["instruction.md"],
    "instruction_test": ["instruction.md", "tests/test_outputs.py"],
    "spec_test": ["tests/test_outputs.py"],
    "solution_instruction": ["solution/solve.sh", "instruction.md"],
    "solution_spec": ["solution/solve.sh"],
    "solution_test": ["solution/solve.sh", "tests/test_outputs.py"],
    "environment": ["environment/Dockerfile", "task.toml"],
}
LLM_CHECKS_NEED_SPEC = {"instruction_spec", "spec_test"}
