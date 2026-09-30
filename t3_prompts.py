"""Prompts for the Terminus 3 task review.

Every prompt here restates a published rule from the Terminus 3 documentation (quality panel judge
guide, writing tests, prompt styling, difficulty guidelines, reviewer checklist) or the public
Terminal-Bench quality-gate rubric in rubrics/task-implementation.toml. Nothing here comes from the
Edition-2 checker. Each prompt asks for JSON only.
"""

from __future__ import annotations

SYSTEM = (
    "You are a strict, evidence-driven reviewer for Terminus 3 (Terminal-Bench 3.0) expert tasks. "
    "You judge a task package the way the platform's quality stages do. You cite files and line "
    "numbers from the package exactly as given (each file is shown with line numbers). You never "
    "invent files, lines or behaviour that the package does not show. When unsure, say so and use "
    "the lower severity. Reply with a single JSON object and nothing else."
)

SEVERITY_RULES = """Severity scale (from the quality panel judge guide):
- "Major": a serious, concrete defect with a mechanism you can state (a wrong solution that passes,
  a valid solution that fails, a rule the candidate cannot know, a reference that violates the
  contract, a reachable path to the answer, nondeterminism that reaches a grade).
- "Minor": a narrower but real defect of the same kind.
- "Advisory": a recommendation only, e.g. a requirement that has one easy case exercised but no
  demonstrated wrong outcome. Advisory never blocks.
Do not report style preferences. Every Minor or Major needs a concrete mechanism: the input, the
behaviour, and why the grade is wrong. Read the whole view before claiming something is missing:
the fix is often in another file or later in the same file. A claim that depends on how a library,
database engine, interpreter or tool behaves at runtime, where you are not certain and the package
does not show it, is Advisory at most."""

FINDING_SCHEMA = """{
  "verdict": "None" | "Advisory" | "Minor" | "Major",   // the worst non-advisory finding, or None
  "summary": "<two or three sentences>",
  "findings": [
    {
      "severity": "Major" | "Minor" | "Advisory",
      "title": "<one line>",
      "mechanism": "<concrete input or construction and what the grader does with it>",
      "citations": ["<file>:<line or range>", "..."],
      "fix": "<the smallest change that closes it, or 'cut the promise' when it is breadth>",
      "obligations": ["O3", "..."]                          // ids from the obligation list, if any
    }
  ]
}"""

OBLIGATIONS = """Read the candidate-visible normative text of a Terminus 3 task below: instruction.md,
the task.toml prose, and any specification or contract document shipped in environment/. The
quality panel turns every normative sentence into an obligation: something the reference must
satisfy and a test must exercise. Extract them.

Rules:
- One obligation per requirement (split sentences that make several promises).
- Include output paths and names, formats and schemas, rounding and tolerances, ordering and
  tie-breaks, boundaries and limits (both ends), stated input guarantees, lifecycle promises
  (reruns, existing outputs, failures, idempotency, cleanup), restrictions (libraries, network,
  files that must not change), and interfaces (CLI flags, functions).
- kind is one of: interface, format, rule, ordering, limit, input, scoping, disposition,
  guarantee, lifecycle, restriction, other.
- Quote the source text verbatim (short) and cite file:line.
- Mark "core": true when the obligation is part of the hard thing the task is about (its crux),
  false when it is breadth (an extra mode, an extra error case, an extra format).

Return:
{
  "obligations": [
    {"id": "O1", "kind": "...", "text": "<the requirement in your words>",
     "quote": "<verbatim excerpt>", "citation": "<file>:<line>", "core": true}
  ],
  "crux": "<one sentence: the expert decision the task turns on, or 'none found'>"
}"""

AXES = {
    "coherent_contract": {
        "label": "Coherent contract",
        "blocks_on": ["Minor", "Major"],
        "view": "contract + environment + tests (no reference solution)",
        "question": (
            "Could two competent people follow the available instructions and produce different "
            "answers that this grader scores differently?"
        ),
        "checklist": """- For every rule the tests grade, find the sentence or authoritative candidate-visible
  file that defines it. A rule graded but never stated is a hidden requirement (Minor or Major).
  Hidden test INPUTS are fine; hidden REQUIREMENTS are not.
- Unstated tie-breaks, rounding, boundary behaviour, output paths, empty/duplicate/invalid input
  handling, and conflicting sources of truth.
- Output paths and file names in instruction.md match what the verifier reads.
- Examples in the package agree with the stated rules; if inferring a rule is the task, the
  examples determine it uniquely.
- Broken template text, leftover REPLACE/TODO markers, accidental test sentinels.
- A tolerance stated in the instruction equals the tolerance the tests enforce.""",
    },
    "correct_reference_solution": {
        "label": "Correct reference solution",
        "blocks_on": ["Minor", "Major"],
        "view": "contract + environment + reference solution (no tests)",
        "question": (
            "Does solution/solve.sh and the code it produces actually satisfy the contract, "
            "including its boundary cases?"
        ),
        "checklist": """- Trace an ordinary case and each relevant boundary through the actual reference logic.
- Output bytes: widths, precision, padding, delimiters, paths, key names.
- Every branch that updates state: deletion, expiry, replacement, identifier reuse, reruns.
- Lost precision (floats for exact values, 28-digit Decimal context, int/str digit limits, YAML
  octal surprises), incorrect tie-breaks, stale loop state, reused ids treated as unique.
- Recovery and durability claims at the documented interruption points.
- Inputs opened read-write when they are only inputs; network use when the agent has none.
- solve.sh must derive the result, not write the answer.
Give a contract-valid input where the reference diverges for every Minor or Major.""",
    },
    "protected_ground_truth": {
        "label": "Protected ground truth",
        "blocks_on": ["Major"],
        "view": "environment + tests + task.toml (no reference solution)",
        "question": (
            "Can candidate-controlled code obtain or change what the grader treats as the "
            "expected answer, earning credit without doing the task?"
        ),
        "checklist": """- Goldens and held-out data live only in the verifier image, never in environment/ or any
  path the agent can write; nothing byte-identical to the solution or goldens ships to the agent.
- If tests execute candidate code, it runs as an unprivileged uid that cannot read goldens,
  /tests or /logs/verifier, and a test proves that boundary.
- Privileged reads of candidate-influenced paths: symlinks, path traversal, copytree staging.
- Expected values are computed independently of the submitted artifact (no candidate checksum
  certifies itself); no expected values read from agent-writable paths.
- Hidden outcomes leaked through file names, directory names (reject/, expected/), arguments,
  working directories or environment variables.
- The reward channel stays trusted: candidate stdout or an early exit cannot impersonate a pass.
A golden file under tests/ is normal; only a reachable path from candidate code to it is a defect.""",
    },
    "sound_verifier": {
        "label": "Sound verifier",
        "blocks_on": ["Minor", "Major"],
        "view": "contract + environment + tests (no reference solution)",
        "question": (
            "Does the grader distinguish solving the documented task from plausible wrong "
            "behaviour, while accepting valid alternatives?"
        ),
        "checklist": """Walk the obligation list. For each obligation, name the test and assertion that would fail if
it were missing or reversed. Then look for both directions:
- Wrong solutions that pass: proxy checks (existence, keys, counts, first element, schema only),
  a named domain rule enforced only inside mixed held-out data, a stated limit with no visible
  test at its boundary, an objective or tie-break never compared, a documented flag or mode never
  invoked, a lifecycle promise (rerun, existing output, failure, idempotency) never exercised,
  a stated restriction enforced nowhere, a delivered binary graded without rebuilding.
- Valid solutions that fail: exact text comparison where key order or whitespace is free, a
  tighter tolerance than stated, an undocumented ordering, format or implementation choice.
Describe the lazy or plausible-wrong implementation concretely for every Minor or Major.
Many tests or long files are not a defect; missing coverage of a stated promise is.""",
    },
    "deterministic_execution": {
        "label": "Deterministic execution",
        "blocks_on": ["Minor", "Major"],
        "view": "everything",
        "question": (
            "If the submission and package stay the same, can an uncontrolled input or race "
            "change the grade?"
        ),
        "checklist": """- Fresh random seeds, the current date or time, mutable remote data, unpinned dependencies,
  network at build or verify time that changes content.
- Filesystem or dict/set ordering that reaches a graded value; missing sorts where the contract
  requires order; required order where it does not.
- Sleep-based readiness, timing and latency assertions, resource-dependent limits.
- Tests that depend on each other's order or on shared global state.
A random temp filename that cannot affect grading is not a defect.""",
    },
}

AXIS_PROMPT = """You are one reviewer on the Terminus 3 quality panel, axis `{axis}` ({label}).

The question for this axis: {question}

You see this view of the package: {view}. That is deliberate; judge only from it.

What to check:
{checklist}

{severity}

The panel's obligation list for this task (extracted from the candidate-visible text):
{obligations}

Return JSON with this shape:
{schema}

PACKAGE VIEW
{package}
"""

GATE_PROMPT = """You are the Terminus 3 quality gate. Judge the task package below against each criterion.
For each criterion return "pass" or "fail" and one paragraph of reasoning that cites files. Follow
each criterion's guidance literally; it states when to PASS and when to FAIL. Where a criterion does
not apply to this task, return "pass" and say why it does not apply.

Terminus 3 context the guidance may not mention:
- The verifier runs in a separate container (task.toml [verifier].environment_mode = "separate")
  and sees only the paths declared in the top-level `artifacts` list.
- [environment].network_mode is always "public"; [agent] and [verifier] declare their own mode.
- rubrics.txt and README.md are added by the platform and must not be in the package.
- difficulty_explanation must say why the task is hard for a human expert, not quote a pass rate.

CRITERIA
{criteria}

Return:
{{"criteria": [{{"name": "<criterion name>", "result": "pass" | "fail", "reasoning": "<paragraph>"}}]}}

PACKAGE
{package}
"""

LLMAJ_CHECKS = [
    ("behavior_in_task_description",
     "Every behavior the tests check is described in instruction.md (or a candidate-visible file it names as authoritative)."),
    ("behavior_in_tests",
     "Every behavior instruction.md describes is verified by a test."),
    ("informative_test_docstrings",
     "Every test function has a docstring that says what it checks."),
    ("anti_cheating_measures",
     "It is hard for the agent to cheat: tests and answers are not visible or editable, data files cannot be edited to pass, no git history or pinned-source leak."),
    ("structured_data_schema",
     "If the agent produces structured data (JSON, CSV, SQLite, ...), the exact schema is stated."),
    ("hardcoded_solution",
     "solution/solve.sh demonstrates the derivation rather than writing the answer."),
    ("file_reference_mentioned",
     "Every file the tests read or require is named in instruction.md with its path."),
]

LLMAJ_PROMPT = """These are the seven LLM-as-judge checks that `stb harbor check` runs on every Terminus 3 task
(the platform runs them with Claude Sonnet 4.6). Judge the package against each.

{checks}

Return:
{{"checks": [{{"name": "<check name>", "result": "pass" | "fail", "reasoning": "<cite files>"}}]}}

PACKAGE
{package}
"""

INSTRUCTION_PROMPT = """Review instruction.md of a Terminus 3 task against the Prompt Styling guide and the Task
Requirements. The platform screens instruction.md and solve.sh for AI-generated text and fails tasks
above a threshold, and reviewers mark style problems Medium.

Check:
- Written the way an engineer types into a coding agent: direct, specific, plain. No persona
  ("You are an expert..."), no emoji, no marketing tone, no long preamble, little markdown.
- Says WHAT (goal, inputs, outputs with absolute paths, schema, tolerances, tie-breaks, boundary
  behaviour), never HOW (steps, hints, algorithm names, detection guidance, bold solution values).
- About two short paragraphs or up to about 20 bullets is the guidance; longer needs a reason.
- Absolute paths for every file it names; no task name; no "you have N minutes"; no canaries.
- Signs of machine-written text: stock phrases ("ensure that", "robust", "comprehensive",
  "seamlessly", "leverage", "delve"), em or en dashes used as punctuation, symmetrical
  triplets, headings over a two-paragraph task, perfectly parallel bullet lists.
- Spec files it names read like real engineering documents and define what, not how.

Return:
{{"verdict": "ok" | "revise",
  "ai_text_risk": "low" | "medium" | "high",
  "issues": [{{"severity": "Medium" | "Low", "issue": "<what>", "where": "<line or quote>", "fix": "<rewrite hint>"}}],
  "summary": "<two sentences>"}}

instruction.md
{instruction}

task.toml prose fields
{prose}
"""

DIFFICULTY_PROMPT = """Assess whether this Terminus 3 task clears the expertise floor and is likely to meet the
difficulty gate, using the Difficulty Guidelines.

Facts:
- Every task must need genuine domain expertise (graduate-level knowledge or years of
  professional experience) at every tier. Obscure facts, long checklists and sheer volume do not
  count. Expertise comes from choosing between valid methods under real constraints, diagnosing a
  plausible-but-wrong result, or reasoning about interacting constraints.
- The difficulty gate: of 8 runs (4 Claude Opus 5, 4 GPT-5.6) at least 3 must fail genuinely, and
  every test must be passed by at least one run (the solvable gate).
- A fully specified checklist contract is usually transcribed correctly by frontier models;
  difficulty has to come from an interaction the text cannot hand over. Clerical edge cases,
  format trivia, OS/permission details and time pressure are bad sources of difficulty.
- difficulty_explanation must say why a human expert finds it hard, not a pass rate.

Return:
{{"expertise_floor": "clears" | "borderline" | "fails",
  "crux": "<the expert decision, one sentence>",
  "silent_wrong_defaults": ["<natural wrong approach and its plausible wrong output>", "..."],
  "difficulty_risk": "too easy" | "in range" | "too hard or unsolvable",
  "reasoning": "<paragraph>",
  "explanation_quality": "good" | "revise",
  "suggestions": ["<concrete>", "..."]}}

PACKAGE
{package}
"""

SKEPTIC_PROMPT = """You are the skeptic on a Terminus 3 review. Another reviewer made the claims below about this
task package. Your job is to try to REFUTE each one using the files shown (they carry line numbers).

For each claim:
- "confirmed" only if you can quote the exact lines (file:line and the text) that show the defect
  and state the mechanism in one sentence: what input or construction makes the grade wrong, or
  which literal PASS/FAIL condition of a criterion is violated.
- "refuted" if the cited lines do not show it, another part of the package handles it, the claim
  depends on a runtime or library behaviour the package does not demonstrate, or it is a style
  preference rather than a defect.
When in doubt, answer "refuted". Do not add new claims.

What counts as evidence:
- Only instruction.md and files under environment/ are visible to the candidate. tests/, solution/
  and task.toml are NOT. A rule is stated only if its text is in a candidate-visible file.
- A test docstring or comment that says a rule is stated, or quotes the contract, proves nothing:
  check that the quoted text really appears in instruction.md or environment/. If it does not,
  that is evidence FOR a hidden-requirement claim.
- For a claim that no test exercises an obligation (or only an easy case), describe the most
  plausible wrong implementation of it, then confirm the claim unless you can name a test that
  runs a case where that wrong implementation gives a different result and quote the assertion
  that catches it. A test where the rule is trivially satisfied does not refute the claim.
- For a claim that the reference solution is wrong, confirm it only if you can trace the exact
  code path, quoting each line in order, from the entry point to the wrong output for one
  concrete input that the contract allows. If the path returns early, branches away, or another
  line handles the case, the claim is refuted.

CLAIMS
{claims}

Return:
{{"verdicts": [{{"id": "<claim id>", "status": "confirmed" | "refuted",
                "evidence": "<file:line and quoted text>", "reason": "<one or two sentences>"}}]}}

PACKAGE VIEW
{package}
"""

COVERAGE_PROMPT = """You check test coverage the way the quality panel's sound_verifier reviewer does. For each
obligation below, work in this order:

1. wrong_impl: write the most plausible WRONG implementation of that obligation that a capable
   engineer might ship (for example: skips the rule, applies it only in the easy case, truncates
   instead of rounding, forgets to restore a value, keeps the first instead of the last).
2. Look through the tests for a case where that wrong implementation gives a DIFFERENT result
   than a correct one, and an assertion that would then fail. Cases where the rule is trivially
   satisfied (empty input, value already equal, boundary never reached) do not count. Held-out
   data compared as a whole does not count for a named rule unless an assertion targets it.
3. caught: "yes" only if you can name the test function, describe that case, and cite the
   assertion line; otherwise "no".

OBLIGATIONS
{obligations}

Return:
{{"coverage": [{{"id": "<O#>", "wrong_impl": "<one sentence>", "caught": "yes" | "no",
                "test": "<test function or ''>", "case": "<the input that separates them or ''>",
                "assertion": "<file:line or ''>"}}]}}

PACKAGE VIEW (contract, environment and tests)
{package}
"""
