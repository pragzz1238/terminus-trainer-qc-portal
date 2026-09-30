"""Terminus 3 full task review: static checks, similarity, and an LLM rehearsal of the platform's
quality stages (quality gate, stb LLMaJ checks, the five-axis quality panel), built from the
current Terminus 3 documentation.

Stages, in the order the platform applies them:
  1. Package + static checks (t3_static, the CI and preflight rules that can be read from files).
  2. Instruction similarity against the team corpus (similarity_service engine).
  3. LLM review: obligation extraction, the five panel axes on their split views, the 29 public
     quality-gate criteria, the 7 LLMaJ checks, instruction style, and the expertise floor.

What it cannot do: build images, run the oracle or NOP, run real agents, or execute mutants. The
report says so, and a READY verdict here is a rehearsal, not a platform pass.
"""

from __future__ import annotations

import html
import io
import json
import re
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import t3_prompts as P
import t3_static
from t3_static import tomllib

APP_DIR = Path(__file__).resolve().parent
RUBRIC_TOML = APP_DIR / "rubrics" / "task-implementation.toml"

# The platform's quality gate runs the public rubric minus these six (PLATFORM-EVAL-PIPELINE).
GATE_SKIPPED = {"task_name", "resource_configuration", "task_readme", "expert_time_estimate",
                "task_toml_schema", "artifact_efficiency"}
# In returned payloads `difficult` is marked advisory (blocking: false); the others are treated as
# blocking here because the platform's exact blocking list is not published.
GATE_ADVISORY = {"difficult"}

REQUIRED_TOP = ("task.toml", "instruction.md", "environment", "solution", "tests")
MAX_FILE_CHARS = 60_000          # per file shown to a reviewer
DEFAULT_VIEW_CHARS = 280_000     # per LLM view
TEXT_SUFFIXES = {".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".jsonl", ".py", ".sh", ".cfg",
                 ".ini", ".js", ".ts", ".tsx", ".go", ".rs", ".java", ".c", ".h", ".cc", ".cpp",
                 ".hpp", ".cs", ".rb", ".sql", ".csv", ".tsv", ".xml", ".html", ".css", ".lock",
                 ".r", ".jl", ".kt", ".scala", ".swift", ".php", ".lua", ".pl", ".v", ".sv",
                 ".vhd", ".mk", ".cmake", ".gradle", ".proto", ".graphql", ".env", ".conf", ""}


# ----------------------------------------------------------------------------- package
class PackageError(Exception):
    pass


def safe_extract(zip_bytes: bytes, dest: Path) -> Path:
    """Extract without following absolute paths or '..' (zip slip); return the task root."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for info in zf.infolist():
            name = info.filename
            if not name or name.startswith("__MACOSX/") or name.endswith(".DS_Store"):
                continue
            target = (dest / name).resolve()
            if root not in target.parents and target != root:
                raise PackageError(f"zip entry escapes the archive: {name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                continue  # symlink entries are skipped, never followed
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
    return find_task_root(dest)


def find_task_root(dest: Path) -> Path:
    if (dest / "task.toml").is_file():
        return dest
    kids = [p for p in dest.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if len(kids) == 1 and (kids[0] / "task.toml").is_file():
        return kids[0]
    return dest


def zip_layout_notes(zip_bytes: bytes) -> list[str]:
    """What the platform upload will say about the archive layout."""
    notes = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [n for n in zf.namelist() if n and not n.startswith("__MACOSX/")]
    tops = {n.split("/")[0] for n in names}
    if not (set(REQUIRED_TOP) & tops) and len(tops) == 1:
        notes.append(f"Everything is nested under '{next(iter(tops))}/'. Zip the files inside the task "
                     "folder (task.toml, instruction.md, environment/, solution/, tests/), not the folder.")
    return notes


def is_text(path: Path) -> bool:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"Dockerfile", "Makefile"} \
            and not path.name.startswith("Dockerfile"):
        return False
    try:
        chunk = path.read_bytes()[:4096]
    except OSError:
        return False
    return b"\x00" not in chunk


def numbered(text: str) -> str:
    return "\n".join(f"{i:>4}| {line}" for i, line in enumerate(text.splitlines(), 1))


@dataclass
class Package:
    root: Path
    files: dict[str, str] = field(default_factory=dict)      # rel path -> text
    binaries: dict[str, int] = field(default_factory=dict)   # rel path -> size

    @classmethod
    def load(cls, root: Path) -> "Package":
        pkg = cls(root=root)
        for p in sorted(root.rglob("*")):
            if not p.is_file() or "/.git/" in f"/{p.relative_to(root)}":
                continue
            rel = p.relative_to(root).as_posix()
            if is_text(p):
                pkg.files[rel] = p.read_text(encoding="utf-8", errors="replace")
            else:
                pkg.binaries[rel] = p.stat().st_size
        return pkg

    def pick(self, *prefixes: str, exclude: tuple[str, ...] = ()) -> list[str]:
        out = []
        for rel in self.files:
            if any(rel == p or rel.startswith(p) for p in prefixes) and \
                    not any(rel == e or rel.startswith(e) for e in exclude):
                out.append(rel)
        return out

    def render(self, rels: list[str], budget: int) -> str:
        """Files with line numbers inside a character budget; the listing names what was cut."""
        parts, used, cut = [], 0, []
        for rel in rels:
            text = self.files[rel]
            body = numbered(text[:MAX_FILE_CHARS])
            if len(text) > MAX_FILE_CHARS:
                body += f"\n     | ... [file truncated: {len(text) - MAX_FILE_CHARS} more characters]"
            block = f"===== FILE: {rel} =====\n{body}\n"
            if used + len(block) > budget:
                cut.append(rel)
                continue
            parts.append(block)
            used += len(block)
        listing = [f"{rel} ({len(self.files[rel])} chars{', NOT SHOWN: budget' if rel in cut else ''})"
                   for rel in rels]
        listing += [f"{rel} (binary, {size} bytes, not shown)" for rel, size in self.binaries.items()
                    if any(rel.startswith(r.split('/')[0]) for r in rels)]
        return "FILES IN THIS VIEW\n" + "\n".join(listing) + "\n\n" + "".join(parts)

    # The split views the panel uses (judge guide: reviewers see different parts on purpose).
    def contract_rels(self) -> list[str]:
        return [r for r in ("instruction.md", "task.toml") if r in self.files] + self.pick("environment/")

    def view(self, name: str, budget: int) -> str:
        contract = self.contract_rels()
        tests = self.pick("tests/")
        solution = self.pick("solution/")
        if name == "contract":
            rels = contract
        elif name in ("coherent_contract", "sound_verifier"):
            rels = contract + tests
        elif name == "correct_reference_solution":
            rels = contract + solution
        elif name == "protected_ground_truth":
            rels = [r for r in ("task.toml",) if r in self.files] + self.pick("environment/") + tests
        else:
            rels = contract + solution + tests
        seen, ordered = set(), []
        for r in rels:
            if r not in seen:
                seen.add(r)
                ordered.append(r)
        return self.render(ordered, budget)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[`*_]", "", text)).strip().lower()


def docstring_quote_findings(pkg: "Package") -> list[dict[str, Any]]:
    """Q01: a test docstring that quotes the contract ("...") must quote text that is really in
    instruction.md or environment/. A quote with no source means the rule it tests is not stated
    (coherent_contract), or the contract changed and the test did not."""
    import ast as _ast

    visible = _norm(" ".join(pkg.files[r] for r in pkg.contract_rels() if r != "task.toml"))
    out = []
    for rel in pkg.pick("tests/"):
        if not rel.endswith(".py"):
            continue
        try:
            tree = _ast.parse(pkg.files[rel])
        except SyntaxError:
            continue
        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                continue
            doc = _ast.get_docstring(node) or ""
            for q in re.findall(r'["\u201c]([^"\u201d]{25,})["\u201d]', doc):
                if len(q.split()) < 6:
                    continue
                pieces = [x for x in re.split(r"\s*(?:\.\.\.|…)\s*", q) if len(x.split()) >= 4]
                if pieces and all(_norm(x) not in visible for x in pieces):
                    out.append({"severity": "ERROR", "code": "Q01", "where": f"{rel}:{node.lineno} {node.name}",
                                "message": f"docstring quotes contract text that is not in instruction.md or environment/: "
                                           f"\"{q[:120]}\" -- the rule may be unstated, or the contract changed",
                                "source": "Quality panel: coherent_contract (every graded rule must be stated)"})
    return out


# ----------------------------------------------------------------------------- LLM plumbing
def parse_json(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def call_llm(client: Any, model: str, prompt: str, max_tokens: int = 8000) -> dict[str, Any]:
    from config import chat_completion_kwargs

    messages = [{"role": "system", "content": P.SYSTEM}, {"role": "user", "content": prompt}]
    last: Exception | None = None
    for attempt in range(3):
        kwargs = chat_completion_kwargs(model, messages, max_output_tokens=max_tokens, temperature=0.1)
        if attempt:
            kwargs.pop("temperature", None)  # some reasoning models accept only the default
        try:
            try:
                resp = client.chat.completions.create(response_format={"type": "json_object"}, **kwargs)
            except Exception as e:
                if "response_format" not in str(e):
                    raise
                resp = client.chat.completions.create(**kwargs)
            return parse_json(resp.choices[0].message.content or "")
        except Exception as e:  # retry transient and parameter errors
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed: {last}")


def gate_criteria() -> list[dict[str, str]]:
    data = tomllib.loads(RUBRIC_TOML.read_text())
    return [c for c in data["criteria"] if c["name"] not in GATE_SKIPPED]


# ----------------------------------------------------------------------------- the review
@dataclass
class Review:
    task_name: str
    created: str
    static: dict[str, Any] = field(default_factory=dict)
    layout_notes: list[str] = field(default_factory=list)
    similarity: dict[str, Any] | None = None
    obligations: dict[str, Any] | None = None
    axes: dict[str, dict[str, Any]] = field(default_factory=dict)
    gate: list[dict[str, Any]] = field(default_factory=list)
    llmaj: list[dict[str, Any]] = field(default_factory=list)
    instruction: dict[str, Any] | None = None
    difficulty: dict[str, Any] | None = None
    errors: list[str] = field(default_factory=list)
    llm_ran: bool = False
    model: str = ""
    instruction_text: str = ""
    coverage: list[dict[str, Any]] = field(default_factory=list)

    # --- verdicts, applying the platform's blocking rules
    def static_blocking(self) -> list[dict[str, Any]]:
        return [f for f in self.static.get("findings", []) if f["severity"] == "ERROR"]

    def axis_blocking(self) -> dict[str, str]:
        out = {}
        for key, res in self.axes.items():
            sev = res.get("verdict", "None")
            if sev in P.AXES[key]["blocks_on"]:
                out[key] = sev
        return out

    def gate_failures(self) -> list[dict[str, Any]]:
        return [c for c in self.gate if c.get("result") == "fail" and not c.get("refuted")
                and c.get("name") not in GATE_ADVISORY]

    def llmaj_failures(self) -> list[dict[str, Any]]:
        return [c for c in self.llmaj if c.get("result") == "fail" and not c.get("refuted")]

    def missing_stages(self) -> list[str]:
        out = [f"panel:{k}" for k in P.AXES if k not in self.axes]
        if not self.gate:
            out.append("quality gate")
        if not self.llmaj:
            out.append("LLMaJ")
        if self.obligations is None or not self.obligations.get("obligations"):
            out.append("obligations")
        if self.instruction is None:
            out.append("instruction style")
        if self.difficulty is None:
            out.append("difficulty")
        return out

    def similarity_blocked(self) -> bool:
        return bool(self.similarity and self.similarity.get("blocked"))

    def verdict(self) -> tuple[str, list[str]]:
        reasons = []
        if self.static_blocking():
            reasons.append(f"{len(self.static_blocking())} static/CI error(s) would stop the upload at CI or preflight.")
        if self.similarity_blocked():
            reasons.append("The instruction is too similar to an existing team task.")
        if self.llm_ran:
            if self.llmaj_failures():
                reasons.append(f"{len(self.llmaj_failures())} stb LLMaJ check(s) fail.")
            if self.gate_failures():
                reasons.append(f"{len(self.gate_failures())} quality-gate criteria fail.")
            for key, sev in self.axis_blocking().items():
                reasons.append(f"Quality panel {key}: {sev} (blocks).")
        if reasons:
            return "NOT READY", reasons
        if not self.llm_ran:
            return "STATIC ONLY", ["Static checks pass; the LLM review did not run, so the panel stages are unchecked."]
        missing = self.missing_stages()
        if missing:
            return "INCOMPLETE", [f"These LLM stages did not finish, so nothing is known about them: {', '.join(missing)}. "
                                  "See Run notes (often an API key, quota or model problem) and run again."]
        return "READY TO UPLOAD", ["No blocking finding in this rehearsal. The platform still runs the oracle, "
                                   "the real panel (two models per axis) and the 8-run difficulty gate."]


def run_review(
    zip_bytes: bytes,
    zip_name: str,
    work_dir: Path,
    *,
    rubric_text: str = "",
    similarity_fn: Callable[[str], dict[str, Any]] | None = None,
    client: Any = None,
    model: str = "",
    run_llm: bool = True,
    view_chars: int = DEFAULT_VIEW_CHARS,
    parallel: int = 6,
    skeptic: bool = True,
    on_progress: Callable[[str, float], None] | None = None,
) -> Review:
    progress = on_progress or (lambda *_: None)
    review = Review(task_name=Path(zip_name).stem, created=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                    model=model)
    progress("Unpacking the archive", 0.03)
    review.layout_notes = zip_layout_notes(zip_bytes)
    root = safe_extract(zip_bytes, work_dir / "pkg")
    zpath = work_dir / "upload.zip"
    zpath.write_bytes(zip_bytes)

    progress("Static and CI checks", 0.08)
    review.static = t3_static.run_all(root, rubric_text=rubric_text, zip_path=zpath)
    try:
        name = tomllib.loads((root / "task.toml").read_text()).get("name")
        if name:
            review.task_name = str(name)
    except Exception:
        pass

    pkg = Package.load(root)
    review.static["findings"].extend(docstring_quote_findings(pkg))
    review.static["warnings"] = sum(1 for f in review.static["findings"] if f["severity"] == "WARN")
    review.static["errors"] = sum(1 for f in review.static["findings"] if f["severity"] == "ERROR")
    instruction = pkg.files.get("instruction.md", "")
    review.instruction_text = instruction
    if similarity_fn and instruction.strip():
        progress("Instruction similarity", 0.14)
        try:
            review.similarity = similarity_fn(instruction)
        except Exception as e:
            review.errors.append(f"similarity check failed: {e}")

    missing = [r for r in REQUIRED_TOP if not (root / r).exists()]
    if not (run_llm and client and model):
        return review
    if missing:
        review.errors.append(f"LLM review skipped: package is missing {', '.join(missing)}.")
        return review

    review.llm_ran = True
    progress("Extracting obligations from the contract", 0.2)
    try:
        review.obligations = call_llm(client, model, P.OBLIGATIONS + "\n\nCANDIDATE-VISIBLE TEXT\n"
                                      + pkg.view("contract", view_chars), max_tokens=8000)
    except Exception as e:
        review.errors.append(f"obligation extraction failed: {e}")
        review.obligations = {"obligations": [], "crux": ""}
    ob_text = "\n".join(f"{o.get('id')}: [{o.get('kind')}{', core' if o.get('core') else ''}] {o.get('text')} "
                        f"({o.get('citation')})" for o in review.obligations.get("obligations", [])) or "(none extracted)"

    jobs: dict[str, Callable[[], Any]] = {}
    for key, axis in P.AXES.items():
        prompt = P.AXIS_PROMPT.format(axis=key, label=axis["label"], question=axis["question"],
                                      view=axis["view"], checklist=axis["checklist"],
                                      severity=P.SEVERITY_RULES, obligations=ob_text,
                                      schema=P.FINDING_SCHEMA, package=pkg.view(key, view_chars))
        jobs[f"axis:{key}"] = (lambda p=prompt: call_llm(client, model, p, max_tokens=9000))
    criteria = gate_criteria()
    crit_text = "\n\n".join(f"### {c['name']}\n{c['description']}\n{c['guidance'].strip()}" for c in criteria)
    jobs["gate"] = lambda: call_llm(client, model, P.GATE_PROMPT.format(
        criteria=crit_text, package=pkg.view("all", view_chars)), max_tokens=12000)
    llmaj_text = "\n".join(f"- {n}: {d}" for n, d in P.LLMAJ_CHECKS)
    jobs["llmaj"] = lambda: call_llm(client, model, P.LLMAJ_PROMPT.format(
        checks=llmaj_text, package=pkg.view("all", view_chars)), max_tokens=5000)
    prose = ""
    try:
        meta = tomllib.loads(pkg.files.get("task.toml", "")).get("metadata", {})
        prose = "\n\n".join(f"{k}:\n{meta.get(k, '')}" for k in (
            "difficulty_explanation", "solution_explanation", "verification_explanation", "relevant_experience"))
    except Exception:
        pass
    jobs["instruction"] = lambda: call_llm(client, model, P.INSTRUCTION_PROMPT.format(
        instruction=numbered(instruction), prose=prose), max_tokens=4000)
    jobs["difficulty"] = lambda: call_llm(client, model, P.DIFFICULTY_PROMPT.format(
        package=pkg.view("all", view_chars)), max_tokens=5000)

    core = [o for o in review.obligations.get("obligations", []) if o.get("core")]
    for i in range(0, len(core), 10):
        batch = core[i:i + 10]
        text = "\n".join(f"{o.get('id')}: {o.get('text')} (quote: \"{o.get('quote', '')}\", {o.get('citation')})"
                         for o in batch)
        jobs[f"coverage:{i // 10}"] = (lambda t=text: call_llm(client, model, P.COVERAGE_PROMPT.format(
            obligations=t, package=pkg.view("sound_verifier", view_chars)), max_tokens=7000))

    coverage: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(fn): key for key, fn in jobs.items()}
        for fut in as_completed(futs):
            key = futs[fut]
            done += 1
            progress(f"LLM review {done}/{len(jobs)}: {key}", 0.25 + 0.7 * done / len(jobs))
            try:
                res = fut.result()
            except Exception as e:
                review.errors.append(f"{key} failed: {e}")
                continue
            if key.startswith("axis:"):
                res.setdefault("findings", [])
                res["verdict"] = _worst(res)
                review.axes[key[5:]] = res
            elif key == "gate":
                names = {c["name"] for c in criteria}
                review.gate = [c for c in res.get("criteria", []) if c.get("name") in names]
            elif key == "llmaj":
                review.llmaj = res.get("checks", [])
            elif key == "instruction":
                review.instruction = res
            elif key == "difficulty":
                review.difficulty = res
            elif key.startswith("coverage:"):
                coverage.extend(res.get("coverage", []))
    review.coverage = coverage
    if "sound_verifier" in review.axes:
        _coverage_findings(review.axes["sound_verifier"], review.obligations or {}, coverage)
        review.axes["sound_verifier"]["verdict"] = _worst(review.axes["sound_verifier"])
    if skeptic:
        progress("Skeptic pass: trying to refute every blocking claim", 0.93)
        run_skeptic(review, pkg, client, model, view_chars, parallel)
    progress("Done", 1.0)
    return review


def run_skeptic(review: Review, pkg: Package, client: Any, model: str, view_chars: int, parallel: int) -> None:
    """Send every Minor/Major finding and every failed gate or LLMaJ item to a refuter that must
    quote the lines proving it. Refuted items stay in the report but stop counting as blocking."""
    batches: dict[str, tuple[str, list[tuple[str, dict[str, Any], str]]]] = {}
    for key, res in review.axes.items():
        items = [(f"{key}#{i}", f, f"[{f.get('severity')}] {f.get('title')}: {f.get('mechanism')} "
                                  f"(cited {', '.join(f.get('citations') or [])})")
                 for i, f in enumerate(res.get("findings", []), 1) if f.get("severity") in ("Minor", "Major")]
        if items:
            batches[f"axis:{key}"] = (key, items)
    gate_items = [(f"gate:{c.get('name')}", c, f"Quality-gate criterion {c.get('name')} FAILS: {c.get('reasoning')}")
                  for c in review.gate if c.get("result") == "fail"]
    llmaj_items = [(f"llmaj:{c.get('name')}", c, f"LLMaJ check {c.get('name')} FAILS: {c.get('reasoning')}")
                   for c in review.llmaj if c.get("result") == "fail"]
    if gate_items or llmaj_items:
        batches["checks"] = ("all", gate_items + llmaj_items)
    if not batches:
        return

    def one(view_name: str, items: list[tuple[str, dict[str, Any], str]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        claims = "\n".join(f"{cid}: {text}" for cid, _, text in items)
        res = call_llm(client, model, P.SKEPTIC_PROMPT.format(claims=claims, package=pkg.view(view_name, view_chars)),
                       max_tokens=6000)
        by_id = {v.get("id"): v for v in res.get("verdicts", [])}
        return [(obj, by_id.get(cid, {"status": "unanswered"})) for cid, obj, _ in items]

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        futs = {ex.submit(one, view, items): name for name, (view, items) in batches.items()}
        for fut in as_completed(futs):
            try:
                pairs = fut.result()
            except Exception as e:
                review.errors.append(f"skeptic pass {futs[fut]} failed (claims kept as found): {e}")
                continue
            for obj, verdict in pairs:
                obj["skeptic"] = verdict
                obj["refuted"] = verdict.get("status") == "refuted"
    for key, res in review.axes.items():
        res["verdict"] = _worst(res)


def _coverage_findings(res: dict[str, Any], obligations: dict[str, Any], coverage: list[dict[str, Any]]) -> None:
    """A core obligation whose plausible wrong implementation no visible test catches becomes a
    Minor sound_verifier finding (the skeptic pass then checks it). Obligations already named in a
    finding are left to that finding."""
    core = {o.get("id"): o for o in obligations.get("obligations", []) if o.get("core")}
    named = {oid for f in res.get("findings", []) for oid in (f.get("obligations") or [])}
    for c in coverage:
        oid = c.get("id")
        if c.get("caught") != "no" or oid not in core or oid in named:
            continue
        o = core[oid]
        res["findings"].append({
            "severity": "Minor",
            "title": f"No visible test catches a wrong {oid}: {o.get('text', '')[:120]}",
            "mechanism": f"Plausible wrong implementation: {c.get('wrong_impl', '')} It would still pass every "
                         "visible test; the panel's sound_verifier axis reports that as a blocking gap.",
            "citations": [o.get("citation", "")],
            "fix": "Add a focused test with a case whose outcome flips when this rule is wrong, or cut the promise.",
            "obligations": [oid],
            "source": "coverage",
        })


def _worst(res: dict[str, Any]) -> str:
    order = {"None": 0, "Advisory": 1, "Minor": 2, "Major": 3}
    worst = "None"
    for f in res.get("findings", []):
        if f.get("refuted"):
            continue
        sev = f.get("severity", "Advisory")
        if order.get(sev, 0) > order[worst]:
            worst = sev
    return worst


# ----------------------------------------------------------------------------- reports
def to_dict(r: Review) -> dict[str, Any]:
    verdict, reasons = r.verdict()
    sim = None
    if r.similarity:
        sim = {k: v for k, v in r.similarity.items() if k in ("blocked", "message", "corpus_size", "corpus_count")}
        sim["top_matches"] = [
            {"task": getattr(m, "task_id", ""), "trainer": getattr(m, "trainer", ""),
             "word_overlap": round(getattr(m, "lexical_score", None) or getattr(m, "score", 0.0), 3),
             "meaning": None if getattr(m, "semantic_score", None) is None else round(m.semantic_score, 3)}
            for m in (r.similarity.get("matches") or [])[:5]]
    return {
        "task": r.task_name, "created": r.created, "model": r.model,
        "verdict": verdict, "reasons": reasons,
        "static": r.static, "zip_layout": r.layout_notes, "similarity": sim,
        "obligations": r.obligations, "quality_panel": r.axes, "quality_gate": r.gate,
        "llmaj": r.llmaj, "instruction_style": r.instruction, "difficulty": r.difficulty,
        "coverage": r.coverage,
        "errors": r.errors,
        "not_checked_here": ["docker build", "oracle x3 and NOP runs", "8 real-agent difficulty runs",
                             "mutant and alternative execution", "platform AI-text screening"],
    }


def _e(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def render_html(r: Review) -> str:
    verdict, reasons = r.verdict()
    tone = {"READY TO UPLOAD": "#15803d", "STATIC ONLY": "#b45309", "INCOMPLETE": "#b45309",
            "NOT READY": "#b91c1c"}[verdict]
    out = [f"""<!doctype html><html><head><meta charset="utf-8"><title>{_e(r.task_name)} - Terminus 3 QC</title>
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#0f172a}}
h1{{font-size:22px}}h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #e2e8f0;padding-bottom:4px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}td,th{{border:1px solid #e2e8f0;padding:6px 8px;vertical-align:top;text-align:left}}
th{{background:#f8fafc}}.b{{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600;color:#fff}}
.Major,.ERROR,.fail{{background:#b91c1c}}.Minor,.WARN{{background:#c2410c}}.Advisory,.INFO{{background:#64748b}}.None,.pass{{background:#15803d}}
code{{background:#f1f5f9;padding:0 4px;border-radius:4px}}</style></head><body>
<h1>{_e(r.task_name)} &middot; Terminus 3 task QC</h1>
<p>{_e(r.created)} &middot; model <code>{_e(r.model or 'none')}</code></p>
<p style="font-size:18px;font-weight:700;color:{tone}">{_e(verdict)}</p><ul>"""]
    out += [f"<li>{_e(x)}</li>" for x in reasons] + ["</ul>"]
    out.append("<p><em>A rehearsal of the platform stages. It does not build images, run the oracle or NOP, "
               "execute mutants or run real agents.</em></p>")
    out.append("<h2>1. Static and CI checks</h2>")
    for n in r.layout_notes:
        out.append(f"<p><span class='b ERROR'>ERROR</span> {_e(n)}</p>")
    rows = "".join(f"<tr><td><span class='b {f['severity']}'>{f['severity']}</span></td><td>{_e(f['code'])}</td>"
                   f"<td>{_e(f['where'])}</td><td>{_e(f['message'])}</td><td>{_e(f['source'])}</td></tr>"
                   for f in r.static.get("findings", []) if f["severity"] != "INFO")
    out.append(f"<p>{r.static.get('errors', 0)} error(s), {r.static.get('warnings', 0)} warning(s).</p>")
    if rows:
        out.append(f"<table><tr><th>Sev</th><th>Code</th><th>Where</th><th>Finding</th><th>Rule</th></tr>{rows}</table>")
    if r.similarity:
        out.append("<h2>2. Instruction similarity</h2>")
        out.append(f"<p>{_e(r.similarity.get('message', ''))}</p>")
    if r.llm_ran:
        out.append("<h2>3. Quality panel rehearsal</h2>")
        for key, axis in P.AXES.items():
            res = r.axes.get(key)
            if not res:
                out.append(f"<p><b>{_e(key)}</b>: not completed</p>")
                continue
            blocks = res["verdict"] in axis["blocks_on"]
            out.append(f"<h3>{_e(key)} <span class='b {res['verdict']}'>{_e(res['verdict'])}</span>"
                       f"{' (blocks)' if blocks else ''}</h3><p>{_e(res.get('summary', ''))}</p>")
            for i, f in enumerate(res.get("findings", []), 1):
                sk = f.get("skeptic") or {}
                tag = " <i>(refuted by the skeptic pass: " + _e(sk.get("reason")) + ")</i>" if f.get("refuted") else (
                    " <i>(confirmed: " + _e(sk.get("evidence")) + ")</i>" if sk.get("status") == "confirmed" else "")
                out.append(f"<p><span class='b {_e(f.get('severity'))}'>{_e(f.get('severity'))}</span> <b>{i}. {_e(f.get('title'))}</b>{tag}<br>"
                           f"{_e(f.get('mechanism'))}<br><small>Cited: {_e(', '.join(f.get('citations') or []))}</small><br>"
                           f"<small>Fix: {_e(f.get('fix'))}</small></p>")
        out.append("<h2>4. Quality gate (29 public criteria)</h2><table><tr><th>Criterion</th><th>Result</th><th>Reasoning</th></tr>")
        for c in r.gate:
            res = c.get("result", "?")
            note = " (advisory)" if c.get("name") in GATE_ADVISORY else ""
            if c.get("refuted"):
                note += " (fail refuted by the skeptic pass: " + _e((c.get("skeptic") or {}).get("reason")) + ")"
            out.append(f"<tr><td>{_e(c.get('name'))}{note}</td><td><span class='b {res}'>{_e(res)}</span></td><td>{_e(c.get('reasoning'))}</td></tr>")
        out.append("</table><h2>5. stb LLMaJ checks</h2><table><tr><th>Check</th><th>Result</th><th>Reasoning</th></tr>")
        for c in r.llmaj:
            res = c.get("result", "?")
            out.append(f"<tr><td>{_e(c.get('name'))}</td><td><span class='b {res}'>{_e(res)}</span></td><td>{_e(c.get('reasoning'))}</td></tr>")
        out.append("</table>")
        if r.instruction:
            out.append(f"<h2>6. Instruction style</h2><p>Verdict: <b>{_e(r.instruction.get('verdict'))}</b>, AI-text risk "
                       f"<b>{_e(r.instruction.get('ai_text_risk'))}</b>. {_e(r.instruction.get('summary'))}</p><ul>")
            out += [f"<li>[{_e(i.get('severity'))}] {_e(i.get('issue'))} <small>({_e(i.get('where'))})</small> Fix: {_e(i.get('fix'))}</li>"
                    for i in r.instruction.get("issues", [])]
            out.append("</ul>")
        if r.difficulty:
            d = r.difficulty
            out.append(f"<h2>7. Expertise floor and difficulty</h2><p>Expertise floor: <b>{_e(d.get('expertise_floor'))}</b>; "
                       f"difficulty risk: <b>{_e(d.get('difficulty_risk'))}</b>; explanation: <b>{_e(d.get('explanation_quality'))}</b></p>"
                       f"<p>Crux: {_e(d.get('crux'))}</p><p>{_e(d.get('reasoning'))}</p><ul>")
            out += [f"<li>{_e(s)}</li>" for s in d.get("suggestions", [])]
            out.append("</ul>")
        if r.obligations:
            out.append("<h2>8. Obligations the panel will extract</h2><table><tr><th>Id</th><th>Kind</th><th>Obligation</th><th>Source</th></tr>")
            for o in r.obligations.get("obligations", []):
                out.append(f"<tr><td>{_e(o.get('id'))}</td><td>{_e(o.get('kind'))}{' core' if o.get('core') else ''}</td>"
                           f"<td>{_e(o.get('text'))}</td><td>{_e(o.get('citation'))}</td></tr>")
            out.append("</table>")
    if r.errors:
        out.append("<h2>Run notes</h2><ul>" + "".join(f"<li>{_e(e)}</li>" for e in r.errors) + "</ul>")
    out.append("</body></html>")
    return "".join(out)
