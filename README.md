# Terminus Trainer QC Portal

Professional web portal for trainers to QC task zips before platform submission.

**Primary check:** LLM alignment judge (9 prompts) comparing your task against **all 8 accepted reference tasks** with exact file-level issues.

## What trainers see

1. Open the shared link
2. Upload task `.zip`
3. Click **Run Full QC Assessment**
4. Download `<task-name>_qc_report.html`

Trainers never see or enter an API key.

## What it checks

| Layer | Role | Method |
|-------|------|--------|
| **LLM Alignment** | Primary | 9 prompts vs 8 accepted refs + cross-file alignment |
| **Similarity** | Secondary | TF-IDF on instruction + SPEC vs team sheet |
| **Static** | Quick gate | 21 regex/rule checks |

### LLM alignment (9 checks)

1. **Instruction vs All 8 References** — per-task comparison against every accepted task
2. Accepted Task Pattern — 3-paragraph structure
3. Instruction ↔ SPEC
4. Instruction ↔ Tests
5. SPEC ↔ Tests
6. Solution ↔ Instruction
7. Solution ↔ SPEC
8. Solution ↔ Tests
9. Environment ↔ Task

Default model: **`gpt-4.1`** (configurable — see below).

---

## Admin setup — API key (trainers must NOT see this)

### Option A: Streamlit Cloud (recommended for sharing)

1. Push this repo to GitHub (see **Publish to GitHub** below)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Point to repo, set **Main file path**: `app.py`
4. **App Settings → Secrets** — paste (replace with your real key):

```toml
OPENAI_API_KEY = "sk-proj-..."
QC_LLM_MODEL = "gpt-5.2"
QC_EMBED_MODEL = "text-embedding-3-small"
```

Remove any `OPENROUTER_API_KEY` or `OPENAI_BASE_URL` lines if you switched back from OpenRouter.

Plus sheet block:
```toml
[sheet]
url = "https://docs.google.com/spreadsheets/d/1XR_EFXtUt4GQ_d6zlkPT4arZi-BG-zFGHx4ctSSG51s/edit"
worksheet = "May 1st - 31st"
task_col = "Task name"
instruction_col = "Task Instruction"
trainer_col = "Trainer Name"
instruction_col_index = "16"
```

5. Deploy → share the public URL with trainers

### Option B: Local / server

```bash
cd trainer-qc-portal
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit secrets.toml — add your OPENAI_API_KEY
pip install -r requirements.txt
streamlit run app.py
```

Or use environment variables (no file):

```bash
export OPENAI_API_KEY="sk-proj-..."
export QC_LLM_MODEL="gpt-5.2"
export QC_EMBED_MODEL="text-embedding-3-small"
streamlit run app.py
```

### Model choice

| Model | When to use |
|-------|-------------|
| `gpt-4.1` | Default — best balance of quality + speed for code review |
| `gpt-5.1` | Set `QC_LLM_MODEL = "gpt-5.1"` if your OpenAI org has it |
| `o4-mini` | Slower, more reasoning-heavy reviews |

Set via `QC_LLM_MODEL` in secrets or env — trainers never choose the model.

---

## Hosting options

| Platform | Cost | Best for |
|----------|------|----------|
| **Streamlit Cloud** | Free | Sharing with team — one URL, secrets in dashboard |
| **Railway / Render** | ~$5/mo | Private deployment with env vars |
| **Local machine** | Free | Dev/testing — `streamlit run app.py` |

**Recommended:** Streamlit Cloud — zero ops, secrets hidden, shareable link.

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Trainer-facing UI |
| `qc_engine.py` | Assessment engine |
| `alignment_prompts.py` | 9 LLM prompts with accepted references |
| `accepted_references.json` | All 8 accepted tasks (instruction, toml, solve.sh, Dockerfile, test metadata) |
| `config.py` | Admin secrets resolution |
| `.streamlit/secrets.toml` | Local admin config (gitignored) |

## Google Sheet (similarity) — Terminus Task Tracker

Pre-wired to **Project Terminus Task Tracker** (Apps Script v2):

| Setting | Value |
|---------|-------|
| Sheet ID | `1XR_EFXtUt4GQ_d6zlkPT4arZi-BG-zFGHx4ctSSG51s` |
| Tab | `May 1st - 31st` |
| Task column | `Task Name` (col D) |
| Instruction column | `Task Instruction` (col P / 16) |
| Trainer column | `Trainer Name` (col A) |

Sheet must be **anyone with the link can view** (CSV export). Instruction similarity runs **twice**: (1) paste/upload `instruction.md` before zip, (2) re-check from zip. Lexical and embedding run **in parallel**; if **both** are ≥ **60%** on the same tracker row → **change the task**. LLM alignment judge is the **primary** pass/fail gate. Configured in admin secrets only.

## Publish to GitHub

This folder is a **standalone git repo** (not the full Terminus workspace).

```bash
cd trainer-qc-portal   # or clone after you create the remote repo
git add .
git commit -m "Add Terminus Trainer QC Portal"
git remote add origin https://github.com/YOUR_USER/terminus-trainer-qc-portal.git
git push -u origin main
```

**Never commit** `.streamlit/secrets.toml` — it is gitignored. Use `secrets.toml.example` as the template.

On GitHub: **Settings → Secrets and variables** is not used for Streamlit; add secrets in **Streamlit Cloud → App Settings → Secrets** after connecting the repo.
