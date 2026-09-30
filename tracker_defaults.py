"""Terminus Task Tracker sheet layout — matches Project Terminus Apps Script v2.

Do NOT put API keys or sync tokens here. Sheet must be shared:
  Anyone with the link can view (for CSV export).
"""

from __future__ import annotations

# From Apps Script: SPREADSHEET_ID, DATA_SHEET, TASK_INSTRUCTION_*
TRACKER_SPREADSHEET_ID = "1XR_EFXtUt4GQ_d6zlkPT4arZi-BG-zFGHx4ctSSG51s"
TRACKER_WORKSHEET = "Task Instructions"

TRACKER_SHEET_URL = (
    f"https://docs.google.com/spreadsheets/d/{TRACKER_SPREADSHEET_ID}/edit"
)

# Column headers (row 1) on the dedicated Task Instructions tab
#   A Task ID | B Task name | C Trainer Name | D Task Status |
#   E Language | F Domain | G Task Instruction | H Source
TRACKER_COL_TRAINER = 3          # C — Trainer Name
TRACKER_COL_TASK_ID = 1          # A — Task ID (UUID)
TRACKER_COL_TASK_STATUS = 4      # D
TRACKER_COL_TASK_NAME = 2        # B — Task Name
TRACKER_COL_TASK_INSTRUCTION = 7  # G — "Task Instruction"

TASK_INSTRUCTION_HEADER = "Task Instruction"

# Dual-block: BOTH lexical AND embedding >= 60% flags duplicate.
INSTRUCTION_SIM_THRESHOLD = 0.60
INSTRUCTION_LEXICAL_GATE = INSTRUCTION_SIM_THRESHOLD
INSTRUCTION_SEMANTIC_THRESHOLD = INSTRUCTION_SIM_THRESHOLD
# Meaning-only block: semantic >= 70% flags even when word overlap is lower.
INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD = 0.70
INSTRUCTION_SIM_WARN = INSTRUCTION_SIM_THRESHOLD
INSTRUCTION_SIM_BLOCK = INSTRUCTION_SIM_THRESHOLD

DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_EMBED_ENDPOINT = "https://api.openai.com/v1/embeddings"

# Default column names for CSV header row
DEFAULT_TASK_COL = "Task name"
DEFAULT_INSTRUCTION_COL = TASK_INSTRUCTION_HEADER
DEFAULT_TRAINER_COL = "Trainer Name"
