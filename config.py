"""Admin-only configuration — API keys and model come from secrets/env, never from trainers."""

from __future__ import annotations

import os
from typing import Any


DEFAULT_LLM_MODEL = "gpt-4.1"


def _get_secrets() -> dict[str, Any]:
    try:
        import streamlit as st

        return dict(st.secrets)
    except Exception:
        return {}


def resolve_openai_api_key() -> str:
    secrets = _get_secrets()
    return (
        secrets.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    ).strip()


def resolve_llm_model() -> str:
    secrets = _get_secrets()
    return (
        secrets.get("QC_LLM_MODEL", "")
        or os.environ.get("QC_LLM_MODEL", "")
        or DEFAULT_LLM_MODEL
    ).strip()


def resolve_sheet_defaults() -> dict[str, str]:
    from tracker_defaults import (
        DEFAULT_INSTRUCTION_COL,
        DEFAULT_TASK_COL,
        DEFAULT_TRAINER_COL,
        TRACKER_SHEET_URL,
        TRACKER_WORKSHEET,
    )

    secrets = _get_secrets()
    sheet = secrets.get("sheet", {})
    if isinstance(sheet, dict) and sheet:
        merged = {
            "url": str(sheet.get("url", "") or TRACKER_SHEET_URL),
            "worksheet": str(sheet.get("worksheet", "") or TRACKER_WORKSHEET),
            "task_col": str(sheet.get("task_col", "") or DEFAULT_TASK_COL),
            "instruction_col": str(sheet.get("instruction_col", "") or DEFAULT_INSTRUCTION_COL),
            "trainer_col": str(sheet.get("trainer_col", "") or DEFAULT_TRAINER_COL),
            "instruction_col_index": str(sheet.get("instruction_col_index", "16")),
            "spec_col": str(sheet.get("spec_col", "")),
        }
        return {k: v for k, v in merged.items() if v}
    return {
        "url": os.environ.get("QC_SHEET_URL", TRACKER_SHEET_URL),
        "worksheet": os.environ.get("QC_SHEET_WORKSHEET", TRACKER_WORKSHEET),
        "task_col": os.environ.get("QC_SHEET_TASK_COL", DEFAULT_TASK_COL),
        "instruction_col": os.environ.get("QC_SHEET_INSTRUCTION_COL", DEFAULT_INSTRUCTION_COL),
        "trainer_col": os.environ.get("QC_SHEET_TRAINER_COL", DEFAULT_TRAINER_COL),
        "instruction_col_index": os.environ.get("QC_SHEET_INSTRUCTION_COL_INDEX", "16"),
        "spec_col": os.environ.get("QC_SHEET_SPEC_COL", ""),
    }


def llm_configured() -> bool:
    return bool(resolve_openai_api_key())
