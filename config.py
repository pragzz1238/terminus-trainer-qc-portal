"""Admin-only configuration — API keys and model come from secrets/env, never from trainers."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_LLM_MODEL = "gpt-4.1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_APP_URL = "https://cognyzer-terminus-trainer-qc-app.streamlit.app"


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
        or secrets.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
    ).strip()


def _explicit_base_url() -> str:
    secrets = _get_secrets()
    return (
        secrets.get("OPENAI_BASE_URL", "")
        or os.environ.get("OPENAI_BASE_URL", "")
    ).strip()


def uses_openrouter(api_key: str = "") -> bool:
    key = (api_key or resolve_openai_api_key()).strip()
    if key.startswith("sk-") and not key.startswith("sk-or-"):
        return False
    if key.startswith("sk-or-"):
        return True
    return "openrouter.ai" in _explicit_base_url()


def resolve_openai_base_url(api_key: str = "") -> str:
    explicit = _explicit_base_url()
    if explicit:
        return explicit.rstrip("/")
    key = (api_key or resolve_openai_api_key()).strip()
    if key.startswith("sk-or-"):
        return DEFAULT_OPENROUTER_BASE_URL
    return DEFAULT_OPENAI_BASE_URL


def resolve_embed_model(api_key: str = "") -> str:
    secrets = _get_secrets()
    explicit = (
        secrets.get("QC_EMBED_MODEL", "")
        or os.environ.get("QC_EMBED_MODEL", "")
    ).strip()
    if explicit:
        return explicit
    if uses_openrouter(api_key):
        return "openai/text-embedding-3-small"
    return "text-embedding-3-small"


def resolve_llm_model(api_key: str = "") -> str:
    secrets = _get_secrets()
    model = (
        secrets.get("QC_LLM_MODEL", "")
        or os.environ.get("QC_LLM_MODEL", "")
        or DEFAULT_LLM_MODEL
    ).strip()
    if uses_openrouter(api_key) and "/" not in model:
        return f"openai/{model}"
    return model


def build_openai_client(api_key: str = ""):
    """OpenAI-compatible client — works with OpenAI or OpenRouter."""
    from openai import OpenAI

    key = (api_key or resolve_openai_api_key()).strip()
    base_url = resolve_openai_base_url(key)
    kwargs: dict[str, Any] = {"api_key": key, "base_url": base_url}
    if uses_openrouter(key):
        kwargs["default_headers"] = {
            "HTTP-Referer": OPENROUTER_APP_URL,
            "X-Title": "Terminus QC Portal",
        }
    return OpenAI(**kwargs)


def api_provider_label(api_key: str = "") -> str:
    return "OpenRouter" if uses_openrouter(api_key) else "OpenAI"


def _model_slug(model: str) -> str:
    """Normalize provider-prefixed ids, e.g. openai/gpt-5.2 -> gpt-5.2."""
    return (model or "").strip().split("/")[-1].lower()


def uses_max_completion_tokens(model: str) -> bool:
    """GPT-5 / o-series chat models reject legacy max_tokens."""
    slug = _model_slug(model)
    return (
        slug.startswith("gpt-5")
        or slug.startswith("o1")
        or slug.startswith("o3")
        or slug.startswith("o4")
    )


def chat_completion_kwargs(
    model: str,
    messages: list[dict[str, str]],
    *,
    max_output_tokens: int = 2500,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Build chat.completions.create kwargs compatible with legacy and GPT-5 models."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if uses_max_completion_tokens(model):
        kwargs["max_completion_tokens"] = max_output_tokens
    else:
        kwargs["max_tokens"] = max_output_tokens
    return kwargs


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
