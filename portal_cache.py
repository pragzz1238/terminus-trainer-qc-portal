"""Streamlit caches for tracker corpus and embedding vectors."""

from __future__ import annotations

import hashlib
from typing import Any

import streamlit as st


def corpus_text_signature(keys_and_texts: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for key, text in sorted(keys_and_texts):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_load_similarity_corpus(
    sheet_url: str,
    worksheet: str,
    task_col: str,
    instruction_col: str,
    spec_col: str,
    trainer_col: str,
    instruction_col_index: int,
    corpus_json_path: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, dict[str, str]], list[str]]:
    from qc_engine import load_similarity_corpus

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


@st.cache_data(ttl=86400, show_spinner=False)
def cached_tracker_embedding_vectors(
    corpus_signature: str,
    embed_model: str,
    sheet_url: str,
    worksheet: str,
    task_col: str,
    instruction_col: str,
    spec_col: str,
    trainer_col: str,
    instruction_col_index: int,
    corpus_json_path: str,
) -> tuple[tuple[str, ...], tuple[tuple[float, ...], ...]]:
    """Embed tracker rows once per corpus revision.

    Cache key uses only the signature + sheet coordinates — never the full instruction texts.
    """
    from config import resolve_openai_api_key
    from similarity_engine import embed_texts_batch

    _, _, meta, _ = cached_load_similarity_corpus(
        sheet_url=sheet_url,
        worksheet=worksheet,
        task_col=task_col,
        instruction_col=instruction_col,
        spec_col=spec_col,
        trainer_col=trainer_col,
        instruction_col_index=instruction_col_index,
        corpus_json_path=corpus_json_path,
    )
    items = sorted(
        (key, (meta[key].get("instruction") or "").strip())
        for key in meta
        if (meta[key].get("instruction") or "").strip()
    )
    if not items:
        return (), ()

    computed_sig = corpus_text_signature(items)
    if computed_sig != corpus_signature:
        # Sheet changed since the caller computed the signature — embed fresh corpus.
        pass

    keys = tuple(key for key, _ in items)
    texts = [text for _, text in items]
    vectors = embed_texts_batch(texts, resolve_openai_api_key(), embed_model)
    return keys, tuple(tuple(vector) for vector in vectors)


def tracker_cache_params(
    *,
    sheet_url: str = "",
    worksheet: str = "",
    task_col: str = "",
    instruction_col: str = "",
    spec_col: str = "",
    trainer_col: str = "",
    instruction_col_index: int = 16,
    corpus_json_path: str = "",
) -> dict[str, Any]:
    return {
        "sheet_url": sheet_url,
        "worksheet": worksheet,
        "task_col": task_col,
        "instruction_col": instruction_col,
        "spec_col": spec_col,
        "trainer_col": trainer_col,
        "instruction_col_index": int(instruction_col_index),
        "corpus_json_path": corpus_json_path,
    }
