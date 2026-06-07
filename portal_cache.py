"""Streamlit caches for tracker corpus and embedding vectors."""

from __future__ import annotations

import hashlib
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
def cached_instruction_embeddings(
    corpus_signature: str,
    embed_model: str,
    texts: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Embed tracker instructions once per corpus revision (24h cache)."""
    from config import resolve_openai_api_key
    from similarity_engine import embed_texts_batch

    if not texts:
        return ()
    vectors = embed_texts_batch(list(texts), resolve_openai_api_key(), embed_model)
    return tuple(tuple(vector) for vector in vectors)
