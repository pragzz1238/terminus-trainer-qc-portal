"""Instruction similarity — lexical + embedding run in parallel.

Block rules (same tracker row):
- BOTH word overlap and meaning >= 60%, OR
- meaning alone >= 70%.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tracker_defaults import (
    DEFAULT_EMBED_MODEL,
    INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD,
    INSTRUCTION_SIM_THRESHOLD,
)


@dataclass
class SimilarityHit:
    task_key: str
    trainer: str
    lexical_score: float
    semantic_score: float | None
    method: str
    instruction_preview: str
    matched_instruction: str = ""
    dual_block: bool = False
    block_reason: str = ""


@dataclass
class SimilarityRunMeta:
    api_key_present: bool
    embedding_ran: bool
    embed_model: str
    embedding_error: str | None
    corpus_size: int


@dataclass
class SimilarityCompareResult:
    hits: list[SimilarityHit]
    meta: SimilarityRunMeta


def normalize_instruction_text(text: str) -> str:
    """Same normalization as Apps Script normalizeInstructionText_."""
    return (
        re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", (text or "").lower()))
        .strip()
    )


def lexical_similarity(a: str, b: str) -> float:
    """Dice coefficient on word tokens — matches tracker instructionSimilarity_."""
    x = normalize_instruction_text(a)
    y = normalize_instruction_text(b)
    if not x or not y:
        return 0.0
    if x == y:
        return 1.0
    x_set = {t for t in x.split() if t}
    y_set = {t for t in y.split() if t}
    if not x_set or not y_set:
        return 0.0
    intersect = len(x_set & y_set)
    return (2 * intersect) / (len(x_set) + len(y_set))


def evaluate_similarity_block(
    lexical_score: float,
    semantic_score: float | None,
    dual_threshold: float = INSTRUCTION_SIM_THRESHOLD,
    semantic_block_threshold: float = INSTRUCTION_SEMANTIC_BLOCK_THRESHOLD,
) -> tuple[bool, str]:
    """Return (flagged, reason) where reason is '', 'dual', or 'meaning'."""
    if semantic_score is None:
        return False, ""
    if lexical_score >= dual_threshold and semantic_score >= dual_threshold:
        return True, "dual"
    if semantic_score >= semantic_block_threshold:
        return True, "meaning"
    return False, ""


def is_dual_duplicate(
    lexical_score: float,
    semantic_score: float | None,
    threshold: float = INSTRUCTION_SIM_THRESHOLD,
) -> bool:
    """Backward-compatible: True when evaluate_similarity_block would flag."""
    flagged, _ = evaluate_similarity_block(lexical_score, semantic_score, threshold)
    return flagged


def embed_texts_batch(
    texts: list[str],
    api_key: str,
    model: str | None = None,
) -> list[list[float]]:
    """Public batch embed helper (used by portal cache)."""
    return _get_embeddings(texts, api_key, model)


def _get_embeddings(
    texts: list[str],
    api_key: str,
    model: str | None = None,
) -> list[list[float]]:
    from config import build_openai_client, resolve_embed_model

    embed_model = model or resolve_embed_model(api_key)
    client = build_openai_client(api_key)
    out: list[list[float]] = []
    batch_size = 60
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        out.extend([item.embedding for item in resp.data])
    return out


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if not na or not nb:
        return 0.0
    return dot / (na * nb)


def _compute_lexical_scores(
    query: str,
    candidates: list[tuple[str, dict[str, str]]],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for key, meta in candidates:
        inst = meta.get("instruction", "")
        if not inst.strip():
            continue
        scores[key] = lexical_similarity(query, inst)
    return scores


def _compute_semantic_scores(
    query: str,
    candidates: list[tuple[str, dict[str, str]]],
    api_key: str,
    tracker_cache: dict[str, Any] | None = None,
) -> dict[str, float]:
    if not api_key or not candidates:
        return {}
    from config import resolve_embed_model

    embed_model = resolve_embed_model(api_key)
    keys = [key for key, _ in candidates]
    corpus_texts = [meta.get("instruction", "") for _, meta in candidates]

    if tracker_cache:
        try:
            from portal_cache import cached_tracker_embedding_vectors, corpus_text_signature

            signature = corpus_text_signature(list(zip(keys, corpus_texts)))
            cached_keys, cached_vectors = cached_tracker_embedding_vectors(
                signature,
                embed_model,
                **tracker_cache,
            )
            key_to_vector = dict(zip(cached_keys, cached_vectors))
            query_emb = _get_embeddings([query], api_key, embed_model)[0]
            return {
                key: _cosine(query_emb, list(key_to_vector[key]))
                for key in keys
                if key in key_to_vector
            }
        except Exception:
            pass

    texts = [query] + corpus_texts
    embeddings = _get_embeddings(texts, api_key, embed_model)
    source_emb = embeddings[0]
    return {
        key: _cosine(source_emb, embeddings[idx + 1])
        for idx, key in enumerate(keys)
    }


def compare_instruction_to_corpus(
    query_instruction: str,
    corpus: dict[str, dict[str, str]],
    exclude_keys: set[str] | None = None,
    api_key: str = "",
    top_n: int = 10,
    threshold: float = INSTRUCTION_SIM_THRESHOLD,
) -> list[SimilarityHit]:
    """Backward-compatible wrapper — returns ranked hits only."""
    return compare_instruction_to_corpus_full(
        query_instruction,
        corpus,
        exclude_keys=exclude_keys,
        api_key=api_key,
        top_n=top_n,
        threshold=threshold,
    ).hits


def compare_instruction_to_corpus_full(
    query_instruction: str,
    corpus: dict[str, dict[str, str]],
    exclude_keys: set[str] | None = None,
    api_key: str = "",
    top_n: int = 10,
    threshold: float = INSTRUCTION_SIM_THRESHOLD,
    tracker_cache: dict[str, Any] | None = None,
) -> SimilarityCompareResult:
    """
    Run lexical and embedding similarity in parallel against the full corpus.
    Always returns top-N rows with both scores visible (even below threshold).
    dual_block=True when dual >= 60% OR meaning >= 70%.
    """
    from config import resolve_embed_model, resolve_openai_api_key

    exclude = exclude_keys or set()
    query = (query_instruction or "").strip()
    api_key = (api_key or resolve_openai_api_key()).strip()
    embed_model = resolve_embed_model(api_key)
    empty_meta = SimilarityRunMeta(
        api_key_present=bool(api_key),
        embedding_ran=False,
        embed_model=embed_model,
        embedding_error=None,
        corpus_size=0,
    )
    if not query or not corpus:
        return SimilarityCompareResult(hits=[], meta=empty_meta)

    candidates: list[tuple[str, dict[str, str]]] = []
    for key, meta in corpus.items():
        if key in exclude:
            continue
        if not (meta.get("instruction", "") or "").strip():
            continue
        candidates.append((key, meta))

    if not candidates:
        return SimilarityCompareResult(hits=[], meta=empty_meta)

    lexical_scores: dict[str, float] = {}
    semantic_scores: dict[str, float] = {}
    embedding_error: str | None = None
    embedding_ran = False

    lexical_scores = _compute_lexical_scores(query, candidates)
    if api_key:
        try:
            semantic_scores = _compute_semantic_scores(
                query, candidates, api_key, tracker_cache=tracker_cache,
            )
            embedding_ran = len(semantic_scores) > 0
            if not embedding_ran:
                embedding_error = "Embedding API returned no scores."
        except Exception as exc:
            embedding_error = str(exc)
            semantic_scores = {}
    else:
        embedding_error = "No OPENAI_API_KEY configured in Streamlit secrets."

    hits: list[SimilarityHit] = []
    for key, meta in candidates:
        inst_full = meta.get("instruction", "")
        lex = lexical_scores.get(key, 0.0)
        sem = semantic_scores.get(key)
        flagged, reason = evaluate_similarity_block(lex, sem, threshold)
        if reason == "dual":
            method = "dual-60"
        elif reason == "meaning":
            method = "meaning-70"
        elif sem is not None and sem >= threshold:
            method = "semantic-high"
        elif lex >= threshold:
            method = "lexical-only"
        else:
            method = "below-threshold"
        hits.append(
            SimilarityHit(
                task_key=key,
                trainer=meta.get("trainer", ""),
                lexical_score=lex,
                semantic_score=sem,
                method=method,
                instruction_preview=inst_full[:200],
                matched_instruction=inst_full,
                dual_block=flagged,
                block_reason=reason,
            )
        )

    hits.sort(
        key=lambda h: (
            h.dual_block,
            h.semantic_score if h.semantic_score is not None else -1.0,
            h.lexical_score,
        ),
        reverse=True,
    )
    meta = SimilarityRunMeta(
        api_key_present=bool(api_key),
        embedding_ran=embedding_ran,
        embed_model=embed_model,
        embedding_error=embedding_error,
        corpus_size=len(candidates),
    )
    return SimilarityCompareResult(hits=hits[:top_n], meta=meta)


def tfidf_similarity(
    query_text: str,
    corpus: dict[str, str],
    exclude_name: str = "",
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Legacy TF-IDF for SPEC.md comparison."""
    filtered = {
        k: v for k, v in corpus.items()
        if v.strip() and exclude_name.lower() not in k.lower()
    }
    if not query_text or not filtered:
        return []
    ids = list(filtered.keys())
    docs = [filtered[i] for i in ids]
    vec = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), max_features=5000,
        lowercase=True, strip_accents="unicode",
    )
    tb = vec.fit_transform(docs)
    qv = vec.transform([query_text])
    sims = cosine_similarity(qv, tb).flatten()
    ranked = sorted(zip(ids, sims), key=lambda x: -x[1])
    return ranked[:top_n]
