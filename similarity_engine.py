"""Instruction similarity — lexical + embedding run in parallel.

Block rule: BOTH lexical AND embedding scores must be >= 60% on the same
corpus row before the portal says to change the task.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tracker_defaults import (
    DEFAULT_EMBED_MODEL,
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
    dual_block: bool = False


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


def is_dual_duplicate(
    lexical_score: float,
    semantic_score: float | None,
    threshold: float = INSTRUCTION_SIM_THRESHOLD,
) -> bool:
    """Both checks must clear the threshold to block."""
    if semantic_score is None:
        return False
    return lexical_score >= threshold and semantic_score >= threshold


def _get_embeddings(texts: list[str], api_key: str, model: str = DEFAULT_EMBED_MODEL) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    out: list[list[float]] = []
    batch_size = 60
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=model, input=batch)
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
) -> dict[str, float]:
    if not api_key or not candidates:
        return {}
    texts = [query] + [meta.get("instruction", "") for _, meta in candidates]
    embeddings = _get_embeddings(texts, api_key)
    source_emb = embeddings[0]
    scores: dict[str, float] = {}
    for idx, (key, _) in enumerate(candidates):
        scores[key] = _cosine(source_emb, embeddings[idx + 1])
    return scores


def compare_instruction_to_corpus(
    query_instruction: str,
    corpus: dict[str, dict[str, str]],
    exclude_keys: set[str] | None = None,
    api_key: str = "",
    top_n: int = 10,
    threshold: float = INSTRUCTION_SIM_THRESHOLD,
) -> list[SimilarityHit]:
    """
    Run lexical and embedding similarity in parallel against the full corpus.
    Returns hits sorted by combined strength; dual_block=True only when BOTH >= threshold.
    """
    exclude = exclude_keys or set()
    query = (query_instruction or "").strip()
    if not query or not corpus:
        return []

    candidates: list[tuple[str, dict[str, str]]] = []
    for key, meta in corpus.items():
        if key in exclude:
            continue
        if not (meta.get("instruction", "") or "").strip():
            continue
        candidates.append((key, meta))

    if not candidates:
        return []

    lexical_scores: dict[str, float] = {}
    semantic_scores: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        lex_future = executor.submit(_compute_lexical_scores, query, candidates)
        sem_future = executor.submit(_compute_semantic_scores, query, candidates, api_key)
        lexical_scores = lex_future.result()
        try:
            semantic_scores = sem_future.result()
        except Exception:
            semantic_scores = {}

    hits: list[SimilarityHit] = []
    for key, meta in candidates:
        lex = lexical_scores.get(key, 0.0)
        sem = semantic_scores.get(key)
        if lex < threshold and (sem is None or sem < threshold):
            continue
        dual = is_dual_duplicate(lex, sem, threshold)
        method = "lexical+semantic-dual" if dual else (
            "semantic-only" if sem is not None and sem >= threshold else "lexical-only"
        )
        hits.append(
            SimilarityHit(
                task_key=key,
                trainer=meta.get("trainer", ""),
                lexical_score=lex,
                semantic_score=sem,
                method=method,
                instruction_preview=meta.get("instruction", "")[:200],
                dual_block=dual,
            )
        )

    hits.sort(
        key=lambda h: (
            h.dual_block,
            h.semantic_score or 0,
            h.lexical_score,
        ),
        reverse=True,
    )
    return hits[:top_n]


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
