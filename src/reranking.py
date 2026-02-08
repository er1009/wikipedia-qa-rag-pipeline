"""Bi-encoder and cross-encoder reranking stages."""

from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


def stage2_biencoder_rerank(
    bi_encoder: SentenceTransformer,
    queries: list[str],
    all_passages: list[list[str]],
    k: int = 20,
    batch_size: int = 256,
) -> list[list[str]]:
    """Stage 2: Bi-encoder semantic reranking.

    Key optimization: encodes ALL unique passages at once (not per-query),
    reducing encoding from O(N*M) to O(unique_passages).

    Args:
        bi_encoder: SentenceTransformer bi-encoder model.
        queries: Original or expanded queries.
        all_passages: Passage lists from Stage 1.
        k: Number of passages to retain per query.
        batch_size: Encoding batch size.

    Returns:
        Reranked passage lists.
    """
    # Deduplicate passages for efficient encoding
    passage_to_idx: dict[str, int] = {}
    unique_passages: list[str] = []
    query_passage_indices: list[list[int]] = []

    for passages in all_passages:
        indices = []
        for p in passages:
            if p not in passage_to_idx:
                passage_to_idx[p] = len(unique_passages)
                unique_passages.append(p)
            indices.append(passage_to_idx[p])
        query_passage_indices.append(indices)

    total = sum(len(p) for p in all_passages)
    logger.info("Total passages: %s → Unique: %s", f"{total:,}", f"{len(unique_passages):,}")

    if not unique_passages:
        return [[] for _ in queries]

    # Encode queries and passages in batch
    query_embs = bi_encoder.encode(
        queries, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True
    )
    passage_embs = bi_encoder.encode(
        unique_passages, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True
    )

    # Score and rerank
    all_reranked = []
    for q_idx in tqdm(range(len(queries)), desc="Stage 2: Bi-Encoder"):
        indices = query_passage_indices[q_idx]
        passages = all_passages[q_idx]

        if not passages or len(passages) <= k:
            all_reranked.append(passages)
            continue

        scores = (query_embs[q_idx] @ passage_embs[indices].T).cpu().numpy()
        top_indices = scores.argsort()[::-1][:k]
        all_reranked.append([passages[i] for i in top_indices])

    return all_reranked


def stage3_crossencoder_rerank(
    cross_encoder: CrossEncoder,
    queries: list[str],
    all_passages: list[list[str]],
    k: int = 5,
    batch_size: int = 64,
) -> list[list[str]]:
    """Stage 3: Cross-encoder precision reranking.

    Uses joint query-passage encoding with cross-attention for
    the most accurate relevance scores.

    Args:
        cross_encoder: CrossEncoder model.
        queries: Original queries (not expanded -- precision matters here).
        all_passages: Passage lists from Stage 2.
        k: Number of passages to retain per query.
        batch_size: Scoring batch size.

    Returns:
        Reranked passage lists (top-k most relevant per query).
    """
    # Build all (query, passage) pairs
    all_pairs = []
    pair_map = []
    for q_idx, (query, passages) in enumerate(zip(queries, all_passages)):
        for p_idx, passage in enumerate(passages):
            all_pairs.append([query, passage])
            pair_map.append((q_idx, p_idx))

    if not all_pairs:
        return [[] for _ in queries]

    # Batch scoring
    all_scores = []
    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Stage 3: Cross-Encoder"):
        batch = all_pairs[i : i + batch_size]
        scores = cross_encoder.predict(batch)
        all_scores.extend(scores)

    # Group scores by query and select top-k
    query_scores: list[list[tuple[float, str]]] = [[] for _ in queries]
    for idx, score in enumerate(all_scores):
        q_idx, p_idx = pair_map[idx]
        query_scores[q_idx].append((score, all_passages[q_idx][p_idx]))

    return [
        [p for _, p in sorted(qs, reverse=True)[:k]]
        for qs in query_scores
    ]
