"""BM25 document retrieval and passage filtering."""

from __future__ import annotations

import json
import logging
import re

from pyserini.search.lucene import LuceneSearcher
from rank_bm25 import BM25Plus
from tqdm import tqdm

from .chunking import chunk_document

logger = logging.getLogger(__name__)

# Stopwords for BM25 passage filtering
BM25_STOPWORDS = frozenset([
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
    "be", "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare", "ought",
    "used", "it", "its", "this", "that", "these", "those", "i", "you", "he",
    "she", "we", "they", "what", "which", "who", "whom", "when", "where", "why",
    "how", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "also", "now", "here", "there", "then", "once",
])


def get_document_text(searcher: LuceneSearcher, docid: str) -> str:
    """Extract document text from BM25 index."""
    doc = searcher.doc(docid)
    if doc is None:
        return ""
    raw = doc.raw()
    if raw:
        try:
            data = json.loads(raw)
            return data.get("contents", raw)
        except (json.JSONDecodeError, TypeError):
            return raw
    return ""


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text for BM25: lowercase, remove punctuation, filter stopwords."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if t not in BM25_STOPWORDS and len(t) > 1]


def bm25_filter_passages(
    query: str,
    passages: list[str],
    top_k: int = 200,
) -> list[str]:
    """Filter passages using BM25Plus scoring (best MRR performance)."""
    if not passages or len(passages) <= top_k:
        return passages

    tokenized = [tokenize_for_bm25(p) for p in passages]

    # Keep only passages with non-empty tokens
    valid_data = [
        (passages[i], tokens) for i, tokens in enumerate(tokenized) if tokens
    ]
    if not valid_data:
        return passages[:top_k]

    valid_passages, valid_tokens = zip(*valid_data)
    bm25 = BM25Plus(list(valid_tokens))

    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return passages[:top_k]

    scores = bm25.get_scores(query_tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    return [valid_passages[i] for i in top_indices]


def stage1_bm25_retrieve(
    searcher: LuceneSearcher,
    search_queries: list[str],
    k_docs: int = 100,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
    k_passages: int = 200,
) -> list[list[str]]:
    """Stage 1: BM25 document retrieval -> chunking -> BM25 passage filtering.

    Args:
        searcher: Pyserini BM25 searcher.
        search_queries: Queries (optionally expanded).
        k_docs: Number of documents to retrieve per query.
        chunk_size: Chunk size in words.
        chunk_overlap: Chunk overlap in words.
        k_passages: Number of passages to retain after filtering.

    Returns:
        List of passage lists, one per query.
    """
    all_passages = []
    total_docs, total_chunks, total_filtered = 0, 0, 0

    for query in tqdm(search_queries, desc="Stage 1: BM25 → Chunk → Filter"):
        hits = searcher.search(query, k=k_docs)

        # Chunk all retrieved documents
        passages = []
        for hit in hits:
            doc_text = get_document_text(searcher, hit.docid)
            if doc_text:
                total_docs += 1
                passages.extend(chunk_document(doc_text, chunk_size, chunk_overlap))

        total_chunks += len(passages)

        # BM25 passage filtering
        filtered = bm25_filter_passages(query, passages, top_k=k_passages)
        total_filtered += len(filtered)
        all_passages.append(filtered)

    n = len(search_queries) or 1
    logger.info(
        "%d docs → %d chunks → %d passages (avg %d → %d per query)",
        total_docs, total_chunks, total_filtered, total_chunks // n, total_filtered // n,
    )

    return all_passages
