"""Multi-stage RAG pipeline for Wikipedia question answering.

Implements a 6-stage pipeline: Query2Doc expansion → BM25 retrieval →
Contextual chunking → Bi-encoder reranking → Cross-encoder reranking →
LLM answer generation with self-consistency voting.
"""

from .pipeline import RAGPipeline
from .config import FAST_CONFIG, BALANCED_CONFIG, COMPETITION_CONFIG, MAX_QUALITY_CONFIG

__all__ = [
    "RAGPipeline",
    "FAST_CONFIG",
    "BALANCED_CONFIG",
    "COMPETITION_CONFIG",
    "MAX_QUALITY_CONFIG",
]
