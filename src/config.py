"""Pipeline configuration presets.

Each config dict controls all pipeline stages:
    use_query_expansion : bool  -- Enable LLM-based Query2Doc expansion (Stage 0)
    expansion_batch     : int   -- Batch size for query expansion
    k_docs              : int   -- BM25 documents to retrieve per query (Stage 1)
    chunk_size          : int   -- Target chunk size in words (~6 chars/word)
    chunk_overlap       : int   -- Overlap between chunks in words
    k_passages          : int   -- Passages to retain after BM25 filtering (Stage 1)
    k_biencoder         : int   -- Passages to retain after bi-encoder (Stage 2)
    biencoder_batch     : int   -- Bi-encoder encoding batch size
    k_crossencoder      : int   -- Passages to retain after cross-encoder (Stage 3)
    crossencoder_batch  : int   -- Cross-encoder scoring batch size
    llm_batch           : int   -- LLM generation batch size (Stage 4)
    temperature         : float -- LLM sampling temperature
    max_new_tokens      : int   -- Max tokens per LLM response
    use_self_consistency: bool  -- Enable multi-sample voting (Stage 5)
    num_samples         : int   -- Number of LLM samples for self-consistency
"""

from __future__ import annotations

# Fast mode -- no extras, fastest execution
FAST_CONFIG = {
    "use_query_expansion": False,
    "expansion_batch": 32,
    "k_docs": 20,
    "chunk_size": 100,
    "chunk_overlap": 20,
    "k_passages": 50,
    "k_biencoder": 15,
    "biencoder_batch": 256,
    "k_crossencoder": 3,
    "crossencoder_batch": 64,
    "llm_batch": 32,
    "temperature": 0.1,
    "max_new_tokens": 32,
    "use_self_consistency": False,
    "num_samples": 1,
}

# Balanced mode -- good quality/speed tradeoff
BALANCED_CONFIG = {
    "use_query_expansion": False,
    "expansion_batch": 32,
    "k_docs": 50,
    "chunk_size": 100,
    "chunk_overlap": 20,
    "k_passages": 150,
    "k_biencoder": 20,
    "biencoder_batch": 256,
    "k_crossencoder": 5,
    "crossencoder_batch": 64,
    "llm_batch": 32,
    "temperature": 0.3,
    "max_new_tokens": 32,
    "use_self_consistency": False,
    "num_samples": 3,
}

# Competition mode -- optimized for maximum F1 score (F1 = 37.5)
COMPETITION_CONFIG = {
    "use_query_expansion": True,
    "expansion_batch": 128,
    "k_docs": 100,
    "chunk_size": 100,
    "chunk_overlap": 20,
    "k_passages": 200,
    "k_biencoder": 25,
    "biencoder_batch": 256,
    "k_crossencoder": 5,
    "crossencoder_batch": 64,
    "llm_batch": 128,
    "temperature": 0.1,
    "max_new_tokens": 32,
    "use_self_consistency": True,
    "num_samples": 3,
}

# Max quality mode -- all features, highest accuracy (slowest)
MAX_QUALITY_CONFIG = {
    "use_query_expansion": True,
    "expansion_batch": 32,
    "k_docs": 150,
    "chunk_size": 100,
    "chunk_overlap": 25,
    "k_passages": 300,
    "k_biencoder": 30,
    "biencoder_batch": 256,
    "k_crossencoder": 5,
    "crossencoder_batch": 64,
    "llm_batch": 32,
    "temperature": 0.3,
    "max_new_tokens": 32,
    "use_self_consistency": True,
    "num_samples": 5,
}
