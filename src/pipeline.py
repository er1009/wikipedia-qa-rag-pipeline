"""Main multi-stage RAG pipeline orchestration."""

from __future__ import annotations

import logging

import numpy as np
import torch
import transformers
from pyserini.search.lucene import LuceneSearcher
from sentence_transformers import CrossEncoder, SentenceTransformer

from .config import COMPETITION_CONFIG
from .generation import stage0_expand_queries, stage4_generate_answers
from .metrics import f1_score, max_over_ground_truths
from .reranking import stage2_biencoder_rerank, stage3_crossencoder_rerank
from .retrieval import stage1_bm25_retrieve

logger = logging.getLogger(__name__)

# Default model names (override via constructor)
DEFAULT_BI_ENCODER = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
DEFAULT_CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-12-v2"
DEFAULT_LLM = "meta-llama/Llama-3.2-1B-Instruct"


class RAGPipeline:
    """Multi-stage RAG pipeline for Wikipedia question answering.

    Pipeline stages:
        0. Query2Doc expansion (LLM generates pseudo-document)
        1. BM25 document retrieval -> contextual chunking -> BM25 passage filtering
        2. Bi-encoder semantic reranking
        3. Cross-encoder precision reranking
        4. LLM answer generation with competition-optimized prompts
        5. Self-consistency voting + F1-aware post-processing
    """

    def __init__(
        self,
        bm25_index_path: str,
        bi_encoder_model: str = DEFAULT_BI_ENCODER,
        cross_encoder_model: str = DEFAULT_CROSS_ENCODER,
        llm_model: str = DEFAULT_LLM,
        device: str = "cuda",
        use_flash_attn: bool = True,
    ):
        # Stage 1: BM25 searcher
        logger.info("Loading BM25 index: %s", bm25_index_path)
        self.searcher = LuceneSearcher(bm25_index_path)
        logger.info("  %s documents indexed", f"{self.searcher.num_docs:,}")

        # Stage 2: Bi-encoder
        logger.info("Loading bi-encoder: %s", bi_encoder_model)
        self.bi_encoder = SentenceTransformer(bi_encoder_model)
        self.bi_encoder.to(device)

        # Stage 3: Cross-encoder
        logger.info("Loading cross-encoder: %s", cross_encoder_model)
        self.cross_encoder = CrossEncoder(cross_encoder_model, max_length=512)

        # Stage 0 + 4: LLM pipeline
        logger.info("Loading LLM: %s", llm_model)
        model_kwargs: dict = {"torch_dtype": torch.bfloat16}
        if use_flash_attn:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.pipe = transformers.pipeline(
            "text-generation",
            model=llm_model,
            model_kwargs=model_kwargs,
            device_map="auto",
        )
        self.pipe.tokenizer.pad_token_id = self.pipe.tokenizer.eos_token_id
        self.pipe.tokenizer.padding_side = "left"
        logger.info("All models loaded successfully")

    def answer(
        self,
        queries: list[str],
        config: dict | None = None,
    ) -> list[str]:
        """Run the full multi-stage pipeline on a batch of queries.

        Args:
            queries: List of natural language questions.
            config: Pipeline configuration (defaults to COMPETITION_CONFIG).

        Returns:
            List of extracted answers.
        """
        config = config or COMPETITION_CONFIG

        logger.info(
            "Pipeline: %d queries | Query2Doc=%s | Self-Consistency=%s",
            len(queries),
            "ON" if config["use_query_expansion"] else "OFF",
            "ON" if config["use_self_consistency"] else "OFF",
        )

        # Stage 0: Query expansion
        if config["use_query_expansion"]:
            search_queries = stage0_expand_queries(
                self.pipe, queries, config["expansion_batch"]
            )
        else:
            search_queries = queries

        # Stage 1: BM25 retrieval + chunking + filtering
        all_passages = stage1_bm25_retrieve(
            self.searcher,
            search_queries,
            k_docs=config["k_docs"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            k_passages=config["k_passages"],
        )

        # Stage 2: Bi-encoder reranking
        all_passages = stage2_biencoder_rerank(
            self.bi_encoder,
            search_queries,
            all_passages,
            k=config["k_biencoder"],
            batch_size=config["biencoder_batch"],
        )

        # Stage 3: Cross-encoder reranking (use original queries for precision)
        all_passages = stage3_crossencoder_rerank(
            self.cross_encoder,
            queries,
            all_passages,
            k=config["k_crossencoder"],
            batch_size=config["crossencoder_batch"],
        )

        # Stage 4+5: LLM generation + post-processing
        answers = stage4_generate_answers(self.pipe, queries, all_passages, config)

        logger.info("Pipeline complete -- generated %d answers", len(answers))
        return answers

    def evaluate(
        self,
        questions: list[str],
        ground_truths: list[list[str]],
        config: dict | None = None,
    ) -> dict:
        """Run pipeline and compute F1 scores.

        Args:
            questions: List of questions.
            ground_truths: List of answer lists (multiple valid answers per question).
            config: Pipeline configuration.

        Returns:
            Dict with avg_f1, median_f1, predictions, and per-query f1_scores.
        """
        answers = self.answer(questions, config)

        f1_scores_list = [
            max_over_ground_truths(f1_score, pred, gts)
            for pred, gts in zip(answers, ground_truths)
        ]

        avg = np.mean(f1_scores_list) * 100
        median = np.median(f1_scores_list) * 100

        logger.info("Average F1: %.2f%%  |  Median F1: %.2f%%", avg, median)

        return {
            "avg_f1": avg,
            "median_f1": median,
            "predictions": answers,
            "f1_scores": f1_scores_list,
        }
