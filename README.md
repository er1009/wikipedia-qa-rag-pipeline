# Multi-Stage RAG Pipeline for Wikipedia QA

A high-performance Retrieval-Augmented Generation pipeline for open-domain question answering over Wikipedia. Combines **BM25 retrieval**, **contextual chunking**, **bi-encoder/cross-encoder reranking**, and **LLM answer extraction** with self-consistency voting, achieving **F1 = 37.5** on a Wikipedia QA benchmark.

## Highlights

- **6-stage pipeline**: Query expansion → BM25 retrieval → Contextual chunking → Bi-encoder → Cross-encoder → LLM generation
- **Query2Doc expansion**: LLM generates pseudo-documents to bridge the query-document vocabulary gap
- **Contextual chunking**: Prepends document context (title) to every chunk -- inspired by [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (improves BM25 by 35%+)
- **Two-stage neural reranking**: Fast bi-encoder filtering followed by precision cross-encoder scoring
- **Self-consistency voting**: Multiple LLM samples with F1-aware majority vote
- **Optimized batching**: Deduplicates passages before encoding for 3-5x speedup

## Results

| Configuration | Query2Doc | Self-Consistency | F1 Score |
|---------------|-----------|-----------------|----------|
| Baseline (BM25 + LLM only) | OFF | OFF | ~22.0 |
| Balanced (+ reranking) | OFF | OFF | ~30.0 |
| **Competition (full pipeline)** | **ON** | **ON** | **37.5** |

The full pipeline with all stages enabled achieves **F1 = 37.5**, a **70%+ improvement** over the BM25-only baseline.

## Architecture

```
Query ─────────────────────────────────────────────────────────────────► Answer

Stage 0         Stage 1              Stage 2          Stage 3       Stage 4+5
Query2Doc  →  BM25 Retrieval  →   Bi-Encoder   →  Cross-Encoder  →   LLM
Expansion     + Chunking          Reranking        Reranking       Generation
              + BM25 Filter                                        + Voting
              
(LLM)      100 docs → 5000     200 passages →   25 passages →    5 passages →
            chunks → 200          25                5              Answer
            passages
```

### Pipeline Detail

```
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 0: Query2Doc Expansion (Llama-3.2-1B)                        │
│  "Who painted the Mona Lisa?"                                       │
│  → "Who painted the Mona Lisa? Leonardo da Vinci painted the        │
│     Mona Lisa, one of the most famous paintings in the world."      │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  Stage 1a: BM25 Document Retrieval (Pyserini, 100 docs)             │
│  Stage 1b: Contextual Chunking ([Title] prepended to each chunk)    │
│  Stage 1c: BM25Plus Passage Filtering (→ 200 passages)              │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  Stage 2: Bi-Encoder Reranking (multi-qa-mpnet, 200 → 25)           │
│  Fast semantic filtering with deduplicated batch encoding            │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  Stage 3: Cross-Encoder Reranking (ms-marco-MiniLM-L-12, 25 → 5)   │
│  Joint query-passage attention for precision scoring                 │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │
┌────────────────────────────────▼─────────────────────────────────────┐
│  Stage 4+5: LLM Generation + Self-Consistency (Llama-3.2-1B)        │
│  Few-shot extraction prompt → 3 samples → F1-aware majority vote    │
│  → Post-processing → Answer                                         │
└──────────────────────────────────────────────────────────────────────┘
```

## Models

| Stage | Model | Params | Role |
|-------|-------|--------|------|
| Query Expansion | meta-llama/Llama-3.2-1B-Instruct | 1.2B | Generate pseudo-documents for BM25 |
| BM25 Retrieval | Pyserini (wikipedia-kilt-doc) | - | Lexical document retrieval |
| Bi-Encoder | multi-qa-mpnet-base-dot-v1 | 110M | Fast semantic passage filtering |
| Cross-Encoder | ms-marco-MiniLM-L-12-v2 | 33M | Precision passage reranking |
| Answer Generation | meta-llama/Llama-3.2-1B-Instruct | 1.2B | Extractive QA with few-shot prompt |

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (16GB+ VRAM)
- Java 21 (required by Pyserini)
- [Hugging Face account](https://huggingface.co/join) with [Llama access](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)

### Installation

```bash
git clone https://github.com/er1009/wikipedia-qa-rag-pipeline.git
cd wikipedia-qa-rag-pipeline

pip install -r requirements.txt

# Java (if not installed)
apt-get install openjdk-21-jdk-headless

# Hugging Face authentication
cp .env.example .env
# Edit .env with your HF token, then:
huggingface-cli login --token $(grep HF_TOKEN .env | cut -d= -f2)
```

### Usage

```python
from src.pipeline import RAGPipeline
from src.config import COMPETITION_CONFIG

# Initialize pipeline (downloads models on first run)
pipeline = RAGPipeline(
    bm25_index_path="path/to/wikipedia-kilt-doc",
    device="cuda",
)

# Answer questions
answers = pipeline.answer(
    queries=["Who painted the Mona Lisa?", "What is the capital of France?"],
    config=COMPETITION_CONFIG,
)
print(answers)  # ["Leonardo da Vinci", "Paris"]

# Evaluate on labeled data
results = pipeline.evaluate(
    questions=["Who painted the Mona Lisa?"],
    ground_truths=[["Leonardo da Vinci", "Da Vinci"]],
)
print(f"F1: {results['avg_f1']:.1f}%")
```

### Configuration Presets

| Preset | Query2Doc | Self-Consistency | Speed | Quality |
|--------|-----------|-----------------|-------|---------|
| `FAST_CONFIG` | OFF | OFF | Fastest | Good |
| `BALANCED_CONFIG` | OFF | OFF | Medium | Better |
| `COMPETITION_CONFIG` | ON | ON (3 samples) | Slower | Best |
| `MAX_QUALITY_CONFIG` | ON | ON (5 samples) | Slowest | Maximum |

## Project Structure

```
wikipedia-qa-rag-pipeline/
├── src/
│   ├── pipeline.py        # Main RAGPipeline class (orchestration)
│   ├── config.py          # Configuration presets (Fast/Balanced/Competition/Max)
│   ├── retrieval.py       # BM25 document retrieval + passage filtering
│   ├── chunking.py        # Contextual chunking (Anthropic-inspired)
│   ├── reranking.py       # Bi-encoder + cross-encoder reranking
│   ├── generation.py      # LLM prompts, Query2Doc, post-processing, voting
│   └── metrics.py         # F1 score, exact match evaluation
├── data/
├── run_pipeline.ipynb     # Google Colab notebook
├── requirements.txt
├── .env.example
└── README.md
```

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **Contextual chunking** | Prepending document title to chunks improves BM25 matching by 35%+ (Anthropic, 2024) |
| **BM25Plus passage filter** | Outperforms BM25Okapi for passage-level ranking; reduces bi-encoder input by 25x |
| **Deduplicated encoding** | Encoding unique passages once (not per-query) gives 3-5x speedup |
| **Cross-encoder on original queries** | Expanded queries help recall (Stage 1-2); original queries help precision (Stage 3) |
| **Few-shot extraction prompt** | 8 diverse QA examples prime the LLM for concise, factual extraction |
| **F1-aware majority vote** | Normalizes answers like the F1 metric before voting, avoiding surface-form mismatches |
| **Post-processing pipeline** | Strips prefixes, articles, punctuation that hurt token-level F1 |

## References

- Anthropic (2024) -- [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- Wang et al. (2023) -- [Query2Doc: Query Expansion with LLMs](https://arxiv.org/abs/2303.07678)
- Nogueira & Cho (2019) -- [Passage Re-ranking with BERT](https://arxiv.org/abs/1901.04085)
- Wang et al. (2022) -- [Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171)
- Robertson & Zaragoza (2009) -- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.nowpublishers.com/article/Details/INR-019)

## License

MIT
