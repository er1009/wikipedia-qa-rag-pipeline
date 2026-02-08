"""Contextual document chunking using Anthropic's Contextual Retrieval approach.

Prepends document context (title/first sentence) to each chunk for better
BM25 matching -- shown to improve retrieval by 35-67%.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Constants
_MIN_LINE_LENGTH = 10          # Minimum chars for a meaningful context line
_MAX_CONTEXT_LENGTH = 150      # Maximum chars for extracted document context
_CONTEXT_CHECK_PREFIX = 30     # Chars to check for duplicate context prefix
_CHARS_PER_WORD = 6            # Approximate chars per word for size conversion

# Semantic separators: paragraphs -> sentences -> clauses -> words
CHUNK_SEPARATORS = [
    "\n\n",  # Paragraphs
    "\n",    # Line breaks
    ". ",    # Sentences
    "? ",
    "! ",
    "; ",    # Clauses
    ", ",
    " ",     # Words
    "",      # Characters (fallback)
]


@lru_cache(maxsize=4)
def _get_splitter(chunk_size_chars: int, overlap_chars: int) -> RecursiveCharacterTextSplitter:
    """Get a cached text splitter instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size_chars,
        chunk_overlap=overlap_chars,
        length_function=len,
        separators=CHUNK_SEPARATORS,
        is_separator_regex=False,
    )


def extract_document_context(text: str) -> str:
    """Extract document title/context (first meaningful line or sentence).

    Based on Anthropic's Contextual Retrieval: prepending context to chunks
    improves BM25 retrieval by 35%+.
    """
    if not text:
        return ""

    for line in text.strip().split("\n"):
        line = line.strip()
        if line and len(line) > _MIN_LINE_LENGTH:
            if len(line) > _MAX_CONTEXT_LENGTH:
                for end_char in [". ", "? ", "! "]:
                    idx = line.find(end_char)
                    if 20 < idx < _MAX_CONTEXT_LENGTH:
                        return line[: idx + 1]
            return line[:_MAX_CONTEXT_LENGTH]
    return ""


def chunk_document(
    text: str,
    chunk_size: int = 100,
    chunk_overlap: int = 20,
) -> list[str]:
    """Split document into contextual chunks.

    1. Extract document context (title/first sentence)
    2. Split into chunks using RecursiveCharacterTextSplitter
    3. Prepend context to each chunk for better BM25 matching

    Args:
        text: Document text to chunk.
        chunk_size: Target chunk size in words (~6 chars/word).
        chunk_overlap: Overlap between chunks in words.

    Returns:
        List of contextual chunks with document context prepended.
    """
    if not text or not text.strip():
        return []

    doc_context = extract_document_context(text)

    # Convert word count to approximate character count
    chunk_chars = chunk_size * _CHARS_PER_WORD
    overlap_chars = chunk_overlap * _CHARS_PER_WORD

    splitter = _get_splitter(chunk_chars, overlap_chars)
    raw_chunks = splitter.split_text(text)

    # Prepend context to each chunk (Contextual Chunking)
    contextual_chunks = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        if doc_context and not chunk.startswith(doc_context[:_CONTEXT_CHECK_PREFIX]):
            chunk = f"[{doc_context}] {chunk}"
        contextual_chunks.append(chunk)

    return contextual_chunks
