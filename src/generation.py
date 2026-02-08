"""LLM answer generation, prompt engineering, and post-processing."""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from transformers import Pipeline as HFPipeline


# =============================================================================
# Prompt Templates
# =============================================================================

def create_competition_prompt(query: str, passages: list[str]) -> list[dict]:
    """Competition-optimized prompt with few-shot examples for max F1."""
    context = "\n---\n".join(passages[:3])
    return [
        {
            "role": "system",
            "content": (
                "You are a factual QA system. Extract the exact answer from the context.\n\n"
                "RULES:\n"
                "- Output ONLY the answer (1-5 words max)\n"
                "- No explanations, no sentences, no lists\n"
                "- No prefixes like 'The answer is' or 'It is'\n"
                "- Proper nouns: keep original capitalization\n"
                "- If uncertain, give your best guess from context\n\n"
                "EXAMPLES:\n"
                "Q: Who wrote Romeo and Juliet? → William Shakespeare\n"
                "Q: What is the capital of France? → Paris\n"
                "Q: When did World War 2 end? → 1945\n"
                "Q: What language do Brazilians speak? → Portuguese\n"
                "Q: Who painted the Mona Lisa? → Leonardo da Vinci\n"
                "Q: What is the largest planet? → Jupiter\n"
                "Q: Who invented the telephone? → Alexander Graham Bell\n"
                "Q: What year did the Titanic sink? → 1912"
            ),
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQ: {query}\nA:"},
    ]


def create_terse_prompt(query: str, passages: list[str]) -> list[dict]:
    """Minimal prompt for shortest possible answers."""
    context = "\n---\n".join(passages)
    return [
        {
            "role": "system",
            "content": "Output ONLY the answer in 1-5 words. No explanations. If not found, say 'unknown'.",
        },
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"},
    ]


def query2doc_prompt(query: str) -> list[dict]:
    """Query2Doc expansion: generate a pseudo-document for better BM25 retrieval."""
    return [
        {
            "role": "system",
            "content": (
                'Write ONE factual sentence answering the question. State the answer directly. '
                'No hedging, no "there is no", no questions back.\n\n'
                "Q: Who was the first US president?\n"
                "George Washington served as the first President of the United States.\n\n"
                "Q: What character did Natalie Portman play in Star Wars?\n"
                "Natalie Portman played Padmé Amidala in the Star Wars prequel trilogy.\n\n"
                "Q: What year did LeBron James join the NBA?\n"
                "LeBron James was drafted to the NBA in 2003 by the Cleveland Cavaliers.\n\n"
                "Q: Who did Draco Malfoy marry?\n"
                "Draco Malfoy married Astoria Greengrass in the Harry Potter series.\n\n"
                "Q: Where does the Lena River end?\n"
                "The Lena River flows into the Laptev Sea in the Arctic Ocean."
            ),
        },
        {"role": "user", "content": f"Q: {query}"},
    ]


# =============================================================================
# Post-Processing
# =============================================================================

def post_process_for_f1(answer: str) -> str:
    """Clean LLM answer to maximize F1 score."""
    if not answer:
        return ""
    answer = answer.strip()

    # Remove numbered lists
    answer = re.sub(r"^\d+[\.)\s]+", "", answer)

    # Take first line/sentence only
    answer = answer.split("\n")[0]
    first_period = answer.find(". ")
    if 0 < first_period < 50:
        answer = answer[:first_period]

    # Remove common prefixes that hurt F1
    prefixes = [
        "the answer is ", "answer: ", "it is ", "they are ",
        "he is ", "she is ", "it was ", "a: ",
    ]
    lower = answer.lower()
    for p in prefixes:
        if lower.startswith(p):
            answer = answer[len(p) :]
            lower = answer.lower()

    # Remove trailing punctuation and quotes
    answer = answer.rstrip(".,;:!?")
    answer = answer.strip("\"'")

    # Handle failure indicators
    fail_indicators = [
        "dont know", "don't know", "unknown", "no answer",
        "not found", "cannot", "no information", "not mentioned",
    ]
    if any(x in answer.lower() for x in fail_indicators):
        return ""

    return answer.strip()


def clean_query2doc_output(text: str) -> str:
    """Post-process Query2Doc expansion output to remove unhelpful content."""
    if not text:
        return ""

    # Remove Q: A: format artifacts
    text = re.sub(r"^[QA]:\s*", "", text, flags=re.MULTILINE)

    # Remove conversational endings
    text = re.sub(r"Is there anything else.*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Let me know if.*$", "", text, flags=re.IGNORECASE)

    # Handle "There is no X. However, Y" pattern - keep only Y
    match = re.search(r"There is no[^.]*\.\s*However,\s*(.+)", text, flags=re.IGNORECASE)
    if match:
        text = match.group(1)

    # Remove unhelpful openers
    for pattern in [
        r"^There is no (record|information|credible|direct|specific)[^.]*\.",
        r"^There are no[^.]*\.",
        r"^I couldn't find[^.]*\.",
        r"^Unfortunately[^.]*\.",
    ]:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)

    # Take only first 2 sentences
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s.strip()]
    text = " ".join(sentences[:2])

    text = " ".join(text.split())

    if text.lower().startswith(("there is no", "i cannot", "unfortunately")):
        return ""

    return text.strip()


def competition_majority_vote(answers: list[str]) -> str:
    """Self-consistency voting with F1-aware normalization."""
    cleaned = [post_process_for_f1(a) for a in answers]
    cleaned = [a for a in cleaned if a]
    if not cleaned:
        return ""

    def normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"[^\w\s]", "", s)
        return " ".join(s.split())

    normalized = [normalize(a) for a in cleaned]
    counts = Counter(normalized)
    winner_norm = counts.most_common(1)[0][0]

    for orig, norm in zip(cleaned, normalized):
        if norm == winner_norm:
            return orig
    return cleaned[0]


# =============================================================================
# Query Expansion
# =============================================================================

def stage0_expand_queries(pipe: HFPipeline, queries: list[str], batch_size: int = 32) -> list[str]:
    """Stage 0: Query2Doc expansion using LLM-generated pseudo-documents."""
    messages = [query2doc_prompt(q) for q in queries]
    expanded = []

    for i in tqdm(range(0, len(messages), batch_size), desc="Stage 0: Query2Doc"):
        batch = messages[i : i + batch_size]
        outputs = pipe(
            batch,
            max_new_tokens=60,
            eos_token_id=pipe.tokenizer.eos_token_id,
            do_sample=False,
            batch_size=len(batch),
        )
        for j, out in enumerate(outputs):
            raw = out[0]["generated_text"][-1].get("content", "").strip()
            pseudo_doc = clean_query2doc_output(raw)
            original = queries[i + j]
            expanded.append(f"{original} {pseudo_doc}" if pseudo_doc else original)

    return expanded


# =============================================================================
# Answer Generation
# =============================================================================

def stage4_generate_answers(
    pipe: HFPipeline,
    queries: list[str],
    all_passages: list[list[str]],
    config: dict,
) -> list[str]:
    """Stage 4+5: LLM answer generation with self-consistency and post-processing."""
    llm_batch = config["llm_batch"]
    temperature = config["temperature"]
    max_new_tokens = config["max_new_tokens"]
    use_self_consistency = config.get("use_self_consistency", False)
    num_samples = config.get("num_samples", 3)
    # Use competition prompt when query expansion is active (full pipeline mode)
    competition_mode = config.get("use_query_expansion", False)

    # Build prompts
    prompt_fn = create_competition_prompt if competition_mode else create_terse_prompt
    all_messages = [
        prompt_fn(q, p if p else ["No relevant information found."])
        for q, p in zip(queries, all_passages)
    ]

    if use_self_consistency and num_samples > 1:
        # Generate multiple samples for voting
        replicated = all_messages * num_samples
        all_answers = []

        for i in tqdm(range(0, len(replicated), llm_batch), desc=f"Stage 4+5: LLM ({num_samples} samples)"):
            batch = replicated[i : i + llm_batch]
            outputs = pipe(
                batch,
                max_new_tokens=max_new_tokens,
                eos_token_id=pipe.tokenizer.eos_token_id,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                batch_size=len(batch),
            )
            for out in outputs:
                all_answers.append(out[0]["generated_text"][-1].get("content", "").strip())

        # Majority vote
        n = len(queries)
        answers = [
            competition_majority_vote([all_answers[q + i * n] for i in range(num_samples)])
            for q in range(n)
        ]
    else:
        # Single answer per query
        answers = []
        for i in tqdm(range(0, len(all_messages), llm_batch), desc="Stage 4: LLM"):
            batch = all_messages[i : i + llm_batch]
            outputs = pipe(
                batch,
                max_new_tokens=max_new_tokens,
                eos_token_id=pipe.tokenizer.eos_token_id,
                do_sample=False,
                batch_size=len(batch),
            )
            for out in outputs:
                answers.append(out[0]["generated_text"][-1].get("content", "").strip())

    # Post-processing
    if competition_mode:
        answers = [post_process_for_f1(a) or "unknown" for a in answers]

    return answers
