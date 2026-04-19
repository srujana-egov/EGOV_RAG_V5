import os
import time
import logging
from dotenv import load_dotenv
import openai

logger = logging.getLogger(__name__)
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

load_dotenv()

# Domain identity — set APP_DOMAIN env var to change for other deployments (e.g. "HCM")
_APP_DOMAIN = os.environ.get("APP_DOMAIN", "DIGIT Studio")

# ─────────────────────────────────────────────
# Single OpenAI client
# ─────────────────────────────────────────────
_client = None

def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        from utils import get_env_var
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing.")
        _client = openai.OpenAI(api_key=api_key)
    return _client


# ─────────────────────────────────────────────
# Retry decorator — covers RateLimitError and transient 5xx errors
# ─────────────────────────────────────────────
_openai_retry = retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIStatusError,
        openai.APIConnectionError,
    )),
    reraise=True,
)


# ─────────────────────────────────────────────
# Domain detection threshold
# ─────────────────────────────────────────────
OUT_OF_DOMAIN_THRESHOLD = float(os.environ.get("OUT_OF_DOMAIN_THRESHOLD", "0.35"))  # cosine similarity below this = out of DIGIT Studio domain

OUT_OF_DOMAIN_MSG = (
    f"I'm sorry, that question appears to be outside the scope of {_APP_DOMAIN} documentation.\n\n"
    f"I can only answer questions related to **{_APP_DOMAIN}**.\n\n"
    f"Please ask a question related to {_APP_DOMAIN}."
)


# ─────────────────────────────────────────────
# Query rewriting — skips GPT call for simple short queries
# ─────────────────────────────────────────────
def _is_simple_query(query: str) -> bool:
    """
    Return True if the query is already short and clean — no need to rewrite.
    Heuristic: ≤5 words AND ≤50 characters.
    """
    words = query.strip().split()
    return len(words) <= 5 and len(query.strip()) <= 50


@_openai_retry
def _call_rewrite(query: str) -> str:
    response = _get_client().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a search query optimizer for a {_APP_DOMAIN} documentation chatbot. "
                    "Rewrite the user's question to maximize retrieval of relevant documentation. "
                    "Expand abbreviations, add relevant synonyms, make it more specific. "
                    "Return ONLY the rewritten query, nothing else. 15 words or fewer."
                )
            },
            {"role": "user", "content": query}
        ],
        max_tokens=100,
        temperature=0.0,
        timeout=30
    )
    return response.choices[0].message.content.strip()


def rewrite_query(query: str) -> str:
    if _is_simple_query(query):
        logger.info("Query Rewrite: Skipped (short query): '%s'", query)
        return query
    try:
        rewritten = _call_rewrite(query)
        logger.info("Query Rewrite: '%s' → '%s'", query, rewritten)
        return rewritten
    except Exception as e:
        logger.warning("Query Rewrite: Failed after retries, using original: %s", e)
        return query


@_openai_retry
def _call_variants(query: str) -> list:
    """Call GPT-3.5 to generate 2 alternative phrasings of the query."""
    response = _get_client().chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimizer for a documentation chatbot. "
                    "Given a user question, generate exactly 2 alternative phrasings that "
                    "capture the same intent using different terminology. "
                    "Return ONLY the 2 alternatives, one per line, no numbering, no explanation."
                )
            },
            {"role": "user", "content": query}
        ],
        max_tokens=120,
        temperature=0.3,
        timeout=30
    )
    raw = response.choices[0].message.content.strip()
    variants = [line.strip() for line in raw.splitlines() if line.strip()]
    return variants[:2]  # cap at 2


def generate_query_variants(query: str) -> list[str]:
    """
    Return [original_query] + up to 2 paraphrases.
    Falls back to [original_query] on any error.
    Only generates variants for non-trivial queries (>5 words).
    """
    if _is_simple_query(query):
        return [query]
    try:
        variants = _call_variants(query)
        all_queries = [query] + [v for v in variants if v and v.lower() != query.lower()]
        logger.info("MultiQuery: %d variants for '%s'", len(all_queries), query[:60])
        return all_queries
    except Exception as e:
        logger.warning("MultiQuery: variant generation failed, using original: %s", e)
        return [query]


# ─────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are a helpful assistant for {_APP_DOMAIN}.

Answer questions clearly and accurately using only the provided context.

Guidelines:
- Answer in plain, friendly language.
- If the answer has sequential steps, use a numbered list.
- If listing features or options, use bullet points.
- Never use headers (###) — keep responses flat and readable in chat.
- Max 300 words unless the question genuinely requires more detail.
- If information is missing from the context, say so and suggest checking the documentation."""


def _build_messages(query: str, docs: list, history: list = None) -> list:
    context = "\n\n".join([
        f"--- {doc.get('title', 'No Title')} ---\n{doc['content']}"
        if isinstance(doc, dict) else str(doc)
        for doc in docs
    ])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        # Keep last 6 turns; truncate each entry to 500 chars to avoid token bloat
        for msg in history[-6:]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"][:500] if len(msg["content"]) > 500 else msg["content"],
            })
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return messages


# ─────────────────────────────────────────────
# Non-streaming answer (fallback)
# ─────────────────────────────────────────────
@_openai_retry
def chat_with_assistant(query: str, docs: list, history: list = None, model: str = "gpt-4") -> str:
    messages = _build_messages(query, docs, history)
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.2,
        timeout=30
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Streaming answer
# ─────────────────────────────────────────────
def stream_rag_answer(query: str, docs: list, history: list = None, model: str = "gpt-4"):
    messages = _build_messages(query, docs, history)

    # Retry wraps the *creation* call only — not the iteration
    @_openai_retry
    def _create_stream():
        return _get_client().chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2000,
            temperature=0.2,
            stream=True,
            timeout=30,
        )

    try:
        stream = _create_stream()
    except Exception as e:
        raise RuntimeError("OpenAI service temporarily unavailable. Please try again in a moment.") from e

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# ─────────────────────────────────────────────
# Full RAG pipeline (non-streaming)
# ─────────────────────────────────────────────
def generate_rag_answer(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None
) -> str:
    rewritten = rewrite_query(query)
    docs_and_meta = hybrid_retrieve_pg(rewritten, top_k)

    if not docs_and_meta:
        return OUT_OF_DOMAIN_MSG

    max_score = max(
        meta.get("vector_score", meta.get("score", 0))
        for _, meta in docs_and_meta
    )
    if max_score < OUT_OF_DOMAIN_THRESHOLD:
        return OUT_OF_DOMAIN_MSG

    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        chunk_id = meta.get("id", f"chunk-{i}")
        section = meta.get("section", "")
        title = f"{section} / {chunk_id}" if section else chunk_id
        docs.append({"title": title, "content": doc})

    return chat_with_assistant(query, docs, history=history, model=model)


# ─────────────────────────────────────────────
# Full RAG pipeline (streaming, used by app.py)
# ─────────────────────────────────────────────
def stream_rag_pipeline(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None,
    collected_sources: list = None,
    timings: dict = None,          # Optional — populated with per-phase ms if provided
):
    """
    Generator that rewrites query, retrieves docs, checks domain, then streams the answer.
    Yields str chunks. If out-of-domain, yields the OUT_OF_DOMAIN_MSG as a single chunk.

    If `timings` dict is provided it will be populated with:
        timings['rewrite_ms']   — query rewrite latency
        timings['retrieve_ms']  — hybrid retrieval latency
        timings['top_score']    — max cosine similarity of retrieved docs
    """
    # ── Phase 1: Query rewrite ──
    t0 = time.perf_counter()
    rewritten = rewrite_query(query)
    if timings is not None:
        timings["rewrite_ms"] = int((time.perf_counter() - t0) * 1000)

    # ── Phase 2: Multi-query retrieval with section filtering ──
    t1 = time.perf_counter()
    from retrieval import detect_section_hint, multi_query_retrieve
    section_hint = detect_section_hint(rewritten)
    query_variants = generate_query_variants(rewritten)
    docs_and_meta = multi_query_retrieve(
        query_variants,
        top_k=top_k,
        section_hint=section_hint,
    )
    if timings is not None:
        timings["retrieve_ms"] = int((time.perf_counter() - t1) * 1000)
        timings["query_variants"] = len(query_variants)
        timings["section_hint"] = section_hint or ""

    # No results at all → out of domain
    if not docs_and_meta:
        yield OUT_OF_DOMAIN_MSG
        return

    # Use vector_score (cosine similarity, 0-1) for domain detection.
    max_score = max(
        meta.get("vector_score", meta.get("score", 0))
        for _, meta in docs_and_meta
    )
    if timings is not None:
        timings["top_score"] = round(max_score, 4)

    logger.info("Domain: Max vector score: %.3f (threshold=%s)", max_score, OUT_OF_DOMAIN_THRESHOLD)

    if max_score < OUT_OF_DOMAIN_THRESHOLD:
        yield OUT_OF_DOMAIN_MSG
        return

    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        chunk_id = meta.get("id", f"chunk-{i}")
        section = meta.get("section", "")
        title = f"{section} / {chunk_id}" if section else chunk_id
        docs.append({"title": title, "content": doc})

    if collected_sources is not None:
        for doc, meta in docs_and_meta[:top_k]:
            collected_sources.append({
                "id": meta.get("id", ""),
                "section": meta.get("section", ""),
            })

    yield from stream_rag_answer(query, docs, history=history, model=model)


if __name__ == "__main__":
    from retrieval import hybrid_retrieve_pg
    q = input("Question: ")
    for chunk in stream_rag_pipeline(q, hybrid_retrieve_pg):
        print(chunk, end="", flush=True)
    print()
