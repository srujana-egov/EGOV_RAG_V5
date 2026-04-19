"""
DIGIT Studio Support Bot
"""

import time
import uuid
import logging
import streamlit as st

logger = logging.getLogger(__name__)
from generator import stream_rag_pipeline, OUT_OF_DOMAIN_MSG
from retrieval import hybrid_retrieve_pg, get_embedding, get_embeddings_batch, ensure_fts_index

from utils import (
    get_conn,
    get_env_var,
    log_feedback,
    log_query,
    log_vote,
    ensure_feedback_table,
    ensure_qa_table_full,
    ensure_query_history_table,
    ensure_vote_log_table,
    update_qa_votes_and_promote,
    ensure_section_column,
    register_cache_invalidation_callback,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

# ─────────────────────────────────────────────
# Auth gate — password protect the app
# Set APP_PASSWORD env var (or Streamlit secret) to enable.
# Leave unset to run without auth (dev mode).
# ─────────────────────────────────────────────
_APP_PASSWORD = get_env_var("APP_PASSWORD", "")

if _APP_PASSWORD:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 DIGIT Studio Assistant")
        with st.form("login_form"):
            pwd = st.text_input("Enter access password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if pwd == _APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password. Please try again.")
        st.stop()   # halt rendering until authenticated

# ─────────────────────────────────────────────
# Per-session rate limiter
# Limits: MAX_QUERIES_PER_WINDOW queries per RATE_WINDOW_SECONDS
# Defaults: 20 queries / 60 seconds (override via env vars)
# ─────────────────────────────────────────────
_RATE_WINDOW = int(get_env_var("RATE_WINDOW_SECONDS", "60"))
_RATE_MAX    = int(get_env_var("MAX_QUERIES_PER_WINDOW", "20"))

_MAX_QUERY_LEN = 500
_INJECTION_PATTERNS = [
    "ignore all previous instructions",
    "ignore previous instructions",
    "disregard your instructions",
    "you are now",
    "new instructions:",
    "system prompt:",
    "forget everything",
]

if "rate_timestamps" not in st.session_state:
    st.session_state.rate_timestamps = []

def _check_rate_limit() -> bool:
    """Return True if the request is allowed, False if rate-limited."""
    now = time.time()
    # Drop timestamps outside the current window
    st.session_state.rate_timestamps = [
        t for t in st.session_state.rate_timestamps
        if now - t < _RATE_WINDOW
    ]
    if len(st.session_state.rate_timestamps) >= _RATE_MAX:
        return False
    st.session_state.rate_timestamps.append(now)
    return True


def _validate_query(query: str):
    """
    Returns (clean_query, error_msg_or_None).
    Truncates overlength queries; rejects injection attempts.
    """
    q = query.strip()
    if len(q) > _MAX_QUERY_LEN:
        q = q[:_MAX_QUERY_LEN]
        logger.warning("Query truncated to %d chars", _MAX_QUERY_LEN)
    lower = q.lower()
    for pattern in _INJECTION_PATTERNS:
        if pattern in lower:
            logger.warning("Prompt injection attempt detected: %r", q[:80])
            return None, "⚠️ That message contains content that can't be processed. Please ask a question about DIGIT Studio."
    return q, None

# Ensure DB tables and indexes exist
try:
    ensure_feedback_table()
    ensure_qa_table_full()
    ensure_query_history_table()
    ensure_vote_log_table()
    ensure_fts_index()
    ensure_section_column()
except Exception as e:
    logger.error("Startup DB initialisation failed: %s", e)
    st.error(
        "⚠️ Database initialisation failed. Some features may not work correctly. "
        "Check your database configuration and restart the app."
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_qa_cache():
    """
    Load all predetermined Q&A rows into memory (refreshes every 5 min
    to pick up auto-promoted entries). No DB round-trip on every message.
    """
    with get_conn() as conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, question, answer, confidence FROM predetermined_qa "
                    "WHERE status = 'active' OR status IS NULL"
                )
                return cur.fetchall()
        except Exception as e:
            logger.error("QACache: Load failed: %s", e)
            return []


@st.cache_data(ttl=300, show_spinner=False)
def _load_faq_embeddings():
    """
    Embed every FAQ question in one batch API call; cache for 1 hour.
    Returns a list of dicts with keys: id, question, answer, confidence, embedding.
    """
    rows = _load_qa_cache()
    if not rows:
        return []
    questions = [r[1] for r in rows]
    embeddings = get_embeddings_batch(questions)
    return [
        {
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "confidence": r[3],
            "embedding": emb,
        }
        for r, emb in zip(rows, embeddings)
    ]


def _clear_faq_caches():
    _load_qa_cache.clear()
    _load_faq_embeddings.clear()

register_cache_invalidation_callback(_clear_faq_caches)


# ─────────────────────────────────────────────
# Semantic FAQ matching
# ─────────────────────────────────────────────

def _cosine_sim(a: list, b: list) -> float:
    """
    Cosine similarity via dot product. Safe because text-embedding-3-small
    vectors are already L2-normalised (unit length).
    """
    return sum(x * y for x, y in zip(a, b))


def semantic_faq_search(query: str):
    """
    Compare query against all FAQ question embeddings.

    Returns one of:
        ("direct", {"id", "answer", "confidence"})   — score > 0.85
        ("chips",  [{"question", "answer", "score"}, ...])  — score 0.65–0.85, top 3
        ("rag",    None)                             — score < 0.65
    """
    faq_items = _load_faq_embeddings()
    if not faq_items:
        return ("rag", None)

    query_emb = get_embedding(query)

    scored = sorted(
        [(item, _cosine_sim(query_emb, item["embedding"])) for item in faq_items],
        key=lambda x: x[1],
        reverse=True,
    )

    top_score = scored[0][1] if scored else 0.0

    if top_score > 0.80:
        item, score = scored[0]
        return ("direct", {"id": item["id"], "answer": item["answer"], "confidence": score})

    if top_score >= 0.68:
        # Only include chips that are individually relevant (within 0.10 of top score
        # and above a minimum threshold), so we don't pad with loosely related questions.
        min_chip_score = max(0.68, top_score - 0.10)
        chips = [
            {"question": item["question"], "answer": item["answer"], "score": score}
            for item, score in scored[:3]
            if score >= min_chip_score
        ]
        if not chips:
            return ("rag", None)
        return ("chips", chips)

    return ("rag", None)


# ─────────────────────────────────────────────
# Contextual query resolution
# Handles short/ambiguous follow-ups using conversation history
# ─────────────────────────────────────────────
_NEGATIVE_REPLIES = {"no", "nope", "none", "neither", "n", "nah", "not really", "no thanks"}


def _resolve_effective_query(query: str, messages: list) -> str:
    """
    Return the best query to actually search for, given conversation context.

    Cases handled:
    1. Negative reply ("no", "nope", etc.) after chips → use the original user question
       so we fall through to RAG instead of dead-ending with an out-of-domain message.
    2. Short follow-up (≤5 words) after any exchange → prepend the last substantive
       user question for context (e.g. "tell me more" → "what is digit studio tell me more").
    3. Everything else → use the query unchanged.
    """
    q = query.strip()
    words = q.split()

    if len(words) > 5:
        return q  # long enough to stand alone

    normalised = q.lower().rstrip(".,!?")

    # ── Case 1: negative reply after chips ──
    last_assistant = next(
        (m for m in reversed(messages) if m["role"] == "assistant"), None
    )
    if last_assistant and last_assistant.get("source") == "chips":
        if normalised in _NEGATIVE_REPLIES:
            # Find the last non-trivial user message (not the "no" itself)
            original = next(
                (m["content"] for m in reversed(messages)
                 if m["role"] == "user"
                 and m["content"].strip().lower().rstrip(".,!?") != normalised
                 and len(m["content"].strip().split()) > 3),
                None,
            )
            if original:
                return original  # re-run original question through RAG

    # ── Case 2: short follow-up — prepend last substantive user question ──
    last_substantive = next(
        (m["content"] for m in reversed(messages)
         if m["role"] == "user"
         and m["content"].strip().lower().rstrip(".,!?") != normalised
         and len(m["content"].strip().split()) > 5),
        None,
    )
    if last_substantive:
        return f"{last_substantive} {q}"

    return q


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_chip_query" not in st.session_state:
    st.session_state.pending_chip_query = None


# ─────────────────────────────────────────────
# UI header
# ─────────────────────────────────────────────
st.title("🛠️ DIGIT Studio Assistant")
st.caption(
    "Ask anything about DIGIT Studio  •  "
    "Conversation is remembered within this session  •  "
    "⚡ = instant cached answer"
)

# ─────────────────────────────────────────────
# Render conversation history with feedback buttons
# ─────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            source = msg.get("source", "rag")

            # Chip suggestions
            if source == "chips" and msg.get("chips"):
                st.markdown("**Did you mean one of these?**")
                for j, chip in enumerate(msg["chips"]):
                    if st.button(chip["question"], key=f"hist_chip_{i}_{j}"):
                        st.session_state.pending_chip_query = chip["question"]
                        st.rerun()

            # Source badge
            if source == "cache":
                conf = msg.get("faq_confidence")
                conf_pct = f" · {conf:.0%} match" if conf else ""
                st.caption(f"⚡ Instant answer{conf_pct}")

            if source == "rag" and msg.get("sources"):
                with st.expander("📚 Sources used", expanded=False):
                    for s in msg["sources"]:
                        label = f"{s['section']} / {s['id']}" if s.get('section') else s['id']
                        st.caption(label)

            # Feedback buttons — only for cache/rag answers, not chips/out-of-domain
            if source in ("cache", "rag"):
                fb = msg.get("feedback")
                if fb is None:
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{i}", help="This was helpful"):
                            # Positive votes go to vote_log only (not bot_feedback)
                            log_vote(msg.get("query", ""), msg["content"], "positive")
                            promoted = update_qa_votes_and_promote(
                                msg.get("query", ""), msg["content"], "positive"
                            )
                            _load_qa_cache.clear()
                            _load_faq_embeddings.clear()
                            if promoted:
                                st.toast("📋 Answer flagged for admin review before joining the FAQ cache.", icon="📋")
                            st.session_state.messages[i]["feedback"] = "positive"
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{i}", help="This wasn't helpful"):
                            # Negative votes go to bot_feedback (for review) AND vote_log
                            log_feedback(
                                msg.get("query", ""), msg["content"],
                                "negative", source
                            )
                            log_vote(msg.get("query", ""), msg["content"], "negative")
                            update_qa_votes_and_promote(
                                msg.get("query", ""), msg["content"], "negative"
                            )
                            st.session_state.messages[i]["feedback"] = "negative"
                            st.rerun()
                else:
                    st.caption("👍 Thanks!" if fb == "positive" else "👎 Noted — flagged for review")


# ─────────────────────────────────────────────
# Query input (chat box always rendered; chip clicks override it)
# ─────────────────────────────────────────────
chat_input = st.chat_input("Ask a question about DIGIT Studio...")

# Chip click sets pending_chip_query; on next render it becomes the query
if st.session_state.pending_chip_query:
    query = st.session_state.pending_chip_query
    st.session_state.pending_chip_query = None
else:
    query = chat_input

if query:
    # ── Rate limit check ──
    if not _check_rate_limit():
        st.warning(
            f"⏳ You've sent too many messages. Please wait a moment before asking again. "
            f"(Limit: {_RATE_MAX} messages per {_RATE_WINDOW}s)"
        )
        st.stop()

    # ── Input validation + injection defence ──
    query, _validation_error = _validate_query(query)
    if _validation_error:
        st.warning(_validation_error)
        st.stop()

    # Append user message (cap history at 20 to avoid unbounded growth)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "user", "content": query})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        sources = []
        timings = {}           # populated by stream_rag_pipeline
        faq_confidence = None
        t_query_start = time.perf_counter()

        with st.status("Thinking...", expanded=True) as status:

            # ── Contextual query resolution ──
            # Expand/replace short or ambiguous follow-ups using conversation history
            req_id = uuid.uuid4().hex[:8]
            logger.info("query_start req=%s query=%r", req_id, query)
            effective_query = _resolve_effective_query(
                query, st.session_state.messages[:-1]  # exclude the just-appended user msg
            )
            if effective_query != query:
                logger.info("QueryResolve: '%s' → '%s'", query, effective_query)

            # ── STEP 1: Semantic FAQ search ──
            n_faq = len(_load_qa_cache())
            st.write(f"🔎 Step 1: Semantic search across {n_faq} FAQ entries...")
            result_type, result_data = semantic_faq_search(effective_query)

            if result_type == "direct":
                st.write("✅ High-confidence FAQ match — returning instant answer.")
                status.update(label="⚡ Answered from FAQ", state="complete", expanded=False)
                answer = result_data["answer"]
                source = "cache"
                chips = None
                faq_confidence = result_data["confidence"]

            elif result_type == "chips":
                st.write("💡 Found similar questions — showing suggestions.")
                status.update(label="💡 Did you mean…", state="complete", expanded=False)
                answer = "I found some related questions that might help:"
                source = "chips"
                chips = result_data

            else:
                st.write("❌ No close FAQ match (score < 0.68).")
                st.write("🔎 Step 2: Searching studio_manual documents...")

                # ── STEP 2: RAG pipeline ──
                source = "rag"
                chips = None
                full_answer = ""

                try:
                    rag_gen = stream_rag_pipeline(
                        query=effective_query,
                        hybrid_retrieve_pg=hybrid_retrieve_pg,
                        top_k=8,
                        model="gpt-4o",
                        history=st.session_state.history,
                        collected_sources=sources,
                        timings=timings,
                    )
                    # Peek at first chunk to detect out-of-domain before streaming
                    first_chunk = next(rag_gen, "")
                    full_answer = first_chunk

                    if first_chunk.strip().startswith("I'm sorry, that question appears"):
                        # Out-of-domain — collect remaining and show static
                        for chunk in rag_gen:
                            full_answer += chunk
                        source = "out_of_domain"
                        st.write("⚠️ Query is outside DIGIT Studio scope.")
                        status.update(label="⚠️ Outside domain", state="complete", expanded=False)
                    else:
                        st.write("✅ Relevant content found — generating answer.")
                        status.update(label="✅ Answered from docs", state="complete", expanded=False)

                except Exception as e:
                    full_answer = str(e) if "temporarily unavailable" in str(e) \
                        else "Something went wrong. Please try again."
                    source = "error"
                    status.update(label="❌ Error", state="error", expanded=False)

                answer = full_answer

        if source in ("rag",):
            # Stream remaining chunks after status closes
            t_stream_start = time.perf_counter()

            def _remaining_gen():
                yield answer  # already have first chunk(s)
                try:
                    for chunk in rag_gen:
                        yield chunk
                except Exception:
                    pass

            answer = st.write_stream(_remaining_gen())
            timings["generate_ms"] = int((time.perf_counter() - t_stream_start) * 1000)
        else:
            st.markdown(answer)

        # ── Latency caption (RAG answers only) ──
        if source == "rag" and timings:
            rewrite_ms  = timings.get("rewrite_ms", 0)
            retrieve_ms = timings.get("retrieve_ms", 0)
            generate_ms = timings.get("generate_ms", 0)
            total_ms    = int((time.perf_counter() - t_query_start) * 1000)
            logger.info("Latency: rewrite=%dms retrieve=%dms generate=%dms total=%dms",
                        rewrite_ms, retrieve_ms, generate_ms, total_ms)
            st.caption(
                f"⏱ Rewrite {rewrite_ms}ms · Retrieve {retrieve_ms}ms · "
                f"Generate {generate_ms}ms · Total {total_ms}ms"
            )

        # Render chip buttons immediately after the answer
        if source == "chips" and chips:
            st.markdown("**Did you mean one of these?**")
            for idx, chip in enumerate(chips):
                if st.button(chip["question"], key=f"new_chip_{idx}"):
                    st.session_state.pending_chip_query = chip["question"]
                    st.rerun()

        if source == "cache":
            conf_pct = f" · {faq_confidence:.0%} match" if faq_confidence else ""
            st.caption(f"⚡ Instant answer{conf_pct}")

        if source == "rag" and sources:
            with st.expander("📚 Sources used", expanded=False):
                for s in sources:
                    label = f"{s['section']} / {s['id']}" if s.get('section') else s['id']
                    st.caption(label)

    # Log every query + answer to query_history (with latency + top_score)
    try:
        total_ms   = int((time.perf_counter() - t_query_start) * 1000)
        top_score  = timings.get("top_score")
        logger.info("query_done req=%s source=%s latency_ms=%s", req_id, source, int((time.perf_counter() - t_query_start)*1000))
        log_query(query, answer, source, latency_ms=total_ms, top_score=top_score)
    except Exception:
        pass

    # Store message with chip data
    msg_data = {
        "role": "assistant",
        "content": answer,
        "query": query,
        "source": source,
        "feedback": None,
        "faq_confidence": faq_confidence,
    }
    if chips:
        msg_data["chips"] = chips
    if sources:
        msg_data["sources"] = sources

    st.session_state.messages.append(msg_data)
    st.session_state.history.append({"role": "assistant", "content": answer})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

    st.rerun()
