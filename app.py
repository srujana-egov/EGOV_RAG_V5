"""
DIGIT Studio Support Bot
"""

import re
import streamlit as st
from generator import stream_rag_pipeline, OUT_OF_DOMAIN_MSG
from retrieval import hybrid_retrieve_pg

from utils import (
    get_conn,
    log_feedback,
    ensure_feedback_table,
    ensure_qa_table_full,
    update_qa_votes_and_promote,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

# Ensure DB tables exist
try:
    ensure_feedback_table()
    ensure_qa_table_full()
except Exception:
    pass


@st.cache_data(ttl=300, show_spinner=False)
def _load_qa_cache():
    """
    Load all predetermined Q&A rows into memory once per process (refreshes
    every 5 min to pick up auto-promoted entries). Avoids a DB round-trip on
    every chat message.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, confidence FROM predetermined_qa")
            return cur.fetchall()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Predetermined Q&A matching
# ─────────────────────────────────────────────

# Common English stop words to ignore when matching.
# "digit" and "studio" are added because they appear in every single Q&A in
# this dataset — they carry zero discriminating power and cause false matches
# (e.g. "what is DIGIT Studio" matching "What is a Service in DIGIT Studio").
_STOP_WORDS = {
    "what", "when", "where", "which", "that", "this", "with", "from",
    "have", "does", "will", "can", "how", "why", "who", "are", "the",
    "and", "for", "not", "your", "my", "do", "is", "in", "an", "a",
    "to", "of", "on", "at", "by", "or", "be", "it", "as", "up",
    "about", "into", "after", "before", "during", "while", "there",
    "digit", "studio",
}

# Generic action/modifier words that are too common to be meaningful on their own
_WEAK_WORDS = {"create", "new", "make", "add", "get", "set", "use", "see",
               "view", "edit", "show", "find", "list", "all", "any"}


def _stem(word: str) -> str:
    """Strip common English suffixes so 'workflows' matches 'workflow', etc."""
    for suffix in ("ings", "ing", "tion", "tions", "ed", "ers", "er", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> set:
    """Extract stemmed, meaningful tokens from text."""
    return set(
        _stem(w) for w in re.findall(r'[a-z0-9]+', text.strip().lower())
        if len(w) >= 3 and w not in _STOP_WORDS
    )


def get_predetermined_answer(query: str):
    """
    Match query against predetermined Q&A cache using Jaccard similarity
    on stemmed tokens. Returns the best match dict if score >= 0.25,
    or None to fall through to RAG.

    Jaccard = |common| / |union| guards against one-sided matches where
    generic shared words (e.g. 'create', 'new') inflate a one-directional
    coverage score.

    Rows are loaded from _load_qa_cache() (in-memory, refreshed every 5 min)
    so no DB hit happens per message.
    """
    rows = _load_qa_cache()

    q = query.strip().lower()
    q_words = _tokenize(q)

    if not q_words:
        return None

    # Downweight tokens that are too generic to distinguish topics
    q_strong = q_words - _WEAK_WORDS

    best_match = None
    best_score = 0.0

    for row_id, question, answer, confidence in rows:
        p = question.strip().lower()

        # Exact match → return immediately with full confidence
        if q == p:
            return {"id": row_id, "answer": answer, "confidence": 1.0}

        p_words = _tokenize(p)
        if not p_words:
            continue

        common = q_words & p_words
        if not common:
            continue

        # Jaccard similarity over all tokens
        jaccard = len(common) / len(q_words | p_words)

        # Bonus: if strong (domain-specific) tokens match, boost the score
        strong_common = q_strong & p_words
        if q_strong and strong_common:
            strong_bonus = len(strong_common) / len(q_strong)
            score = jaccard + 0.2 * strong_bonus
        else:
            # No domain-specific token matched — penalise heavily
            score = jaccard * 0.5

        if score >= 0.45 and score > best_score:
            best_score = score
            effective_confidence = min(float(confidence or 1.0), score)
            best_match = {
                "id": row_id,
                "answer": answer,
                "confidence": effective_confidence,
            }

    return best_match


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []   # sent to LLM for context

if "messages" not in st.session_state:
    # Each assistant message also stores: query, source, feedback
    st.session_state.messages = []


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

            # Source badge
            if source == "cache":
                st.caption("⚡ Instant answer")

            # Feedback buttons — only for cache/rag answers, not out-of-domain
            if source in ("cache", "rag"):
                fb = msg.get("feedback")
                if fb is None:
                    col1, col2, _ = st.columns([1, 1, 8])
                    with col1:
                        if st.button("👍", key=f"up_{i}", help="This was helpful"):
                            log_feedback(
                                msg.get("query", ""), msg["content"],
                                "positive", source
                            )
                            update_qa_votes_and_promote(
                                msg.get("query", ""), msg["content"], "positive"
                            )
                            _load_qa_cache.clear()  # bust in-memory cache so promotion is visible
                            st.session_state.messages[i]["feedback"] = "positive"
                            st.rerun()
                    with col2:
                        if st.button("👎", key=f"down_{i}", help="This wasn't helpful"):
                            log_feedback(
                                msg.get("query", ""), msg["content"],
                                "negative", source
                            )
                            st.session_state.messages[i]["feedback"] = "negative"
                            st.rerun()
                else:
                    st.caption("👍 Thanks!" if fb == "positive" else "👎 Noted — flagged for review")


# ─────────────────────────────────────────────
# New query input
# ─────────────────────────────────────────────
query = st.chat_input("Ask a question about DIGIT Studio...")

if query:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        with st.status("Thinking...", expanded=True) as status:

            # ── STEP 1: Predetermined Q&A cache ──
            st.write("🔎 Step 1: Checking Q&A cache (44 preloaded answers)...")
            cached = get_predetermined_answer(query)

            if cached:
                st.write("✅ Found in Q&A cache — returning instant answer.")
                status.update(label="⚡ Answered from cache", state="complete", expanded=False)
                answer = cached["answer"]
                source = "cache"

            else:
                st.write("❌ Not in cache.")
                st.write("🔎 Step 2: Searching studio_manual (documents + Supademo guide)...")

                # ── STEP 2: RAG pipeline (studio_manual DB) ──
                full_answer = ""
                source = "rag"

                try:
                    chunks_collected = []
                    for chunk in stream_rag_pipeline(
                        query=query,
                        hybrid_retrieve_pg=hybrid_retrieve_pg,
                        top_k=5,
                        model="gpt-4",
                        history=st.session_state.history,
                    ):
                        full_answer += chunk
                        chunks_collected.append(chunk)

                    if full_answer.strip() == OUT_OF_DOMAIN_MSG.strip():
                        source = "out_of_domain"
                        st.write("⚠️ Query is outside DIGIT Studio scope.")
                        status.update(label="⚠️ Outside domain", state="complete", expanded=False)
                    else:
                        st.write("✅ Relevant content found — generating answer.")
                        status.update(label="✅ Answered from docs", state="complete", expanded=False)

                except Exception as e:
                    full_answer = "Something went wrong. Please try again."
                    source = "error"
                    status.update(label="❌ Error", state="error", expanded=False)

                answer = full_answer

        st.markdown(answer)
        if source == "cache":
            st.caption("⚡ Instant answer")

    # Append assistant message with metadata for feedback buttons
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "query": query,
        "source": source,
        "feedback": None,
    })
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Rerun so feedback buttons appear via the main message loop
    st.rerun()
