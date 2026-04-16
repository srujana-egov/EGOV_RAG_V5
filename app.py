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


# ─────────────────────────────────────────────
# Predetermined Q&A matching
# ─────────────────────────────────────────────

# Common English stop words to ignore when matching
_STOP_WORDS = {
    "what", "when", "where", "which", "that", "this", "with", "from",
    "have", "does", "will", "can", "how", "why", "who", "are", "the",
    "and", "for", "not", "your", "my", "do", "is", "in", "an", "a",
    "to", "of", "on", "at", "by", "or", "be", "it", "as", "up",
    "about", "into", "after", "before", "during", "while", "there",
}


def get_predetermined_answer(query: str):
    """
    Match query against predetermined Q&A cache.
    Returns the best match dict if confident (>= 50% query coverage),
    or None to fall through to RAG.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, confidence FROM predetermined_qa")
            rows = cur.fetchall()

        q = query.strip().lower()
        # Use regex to extract clean alphanumeric tokens — strips punctuation,
        # quotes, brackets, slashes so "sms/email", '"Service"', "(dev," all
        # split and clean correctly.
        q_words = set(
            w for w in re.findall(r'[a-z0-9]+', q)
            if len(w) >= 3 and w not in _STOP_WORDS
        )

        if not q_words:
            return None

        best_match = None
        best_score = 0.0

        for row_id, question, answer, confidence in rows:
            p = question.strip().lower()

            # Exact match → return immediately with full confidence
            if q == p:
                return {"id": row_id, "answer": answer, "confidence": 1.0}

            p_words = set(
                w for w in re.findall(r'[a-z0-9]+', p)
                if len(w) >= 3 and w not in _STOP_WORDS
            )
            if not p_words:
                continue

            common = q_words & p_words
            if not common:
                continue

            # Coverage = what fraction of the query's key words appear in the question
            query_coverage = len(common) / len(q_words)

            # Require >= 50% coverage to be a "sure" match
            if query_coverage >= 0.5 and query_coverage > best_score:
                best_score = query_coverage
                # Use the lower of DB confidence and match score
                effective_confidence = min(float(confidence or 1.0), query_coverage)
                best_match = {
                    "id": row_id,
                    "answer": answer,
                    "confidence": effective_confidence,
                }

        return best_match

    finally:
        conn.close()


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

        # ── STEP 1: Predetermined Q&A cache ──
        cached = get_predetermined_answer(query)

        if cached:
            answer = cached["answer"]
            source = "cache"
            st.markdown(answer)
            st.caption("⚡ Instant answer")

        else:
            # ── STEP 2: RAG pipeline ──
            full_answer = ""
            source = "rag"
            container = st.empty()

            try:
                for chunk in stream_rag_pipeline(
                    query=query,
                    hybrid_retrieve_pg=hybrid_retrieve_pg,
                    top_k=5,
                    model="gpt-4",
                    history=st.session_state.history,
                ):
                    full_answer += chunk
                    container.markdown(full_answer + "▌")

                # Detect out-of-domain response
                if full_answer.strip() == OUT_OF_DOMAIN_MSG.strip():
                    source = "out_of_domain"

                container.markdown(full_answer)

            except Exception as e:
                full_answer = "Something went wrong. Please try again."
                source = "error"
                container.markdown(full_answer)

            answer = full_answer

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
