"""
DIGIT Studio Support Bot
"""

import streamlit as st
from generator import stream_rag_pipeline
from retrieval import hybrid_retrieve_pg

from utils import (
    get_conn,
    log_feedback,
    ensure_feedback_table
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

try:
    ensure_feedback_table()
except Exception:
    pass


# ─────────────────────────────────────────────
# Ensure Q&A table
# ─────────────────────────────────────────────
def _ensure_qa_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predetermined_qa (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0
                )
            """)
        conn.commit()
    finally:
        conn.close()


_ensure_qa_table()


# ─────────────────────────────────────────────
# Cache logic (FIXED)
# ─────────────────────────────────────────────
def get_predetermined_answer(query: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer FROM predetermined_qa")
            rows = cur.fetchall()

        q = query.strip().lower()
        q_words = set(w for w in q.split() if len(w) > 3)

        best_match = None
        best_score = 0.0

        for row_id, question, answer in rows:
            p = question.strip().lower()

            # ✅ EXACT MATCH
            if q == p:
                return {
                    "id": row_id,
                    "answer": answer,
                    "confidence": 1.0
                }

            p_words = set(w for w in p.split() if len(w) > 3)

            common = q_words & p_words

            if len(common) < 1:
                continue

            overlap = len(common) / len(q_words)

            if overlap >= 0.3 and overlap > best_score:
                best_score = overlap
                best_match = {
                    "id": row_id,
                    "answer": answer,
                    "confidence": overlap
                }

        return best_match

    finally:
        conn.close()


# ─────────────────────────────────────────────
# Session
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("🛠️ DIGIT Studio Assistant")
st.caption("Ask anything about DIGIT Studio")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask a question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):

        # STEP 1: Cache
        cached = get_predetermined_answer(query)

        if cached:
            answer = cached["answer"]
            source = "cache"

            st.markdown(answer)
            st.caption("⚡ Instant answer")

        else:
            # STEP 2: RAG
            full_answer = ""
            container = st.empty()

            try:
                has_content = False

                for chunk in stream_rag_pipeline(
                    query=query,
                    hybrid_retrieve_pg=hybrid_retrieve_pg,
                    top_k=5,
                    model="gpt-4",
                    history=st.session_state.history
                ):
                    has_content = True
                    full_answer += chunk
                    container.markdown(full_answer + "▌")

                if not has_content or "I don't have enough information" in full_answer:
                    full_answer = (
                        "I couldn't find a clear answer in DIGIT Studio documentation.\n\n"
                        "Try rephrasing your question."
                    )

                container.markdown(full_answer)

            except Exception:
                full_answer = "Something went wrong. Please try again."
                container.markdown(full_answer)

            answer = full_answer
            source = "rag"

        # Feedback
        st.divider()

        if st.button("👍"):
            log_feedback(query, answer, "positive", source)
            st.success("Thanks!")

        if st.button("👎"):
            log_feedback(query, answer, "negative", source)
            st.info("Feedback noted")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})
