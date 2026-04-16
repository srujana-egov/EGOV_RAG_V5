"""
DIGIT Studio Support Bot
"""

import streamlit as st
from generator import stream_rag_pipeline
from retrieval import hybrid_retrieve_pg

from utils import (
    get_conn,
    get_env_var,
    log_feedback,
    get_recent_feedback,
    get_feedback_stats,
    insert_chunk,
    ensure_feedback_table
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

# Ensure DB tables exist
try:
    ensure_feedback_table()
except Exception:
    pass


# ─────────────────────────────────────────────
# Predetermined Q&A table
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
                    confidence FLOAT DEFAULT 1.0,
                    positive_votes INT DEFAULT 0,
                    negative_votes INT DEFAULT 0,
                    source VARCHAR(20) DEFAULT 'manual',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        conn.close()


_ensure_qa_table()


# ─────────────────────────────────────────────
# Predetermined answer logic
# ─────────────────────────────────────────────
def get_predetermined_answer(query: str, threshold: float = 0.85):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, question, answer, confidence
                FROM predetermined_qa
                WHERE confidence >= %s
            """, (threshold,))
            rows = cur.fetchall()

        q_words = set(w for w in query.lower().split() if len(w) > 3)

        # Domain filter
        if not any(word in query.lower() for word in ["digit", "studio", "service", "workflow", "api", "config"]):
            return None

        best_match = None
        best_score = 0.0

        for row_id, question, answer, confidence in rows:
            p_words = set(w for w in question.lower().split() if len(w) > 3)

            if not q_words or not p_words:
                continue

            common_words = q_words & p_words

            if len(common_words) < 2:
                continue

            overlap = len(common_words) / len(q_words)

            if overlap >= 0.5 and overlap > best_score:
                best_score = overlap
                best_match = {
                    "id": row_id,
                    "answer": answer,
                    "confidence": confidence,
                    "overlap": overlap
                }

        return best_match

    except Exception as e:
        print("[QA Cache Error]", e)
        return None

    finally:
        conn.close()


# ─────────────────────────────────────────────
# Update confidence
# ─────────────────────────────────────────────
def update_qa_confidence(qa_id: int, positive: bool):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if positive:
                cur.execute("""
                    UPDATE predetermined_qa
                    SET positive_votes = positive_votes + 1,
                        confidence = (positive_votes + 1.0) /
                                     NULLIF(positive_votes + negative_votes + 1, 0),
                        updated_at = NOW()
                    WHERE id = %s
                """, (qa_id,))
            else:
                cur.execute("""
                    UPDATE predetermined_qa
                    SET negative_votes = negative_votes + 1,
                        confidence = positive_votes::float /
                                     NULLIF(positive_votes + negative_votes + 1, 0),
                        updated_at = NOW()
                    WHERE id = %s
                """, (qa_id,))
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Promote RAG answer to cache
# ─────────────────────────────────────────────
def promote_to_cache(query: str, answer: str, confidence: float = 0.7):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predetermined_qa (question, answer, confidence, source)
                VALUES (%s, %s, %s, 'rag_promoted')
            """, (query, answer, confidence))
        conn.commit()
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Session state
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

# Display chat
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
            source = "predetermined"
            qa_id = cached["id"]

            st.markdown(answer)
            st.caption(f"⚡ Instant answer · {cached['confidence']:.0%}")

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

                # Fallback if no useful answer
                if not has_content or "I don't have enough information" in full_answer:
                    full_answer = (
                        "This assistant is designed to answer questions about DIGIT Studio.\n\n"
                        "I may not be able to help with general knowledge questions like this.\n\n"
                        "📎 Try asking about:\n"
                        "- Services\n"
                        "- Workflows\n"
                        "- Configuration\n"
                        "- DIGIT Studio features"
                    )

                container.markdown(full_answer)

            except Exception as e:
                full_answer = (
                    "Something went wrong while retrieving the answer.\n\n"
                    "📎 https://docs.digit.org/studio"
                )
                container.markdown(full_answer)

            answer = full_answer
            source = "rag"
            qa_id = None

        # STEP 3: Feedback
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍"):
                log_feedback(query, answer, "positive", source)

                if source == "predetermined" and qa_id:
                    update_qa_confidence(qa_id, True)
                elif source == "rag":
                    promote_to_cache(query, answer)

                st.success("Thanks!")

        with col2:
            if st.button("👎"):
                log_feedback(query, answer, "negative", source)

                if source == "predetermined" and qa_id:
                    update_qa_confidence(qa_id, False)

                st.info("Feedback noted")

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})
