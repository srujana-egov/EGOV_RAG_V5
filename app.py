"""
DIGIT Studio Support Bot
========================
Flow:
  1. User submits query
  2. Check predetermined Q&A cache — if match found, return instantly (no API call, no wait)
  3. If not in cache — run RAG pipeline with streaming
  4. Ask user for 👍/👎 feedback
  5. Positive feedback on RAG answers increments confidence score
  6. Once confidence >= threshold, answer is promoted to predetermined cache
  7. Owner dashboard: see all queries, feedback, cache stats
"""

import json
import streamlit as st
from generator import stream_rag_pipeline
from retrieval import hybrid_retrieve_pg
from utils import (
    log_feedback, get_recent_feedback, get_feedback_stats,
    insert_chunk, get_env_var, ensure_feedback_table, get_conn, release_conn
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

# Ensure DB tables exist
try:
    ensure_feedback_table()
    _ensure_qa_table()
except Exception:
    pass


# ─────────────────────────────────────────────
# Predetermined Q&A store (DB-backed)
# ─────────────────────────────────────────────
def _ensure_qa_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predetermined_qa (
                    id              SERIAL PRIMARY KEY,
                    question        TEXT NOT NULL,
                    answer          TEXT NOT NULL,
                    confidence      FLOAT DEFAULT 1.0,
                    positive_votes  INT DEFAULT 0,
                    negative_votes  INT DEFAULT 0,
                    source          VARCHAR(20) DEFAULT 'manual',
                    created_at      TIMESTAMPTZ DEFAULT NOW(),
                    updated_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()
    finally:
        release_conn(conn)


def get_predetermined_answer(query: str, threshold: float = 0.85) -> dict | None:
    """
    Returns a predetermined answer if one matches the query closely enough.
    Uses simple lowercased keyword overlap for now — fast, zero API cost.
    Returns dict with 'answer' and 'confidence' or None.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, confidence FROM predetermined_qa WHERE confidence >= %s", (threshold,))
            rows = cur.fetchall()
        q_lower = query.lower().strip()
        q_words = set(w for w in q_lower.split() if len(w) > 3)
        best_match = None
        best_score = 0.0
        for row_id, question, answer, confidence in rows:
            p_words = set(w for w in question.lower().split() if len(w) > 3)
            if not q_words or not p_words:
                continue
            overlap = len(q_words & p_words) / len(q_words | p_words)
            if overlap > best_score and overlap >= 0.6:
                best_score = overlap
                best_match = {"id": row_id, "answer": answer, "confidence": confidence, "overlap": overlap}
        return best_match
    except Exception as e:
        print(f"[QA Cache] Lookup failed: {e}")
        return None
    finally:
        release_conn(conn)


def update_qa_confidence(qa_id: int, positive: bool):
    """Increments vote count and recalculates confidence for a cached answer."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            if positive:
                cur.execute("""
                    UPDATE predetermined_qa
                    SET positive_votes = positive_votes + 1,
                        confidence = (positive_votes + 1.0) / NULLIF(positive_votes + negative_votes + 1, 0),
                        updated_at = NOW()
                    WHERE id = %s
                """, (qa_id,))
            else:
                cur.execute("""
                    UPDATE predetermined_qa
                    SET negative_votes = negative_votes + 1,
                        confidence = positive_votes::float / NULLIF(positive_votes + negative_votes + 1, 0),
                        updated_at = NOW()
                    WHERE id = %s
                """, (qa_id,))
        conn.commit()
    finally:
        release_conn(conn)


def promote_to_cache(query: str, answer: str, confidence: float = 0.7):
    """Adds a RAG-generated answer to the predetermined cache."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO predetermined_qa (question, answer, confidence, source)
                VALUES (%s, %s, %s, 'rag_promoted')
                ON CONFLICT DO NOTHING
            """, (query, answer, confidence))
        conn.commit()
        print(f"[QA Cache] Promoted to cache: '{query[:60]}'")
    except Exception as e:
        print(f"[QA Cache] Promote failed: {e}")
    finally:
        release_conn(conn)


def get_all_qa_entries() -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, question, answer, confidence, positive_votes, negative_votes, source, created_at
                FROM predetermined_qa ORDER BY confidence DESC
            """)
            rows = cur.fetchall()
            return [{"id": r[0], "question": r[1], "answer": r[2], "confidence": r[3],
                     "pos": r[4], "neg": r[5], "source": r[6], "created_at": r[7]} for r in rows]
    except Exception:
        return []
    finally:
        release_conn(conn)


# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
defaults = {
    "history": [],
    "display_messages": [],
    "last_query": "",
    "last_answer": "",
    "last_source": "rag",
    "last_qa_id": None,
    "pending_feedback": False,
    "admin_mode": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    try:
        st.image("eGov-logo.png", width=120)
    except Exception:
        pass
    st.markdown("### DIGIT Studio Assistant")
    st.caption("Powered by eGov")
    st.divider()

    model_choice = st.selectbox("Model", ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"], index=0)
    top_k = st.slider("Chunks to retrieve", 3, 15, 5)
    confidence_threshold = st.slider("Cache confidence threshold", 0.5, 1.0, 0.85, 0.05,
                                     help="Answers with score above this are served from cache instantly")

    st.divider()
    if st.button("🗑️ Clear conversation"):
        for k in ["history", "display_messages", "last_query", "last_answer", "pending_feedback"]:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else ""
        st.session_state.pending_feedback = False
        st.rerun()

    st.divider()
    admin_pw = st.text_input("Admin password", type="password", placeholder="Enter to unlock dashboard")
    if admin_pw == get_env_var("ADMIN_PASSWORD", "admin123"):
        st.session_state.admin_mode = True
    elif admin_pw:
        st.error("Wrong password")
        st.session_state.admin_mode = False


# ─────────────────────────────────────────────
# Main layout: Chat | Admin tabs
# ─────────────────────────────────────────────
if st.session_state.admin_mode:
    tab_chat, tab_queries, tab_cache, tab_add = st.tabs(["💬 Chat", "📊 Queries & Feedback", "🗄️ Q&A Cache", "📝 Add Content"])
else:
    tab_chat, = st.tabs(["💬 Chat"])
    tab_queries = tab_cache = tab_add = None


# ─────────────────────────────────────────────
# TAB 1: Chat
# ─────────────────────────────────────────────
with tab_chat:
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown("## 🛠️")
    with col_title:
        st.markdown("## DIGIT Studio Assistant")
        st.caption("Ask anything about DIGIT Studio")

    st.divider()

    # Render conversation history
    for msg in st.session_state.display_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("badge"):
                st.caption(msg["badge"])

    # Empty state suggestions
    if not st.session_state.display_messages:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What is DIGIT Studio?",
            "How do I create a form in Studio?",
            "How do I deploy a workflow?",
            "What are the available components?",
            "How do I configure user roles?",
        ]
        cols = st.columns(2)
        for i, s in enumerate(suggestions):
            with cols[i % 2]:
                if st.button(s, key=f"sug_{i}", use_container_width=True):
                    st.session_state["_pending_query"] = s
                    st.rerun()

    # Handle suggestion click
    if "_pending_query" in st.session_state:
        query = st.session_state.pop("_pending_query")
    else:
        query = st.chat_input("Ask anything about DIGIT Studio...")

    if query and query.strip():
        st.session_state.display_messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            # ── Step 1: Check predetermined cache first
            cached = get_predetermined_answer(query, threshold=confidence_threshold)

            if cached:
                # Instant answer — no API call, no wait
                st.markdown(cached["answer"])
                badge = f"⚡ Instant answer · confidence {cached['confidence']:.0%}"
                st.caption(badge)
                answer = cached["answer"]
                source = "predetermined"
                qa_id = cached["id"]
            else:
                # ── Step 2: Stream RAG answer
                response_container = st.empty()
                full_answer = ""
                try:
                    for chunk in stream_rag_pipeline(
                        query=query,
                        hybrid_retrieve_pg=hybrid_retrieve_pg,
                        top_k=top_k,
                        model=model_choice,
                        history=st.session_state.history
                    ):
                        full_answer += chunk
                        response_container.markdown(full_answer + "▌")
                    response_container.markdown(full_answer)
                except Exception as e:
                    full_answer = f"Sorry, I ran into an error: {e}\n\n📎 Check the docs: https://docs.digit.org/studio"
                    response_container.markdown(full_answer)

                answer = full_answer
                source = "rag"
                qa_id = None
                st.caption("💡 Answer generated from documentation")

            # ── Step 3: Feedback buttons
            st.divider()
            st.caption("Was this helpful?")
            fc1, fc2, fc3 = st.columns([1, 1, 8])
            with fc1:
                if st.button("👍", key=f"pos_{len(st.session_state.display_messages)}"):
                    log_feedback(query, answer, "positive", source)
                    if source == "predetermined" and qa_id:
                        update_qa_confidence(qa_id, positive=True)
                    elif source == "rag":
                        # RAG answer with positive feedback → promote to cache with initial confidence
                        promote_to_cache(query, answer, confidence=0.7)
                    st.success("Thanks! 🙌")
            with fc2:
                if st.button("👎", key=f"neg_{len(st.session_state.display_messages)}"):
                    log_feedback(query, answer, "negative", source)
                    if source == "predetermined" and qa_id:
                        update_qa_confidence(qa_id, positive=False)
                    st.info("Thanks for the feedback.")

        # Update session
        st.session_state.display_messages.append({
            "role": "assistant",
            "content": answer,
            "badge": f"{'⚡ Cache' if source == 'predetermined' else '🔍 RAG'}"
        })
        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": answer})
        st.session_state.last_query = query
        st.session_state.last_answer = answer
        st.session_state.last_source = source


# ─────────────────────────────────────────────
# TAB 2: Queries & Feedback (admin only)
# ─────────────────────────────────────────────
if tab_queries and st.session_state.admin_mode:
    with tab_queries:
        st.subheader("📊 Usage & Feedback")

        stats = get_feedback_stats()
        if stats:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total queries", stats.get("total", 0))
            c2.metric("👍 Positive", stats.get("positive", 0))
            c3.metric("👎 Negative", stats.get("negative", 0))
            c4.metric("Satisfaction", f"{stats.get('satisfaction', 0)}%")
            c5.metric("From cache", stats.get("from_cache", 0))

        st.divider()
        st.subheader("Recent feedback")

        feedback = get_recent_feedback(50)
        if feedback:
            for f in feedback:
                icon = "👍" if f["rating"] == "positive" else "👎"
                src = "⚡" if f["source"] == "predetermined" else "🔍"
                with st.expander(f"{icon} {src} {str(f['created_at'])[:16]} — {f['query'][:80]}"):
                    st.markdown(f"**Rating:** {f['rating']}  |  **Source:** {f['source']}")
                    if f["comment"]:
                        st.markdown(f"**Comment:** {f['comment']}")
        else:
            st.info("No feedback yet.")


# ─────────────────────────────────────────────
# TAB 3: Q&A Cache (admin only)
# ─────────────────────────────────────────────
if tab_cache and st.session_state.admin_mode:
    with tab_cache:
        st.subheader("🗄️ Predetermined Q&A Cache")
        st.caption("Answers here are served instantly with no API call. RAG answers get promoted here when users rate them 👍.")

        # Add manual Q&A
        with st.expander("➕ Add manual Q&A"):
            new_q = st.text_input("Question")
            new_a = st.text_area("Answer", height=150)
            if st.button("Add to cache", type="primary"):
                if new_q and new_a:
                    promote_to_cache(new_q, new_a, confidence=1.0)
                    st.success("Added!")
                    st.rerun()

        st.divider()
        entries = get_all_qa_entries()
        if entries:
            for e in entries:
                conf_color = "🟢" if e["confidence"] >= 0.85 else "🟡" if e["confidence"] >= 0.6 else "🔴"
                with st.expander(f"{conf_color} [{e['confidence']:.0%}] {e['question'][:80]}"):
                    st.markdown(f"**Answer:** {e['answer'][:400]}...")
                    st.caption(f"Source: {e['source']} · 👍 {e['pos']} · 👎 {e['neg']} · Added: {str(e['created_at'])[:10]}")
                    if st.button("🗑️ Delete", key=f"del_{e['id']}"):
                        conn = get_conn()
                        try:
                            with conn.cursor() as cur:
                                cur.execute("DELETE FROM predetermined_qa WHERE id = %s", (e["id"],))
                            conn.commit()
                        finally:
                            release_conn(conn)
                        st.rerun()
        else:
            st.info("Cache is empty. Answers will be added here when users rate RAG responses positively.")


# ─────────────────────────────────────────────
# TAB 4: Add Content (admin only)
# ─────────────────────────────────────────────
if tab_add and st.session_state.admin_mode:
    with tab_add:
        st.subheader("📝 Add content to knowledge base")
        st.caption("Paste documentation directly — no JSON needed. Content is embedded and searchable immediately.")

        admin_title = st.text_input("Page title", placeholder="e.g. How to create a form in Studio")
        admin_url = st.text_input("Source URL", placeholder="https://docs.digit.org/studio/...")
        admin_content = st.text_area("Content", placeholder="Paste documentation here...", height=300)
        admin_tag = st.text_input("Tag (optional)", placeholder="e.g. studio, forms, workflow")

        if st.button("✅ Add to knowledge base", type="primary"):
            if not admin_title or not admin_content:
                st.error("Title and content are required.")
            else:
                with st.spinner("Embedding and storing..."):
                    try:
                        from retrieval import get_embedding
                        doc_id = f"manual/{admin_title.lower().replace(' ', '_')}"
                        insert_chunk(doc_id, admin_content, {
                            "url": admin_url, "tag": admin_tag, "version": "manual"
                        }, get_embedding)
                        st.success(f"✅ '{admin_title}' added to the knowledge base!")
                    except Exception as e:
                        st.error(f"Failed: {e}")
