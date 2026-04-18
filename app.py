"""
DIGIT Studio Support Bot
"""

import streamlit as st
from generator import stream_rag_pipeline, OUT_OF_DOMAIN_MSG
from retrieval import hybrid_retrieve_pg, get_embedding, get_embeddings_batch, ensure_fts_index

from utils import (
    get_conn,
    log_feedback,
    log_query,
    log_vote,
    ensure_feedback_table,
    ensure_qa_table_full,
    ensure_query_history_table,
    ensure_vote_log_table,
    update_qa_votes_and_promote,
    ensure_section_column,
)

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(page_title="DIGIT Studio Assistant", page_icon="🛠️", layout="wide")

# Ensure DB tables and indexes exist
try:
    ensure_feedback_table()
    ensure_qa_table_full()
    ensure_query_history_table()
    ensure_vote_log_table()
    ensure_fts_index()
    ensure_section_column()
except Exception:
    pass


@st.cache_data(ttl=300, show_spinner=False)
def _load_qa_cache():
    """
    Load all predetermined Q&A rows into memory (refreshes every 5 min
    to pick up auto-promoted entries). No DB round-trip on every message.
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, confidence FROM predetermined_qa")
            return cur.fetchall()
    finally:
        conn.close()


@st.cache_data(ttl=3600, show_spinner=False)
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
                st.caption("⚡ Instant answer")

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
                            update_qa_votes_and_promote(
                                msg.get("query", ""), msg["content"], "positive"
                            )
                            _load_qa_cache.clear()
                            _load_faq_embeddings.clear()
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
    # Append user message (cap history at 20 to avoid unbounded growth)
    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "user", "content": query})
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        sources = []

        with st.status("Thinking...", expanded=True) as status:

            # ── STEP 1: Semantic FAQ search ──
            n_faq = len(_load_qa_cache())
            st.write(f"🔎 Step 1: Semantic search across {n_faq} FAQ entries...")
            result_type, result_data = semantic_faq_search(query)

            if result_type == "direct":
                st.write("✅ High-confidence FAQ match — returning instant answer.")
                status.update(label="⚡ Answered from FAQ", state="complete", expanded=False)
                answer = result_data["answer"]
                source = "cache"
                chips = None

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
                        query=query,
                        hybrid_retrieve_pg=hybrid_retrieve_pg,
                        top_k=8,
                        model="gpt-4o",
                        history=st.session_state.history,
                        collected_sources=sources,
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

                except Exception:
                    full_answer = "Something went wrong. Please try again."
                    source = "error"
                    status.update(label="❌ Error", state="error", expanded=False)

                answer = full_answer

        if source in ("rag",):
            # Stream remaining chunks after status closes
            def _remaining_gen():
                yield answer  # already have first chunk(s)
                # rag_gen may still have chunks if we didn't exhaust it above
                try:
                    for chunk in rag_gen:
                        yield chunk
                except Exception:
                    pass

            answer = st.write_stream(_remaining_gen())
        else:
            st.markdown(answer)

        # Render chip buttons immediately after the answer
        if source == "chips" and chips:
            st.markdown("**Did you mean one of these?**")
            for idx, chip in enumerate(chips):
                if st.button(chip["question"], key=f"new_chip_{idx}"):
                    st.session_state.pending_chip_query = chip["question"]
                    st.rerun()

        if source == "cache":
            st.caption("⚡ Instant answer")

        if source == "rag" and sources:
            with st.expander("📚 Sources used", expanded=False):
                for s in sources:
                    label = f"{s['section']} / {s['id']}" if s.get('section') else s['id']
                    st.caption(label)

    # Log every query + answer to query_history
    try:
        log_query(query, answer, source)
    except Exception:
        pass

    # Store message with chip data
    msg_data = {
        "role": "assistant",
        "content": answer,
        "query": query,
        "source": source,
        "feedback": None,
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
