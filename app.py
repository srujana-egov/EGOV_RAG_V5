import re
import json
import streamlit as st
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg
from utils import log_feedback, insert_chunk, get_env_var

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DIGIT Studio Assistant",
    page_icon="🛠️",
    layout="wide"
)

# ─────────────────────────────────────────────
# Branding
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 6])
with col_logo:
    try:
        st.image("eGov-logo.png", width=120)
    except Exception:
        pass
with col_title:
    st.title("🛠️ DIGIT Studio Assistant")
    st.caption("Your guide to building digital services with DIGIT Studio")

st.divider()

# ─────────────────────────────────────────────
# Session state initialisation
# ─────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []          # conversation turns for context
if "display_messages" not in st.session_state:
    st.session_state.display_messages = [] # what shows in the chat UI
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# ─────────────────────────────────────────────
# Sidebar: Admin panel + settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        index=0
    )

    top_k = st.slider("Chunks to retrieve", min_value=3, max_value=15, value=5)

    st.divider()

    # ── Conversation controls
    st.subheader("💬 Conversation")
    if st.button("🗑️ Clear conversation"):
        st.session_state.history = []
        st.session_state.display_messages = []
        st.session_state.last_query = ""
        st.session_state.last_answer = ""
        st.session_state.feedback_given = False
        st.rerun()

    st.divider()

    # ── Admin: Add content
    st.subheader("📝 Add Content")
    st.caption("Paste Studio documentation directly — no JSON needed")

    with st.expander("➕ Add new content to knowledge base"):
        admin_title = st.text_input("Page title", placeholder="e.g. How to create a form in Studio")
        admin_url = st.text_input("Source URL", placeholder="https://docs.digit.org/studio/...")
        admin_content = st.text_area(
            "Content",
            placeholder="Paste the documentation content here...",
            height=200
        )
        admin_tag = st.text_input("Tag (optional)", placeholder="e.g. studio, forms, workflow")

        if st.button("✅ Add to knowledge base", type="primary"):
            if not admin_title or not admin_content:
                st.error("Title and content are required.")
            else:
                try:
                    from retrieval import get_embedding
                    doc_id = f"manual/{admin_title.lower().replace(' ', '_')}"
                    metadata = {
                        "url": admin_url,
                        "tag": admin_tag,
                        "version": "manual"
                    }
                    insert_chunk(doc_id, admin_content, metadata, get_embedding)
                    st.success(f"✅ '{admin_title}' added to the knowledge base!")
                except Exception as e:
                    st.error(f"Failed to add content: {e}")

    st.divider()

    # ── Feedback log viewer
    st.subheader("📊 Feedback Log")
    if st.button("View recent feedback"):
        try:
            with open("feedback_log.jsonl", "r") as f:
                lines = f.readlines()[-10:]
            for line in lines:
                record = json.loads(line)
                icon = "👍" if record["rating"] == "positive" else "👎"
                st.write(f"{icon} {record['timestamp'][:10]} — {record['query'][:60]}")
        except FileNotFoundError:
            st.info("No feedback logged yet.")

# ─────────────────────────────────────────────
# Main chat area
# ─────────────────────────────────────────────

# Display conversation history
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─────────────────────────────────────────────
# Query input
# ─────────────────────────────────────────────
query = st.chat_input("Ask anything about DIGIT Studio...")

if query and query.strip():
    # Show user message immediately
    st.session_state.display_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            try:
                answer = generate_rag_answer(
                    query=query,
                    hybrid_retrieve_pg=hybrid_retrieve_pg,
                    top_k=top_k,
                    model=model_choice,
                    history=st.session_state.history
                )
            except Exception as e:
                answer = f"Sorry, I ran into an error: {e}\n\nPlease try again or check the DIGIT Studio docs directly."

        st.markdown(answer)

        # ── Feedback buttons
        st.divider()
        st.caption("Was this answer helpful?")
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 6])

        with fb_col1:
            if st.button("👍 Yes", key=f"pos_{len(st.session_state.display_messages)}"):
                log_feedback(query, answer, "positive")
                st.success("Thanks for the feedback!")

        with fb_col2:
            if st.button("👎 No", key=f"neg_{len(st.session_state.display_messages)}"):
                log_feedback(query, answer, "negative")
                st.info("Thanks — we'll use this to improve.")

    # Update session state
    st.session_state.display_messages.append({"role": "assistant", "content": answer})

    # Update history for context (OpenAI format)
    st.session_state.history.append({"role": "user", "content": query})
    st.session_state.history.append({"role": "assistant", "content": answer})

    # Store for feedback reference
    st.session_state.last_query = query
    st.session_state.last_answer = answer

# ─────────────────────────────────────────────
# Empty state — show suggested questions
# ─────────────────────────────────────────────
if not st.session_state.display_messages:
    st.markdown("### 💡 Try asking:")
    suggestions = [
        "How do I create a new form in DIGIT Studio?",
        "What is DIGIT Studio used for?",
        "How do I deploy a workflow?",
        "What are the components available in Studio?",
        "How do I configure roles and permissions?",
    ]
    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"suggestion_{i}", use_container_width=True):
                # Trigger the query by rerunning with the suggestion
                st.session_state["_pending_query"] = suggestion
                st.rerun()

# Handle suggestion click
if "_pending_query" in st.session_state:
    pending = st.session_state.pop("_pending_query")
    st.session_state.display_messages.append({"role": "user", "content": pending})
    with st.spinner("Searching..."):
        try:
            answer = generate_rag_answer(
                query=pending,
                hybrid_retrieve_pg=hybrid_retrieve_pg,
                top_k=top_k,
                model=model_choice,
                history=st.session_state.history
            )
        except Exception as e:
            answer = f"Sorry, I ran into an error: {e}"
    st.session_state.display_messages.append({"role": "assistant", "content": answer})
    st.session_state.history.append({"role": "user", "content": pending})
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.rerun()
