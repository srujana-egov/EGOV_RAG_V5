"""
DIGIT Studio Bot — Query Analytics
Navigate to this page via the Streamlit sidebar.
"""
import streamlit as st
import pandas as pd
from utils import get_conn

st.set_page_config(page_title="Bot Analytics", page_icon="📊", layout="wide")
st.title("📊 Query Analytics")

@st.cache_data(ttl=60, show_spinner=False)
def load_query_history():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, answer, source, created_at
                FROM query_history
                ORDER BY created_at DESC
                LIMIT 500
            """)
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=["Query", "Answer", "Source", "Time"])
    finally:
        conn.close()

@st.cache_data(ttl=60, show_spinner=False)
def load_feedback():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, response, feedback_type, source, created_at
                FROM bot_feedback
                ORDER BY created_at DESC
                LIMIT 200
            """)
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=["Query", "Response", "Feedback", "Source", "Time"])
    finally:
        conn.close()

@st.cache_data(ttl=60, show_spinner=False)
def load_votes():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT query, vote_type, created_at
                FROM vote_log
                ORDER BY created_at DESC
                LIMIT 500
            """)
            rows = cur.fetchall()
            return pd.DataFrame(rows, columns=["Query", "Vote", "Time"])
    finally:
        conn.close()

# ── Summary metrics ──
col1, col2, col3, col4 = st.columns(4)

history_df = load_query_history()
feedback_df = load_feedback()
votes_df = load_votes()

with col1:
    st.metric("Total Queries", len(history_df))
with col2:
    rag_count = (history_df["Source"] == "rag").sum() if not history_df.empty else 0
    cache_count = (history_df["Source"] == "cache").sum() if not history_df.empty else 0
    st.metric("FAQ Hits", int(cache_count))
with col3:
    pos = (votes_df["Vote"] == "positive").sum() if not votes_df.empty else 0
    st.metric("👍 Positive Votes", int(pos))
with col4:
    neg = (votes_df["Vote"] == "negative").sum() if not votes_df.empty else 0
    st.metric("👎 Negative Votes", int(neg))

st.divider()

# ── Source breakdown ──
if not history_df.empty:
    st.subheader("Answer Source Breakdown")
    source_counts = history_df["Source"].value_counts().reset_index()
    source_counts.columns = ["Source", "Count"]
    st.bar_chart(source_counts.set_index("Source"))

st.divider()

# ── Recent queries ──
st.subheader("Recent Queries")
if not history_df.empty:
    st.dataframe(
        history_df[["Time", "Query", "Source"]].head(50),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No queries logged yet.")

st.divider()

# ── Negative feedback ──
st.subheader("Flagged (Negative Feedback)")
if not feedback_df.empty:
    neg_df = feedback_df[feedback_df["Feedback"] == "negative"]
    if not neg_df.empty:
        st.dataframe(neg_df[["Time", "Query", "Source"]].head(50), use_container_width=True, hide_index=True)
    else:
        st.success("No negative feedback yet! 🎉")
else:
    st.info("No feedback logged yet.")

st.caption("Refreshes every 60 seconds. Data from query_history, bot_feedback, vote_log tables.")
