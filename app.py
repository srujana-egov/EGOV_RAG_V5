# app.py
import re
import json
import urllib.parse
import streamlit as st
import pandas as pd
import requests
from typing import Any, List, Union
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")

# -----------------------
# (Keep your existing media helpers here: fetch_image_bytes, _embed_url, render_text_with_media, etc.)
# For brevity, paste the same media helper implementations you already have.
# I'll assume the following helper functions exist below:
# - render_text_with_media(text)
# - extract_human_summary(answer)  # used only for display; doesn't modify answer variable
# -----------------------

# If you don't have extract_human_summary, include the robust one from earlier messages.
# (I'll include a short fallback here — replace with your robust extractor if present.)
URL_REGEX = r'https?://[^\s)>\]"]+'
def extract_human_summary(answer: str) -> str:
    # Lightweight fallback: return first long paragraph or first 2 sentences
    if not answer: return ""
    ans = answer.replace('\r\n', '\n').replace('\r', '\n')
    # ignore leading single 'Link' tokens
    ans = re.sub(r'^\s*Link\s*\n+', '', ans, flags=re.IGNORECASE)
    # split paragraphs
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', ans) if p.strip()]
    for p in paragraphs:
        if len(p) > 60 and not p.strip().startswith('{'):
            return p
    # fallback to first two sentences
    sents = re.split(r'(?<=[.!?])\s+', ans.strip())
    return ' '.join(sents[:2]).strip()

# -----------------------
# New: Friendly JSON renderer (appearance only)
# -----------------------
def render_structured_json_friendly(parsed_json: Any):
    """
    Present the parsed JSON in a friendly, non-technical UI without mutating content.
    - Top-level keys are sections.
    - Lists of dicts -> DataFrame tables.
    - Lists of strings -> bullet lists.
    - Strings -> paragraphs.
    All inside expanders so the screen is clean.
    """
    if parsed_json is None:
        return

    # If root is a list of dicts, show that as a table under an expander
    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
        with st.expander("Structured results (table)", expanded=False):
            try:
                df = pd.json_normalize(parsed_json)
                st.dataframe(df)
            except Exception:
                st.write(parsed_json)
        return

    # If root is a dict: create a two-column-like layout using columns for headings + content
    if isinstance(parsed_json, dict):
        # top-level summary/description first if present
        preferred = None
        for key in ("summary", "description", "overview", "title"):
            if key in parsed_json and isinstance(parsed_json[key], str) and len(parsed_json[key].strip()) > 20:
                preferred = parsed_json[key].strip()
                break
        if preferred:
            st.markdown(f"**Summary:** {preferred}")

        # Render other top-level keys in expanders
        for k, v in parsed_json.items():
            # Avoid reprinting the summary chosen above
            if preferred and isinstance(v, str) and v.strip() == preferred:
                continue

            label = str(k)
            with st.expander(label, expanded=False):
                # Strings -> simple paragraph
                if isinstance(v, str):
                    st.markdown(v)
                # list of simple strings -> bullets
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    for item in v:
                        st.write(f"- {item}")
                # list of dicts -> show as table
                elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    try:
                        df = pd.json_normalize(v)
                        st.dataframe(df)
                    except Exception:
                        st.write(v)
                # dict -> key:value pairs (nicely)
                elif isinstance(v, dict):
                    # show key: value lines. If value is complex, pretty-print JSON inside a nested expander.
                    simple_pairs = {}
                    complex_items = {}
                    for subk, subv in v.items():
                        if isinstance(subv, (str, int, float, bool)) or (isinstance(subv, list) and all(isinstance(x, str) for x in subv)):
                            simple_pairs[subk] = subv
                        else:
                            complex_items[subk] = subv
                    if simple_pairs:
                        # show as two-column table
                        df_pairs = pd.DataFrame(list(simple_pairs.items()), columns=["Field", "Value"])
                        st.table(df_pairs)
                    for ck, cv in complex_items.items():
                        with st.expander(f"{ck} (details)"):
                            st.write(cv)
                else:
                    # Catch-all: print whatever it is
                    st.write(v)
    else:
        # If JSON is primitive or unknown, just show it
        st.write(parsed_json)

# -----------------------
# UI: Header
# -----------------------
st.title("Tentative eGov RAG — Friendly View")
st.write("Summary shown first for non-technical users. Structured details available in expanders below.")

# -----------------------
# RAG Query
# -----------------------
query = st.text_input("Enter your question:", placeholder="e.g., What is HCM?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving answer..."):
            # 1. Get docs and answer (unchanged behavior)
            docs_and_meta = hybrid_retrieve_pg(query, top_k=5)
            answer = generate_rag_answer(query, lambda q, top_k=5: docs_and_meta)

            # 2. Show a human-friendly summary (display only; original answer remains unchanged)
            lead = extract_human_summary(answer)
            if lead:
                st.markdown(f"### {lead}")
            else:
                st.info("No summary detected. See structured output below.")

            # 3. Show media found in answer (optional; display only)
            # If you already have render_text_with_media, call it; otherwise show answer raw
            try:
                # render_text_with_media will embed videos/images and show text in order
                render_text_with_media(answer)
            except Exception:
                # fallback: do nothing (avoid clutter) - raw JSON shown below
                pass

            # 4. Show structured JSON in friendly form (appearance only)
            parsed_json = None
            try:
                parsed_json = json.loads(answer)
            except Exception:
                parsed_json = None

            if parsed_json is not None:
                st.markdown("---")
                st.markdown("## More details")
                # Friendly render
                render_structured_json_friendly(parsed_json)

                # Allow raw JSON view and download
                with st.expander("Show raw JSON"):
                    st.json(parsed_json)

                # Download button for JSON
                json_bytes = json.dumps(parsed_json, indent=2).encode("utf-8")
                st.download_button("Download JSON", data=json_bytes, file_name="answer.json", mime="application/json")
            else:
                # If not JSON, still offer raw view + download
                with st.expander("Show raw answer"):
                    st.code(answer)
                st.download_button("Download answer.txt", data=answer.encode("utf-8"), file_name="answer.txt", mime="text/plain")

            # 5. Finally: transparency — show retrieved chunks in expanders (unchanged)
            st.markdown("---")
            st.subheader("Retrieved Chunks Used")
            try:
                if not docs_and_meta:
                    st.info("No retrieved chunks returned by the retriever.")
                else:
                    for i, item in enumerate(docs_and_meta, start=1):
                        try:
                            doc, meta = item
                        except Exception:
                            if isinstance(item, dict):
                                doc = item.get("doc") or item.get("text") or str(item)
                                meta = item.get("meta", {})
                            else:
                                doc = str(item)
                                meta = {}
                        score_str = ""
                        try:
                            score = meta.get("score")
                            if score is not None:
                                score_str = f" (Score: {float(score):.4f})"
                        except Exception:
                            score_str = ""
                        with st.expander(f"Chunk {i}{score_str}", expanded=False):
                            if isinstance(doc, (dict, list)):
                                st.write(doc)
                            else:
                                st.write(doc)
                            source = None
                            try:
                                source = meta.get("source") or meta.get("url") or meta.get("source_url")
                            except Exception:
                                source = None
                            if source:
                                st.markdown(f"[Source]({source})")
            except Exception as e:
                st.warning(f"Could not fetch retrieved chunks: {e}")
