import streamlit as st
import json
import pandas as pd
import re
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")

# -----------------------
# Helper: detect media in text
# -----------------------
YOUTUBE_REGEX = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+(?:[^\s]*)|youtu\.be/[\w-]+(?:[^\s]*))'
IMAGE_REGEX = r'https?://\S+(?:\.(?:png|jpg|jpeg|gif|bmp|webp)(?:\?\S*)?|image\?\S+|media\?\S+)'

def has_media(text: str) -> bool:
    if not text:
        return False
    return bool(re.search(YOUTUBE_REGEX, text, flags=re.IGNORECASE) or re.search(IMAGE_REGEX, text, flags=re.IGNORECASE))

# -----------------------
# Helper: render text with media embedding
# -----------------------
def render_text_with_media(text: str):
    """
    Render text in Streamlit and automatically embed:
      - YouTube links (youtube.com/watch?v=... or youtu.be/...)
      - Image links ending with png/jpg/jpeg/gif/bmp/webp (with optional query params)
    Keeps original text order and renders non-media as markdown.
    """
    if not text:
        return

    combined_pattern = f'({YOUTUBE_REGEX})|({IMAGE_REGEX})'
    last_end = 0
    for match in re.finditer(combined_pattern, text, flags=re.IGNORECASE):
        start, end = match.span()
        # text before match
        if start > last_end:
            chunk = text[last_end:start].strip()
            if chunk:
                st.markdown(chunk)

        url = match.group(0).strip()
        # embed video
        if re.match(YOUTUBE_REGEX, url, flags=re.IGNORECASE):
            try:
                st.video(url)
            except Exception:
                st.markdown(f"[Video link]({url})")
        # embed image
        elif re.match(IMAGE_REGEX, url, flags=re.IGNORECASE):
            try:
                st.image(url, use_container_width=True)
            except Exception:
                st.markdown(f"[Image link]({url})")
        else:
            st.markdown(f"[Link]({url})")

        last_end = end

    # remainder
    if last_end < len(text):
        remainder = text[last_end:].strip()
        if remainder:
            st.markdown(remainder)

# -----------------------
# UI: Header & optional demo/reference sections
# -----------------------
st.title("Tentative eGov RAG - As of Oct 6th, 2025")
st.subheader(
    "Important note: Work under progress, only answers questions until ACCESS subsection of HCM gitbook. "
    "Updated information on console and dashboard to be added."
)

# -----------------------
# RAG Query Interface
# -----------------------
query = st.text_input("Enter your question:", placeholder="e.g., How to pay with HCM?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer = generate_rag_answer(query, hybrid_retrieve_pg)  # expected: string
                st.success("Answer:")

                # 1) If media present in raw answer -> embed media first (and surrounding text)
                if has_media(answer):
                    render_text_with_media(answer)
                    # AFTER embedding media, also try to show structured JSON if present
                    try:
                        parsed_json = json.loads(answer)
                    except Exception:
                        parsed_json = None

                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        df = pd.json_normalize(parsed_json)
                        st.subheader("Structured Output (table)")
                        st.dataframe(df)
                    elif isinstance(parsed_json, dict):
                        st.subheader("Structured Output (JSON)")
                        st.json(parsed_json)

                else:
                    # 2) No media found -> try JSON/table first (so table appears above plain text)
                    parsed_json = None
                    try:
                        parsed_json = json.loads(answer)
                    except Exception:
                        parsed_json = None

                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        df = pd.json_normalize(parsed_json)
                        st.dataframe(df)
                    elif isinstance(parsed_json, dict):
                        st.json(parsed_json)
                    else:
                        # 3) Fall back to plain text (with any media embeddings inside — though none expected)
                        render_text_with_media(answer)

                # -----------------------
                # Transparency: show retrieved chunks
                # -----------------------
                st.subheader("Retrieved Chunks Used")
                try:
                    docs_and_meta = hybrid_retrieve_pg(query, top_k=5)
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

                            with st.expander(f"Chunk {i}{score_str}"):
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

            except Exception as e:
                st.error(f"Error: {str(e)}")
