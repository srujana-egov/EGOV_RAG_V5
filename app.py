import streamlit as st
import json
import pandas as pd
import re
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")

# -----------------------
# Helper: render text with media embedding
# -----------------------
def render_text_with_media(text: str):
    """
    Render text in Streamlit and automatically embed:
      - YouTube links (youtube.com/watch?v=... or youtu.be/...)
      - Image links ending with png/jpg/jpeg/gif/bmp/webp (with optional query params)
    Other URLs are left as markdown links.
    """
    if not text:
        return

    # Patterns
    youtube_pattern = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+(?:[^\s]*)|youtu\.be/[\w-]+(?:[^\s]*))'
    image_pattern = r'https?://\S+\.(?:png|jpg|jpeg|gif|bmp|webp)(?:\?\S*)?'

    # Combined pattern to find either youtube or image (keeps order)
    combined_pattern = f'({youtube_pattern})|({image_pattern})'

    last_end = 0
    for match in re.finditer(combined_pattern, text, flags=re.IGNORECASE):
        start, end = match.span()
        # write text before the match (if any)
        if start > last_end:
            chunk = text[last_end:start].strip()
            if chunk:
                st.markdown(chunk)

        url = match.group(0).strip()

        # Determine type and embed
        if re.match(youtube_pattern, url, flags=re.IGNORECASE):
            try:
                st.video(url)
            except Exception:
                # fallback to link if embed fails
                st.markdown(f"[Video link]({url})")
        elif re.match(image_pattern, url, flags=re.IGNORECASE):
            try:
                st.image(url, use_container_width=True)
            except Exception:
                st.markdown(f"[Image link]({url})")
        else:
            st.markdown(f"[Link]({url})")

        last_end = end

    # remaining text after last match
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
                # generate RAG answer (string expected)
                answer = generate_rag_answer(query, hybrid_retrieve_pg)

                st.success("Answer:")

                # Try to parse as JSON for nicer formatting, but always embed media found in raw 'answer'
                parsed_json = None
                try:
                    parsed_json = json.loads(answer)
                except Exception:
                    parsed_json = None

                # If parsed JSON is a list of dicts -> show dataframe and also embed any detected media from raw text
                if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                    df = pd.json_normalize(parsed_json)
                    st.dataframe(df)
                    # Also scan raw answer string for media links and embed
                    render_text_with_media(answer)
                # If it's a dict -> show json viewer and embed media if any in the raw string
                elif isinstance(parsed_json, dict):
                    st.json(parsed_json)
                    render_text_with_media(answer)
                # Not JSON -> directly render text with media embedding
                else:
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
                            # Support two common return shapes: (doc, meta) or dicts/tuples
                            try:
                                doc, meta = item
                            except Exception:
                                # try fallback if item itself is dict
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
                                # show document text
                                if isinstance(doc, (dict, list)):
                                    st.write(doc)
                                else:
                                    st.write(doc)

                                # show source link if present
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
