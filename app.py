# app.py
import re
import json
import urllib.parse
import streamlit as st
import pandas as pd
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")


# -----------------------
# Media detection & embedding helpers
# -----------------------
YOUTUBE_REGEX = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+(?:[^\s]*)|youtu\.be/[\w-]+(?:[^\s]*))'
URL_REGEX = r'https?://[^\s)>\]"]+'

def _extract_inner_image_from_gitbook(url: str) -> str | None:
    """
    If url contains an encoded image link in query params (e.g. ?url=...), decode & return it.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        q = urllib.parse.parse_qs(parsed.query)
        for key in ('url', 'src', 'image', 'file'):
            if key in q and q[key]:
                inner = q[key][0]
                return urllib.parse.unquote(inner)
    except Exception:
        return None
    return None

def _looks_like_image(url: str) -> bool:
    """
    Heuristics to decide whether a URL points to an image:
      - direct extension (.png/.jpg/...) possibly followed by ? or #
      - percent-encoded .png/.jpg in the URL (e.g. %2Epng)
      - contains 'image?' pattern (gitbook style)
    """
    u = url.lower()
    if re.search(r'\.(png|jpe?g|gif|bmp|webp)(?:[?#]|$)', u):
        return True
    if re.search(r'%2e(?:png|jpe?g|gif|bmp|webp)', u):
        return True
    if 'image?' in u or 'image%3f' in u or 'media?' in u:
        return True
    return False

def _embed_url(url: str):
    """
    Try to embed the URL as a YouTube video or image. Fall back to a clickable markdown link.
    """
    url = url.rstrip('.,;:')  # strip trailing punctuation
    # YouTube
    if re.match(YOUTUBE_REGEX, url, flags=re.IGNORECASE):
        try:
            st.video(url)
            return
        except Exception:
            st.markdown(f"[Video link]({url})")
            return

    # GitBook / encoded inner image
    inner = _extract_inner_image_from_gitbook(url)
    if inner:
        # If inner appears image-like, try it first
        if _looks_like_image(inner):
            try:
                st.image(inner, use_container_width=True)
                return
            except Exception:
                # fallthrough to try outer url
                pass
        try:
            st.image(inner, use_container_width=True)
            return
        except Exception:
            pass

    # If the url itself looks like an image, try embedding
    if _looks_like_image(url):
        try:
            st.image(url, use_container_width=True)
            return
        except Exception:
            st.markdown(f"[Image link]({url})")
            return

    # Final fallback: clickable link
    st.markdown(f"[Link]({url})")

def has_media(text: str) -> bool:
    """
    Quick heuristic: returns True if text contains a YouTube link or any image-like URL.
    """
    if not text:
        return False
    if re.search(YOUTUBE_REGEX, text, flags=re.IGNORECASE):
        return True
    for m in re.finditer(URL_REGEX, text):
        u = m.group(0)
        if _looks_like_image(u) or _extract_inner_image_from_gitbook(u):
            return True
    return False

def render_text_with_media(text: str):
    """
    Walks the text, splits on URLs, and embeds/prints in original order.
    """
    if not text:
        return
    last_end = 0
    for m in re.finditer(URL_REGEX, text):
        start, end = m.span()
        # text before URL
        if start > last_end:
            chunk = text[last_end:start].strip()
            if chunk:
                st.markdown(chunk)
        url = m.group(0).rstrip('.,;:')
        _embed_url(url)
        last_end = end
    # remainder
    if last_end < len(text):
        tail = text[last_end:].strip()
        if tail:
            st.markdown(tail)


# -----------------------
# UI: Header, Demo & References
# -----------------------
st.title("Tentative eGov RAG - As of Oct 6th, 2025")
st.subheader(
    "Important note: Work under progress, only answers questions until ACCESS subsection of HCM gitbook. "
    "Updated information on console and dashboard to be added."
)

demo_videos = [
    {"Title": "Health Campaign Management Demo Video", "URL": "https://www.youtube.com/watch?v=_yxD9Wjqkfw&t=3s"},
    {"Title": "Features & Workflows Walkthrough Video of HCM", "URL": "https://www.youtube.com/watch?v=NB54Ve_smv0"}
]

with st.expander("🎥 Demo Videos"):
    for v in demo_videos:
        st.markdown(f"**{v['Title']}**")
        st.video(v["URL"])
        st.divider()

reference_urls = [
    "https://docs.digit.org/health/introducing-public-health/whats-new",
    "https://docs.digit.org/health/access/public-health-product-suite/health-campaign-management-hcm/demo"
]
with st.expander("🔗 Reference URLs"):
    for url in reference_urls:
        st.markdown(f"- [{url}]({url})")


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
                answer = generate_rag_answer(query, hybrid_retrieve_pg)  # expected string
                st.success("Answer:")

                # 1) If raw answer contains media -> embed media first (preserving surrounding text)
                if has_media(answer):
                    render_text_with_media(answer)

                    # After media embedding, still try to present structured JSON (if present)
                    try:
                        parsed_json = json.loads(answer)
                    except Exception:
                        parsed_json = None

                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        st.subheader("Structured Output (table)")
                        df = pd.json_normalize(parsed_json)
                        st.dataframe(df)
                    elif isinstance(parsed_json, dict):
                        st.subheader("Structured Output (JSON)")
                        st.json(parsed_json)

                else:
                    # 2) No media present -> try JSON/table first (table appears above plain text)
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
                        # 3) Fallback: render text (this will still embed any unusual media patterns)
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
