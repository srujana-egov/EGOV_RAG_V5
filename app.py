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
# HTTP fetcher for images (server-side)
# -----------------------
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

@st.cache_data(show_spinner=False)
def fetch_image_bytes(url: str, timeout: int = 10) -> Union[bytes, None]:
    """
    Try to fetch image bytes server-side and return bytes. Return None on failure.
    Cached to avoid repeated downloads.
    """
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
        resp.raise_for_status()
        return resp.content
    except Exception:
        return None

# -----------------------
# Patterns & helpers
# -----------------------
YOUTUBE_REGEX = r'https?://(?:www\.)?(?:youtube\.com/watch\?v=[\w-]+(?:[^\s]*)|youtu\.be/[\w-]+(?:[^\s]*))'
URL_REGEX = r'https?://[^\s)>\]"]+'

def find_urls_in_text(text: str) -> List[str]:
    if not text:
        return []
    return [m.group(0).rstrip('.,;:') for m in re.finditer(URL_REGEX, text)]

def _extract_inner_image_from_gitbook(url: str) -> Union[str, None]:
    """
    Decode inner image URL from GitBook-style URLs: '?url=<encoded_image_url>'
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
    u = url.lower()
    if re.search(r'\.(png|jpe?g|gif|bmp|webp)(?:[?#]|$)', u):
        return True
    if re.search(r'%2e(?:png|jpe?g|gif|bmp|webp)', u):
        return True
    if 'image?' in u or 'image%3f' in u or 'media?' in u:
        return True
    return False

def _show_link_and_button(url: str, label: str = "Open image"):
    st.markdown(f"[Link]({url})")
    # streamlit doesn't have a native open-in-new-tab button, but we can render HTML anchor that opens in new tab
    btn_html = f"""<a target="_blank" href="{url}"><button style="padding:6px 10px;border-radius:6px;border:1px solid #ccc;background:#f6f6f6">🔗 {label}</button></a>"""
    st.markdown(btn_html, unsafe_allow_html=True)

def _embed_gitbook_proxy_if_present_and_fetch(url: str) -> bool:
    """
    If url is a GitBook proxy like /~gitbook/image?url=..., decode inner URL and try to fetch & embed.
    Returns True if handled (embedded or fallback link shown); False otherwise.
    """
    lowered = url.lower()
    if '/~gitbook/image' in lowered or 'gitbook/image?' in lowered or 'docs.digit.org/~gitbook/image' in lowered:
        try:
            parsed = urllib.parse.urlparse(url)
            q = urllib.parse.parse_qs(parsed.query)
            inner_encoded = q.get("url", [""])[0]
            if inner_encoded:
                inner_decoded = urllib.parse.unquote(inner_encoded)
                # try server-side fetch
                img_bytes = fetch_image_bytes(inner_decoded)
                if img_bytes:
                    st.image(img_bytes, caption="Diagram", use_container_width=True)
                    return True
                else:
                    # attempt fallback: try outer proxy URL server-side too (rarely works)
                    outer_bytes = fetch_image_bytes(url)
                    if outer_bytes:
                        st.image(outer_bytes, caption="Diagram (via proxy)", use_container_width=True)
                        return True
                    # fallback to link + open-in-new-tab button
                    st.markdown("**Diagram (could not embed — opened as link)**")
                    _show_link_and_button(inner_decoded, "Open decoded image in new tab")
                    return True
        except Exception:
            # if something goes wrong, return False to allow normal handler
            return False
    return False

def _embed_url(url: str):
    """
    Try to embed the URL as a YouTube video or image. Uses server-side fetch for images to avoid hotlink issues.
    Fallbacks to link + button.
    """
    url = url.rstrip('.,;:')  # strip trailing punctuation

    # 1) GitBook proxy explicit handler (highest priority)
    try:
        handled = _embed_gitbook_proxy_if_present_and_fetch(url)
        if handled:
            return
    except Exception:
        pass

    # 2) YouTube (embed client-side)
    if re.match(YOUTUBE_REGEX, url, flags=re.IGNORECASE):
        try:
            st.video(url)
            return
        except Exception:
            st.markdown(f"[Video link]({url})")
            return

    # 3) Try decode inner url generically (other hosts using ?url=)
    inner = _extract_inner_image_from_gitbook(url)
    if inner:
        if _looks_like_image(inner):
            bytes_inner = fetch_image_bytes(inner)
            if bytes_inner:
                st.image(bytes_inner, use_container_width=True)
                return
            else:
                # try outer
                outer_bytes = fetch_image_bytes(url)
                if outer_bytes:
                    st.image(outer_bytes, use_container_width=True)
                    return
                st.markdown("**Image (could not embed) — opened as link**")
                _show_link_and_button(inner, "Open decoded image in new tab")
                return
        else:
            # if inner isn't image-like, still try fetching inner bytes (maybe direct)
            bytes_inner = fetch_image_bytes(inner)
            if bytes_inner:
                st.image(bytes_inner, use_container_width=True)
                return

    # 4) If the url itself looks like an image, try fetching server-side first
    if _looks_like_image(url):
        img_bytes = fetch_image_bytes(url)
        if img_bytes:
            st.image(img_bytes, use_container_width=True)
            return
        else:
            st.markdown("**Image (could not embed) — opened as link**")
            _show_link_and_button(url, "Open image in new tab")
            return

    # 5) Final fallback: clickable link
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
        if _looks_like_image(u) or _extract_inner_image_from_gitbook(u) or '/~gitbook/image' in u.lower():
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
# JSON helpers: extract URLs from JSON-like structures
# -----------------------
def extract_urls_from_json(obj: Any) -> List[str]:
    """
    Recursively walk JSON-like object and return any URLs found in strings.
    """
    urls: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            urls.extend(extract_urls_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            urls.extend(extract_urls_from_json(item))
    elif isinstance(obj, str):
        urls.extend(find_urls_in_text(obj))
    # deduplicate while preserving order
    return list(dict.fromkeys(urls))


# -----------------------
# UI: Header, Demo & References
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
                # 1. Call retriever once (we'll reuse docs)
                docs_and_meta = hybrid_retrieve_pg(query, top_k=5)

                # 2. Generate answer (string expected). Provide retriever result wrapper so generator can use it.
                answer = generate_rag_answer(query, lambda q, top_k=5: docs_and_meta)

                st.success("Answer:")

                # 3. Inspect answer (raw text) and JSON inside it for media URLs
                raw_urls = find_urls_in_text(answer)

                parsed_json = None
                try:
                    parsed_json = json.loads(answer)
                except Exception:
                    parsed_json = None

                json_urls = extract_urls_from_json(parsed_json) if parsed_json is not None else []
                combined_answer_urls = list(dict.fromkeys(raw_urls + json_urls))

                # 4. If we found media in the answer's raw text or JSON -> embed them first
                if combined_answer_urls:
                    for u in combined_answer_urls:
                        _embed_url(u)
                    # After embedding, show structured JSON/table if present
                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        st.subheader("Structured Output (table)")
                        df = pd.json_normalize(parsed_json)
                        st.dataframe(df)
                    elif isinstance(parsed_json, dict):
                        st.subheader("Structured Output (JSON)")
                        st.json(parsed_json)
                    else:
                        # Also render the (possibly non-JSON) remainder in the answer
                        render_text_with_media(answer)
                else:
                    # 5. No media in answer -> scan retrieved chunks for media and embed them
                    embedded_from_chunks = False
                    if docs_and_meta:
                        for i, item in enumerate(docs_and_meta, start=1):
                            # item may be (doc, meta) or a dict
                            try:
                                doc, meta = item
                            except Exception:
                                if isinstance(item, dict):
                                    doc = item.get("doc") or item.get("text") or ""
                                    meta = item.get("meta", {})
                                else:
                                    doc = str(item)
                                    meta = {}

                            # collect urls from doc text and meta fields
                            chunk_urls: List[str] = []
                            if isinstance(doc, str):
                                chunk_urls.extend(find_urls_in_text(doc))
                            else:
                                try:
                                    chunk_urls.extend(find_urls_in_text(json.dumps(doc)))
                                except Exception:
                                    pass

                            # meta might have source url(s)
                            try:
                                if isinstance(meta, dict):
                                    for v in meta.values():
                                        if isinstance(v, str):
                                            chunk_urls.extend(find_urls_in_text(v))
                                        else:
                                            try:
                                                chunk_urls.extend(find_urls_in_text(json.dumps(v)))
                                            except Exception:
                                                pass
                            except Exception:
                                pass

                            chunk_urls = list(dict.fromkeys(chunk_urls))
                            if chunk_urls:
                                st.subheader(f"Embedded media from Retrieved Chunk {i}")
                                for u in chunk_urls:
                                    _embed_url(u)
                                    embedded_from_chunks = True

                    # 6. Show structured JSON / table / text (answer)
                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        st.subheader("Structured Output (table)")
                        df = pd.json_normalize(parsed_json)
                        st.dataframe(df)
                    elif isinstance(parsed_json, dict):
                        st.subheader("Structured Output (JSON)")
                        st.json(parsed_json)
                    else:
                        render_text_with_media(answer)

                    if not combined_answer_urls and not embedded_from_chunks:
                        st.info("No embeddable media found in the answer or retrieved chunks.")

                # -----------------------
                # Finally: show retrieved chunks with expanders (transparency)
                # -----------------------
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
