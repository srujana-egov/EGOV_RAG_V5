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

st.set_page_config(page_title="RAG-TDD — Friendly View", layout="wide")

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
# Media detection & embedding helpers
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
                img_bytes = fetch_image_bytes(inner_decoded)
                if img_bytes:
                    st.image(img_bytes, caption="Diagram", use_container_width=True)
                    return True
                else:
                    outer_bytes = fetch_image_bytes(url)
                    if outer_bytes:
                        st.image(outer_bytes, caption="Diagram (via proxy)", use_container_width=True)
                        return True
                    st.markdown("**Diagram (could not embed — opened as link)**")
                    _show_link_and_button(inner_decoded, "Open decoded image in new tab")
                    return True
        except Exception:
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
                outer_bytes = fetch_image_bytes(url)
                if outer_bytes:
                    st.image(outer_bytes, use_container_width=True)
                    return
                st.markdown("**Image (could not embed) — opened as link**")
                _show_link_and_button(inner, "Open decoded image in new tab")
                return
        else:
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
# JSON helpers: extract URLs and texts from JSON-like structures
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
    return list(dict.fromkeys(urls))  # deduplicate while preserving order

# -----------------------
# Human summary extractor (display-only; does NOT modify original content)
# -----------------------
URL_REGEX_SIMPLE = r'https?://[^\s)>\]"]+'
def _is_url(s: str) -> bool:
    return bool(re.search(URL_REGEX_SIMPLE, s))

def _gather_texts_from_json(obj: Any) -> List[str]:
    texts = []
    if isinstance(obj, dict):
        # preferred keys
        for key in ("summary", "description", "overview", "title", "content"):
            if key in obj and isinstance(obj[key], str) and len(obj[key].strip()) > 30 and not _is_url(obj[key]):
                texts.append(obj[key].strip())
        for v in obj.values():
            texts.extend(_gather_texts_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            texts.extend(_gather_texts_from_json(item))
    elif isinstance(obj, str):
        s = obj.strip()
        if len(s) >= 40 and not _is_url(s):
            texts.append(s)
    return texts

def extract_human_summary(answer: str) -> str:
    """
    Robustly extract a single human-readable paragraph from a model answer that may contain JSON.
    This function only returns a display string and does not modify the original answer content.
    """
    if not answer:
        return ""

    ans = answer.replace('\r\n', '\n').replace('\r', '\n')

    # Try parse JSON directly
    parsed_json = None
    try:
        parsed_json = json.loads(ans)
    except Exception:
        # attempt to find JSON substring
        m = re.search(r'(\{|\[)', ans)
        if m:
            pos = m.start()
            candidate = ans[pos:]
            for end in range(len(candidate), 0, -1):
                try:
                    parsed_json = json.loads(candidate[:end])
                    break
                except Exception:
                    continue

    if parsed_json is not None:
        # prefer explicit keys
        if isinstance(parsed_json, dict):
            for k in ("summary", "description", "overview", "title"):
                v = parsed_json.get(k)
                if isinstance(v, str) and len(v.strip()) >= 30 and not _is_url(v):
                    return v.strip()
        candidates = _gather_texts_from_json(parsed_json)
        if candidates:
            return max(candidates, key=len).strip()

    # Fallback: cut before markers or JSON and take first paragraph
    markers = ['Reference URLs', 'Reference URLs:', 'Source', 'Source:', 'Link', 'Links', 'Reference:', 'Retrieved Chunks']
    cut_positions = []
    for m in markers:
        p = ans.find(m)
        if p != -1:
            cut_positions.append(p)
    b = ans.find('{')
    if b != -1:
        cut_positions.append(b)
    b2 = ans.find('[')
    if b2 != -1:
        cut_positions.append(b2)
    if cut_positions:
        cut_at = min(cut_positions)
        leading_text = ans[:cut_at].strip()
    else:
        leading_text = ans.strip()

    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', leading_text) if p.strip()]
    for p in paragraphs:
        if len(p) >= 50 and p.lower() not in ('link', 'link:', 'source'):
            return p
    if paragraphs:
        return paragraphs[0]

    sents = re.split(r'(?<=[.!?])\s+', ans.strip())
    if sents:
        return ' '.join(sents[:2]).strip()

    return ans.strip()

# -----------------------
# Friendly JSON renderer (appearance only)
# -----------------------
def render_structured_json_friendly(parsed_json: Any):
    """
    Present the parsed JSON in a friendly, non-technical UI without mutating content.
    """
    if parsed_json is None:
        return

    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
        with st.expander("Structured results (table)", expanded=False):
            try:
                df = pd.json_normalize(parsed_json)
                st.dataframe(df)
            except Exception:
                st.write(parsed_json)
        return

    if isinstance(parsed_json, dict):
        # top-level preferred summary
        preferred = None
        for key in ("summary", "description", "overview", "title"):
            if key in parsed_json and isinstance(parsed_json[key], str) and len(parsed_json[key].strip()) > 20:
                preferred = parsed_json[key].strip()
                break
        if preferred:
            st.markdown(f"**Summary:** {preferred}")

        for k, v in parsed_json.items():
            if preferred and isinstance(v, str) and v.strip() == preferred:
                continue
            label = str(k)
            with st.expander(label, expanded=False):
                if isinstance(v, str):
                    st.markdown(v)
                elif isinstance(v, list) and all(isinstance(i, str) for i in v):
                    for item in v:
                        st.write(f"- {item}")
                elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                    try:
                        df = pd.json_normalize(v)
                        st.dataframe(df)
                    except Exception:
                        st.write(v)
                elif isinstance(v, dict):
                    simple_pairs = {}
                    complex_items = {}
                    for subk, subv in v.items():
                        if isinstance(subv, (str, int, float, bool)) or (isinstance(subv, list) and all(isinstance(x, str) for x in subv)):
                            simple_pairs[subk] = subv
                        else:
                            complex_items[subk] = subv
                    if simple_pairs:
                        df_pairs = pd.DataFrame(list(simple_pairs.items()), columns=["Field", "Value"])
                        st.table(df_pairs)
                    for ck, cv in complex_items.items():
                        with st.expander(f"{ck} (details)"):
                            st.write(cv)
                else:
                    st.write(v)
    else:
        st.write(parsed_json)

# -----------------------
# UI: Header
# -----------------------
st.title("Tentative eGov RAG — Friendly View")
st.write("Summary shown first for non-technical users. Structured details available in expanders below.")

# -----------------------
# RAG Query Interface
# -----------------------
query = st.text_input("Enter your question:", placeholder="e.g., What is HCM?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving answer..."):
            try:
                # 1. Get docs and answer
                docs_and_meta = hybrid_retrieve_pg(query, top_k=5)
                # pass docs to generator (preserves existing behavior)
                answer = generate_rag_answer(query, lambda q, top_k=5: docs_and_meta)

                st.success("Answer:")

                # 2. Show human-friendly summary (display only)
                lead = extract_human_summary(answer)
                if lead:
                    st.markdown(f"### {lead}")
                else:
                    st.info("No summary detected. See structured output below.")

                # 3. Show embedded media found in answer (display-only, WITHOUT reprinting raw JSON)
                # We only embed URLs if the answer is NOT valid JSON (i.e., text mode)
                try:
                    # If answer is not JSON, embed any videos/images
                    try:
                        json.loads(answer)
                        # If this line succeeds, skip embedding (we’ll handle JSON later)
                        pass
                    except json.JSONDecodeError:
                        render_text_with_media(answer)
                except Exception:
                    pass

                # 4. Show structured JSON in a friendly form (appearance only)
                parsed_json = None
                try:
                    parsed_json = json.loads(answer)
                except Exception:
                    parsed_json = None

                if parsed_json is not None:
                    st.markdown("---")
                    st.markdown("## More details")
                    render_structured_json_friendly(parsed_json)

                    # Raw JSON expander and download
                    with st.expander("Show raw JSON"):
                        st.json(parsed_json)
                    json_bytes = json.dumps(parsed_json, indent=2).encode("utf-8")
                    st.download_button("Download JSON", data=json_bytes, file_name="answer.json", mime="application/json")
                else:
                    # Not JSON: offer raw view & download
                    with st.expander("Show raw answer"):
                        st.code(answer)
                    st.download_button("Download answer.txt", data=answer.encode("utf-8"), file_name="answer.txt", mime="text/plain")

                # 5. Transparency: list retrieved chunks in expanders (unchanged behavior)
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

            except Exception as e:
                st.error(f"Error: {str(e)}")
