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

# --- Top Logo ---
st.image("eGov-logo.png", width=200)

st.set_page_config(page_title="miGo", layout="wide")

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
    """
    Robust summary extractor that also turns list-of-dicts JSON into a natural-language sentence.
    Display-only: does not modify original answer.
    """
    if not answer:
        return ""

    ans = answer.replace('\r\n', '\n').replace('\r', '\n').strip()

    url_regex = re.compile(r'https?://[^\s)>\]"]+', flags=re.IGNORECASE)
    def looks_like_url(s: str) -> bool:
        return bool(url_regex.search(s))

    def gather_texts(obj):
        texts = []
        if isinstance(obj, dict):
            for v in obj.values():
                texts.extend(gather_texts(v))
        elif isinstance(obj, list):
            for item in obj:
                texts.extend(gather_texts(item))
        elif isinstance(obj, str):
            s = obj.strip()
            if len(s) > 20 and not looks_like_url(s):
                texts.append(s)
        return texts

    # try parse JSON
    parsed = None
    try:
        parsed = json.loads(ans)
    except Exception:
        # try to locate first JSON substring
        m = re.search(r'(\{|\[)', ans)
        if m:
            candidate = ans[m.start():]
            for end in range(len(candidate), 0, -1):
                try:
                    parsed = json.loads(candidate[:end])
                    break
                except Exception:
                    continue

    # If parsed is a list containing a single dict (common), simplify it
    if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
        obj = parsed[0]
        # if keys look like "HCM v1.8 Features" or "HCM v1.8", build a readable summary
        parts = []
        for key, val in obj.items():
            # normalize key text
            ktext = str(key).strip()
            # if the value is a list of feature strings or dicts with 'feature' keys
            features = []
            if isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        features.append(item.strip())
                    elif isinstance(item, dict):
                        # try common subkeys
                        if 'feature' in item and isinstance(item['feature'], str):
                            features.append(item['feature'].strip())
                        else:
                            # gather any string values inside dict
                            for sv in item.values():
                                if isinstance(sv, str):
                                    features.append(sv.strip())
                # remove empties and dedupe small set
                features = [f for f in features if f]
            # If features found, render short clause
            if features:
                if len(features) <= 6:
                    feat_txt = ", ".join(features[:-1]) + (", and " + features[-1] if len(features) > 1 else features[0])
                else:
                    feat_txt = ", ".join(features[:6]) + ", etc."
                parts.append(f"{ktext}: {feat_txt}")
            else:
                # fallback: if val contains readable strings deeper, pick them
                candidates = gather_texts(val)
                if candidates:
                    parts.append(f"{ktext}: {candidates[0]}")
        if parts:
            # join clauses into one summary sentence
            summary = "; ".join(parts)
            # short friendly prefix
            if any("feature" in k.lower() or "features" in k.lower() for k in obj.keys()):
                return "This release includes: " + summary
            return summary

    # If parsed is a dict, prefer explicit human fields
    if isinstance(parsed, dict):
        for key in ("summary", "description", "overview", "title", "HCM", "Platform Name"):
            if key in parsed and isinstance(parsed[key], str) and len(parsed[key].strip()) > 20:
                s = parsed[key].strip()
                if not looks_like_url(s):
                    return s
        # otherwise gather strings and return the longest
        candidates = gather_texts(parsed)
        if candidates:
            return max(candidates, key=len).strip()

    # No useful JSON -> fallback to raw-text heuristics (take first non-JSON paragraph)
    clean = re.sub(r'^\s*(Answer:|Link:|Link)\s*', '', ans, flags=re.IGNORECASE).strip()
    # cut at start of JSON if present
    json_pos = min([p for p in [clean.find('{') if '{' in clean else -1, clean.find('[') if '[' in clean else -1] if p != -1] + [len(clean)])
    raw_before_json = clean[:json_pos].strip() if json_pos > 0 else clean
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_before_json) if p.strip()]
    for p in paragraphs:
        if len(p) >= 40 and not p.strip().startswith('{'):
            return p
    sents = re.split(r'(?<=[.!?])\s+', clean)
    if sents:
        return ' '.join(sents[:2]).strip()
    return (clean[:1000] + '...') if len(clean) > 1000 else clean

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
st.title("miGo")
st.write("Your assistant to explore all things eGov!")

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
                st.write(lead)
            else:
                st.info("No summary detected. See structured output below.")

            # 3. Media handling:
            #  - If answer is not JSON -> embed inline media from the raw answer text (renders images/videos inline)
            #  - If answer is JSON -> do NOT reprint the JSON; instead extract any URLs from the JSON and embed those media items
            parsed_json = None
            try:
                parsed_json = json.loads(answer)
            except Exception:
                parsed_json = None

            if parsed_json is None:
                # answer is plain text (not JSON) -> show media and text in original order
                try:
                    render_text_with_media(answer)
                except Exception:
                    # fail silently; do not print raw JSON
                    pass
            else:
                # answer is structured JSON -> do NOT print the JSON raw here
                # but embed any images/videos referenced inside the JSON (display-only)
                try:
                    json_urls = extract_urls_from_json(parsed_json)
                    if json_urls:
                        for u in json_urls:
                            _embed_url(u)
                except Exception:
                    # swallow errors (we do not want to alter content)
                    pass

            # 4. Show structured JSON in friendly form (appearance only)
            if parsed_json is not None:
                st.markdown("---")
                st.markdown("## More details")

                # Friendly, non-technical JSON rendering
                render_structured_json_friendly(parsed_json)

                # Allow raw JSON view and download (opt-in)
                if st.checkbox("Show technical JSON details", value=False):
                    st.json(parsed_json)

                # Download JSON file
                json_bytes = json.dumps(parsed_json, indent=2).encode("utf-8")
                st.download_button("Download JSON", data=json_bytes, file_name="answer.json", mime="application/json")
            else:
                # If not JSON, still offer raw text view + download
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
