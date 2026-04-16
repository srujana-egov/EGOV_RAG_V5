"""
Ingestion Console
=================
A standalone Streamlit app for uploading documents (PDF, CSV, Excel)
and converting them into the JSONL format required by the bot.

Run with:
    streamlit run ingest_console.py
"""

import os
import json
import hashlib
import streamlit as st
import pandas as pd
from io import BytesIO

st.set_page_config(page_title="DIGIT Studio — Ingestion Console", page_icon="📥", layout="wide")

st.title("📥 Ingestion Console")
st.caption("Upload PDF, CSV, or Excel files. Preview and store them into the knowledge base.")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def make_id(source: str, index: int) -> str:
    base = f"{source}_{index}"
    return hashlib.md5(base.encode()).hexdigest()[:12]


def parse_pdf(file_bytes: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        return f"[PDF parse error: {e}]"


def parse_csv(file_bytes: bytes) -> str:
    df = pd.read_csv(BytesIO(file_bytes))
    return df.to_string(index=False)


def parse_excel(file_bytes: bytes) -> str:
    xl = pd.ExcelFile(BytesIO(file_bytes))
    parts = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        parts.append(f"Sheet: {sheet}\n{df.to_string(index=False)}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────
# Upload
# ─────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "csv", "xlsx", "xls"],
    accept_multiple_files=True,
    help="PDF, CSV, or Excel files. Each file will be chunked and embedded."
)

source_tag = st.text_input("Tag these documents", placeholder="e.g. studio, forms, onboarding")
source_url = st.text_input("Base URL (optional)", placeholder="https://docs.digit.org/studio")
chunk_size = st.slider("Words per chunk", 200, 1500, 800, 50)
overlap = st.slider("Overlap words", 0, 200, 100, 10)

if uploaded_files:
    st.divider()
    all_chunks = []

    for file in uploaded_files:
        st.subheader(f"📄 {file.name}")
        raw_bytes = file.read()

        if file.name.endswith(".pdf"):
            text = parse_pdf(raw_bytes)
        elif file.name.endswith(".csv"):
            text = parse_csv(raw_bytes)
        elif file.name.endswith((".xlsx", ".xls")):
            text = parse_excel(raw_bytes)
        else:
            text = raw_bytes.decode("utf-8", errors="ignore")

        if not text.strip():
            st.warning(f"No text extracted from {file.name}")
            continue

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        st.caption(f"Extracted {len(text)} characters → {len(chunks)} chunks")

        with st.expander("Preview first chunk"):
            st.text(chunks[0][:600] + "..." if len(chunks[0]) > 600 else chunks[0])

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "id": f"{file.name}/{make_id(file.name, i)}",
                "title": file.name,
                "document": chunk,
                "url": source_url or "",
                "tag": source_tag or "",
                "version": "uploaded"
            })

    st.divider()
    st.success(f"Total: **{len(all_chunks)} chunks** ready from {len(uploaded_files)} file(s)")

    col1, col2 = st.columns(2)

    # ── Option A: Download JSONL
    with col1:
        st.markdown("**Option A — Download JSONL**")
        st.caption("Download and run `ingest_simple.py --file your_file.jsonl`")
        jsonl_str = "\n".join(json.dumps(c, ensure_ascii=False) for c in all_chunks)
        st.download_button(
            "⬇️ Download JSONL",
            data=jsonl_str.encode("utf-8"),
            file_name="studio_chunks.jsonl",
            mime="application/jsonl"
        )

    # ── Option B: Ingest directly to DB
    with col2:
        st.markdown("**Option B — Ingest directly to DB**")
        st.caption("Embeds and stores all chunks into the vector database now")
        if st.button("🚀 Ingest to database", type="primary"):
            try:
                from utils import insert_chunk
                from retrieval import get_embedding

                progress = st.progress(0, text="Starting...")
                errors = 0

                for i, chunk in enumerate(all_chunks):
                    try:
                        insert_chunk(
                            doc_id=chunk["id"],
                            text=chunk["document"],
                            metadata={"url": chunk["url"], "tag": chunk["tag"], "version": chunk["version"]},
                            get_embedding=get_embedding
                        )
                    except Exception as e:
                        errors += 1
                        st.warning(f"Chunk {i} failed: {e}")

                    progress.progress((i + 1) / len(all_chunks), text=f"Ingesting {i + 1}/{len(all_chunks)}...")

                progress.empty()
                if errors == 0:
                    st.success(f"✅ All {len(all_chunks)} chunks ingested successfully!")
                else:
                    st.warning(f"Done with {errors} errors. {len(all_chunks) - errors} chunks ingested.")

            except ImportError as e:
                st.error(f"Missing dependency: {e}. Make sure .env is configured.")
            except Exception as e:
                st.error(f"Ingestion failed: {e}")
