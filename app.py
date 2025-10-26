import streamlit as st
import json
import pandas as pd
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")

# -----------------------
# UI
# -----------------------
st.title("Tentative eGov RAG - As of Oct 6th, 2025")
st.subheader(
    "Important note: Work under progress, only answers questions until ACCESS subsection of HCM gitbook. "
    "Updated information on console and dashboard to be added."
)

query = st.text_input("Enter your question:", placeholder="e.g., How to pay with HCM?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                # Generate answer
                answer = generate_rag_answer(query, hybrid_retrieve_pg)
                
                st.success("Answer:")

                # Try to parse as JSON for nice formatting
                try:
                    parsed_json = json.loads(answer)

                    # If it's a list of dicts, show as table
                    if isinstance(parsed_json, list) and all(isinstance(i, dict) for i in parsed_json):
                        df = pd.json_normalize(parsed_json)
                        st.dataframe(df)
                    else:
                        st.json(parsed_json)  # Pretty JSON viewer
                except json.JSONDecodeError:
                    # Not JSON? Show as plain text
                    st.write(answer)

                # Transparency: show retrieved chunks
                st.subheader("Retrieved Chunks Used")
                docs_and_meta = hybrid_retrieve_pg(query, top_k=5)
                for i, (doc, meta) in enumerate(docs_and_meta, start=1):
                    with st.expander(f"Chunk {i} (Score: {meta.get('score'):.4f})"):
                        st.write(doc)
                        if meta.get("source"):
                            st.markdown(f"[Source]({meta['source']})")

            except Exception as e:
                st.error(f"Error: {str(e)}")
