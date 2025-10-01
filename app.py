import streamlit as st
from generator import generate_rag_answer
from retrieval import hybrid_retrieve_pg

st.set_page_config(page_title="RAG-TDD Demo", layout="wide")

# -----------------------
# UI
# -----------------------
st.title("📖 RAG-TDD Demo (DIGIT HCM Knowledge Base)")

query = st.text_input("Enter your question:", placeholder="e.g., How to pay with HCM?")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                answer = generate_rag_answer(query, hybrid_retrieve_pg)
                st.success("Answer:")
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
