import streamlit as st
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
                # Generate RAG answer
                answer = generate_rag_answer(query, hybrid_retrieve_pg)
                st.success("Answer:")
                st.markdown(answer)  # use markdown for better formatting

                # Transparency: show retrieved chunks
                st.subheader("Retrieved Chunks Used")
                docs_and_meta = hybrid_retrieve_pg(query, top_k=5)

                for i, item in enumerate(docs_and_meta, start=1):
                    # Unpack safely
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, meta = item
                    else:
                        doc, meta = item, {}

                    score = meta.get("score", None)
                    source = meta.get("source", None)

                    expander_label = f"Chunk {i}"
                    if score is not None:
                        expander_label += f" (Score: {score:.4f})"

                    with st.expander(expander_label):
                        st.markdown(doc)
                        if source:
                            st.markdown(f"[Source]({source})")
            except Exception as e:
                st.error(f"Error: {str(e)}")
