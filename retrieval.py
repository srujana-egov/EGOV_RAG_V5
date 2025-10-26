import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# Load chunks from your JSON file (replace path as needed)
with open("chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

documents = []

# ✅ Build Documents with TITLE + URL to improve retrieval accuracy
for chunk in data:
    title = chunk.get("title", "")
    url = chunk.get("url", "")
    content = chunk.get("document", "")

    # Build embedding text with extra metadata context
    embedding_text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{content}"

    documents.append(
        Document(
            page_content=embedding_text,
            metadata={
                "id": chunk.get("id"),
                "title": title,
                "url": url
            },
        )
    )

# Create vectorstore
vectorstore = FAISS.from_documents(documents, embedding_model)

# ✅ Retrieval function
def retrieve(query, top_k=5):
    """Retrieve top K most relevant chunks for a query."""
    results = vectorstore.similarity_search_with_score(query, k=top_k)
    print("\n🔍 Retrieved Chunks Used")
    for i, (doc, score) in enumerate(results, start=1):
        print(f"\nChunk {i} (Score: {score:.4f})\n")
        print(doc.page_content[:1500])  # print partial content for readability
    return results

if __name__ == "__main__":
    # Example query test
    query = "campaign setup specifications"
    retrieve(query)
