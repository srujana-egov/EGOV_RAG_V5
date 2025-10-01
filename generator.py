import os
from dotenv import load_dotenv
load_dotenv()
import openai

def chat_with_assistant(query, docs, model="gpt-3.5-turbo"):
    """
    Calls OpenAI (or compatible) chat completion model with query and supporting docs.
    Returns the generated answer as a string.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")
    client = openai.OpenAI(api_key=api_key)
    context = "\n\n".join(docs)
    prompt = f"""You are a precise assistant that answers questions strictly from the provided context.

You are a precise assistant that answers questions using the provided context.

Rules:
1. Use ONLY information from the context as the primary source of truth. Do not invent details.
2. If the context partially addresses the query, provide the available information and clearly state what is missing.
3. If multiple code snippets are in the context, return them exactly as-is, combined in the correct order. Keep imports, setup, and comments intact.
4. For procedures, return exact step-by-step instructions in the order they appear.
5. If multiple chunks provide complementary information, merge them into a single coherent answer.
6. If different chunks provide conflicting information, explain the conflict and cite which chunk(s) each version came from.
7. If the context contains URLs, include them at the end of the answer for reference.
8. If the context includes titles or sections, prefix your answer with them.
9. Only if there is absolutely no relevant information in the context, say: "I don't have enough information in the knowledge base to answer that."

Context:
{context}

Question: {query}
Answer:"""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=3000,  # allow longer answers for recall coverage
        temperature=0.3#removed for model compatibility
    )
    return response.choices[0].message.content.strip()

def generate_rag_answer(query, hybrid_retrieve_pg, top_k=5, model="gpt-3.5-turbo"):
    docs_and_meta = hybrid_retrieve_pg(query, top_k)
    if not docs_and_meta:
        return "I don't have enough information in the knowledge base to answer that. Please check our documentation for more details: https://docs.digit.org/health."

    print("\n[Retrieved Chunks Used:]\n")
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {meta.get('id')}")
        print(f"Score: {meta.get('score')}")
        print(f"Snippet: {(doc or '').strip()[:300]}...\n")

    docs = [doc for doc, meta in docs_and_meta]
    return chat_with_assistant(query, docs, model=model)

if __name__ == "__main__":
    from .retrieval import hybrid_retrieve_pg
    q = input("Enter user query: ")
    answer = generate_rag_answer(q, hybrid_retrieve_pg, model="gpt-3.5-turbo")
    print("\nRAG Answer:\n", answer)
