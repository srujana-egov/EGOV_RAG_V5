import os
from dotenv import load_dotenv
import openai
import json

load_dotenv()

# ─────────────────────────────────────────────
# Query rewriting — makes retrieval much better
# ─────────────────────────────────────────────
def rewrite_query(query: str, client: openai.OpenAI) -> str:
    """
    Rewrites the user query to be more search-friendly before retrieval.
    Falls back to original query if rewriting fails.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": (
                    "You are a search query optimizer for a DIGIT Studio documentation chatbot. "
                    "Rewrite the user's question to maximize retrieval of relevant documentation. "
                    "Expand abbreviations, add relevant synonyms, and make it more specific. "
                    "Return ONLY the rewritten query, nothing else."
                )
            }, {
                "role": "user",
                "content": query
            }],
            max_tokens=150,
            temperature=0.0
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"[Query Rewrite] Original: '{query}' → Rewritten: '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewrite] Failed, using original: {e}")
        return query


# ─────────────────────────────────────────────
# Main answer generator
# ─────────────────────────────────────────────
def chat_with_assistant(query: str, docs: list, history: list = None, model: str = "gpt-4") -> str:
    """
    Calls OpenAI with query, supporting docs, and optional conversation history.
    Returns a natural language answer — no forced JSON.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")

    client = openai.OpenAI(api_key=api_key)

    # Build context from retrieved docs
    context = "\n\n".join([
        f"--- {doc.get('title', 'No Title')} ---\n{doc['content']}"
        if isinstance(doc, dict) else str(doc)
        for doc in docs
    ])

    system_prompt = """You are a helpful assistant for DIGIT Studio — a low-code platform for building government digital services.

Your job is to answer questions clearly and accurately using only the provided context.

Guidelines:
- Answer in plain, friendly language. Do NOT force JSON output.
- Use bullet points or numbered lists only when the answer genuinely benefits from structure (e.g. step-by-step instructions, feature lists).
- If the answer involves steps, format them clearly as a numbered list.
- If information is missing from the context, say so clearly and suggest the user check the documentation.
- Always include relevant source URLs at the end of your answer under a "📎 References" section.
- DIGIT Studio is a low-code platform — keep that context in mind when answering.
- Do not mention "HCM" or "Health Campaign Management" unless the user specifically asks about it.
- Keep answers concise but complete. Avoid padding."""

    # Build messages including conversation history
    messages = [{"role": "system", "content": system_prompt}]

    if history:
        for turn in history[-6:]:  # last 3 exchanges to keep context window manageable
            messages.append(turn)

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {query}"
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Full RAG pipeline
# ─────────────────────────────────────────────
def generate_rag_answer(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None
) -> str:
    """
    Full RAG pipeline:
    1. Rewrite query for better retrieval
    2. Retrieve relevant docs
    3. Generate a natural language answer
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing.")
    client = openai.OpenAI(api_key=api_key)

    # Step 1: Rewrite query
    rewritten_query = rewrite_query(query, client)

    # Step 2: Retrieve docs using rewritten query
    docs_and_meta = hybrid_retrieve_pg(rewritten_query, top_k)

    if not docs_and_meta:
        return (
            "I don't have enough information in the knowledge base to answer that.\n\n"
            "📎 Please check the DIGIT Studio documentation: https://docs.digit.org/studio"
        )

    print("\n[Retrieved Chunks Used:]\n")
    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {meta.get('id')}")
        print(f"Score: {meta.get('score')}")
        print(f"Snippet: {(doc or '').strip()[:300]}...\n")
        docs.append({
            "title": meta.get("title", f"Chunk {i}"),
            "content": doc,
            "url": meta.get("url", "")
        })

    # Step 3: Generate answer
    answer = chat_with_assistant(query, docs, history=history, model=model)
    return answer


if __name__ == "__main__":
    from retrieval import hybrid_retrieve_pg
    q = input("Enter your question: ")
    answer = generate_rag_answer(q, hybrid_retrieve_pg, model="gpt-4")
    print("\nAnswer:\n")
    print(answer)
