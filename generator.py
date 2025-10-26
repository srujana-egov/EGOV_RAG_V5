import os
from dotenv import load_dotenv
load_dotenv()
import openai
import json

def chat_with_assistant(query, docs, model="gpt-4"):
    """
    Calls OpenAI chat completion model with query and supporting docs.
    Returns a structured JSON answer as a string.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")
    
    client = openai.OpenAI(api_key=api_key)

    # Preserve chunk titles/IDs to help model understand context
    context = "\n\n".join([
        f"--- {doc.get('title','No Title')} ---\n{doc['content']}" if isinstance(doc, dict) else doc 
        for doc in docs
    ])

    prompt = f"""
You are a precise assistant that answers questions using only the provided context.

Instructions:
1. Use only information available in the context.
2. Campaign Setup is different from Campaign Type Setup.
3. Merge complementary information from multiple chunks if needed.
4. Include URLs at the end as a separate JSON field: "Reference URLs": []
5. If information is missing, indicate it clearly using null or an appropriate placeholder.
6. HCM stands for Health Campaign Management.
7. **Important:** 
   - When returning JSON, do NOT include any text outside the JSON array.
   - All boolean values must be lowercase (`true`/`false`), missing values must be `null`, and all strings must use double quotes.
8. If the question does **not** require JSON, answer in **plain text** based on the context.

Context:
{context}

Question: {query}

Answer accordingly. If possible, format the answer as a JSON array, otherwise return plain text.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=4000,
        temperature=0.2
    )

    return response.choices[0].message.content.strip()


def generate_rag_answer(query, hybrid_retrieve_pg, top_k=5, model="gpt-4"):
    docs_and_meta = hybrid_retrieve_pg(query, top_k)
    if not docs_and_meta:
        return "I don't have enough information in the knowledge base to answer that. Please check our documentation for more details: https://docs.digit.org/health."

    print("\n[Retrieved Chunks Used:]\n")
    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {meta.get('id')}")
        print(f"Score: {meta.get('score')}")
        print(f"Snippet: {(doc or '').strip()[:300]}...\n")
        docs.append({"title": meta.get('title', f"Chunk {i}"), "content": doc})

    answer = chat_with_assistant(query, docs, model=model)

    # Try to parse as JSON and pretty-print
    try:
        parsed = json.loads(answer)
        # Always return pretty JSON if possible
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        # If not JSON, return as-is (plain text)
        return answer


if __name__ == "__main__":
    from .retrieval import hybrid_retrieve_pg
    q = input("Enter user query: ")
    answer = generate_rag_answer(q, hybrid_retrieve_pg, model="gpt-4")
    print("\nRAG Answer:\n", answer)
