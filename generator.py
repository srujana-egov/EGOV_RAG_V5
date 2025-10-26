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
    context = "\n\n".join([f"--- {doc.get('title','No Title')} ---\n{doc['content']}" if isinstance(doc, dict) else doc for doc in docs])

    # Explicit prompt to get structured JSON output
    prompt = f"""
You are a precise assistant that answers questions strictly using the provided context.

Rules:
1. Use ONLY the information from the context. Do not invent any details.
2. Return a **complete JSON array**, where each element corresponds to a field in the Campaign Setup.
3. Each JSON object must include:
   {{
     "Field Name": "",
     "Description": "",
     "Mandatory": true/false,
     "Input": "User/System",
     "Data Type": "",
     "Min Length": int or null,
     "Max Length": int or null,
     "Minimum Value": number or null,
     "Maximum Value": number or null,
     "Need Data from Program or State": true/false
   }}
4. Merge complementary information from multiple chunks.
5. Include URLs at the end as a separate JSON field: "Reference URLs": []
6. If a field is partially described, include what is available and indicate missing info clearly.

Context:
{context}

Question: {query}

Answer with the full JSON array:
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
        # Pass structured dict to preserve title
        docs.append({"title": meta.get('title', f"Chunk {i}"), "content": doc})

    answer = chat_with_assistant(query, docs, model=model)
    
    # Optional: validate JSON
    try:
        parsed = json.loads(answer)
        return json.dumps(parsed, indent=2)
    except json.JSONDecodeError:
        # If the model output is not valid JSON, return raw
        return answer


if __name__ == "__main__":
    from .retrieval import hybrid_retrieve_pg
    q = input("Enter user query: ")
    answer = generate_rag_answer(q, hybrid_retrieve_pg, model="gpt-4")
    print("\nRAG Answer:\n", answer)
