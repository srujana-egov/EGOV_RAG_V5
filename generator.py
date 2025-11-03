import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import json

def chat_with_assistant(query, docs, model="gpt-4o-mini"):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing in environment variables.")

    client = OpenAI(api_key=api_key)

    context = "\n\n".join([
        f"--- {doc.get('title','No Title')} ---\n{doc['content']}" if isinstance(doc, dict) else doc
        for doc in docs
    ])

    system_msg = (
        "You are a precise assistant that answers using only the provided context. "
        "Campaign Setup is different from Campaign Type Setup. Merge complementary info. "
        'Include URLs in a field named "Reference URLs". '
        "If information is missing, say so. HCM = Health Campaign Management. "
        "When returning JSON, output a JSON array only; booleans true/false; missing -> null; double quotes."
    )

    user_msg = f"Context:\n{context}\n\nQuestion: {query}\n\n" \
               "Answer accordingly. If possible, format the answer as a JSON array; otherwise return plain text."

    try:
        resp = client.chat.completions.create(
            model=model,                       # "gpt-4o" also fine
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
            max_tokens=4000                   # safer cap for RAG
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        # Make failures visible to your app instead of failing silently
        return f"(Model error) {e}"

def generate_rag_answer(query, hybrid_retrieve_pg, top_k=5, model="gpt-4o-mini"):
    docs_and_meta = hybrid_retrieve_pg(query, top_k)
    if not docs_and_meta:
        return ("I don't have enough information in the knowledge base to answer that. "
                "Please check our documentation: https://docs.digit.org/health.")

    print("\n[Retrieved Chunks Used:]\n")
    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        print(f"--- Chunk {i} ---")
        print(f"ID: {meta.get('id')}")
        print(f"Score: {meta.get('score')}")
        print(f"Snippet: {(doc or '').strip()[:300]}...\n")
        docs.append({"title": meta.get('title', f"Chunk {i}"), "content": doc})

    answer = chat_with_assistant(query, docs, model=model)

    try:
        parsed = json.loads(answer)
        return json.dumps(parsed, indent=2, ensure_ascii=False)
    except json.JSONDecodeError:
        return answer
