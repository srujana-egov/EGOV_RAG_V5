import os
from dotenv import load_dotenv
import openai

load_dotenv()

# ─────────────────────────────────────────────
# Single OpenAI client — not recreated per query
# ─────────────────────────────────────────────
_client = None

def _get_client() -> openai.OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing.")
        _client = openai.OpenAI(api_key=api_key)
    return _client


# ─────────────────────────────────────────────
# Query rewriting — uses gpt-3.5-turbo (10x cheaper, same quality for this task)
# ─────────────────────────────────────────────
def rewrite_query(query: str) -> str:
    try:
        response = _get_client().chat.completions.create(
            model="gpt-3.5-turbo",   # ← was gpt-4, same quality for rewriting, 10x cheaper
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query optimizer for a DIGIT Studio documentation chatbot. "
                        "Rewrite the user's question to maximize retrieval of relevant documentation. "
                        "Expand abbreviations, add relevant synonyms, make it more specific. "
                        "Return ONLY the rewritten query, nothing else."
                    )
                },
                {"role": "user", "content": query}
            ],
            max_tokens=100,
            temperature=0.0
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"[Query Rewrite] '{query}' → '{rewritten}'")
        return rewritten
    except Exception as e:
        print(f"[Query Rewrite] Failed, using original: {e}")
        return query


# ─────────────────────────────────────────────
# Build messages list (shared between streaming and non-streaming)
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful assistant for DIGIT Studio — a low-code platform for building government digital services.

Answer questions clearly and accurately using only the provided context.

Guidelines:
- Answer in plain, friendly language.
- Use numbered lists for step-by-step instructions, bullet points for feature lists.
- If information is missing from the context, say so and suggest checking the documentation.
- Always include relevant source URLs at the end under a "📎 References" section.
- Do not mention HCM or Health Campaign Management unless the user asks.
- Keep answers concise but complete."""


def _build_messages(query: str, docs: list, history: list = None) -> list:
    context = "\n\n".join([
        f"--- {doc.get('title', 'No Title')} ---\n{doc['content']}"
        if isinstance(doc, dict) else str(doc)
        for doc in docs
    ])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"})
    return messages


# ─────────────────────────────────────────────
# Non-streaming answer (used as fallback)
# ─────────────────────────────────────────────
def chat_with_assistant(query: str, docs: list, history: list = None, model: str = "gpt-4") -> str:
    messages = _build_messages(query, docs, history)
    response = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.2
    )
    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────
# Streaming answer — yields chunks as they arrive
# ─────────────────────────────────────────────
def stream_rag_answer(query: str, docs: list, history: list = None, model: str = "gpt-4"):
    """
    Generator that yields text chunks as they stream from OpenAI.
    Use with Streamlit's st.write_stream() or manual accumulation.
    """
    messages = _build_messages(query, docs, history)
    with _get_client().chat.completions.stream(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.2
    ) as stream:
        for text in stream.text_stream:
            yield text


# ─────────────────────────────────────────────
# Full RAG pipeline (non-streaming, for internal use)
# ─────────────────────────────────────────────
def generate_rag_answer(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None
) -> str:
    rewritten = rewrite_query(query)
    docs_and_meta = hybrid_retrieve_pg(rewritten, top_k)

    if not docs_and_meta:
        return (
            "I don't have enough information in the knowledge base to answer that.\n\n"
            "📎 Please check the DIGIT Studio documentation: https://docs.digit.org/studio"
        )

    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        docs.append({"title": meta.get("title", f"Chunk {i}"), "content": doc, "url": meta.get("url", "")})

    return chat_with_assistant(query, docs, history=history, model=model)


# ─────────────────────────────────────────────
# Full RAG pipeline (streaming version for app.py)
# ─────────────────────────────────────────────
def stream_rag_pipeline(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None
):
    """
    Generator: rewrites query, retrieves docs, then streams the answer.
    Yields either str chunks or a special {"type": "no_results"} dict.
    """
    rewritten = rewrite_query(query)
    docs_and_meta = hybrid_retrieve_pg(rewritten, top_k)

    if not docs_and_meta:
        yield "I don't have enough information in the knowledge base to answer that.\n\n📎 Please check the DIGIT Studio documentation: https://docs.digit.org/studio"
        return

    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        docs.append({"title": meta.get("title", f"Chunk {i}"), "content": doc, "url": meta.get("url", "")})

    yield from stream_rag_answer(query, docs, history=history, model=model)


if __name__ == "__main__":
    from retrieval import hybrid_retrieve_pg
    q = input("Question: ")
    for chunk in stream_rag_pipeline(q, hybrid_retrieve_pg):
        print(chunk, end="", flush=True)
    print()
