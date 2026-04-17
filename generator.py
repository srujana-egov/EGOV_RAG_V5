import os
from dotenv import load_dotenv
import openai

load_dotenv()

# ─────────────────────────────────────────────
# Single OpenAI client
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
# Domain detection threshold
# ─────────────────────────────────────────────
OUT_OF_DOMAIN_THRESHOLD = 0.35  # cosine similarity below this = out of DIGIT Studio domain

OUT_OF_DOMAIN_MSG = (
    "I'm sorry, that question appears to be outside the scope of DIGIT Studio documentation.\n\n"
    "I can only answer questions related to **DIGIT Studio** — eGovernments Foundation's platform "
    "for building digital public services.\n\n"
    "**Topics I can help with:**\n"
    "- Building services, forms, workflows, and checklists\n"
    "- Roles, permissions, and user management\n"
    "- Notifications (SMS/email), documents, and integrations\n"
    "- Deployment, environments, and configuration\n"
    "- Citizen and employee app behaviour\n\n"
    "Please ask a question related to DIGIT Studio."
)


# ─────────────────────────────────────────────
# Query rewriting (gpt-3.5-turbo — cheaper, fine for this task)
# ─────────────────────────────────────────────
def rewrite_query(query: str) -> str:
    try:
        response = _get_client().chat.completions.create(
            model="gpt-3.5-turbo",
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
# System prompt
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
# Non-streaming answer (fallback)
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
# Streaming answer
# ─────────────────────────────────────────────
def stream_rag_answer(query: str, docs: list, history: list = None, model: str = "gpt-4"):
    messages = _build_messages(query, docs, history)

    stream = _get_client().chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content


# ─────────────────────────────────────────────
# Full RAG pipeline (non-streaming)
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
        return OUT_OF_DOMAIN_MSG

    max_score = max(meta.get("score", 0) for _, meta in docs_and_meta)
    if max_score < OUT_OF_DOMAIN_THRESHOLD:
        return OUT_OF_DOMAIN_MSG

    docs = []
    for i, (doc, meta) in enumerate(docs_and_meta, start=1):
        docs.append({"title": meta.get("title", f"Chunk {i}"), "content": doc, "url": meta.get("url", "")})

    return chat_with_assistant(query, docs, history=history, model=model)


# ─────────────────────────────────────────────
# Full RAG pipeline (streaming, used by app.py)
# ─────────────────────────────────────────────
def stream_rag_pipeline(
    query: str,
    hybrid_retrieve_pg,
    top_k: int = 5,
    model: str = "gpt-4",
    history: list = None
):
    """
    Generator that rewrites query, retrieves docs, checks domain, then streams the answer.
    Yields str chunks. If out-of-domain, yields the OUT_OF_DOMAIN_MSG as a single chunk.
    """
    rewritten = rewrite_query(query)
    docs_and_meta = hybrid_retrieve_pg(rewritten, top_k)

    # No results at all → out of domain
    if not docs_and_meta:
        yield OUT_OF_DOMAIN_MSG
        return

    # Check best similarity score
    max_score = max(meta.get("score", 0) for _, meta in docs_and_meta)
    print(f"[Domain] Max retrieval score: {max_score:.3f} (threshold={OUT_OF_DOMAIN_THRESHOLD})")

    if max_score < OUT_OF_DOMAIN_THRESHOLD:
        yield OUT_OF_DOMAIN_MSG
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
