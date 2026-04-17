from typing import List, Tuple, Dict
from utils import get_conn, get_env_var
import openai

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    return _client

TABLE = get_env_var("DB_TABLE", "studio_manual")


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
def get_embedding(text: str) -> List[float]:
    resp = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


# ─────────────────────────────────────────────
# Retrieval
# ─────────────────────────────────────────────
def hybrid_retrieve_pg(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    conn = get_conn()

    try:
        with conn.cursor() as cur:
            query_embedding = get_embedding(query)

            cur.execute(f"""
                SELECT document, url,
                       (1 - (embedding <=> %s::vector)) AS score
                FROM {TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))

            rows = cur.fetchall()

            return [
                (row[0], {"url": row[1], "score": row[2]})
                for row in rows
            ]

    finally:
        conn.close()
