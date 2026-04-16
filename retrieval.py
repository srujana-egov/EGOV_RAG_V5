from typing import List, Tuple, Dict
from utils import get_conn, get_env_var
from generator import rewrite_query

import openai

client = openai.OpenAI(api_key=get_env_var("OPENAI_API_KEY"))

TABLE = get_env_var("DB_TABLE", "studio_manual")


def get_embedding(text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return resp.data[0].embedding


def hybrid_retrieve_pg(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    conn = get_conn()

    try:
        with conn.cursor() as cur:
            # 🔥 query rewrite (you already built it — use it!)
            rewritten = rewrite_query(query)

            query_embedding = get_embedding(rewritten)

            cur.execute(f"""
                SELECT id, document, url,
                       (1 - (embedding <=> %s::vector)) AS score
                FROM {TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, top_k))

            rows = cur.fetchall()

            results = []
            for row in rows:
                results.append((
                    row[1],
                    {"url": row[2], "score": row[3]}
                ))

            return results

    finally:
        conn.close()
