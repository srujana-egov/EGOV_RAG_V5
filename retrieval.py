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


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts in a single API call (preserves order)."""
    resp = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    items = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in items]


# ─────────────────────────────────────────────
# Ensure FTS index exists (called once on startup)
# ─────────────────────────────────────────────
def ensure_fts_index():
    """Add tsvector column + GIN index to studio_manual if not present."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Add fts column if missing
            cur.execute(f"""
                ALTER TABLE {TABLE}
                ADD COLUMN IF NOT EXISTS fts tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
            """)
            # Add GIN index if missing
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {TABLE}_fts_idx
                ON {TABLE} USING GIN(fts);
            """)
        conn.commit()
        print("[FTS] tsvector column + GIN index ensured.")
    except Exception as e:
        conn.rollback()
        print(f"[FTS] Could not create FTS index (non-fatal): {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Reciprocal Rank Fusion
# ─────────────────────────────────────────────
def _rrf(vector_rows: list, bm25_rows: list, k: int = 60) -> List[Tuple[str, float]]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    Returns list of (document, rrf_score) sorted descending.
    """
    scores: Dict[str, float] = {}
    doc_text: Dict[str, str] = {}

    for rank, (doc, _) in enumerate(vector_rows):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        doc_text[doc] = doc

    for rank, (doc, _) in enumerate(bm25_rows):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (k + rank + 1)
        doc_text[doc] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_text[doc], score) for doc, score in ranked]


# ─────────────────────────────────────────────
# Hybrid retrieval: vector + BM25 → RRF
# ─────────────────────────────────────────────
def hybrid_retrieve_pg(query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
    """
    1. Vector search (top 2*k candidates via pgvector cosine)
    2. BM25 full-text search (top 2*k candidates via PostgreSQL tsvector)
    3. Fuse with Reciprocal Rank Fusion → return top_k
    """
    conn = get_conn()

    try:
        with conn.cursor() as cur:
            query_embedding = get_embedding(query)
            fetch = top_k * 2  # retrieve more candidates for fusion

            # ── Vector search ──
            cur.execute(f"""
                SELECT id, document,
                       (1 - (embedding <=> %s::vector)) AS score
                FROM {TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, fetch))
            vector_rows = [(row[1], {"id": row[0], "score": row[2]}) for row in cur.fetchall()]

            # ── BM25 full-text search ──
            cur.execute(f"""
                SELECT id, document,
                       ts_rank_cd(fts, plainto_tsquery('english', %s)) AS score
                FROM {TABLE}
                WHERE fts @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, fetch))
            bm25_rows = [(row[1], {"id": row[0], "score": row[2]}) for row in cur.fetchall()]

            print(f"[Retrieval] Vector: {len(vector_rows)} | BM25: {len(bm25_rows)}")

    finally:
        conn.close()

    if not vector_rows and not bm25_rows:
        return []

    # ── Fuse with RRF ──
    fused = _rrf(vector_rows, bm25_rows)

    # Preserve original cosine similarity and chunk id in metadata so domain detection
    # in generator.py can use a stable 0-1 score (RRF scores are ~0.01-0.03).
    vector_meta = {doc: meta for doc, meta in vector_rows}
    bm25_meta = {doc: meta for doc, meta in bm25_rows}
    return [
        (doc, {
            "score": rrf_score,
            "vector_score": vector_meta.get(doc, bm25_meta.get(doc, {})).get("score", 0.0)
                            if doc in vector_meta else 0.0,
            "id": (vector_meta.get(doc) or bm25_meta.get(doc, {})).get("id", ""),
        })
        for doc, rrf_score in fused[:top_k]
    ]
