import logging
from typing import List, Tuple, Dict
from utils import get_conn, get_env_var
import openai

logger = logging.getLogger(__name__)
from tenacity import (
    retry,
    wait_exponential,
    stop_after_attempt,
    retry_if_exception_type,
)

_client = None

def _get_client():
    global _client
    if _client is None:
        _client = openai.OpenAI(api_key=get_env_var("OPENAI_API_KEY"))
    return _client


TABLE = get_env_var("DB_TABLE", "studio_manual")

# Shared retry decorator for embedding API calls
_embed_retry = retry(
    wait=wait_exponential(multiplier=1, min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((
        openai.RateLimitError,
        openai.APIStatusError,
        openai.APIConnectionError,
    )),
    reraise=True,
)


# ─────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────
@_embed_retry
def get_embedding(text: str) -> List[float]:
    resp = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=text,
        timeout=20
    )
    return resp.data[0].embedding


@_embed_retry
def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Embed multiple texts in a single API call (preserves order)."""
    resp = _get_client().embeddings.create(
        model="text-embedding-3-small",
        input=texts,
        timeout=20
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
            cur.execute(f"""
                CREATE INDEX IF NOT EXISTS {TABLE}_hnsw_idx
                ON {TABLE} USING hnsw (embedding vector_cosine_ops);
            """)
        conn.commit()
        logger.info("FTS: tsvector column + GIN index ensured.")
    except Exception as e:
        conn.rollback()
        logger.warning("FTS: Could not create FTS index (non-fatal): %s", e)
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
                SELECT id, document, section,
                       (1 - (embedding <=> %s::vector)) AS score
                FROM {TABLE}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, fetch))
            vector_rows = [(row[1], {"id": row[0], "section": row[2], "score": row[3]}) for row in cur.fetchall()]

            # ── BM25 full-text search ──
            cur.execute(f"""
                SELECT id, document, section,
                       ts_rank_cd(fts, websearch_to_tsquery('english', %s)) AS score
                FROM {TABLE}
                WHERE fts @@ websearch_to_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
            """, (query, query, fetch))
            bm25_rows = [(row[1], {"id": row[0], "section": row[2], "score": row[3]}) for row in cur.fetchall()]

            logger.info("Retrieval: Vector: %d | BM25: %d", len(vector_rows), len(bm25_rows))

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
            "vector_score": vector_meta[doc]["score"] if doc in vector_meta else 0.0,
            "id": (vector_meta.get(doc) or bm25_meta.get(doc, {})).get("id", ""),
            "section": (vector_meta.get(doc) or bm25_meta.get(doc, {})).get("section", ""),
        })
        for doc, rrf_score in fused[:top_k]
    ]
