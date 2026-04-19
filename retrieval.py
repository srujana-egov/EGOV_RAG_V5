import logging
import re as _re
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
import re as _re
if not _re.match(r'^[A-Za-z0-9_]+$', str(TABLE)):
    raise ValueError(f"DB_TABLE env var contains invalid characters: {TABLE!r}")

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
    with get_conn() as conn:
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
# Section hint detection — lightweight keyword → section mapper
# ─────────────────────────────────────────────

_SECTION_KEYWORDS: list = [
    # (keywords_tuple, section_substring)
    (("notification", "sms", "email alert", "notify"), "notification"),
    (("workflow", "process", "state", "transition", "action"), "workflow"),
    (("role", "permission", "access", "privilege", "user management"), "role"),
    (("field", "form", "input", "dropdown", "text box", "checkbox"), "field"),
    (("document", "upload", "attachment", "pdf", "file"), "document"),
    (("checklist", "inspection", "question", "task"), "checklist"),
    (("billing", "calculator", "fee", "payment", "charge"), "billing"),
    (("inbox", "search", "filter", "assignee"), "inbox"),
    (("deploy", "deployment", "environment", "release", "go live", "go-live"), "deploy"),
    (("localisation", "localization", "language", "translation", "i18n"), "localisation"),
    (("service", "module", "application", "mdms", "master data"), "service"),
    (("address", "boundary", "location", "geography"), "address"),
    # HCM-specific
    (("campaign", "health campaign", "hcm"), "campaign"),
    (("beneficiary", "household", "registration"), "beneficiary"),
    (("stock", "inventory", "warehouse", "supply"), "stock"),
    (("supervisor", "field worker", "distributor"), "staff"),
]


def detect_section_hint(query: str) -> str | None:
    """
    Return a section keyword hint if the query strongly matches a known section,
    else None. Used to pre-filter retrieval candidates.

    Returns None if no confident match (better to retrieve broadly than miss).
    """
    q_lower = query.lower()
    matches = []
    for keywords, section in _SECTION_KEYWORDS:
        for kw in keywords:
            if kw in q_lower:
                matches.append(section)
                break  # one match per section group is enough

    if len(matches) == 1:
        # Confident single-section match → filter
        logger.info("SectionHint: '%s' → section filter: '%s'", query[:60], matches[0])
        return matches[0]
    elif len(matches) > 1:
        # Ambiguous — multiple sections match, don't filter (avoid missing results)
        logger.info("SectionHint: '%s' → ambiguous (%s), no filter", query[:60], matches)
        return None
    return None


# ─────────────────────────────────────────────
# Hybrid retrieval: vector + BM25 → RRF
# ─────────────────────────────────────────────
def hybrid_retrieve_pg(query: str, top_k: int = 5, section_hint: str = None) -> List[Tuple[str, Dict]]:
    """
    1. Vector search (top 2*k candidates via pgvector cosine)
    2. BM25 full-text search (top 2*k candidates via PostgreSQL tsvector)
    3. Fuse with Reciprocal Rank Fusion → return top_k
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            query_embedding = get_embedding(query)
            fetch = top_k * 2  # retrieve more candidates for fusion

            # build section filter clause
            section_filter = ""
            section_params_vector = [query_embedding, query_embedding, fetch]
            section_params_bm25 = [query, query, fetch]
            if section_hint:
                section_filter = "AND section ILIKE %s"
                section_params_vector = [query_embedding, query_embedding, f"%{section_hint}%", fetch]
                section_params_bm25 = [query, query, f"%{section_hint}%", fetch]

            # ── Vector search ──
            cur.execute(f"""
                SELECT id, document, section,
                       (1 - (embedding <=> %s::vector)) AS score
                FROM {TABLE}
                WHERE 1=1 {section_filter}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, section_params_vector)
            vector_rows = [(row[1], {"id": row[0], "section": row[2], "score": row[3]}) for row in cur.fetchall()]

            # ── BM25 full-text search ──
            cur.execute(f"""
                SELECT id, document, section,
                       ts_rank_cd(fts, websearch_to_tsquery('english', %s)) AS score
                FROM {TABLE}
                WHERE fts @@ websearch_to_tsquery('english', %s) {section_filter}
                ORDER BY score DESC
                LIMIT %s
            """, section_params_bm25)
            bm25_rows = [(row[1], {"id": row[0], "section": row[2], "score": row[3]}) for row in cur.fetchall()]

            logger.info("Retrieval: Vector: %d | BM25: %d", len(vector_rows), len(bm25_rows))

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


def multi_query_retrieve(
    queries: list,
    top_k: int = 5,
    section_hint: str = None,
) -> list:
    """
    Run hybrid_retrieve_pg for each query variant, deduplicate by chunk ID,
    then re-rank the merged pool using RRF across all result lists.

    Args:
        queries: List of query strings (original + variants from generate_query_variants)
        top_k: Final number of chunks to return
        section_hint: Optional section name to filter results (from detect_section_hint)

    Returns same format as hybrid_retrieve_pg: List[Tuple[str, Dict]]
    """
    if not queries:
        return []

    # Collect ranked lists from each query variant
    all_ranked_lists = []
    seen_ids: set = set()
    all_docs_meta: dict = {}  # chunk_id → (doc_text, meta)

    for q in queries:
        results = hybrid_retrieve_pg(q, top_k=top_k * 2, section_hint=section_hint)
        if results:
            # Build a ranked list of (doc_text, score) for RRF
            ranked = [(doc, {"score": meta["score"]}) for doc, meta in results]
            all_ranked_lists.append(ranked)
            for doc, meta in results:
                chunk_id = meta.get("id", doc[:40])
                if chunk_id not in seen_ids:
                    seen_ids.add(chunk_id)
                    all_docs_meta[chunk_id] = (doc, meta)

    if not all_ranked_lists:
        return []

    # RRF across all ranked lists (generalised to N lists)
    from collections import defaultdict
    k = 60
    rrf_scores: dict = defaultdict(float)
    id_to_doc: dict = {}

    for ranked_list in all_ranked_lists:
        for rank, (doc, _) in enumerate(ranked_list):
            # Map doc text back to chunk_id
            chunk_id = next(
                (cid for cid, (d, _) in all_docs_meta.items() if d == doc),
                doc[:40]
            )
            rrf_scores[chunk_id] += 1.0 / (k + rank + 1)
            id_to_doc[chunk_id] = doc

    # Sort by RRF score
    ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    # Rebuild results with original metadata, adding rrf_score
    results_out = []
    for chunk_id, rrf_score in ranked_ids[:top_k]:
        if chunk_id in all_docs_meta:
            doc_text, meta = all_docs_meta[chunk_id]
            results_out.append((doc_text, {**meta, "score": rrf_score}))

    logger.info(
        "MultiQuery: %d queries → %d unique chunks → top %d after RRF",
        len(queries), len(seen_ids), len(results_out)
    )
    return results_out
