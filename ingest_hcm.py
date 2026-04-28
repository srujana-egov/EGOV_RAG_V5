"""
ingest_hcm.py — Ingest HCM knowledge base chunks into the hcm_manual table.

Usage:
    # Activate venv and source HCM env vars first:
    set -a && source .env.hcm && set +a
    python ingest_hcm.py [path/to/hcm_chunks.jsonl]

The target table (default: hcm_manual) is read from the DB_TABLE env var set in .env.hcm.
Schema created matches studio_manual exactly so the same retrieval.py code works for both.
"""
import json
import os
import sys
import re
import logging

import psycopg2
from psycopg2.extras import execute_values

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TABLE = os.environ.get("DB_TABLE", "hcm_manual")
_TABLE_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]{0,62}$")
if not _TABLE_RE.match(TABLE):
    raise ValueError(f"DB_TABLE contains invalid characters: {TABLE!r}")

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "1536"))


# ── DB connection ─────────────────────────────────────────────────────────────
def get_conn():
    conn = psycopg2.connect(
        user=os.environ["PGUSER"],
        password=os.environ["PGPASSWORD"],
        host=os.environ["PGHOST"],
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ["PGDATABASE"],
        sslmode=os.environ.get("PGSSLMODE", "require"),
    )
    try:
        from pgvector.psycopg2 import register_vector
        register_vector(conn)
    except Exception as e:
        logger.warning("pgvector register_vector failed (non-fatal): %s", e)
    return conn


# ── Table setup ───────────────────────────────────────────────────────────────
def ensure_table(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id      TEXT PRIMARY KEY,
                document TEXT NOT NULL,
                embedding vector({EMBED_DIM}),
                section TEXT
            );
        """)
        # FTS column (GENERATED … STORED requires PG 12+)
        cur.execute(f"""
            ALTER TABLE {TABLE}
            ADD COLUMN IF NOT EXISTS fts tsvector
                GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
        """)
        # HNSW index for vector search
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_hnsw_idx
            ON {TABLE} USING hnsw (embedding vector_cosine_ops);
        """)
        # GIN index for BM25 full-text search
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {TABLE}_fts_gin_idx
            ON {TABLE} USING gin (fts);
        """)
    conn.commit()
    logger.info("Table %s ready.", TABLE)


# ── Embeddings ────────────────────────────────────────────────────────────────
def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts in one API call."""
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts, timeout=60)
    items = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in items]


# ── Ingest ────────────────────────────────────────────────────────────────────
def ingest(file_path: str, batch_size: int = 20):
    with open(file_path, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d chunks from %s", len(chunks), file_path)

    conn = get_conn()
    ensure_table(conn)

    inserted = 0
    skipped = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start: batch_start + batch_size]
        texts = [c["document"] for c in batch]

        try:
            embeddings = get_embeddings_batch(texts)
        except Exception as e:
            logger.error("Embedding failed for batch starting at %d: %s", batch_start, e)
            continue

        rows = [
            (
                c["id"],
                c["document"],
                emb,
                c.get("section", ""),
            )
            for c, emb in zip(batch, embeddings)
        ]

        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {TABLE} (id, document, embedding, section)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE
                        SET document  = EXCLUDED.document,
                            embedding = EXCLUDED.embedding,
                            section   = EXCLUDED.section
                    """,
                    rows,
                )
            conn.commit()
            inserted += len(rows)
            logger.info("Inserted batch %d–%d (%d total so far)",
                        batch_start + 1, batch_start + len(batch), inserted)
        except Exception as e:
            conn.rollback()
            logger.error("DB insert failed for batch at %d: %s", batch_start, e)
            skipped += len(batch)

    conn.close()
    logger.info("✅ Done. Inserted/updated: %d | Skipped: %d", inserted, skipped)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/hcm_chunks.jsonl"
    ingest(path)
