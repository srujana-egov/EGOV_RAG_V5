"""
Ingest new_chunks.json into the studio_manual table.

Steps:
  1. Reads new_chunks.json (list of {id, document, section}).
  2. Generates embeddings via OpenAI text-embedding-3-small.
  3. Upserts each row into Postgres (INSERT ... ON CONFLICT DO UPDATE).
  4. Optionally deletes the stale chunk `us_configurable_address_criteria`.

Run:
    python ingest_new_chunks.py
"""

import json
import os
import sys
from dotenv import load_dotenv
import openai

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────
CHUNKS_FILE   = "new_chunks.json"
TABLE         = os.environ.get("DB_TABLE", "studio_manual")
DELETE_STALE  = ["us_configurable_address_criteria"]   # confirmed bad chunk
EMBEDDING_MODEL = "text-embedding-3-small"
BATCH_SIZE    = 20  # max texts per embedding API call

# ── Clients ────────────────────────────────────────────────────────────────
from utils import get_conn, get_env_var

client = openai.OpenAI(api_key=get_env_var("OPENAI_API_KEY"))


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    items = sorted(resp.data, key=lambda x: x.index)
    return [item.embedding for item in items]


def main():
    # ── Load chunks ──────────────────────────────────────────────────────
    with open(CHUNKS_FILE, "r") as f:
        chunks = json.load(f)
    print(f"[Ingest] Loaded {len(chunks)} chunks from {CHUNKS_FILE}")

    # ── Generate embeddings in batches ───────────────────────────────────
    texts = [c["document"] for c in chunks]
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"[Embed]  Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} texts)…")
        embeddings.extend(embed_batch(batch))
    print(f"[Embed]  Done — {len(embeddings)} embeddings generated.")

    # ── Upsert into Postgres ─────────────────────────────────────────────
    conn = get_conn()
    inserted = 0
    updated  = 0
    try:
        with conn.cursor() as cur:
            for chunk, emb in zip(chunks, embeddings):
                cur.execute(
                    f"""
                    INSERT INTO {TABLE} (id, document, section, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (id) DO UPDATE
                        SET document  = EXCLUDED.document,
                            section   = EXCLUDED.section,
                            embedding = EXCLUDED.embedding
                    """,
                    (chunk["id"], chunk["document"], chunk.get("section", ""), emb),
                )
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    updated += 1
                print(f"  ✓  {chunk['id']}")

            # ── Delete stale chunks ──────────────────────────────────────
            for stale_id in DELETE_STALE:
                cur.execute(f"DELETE FROM {TABLE} WHERE id = %s", (stale_id,))
                if cur.rowcount:
                    print(f"  🗑  Deleted stale chunk: {stale_id}")
                else:
                    print(f"  ℹ  Stale chunk not found (already deleted?): {stale_id}")

        conn.commit()
        print(f"\n[Ingest] Complete — {inserted} inserted, {updated} updated, "
              f"{len(DELETE_STALE)} stale chunk(s) removed.")
    except Exception as e:
        conn.rollback()
        print(f"[Ingest] ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        conn.close()

    # ── Quick row count ──────────────────────────────────────────────────
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
            total = cur.fetchone()[0]
        print(f"[DB]     Total rows in {TABLE}: {total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
