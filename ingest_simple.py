import os
import json
import time
import argparse
import psycopg2
from tqdm import tqdm
import openai

from utils import get_env_var, get_conn, release_conn

# ─────────────────────────────────────────────
# Table config — change here or via .env
# ─────────────────────────────────────────────
TABLE = get_env_var("DB_TABLE", "studio_manual")
ID_COL, TXT_COL, URL_COL, TAG_COL, VER_COL, EMB_COL = (
    "id", "document", "url", "tag", "version", "embedding"
)


def get_embedding(text: str) -> list:
    from retrieval import get_embedding as _get_embedding
    return [float(x) for x in _get_embedding(text)]


def safe_get_embedding(text: str, retries: int = 5) -> list:
    for i in range(retries):
        try:
            return get_embedding(text)
        except openai.RateLimitError:
            wait = 2 ** i
            print(f"⚠️  Rate limit, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i
            print(f"⚠️  Embedding error ({e}), retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError("❌ Failed to get embedding after retries")


def ingest_jsonl(jsonl_path: str, limit: int = None, commit_every: int = 100):
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"❌ JSONL file not found: {jsonl_path}")

    total_lines = sum(1 for _ in open(jsonl_path, "r", encoding="utf-8"))
    print(f"👉 Found {total_lines} chunks in {jsonl_path}")

    conn = get_conn()
    cur = conn.cursor()
    inserted = 0
    failed = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(tqdm(f, total=(limit or total_lines), desc="Ingesting")):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)
            chunk_id = row["id"]
            text     = row.get("document", "")
            url      = row.get("url", None)
            tag      = row.get("tag", None)
            version  = row.get("version", None)

            try:
                emb = safe_get_embedding(text)
                cur.execute(f"""
                    INSERT INTO {TABLE}
                        ({ID_COL}, {TXT_COL}, {URL_COL}, {TAG_COL}, {VER_COL}, {EMB_COL})
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT ({ID_COL}) DO UPDATE SET
                        {TXT_COL} = EXCLUDED.{TXT_COL},
                        {URL_COL} = EXCLUDED.{URL_COL},
                        {TAG_COL} = EXCLUDED.{TAG_COL},
                        {VER_COL} = EXCLUDED.{VER_COL},
                        {EMB_COL} = EXCLUDED.{EMB_COL}
                """, (chunk_id, text, url, tag, version, emb))

                inserted += 1
                if inserted % commit_every == 0:
                    conn.commit()
                    print(f"💾 Committed {inserted} so far...")

            except Exception as e:
                print(f"❌ Failed for {chunk_id}: {e}")
                failed += 1
                with open("failed_chunks.jsonl", "a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")

    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
    count = cur.fetchone()[0]
    print(f"📊 Total rows in {TABLE}: {count}")

    cur.close()
    release_conn(conn)
    print(f"🎉 Done. Inserted/updated {inserted}, failed {failed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest JSONL chunks into the vector DB")
    parser.add_argument(
        "--file",
        type=str,
        # ← relative path now, not hardcoded to /Users/srujana/...
        default=os.path.join("data", "normalized_chunks.jsonl"),
        help="Path to the JSONL file to ingest"
    )
    parser.add_argument("--limit", type=int, default=None, help="Max rows to ingest")
    args = parser.parse_args()

    ingest_jsonl(args.file, limit=args.limit)
