import os, json, time, psycopg2
from tqdm import tqdm
import openai
from src.utils import get_env_var, get_conn
from src.retrieval import get_embedding  # <-- add this

# --- Table config ---
TABLE = "hcm_manual"
ID_COL, TXT_COL, URL_COL, TAG_COL, VER_COL, EMB_COL = "id", "document", "url", "tag", "version", "embedding"


def safe_get_embedding(text, retries=5):
    """
    Get embedding with retries (handles OpenAI rate limits).
    """
    for i in range(retries):
        try:
            return [float(x) for x in get_embedding(text)]
        except openai.error.RateLimitError:
            wait = 2 ** i
            print(f"⚠️ Rate limit hit, retrying in {wait}s...")
            time.sleep(wait)
        except Exception as e:
            if i == retries - 1:
                raise
            wait = 2 ** i
            print(f"⚠️ Embedding error {e}, retrying in {wait}s...")
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
        for i, line in enumerate(tqdm(f, total=(limit or total_lines), desc="Ingesting chunks")):
            if limit is not None and i >= limit:
                break

            row = json.loads(line)
            chunk_id = row["id"]
            text = row.get("document", "")
            url = row.get("url", None)
            tag = row.get("tag", None)
            version = row.get("version", None)

            try:
                emb = safe_get_embedding(text)

                cur.execute(
                    f"""
                    INSERT INTO {TABLE} ({ID_COL}, {TXT_COL}, {URL_COL}, {TAG_COL}, {VER_COL}, {EMB_COL})
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT ({ID_COL})
                    DO UPDATE SET {TXT_COL} = EXCLUDED.{TXT_COL},
                                  {URL_COL} = EXCLUDED.{URL_COL},
                                  {TAG_COL} = EXCLUDED.{TAG_COL},
                                  {VER_COL} = EXCLUDED.{VER_COL},
                                  {EMB_COL} = EXCLUDED.{EMB_COL}
                    """,
                    (chunk_id, text, url, tag, version, emb),
                )
                inserted += 1
                if inserted % commit_every == 0:
                    conn.commit()
                    print(f"💾 Committed {inserted} so far...")

            except Exception as e:
                print(f"❌ Failed for {chunk_id}: {e}")
                failed += 1
                with open("failed_chunks.jsonl", "a", encoding="utf-8") as out:
                    out.write(json.dumps(row) + "\n")
                continue

    print("✅ Final commit...")
    conn.commit()

    cur.execute(f"SELECT COUNT(*) FROM {TABLE}")
    count = cur.fetchone()[0]
    print(f"📊 Row count in {TABLE}: {count}")

    cur.close()
    conn.close()
    print(f"🎉 Done. Inserted {inserted} chunks, {failed} failed, into {TABLE}.")


if __name__ == "__main__":
    jsonl_file = "/Users/srujana/Desktop/RAG-TDD/data/normalized_chunks.jsonl"  # path to your JSONL file
    ingest_jsonl(jsonl_file, limit=None)  # run full dataset

