import json
from utils import get_conn
from retrieval import get_embedding

TABLE = "studio_manual"


def create_table(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE} (
                id TEXT PRIMARY KEY,
                document TEXT,
                embedding vector(1536)
            );
        """)
    conn.commit()


def ingest(file_path):
    conn = get_conn()
    create_table(conn)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Ingesting {len(lines)} chunks...")

    try:
        with conn.cursor() as cur:
            for i, line in enumerate(lines):
                chunk = json.loads(line)

                try:
                    emb = get_embedding(chunk["document"])

                    cur.execute(f"""
                        INSERT INTO {TABLE} (id, document, embedding)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (id) DO NOTHING
                    """, (
                        chunk["id"],
                        chunk["document"],
                        emb,
                    ))

                    if i % 5 == 0:
                        print(f"Inserted {i+1}/{len(lines)}")

                except Exception as e:
                    print(f"❌ Failed at chunk {i}: {e}")

        conn.commit()

    finally:
        conn.close()

    print("✅ Ingestion complete!")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/studio_chunks.jsonl"
    ingest(path)
