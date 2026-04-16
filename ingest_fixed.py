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
                embedding vector(1536),
                url TEXT,
                tag TEXT,
                version TEXT
            );
        """)
    conn.commit()


def ingest(file_path):
    conn = get_conn()
    create_table(conn)

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"Ingesting {len(lines)} chunks...")

    with conn.cursor() as cur:
        for i, line in enumerate(lines):
            chunk = json.loads(line)

            try:
                emb = get_embedding(chunk["document"])

                cur.execute(f"""
                    INSERT INTO {TABLE} (id, document, embedding, url, tag, version)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT (id) DO NOTHING
                """, (
                    chunk["id"],
                    chunk["document"],
                    emb,
                    chunk.get("url"),
                    chunk.get("tag"),
                    chunk.get("version")
                ))

                if i % 5 == 0:
                    print(f"Inserted {i+1}/{len(lines)}")

            except Exception as e:
                print(f"❌ Failed at chunk {i}: {e}")

    conn.commit()
    conn.close()
    print("✅ Ingestion complete!")


if __name__ == "__main__":
    ingest("data/studio_chunks.jsonl")
