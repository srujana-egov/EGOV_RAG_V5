"""
Migration: drop url, tag, version columns from studio_manual table.
Run once: python migrate_drop_columns.py
"""
from utils import get_conn


def migrate():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                ALTER TABLE studio_manual
                    DROP COLUMN IF EXISTS url,
                    DROP COLUMN IF EXISTS tag,
                    DROP COLUMN IF EXISTS version;
            """)
        conn.commit()
        print("✅ Dropped url, tag, version columns from studio_manual.")
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
