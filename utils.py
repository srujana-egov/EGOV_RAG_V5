import os
import json
import datetime
import tiktoken
from typing import Optional, Callable, Any
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
    from psycopg2 import pool
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None
    pool = None

try:
    import streamlit as st
except ImportError:
    st = None


# ─────────────────────────────────────────────
# Env var helper
# ─────────────────────────────────────────────
def get_env_var(key: str, default=None):
    if st and hasattr(st, "secrets"):
        try:
            if key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.environ.get(key, default)


# ─────────────────────────────────────────────
# Connection pool
# ─────────────────────────────────────────────
_pool = None

def _get_pool():
    global _pool
    if _pool is None:
        _pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dbname=get_env_var("PGDATABASE"),
            user=get_env_var("PGUSER"),
            password=get_env_var("PGPASSWORD"),
            host=get_env_var("PGHOST"),
            port=get_env_var("PGPORT", "5432"),
            sslmode=get_env_var("PGSSLMODE", "require")
        )
    return _pool


def get_conn():
    p = _get_pool()
    conn = p.getconn()
    register_vector(conn)
    return conn


def release_conn(conn):
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


# ─────────────────────────────────────────────
# Ensure feedback table exists (run once on startup)
# ─────────────────────────────────────────────
def ensure_feedback_table():
    """Creates the feedback table if it doesn't exist."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_feedback (
                    id          SERIAL PRIMARY KEY,
                    created_at  TIMESTAMPTZ DEFAULT NOW(),
                    query       TEXT,
                    answer_snippet TEXT,
                    rating      VARCHAR(10),   -- 'positive' or 'negative'
                    source      VARCHAR(20),   -- 'predetermined' or 'rag'
                    comment     TEXT
                )
            """)
        conn.commit()
    except Exception as e:
        print(f"[DB] Could not create feedback table: {e}")
    finally:
        release_conn(conn)


# ─────────────────────────────────────────────
# Log feedback to DB (not a flat file)
# ─────────────────────────────────────────────
def log_feedback(query: str, answer: str, rating: str, source: str = "rag", comment: str = ""):
    """
    Persists user feedback to the database.
    rating: 'positive' | 'negative'
    source: 'predetermined' | 'rag'
    """
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_feedback (query, answer_snippet, rating, source, comment)
                VALUES (%s, %s, %s, %s, %s)
            """, (query, answer[:500], rating, source, comment))
        conn.commit()
        print(f"[Feedback] {rating} logged for: '{query[:60]}'")
    except Exception as e:
        print(f"[Feedback] DB log failed: {e}")
        # Fallback to file so feedback is never silently lost
        _log_feedback_file(query, answer, rating, source, comment)
    finally:
        release_conn(conn)


def _log_feedback_file(query: str, answer: str, rating: str, source: str, comment: str):
    """Fallback file logger in case DB is unavailable."""
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "answer_snippet": answer[:300],
        "rating": rating,
        "source": source,
        "comment": comment
    }
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────
# Fetch feedback for admin dashboard
# ─────────────────────────────────────────────
def get_recent_feedback(limit: int = 50) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, query, rating, source, comment
                FROM bot_feedback
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            return [
                {"created_at": r[0], "query": r[1], "rating": r[2], "source": r[3], "comment": r[4]}
                for r in rows
            ]
    except Exception as e:
        print(f"[Feedback] Fetch failed: {e}")
        return []
    finally:
        release_conn(conn)


def get_feedback_stats() -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END) AS negative,
                    SUM(CASE WHEN source = 'predetermined' THEN 1 ELSE 0 END) AS from_cache,
                    SUM(CASE WHEN source = 'rag' THEN 1 ELSE 0 END) AS from_rag
                FROM bot_feedback
            """)
            row = cur.fetchone()
            total = row[0] or 0
            return {
                "total": total,
                "positive": row[1] or 0,
                "negative": row[2] or 0,
                "satisfaction": round((row[1] or 0) / total * 100) if total > 0 else 0,
                "from_cache": row[3] or 0,
                "from_rag": row[4] or 0,
            }
    except Exception as e:
        print(f"[Feedback] Stats failed: {e}")
        return {}
    finally:
        release_conn(conn)


# ─────────────────────────────────────────────
# Chunk insert helper
# ─────────────────────────────────────────────
def insert_chunk(doc_id: str, text: str, metadata: dict, get_embedding, table: str = "studio_manual"):
    emb = get_embedding(text)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table} (id, document, embedding, url, tag, version)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document  = EXCLUDED.document,
                    embedding = EXCLUDED.embedding,
                    url       = EXCLUDED.url,
                    tag       = EXCLUDED.tag,
                    version   = EXCLUDED.version
            """, (doc_id, text, emb, metadata.get("url"), metadata.get("tag"), metadata.get("version")))
        conn.commit()
    finally:
        release_conn(conn)


# ─────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────
def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


def get_required_env(name: str, cast: Optional[Callable[[str], Any]] = None) -> Any:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Required env var {name!r} is not set.")
    if cast is not None:
        return cast(v)
    return v
