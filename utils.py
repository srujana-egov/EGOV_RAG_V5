import os
import json
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
# Env var helper — Streamlit secrets → OS env
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
# Connection pool  ← replaces open/close per query
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
    """
    Get a connection from the pool.
    Always call conn.close() after use — this returns it to the pool,
    it does NOT close the actual connection.
    """
    p = _get_pool()
    conn = p.getconn()
    register_vector(conn)
    return conn


def release_conn(conn):
    """Explicitly return a connection to the pool."""
    try:
        _get_pool().putconn(conn)
    except Exception:
        pass


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
            """, (
                doc_id,
                text,
                emb,
                metadata.get("url"),
                metadata.get("tag"),
                metadata.get("version")
            ))
        conn.commit()
    finally:
        release_conn(conn)


# ─────────────────────────────────────────────
# Token counting
# ─────────────────────────────────────────────
def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# ─────────────────────────────────────────────
# Required env var (raises if missing)
# ─────────────────────────────────────────────
def get_required_env(name: str, cast: Optional[Callable[[str], Any]] = None) -> Any:
    v = os.getenv(name)
    if v is None or str(v).strip() == "":
        raise RuntimeError(f"Required environment variable {name!r} is not set or empty.")
    if cast is not None:
        try:
            return cast(v)
        except Exception as e:
            raise RuntimeError(f"Error casting env var {name!r} = {v!r}: {e}")
    return v


# ─────────────────────────────────────────────
# Feedback logger
# ─────────────────────────────────────────────
def log_feedback(query: str, answer: str, rating: str, comment: str = ""):
    """
    Logs user feedback to a JSONL file for later analysis.
    rating: 'positive' or 'negative'
    """
    import datetime
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query": query,
        "answer_snippet": answer[:300],
        "rating": rating,
        "comment": comment
    }
    with open("feedback_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[Feedback] Logged: {rating} for query: '{query[:60]}'")
