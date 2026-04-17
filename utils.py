import os
import json
import datetime
import urllib.request
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tiktoken
from typing import Optional, Callable, Any
from dotenv import load_dotenv

load_dotenv()

try:
    import psycopg2
    from pgvector.psycopg2 import register_vector
except ImportError:
    psycopg2 = None

try:
    import streamlit as st
except ImportError:
    st = None

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
PROMOTION_THRESHOLD = 5  # positive votes before auto-promoting RAG answer to Q&A cache


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
# DB Connection
# ─────────────────────────────────────────────
def get_conn():
    conn = psycopg2.connect(
        dbname=get_env_var("PGDATABASE"),
        user=get_env_var("PGUSER"),
        password=get_env_var("PGPASSWORD"),
        host=get_env_var("PGHOST"),
        port=get_env_var("PGPORT", "5432"),
        sslmode="require",
        connect_timeout=5
    )
    register_vector(conn)
    return conn


# ─────────────────────────────────────────────
# Query history table
# ─────────────────────────────────────────────
def ensure_query_history_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS query_history (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    query TEXT,
                    answer_snippet TEXT,
                    source VARCHAR(20)
                )
            """)
        conn.commit()
    except Exception as e:
        print(f"[DB] Could not create query_history table: {e}")
    finally:
        conn.close()


def ensure_vote_log_table():
    """Lightweight table for tracking thumbs-up/down counts used for auto-promotion."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS vote_log (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    query TEXT,
                    answer_snippet TEXT,
                    rating VARCHAR(10)
                )
            """)
        conn.commit()
    except Exception as e:
        print(f"[DB] Could not create vote_log table: {e}")
    finally:
        conn.close()


def log_vote(query: str, answer: str, rating: str):
    """Record a vote (positive or negative) for auto-promotion tracking."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO vote_log (query, answer_snippet, rating)
                VALUES (%s, %s, %s)
            """, (query, answer[:500], rating))
        conn.commit()
    except Exception as e:
        print(f"[VoteLog] Log failed: {e}")
    finally:
        conn.close()


def log_query(query: str, answer: str, source: str):
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO query_history (query, answer_snippet, source)
                VALUES (%s, %s, %s)
            """, (query, answer[:500], source))
        conn.commit()
    except Exception as e:
        print(f"[QueryHistory] Log failed: {e}")
    finally:
        conn.close()


def get_query_history(limit: int = 200) -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, query, answer_snippet, source
                FROM query_history
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
            return [
                {"created_at": r[0], "query": r[1], "answer_snippet": r[2], "source": r[3]}
                for r in rows
            ]
    except Exception as e:
        print(f"[QueryHistory] Fetch failed: {e}")
        return []
    finally:
        conn.close()


def get_flagged_queries() -> list:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, query, answer_snippet, source, comment
                FROM bot_feedback
                WHERE is_flagged = TRUE
                ORDER BY created_at DESC
            """)
            rows = cur.fetchall()
            return [
                {"created_at": r[0], "query": r[1], "answer_snippet": r[2],
                 "source": r[3], "comment": r[4] or ""}
                for r in rows
            ]
    except Exception as e:
        print(f"[QueryHistory] Flagged fetch failed: {e}")
        return []
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Ensure feedback table (with is_flagged column)
# ─────────────────────────────────────────────
def ensure_feedback_table():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS bot_feedback (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    query TEXT,
                    answer_snippet TEXT,
                    rating VARCHAR(10),
                    source VARCHAR(20),
                    comment TEXT,
                    is_flagged BOOLEAN DEFAULT FALSE
                )
            """)
            # Add is_flagged column if missing on older tables
            cur.execute("""
                ALTER TABLE bot_feedback
                ADD COLUMN IF NOT EXISTS is_flagged BOOLEAN DEFAULT FALSE
            """)
        conn.commit()
    except Exception as e:
        print(f"[DB] Could not create feedback table: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Ensure Q&A table with full schema
# ─────────────────────────────────────────────
def ensure_qa_table_full():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS predetermined_qa (
                    id SERIAL PRIMARY KEY,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    confidence FLOAT DEFAULT 1.0,
                    positive_votes INT DEFAULT 0,
                    negative_votes INT DEFAULT 0,
                    source VARCHAR(20) DEFAULT 'manual',
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            # Add missing columns on older tables
            for stmt in [
                "ALTER TABLE predetermined_qa ADD COLUMN IF NOT EXISTS positive_votes INT DEFAULT 0",
                "ALTER TABLE predetermined_qa ADD COLUMN IF NOT EXISTS negative_votes INT DEFAULT 0",
                "ALTER TABLE predetermined_qa ADD COLUMN IF NOT EXISTS source VARCHAR(20) DEFAULT 'manual'",
                "ALTER TABLE predetermined_qa ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
                "ALTER TABLE predetermined_qa ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
            ]:
                try:
                    cur.execute(stmt)
                except Exception:
                    conn.rollback()
        conn.commit()
    except Exception as e:
        print(f"[DB] Could not ensure QA table: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Feedback logging
# ─────────────────────────────────────────────
def log_feedback(query: str, answer: str, rating: str, source: str = "rag", comment: str = ""):
    is_flagged = (rating == "negative")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO bot_feedback (query, answer_snippet, rating, source, comment, is_flagged)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (query, answer[:500], rating, source, comment, is_flagged))
        conn.commit()
        print(f"[Feedback] {rating} logged (flagged={is_flagged})")
    except Exception as e:
        print(f"[Feedback] DB log failed: {e}")
        _log_feedback_file(query, answer, rating, source, comment)
    finally:
        conn.close()


def _log_feedback_file(query: str, answer: str, rating: str, source: str, comment: str):
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
# Vote tracking + auto-promotion to Q&A cache
# ─────────────────────────────────────────────
def update_qa_votes_and_promote(query: str, answer: str, rating: str):
    """
    Track votes per query. On positive rating:
    - Count all positive/negative feedback for this exact query
    - If positive_votes >= PROMOTION_THRESHOLD, auto-promote to predetermined_qa
    - Update confidence for entries already in the cache
    """
    if not query or not answer:
        return

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            # Count votes for this query from vote_log (case-insensitive)
            cur.execute("""
                SELECT
                    SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END),
                    SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END)
                FROM vote_log
                WHERE LOWER(TRIM(query)) = LOWER(TRIM(%s))
            """, (query,))
            row = cur.fetchone()
            pos_votes = int(row[0] or 0)
            neg_votes = int(row[1] or 0)
            total = pos_votes + neg_votes
            confidence = round(pos_votes / total, 3) if total > 0 else 1.0

            # Check if already in predetermined_qa
            cur.execute("""
                SELECT id FROM predetermined_qa
                WHERE LOWER(TRIM(question)) = LOWER(TRIM(%s))
            """, (query,))
            existing = cur.fetchone()

            if existing:
                # Update vote counts and confidence for existing entry
                cur.execute("""
                    UPDATE predetermined_qa
                    SET positive_votes = %s,
                        negative_votes = %s,
                        confidence = %s,
                        updated_at = NOW()
                    WHERE id = %s
                """, (pos_votes, neg_votes, confidence, existing[0]))
                print(f"[Votes] Updated existing Q&A confidence to {confidence}")
            elif rating == "positive" and pos_votes >= PROMOTION_THRESHOLD:
                # Auto-promote this RAG answer to the Q&A cache
                cur.execute("""
                    INSERT INTO predetermined_qa
                        (question, answer, confidence, positive_votes, negative_votes, source)
                    VALUES (%s, %s, %s, %s, %s, 'auto_promoted')
                """, (query, answer, confidence, pos_votes, neg_votes))
                print(f"[Promote] Auto-promoted to Q&A cache after {pos_votes} votes: {query[:60]}...")

        conn.commit()
    except Exception as e:
        print(f"[Votes] Error updating votes: {e}")
    finally:
        conn.close()


# ─────────────────────────────────────────────
# Fetch feedback
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
        conn.close()


def get_feedback_stats() -> dict:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN rating = 'positive' THEN 1 ELSE 0 END) AS positive,
                    SUM(CASE WHEN rating = 'negative' THEN 1 ELSE 0 END) AS negative,
                    SUM(CASE WHEN source = 'cache' THEN 1 ELSE 0 END) AS from_cache,
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
        conn.close()


# ─────────────────────────────────────────────
# Weekly report data
# ─────────────────────────────────────────────
def get_flagged_feedback_for_report(days: int = 7) -> list:
    """Return all negative/flagged feedback from the past N days."""
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT created_at, query, answer_snippet, source, comment
                FROM bot_feedback
                WHERE rating = 'negative'
                  AND created_at >= NOW() - INTERVAL '%s days'
                ORDER BY created_at DESC
            """ % int(days))
            rows = cur.fetchall()
            return [
                {
                    "created_at": str(r[0]),
                    "query": r[1],
                    "answer_snippet": r[2],
                    "source": r[3],
                    "comment": r[4] or ""
                }
                for r in rows
            ]
    except Exception as e:
        print(f"[Report] Fetch failed: {e}")
        return []
    finally:
        conn.close()


def generate_weekly_report(days: int = 7) -> dict:
    """Compile stats and flagged items into a report dict."""
    stats = get_feedback_stats()
    flagged = get_flagged_feedback_for_report(days)

    # Count auto-promoted Q&As
    promoted_count = 0
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*) FROM predetermined_qa
                WHERE source = 'auto_promoted'
                  AND updated_at >= NOW() - INTERVAL '%s days'
            """ % int(days))
            promoted_count = cur.fetchone()[0] or 0
    except Exception:
        pass
    finally:
        conn.close()

    return {
        "stats": stats,
        "flagged_count": len(flagged),
        "flagged_items": flagged,
        "auto_promoted_count": promoted_count,
        "period_days": days,
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z"
    }


# ─────────────────────────────────────────────
# Send Slack report
# ─────────────────────────────────────────────
def send_slack_report(webhook_url: str, report: dict):
    """POST weekly report to a Slack Incoming Webhook."""
    stats = report.get("stats", {})
    flagged = report.get("flagged_items", [])
    promoted = report.get("auto_promoted_count", 0)

    lines = [
        f"*DIGIT Studio Bot — Weekly Report* (last {report['period_days']} days)",
        f"Generated: {report['generated_at']}",
        "",
        "*Summary*",
        f"• Total queries: {stats.get('total', 0)}",
        f"• 👍 Satisfaction: {stats.get('satisfaction', 0)}%  "
        f"({stats.get('positive', 0)} positive / {stats.get('negative', 0)} negative)",
        f"• ⚡ From cache: {stats.get('from_cache', 0)}  |  🔍 From RAG: {stats.get('from_rag', 0)}",
        f"• 🚀 Auto-promoted to Q&A cache: {promoted}",
        f"• 🚩 Flagged (thumbs down): {report['flagged_count']}",
    ]

    if flagged:
        lines += ["", "*Flagged Questions (need review):*"]
        for item in flagged[:15]:
            ts = item["created_at"][:10] if item["created_at"] else "?"
            lines.append(f"  • `{item['query'][:100]}` _{ts}_")

    payload = json.dumps({"text": "\n".join(lines)}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            print(f"[Slack] Report sent, status={resp.status}")
            return True
    except Exception as e:
        print(f"[Slack] Failed to send report: {e}")
        raise


# ─────────────────────────────────────────────
# Send email report
# ─────────────────────────────────────────────
def send_email_report(report: dict, to_email: str = None):
    """Send weekly report via SMTP. Reads config from env vars."""
    smtp_host = get_env_var("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(get_env_var("SMTP_PORT", "587"))
    smtp_user = get_env_var("SMTP_USER", "")
    smtp_pass = get_env_var("SMTP_PASS", "")
    recipient = to_email or get_env_var("REPORT_EMAIL", "")

    if not recipient:
        raise ValueError("No recipient email — set REPORT_EMAIL env var or pass to_email.")

    stats = report.get("stats", {})
    flagged = report.get("flagged_items", [])
    promoted = report.get("auto_promoted_count", 0)

    flagged_rows = "".join(
        f"<tr><td>{i['created_at'][:10]}</td><td>{i['query']}</td>"
        f"<td>{i.get('source','')}</td><td>{i.get('comment','')}</td></tr>"
        for i in flagged
    ) or "<tr><td colspan='4'>No flagged items this week 🎉</td></tr>"

    html = f"""
    <html><body style="font-family:sans-serif;max-width:700px;margin:auto;">
    <h2>DIGIT Studio Bot — Weekly Report</h2>
    <p><b>Period:</b> Last {report['period_days']} days &nbsp;|&nbsp;
       <b>Generated:</b> {report['generated_at']}</p>

    <h3>Summary</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;">
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Total queries</td><td>{stats.get('total', 0)}</td></tr>
      <tr><td>Satisfaction</td><td>{stats.get('satisfaction', 0)}%
          ({stats.get('positive', 0)} 👍 / {stats.get('negative', 0)} 👎)</td></tr>
      <tr><td>From cache (instant)</td><td>{stats.get('from_cache', 0)}</td></tr>
      <tr><td>From RAG</td><td>{stats.get('from_rag', 0)}</td></tr>
      <tr><td>Auto-promoted to Q&amp;A cache</td><td>{promoted}</td></tr>
      <tr><td>Flagged (thumbs down)</td><td><b>{report['flagged_count']}</b></td></tr>
    </table>

    <h3>Flagged Questions (Thumbs Down) — Need Review</h3>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">
      <tr><th>Date</th><th>Query</th><th>Source</th><th>Comment</th></tr>
      {flagged_rows}
    </table>

    <p style="color:#888;font-size:12px;">
      DIGIT Studio Support Bot · Auto-generated report
    </p>
    </body></html>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"DIGIT Studio Bot — Weekly Report ({report['generated_at'][:10]})"
    msg["From"] = smtp_user
    msg["To"] = recipient
    msg.attach(MIMEText(html, "html"))

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        if smtp_user and smtp_pass:
            server.login(smtp_user, smtp_pass)
        server.send_message(msg)
    print(f"[Email] Report sent to {recipient}")
    return True


# ─────────────────────────────────────────────
# Insert chunk (for ingestion)
# ─────────────────────────────────────────────
def insert_chunk(doc_id: str, text: str, metadata: dict, get_embedding, table: str = "studio_manual"):
    emb = get_embedding(text)
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {table} (id, document, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    document = EXCLUDED.document,
                    embedding = EXCLUDED.embedding
            """, (doc_id, text, emb))
        conn.commit()
    finally:
        conn.close()


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
