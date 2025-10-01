import math
from typing import Any, Dict, List, Tuple
import numpy as np

from utils import get_conn, get_env_var  # shared utils

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None

# ---------------------------
# Config
# ---------------------------
TABLE = "hcm_manual"
ID_COL, TXT_COL, URL_COL, TAG_COL, VER_COL = "id", "document", "url", "tag", "version"
EMB_COL = get_env_var("EMBED_COL", "embedding")

EMBED_MODEL     = get_env_var("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM       = int(get_env_var("EMBED_DIM", "1536"))
EMBED_NORMALIZE = get_env_var("EMBED_NORMALIZE", "1") == "1"

CAND_MULT       = int(get_env_var("RETRIEVE_CAND_MULT", "10"))
MAX_SQL_LIMIT   = int(get_env_var("RETRIEVE_SQL_LIMIT", "300"))
MMR_LAMBDA      = float(get_env_var("MMR_LAMBDA", "0.7"))

# ---------------------------
# Embeddings
# ---------------------------
def get_embedding(text: str) -> List[float]:
    api_key = get_env_var("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY missing for embeddings.")
    
    import openai
    client = openai.OpenAI(api_key=api_key)
    
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding

# ---------------------------
# Text utils
# ---------------------------
def _tf_dict(s: str) -> Dict[str, float]:
    from collections import Counter
    toks = (s or "").lower().split()
    if not toks: return {}
    c = Counter(toks); tot = float(sum(c.values()))
    return {t: v/tot for t, v in c.items()} if tot > 0 else {}

def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    common = set(a) & set(b)
    num = sum(a[t]*b[t] for t in common)
    na = math.sqrt(sum(v*v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v*v for v in b.values())) or 1.0
    return num/(na*nb)

# ---------------------------
# MMR reranking
# ---------------------------
def mmr_select(scored: List[Any], q_vec=None, k: int=10, lambda_: float=MMR_LAMBDA) -> List[Any]:
    n = len(scored)
    if n==0 or k<=0: return []
    k = min(k,n)
    idx = list(range(n))
    idx.sort(key=lambda i: float(scored[i].get("score", 0.0)), reverse=True)
    selected = [idx[0]]; avail = idx[1:]
    vecs = [c.get("tfidf") for c in scored]
    rels = [c.get("score", 0.0) for c in scored]
    while avail and len(selected)<k:
        best_i, best_val = None, -1e18
        for i in avail:
            rel = rels[i]
            div = max(_cosine(vecs[i], vecs[j]) for j in selected if vecs[j]) if vecs[i] else 0.0
            val = lambda_*rel - (1.0-lambda_)*div
            if val>best_val: best_val, best_i = val, i
        selected.append(best_i); avail.remove(best_i)
    return [scored[i] for i in selected]

# ---------------------------
# Candidate retrieval
# ---------------------------
def vector_candidates(conn, query: str, need: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        qvec = get_embedding(query)
        sql_vec = f"""
          SELECT
            {ID_COL} AS id, {TXT_COL} AS document, {URL_COL}, {TAG_COL}, {VER_COL},
            (1.0 - ({EMB_COL} <=> %s::vector))::float AS score
          FROM {TABLE}
          ORDER BY {EMB_COL} <=> %s::vector
          LIMIT %s
        """
        cur.execute(sql_vec, [qvec, qvec, need])
        rows = cur.fetchall()
        for r in rows:
            r["metadata"] = {
                "url": r.get("url"),
                "tag": r.get("tag"),
                "version": r.get("version"),
            }
    return rows

def hybrid_retrieve_pg(query: str, top_k: int = 20, mmr_lambda: float = MMR_LAMBDA) -> List[Tuple[str, Dict[str, Any]]]:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM hcm_manual")
    print("[DB] Total rows in table:", cur.fetchone())

    try:
        need = min(max(top_k * CAND_MULT, 50), MAX_SQL_LIMIT)
        base = vector_candidates(conn, query, need)
    except Exception as e:
        print("[ERROR] vector_candidates failed:", e)
        base = []
    finally:
        try: conn.close()
        except Exception: pass

    if not base:
        print("[DEBUG] No candidates found for query:", query)
        return []

    print(f"[DEBUG] Retrieved {len(base)} candidates for query: '{query}'")
    for r in base[:5]:
        print(f"   id={r['id']}  score={r['score']:.4f}  text_snippet='{(r['document'] or '')[:80]}'")

    # Prep for MMR
    cands: List[Dict[str, Any]] = []
    for r in base:
        text = r.get("document") or ""
        meta_raw = r.get("metadata", {})
        meta = {
            "id": str(r.get("id")) if r.get("id") else None,
            "source": meta_raw.get("url"),
            "score": float(r.get("score") or 0.0),
            "tfidf": _tf_dict(text),
        }
        cands.append({"text": text, "score": meta["score"], "tfidf": meta["tfidf"], "meta": meta})

    q_vec = _tf_dict(query)
    selected = mmr_select(cands, q_vec=q_vec, k=max(top_k, 5), lambda_=mmr_lambda)

    print(f"[DEBUG] After MMR, selected {len(selected)} results")
    for s in selected[:5]:
        print(f"   id={s['meta']['id']}  score={s['score']:.4f}  text_snippet='{s['text'][:80]}'")

    out: List[Tuple[str, Dict[str, Any]]] = []
    for s in selected[:top_k]:
        meta = dict(s["meta"])
        out.append((s["text"], meta))
    return out


# ---------------------------
# Formatter
# ---------------------------
def format_result(row):
    meta = row.get("metadata", {})
    url = (meta.get("url") or "").strip().lower().rstrip("/") if isinstance(meta, dict) else ""
    return {
        "doc_id": url if url else row["id"],
        "chunk_id": row["id"],
        "title": row.get("document", ""),
        "score": row.get("score", 0.0),
        "url": url,
    }
