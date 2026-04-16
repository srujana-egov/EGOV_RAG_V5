import math
import traceback
import os
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

from utils import get_conn, get_env_var, release_conn   # ← release_conn added

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None

try:
    import cohere
except ImportError:
    cohere = None

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
TABLE   = get_env_var("DB_TABLE", "studio_manual")
ID_COL  = "id"
TXT_COL = "document"
URL_COL = "url"
TAG_COL = "tag"
VER_COL = "version"
EMB_COL = get_env_var("EMBED_COL", "embedding")

EMBED_MODEL     = get_env_var("EMBEDDING_MODEL", "text-embedding-3-small")
EMBED_DIM       = int(get_env_var("EMBED_DIM", "1536"))
EMBED_NORMALIZE = get_env_var("EMBED_NORMALIZE", "1") == "1"

CAND_MULT     = int(get_env_var("RETRIEVE_CAND_MULT", "10"))
MAX_SQL_LIMIT = int(get_env_var("RETRIEVE_SQL_LIMIT", "300"))
MMR_LAMBDA    = float(get_env_var("MMR_LAMBDA", "0.7"))

RERANK_ENABLED = get_env_var("RERANK_ENABLED", "1") == "1"
RERANK_TOPK    = int(get_env_var("RERANK_TOPK", "50"))
RERANK_FINAL_K = int(get_env_var("RERANK_FINAL_K", "10"))
RERANK_MODEL   = get_env_var("RERANK_MODEL", "rerank-english-v3.0")


# ─────────────────────────────────────────────
# OpenAI client — created once, not per query
# ─────────────────────────────────────────────
_openai_client = None

def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        api_key = get_env_var("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY missing.")
        _openai_client = openai.OpenAI(api_key=api_key)
    return _openai_client


# ─────────────────────────────────────────────
# Embeddings
# ─────────────────────────────────────────────
def get_embedding(text: str) -> List[float]:
    client = _get_openai_client()
    resp = client.embeddings.create(model=EMBED_MODEL, input=text)
    return resp.data[0].embedding


# ─────────────────────────────────────────────
# Cohere reranker
# ─────────────────────────────────────────────
def cohere_rerank(query: str, candidates: List[Dict[str, Any]], top_k: int = RERANK_FINAL_K) -> List[Dict[str, Any]]:
    if not RERANK_ENABLED:
        return candidates[:top_k]
    if cohere is None:
        print("[Rerank] cohere not installed, skipping.")
        return candidates[:top_k]
    api_key = get_env_var("COHERE_API_KEY")
    if not api_key:
        print("[Rerank] COHERE_API_KEY not set, skipping.")
        return candidates[:top_k]
    try:
        co = cohere.Client(api_key)
        docs = [c.get("text", "") for c in candidates]
        results = co.rerank(
            model=RERANK_MODEL,
            query=query,
            documents=docs,
            top_n=min(top_k, len(candidates))
        )
        reranked = []
        for r in results.results:
            c = candidates[r.index].copy()
            c["rerank_score"] = r.relevance_score
            c["score"] = 0.3 * float(c.get("score", 0.0)) + 0.7 * r.relevance_score
            reranked.append(c)
        print(f"[Rerank] {len(candidates)} → top {len(reranked)}")
        return reranked
    except Exception as e:
        print(f"[Rerank] Failed: {e}, using original order.")
        return candidates[:top_k]


# ─────────────────────────────────────────────
# Text utils
# ─────────────────────────────────────────────
def _tf_dict(s: str) -> Dict[str, float]:
    toks = (s or "").lower().split()
    if not toks:
        return {}
    c = Counter(toks)
    tot = float(sum(c.values()))
    return {t: v / tot for t, v in c.items()} if tot > 0 else {}


def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v * v for v in b.values())) or 1.0
    return num / (na * nb)


def _keyword_boost(text: str, query: str) -> float:
    text_lower = (text or "").lower()
    boost = 0.0
    if query.lower() in text_lower:
        boost += 0.2
    words = [w for w in query.lower().split() if len(w) > 3]
    if words:
        matched = sum(1 for w in words if w in text_lower)
        boost += 0.1 * (matched / len(words))
    return boost


# ─────────────────────────────────────────────
# URL-aware MMR
# ─────────────────────────────────────────────
def mmr_select_url_aware(scored: List[Any], q_vec=None, k: int = 10, lambda_: float = MMR_LAMBDA) -> List[Any]:
    n = len(scored)
    if n == 0 or k <= 0:
        return []
    k = min(k, n)
    idx = sorted(range(n), key=lambda i: float(scored[i].get("score", 0.0)), reverse=True)
    selected = [idx[0]]
    avail = idx[1:]
    vecs = [c.get("tfidf") for c in scored]
    rels = [c.get("score", 0.0) for c in scored]
    urls = [c.get("meta", {}).get("url") for c in scored]
    url_count = Counter(urls)
    while avail and len(selected) < k:
        best_i, best_val = None, -1e18
        for i in avail:
            sims = []
            if vecs[i]:
                for j in selected:
                    sim = _cosine(vecs[i], vecs[j])
                    if urls[i] and urls[j] and urls[i] == urls[j]:
                        sim *= (0.2 + 0.1 / max(url_count[urls[i]], 1))
                    sims.append(sim)
            div = max(sims) if sims else 0.0
            val = lambda_ * rels[i] - (1.0 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        avail.remove(best_i)
    return [scored[i] for i in selected]


# ─────────────────────────────────────────────
# Vector retrieval
# ─────────────────────────────────────────────
def vector_candidates(conn, query: str, need: int) -> List[Dict[str, Any]]:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        qvec = get_embedding(query)
        cur.execute(f"""
            SELECT {ID_COL} AS id, {TXT_COL} AS document, {URL_COL}, {TAG_COL}, {VER_COL},
                   (1.0 - ({EMB_COL} <=> %s::vector))::float AS score
            FROM {TABLE}
            ORDER BY {EMB_COL} <=> %s::vector
            LIMIT %s
        """, [qvec, qvec, need])
        rows = list(cur.fetchall())
        for r in rows:
            r["metadata"] = {"url": r.get("url"), "tag": r.get("tag"), "version": r.get("version")}
        return rows


# ─────────────────────────────────────────────
# Main hybrid retrieval
# ─────────────────────────────────────────────
def hybrid_retrieve_pg(query: str, top_k: int = 20, mmr_lambda: float = MMR_LAMBDA) -> List[Tuple[str, Dict[str, Any]]]:
    conn = get_conn()
    try:
        need = min(max(top_k * CAND_MULT, RERANK_TOPK), MAX_SQL_LIMIT)
        base = vector_candidates(conn, query, need)
    except Exception as e:
        print("[ERROR] vector_candidates failed:", e)
        traceback.print_exc()
        base = []
    finally:
        release_conn(conn)   # ← FIXED: was conn.close() which destroyed the pooled connection

    if not base:
        return []

    cands: List[Dict[str, Any]] = []
    for r in base:
        text = r.get("document") or ""
        meta_raw = r.get("metadata", {})
        score = float(r.get("score") or 0.0) + _keyword_boost(text, query)
        meta = {"id": str(r.get("id")) if r.get("id") else None, "url": meta_raw.get("url"), "score": score}
        cands.append({"text": text, "score": score, "tfidf": _tf_dict(text), "meta": meta})

    if RERANK_ENABLED and cands:
        cands = cohere_rerank(query, cands, top_k=RERANK_TOPK)

    q_vec = _tf_dict(query)
    selected = mmr_select_url_aware(cands, q_vec=q_vec, k=max(top_k, 15), lambda_=mmr_lambda)

    merged: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"text": "", "meta": None, "score": 0.0})
    for s in selected:
        url = (s["meta"]["url"] or s["meta"]["id"] or "").strip().lower().rstrip("/")
        if merged[url]["text"]:
            merged[url]["text"] += "\n" + s["text"]
            merged[url]["score"] = max(merged[url]["score"], s["score"])
        else:
            merged[url] = {"text": s["text"], "meta": s["meta"], "score": s["score"]}

    final = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
    return [(v["text"], v["meta"]) for v in final]
