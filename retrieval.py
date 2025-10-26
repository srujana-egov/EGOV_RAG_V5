import math
from typing import Any, Dict, List, Tuple
from collections import Counter, defaultdict

from utils import get_conn, get_env_var

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
    toks = (s or "").lower().split()
    if not toks: return {}
    c = Counter(toks)
    tot = float(sum(c.values()))
    return {t: v / tot for t, v in c.items()} if tot > 0 else {}

def _cosine(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v*v for v in a.values())) or 1.0
    nb = math.sqrt(sum(v*v for v in b.values())) or 1.0
    return num / (na * nb)

# ---------------------------
# URL-aware MMR
# ---------------------------
def mmr_select_url_aware(scored: List[Any], q_vec=None, k: int = 10, lambda_: float = MMR_LAMBDA) -> List[Any]:
    n = len(scored)
    if n == 0 or k <= 0:
        return []

    k = min(k, n)
    idx = list(range(n))
    idx.sort(key=lambda i: float(scored[i].get("score", 0.0)), reverse=True)
    selected = [idx[0]]
    avail = idx[1:]

    vecs = [c.get("tfidf") for c in scored]
    rels = [c.get("score", 0.0) for c in scored]
    urls = [c.get("meta", {}).get("url") for c in scored]

    url_count = Counter(urls)

    while avail and len(selected) < k:
        best_i, best_val = None, -1e18
        for i in avail:
            rel = rels[i]
            div = 0.0
            if vecs[i]:
                sims = []
                for j in selected:
                    sim = _cosine(vecs[i], vecs[j])
                    if urls[i] and urls[j] and urls[i] == urls[j]:
                        factor = 0.2 + 0.1 / max(url_count[urls[i]], 1)
                        sim *= factor
                    sims.append(sim)
                div = max(sims) if sims else 0.0

            val = lambda_ * rel - (1.0 - lambda_) * div
            if val > best_val:
                best_val, best_i = val, i
        selected.append(best_i)
        avail.remove(best_i)

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

# ---------------------------
# Hybrid retrieval with safe URL merge
# ---------------------------
def hybrid_retrieve_pg(query: str, top_k: int = 5, mmr_lambda: float = 0.5) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Retrieve relevant chunks for a query, merge by URL, and rank by score.
    
    Returns a list of (text, meta) tuples.
    """
    # Step 1: Retrieve top candidate chunks
    # Replace `vectorstore.similarity_search_with_score` with your retrieval call
    retrieved_chunks: List[Dict[str, Any]] = vectorstore.similarity_search_with_score(query, k=top_k*2)
    
    # Step 2: Merge chunks by URL
    merged_by_url: Dict[str, Dict[str, Any]] = {}
    
    for chunk, score in retrieved_chunks:
        url = chunk["metadata"]["url"]
        if url in merged_by_url:
            # Append text and update score to max
            merged_by_url[url]["text"] += "\n\n" + chunk["text"]
            merged_by_url[url]["score"] = max(merged_by_url[url]["score"], score)
        else:
            merged_by_url[url] = {
                "text": chunk["text"],
                "meta": {**chunk["metadata"], "score": score},
                "score": score
            }

    # Step 3: Apply MMR on merged documents (optional)
    # Here you can integrate your MMR ranking function if desired
    # For simplicity, we sort by score descending
    final = sorted(merged_by_url.values(), key=lambda x: x["score"], reverse=True)

    # Step 4: Format output
    out: List[Tuple[str, Dict[str, Any]]] = [(v["text"], v["meta"]) for v in final]

    # Debug info
    print(f"[DEBUG] Final merged results count: {len(out)}")
    for o in out[:5]:
        print(f"   url={o[1]['url']}  score={o[1]['score']:.4f}  snippet='{o[0][:80]}'")

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
