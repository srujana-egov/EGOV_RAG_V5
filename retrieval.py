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
TITLE_COL = "title"  # << added so we can use title field
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
    
    # ✅ Include title+url context in the embedding
    emb_input = f"QUERY CONTEXT:\n{query}"
    qvec = get_embedding(emb_input)

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        sql_vec = f"""
          SELECT
            {ID_COL} AS id, {TITLE_COL} AS title, {TXT_COL} AS document, {URL_COL}, {TAG_COL}, {VER_COL},
            (1.0 - ({EMB_COL} <=> %s::vector))::float AS score
          FROM {TABLE}
          ORDER BY {EMB_COL} <=> %s::vector
          LIMIT %s
        """
        cur.execute(sql_vec, [qvec, qvec, need])
        rows = cur.fetchall()
        for r in rows:
            # ✅ merge title + url + content into a single logical text for downstream
            title = (r.get("title") or "").strip()
            url = (r.get("url") or "").strip()
            doc = (r.get("document") or "").strip()
            combined_text = f"TITLE: {title}\nURL: {url}\nCONTENT:\n{doc}"
            r["document"] = combined_text
            r["metadata"] = {
                "url": url,
                "tag": r.get("tag"),
                "version": r.get("version"),
            }
    return rows

# ---------------------------
# Hybrid retrieval with safe URL merge
# ---------------------------
def hybrid_retrieve_pg(query: str, top_k: int = 20, mmr_lambda: float = MMR_LAMBDA) -> List[Tuple[str, Dict[str, Any]]]:
    conn = get_conn()
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

    # prepare candidates
    cands: List[Dict[str, Any]] = []
    for r in base:
        text = r.get("document") or ""
        meta_raw = r.get("metadata", {})
        meta = {
            "id": str(r.get("id")) if r.get("id") else None,
            "url": meta_raw.get("url"),
            "score": float(r.get("score") or 0.0),
            "tfidf": _tf_dict(text),
        }
        cands.append({"text": text, "score": meta["score"], "tfidf": meta["tfidf"], "meta": meta})

    q_vec = _tf_dict(query)
    selected = mmr_select_url_aware(cands, q_vec=q_vec, k=max(top_k, 5), lambda_=mmr_lambda)

    # merge chunks safely by normalized URL
    merged_by_url: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"text": "", "meta": None, "score": 0.0})
    for s in selected:
        raw_url = s["meta"]["url"] or s["meta"]["id"] or ""
        url = raw_url.strip().lower().rstrip("/")
        print(f"[DEBUG] Merging chunk id={s['meta']['id']} url={url} score={s['score']:.4f}")  # << ADDED DEBUG
        if merged_by_url[url]["text"]:
            merged_by_url[url]["text"] += "\n" + s["text"]
            merged_by_url[url]["score"] = max(merged_by_url[url]["score"], s["score"])
        else:
            merged_by_url[url]["text"] = s["text"]
            merged_by_url[url]["meta"] = s["meta"]
            merged_by_url[url]["score"] = s["score"]

    print("[DEBUG] merged_by_url keys:", list(merged_by_url.keys()))

    # final sorted output, keep all merged chunks
    final = sorted(merged_by_url.values(), key=lambda x: x["score"], reverse=True)
    out: List[Tuple[str, Dict[str, Any]]] = [(v["text"], v["meta"]) for v in final]

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
