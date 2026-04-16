"""
Test all predetermined questions against the Q&A cache matching logic.
Reports: CACHE HIT / MISS / coverage score for each question.

Run: python test_qa_matching.py
"""

import re
import sys
from dotenv import load_dotenv
load_dotenv()

from utils import get_conn

# ── Same stop words and matching logic as app.py ──
_STOP_WORDS = {
    "what", "when", "where", "which", "that", "this", "with", "from",
    "have", "does", "will", "can", "how", "why", "who", "are", "the",
    "and", "for", "not", "your", "my", "do", "is", "in", "an", "a",
    "to", "of", "on", "at", "by", "or", "be", "it", "as", "up",
    "about", "into", "after", "before", "during", "while", "there",
}

_WEAK_WORDS = {"create", "new", "make", "add", "get", "set", "use", "see",
               "view", "edit", "show", "find", "list", "all", "any"}


def _stem(word: str) -> str:
    for suffix in ("ings", "ing", "tion", "tions", "ed", "ers", "er", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def _tokenize(text: str) -> set:
    """Extract stemmed, meaningful tokens — mirrors app.py _tokenize."""
    return set(
        _stem(w) for w in re.findall(r'[a-z0-9]+', text.strip().lower())
        if len(w) >= 3 and w not in _STOP_WORDS
    )


def get_predetermined_answer(query: str, rows):
    q = query.strip().lower()
    q_words = _tokenize(q)
    if not q_words:
        return None, 0.0, None

    q_strong = q_words - _WEAK_WORDS

    best_match = None
    best_score = 0.0
    best_question = None

    for row_id, question, answer, confidence in rows:
        p = question.strip().lower()
        if q == p:
            return answer, 1.0, question

        p_words = _tokenize(p)
        if not p_words:
            continue

        common = q_words & p_words
        if not common:
            continue

        jaccard = len(common) / len(q_words | p_words)

        strong_common = q_strong & p_words
        if q_strong and strong_common:
            score = jaccard + 0.2 * (len(strong_common) / len(q_strong))
        else:
            score = jaccard * 0.5

        if score >= 0.25 and score > best_score:
            best_score = score
            best_match = answer
            best_question = question

    return best_match, best_score, best_question


# ── All 40 test questions ──
TEST_QUESTIONS = [
    "What environments are available (dev, staging, prod)?",
    "What can I build using DIGIT Studio?",
    "Can I build XYZ using DIGIT Studio?",
    "How is DIGIT Studio different from traditional low-code/no-code tools?",
    'What is a "Service" in DIGIT Studio?',
    'What is a "module" in DIGIT Studio?',
    "What is the difference between a module and a Service?",
    "What is the difference between a checklist and a form?",
    "How do I preview changes before publishing?",
    "Are apps generated on Studio responsive?",
    "Can I build mobile apps on Studio?",
    "Can I edit my app after publishing?",
    "How do I test workflows before going live?",
    "How will citizens see the services I created?",
    "What is the default citizen role?",
    "How do I assign permissions to roles?",
    "Can dropdown values be used across apps?",
    "How do I create a dependent dropdown?",
    "What is SLA?",
    "Where has this been used in the past?",
    "What use cases can DIGIT Studio support?",
    "How do I configure approvals and rejections?",
    "Can I have multiple languages in an app?",
    "Are applications built on Studio accessible?",
    "How do I add users?",
    "What are the steps to go live?",
    "How do I integrate with SMS/email services?",
    "Can I roll back a deployment?",
    "Can I build grievance applications?",
    "How is authentication handled?",
    "How do I configure notifications?",
    "Can I use my own domain when going live?",
    "How do I deploy my application?",
    "Can I import/export data?",
    "Can workflows integrate with external systems?",
    "How do I create a new UI screen?",
    "How do I change the look and feel of my UI?",
    "How does DIGIT Studio work with DIGIT?",
    "What are states and actions in workflows?",
    "What is the downtime during deployment?",
    "How do I assign roles to workflow steps?",
    "How does role-based access work?",
    "Where is the data stored?",
    "Can I integrate with existing identity systems?",
    "Can I build permit and license applications?",
]


def run_tests():
    print("Loading Q&A cache from database...")
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, question, answer, confidence FROM predetermined_qa")
            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"Loaded {len(rows)} Q&A entries from cache.\n")
    print("=" * 70)
    print(f"{'#':<4} {'RESULT':<10} {'SCORE':<8} QUESTION")
    print("=" * 70)

    hits = 0
    misses = []

    for i, question in enumerate(TEST_QUESTIONS, 1):
        answer, score, matched_q = get_predetermined_answer(question, rows)

        if answer:
            hits += 1
            score_str = f"{score:.0%}"
            # Truncate matched question for display
            mq = (matched_q[:45] + "...") if matched_q and len(matched_q) > 48 else matched_q
            print(f"{i:<4} {'✅ HIT':<10} {score_str:<8} {question[:55]}")
            if matched_q and matched_q.strip().lower() != question.strip().lower():
                print(f"     {'':10} {'':8} └─ matched: \"{mq}\"")
        else:
            misses.append((i, question))
            print(f"{i:<4} {'❌ MISS':<10} {'–':<8} {question[:55]}")

    print("=" * 70)
    print(f"\nResults: {hits}/{len(TEST_QUESTIONS)} cache hits")

    if misses:
        print(f"\n⚠️  {len(misses)} question(s) not matching — would fall through to RAG:")
        for idx, q in misses:
            print(f"  [{idx}] {q}")
        print(
            "\nTo fix misses: add a Q&A entry in setup_studio_data.py whose 'question' "
            "shares ≥50% key words with the query, then re-run setup_studio_data.py."
        )
    else:
        print("\n🎉 All questions matched the cache — no RAG calls needed!")

    return hits, len(TEST_QUESTIONS), misses


if __name__ == "__main__":
    try:
        hits, total, misses = run_tests()
        sys.exit(0 if not misses else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure your .env DB credentials are set and setup_studio_data.py has been run.")
        sys.exit(2)
