# DIGIT Studio Assistant

A domain-specific support chatbot for DIGIT Studio documentation.

## How it works

```
User query
    ↓
Check predetermined Q&A cache (instant, no API cost)
    ↓ (if not found)
Rewrite query → RAG retrieval → Stream answer
    ↓
User gives 👍/👎 feedback
    ↓
👍 on RAG answer → promoted to cache with 70% confidence
Repeated 👍 → confidence rises → served instantly next time
```

## Quick start (local)

**Prerequisites:** Python 3.11+, a Neon PostgreSQL database with pgvector enabled, OpenAI API key, Cohere API key (optional, for reranking).

```bash
git clone https://github.com/srujana-egov/EGOV_RAG_V5.git
cd EGOV_RAG_V5

python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt

cp .env.example .env
# Edit .env and fill in your actual credentials
```

Then run the bot:
```bash
streamlit run app.py
```

Or run the ingestion console:
```bash
streamlit run ingest_console.py
```

## Environment variables

Copy `.env.example` to `.env` and fill in:

| Variable | Description |
|---|---|
| `PGUSER` | Neon DB username |
| `PGPASSWORD` | Neon DB password |
| `PGHOST` | Neon DB host |
| `PGDATABASE` | Database name |
| `OPENAI_API_KEY` | OpenAI API key |
| `COHERE_API_KEY` | Cohere API key (for reranking) |
| `DB_TABLE` | Vector table name (default: `studio_manual`) |
| `ADMIN_PASSWORD` | Password to unlock admin dashboard in the bot |

## Adding content

### Option 1 — Ingestion console (recommended)
```bash
streamlit run ingest_console.py
```
Upload PDFs, CSVs, or Excel files. Preview chunks. Ingest directly to the database or download JSONL.

### Option 2 — Admin panel in the bot
Open the bot → enter admin password in sidebar → "Add Content" tab → paste text directly.

### Option 3 — JSONL ingest script
```bash
python ingest_simple.py --file data/studio_chunks.jsonl
```

## Changing to a different use case

To point this bot at a different documentation set:

1. Update `DB_TABLE` in `.env` (e.g. `health_manual`, `payments_manual`)
2. Ingest new docs via the ingestion console
3. Update the system prompt in `generator.py` (`SYSTEM_PROMPT`)
4. Update suggested questions in `app.py`
5. Update branding (title, logo) in `app.py`

## Architecture

```
app.py              — Streamlit UI (chat + admin tabs)
generator.py        — Query rewriting + OpenAI streaming
retrieval.py        — Vector search + Cohere reranking + MMR
utils.py            — DB pool, feedback logging, helpers
ingest_console.py   — Streamlit UI for document ingestion
ingest_simple.py    — CLI ingestion script
```

## Database tables

| Table | Purpose |
|---|---|
| `studio_manual` | Vector embeddings of documentation chunks |
| `predetermined_qa` | Cached Q&A pairs with confidence scores |
| `bot_feedback` | All user feedback (👍/👎) |

## Admin dashboard

Set `ADMIN_PASSWORD` in `.env`. In the sidebar, enter the password to unlock:
- **Queries & Feedback** — see all queries, ratings, satisfaction %
- **Q&A Cache** — view/add/delete predetermined answers
- **Add Content** — paste new documentation directly
