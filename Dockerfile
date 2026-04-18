# ── Build stage ───────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps for psycopg2-binary
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py generator.py retrieval.py utils.py ./
COPY pages/ pages/ 2>/dev/null || true

# ── Runtime ───────────────────────────────────────────────────
EXPOSE 8501

# Streamlit health check — built-in /_stcore/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
