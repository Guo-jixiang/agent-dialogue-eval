# ── Single stage: Python backend + pre-built frontend ─────────────────────
FROM python:3.11-slim
WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY api/       api/
COPY config/    config/
COPY core/      core/
COPY evaluator/ evaluator/
COPY parser/    parser/
COPY report/    report/
COPY simulator/ simulator/
COPY run_eval.py ./
COPY cli.py      ./

# Copy pre-built frontend (built locally before deploy)
COPY web/dist ./web/dist

# Reports directory (mounted as volume in production)
RUN mkdir -p reports

EXPOSE 8000

# Railway injects $PORT; fall back to 8000 for local/docker-compose usage
CMD ["sh", "-c", "uvicorn api.app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 2"]
