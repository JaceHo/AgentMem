FROM python:3.12-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model so the container starts fast
# Must match LocalProvider._MODEL_NAME in core/embedder.py
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .

# Create non-root user for production security
RUN groupadd -r agentmem && useradd -r -g agentmem -d /app -s /sbin/nologin agentmem \
    && mkdir -p /home/agentmem/.agentmem \
    && chown -R agentmem:agentmem /app /home/agentmem

ENV AGENTMEM_HOST=0.0.0.0 \
    AGENTMEM_PORT=18800 \
    REDIS_URL=redis://redis:6379

EXPOSE 18800

USER agentmem

HEALTHCHECK --interval=30s --timeout=5s --retries=3 --start-period=10s \
    CMD curl -sf http://localhost:18800/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "18800", \
     "--log-level", "info", "--timeout-graceful-shutdown", "30", \
     "--workers", "1", "--loop", "uvloop", "--http", "httptools"]
