# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build tools needed by some packages (uvloop, aiohttp, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy only what's needed for install
COPY requirements-prod.txt ./
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml ./

# Install production dependencies into a prefix we can copy over
RUN pip install --no-cache-dir --prefix=/install -r requirements-prod.txt && \
    pip install --no-cache-dir --prefix=/install --no-deps -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r churn && useradd -r -g churn churn

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Models are mounted at runtime via Docker volume (-v /opt/churn/models:/app/models:ro)
RUN mkdir -p models logs && chown -R churn:churn /app

# Runtime env
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_ENVIRONMENT=production \
    API_PORT=8000

USER churn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "churn.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
