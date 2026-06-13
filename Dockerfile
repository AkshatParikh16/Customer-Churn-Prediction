# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11 AS builder

# Install uv (modern Python package manager)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Copy dependency files first (Docker layer cache)
COPY pyproject.toml uv.lock ./
COPY src/ ./src/
COPY configs/ ./configs/

# Install only production deps into /app/.venv
RUN uv sync --no-dev

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd -r churn && useradd -r -g churn churn

WORKDIR /app

# Copy venv from builder (no pip/uv needed in final image)
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/configs ./configs

# Models are mounted at runtime via Docker volume (-v /opt/churn/models:/app/models:ro)
RUN mkdir -p models

# Runtime env
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    API_ENVIRONMENT=production \
    API_PORT=8000

# Logs dir (writable by churn user)
RUN mkdir -p logs && chown -R churn:churn /app

USER churn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "churn.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
