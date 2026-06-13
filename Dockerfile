FROM python:3.11-slim

# Build tools for C-extension packages (uvloop, aiohttp, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install production dependencies
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy source and config
COPY src/ ./src/
COPY configs/ ./configs/

# Empty models dir — filled at runtime via volume mount
RUN mkdir -p models logs

# Non-root user
RUN groupadd -r churn && useradd -r -g churn churn && \
    chown -R churn:churn /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
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
