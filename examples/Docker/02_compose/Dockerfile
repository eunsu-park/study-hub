# Production-ready Dockerfile for Flask application
# Used by docker-compose.yml

# Why: Unlike 01_multi_stage, this uses a single stage because Compose rebuilds are frequent during development — the simplicity trade-off outweighs the ~50MB size difference
FROM python:3.11-slim

LABEL maintainer="your-email@example.com"
LABEL description="Flask app with PostgreSQL and Redis"

# Install system dependencies
# Why: libpq-dev must stay in the runtime image because psycopg2 dynamically links to libpq at runtime (unlike psycopg2-binary which bundles it)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

WORKDIR /app

# Why: --no-cache-dir avoids storing pip's download cache in the image layer — saves ~20-50MB since containers never re-run pip install
# Copy and install dependencies
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=appuser:appuser app.py .

# Why: Switching to non-root here means that even if the app has an RCE vulnerability, the attacker cannot install packages, modify system files, or escalate privileges
USER appuser

EXPOSE 5000

# Why: start-period is 40s (longer than 01_multi_stage's 5s) because this app connects to Postgres and Redis on startup, which may take time in a Compose environment
HEALTHCHECK --interval=30s --timeout=3s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

CMD ["python", "app.py"]
