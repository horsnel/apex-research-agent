# ── Stage 1: Builder ──
FROM python:3.11-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ──
FROM python:3.11-slim

LABEL maintainer="APEX Research Agent"
LABEL description="Token-efficient hybrid RAG + Live Scraper research AI"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages
COPY --from=builder /install /usr/local

# Create non-root user
RUN groupadd -r apex && useradd -r -g apex -d /app -s /sbin/nologin apex

WORKDIR /app

# Copy application code
COPY --chown=apex:apex . .

# Create data directory
RUN mkdir -p /app/data && chown apex:apex /app/data

# Switch to non-root user
USER apex

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Railway injects PORT env var — default to 8000 for local dev
ENV PORT=8000

# Expose the PORT (Railway uses this to route traffic)
EXPOSE ${PORT}

# Default command: use PORT env var (Railway injects this)
# Railway auto-assigns a port and proxies all traffic to it
CMD uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}
