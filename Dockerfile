# ─────────────────────────────────────────────────────────────
# YouTube Pipeline — LLM Service Dockerfile
# ─────────────────────────────────────────────────────────────
# Multi-stage build:
#   Stage 1 (builder) — installs dependencies
#   Stage 2 (runtime) — lean final image, no build tools
#
# Build:  docker build -t llm-service .
# Run:    docker compose up -d --build
# ─────────────────────────────────────────────────────────────


# ── Stage 1: Builder ──────────────────────────────────────────
FROM python:3.11-slim AS builder

# Prevents Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Prevents Python from buffering stdout/stderr
# (logs appear immediately in docker logs)
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install dependencies into a separate directory
# so we can copy only them into the runtime stage
COPY requirements.txt .
RUN pip install --upgrade pip --quiet && \
    pip install --prefix=/install \
        --no-cache-dir \
        --quiet \
        -r requirements.txt


# ── Stage 2: Runtime ──────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local

# Copy application code
# .dockerignore excludes .env, __pycache__, .git etc
COPY app/ ./app/

# Create non-root user for security
# Never run containers as root
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser
USER appuser

# Expose the service port
# Must match PORT in .env and docker-compose ports mapping
EXPOSE 8001

# Health check — Docker will restart container if this fails
# Interval: check every 30s
# Timeout: fail if no response in 10s
# Retries: mark unhealthy after 3 consecutive failures
# Start period: give app 15s to start before checking
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --retries=3 \
    --start-period=15s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" \
    || exit 1

# Start the service
# --host 0.0.0.0 makes it accessible from outside the container
# --workers 1 is correct for async FastAPI (not sync Django/Flask)
# --loop uvloop for faster async performance
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8001", \
     "--workers", "1", \
     "--loop", "uvloop", \
     "--log-level", "warning"]