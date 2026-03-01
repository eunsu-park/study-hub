# Multi-stage Docker build for a Python Flask application
# This demonstrates best practices for creating slim, secure production images

# ============================================================================
# Stage 1: Builder
# Purpose: Install dependencies and build the application
# ============================================================================
# Why: slim variant (~120MB) instead of full (~900MB) — avoids shipping hundreds of unused Debian packages in the final image
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
# These are only needed during the build process
# Why: --no-install-recommends prevents pulling in transitive packages that bloat the layer; rm of apt lists reclaims ~30MB
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
# If requirements.txt doesn't change, this layer is cached
# Why: Separating dependency install from code COPY means code changes don't invalidate the (slow) pip install layer
COPY requirements.txt .

# Install Python dependencies to a local directory
# This allows us to copy only what's needed to the final stage
# Why: --user installs to ~/.local so we can cherry-pick just the packages into the runtime stage, leaving pip/wheel/setuptools behind
RUN pip install --user --no-cache-dir -r requirements.txt

# ============================================================================
# Stage 2: Runtime
# Purpose: Create a minimal production image
# ============================================================================
FROM python:3.11-slim

# Metadata labels following OCI standard
LABEL maintainer="your-email@example.com"
LABEL description="Flask application with multi-stage build"
LABEL version="1.0"

# Create a non-root user for security
# Running as root in containers is a security risk
# Why: UID 1000 is chosen to match the default host user, avoiding permission issues on bind-mounted volumes
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy Python packages from builder stage
# This includes all installed dependencies without build tools
COPY --from=builder --chown=appuser:appuser /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser app.py .

# Update PATH to include local Python packages
ENV PATH=/home/appuser/.local/bin:$PATH

# Switch to non-root user
USER appuser

# Expose port (documentation only, doesn't publish the port)
EXPOSE 5000

# Health check to verify container is running correctly
# This is used by orchestrators like Kubernetes
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Why: Exec form (JSON array) runs python as PID 1, so Docker's stop signal (SIGTERM) goes directly to the app instead of being swallowed by a shell wrapper
CMD ["python", "app.py"]
