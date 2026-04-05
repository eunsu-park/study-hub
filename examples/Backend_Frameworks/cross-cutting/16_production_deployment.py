"""
Production Deployment — Docker, Reverse Proxy, Health Checks
Demonstrates: Dockerfile generation, health/readiness endpoints,
              graceful shutdown, and nginx config generation.

Run: pip install fastapi uvicorn && uvicorn 16_production_deployment:app --reload
"""

from fastapi import FastAPI
from contextlib import asynccontextmanager
import signal
import time
import os

# --- 1. Application State for Health Checks ---

app_state = {"ready": False, "start_time": 0.0, "shutting_down": False}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle hooks."""
    # Startup: simulate warm-up (DB pool, cache preload)
    app_state["start_time"] = time.time()
    app_state["ready"] = True
    print("[lifecycle] Application started")
    yield
    # Shutdown: graceful drain
    app_state["shutting_down"] = True
    print("[lifecycle] Graceful shutdown initiated")


app = FastAPI(title="Production App", version="1.0.0", lifespan=lifespan)


# --- 2. Health & Readiness Probes ---

@app.get("/healthz")
async def health():
    """Liveness probe: is the process alive?"""
    if app_state["shutting_down"]:
        return {"status": "draining"}, 503
    return {"status": "healthy", "uptime_seconds": round(time.time() - app_state["start_time"], 1)}


@app.get("/readyz")
async def readiness():
    """Readiness probe: can the service accept traffic?"""
    if not app_state["ready"]:
        return {"status": "not ready"}, 503
    return {"status": "ready"}


@app.get("/")
async def root():
    return {"service": "production-app", "env": os.getenv("APP_ENV", "development")}


# --- 3. Dockerfile Generator ---

def generate_dockerfile(
    python_version: str = "3.12",
    app_module: str = "main:app",
    port: int = 8000,
    workers: int = 4,
) -> str:
    """Generate a production-grade multi-stage Dockerfile."""
    return f"""\
# --- Build stage ---
FROM python:{python_version}-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --target=/deps -r requirements.txt

# --- Runtime stage ---
FROM python:{python_version}-slim
WORKDIR /app

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /deps /usr/local/lib/python{python_version}/site-packages
COPY . .

USER app
EXPOSE {port}

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:{port}/healthz')"

CMD ["uvicorn", "{app_module}", "--host", "0.0.0.0", "--port", "{port}", "--workers", "{workers}"]
"""


# --- 4. Nginx Reverse Proxy Config ---

def generate_nginx_config(
    server_name: str = "api.example.com",
    upstream_port: int = 8000,
    ssl: bool = True,
) -> str:
    """Generate an nginx reverse proxy configuration."""
    ssl_block = ""
    if ssl:
        ssl_block = """
    listen 443 ssl;
    ssl_certificate /etc/letsencrypt/live/{server}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/{server}/privkey.pem;
""".format(server=server_name)

    return f"""\
upstream backend {{
    server 127.0.0.1:{upstream_port};
}}

server {{
    listen 80;{ssl_block}
    server_name {server_name};

    location / {{
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }}

    location /healthz {{
        proxy_pass http://backend/healthz;
        access_log off;
    }}
}}
"""


# --- 5. Docker Compose Generator ---

def generate_compose(services: list[dict]) -> str:
    """Generate docker-compose.yaml from service definitions."""
    lines = ["version: '3.8'", "services:"]
    for svc in services:
        lines.append(f"  {svc['name']}:")
        lines.append(f"    build: {svc.get('build', '.')}")
        if "ports" in svc:
            lines.append("    ports:")
            for p in svc["ports"]:
                lines.append(f'      - "{p}"')
        if "env" in svc:
            lines.append("    environment:")
            for k, v in svc["env"].items():
                lines.append(f"      {k}: {v}")
        if svc.get("healthcheck"):
            lines.append("    healthcheck:")
            lines.append(f"      test: {svc['healthcheck']}")
            lines.append("      interval: 30s")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print("=== Dockerfile ===")
    print(generate_dockerfile())
    print("=== Nginx Config ===")
    print(generate_nginx_config())
    print("=== Docker Compose ===")
    print(generate_compose([
        {"name": "api", "ports": ["8000:8000"], "env": {"APP_ENV": "production"},
         "healthcheck": '["CMD", "curl", "-f", "http://localhost:8000/healthz"]'},
        {"name": "redis", "build": "redis:7-alpine", "ports": ["6379:6379"]},
    ]))
