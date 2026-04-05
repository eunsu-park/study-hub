#!/usr/bin/env python3
"""Example: Platform Engineering — Internal Developer Platform Primitives

Demonstrates platform engineering concepts: self-service project scaffolding,
environment provisioning, golden-path templates, developer portal catalog,
and platform API abstraction over infrastructure.
Related lesson: 17_Platform_Engineering.md
"""

# =============================================================================
# WHY PLATFORM ENGINEERING?
# Platform engineering builds an Internal Developer Platform (IDP) that
# provides golden paths for common workflows (create service, deploy, add
# database) so developers get self-service capabilities with built-in
# guardrails, reducing cognitive load and time-to-production.
# =============================================================================

import json
import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


# =============================================================================
# 1. SERVICE CATALOG
# =============================================================================

class ServiceTier(Enum):
    CRITICAL = "critical"    # 99.99% SLO, pager
    STANDARD = "standard"    # 99.9% SLO
    BEST_EFFORT = "best-effort"  # No SLO commitment


@dataclass
class ServiceSpec:
    """A registered service in the platform catalog (Backstage-style)."""
    name: str
    owner: str
    description: str
    tier: ServiceTier = ServiceTier.STANDARD
    language: str = "python"
    dependencies: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    repo_url: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ServiceCatalog:
    """Registry of all services in the organization."""
    services: dict[str, ServiceSpec] = field(default_factory=dict)

    def register(self, spec: ServiceSpec) -> str:
        if spec.name in self.services:
            raise ValueError(f"Service '{spec.name}' already registered")
        self.services[spec.name] = spec
        return f"Registered '{spec.name}' (owner={spec.owner}, tier={spec.tier.value})"

    def get(self, name: str) -> Optional[ServiceSpec]:
        return self.services.get(name)

    def find_by_owner(self, owner: str) -> list[ServiceSpec]:
        return [s for s in self.services.values() if s.owner == owner]

    def dependency_graph(self) -> dict[str, list[str]]:
        """Return the service dependency graph."""
        return {s.name: s.dependencies for s in self.services.values()}


# =============================================================================
# 2. GOLDEN-PATH TEMPLATES
# =============================================================================

@dataclass
class Template:
    """A scaffolding template for new services."""
    name: str
    description: str
    language: str
    files: dict[str, str]  # filename -> content template
    parameters: list[str] = field(default_factory=list)


# Standard templates available to developers
TEMPLATES: dict[str, Template] = {
    "python-fastapi": Template(
        name="python-fastapi",
        description="Python FastAPI microservice with health checks and metrics",
        language="python",
        parameters=["service_name", "owner", "port"],
        files={
            "Dockerfile": (
                "FROM python:3.12-slim\n"
                "WORKDIR /app\nCOPY . .\n"
                "RUN pip install -r requirements.txt\n"
                "CMD [\"uvicorn\", \"main:app\", \"--host\", \"0.0.0.0\", "
                "\"--port\", \"{port}\"]"
            ),
            "main.py": (
                "from fastapi import FastAPI\n\n"
                "app = FastAPI(title=\"{service_name}\")\n\n"
                "@app.get(\"/healthz\")\ndef health(): return {{\"status\": \"ok\"}}\n"
            ),
            "k8s/deployment.yaml": (
                "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n"
                "  name: {service_name}\n  labels:\n    app: {service_name}\n"
                "    owner: {owner}\nspec:\n  replicas: 2\n"
            ),
            ".github/workflows/ci.yml": (
                "name: CI\non: [push]\njobs:\n  test:\n"
                "    runs-on: ubuntu-latest\n    steps:\n"
                "      - uses: actions/checkout@v4\n"
                "      - run: pytest\n"
            ),
        },
    ),
    "go-grpc": Template(
        name="go-grpc",
        description="Go gRPC service with structured logging",
        language="go",
        parameters=["service_name", "owner", "port"],
        files={
            "Dockerfile": (
                "FROM golang:1.22 AS build\nWORKDIR /app\nCOPY . .\n"
                "RUN go build -o /server ./cmd/server\n"
                "FROM gcr.io/distroless/base\nCOPY --from=build /server /server\n"
                "CMD [\"/server\"]"
            ),
            "cmd/server/main.go": (
                "package main\n\nimport \"fmt\"\n\nfunc main() {{\n"
                "    fmt.Println(\"{service_name} starting on :{port}\")\n}}"
            ),
        },
    ),
}


def scaffold_service(template_name: str, params: dict[str, str]) -> dict[str, str]:
    """Generate files from a golden-path template."""
    tmpl = TEMPLATES.get(template_name)
    if not tmpl:
        raise ValueError(f"Unknown template: {template_name}")
    # Validate required parameters
    missing = [p for p in tmpl.parameters if p not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}")
    # Render files
    rendered: dict[str, str] = {}
    for filename, content in tmpl.files.items():
        rendered[filename] = content.format(**params)
    return rendered


# =============================================================================
# 3. ENVIRONMENT PROVISIONING
# =============================================================================

class EnvType(Enum):
    DEV = "dev"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class Environment:
    """A provisioned environment for a service."""
    name: str
    env_type: EnvType
    service: str
    namespace: str = ""
    url: str = ""
    resources: dict[str, Any] = field(default_factory=dict)
    status: str = "provisioning"

    def __post_init__(self):
        self.namespace = f"{self.service}-{self.env_type.value}"
        self.url = f"https://{self.service}.{self.env_type.value}.internal"


@dataclass
class EnvironmentManager:
    """Manages environment lifecycle for services."""
    environments: dict[str, Environment] = field(default_factory=dict)
    provisioning_log: list[str] = field(default_factory=list)

    # Resource defaults by environment type
    RESOURCE_DEFAULTS: dict = field(default_factory=lambda: {
        EnvType.DEV: {"cpu": "500m", "memory": "512Mi", "replicas": 1},
        EnvType.STAGING: {"cpu": "1000m", "memory": "1Gi", "replicas": 2},
        EnvType.PRODUCTION: {"cpu": "2000m", "memory": "4Gi", "replicas": 3},
    })

    def provision(self, service: str, env_type: EnvType) -> Environment:
        key = f"{service}-{env_type.value}"
        if key in self.environments:
            raise ValueError(f"Environment {key} already exists")
        env = Environment(
            name=key,
            env_type=env_type,
            service=service,
            resources=copy.deepcopy(self.RESOURCE_DEFAULTS.get(env_type, {})),
            status="ready",
        )
        self.environments[key] = env
        self.provisioning_log.append(
            f"Provisioned {key}: ns={env.namespace}, url={env.url}"
        )
        return env

    def teardown(self, service: str, env_type: EnvType) -> str:
        key = f"{service}-{env_type.value}"
        if key not in self.environments:
            return f"Environment {key} not found"
        del self.environments[key]
        return f"Torn down {key}"


# =============================================================================
# 4. PLATFORM SCORECARD
# =============================================================================

def compute_scorecard(spec: ServiceSpec, catalog: ServiceCatalog) -> dict[str, Any]:
    """Compute a platform readiness scorecard for a service."""
    checks = {
        "has_owner": bool(spec.owner),
        "has_description": len(spec.description) > 10,
        "has_repo": bool(spec.repo_url),
        "tier_defined": spec.tier != ServiceTier.BEST_EFFORT,
        "dependencies_registered": all(
            d in catalog.services for d in spec.dependencies
        ),
        "has_tags": len(spec.tags) >= 2,
    }
    score = sum(checks.values()) / len(checks) * 100
    return {"service": spec.name, "score": round(score, 1), "checks": checks}


# =============================================================================
# 5. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Service Catalog ---
    print("=" * 60)
    print("Service Catalog")
    print("=" * 60)
    catalog = ServiceCatalog()
    catalog.register(ServiceSpec(
        name="payment-api", owner="team-payments",
        description="Handles payment processing and invoicing",
        tier=ServiceTier.CRITICAL, language="python",
        tags=["payments", "api"], repo_url="https://github.com/org/payment-api",
    ))
    catalog.register(ServiceSpec(
        name="order-svc", owner="team-orders",
        description="Order management service",
        dependencies=["payment-api"],
        tags=["orders", "backend"],
    ))
    catalog.register(ServiceSpec(
        name="notification-worker", owner="team-platform",
        description="Async notification dispatcher",
        tags=["notifications"],
    ))
    for name, svc in catalog.services.items():
        print(f"  {name}: owner={svc.owner}, tier={svc.tier.value}")
    print(f"  Dependency graph: {catalog.dependency_graph()}")

    # --- Golden Path Scaffolding ---
    print(f"\n{'=' * 60}")
    print("Golden Path Scaffolding")
    print("=" * 60)
    files = scaffold_service("python-fastapi", {
        "service_name": "user-profile",
        "owner": "team-identity",
        "port": "8080",
    })
    for fname in files:
        print(f"  Generated: {fname}")

    # --- Environment Provisioning ---
    print(f"\n{'=' * 60}")
    print("Environment Provisioning")
    print("=" * 60)
    em = EnvironmentManager()
    for env_type in [EnvType.DEV, EnvType.STAGING, EnvType.PRODUCTION]:
        env = em.provision("payment-api", env_type)
        print(f"  {env.name}: url={env.url}, replicas={env.resources['replicas']}")

    # --- Platform Scorecard ---
    print(f"\n{'=' * 60}")
    print("Platform Scorecard")
    print("=" * 60)
    for svc in catalog.services.values():
        card = compute_scorecard(svc, catalog)
        print(f"  {card['service']}: {card['score']}%")
        for check, passed in card["checks"].items():
            icon = "PASS" if passed else "FAIL"
            print(f"    [{icon}] {check}")
