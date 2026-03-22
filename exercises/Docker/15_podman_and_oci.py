"""
Exercises for Lesson 15: Podman and OCI
Topic: Docker

Solutions to practice problems from the lesson.
"""

import json
from dataclasses import dataclass, field
from typing import Optional


# === Exercise 1: Docker-to-Podman Command Translator ===
# Problem: Convert Docker CLI commands to Podman equivalents, flagging
#   unsupported features (Swarm) and noting behavioral differences.

UNSUPPORTED = {
    "swarm": "Use Kubernetes instead of Swarm.",
    "service": "Docker Swarm services not available in Podman.",
    "stack": "Docker Swarm stacks not available in Podman.",
}

NOTES = {
    "compose": "Use 'podman compose' (requires podman-compose).",
    "run": "Podman runs rootless by default; ports < 1024 may need sysctl.",
    "pull": "Podman uses unqualified-search-registries from registries.conf.",
    "build": "Podman uses Buildah under the hood. Same Dockerfile syntax.",
}


def translate(docker_cmd: str) -> tuple[str, str, bool]:
    """Return (translated_cmd, notes, is_supported)."""
    parts = docker_cmd.strip().split()
    if not parts or parts[0] != "docker":
        return docker_cmd, "Not a docker command.", True

    if len(parts) >= 2 and parts[1] in UNSUPPORTED:
        return "", UNSUPPORTED[parts[1]], False

    translated = "podman " + " ".join(parts[1:])
    note = NOTES.get(parts[1], "") if len(parts) >= 2 else ""
    return translated, note, True


def exercise_1():
    """Demonstrate Docker-to-Podman translation."""
    print("=== Exercise 1: Docker-to-Podman Translator ===\n")
    cmds = [
        "docker run -d --name web -p 8080:80 nginx:alpine",
        "docker build -t myapp .",
        "docker compose up -d",
        "docker swarm init",
        "docker pull python:3.12-slim",
        "docker exec -it web sh",
        "docker service create --replicas 3 nginx",
    ]
    for cmd in cmds:
        translated, note, ok = translate(cmd)
        tag = "OK" if ok else "UNSUPPORTED"
        print(f"  [{tag:<11}] {cmd}")
        if ok:
            print(f"              -> {translated}")
        if note:
            print(f"              Note: {note}")


# === Exercise 2: OCI Image Manifest Builder ===
# Problem: Construct a simplified OCI image manifest and multi-arch
#   index to understand the image-spec structure.

@dataclass
class OCILayer:
    digest: str
    size_bytes: int
    media_type: str = "application/vnd.oci.image.layer.v1.tar+gzip"

    def to_dict(self) -> dict:
        return {"mediaType": self.media_type,
                "digest": self.digest, "size": self.size_bytes}


@dataclass
class OCIManifest:
    config_digest: str
    config_size: int
    layers: list[OCILayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "config": {"mediaType": "application/vnd.oci.image.config.v1+json",
                       "digest": self.config_digest, "size": self.config_size},
            "layers": [l.to_dict() for l in self.layers],
        }

    @property
    def total_mb(self) -> float:
        return (self.config_size + sum(l.size_bytes for l in self.layers)
                ) / (1024 * 1024)


def build_index(platforms: list[tuple[str, str, str, int]]) -> dict:
    """Build OCI index from (os, arch, digest, size) tuples."""
    return {
        "schemaVersion": 2,
        "mediaType": "application/vnd.oci.image.index.v1+json",
        "manifests": [
            {"mediaType": "application/vnd.oci.image.manifest.v1+json",
             "digest": d, "size": s,
             "platform": {"os": o, "architecture": a}}
            for o, a, d, s in platforms
        ],
    }


def exercise_2():
    """Demonstrate OCI manifest construction."""
    print("=== Exercise 2: OCI Image Manifest Builder ===\n")
    m = OCIManifest("sha256:config_abc", 1024, [
        OCILayer("sha256:base", 3 * 1024 * 1024),
        OCILayer("sha256:app", 5 * 1024 * 1024),
    ])
    print("  Single-platform manifest:")
    print(f"  {json.dumps(m.to_dict(), indent=2)[:200]}...")
    print(f"  Total: {m.total_mb:.1f} MB\n")

    idx = build_index([
        ("linux", "amd64", "sha256:m_amd64", 528),
        ("linux", "arm64", "sha256:m_arm64", 530),
    ])
    print("  Multi-arch index platforms:")
    for entry in idx["manifests"]:
        p = entry["platform"]
        print(f"    {p['os']}/{p['architecture']} -> {entry['digest']}")


# === Exercise 3: Pod Configuration Builder ===
# Problem: Build a Podman pod with multiple containers sharing network
#   namespace and generate Kubernetes YAML.

@dataclass
class PodContainer:
    name: str
    image: str
    ports: list[int] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    volumes: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Pod:
    name: str
    published: list[tuple[int, int]] = field(default_factory=list)
    containers: list[PodContainer] = field(default_factory=list)

    def cli_commands(self) -> list[str]:
        """Generate podman CLI commands to create this pod."""
        ports = " ".join(f"-p {h}:{c}" for h, c in self.published)
        cmds = [f"podman pod create --name {self.name} {ports}".strip()]
        for c in self.containers:
            parts = [f"podman run -d --pod {self.name} --name {c.name}"]
            for k, v in c.env.items():
                parts.append(f"-e {k}={v}")
            for src, dst in c.volumes:
                parts.append(f"-v {src}:{dst}")
            parts.append(c.image)
            cmds.append(" ".join(parts))
        return cmds

    def kube_yaml(self) -> str:
        """Generate Kubernetes-compatible YAML."""
        lines = ["apiVersion: v1", "kind: Pod", "metadata:",
                 f"  name: {self.name}", "  labels:",
                 f"    app: {self.name}", "spec:", "  containers:"]
        for c in self.containers:
            lines.append(f"    - name: {c.name}")
            lines.append(f"      image: {c.image}")
            if c.ports:
                lines.append("      ports:")
                for p in c.ports:
                    lines.append(f"        - containerPort: {p}")
            if c.env:
                lines.append("      env:")
                for k, v in c.env.items():
                    lines.append(f"        - name: {k}")
                    lines.append(f'          value: "{v}"')
        return "\n".join(lines)


def exercise_3():
    """Demonstrate Podman pod configuration."""
    print("=== Exercise 3: Pod Configuration Builder ===\n")
    pod = Pod("webapp", [(8080, 80), (5432, 5432)])
    pod.containers = [
        PodContainer("web", "nginx:alpine", [80]),
        PodContainer("api", "myapp:latest", [3000],
                     {"DB_URL": "postgresql://localhost:5432/mydb"}),
        PodContainer("db", "postgres:16-alpine", [5432],
                     {"POSTGRES_PASSWORD": "secret"},
                     [("pgdata", "/var/lib/postgresql/data")]),
    ]
    print("  CLI commands:")
    for cmd in pod.cli_commands():
        print(f"    {cmd}")
    print(f"\n  Kubernetes YAML:\n{pod.kube_yaml()}")


# === Exercise 4: Migration Effort Estimator ===
# Problem: Estimate effort for Docker-to-Podman migration based on
#   an inventory of resources.

@dataclass
class Inventory:
    images: list[str] = field(default_factory=list)
    volumes: list[str] = field(default_factory=list)
    compose_files: list[str] = field(default_factory=list)
    swarm_services: list[str] = field(default_factory=list)
    uses_socket: bool = False
    ci_pipelines: int = 0


def estimate_migration(inv: Inventory) -> list[tuple[str, float]]:
    """Return list of (task_description, hours) for migration."""
    tasks = [("Install and verify Podman", 1.0)]
    if inv.swarm_services:
        tasks.append((f"Plan K8s migration for {len(inv.swarm_services)} "
                       f"Swarm services", len(inv.swarm_services) * 4.0))
    if inv.uses_socket:
        tasks.append(("Audit docker.sock dependencies", 3.0))
    tasks.append(("Test all images with Podman",
                   max(1.0, len(inv.images) * 0.5)))
    for vol in inv.volumes:
        tasks.append((f"Migrate volume: {vol}", 0.5))
    for cf in inv.compose_files:
        tasks.append((f"Test compose: {cf}", 1.0))
    if inv.ci_pipelines:
        tasks.append((f"Update {inv.ci_pipelines} CI/CD pipelines",
                       inv.ci_pipelines * 2.0))
    tasks.append(("Remove Docker daemon", 1.0))
    return tasks


def exercise_4():
    """Demonstrate migration effort estimation."""
    print("=== Exercise 4: Migration Effort Estimator ===\n")
    inv = Inventory(
        images=["myapp:latest", "nginx:alpine", "postgres:16"],
        volumes=["pgdata", "uploads"],
        compose_files=["docker-compose.yml"],
        uses_socket=True, ci_pipelines=2,
    )
    tasks = estimate_migration(inv)
    total = 0.0
    for desc, hrs in tasks:
        print(f"  [{hrs:4.1f}h] {desc}")
        total += hrs
    print(f"\n  Total: {total:.1f} hours, {len(tasks)} tasks")


# === Exercise 5: Buildah Script Generator ===
# Problem: Generate Buildah scripted builds from a high-level
#   application spec (no Dockerfile needed).

@dataclass
class AppSpec:
    base_image: str
    packages: list[str] = field(default_factory=list)
    copy_files: list[tuple[str, str]] = field(default_factory=list)
    workdir: str = "/app"
    ports: list[int] = field(default_factory=list)
    env: dict = field(default_factory=dict)
    cmd: Optional[str] = None
    user: Optional[str] = None


def buildah_script(app: AppSpec, tag: str) -> str:
    """Generate a Buildah scripted build shell script."""
    lines = ["#!/bin/bash", f"# Buildah build for {tag}",
             "set -euo pipefail", "",
             f"ctr=$(buildah from {app.base_image})", ""]

    if app.packages:
        pkgs = " ".join(app.packages)
        if "alpine" in app.base_image:
            lines.append(f"buildah run $ctr -- apk add --no-cache {pkgs}")
        else:
            lines.append(
                f'buildah run $ctr -- sh -c "apt-get update && '
                f"apt-get install -y --no-install-recommends {pkgs} && "
                f'rm -rf /var/lib/apt/lists/*"')

    lines.append(f"buildah config --workingdir {app.workdir} $ctr")
    for src, dst in app.copy_files:
        lines.append(f"buildah copy $ctr {src} {dst}")
    for k, v in app.env.items():
        lines.append(f'buildah config --env {k}="{v}" $ctr')
    for p in app.ports:
        lines.append(f"buildah config --port {p} $ctr")
    if app.user:
        lines.append(f"buildah config --user {app.user} $ctr")
    if app.cmd:
        lines.append(f"buildah config --cmd '{app.cmd}' $ctr")

    lines.extend(["", f"buildah commit $ctr {tag}",
                   "buildah rm $ctr",
                   f'echo "Built {tag} successfully"'])
    return "\n".join(lines)


def exercise_5():
    """Demonstrate Buildah script generation."""
    print("=== Exercise 5: Buildah Script Generator ===\n")
    app = AppSpec(
        base_image="python:3.12-slim",
        packages=["curl"],
        copy_files=[("./requirements.txt", "/app/requirements.txt"),
                    ("./app", "/app/app")],
        ports=[8000],
        env={"PYTHONUNBUFFERED": "1"},
        cmd='["gunicorn", "app:app", "-b", "0.0.0.0:8000"]',
        user="nobody",
    )
    print(buildah_script(app, "myapp:1.0"))


# === Run all exercises ===

def main():
    exercises = [exercise_1, exercise_2, exercise_3, exercise_4, exercise_5]
    for i, ex in enumerate(exercises):
        if i > 0:
            print("\n")
        ex()
    print("\n\nAll exercises completed!")


if __name__ == "__main__":
    main()
