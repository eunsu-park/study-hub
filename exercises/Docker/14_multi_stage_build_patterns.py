"""
Exercises for Lesson 14: Multi-Stage Build Patterns
Topic: Docker

Solutions to practice problems from the lesson.
"""

import re
import textwrap
from dataclasses import dataclass, field
from typing import Optional


# === Exercise 1: Dockerfile Stage Parser ===
# Problem: Parse a multi-stage Dockerfile and extract stages, base
#   images, and COPY --from dependencies between stages.

@dataclass
class Stage:
    index: int
    name: Optional[str]
    base_image: str
    instructions: list[str] = field(default_factory=list)
    copies_from: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        return self.name if self.name else f"stage-{self.index}"


def parse_stages(dockerfile: str) -> list[Stage]:
    """Parse a multi-stage Dockerfile into Stage objects."""
    stages: list[Stage] = []
    current: Optional[Stage] = None
    idx = 0

    for raw in dockerfile.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        m = re.match(r"FROM\s+(\S+)(?:\s+AS\s+(\S+))?", line, re.I)
        if m:
            current = Stage(idx, m.group(2), m.group(1))
            stages.append(current)
            idx += 1
            continue

        if current is None:
            continue
        current.instructions.append(line)

        cm = re.match(r"COPY\s+--from=(\S+)", line, re.I)
        if cm and cm.group(1) not in current.copies_from:
            current.copies_from.append(cm.group(1))

    return stages


def exercise_1():
    """Demonstrate Dockerfile stage parsing."""
    print("=== Exercise 1: Dockerfile Stage Parser ===\n")
    dockerfile = textwrap.dedent("""\
        FROM golang:1.22 AS builder
        WORKDIR /app
        COPY go.mod go.sum ./
        RUN go mod download
        COPY . .
        RUN CGO_ENABLED=0 go build -o /server .

        FROM node:20 AS frontend
        WORKDIR /app
        COPY frontend/ .
        RUN npm ci && npm run build

        FROM alpine:3.19
        COPY --from=builder /server /server
        COPY --from=frontend /app/dist /static
        CMD ["/server"]
    """)
    for s in parse_stages(dockerfile):
        deps = ", ".join(s.copies_from) or "(none)"
        print(f"  {s.label:<12} base={s.base_image:<22} deps={deps}")


# === Exercise 2: Image Size Estimator ===
# Problem: Compare single-stage vs multi-stage image sizes for
#   different base images.

BASE_SIZES_MB = {
    "scratch": 0, "alpine:3.19": 7, "debian:bookworm-slim": 80,
    "python:3.12": 1020, "python:3.12-slim": 150,
    "node:20": 1100, "node:20-slim": 200,
    "golang:1.22": 820, "golang:1.22-alpine": 260,
    "gcr.io/distroless/static-debian12": 2,
    "nginx:alpine": 45, "rust:1.77": 1400,
}


def estimate_size(base: str, app_mb: float = 10.0,
                  deps_mb: float = 200.0,
                  include_tools: bool = True) -> float:
    """Estimate total image size in MB."""
    base_size = BASE_SIZES_MB.get(base, 100)
    added = (app_mb + deps_mb) if include_tools else app_mb
    return base_size + added


def exercise_2():
    """Demonstrate image size comparison."""
    print("=== Exercise 2: Image Size Estimator ===\n")
    configs = [
        ("Single-stage", "golang:1.22", True),
        ("Multi (alpine)", "alpine:3.19", False),
        ("Multi (scratch)", "scratch", False),
        ("Multi (distroless)", "gcr.io/distroless/static-debian12", False),
    ]
    print(f"  {'Strategy':<22} {'Base Image':<38} {'MB':>6}")
    print("  " + "-" * 68)
    for label, base, tools in configs:
        sz = estimate_size(base, app_mb=8, deps_mb=100,
                           include_tools=tools)
        print(f"  {label:<22} {base:<38} {sz:>6.0f}")

    single = estimate_size("golang:1.22", 8, 100, True)
    scratch = estimate_size("scratch", 8, 100, False)
    print(f"\n  Reduction: {(1 - scratch / single) * 100:.0f}%")


# === Exercise 3: Build Cache Analyzer ===
# Problem: Detect caching anti-patterns in a Dockerfile (COPY . before
#   dependency install, split apt-get, missing cleanup).

@dataclass
class CacheIssue:
    line: int
    severity: str
    instruction: str
    suggestion: str


def analyze_cache(dockerfile: str) -> list[CacheIssue]:
    """Detect layer caching anti-patterns in a Dockerfile."""
    issues = []
    seen_copy_all = False
    seen_apt_update = -1

    for i, raw in enumerate(dockerfile.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue

        if re.match(r"COPY\s+\.\s", line) or line == "COPY . .":
            seen_copy_all = True

        if seen_copy_all and re.match(r"RUN\s+(pip|npm)\s+install", line):
            issues.append(CacheIssue(
                i, "high", line,
                "COPY dependency files first, install, then COPY rest"))

        if "apt-get update" in line and "apt-get install" not in line:
            seen_apt_update = i
        if ("apt-get install" in line and 0 < seen_apt_update != i):
            issues.append(CacheIssue(
                i, "medium", line,
                "Combine apt-get update && install in one RUN"))

        if "pip install" in line and "--no-cache-dir" not in line:
            if "--mount=type=cache" not in line:
                issues.append(CacheIssue(
                    i, "low", line,
                    "Add --no-cache-dir or use cache mount"))

        if ("apt-get install" in line
                and "/var/lib/apt/lists" not in line):
            issues.append(CacheIssue(
                i, "medium", line,
                "Add && rm -rf /var/lib/apt/lists/*"))

    return issues


def exercise_3():
    """Demonstrate cache analysis."""
    print("=== Exercise 3: Build Cache Analyzer ===\n")
    bad = textwrap.dedent("""\
        FROM python:3.12
        WORKDIR /app
        RUN apt-get update
        RUN apt-get install -y gcc
        COPY . .
        RUN pip install -r requirements.txt
        CMD ["python", "app.py"]
    """)
    for issue in analyze_cache(bad):
        print(f"  L{issue.line} [{issue.severity.upper():<6}] {issue.suggestion}")
        print(f"    -> {issue.instruction}")


# === Exercise 4: Multi-Architecture Build Planner ===
# Problem: Generate docker buildx commands for cross-compilation
#   across multiple platforms.

PLATFORMS = {
    "amd64": "linux/amd64",
    "arm64": "linux/arm64",
    "armv7": "linux/arm/v7",
    "s390x": "linux/s390x",
}


def buildx_command(
    tag: str, targets: list[str], push: bool = False,
    build_args: Optional[dict] = None,
) -> str:
    """Generate a docker buildx build command string."""
    plats = [PLATFORMS[t] for t in targets if t in PLATFORMS]
    parts = ["docker buildx build",
             f"--platform {','.join(plats)}", f"-t {tag}"]
    if push:
        parts.append("--push")
    if build_args:
        for k, v in build_args.items():
            parts.append(f"--build-arg {k}={v}")
    parts.append(".")
    return " ".join(parts)


def exercise_4():
    """Demonstrate multi-arch build planning."""
    print("=== Exercise 4: Multi-Architecture Build Planner ===\n")
    cmd = buildx_command("myuser/app:1.0", ["amd64", "arm64"], push=True)
    print(f"  Dual-arch: {cmd}\n")

    cmd = buildx_command("myuser/app:latest",
                         ["amd64", "arm64", "armv7"], push=True,
                         build_args={"VERSION": "2.0"})
    print(f"  Full:      {cmd}\n")

    print("  Available platforms:")
    for name, plat in PLATFORMS.items():
        print(f"    {name:<8} -> {plat}")


# === Exercise 5: Dockerignore Generator ===
# Problem: Generate .dockerignore from project language with estimated
#   build context savings.

IGNORE = {
    "common": [".git", ".gitignore", "Dockerfile", "docker-compose*.yml",
               "*.md", "LICENSE", ".env", ".env.*"],
    "python": ["__pycache__", "*.pyc", ".pytest_cache", ".venv", "venv",
               "*.egg-info", "dist", "build", ".mypy_cache"],
    "node":   ["node_modules", "npm-debug.log*", ".next", "dist",
               "build", "coverage"],
    "go":     ["vendor", "*.test", "bin", "tmp"],
    "rust":   ["target", "*.rs.bk"],
}

SAVINGS = {"node": 60, "python": 25, "go": 15, "rust": 50}


def generate_dockerignore(langs: list[str]) -> tuple[str, float]:
    """Return (.dockerignore content, estimated savings %)."""
    patterns = list(IGNORE["common"])
    for lang in langs:
        patterns.extend(p for p in IGNORE.get(lang, [])
                        if p not in patterns)

    content = "# .dockerignore\n" + "\n".join(patterns) + "\n"
    savings = min(sum(SAVINGS.get(l, 10) for l in langs), 80.0)
    return content, savings


def exercise_5():
    """Demonstrate .dockerignore generation."""
    print("=== Exercise 5: Dockerignore Generator ===\n")
    for langs in [["python"], ["node"], ["python", "node"]]:
        content, pct = generate_dockerignore(langs)
        lines = [l for l in content.splitlines() if l and not l.startswith("#")]
        print(f"  {'+'.join(langs)}: {len(lines)} patterns, "
              f"~{pct:.0f}% context savings")
    print()
    content, _ = generate_dockerignore(["python", "node"])
    print(content)


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
