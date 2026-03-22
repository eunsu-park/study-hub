"""
Exercises for Lesson 01: Docker Basics
Topic: Docker

Solutions to practice problems from the lesson.
"""

import shutil
import subprocess
import textwrap


# === Exercise 1: Docker Installation Checker ===
# Problem: Verify whether Docker is installed, report the version,
#   and determine if the Docker daemon is running.

def check_docker_installation() -> dict:
    """Check Docker installation status and gather environment info."""
    result = {"installed": False, "version": None,
              "daemon_running": False, "compose_available": False}

    if shutil.which("docker") is None:
        return result
    result["installed"] = True

    for key, cmd in [
        ("version", ["docker", "--version"]),
        ("daemon_running", ["docker", "info"]),
        ("compose_available", ["docker", "compose", "version"]),
    ]:
        try:
            proc = subprocess.run(cmd, capture_output=True,
                                  text=True, timeout=10)
            if key == "version" and proc.returncode == 0:
                result["version"] = proc.stdout.strip()
            else:
                result[key] = proc.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return result


def exercise_1():
    """Demonstrate Docker installation checking."""
    print("=== Exercise 1: Docker Installation Checker ===\n")
    info = check_docker_installation()
    for k, v in info.items():
        print(f"  {k:<20} {v}")


# === Exercise 2: Container Lifecycle Simulator ===
# Problem: Model the Docker container lifecycle as a state machine
#   and validate transitions.

class ContainerLifecycle:
    """State machine for a Docker container lifecycle."""

    TRANSITIONS = {
        "created":  {"start": "running", "remove": "removed"},
        "running":  {"pause": "paused", "stop": "stopped",
                     "kill": "stopped", "remove": "removed"},
        "paused":   {"unpause": "running", "stop": "stopped",
                     "kill": "stopped"},
        "stopped":  {"start": "running", "remove": "removed"},
        "removed":  {},
    }

    def __init__(self, name: str):
        self.name = name
        self.state = "created"
        self.history: list[str] = ["created"]

    def transition(self, action: str) -> str:
        """Apply action, return new state. Raises ValueError if invalid."""
        allowed = self.TRANSITIONS.get(self.state, {})
        if action not in allowed:
            raise ValueError(
                f"Cannot '{action}' in state '{self.state}'. "
                f"Valid: {list(allowed.keys())}"
            )
        self.state = allowed[action]
        self.history.append(self.state)
        return self.state


def exercise_2():
    """Demonstrate the container lifecycle state machine."""
    print("=== Exercise 2: Container Lifecycle Simulator ===\n")
    c = ContainerLifecycle("my-nginx")
    for action in ["start", "pause", "unpause", "stop", "start",
                    "kill", "remove"]:
        new = c.transition(action)
        print(f"  {action:>8} -> {new}")
    print(f"\n  History: {' -> '.join(c.history)}")

    c2 = ContainerLifecycle("test")
    try:
        c2.transition("stop")
    except ValueError as exc:
        print(f"  Expected error: {exc}")


# === Exercise 3: Docker Command Builder ===
# Problem: Create a builder that constructs `docker run` commands
#   with port mapping, volumes, env vars, and common flags.

class DockerRunBuilder:
    """Fluent builder for docker run command strings."""

    def __init__(self, image: str):
        self._image = image
        self._flags: list[str] = []
        self._cmd: str | None = None

    def name(self, n: str) -> "DockerRunBuilder":
        self._flags.extend(["--name", n]); return self

    def detach(self) -> "DockerRunBuilder":
        self._flags.append("-d"); return self

    def interactive(self) -> "DockerRunBuilder":
        self._flags.append("-it"); return self

    def auto_remove(self) -> "DockerRunBuilder":
        self._flags.append("--rm"); return self

    def port(self, host: int, container: int) -> "DockerRunBuilder":
        self._flags.extend(["-p", f"{host}:{container}"]); return self

    def volume(self, src: str, dst: str) -> "DockerRunBuilder":
        self._flags.extend(["-v", f"{src}:{dst}"]); return self

    def env(self, key: str, val: str) -> "DockerRunBuilder":
        self._flags.extend(["-e", f"{key}={val}"]); return self

    def network(self, net: str) -> "DockerRunBuilder":
        self._flags.extend(["--network", net]); return self

    def command(self, cmd: str) -> "DockerRunBuilder":
        self._cmd = cmd; return self

    def build(self) -> str:
        parts = ["docker", "run"] + self._flags + [self._image]
        if self._cmd:
            parts.append(self._cmd)
        return " ".join(parts)


def exercise_3():
    """Demonstrate Docker command building."""
    print("=== Exercise 3: Docker Command Builder ===\n")

    cmd1 = (DockerRunBuilder("nginx:alpine")
            .name("web").detach().port(8080, 80).build())
    print(f"  Nginx:    {cmd1}")

    cmd2 = (DockerRunBuilder("node:20")
            .name("dev").interactive().auto_remove()
            .port(3000, 3000).volume("./src", "/app/src")
            .env("NODE_ENV", "development").command("bash").build())
    print(f"  Node dev: {cmd2}")

    cmd3 = (DockerRunBuilder("postgres:16-alpine")
            .name("mydb").detach().port(5432, 5432)
            .volume("pgdata", "/var/lib/postgresql/data")
            .env("POSTGRES_PASSWORD", "secret").network("backend").build())
    print(f"  Postgres: {cmd3}")


# === Exercise 4: Container vs VM Suitability Scorer ===
# Problem: Score containers vs VMs for a workload based on requirements.

COMPARISON = [
    ("Startup time",       "seconds",  "minutes"),
    ("Resource overhead",  "low",      "high"),
    ("Isolation level",    "process",  "hardware"),
    ("OS requirement",     "shared",   "separate"),
    ("Image size",         "MB range", "GB range"),
    ("Security boundary",  "moderate", "strong"),
]


def compute_suitability(fast_start=False, strong_iso=False,
                        hw_access=False, constrained=False,
                        diff_os=False) -> dict[str, float]:
    """Score containers vs VMs (0.0-1.0) based on workload needs."""
    c, v = 0.5, 0.5
    if fast_start:   c += 0.2
    if strong_iso:   v += 0.2
    if hw_access:    v += 0.2
    if constrained:  c += 0.2
    if diff_os:      v += 0.2
    mx = max(c, v)
    return {"container": round(c / mx, 2), "vm": round(v / mx, 2)}


def exercise_4():
    """Demonstrate container vs VM comparison."""
    print("=== Exercise 4: Container vs VM Suitability ===\n")
    print(f"  {'Criterion':<20} {'Container':<12} {'VM':<12}")
    print("  " + "-" * 44)
    for crit, cv, vv in COMPARISON:
        print(f"  {crit:<20} {cv:<12} {vv:<12}")

    print("\n  Workload scores:")
    s = compute_suitability(fast_start=True, constrained=True)
    print(f"    Microservices: container={s['container']} vm={s['vm']}")
    s = compute_suitability(diff_os=True, strong_iso=True, hw_access=True)
    print(f"    Legacy system: container={s['container']} vm={s['vm']}")


# === Exercise 5: Docker Workflow Visualizer ===
# Problem: Visualize the Docker workflow and parse docker ps output.

def parse_docker_ps(raw: str) -> list[dict]:
    """Parse docker ps output into a list of dicts."""
    lines = raw.strip().splitlines()
    if len(lines) < 2:
        return []
    headers = [h.lower() for h in lines[0].split()]
    results = []
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split(None, len(headers) - 1)
        results.append(dict(zip(headers, parts)))
    return results


def visualize_workflow() -> str:
    """Return ASCII diagram of Docker pull-run workflow."""
    return textwrap.dedent("""\
        docker run nginx
              |
              v
        +-----------+     +----------+     +----------+
        |  Docker   | --> |  Docker  | --> |  Docker  |
        |  Client   |     |  Daemon  |     |  Hub     |
        +-----------+     +-----+----+     +-----+----+
                                |               |
                                |  Pull image   |
                                |<--------------+
                                v
                          +-----------+
                          | Container |
                          +-----------+

        1. CLI sends request to daemon
        2. Daemon checks local cache
        3. If missing, pulls from Hub
        4. Creates and starts container""")


def exercise_5():
    """Demonstrate workflow visualization and ps parsing."""
    print("=== Exercise 5: Docker Workflow Visualizer ===\n")
    print(visualize_workflow())

    sample = ("CONTAINER_ID  IMAGE         STATUS          NAMES\n"
              "a1b2c3d4e5f6  nginx:alpine  Up_2_hours      web\n"
              "f6e5d4c3b2a1  redis:7       Exited_(0)      cache\n")
    print("\n  Parsed containers:")
    for c in parse_docker_ps(sample):
        tag = "RUNNING" if "Up" in c.get("status", "") else "STOPPED"
        print(f"    {c['names']:<10} {c['image']:<14} [{tag}]")


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
