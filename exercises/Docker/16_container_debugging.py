"""
Exercises for Lesson 16: Container Debugging
Topic: Docker

Solutions to practice problems from the lesson.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# === Exercise 1: Container Health Monitor ===
# Problem: Simulate health check state transitions (starting ->
#   healthy -> unhealthy) and generate diagnostics.

class HealthState(Enum):
    STARTING = "starting"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthResult:
    exit_code: int
    output: str
    duration_ms: int

    @property
    def passed(self) -> bool:
        return self.exit_code == 0


class HealthMonitor:
    """Tracks health check state transitions per Docker protocol."""

    def __init__(self, retries: int = 3):
        self.retries = retries
        self.state = HealthState.STARTING
        self.consecutive_fails = 0
        self.history: list[tuple[HealthState, HealthResult]] = []

    def record(self, r: HealthResult) -> HealthState:
        if r.passed:
            self.consecutive_fails = 0
            self.state = HealthState.HEALTHY
        else:
            self.consecutive_fails += 1
            if self.consecutive_fails >= self.retries:
                self.state = HealthState.UNHEALTHY
        self.history.append((self.state, r))
        return self.state

    def diagnose(self) -> list[str]:
        """Generate diagnostic recommendations from history."""
        if self.state != HealthState.UNHEALTHY:
            return []
        tips = []
        recent = [r for _, r in self.history[-self.retries:]]
        if any("connection refused" in r.output.lower() for r in recent):
            tips.append("Connection refused -- increase --health-start-period.")
        if any(r.duration_ms > 4500 for r in recent):
            tips.append("Check timing out -- increase --health-timeout.")
        if any(r.exit_code == 127 for r in recent):
            tips.append("Command not found -- install tool or change check.")
        if not tips:
            tips.append("Inspect health log: docker inspect "
                        "--format='{{json .State.Health.Log}}' <ctr> | jq")
        return tips


def exercise_1():
    """Demonstrate container health monitoring."""
    print("=== Exercise 1: Container Health Monitor ===\n")
    mon = HealthMonitor(retries=3)
    checks = [
        HealthResult(1, "curl: connection refused", 150),
        HealthResult(1, "curl: connection refused", 200),
        HealthResult(0, "OK", 50),
        HealthResult(0, "OK", 45),
        HealthResult(1, "HTTP 503", 4800),
        HealthResult(1, "HTTP 503", 4900),
        HealthResult(1, "HTTP 503", 5100),
    ]
    for i, c in enumerate(checks, 1):
        st = mon.record(c)
        tag = "PASS" if c.passed else "FAIL"
        print(f"  Check {i}: [{tag}] {c.output:<28} -> {st.value}")
    for d in mon.diagnose():
        print(f"  Tip: {d}")


# === Exercise 2: Docker Inspect Query Engine ===
# Problem: Extract fields from docker inspect JSON using dot-notation
#   paths and auto-diagnose common issues.

def query_inspect(data: dict, path: str):
    """Extract value from nested dict using dot-separated path."""
    cur = data
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and key.isdigit():
            idx = int(key)
            cur = cur[idx] if idx < len(cur) else None
        else:
            return None
        if cur is None:
            return None
    return cur


def diagnose_inspect(data: dict) -> list[str]:
    """Auto-diagnose common issues from inspect output."""
    findings = []
    ec = query_inspect(data, "State.ExitCode")
    if ec and ec != 0:
        findings.append(f"Non-zero exit code ({ec}). Check docker logs.")
    if query_inspect(data, "State.OOMKilled"):
        findings.append("OOM killed. Increase -m memory limit.")
    rc = query_inspect(data, "RestartCount")
    if rc and rc > 5:
        findings.append(f"Restarted {rc} times. Investigate crash loop.")
    if (query_inspect(data, "HostConfig.Memory") == 0
            and query_inspect(data, "HostConfig.NanoCpus") == 0):
        findings.append("No resource limits. Set -m and --cpus in production.")
    if not query_inspect(data, "Config.Healthcheck"):
        findings.append("No health check configured.")
    return findings


def exercise_2():
    """Demonstrate inspect query engine."""
    print("=== Exercise 2: Docker Inspect Query Engine ===\n")
    data = {
        "State": {"Status": "exited", "ExitCode": 137, "OOMKilled": True},
        "Config": {"Image": "myapp:latest", "Healthcheck": None,
                   "Env": ["APP_ENV=prod"]},
        "HostConfig": {"Memory": 268435456, "NanoCpus": 0},
        "NetworkSettings": {"Networks": {"bridge": {"IPAddress": "172.17.0.3"}}},
        "RestartCount": 8,
    }
    queries = ["State.Status", "State.ExitCode", "State.OOMKilled",
               "Config.Image", "NetworkSettings.Networks.bridge.IPAddress",
               "RestartCount"]
    for q in queries:
        print(f"  {q:<50} = {query_inspect(data, q)}")
    print("\n  Diagnostics:")
    for f in diagnose_inspect(data):
        print(f"    - {f}")


# === Exercise 3: Log Pattern Analyzer ===
# Problem: Parse container logs, detect error patterns, and generate
#   categorized recommendations.

ERROR_PATTERNS = [
    (r"OOM|out of memory", "memory",
     "Increase -m memory limit."),
    (r"connection refused|ECONNREFUSED", "network",
     "Check network and startup order."),
    (r"permission denied|EACCES", "permission",
     "Check USER directive and volume ownership."),
    (r"no such file|ENOENT|not found", "filesystem",
     "Verify COPY instructions and paths."),
    (r"address already in use|EADDRINUSE", "port",
     "Port conflict with another process."),
    (r"timeout|timed out", "timeout",
     "Check connectivity and timeout settings."),
]


def analyze_logs(lines: list[str]) -> tuple[int, int, list[str]]:
    """Return (error_count, warning_count, recommendations)."""
    errors = sum(1 for l in lines
                 if any(w in l.lower() for w in ["error", "fatal"]))
    warnings = sum(1 for l in lines if "warning" in l.lower())
    full = "\n".join(lines)
    recs = []
    for pat, cat, rec in ERROR_PATTERNS:
        if re.search(pat, full, re.I):
            recs.append(f"[{cat}] {rec}")
    return errors, warnings, recs


def exercise_3():
    """Demonstrate log pattern analysis."""
    print("=== Exercise 3: Log Pattern Analyzer ===\n")
    logs = [
        "INFO Starting app on port 8080",
        "WARNING High memory usage",
        "ERROR Connection refused to redis:6379",
        "ERROR ECONNREFUSED - redis unavailable",
        "WARNING Request timeout after 30s",
        "ERROR Permission denied: /data/cache",
        "FATAL Out of memory - cannot allocate",
    ]
    errs, warns, recs = analyze_logs(logs)
    print(f"  Errors: {errs}, Warnings: {warns}")
    for r in recs:
        print(f"  {r}")


# === Exercise 4: Debugging Decision Tree ===
# Problem: Walk a decision tree of symptoms to recommend specific
#   debugging commands.

@dataclass
class DebugNode:
    question: str = ""
    yes: Optional["DebugNode"] = None
    no: Optional["DebugNode"] = None
    diagnosis: str = ""
    commands: list[str] = field(default_factory=list)


def build_tree() -> DebugNode:
    oom = DebugNode(diagnosis="OOM killed", commands=[
        "docker inspect --format='{{.State.OOMKilled}}' <ctr>",
        "docker stats --no-stream <ctr>",
        "docker run -m 512m <image>"])
    perm = DebugNode(diagnosis="Permission issue", commands=[
        "docker exec <ctr> id",
        "docker exec <ctr> ls -la /app/",
        "docker run --user 1000:1000 <image>"])
    net = DebugNode(diagnosis="Network problem", commands=[
        "docker network inspect <net>",
        "docker exec <ctr> ping -c1 <target>",
        "docker exec <ctr> nslookup <svc>"])
    startup = DebugNode(diagnosis="Startup failure", commands=[
        "docker logs --tail 50 <ctr>",
        "docker run -it <image> sh"])

    return DebugNode("Is the container running?",
        yes=DebugNode("Permission denied errors?", yes=perm,
            no=DebugNode("Network errors?", yes=net, no=startup)),
        no=DebugNode("Exit code 137 (OOM)?", yes=oom, no=startup))


def walk_tree(node: DebugNode, answers: list[bool]) -> tuple[str, list[str]]:
    idx = 0
    while node.question and idx < len(answers):
        node = node.yes if answers[idx] else node.no
        idx += 1
    return node.diagnosis, node.commands


def exercise_4():
    """Demonstrate debugging decision tree."""
    print("=== Exercise 4: Debugging Decision Tree ===\n")
    tree = build_tree()
    cases = [
        ("OOM crash", [False, True]),
        ("Network issue", [True, False, True]),
        ("Permission error", [True, True]),
        ("Won't start", [False, False]),
    ]
    for label, ans in cases:
        diag, cmds = walk_tree(tree, ans)
        print(f"  {label}: {diag}")
        for c in cmds[:2]:
            print(f"    $ {c}")


# === Exercise 5: Restart Policy Simulator ===
# Problem: Simulate restart behavior under different policies (no,
#   always, on-failure, unless-stopped) with exponential backoff.

@dataclass
class ExitEvent:
    code: int
    manual_stop: bool = False


class RestartSim:
    """Simulate Docker restart policy behavior."""

    def __init__(self, policy: str, max_retries: int = 0):
        self.policy = policy
        self.max_retries = max_retries
        self.count = 0

    def evaluate(self, e: ExitEvent) -> tuple[bool, str, float]:
        """Return (should_restart, reason, delay_seconds)."""
        restart, reason = self._decide(e)
        delay = 0.0
        if restart:
            self.count += 1
            delay = min(2 ** self.count * 0.1, 60.0)
        return restart, reason, round(delay, 1)

    def _decide(self, e: ExitEvent) -> tuple[bool, str]:
        if self.policy == "no":
            return False, "never restart"
        if self.policy == "always":
            return True, "always restart"
        if self.policy == "unless-stopped":
            if e.manual_stop:
                return False, "manual stop"
            return True, "not manually stopped"
        if self.policy == "on-failure":
            if e.code == 0:
                return False, "clean exit"
            if self.max_retries and self.count >= self.max_retries:
                return False, f"max retries ({self.max_retries}) reached"
            return True, f"non-zero exit ({e.code})"
        return False, "unknown policy"


def exercise_5():
    """Demonstrate restart policy simulation."""
    print("=== Exercise 5: Restart Policy Simulator ===\n")
    events = [ExitEvent(1), ExitEvent(137), ExitEvent(1),
              ExitEvent(1), ExitEvent(0),
              ExitEvent(1, manual_stop=True)]

    for pol, mx in [("no", 0), ("always", 0),
                    ("on-failure", 3), ("unless-stopped", 0)]:
        sim = RestartSim(pol, mx)
        label = f"{pol}:{mx}" if mx else pol
        print(f"  --- {label} ---")
        for i, ev in enumerate(events, 1):
            restart, reason, delay = sim.evaluate(ev)
            act = "RESTART" if restart else "STOP"
            ms = f" (manual)" if ev.manual_stop else ""
            dl = f" delay={delay}s" if restart else ""
            print(f"    Exit {i}: code={ev.code}{ms} [{act}]{dl} {reason}")
        print(f"    Total restarts: {sim.count}\n")


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
