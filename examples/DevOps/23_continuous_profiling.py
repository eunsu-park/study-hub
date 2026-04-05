#!/usr/bin/env python3
"""Example: Continuous Profiling — Always-On Performance Analysis

Demonstrates continuous profiling concepts: CPU and memory flame graph
data modeling, profile aggregation over time, differential profiling
(comparing two time windows), and automated hot-spot detection.
Related lesson: 25_Continuous_Profiling.md
"""

# =============================================================================
# WHY CONTINUOUS PROFILING?
# Traditional profiling is done ad-hoc during debugging. Continuous profiling
# (Pyroscope, Parca, Google Cloud Profiler) runs permanently with minimal
# overhead (~2%), capturing CPU, memory, and lock profiles. This lets you
# answer "what changed?" by comparing profiles before and after a deployment.
# =============================================================================

import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


# =============================================================================
# 1. STACK TRACE AND PROFILE MODELS
# =============================================================================

@dataclass
class StackSample:
    """A single stack trace sample from the profiler."""
    frames: list[str]   # Bottom (main) to top (leaf function)
    value: int           # CPU nanoseconds or bytes allocated
    timestamp: float = field(default_factory=time.time)
    labels: dict[str, str] = field(default_factory=dict)


@dataclass
class Profile:
    """A collection of stack samples for a time period."""
    service: str
    profile_type: str  # cpu, alloc, goroutine, mutex
    start_time: float
    end_time: float
    samples: list[StackSample] = field(default_factory=list)

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time

    @property
    def total_value(self) -> int:
        return sum(s.value for s in self.samples)


# =============================================================================
# 2. FLAME GRAPH DATA STRUCTURE
# =============================================================================

@dataclass
class FlameNode:
    """A node in a flame graph (collapsed stack representation)."""
    name: str
    self_value: int = 0
    total_value: int = 0
    children: dict[str, "FlameNode"] = field(default_factory=dict)

    def add_stack(self, frames: list[str], value: int, depth: int = 0) -> None:
        """Insert a stack trace into the flame graph."""
        self.total_value += value
        if depth >= len(frames):
            self.self_value += value
            return
        frame = frames[depth]
        if frame not in self.children:
            self.children[frame] = FlameNode(name=frame)
        self.children[frame].add_stack(frames, value, depth + 1)

    def top_functions(self, n: int = 10) -> list[tuple[str, int, float]]:
        """Find the top N functions by self time."""
        result: list[tuple[str, int, float]] = []
        self._collect_self(result)
        result.sort(key=lambda x: x[1], reverse=True)
        total = self.total_value or 1
        return [(name, val, val / total * 100) for name, val, _ in result[:n]]

    def _collect_self(self, result: list) -> None:
        if self.self_value > 0:
            result.append((self.name, self.self_value, 0.0))
        for child in self.children.values():
            child._collect_self(result)

    def display(self, indent: int = 0, min_pct: float = 1.0) -> str:
        """ASCII flame graph display."""
        total = self.total_value or 1
        pct = (self.total_value / total) * 100 if indent == 0 else 0
        lines = []
        # Sort children by total_value descending
        sorted_children = sorted(
            self.children.values(), key=lambda c: c.total_value, reverse=True
        )
        for child in sorted_children:
            child_pct = child.total_value / total * 100
            if child_pct < min_pct:
                continue
            prefix = "  " * indent
            lines.append(f"{prefix}{child.name} ({child.total_value}, "
                         f"{child_pct:.1f}%)")
            lines.append(child.display(indent + 1, min_pct))
        return "\n".join(lines)


def build_flame_graph(profile: Profile) -> FlameNode:
    """Build a flame graph from a profile."""
    root = FlameNode(name="root")
    for sample in profile.samples:
        root.add_stack(sample.frames, sample.value)
    return root


# =============================================================================
# 3. DIFFERENTIAL PROFILING
# =============================================================================

def diff_profiles(before: FlameNode, after: FlameNode) -> list[dict[str, Any]]:
    """Compare two flame graphs to find regressions and improvements."""
    before_funcs = {name: val for name, val, _ in before.top_functions(50)}
    after_funcs = {name: val for name, val, _ in after.top_functions(50)}
    all_funcs = set(before_funcs) | set(after_funcs)

    diffs = []
    for func in all_funcs:
        bval = before_funcs.get(func, 0)
        aval = after_funcs.get(func, 0)
        if bval == 0 and aval == 0:
            continue
        change_pct = ((aval - bval) / max(bval, 1)) * 100
        diffs.append({
            "function": func,
            "before": bval,
            "after": aval,
            "delta": aval - bval,
            "change_pct": round(change_pct, 1),
            "status": "regression" if change_pct > 10 else
                      "improvement" if change_pct < -10 else "stable",
        })
    diffs.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return diffs


# =============================================================================
# 4. HOT-SPOT DETECTOR
# =============================================================================

def detect_hotspots(flame: FlameNode, threshold_pct: float = 5.0) -> list[dict]:
    """Detect functions consuming more than threshold% of total time."""
    hotspots = []
    total = flame.total_value or 1
    for name, self_val, pct in flame.top_functions(20):
        if pct >= threshold_pct:
            hotspots.append({
                "function": name,
                "self_value": self_val,
                "pct_of_total": round(pct, 1),
                "recommendation": _recommend(name, pct),
            })
    return hotspots


def _recommend(func_name: str, pct: float) -> str:
    """Generate optimization recommendation based on function name heuristics."""
    lower = func_name.lower()
    if "json" in lower or "marshal" in lower or "serialize" in lower:
        return "Consider binary serialization (protobuf/msgpack) or caching"
    if "gc" in lower or "alloc" in lower:
        return "Reduce allocations; consider object pooling or sync.Pool"
    if "sql" in lower or "query" in lower or "db" in lower:
        return "Check query plans; add indexes or caching layer"
    if "compress" in lower or "gzip" in lower:
        return "Use async compression or lower compression level"
    if "tls" in lower or "crypto" in lower:
        return "Consider TLS session resumption or hardware acceleration"
    if pct > 20:
        return "Major hotspot — profile at line level for optimization"
    return "Review for algorithmic improvements"


# =============================================================================
# 5. SYNTHETIC DATA GENERATOR
# =============================================================================

def generate_profile(service: str, profile_type: str = "cpu",
                     n_samples: int = 500, is_after: bool = False) -> Profile:
    """Generate a synthetic CPU profile."""
    random.seed(42 if not is_after else 99)
    stacks = [
        ["main", "http.Serve", "handler.Orders", "db.Query"],
        ["main", "http.Serve", "handler.Orders", "json.Marshal"],
        ["main", "http.Serve", "handler.Users", "auth.Validate"],
        ["main", "http.Serve", "handler.Users", "db.Query"],
        ["main", "http.Serve", "middleware.Logger"],
        ["main", "http.Serve", "handler.Health"],
        ["main", "runtime.GC", "runtime.gcBgSweep"],
        ["main", "http.Serve", "handler.Orders", "cache.Get"],
        ["main", "http.Serve", "tls.Handshake"],
    ]
    weights = [25, 15, 10, 12, 5, 3, 8, 12, 10]
    if is_after:
        weights = [15, 30, 10, 12, 5, 3, 5, 10, 10]  # json.Marshal regresses

    now = time.time()
    samples = []
    for _ in range(n_samples):
        idx = random.choices(range(len(stacks)), weights=weights)[0]
        value = random.randint(100_000, 10_000_000)
        samples.append(StackSample(frames=stacks[idx], value=value))
    return Profile(
        service=service, profile_type=profile_type,
        start_time=now - 60, end_time=now, samples=samples,
    )


# =============================================================================
# 6. DEMO
# =============================================================================

if __name__ == "__main__":
    # --- Build Flame Graph ---
    print("=" * 60)
    print("Continuous Profiling — CPU Flame Graph")
    print("=" * 60)
    profile = generate_profile("order-svc", n_samples=1000)
    flame = build_flame_graph(profile)
    print(f"  Service: {profile.service}")
    print(f"  Samples: {len(profile.samples)}")
    print(f"  Total CPU: {flame.total_value:,} ns\n")
    print("  Top functions by self time:")
    for name, val, pct in flame.top_functions(8):
        print(f"    {name:<30s} {val:>12,} ns ({pct:.1f}%)")

    # --- Hot-spot Detection ---
    print(f"\n{'=' * 60}")
    print("Hot-spot Detection")
    print("=" * 60)
    for hs in detect_hotspots(flame, threshold_pct=5.0):
        print(f"  {hs['function']}: {hs['pct_of_total']}%")
        print(f"    Recommendation: {hs['recommendation']}")

    # --- Differential Profiling ---
    print(f"\n{'=' * 60}")
    print("Differential Profiling (before vs after deploy)")
    print("=" * 60)
    before_profile = generate_profile("order-svc", n_samples=1000, is_after=False)
    after_profile = generate_profile("order-svc", n_samples=1000, is_after=True)
    before_flame = build_flame_graph(before_profile)
    after_flame = build_flame_graph(after_profile)
    diffs = diff_profiles(before_flame, after_flame)
    for d in diffs[:8]:
        arrow = "^" if d["delta"] > 0 else "v" if d["delta"] < 0 else "="
        print(f"  {d['function']:<30s} {d['before']:>10,} -> {d['after']:>10,} "
              f"({arrow} {d['change_pct']:+.1f}%) [{d['status']}]")
