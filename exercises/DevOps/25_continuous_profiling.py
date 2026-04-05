#!/usr/bin/env python3
"""Exercises for Lesson 25: Continuous Profiling
Topic: DevOps
"""


def exercise_1():
    """Flame graph analysis from CPU sample data."""
    print("=== Exercise 1: Flame Graph Analysis ===\n")
    samples = {
        "serializeJSON → json.Marshal → reflect.Value.String": 350,
        "serializeJSON → json.Marshal → reflect.Value.Int": 150,
        "queryDB → sql.Query → pgx.conn.exec": 200,
        "queryDB → sql.Rows.Scan": 50,
        "authenticate → bcrypt.CompareHashAndPassword": 180,
        "compress → gzip.Writer.Write": 70,
    }
    total = sum(samples.values())

    print(f"Total samples: {total}")
    print()

    groups = {}
    for path, count in samples.items():
        top = path.split(" → ")[0]
        groups[top] = groups.get(top, 0) + count

    print("CPU distribution:")
    for func, count in sorted(groups.items(), key=lambda x: -x[1]):
        print(f"  {func}: {count}/{total} = {count/total*100:.1f}%")

    print(f"\n(a) serializeJSON: {groups['serializeJSON']}/{total} = {groups['serializeJSON']/total*100:.1f}%")
    print("(b) Most impactful: Replace encoding/json with code-generated serializer")

    speedup = 5
    new_serialize = groups["serializeJSON"] // speedup
    new_total = total - groups["serializeJSON"] + new_serialize
    print(f"\n(c) After {speedup}x faster JSON:")
    print(f"  New total: {new_total} (was {total})")
    print(f"  CPU reduction: {(1 - new_total/total)*100:.0f}%")
    print(f"  serializeJSON: {new_serialize}/{new_total} = {new_serialize/new_total*100:.1f}%")
    print(f"  queryDB: {groups['queryDB']}/{new_total} = {groups['queryDB']/new_total*100:.1f}% (now #1)")


def exercise_2():
    """Memory leak detection workflow with pprof."""
    print("\n=== Exercise 2: Memory Leak Detection ===\n")
    steps = [
        "1. Capture baseline heap: curl -o heap1.prof host:6060/debug/pprof/heap",
        "2. Wait 2-4 hours, capture second: curl -o heap2.prof ...",
        "3. Compare: go tool pprof -base heap1.prof heap2.prof",
        "4. Identify growth: (pprof) top20 -inuse_space",
        "5. Drill into source: (pprof) list leakyFunction",
        "6. Check allocs rate: go tool pprof .../allocs → high alloc + growing inuse = leak",
        "7. Check goroutines: go tool pprof .../goroutine → growing count = goroutine leak",
    ]
    print("Step-by-step pprof leak detection:")
    for step in steps:
        print(f"  {step}")

    print("\nLeak patterns:")
    patterns = [
        "Map/slice growing without bounds (no eviction)",
        "Goroutines started but never finished (blocked on channel/IO)",
        "sync.Pool buffers not returned",
        "Global variables holding references to large objects",
    ]
    for p in patterns:
        print(f"  - {p}")


if __name__ == "__main__":
    exercise_1()
    exercise_2()
