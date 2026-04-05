# 25. Continuous Profiling

**Previous**: [eBPF Observability](./24_eBPF_Observability.md) | **Next**: [Incident Response](./26_Incident_Response.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain continuous profiling and how it differs from ad-hoc profiling
2. Read and interpret flame graphs for CPU, memory, and off-CPU profiling
3. Use pprof to profile Go, Python, and JVM applications in production
4. Deploy Pyroscope or Parca for continuous profiling at scale
5. Correlate profiling data with traces and metrics for root cause analysis
6. Apply profiling insights to reduce CPU and memory costs

---

Metrics tell you *what* is happening (CPU usage is 80%). Traces tell you *where* it is happening (the order-service is slow). Profiling tells you *why* at the code level (function `serializeJSON` consumes 40% of CPU due to reflection). Continuous profiling runs profiling in production at all times with low overhead, capturing the data you need before you know you need it.

> **Analogy -- Medical Monitoring vs Diagnostic Tests**: Metrics are like vitals (heart rate, blood pressure) -- always monitored. Traces are like tracking a patient's journey through the hospital (ER → Lab → Surgery). Profiling is like a blood panel or MRI -- it reveals what is happening inside the body at a cellular level. Continuous profiling is like wearing a medical monitoring device (CGM for diabetes) -- it captures detailed data 24/7 so that when an anomaly occurs, you already have the history.

## 1. Profiling Fundamentals

### 1.1 Profile Types

| Profile Type | What It Measures | When to Use |
|-------------|-----------------|-------------|
| **CPU** | Which functions consume CPU time | High CPU usage, slow responses |
| **Heap (Alloc)** | Memory allocated by function | High memory usage, GC pressure |
| **Heap (InUse)** | Memory currently in use by function | Memory leaks |
| **Goroutine** (Go) | Number of goroutines by stack | Goroutine leaks, deadlocks |
| **Mutex** | Time spent waiting for locks | Lock contention |
| **Block** | Time spent blocking on sync primitives | Channel operations, I/O waits |
| **Off-CPU** | Time spent NOT on CPU (I/O, sleep, locks) | I/O-bound latency |
| **Wall clock** | Total elapsed time (CPU + off-CPU) | Overall function duration |

### 1.2 Sampling vs Instrumentation

| Approach | How It Works | Overhead | Accuracy |
|----------|-------------|----------|----------|
| **Sampling** | Periodically interrupt and record stack trace | Very low (~1-3%) | Statistical (more samples = higher accuracy) |
| **Instrumentation** | Insert timing code at function entry/exit | High (10-50%) | Exact (every call measured) |

Continuous profiling uses **sampling** because the overhead must be negligible in production.

### 1.3 How CPU Sampling Works

```
Time →  |-----|-----|-----|-----|-----|-----|-----|-----|
         t1    t2    t3    t4    t5    t6    t7    t8

Sample at 100 Hz (every 10ms):

t1: main → handleRequest → serializeJSON → json.Marshal
t2: main → handleRequest → queryDB → postgres.Query
t3: main → handleRequest → serializeJSON → json.Marshal
t4: main → handleRequest → serializeJSON → reflect.Value.String
t5: main → handleRequest → queryDB → postgres.Query
t6: main → GC → runtime.mallocgc
t7: main → handleRequest → serializeJSON → json.Marshal
t8: main → handleRequest → serializeJSON → json.Marshal

Result:
  serializeJSON: 5/8 samples = 62.5% of CPU
  queryDB:       2/8 samples = 25.0% of CPU
  GC:            1/8 samples = 12.5% of CPU
```

---

## 2. Flame Graphs

### 2.1 Reading Flame Graphs

```
┌──────────────────────────────────────────────────────────┐
│ root                                                      │  100%
├──────────────────────────────────────┬───────────────────┤
│ handleRequest                        │ processQueue      │  70% / 30%
├────────────────────┬─────────────────┤                   │
│ serializeJSON      │ queryDB         │                   │  45% / 25%
├──────────┬─────────┤                 │                   │
│json.Marshal│reflect │ postgres.Query  │                   │  30% / 15%
└──────────┴─────────┴─────────────────┴───────────────────┘
```

**Reading rules:**
- **X-axis**: Width = proportion of samples (NOT time). Wider = more CPU.
- **Y-axis**: Stack depth. Bottom = entry point, top = leaf functions.
- **Color**: Usually random (grouping by package) or indicates hot/cold.
- **Focus on wide bars at the top**: These are the functions where CPU is actually spent.
- **Wide bars at the bottom** just mean they call many things (not necessarily slow themselves).

### 2.2 Flame Graph Types

| Type | X-axis Represents | Bottom | Top |
|------|-------------------|--------|-----|
| **CPU flame graph** | CPU time proportion | Entry point (main) | Leaf function (hot code) |
| **Off-CPU flame graph** | Wait time proportion | Entry point | Blocking function (I/O, lock) |
| **Memory flame graph** | Bytes allocated | Entry point | Allocation site |
| **Differential flame graph** | Change between two profiles | Red = regression, Blue = improvement |

### 2.3 Generating Flame Graphs

```bash
# From perf data (Linux)
perf record -F 99 -g -p $(pidof my-service) -- sleep 30
perf script | stackcollapse-perf.pl | flamegraph.pl > cpu-flame.svg

# From Go pprof
go tool pprof -http=:6060 http://localhost:8080/debug/pprof/profile?seconds=30
# Opens interactive flame graph in browser

# From Python py-spy
py-spy record -o profile.svg --pid $(pidof python3) --duration 30

# From Java async-profiler
asprof -d 30 -f profile.html $(pidof java)
```

---

## 3. Go Profiling with pprof

### 3.1 Enabling pprof in Production

```go
package main

import (
    "net/http"
    _ "net/http/pprof"  // Register pprof handlers
)

func main() {
    // pprof endpoints available at /debug/pprof/
    // In production, serve on a separate port (not public-facing)
    go func() {
        http.ListenAndServe("localhost:6060", nil)
    }()

    // ... application code ...
}
```

### 3.2 pprof Endpoints

| Endpoint | Profile Type | Usage |
|----------|-------------|-------|
| `/debug/pprof/profile?seconds=30` | CPU (30s) | `go tool pprof http://host:6060/debug/pprof/profile?seconds=30` |
| `/debug/pprof/heap` | Heap (current) | `go tool pprof http://host:6060/debug/pprof/heap` |
| `/debug/pprof/allocs` | Heap (cumulative) | Shows all allocations since start |
| `/debug/pprof/goroutine` | Goroutines | `go tool pprof http://host:6060/debug/pprof/goroutine` |
| `/debug/pprof/mutex` | Mutex contention | Requires `runtime.SetMutexProfileFraction(5)` |
| `/debug/pprof/block` | Block (sync) | Requires `runtime.SetBlockProfileRate(1)` |

### 3.3 pprof Analysis Workflow

```bash
# 1. Capture CPU profile
go tool pprof http://localhost:6060/debug/pprof/profile?seconds=60

# 2. In pprof interactive mode:
(pprof) top20              # Top 20 CPU consumers
(pprof) top20 -cum         # Top 20 by cumulative time
(pprof) list serializeJSON # Source-level annotation
(pprof) web                # Open flame graph in browser
(pprof) peek queryDB       # Show callers and callees

# 3. Compare two profiles (before/after optimization)
go tool pprof -base before.prof after.prof
(pprof) top20              # Shows difference
```

### 3.4 Memory Profiling

```bash
# Capture heap profile
go tool pprof http://localhost:6060/debug/pprof/heap

(pprof) top20 -inuse_space    # Currently allocated memory
(pprof) top20 -alloc_space    # Total allocated since start (GC pressure)
(pprof) top20 -alloc_objects  # Number of allocations (GC trigger rate)

# Find memory leak: compare two heap profiles taken minutes apart
go tool pprof -base heap1.prof heap2.prof
(pprof) top20 -inuse_space    # Shows what grew between snapshots
```

---

## 4. Python Profiling

### 4.1 py-spy (Sampling Profiler)

```bash
# Attach to running Python process (no code changes)
py-spy top --pid $(pidof python3)          # Real-time top-like view
py-spy record -o profile.svg --pid $(pidof python3) --duration 30  # Flame graph

# Profile a specific command
py-spy record -o profile.svg -- python3 app.py

# Subprocesses (follow forks)
py-spy record -o profile.svg --subprocesses -- gunicorn app:app
```

### 4.2 cProfile and Scalene

```python
# cProfile: built-in deterministic profiler (high overhead, not for production)
import cProfile
cProfile.run('process_requests()', 'output.prof')

# Analyze with snakeviz
# pip install snakeviz
# snakeviz output.prof  → opens flame graph in browser
```

```bash
# Scalene: low-overhead CPU + memory + GPU profiler
pip install scalene
scalene --cpu --memory --reduced-profile app.py
```

### 4.3 Memory Profiling with memray

```bash
# memray: production-grade memory profiler for Python
pip install memray

# Attach to running process
memray attach $(pidof python3)

# Profile from start
memray run app.py
memray flamegraph memray-output.bin -o memory.html

# Track leaks (show allocations not freed)
memray flamegraph memray-output.bin --leaks -o leaks.html
```

---

## 5. Continuous Profiling Platforms

### 5.1 Pyroscope

Pyroscope is an open-source continuous profiling platform:

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  App + Agent │  │  App + Agent │  │  App + Agent │
│  (SDK or     │  │  (SDK or     │  │  (SDK or     │
│   eBPF)      │  │   eBPF)      │  │   eBPF)      │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
              ┌──────────▼───────────┐
              │   Pyroscope Server   │
              │  - Ingestion         │
              │  - Storage (blocks)  │
              │  - Query engine      │
              │  - Flame graph UI    │
              └──────────────────────┘
```

### 5.2 Pyroscope Integration

```python
# Python: Pyroscope SDK
import pyroscope

pyroscope.configure(
    application_name="payment-service",
    server_address="http://pyroscope:4040",
    tags={
        "environment": "production",
        "region": "us-east",
    },
    # Enable specific profilers
    enabled_profilers=[
        pyroscope.CpuProfiler,
        pyroscope.AllocProfiler,
        pyroscope.LockProfiler,
    ],
    # Sampling rate
    sample_rate=100,  # 100 Hz
)

# Tag specific code paths for filtering
with pyroscope.tag_wrapper({"endpoint": "/api/orders", "method": "POST"}):
    process_order(order)
```

```go
// Go: Pyroscope SDK
import "github.com/grafana/pyroscope-go"

func main() {
    pyroscope.Start(pyroscope.Config{
        ApplicationName: "payment-service",
        ServerAddress:   "http://pyroscope:4040",
        Tags:            map[string]string{"env": "production"},
        ProfileTypes: []pyroscope.ProfileType{
            pyroscope.ProfileCPU,
            pyroscope.ProfileAllocObjects,
            pyroscope.ProfileAllocSpace,
            pyroscope.ProfileInuseObjects,
            pyroscope.ProfileInuseSpace,
            pyroscope.ProfileGoroutines,
            pyroscope.ProfileMutexCount,
            pyroscope.ProfileMutexDuration,
            pyroscope.ProfileBlockCount,
            pyroscope.ProfileBlockDuration,
        },
    })
    defer pyroscope.Stop()
}
```

### 5.3 eBPF-Based Continuous Profiling

For language-agnostic profiling without SDK changes:

```yaml
# Pyroscope eBPF agent (DaemonSet)
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: pyroscope-ebpf
spec:
  template:
    spec:
      containers:
        - name: agent
          image: grafana/pyroscope:latest
          args:
            - "ebpf"
            - "--server-address=http://pyroscope:4040"
            - "--node=$(NODE_NAME)"
          securityContext:
            privileged: true
          env:
            - name: NODE_NAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
          volumeMounts:
            - name: modules
              mountPath: /lib/modules
            - name: debugfs
              mountPath: /sys/kernel/debug
      volumes:
        - name: modules
          hostPath:
            path: /lib/modules
        - name: debugfs
          hostPath:
            path: /sys/kernel/debug
```

---

## 6. Profiling-Trace Correlation

### 6.1 Linking Profiles to Traces

The most powerful debugging workflow: from a slow trace span, jump to the CPU profile of that exact time period.

```
Trace (Tempo/Jaeger):
  order-service: POST /orders (2.5s)
    → createOrder (2.3s)
      → validateInventory (100ms)
      → calculateTotals (2.1s)  ← WHY IS THIS SLOW?
          span.start: 14:00:00.100
          span.end: 14:00:02.200

Profile (Pyroscope):
  Query: service=order-service, from=14:00:00, to=14:00:02
  Flame graph shows:
    calculateTotals → applyDiscountRules → regexp.Compile (85% CPU)
  → Root cause: compiling regexps on every request instead of caching
```

### 6.2 Grafana Integration

```
Grafana Tempo → Pyroscope linking:
  1. In Tempo data source settings:
     - Enable "Traces to Profiles"
     - Link to Pyroscope data source
     - Match on service.name label

  2. When viewing a trace:
     - Click on a span
     - Click "View Profile" button
     - Opens Pyroscope flame graph for that time range and service

  3. Compare profiles:
     - Select a baseline time range (before the incident)
     - Select the incident time range
     - View differential flame graph (red = regression)
```

---

## 7. Cost Optimization with Profiling

### 7.1 Identifying CPU Waste

```
Before profiling optimization:
  payment-service: 8 pods × 2 CPU = 16 CPU cores

After profiling analysis:
  - json.Marshal using reflection: 35% CPU → Switch to jsoniter: 12% CPU
  - regexp.Compile per request: 20% CPU → Cache compiled regexps: 0.1% CPU
  - TLS handshake per request: 15% CPU → Connection pooling: 2% CPU
  - Total CPU reduction: 70% → 30% = 57% reduction

After optimization:
  payment-service: 4 pods × 2 CPU = 8 CPU cores
  Savings: 8 CPU cores × $0.05/hr × 720 hr/mo = $288/mo per service
```

### 7.2 Memory Optimization

```
Profiling reveals:
  - String concatenation in loop: allocates 500MB/min (GC pressure)
    Fix: use strings.Builder → 5MB/min
  - Large struct copying in function args: 200MB in-use
    Fix: pass pointers → 50MB in-use
  - Unbounded cache: grows to 2GB over 24 hours
    Fix: LRU cache with max size → stable at 256MB

Result: memory request reduced from 4Gi to 1Gi per pod
  Savings: 3Gi × 8 pods × $0.004/GiB/hr × 720 hr/mo = $69/mo
```

---

## 8. Best Practices

### 8.1 Production Profiling Checklist

| Practice | Reason |
|----------|--------|
| Use sampling profilers (not instrumenting) | Keep overhead < 2% |
| Profile continuously, not just during incidents | Have baseline data for comparison |
| Set CPU sample rate to 100 Hz | Good accuracy with minimal overhead |
| Profile memory allocation rate (allocs), not just in-use | Reveals GC pressure |
| Add service and environment tags | Filter profiles in multi-service deployments |
| Integrate with traces for context | Jump from slow span to code-level profile |
| Review profiles weekly, not just during incidents | Find gradual regressions |
| Use differential flame graphs for before/after | Validate optimizations objectively |

---

## 9. Next Steps

- [26_Incident_Response.md](./26_Incident_Response.md) -- On-call practices and incident management
- [27_AIOps_Anomaly_Detection.md](./27_AIOps_Anomaly_Detection.md) -- ML-based anomaly detection

---

## Exercises

### Exercise 1: Flame Graph Analysis

Given the following flame graph data (showing CPU sample counts):

```
main → handleRequest → serializeJSON → json.Marshal → reflect.Value.String: 350
main → handleRequest → serializeJSON → json.Marshal → reflect.Value.Int: 150
main → handleRequest → queryDB → sql.Query → pgx.conn.exec: 200
main → handleRequest → queryDB → sql.Rows.Scan: 50
main → handleRequest → authenticate → bcrypt.CompareHashAndPassword: 180
main → handleRequest → compress → gzip.Writer.Write: 70
Total samples: 1000
```

Answer: (a) What percentage of CPU does `serializeJSON` consume? (b) What is the single most impactful optimization you could make? (c) If you replace `json.Marshal` with a code-generated serializer that is 5x faster, what would the new total CPU consumption look like?

<details>
<summary>Show Answer</summary>

**(a) serializeJSON CPU percentage:**
```
serializeJSON samples = 350 + 150 = 500
Percentage = 500 / 1000 = 50%
```

**(b) Most impactful optimization:**
`serializeJSON` via `json.Marshal` (50% of CPU). Specifically, `reflect.Value.String` (35%) is the dominant leaf function. The standard library `json.Marshal` uses reflection for serialization, which is CPU-intensive.

**Optimization**: Replace `encoding/json` with a code-generated serializer like `easyjson`, `jsoniter`, or `sonic` that avoids reflection. Expected speedup: 3-10x for JSON serialization.

**(c) After 5x faster JSON serialization:**
```
Before: serializeJSON = 500 samples (50%)
After:  serializeJSON = 500 / 5 = 100 samples

New total = 1000 - 500 + 100 = 600 samples
New distribution:
  serializeJSON:  100/600 = 16.7% (was 50%)
  queryDB:        250/600 = 41.7% (was 25%)
  authenticate:   180/600 = 30.0% (was 18%)
  compress:        70/600 = 11.7% (was 7%)

Overall: 600/1000 = 40% less CPU total.
This means you could serve the same traffic with ~40% fewer CPU cores.
```

</details>

### Exercise 2: Memory Leak Detection

A Go service's memory usage grows from 500MB to 4GB over 24 hours, then OOM-kills. Describe the step-by-step process to identify the leak using pprof. Include the specific commands, pprof queries, and what patterns to look for in the output.

<details>
<summary>Show Answer</summary>

**Step 1: Capture baseline heap profile**
```bash
# Right after service restart (500MB)
curl -o heap_baseline.prof http://service:6060/debug/pprof/heap
```

**Step 2: Wait and capture second profile**
```bash
# After 2-4 hours (should be ~1-2GB if linear growth)
curl -o heap_after4h.prof http://service:6060/debug/pprof/heap
```

**Step 3: Compare profiles (differential analysis)**
```bash
go tool pprof -base heap_baseline.prof heap_after4h.prof

(pprof) top20 -inuse_space
# Shows functions where in-use memory GREW between snapshots
# The top entries are likely the leak sources

# Expected output (example):
# 1.2GB  leakyCache.Store      (cache that never evicts)
# 200MB  bufPool.Get           (buffers never returned to pool)
```

**Step 4: Drill into the leak source**
```bash
(pprof) list leakyCache.Store
# Shows source code with memory annotations per line
# Line 45:   cache[key] = largeStruct   ← 1.2GB allocated here

(pprof) peek leakyCache.Store
# Shows callers: who is calling this function?
# handleRequest → processData → leakyCache.Store
```

**Step 5: Check allocation rate (GC pressure)**
```bash
go tool pprof http://service:6060/debug/pprof/allocs

(pprof) top20 -alloc_space
# Shows cumulative allocations since start
# High alloc rate + growing inuse = leak
# High alloc rate + stable inuse = just GC pressure (not a leak)
```

**Step 6: Check goroutine count (goroutine leak)**
```bash
go tool pprof http://service:6060/debug/pprof/goroutine

(pprof) top20
# If goroutine count is growing over time, you have a goroutine leak
# Each goroutine holds ~2-8KB stack + whatever data it references
```

**Patterns indicating a leak:**
- `inuse_space` for a function grows linearly over time
- A map or slice that grows without bounds (no eviction/cleanup)
- Goroutines that start but never finish (blocked on channel or I/O)
- Buffers allocated via `sync.Pool` but never returned (broken pool usage)
- Global variables holding references to large objects

</details>

---

## References

- [Pyroscope Documentation](https://pyroscope.io/docs/)
- [Go pprof Documentation](https://pkg.go.dev/net/http/pprof)
- [Brendan Gregg -- Flame Graphs](https://www.brendangregg.com/flamegraphs.html)
- [py-spy -- Sampling Profiler for Python](https://github.com/benfred/py-spy)
- [Grafana Tempo -- Traces to Profiles](https://grafana.com/docs/tempo/latest/)
- [Parca -- Continuous Profiling](https://www.parca.dev/)
