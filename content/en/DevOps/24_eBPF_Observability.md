# 24. eBPF Observability

**Previous**: [OpenTelemetry Pipelines](./23_OpenTelemetry_Pipelines.md) | **Next**: [Continuous Profiling](./25_Continuous_Profiling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain how eBPF works and why it enables observability without application code changes
2. Use bpftrace to write one-liner and scripted observability probes for production debugging
3. Describe Cilium Hubble's architecture for Kubernetes network observability
4. Apply eBPF-based tools to observe system calls, network traffic, and application behavior at the kernel level
5. Compare eBPF-based observability with traditional instrumentation approaches
6. Evaluate when eBPF observability is appropriate versus OpenTelemetry-based instrumentation

---

eBPF (extended Berkeley Packet Filter) is a technology that allows running sandboxed programs inside the Linux kernel without modifying kernel source code or loading kernel modules. For observability, this means you can instrument any application -- regardless of language, framework, or whether you have access to its source code -- by attaching probes at the kernel level.

> **Analogy -- Airport Security Cameras**: Traditional instrumentation (OpenTelemetry) is like asking every passenger to wear a body camera. It requires cooperation, and each passenger must opt in. eBPF is like the airport's security cameras -- they observe everyone passing through, require no cooperation from passengers, and can track movement patterns without anyone changing their behavior. The cameras (eBPF programs) are installed at fixed points (kernel hooks) and observe all traffic.

## 1. eBPF Fundamentals

### 1.1 How eBPF Works

```
User Space                              Kernel Space
┌──────────────────┐                   ┌──────────────────────────┐
│                  │                   │                          │
│  eBPF Program    │  load + verify   │  eBPF Virtual Machine    │
│  (C or Rust)     │ ──────────────→  │  ┌────────────────────┐  │
│                  │                   │  │ JIT Compiled Code  │  │
│  bpftrace script │                   │  │ (runs at kernel    │  │
│  or compiled     │                   │  │  speed)            │  │
│  binary          │                   │  └────────┬───────────┘  │
└──────────────────┘                   │           │              │
                                       │  Attach to hook points:  │
       ┌───────────────────────────────┤  ┌──────────────────┐   │
       │  Results via:                 │  │ - kprobes (kernel │   │
       │  - BPF maps (shared memory)   │  │   function entry) │   │
       │  - Perf events (ring buffer)  │  │ - tracepoints     │   │
       │  - Print to trace pipe        │  │ - uprobes (user   │   │
       │                               │  │   function entry)  │   │
       ▼                               │  │ - XDP (network    │   │
┌──────────────────┐                   │  │   packet hooks)   │   │
│  User-space tool │                   │  │ - perf events     │   │
│  (dashboard,     │                   │  └──────────────────┘   │
│   CLI output,    │                   │                          │
│   Prometheus     │                   └──────────────────────────┘
│   exporter)      │
└──────────────────┘
```

### 1.2 eBPF Safety Model

The kernel verifier ensures eBPF programs are safe:

| Check | Purpose |
|-------|---------|
| **No unbounded loops** | Prevents infinite loops in kernel context |
| **Memory bounds checking** | Prevents buffer overflows |
| **Stack size limit** (512 bytes) | Prevents stack overflow |
| **Instruction count limit** (~1M verified) | Limits program complexity |
| **No arbitrary kernel memory access** | Only allowed through helper functions |
| **No sleeping or blocking** | eBPF programs must complete quickly |

### 1.3 eBPF Hook Points for Observability

| Hook Type | What It Observes | Examples |
|-----------|-----------------|---------|
| **kprobes** | Kernel function entry/exit | `tcp_connect`, `do_sys_open`, `vfs_write` |
| **tracepoints** | Stable kernel event points | `sched:sched_process_exec`, `net:net_dev_xmit` |
| **uprobes** | User-space function entry/exit | `SSL_write` in OpenSSL, `malloc` in libc |
| **USDT** | User-defined static tracepoints | Python GC events, MySQL query events |
| **XDP** | Network packet arrival (pre-stack) | Packet filtering, DDoS mitigation |
| **tc** | Traffic control (post-stack) | Network policy enforcement |
| **perf events** | Hardware and software counters | CPU cache misses, page faults |
| **LSM** | Linux Security Module hooks | Security policy enforcement |

---

## 2. bpftrace

### 2.1 One-Liner Examples

bpftrace is a high-level tracing language for eBPF, ideal for ad-hoc production debugging:

```bash
# Count system calls by name (what is the system doing?)
bpftrace -e 'tracepoint:syscalls:sys_enter_* { @[probe] = count(); }'

# Trace new process creation (what processes are being spawned?)
bpftrace -e 'tracepoint:sched:sched_process_exec { printf("%s executed %s\n", comm, str(args->filename)); }'

# Histogram of read() sizes (how much data are processes reading?)
bpftrace -e 'tracepoint:syscalls:sys_exit_read /args->ret > 0/ { @bytes = hist(args->ret); }'

# Trace TCP connections (who is connecting where?)
bpftrace -e 'kprobe:tcp_connect { printf("%s → %s\n", comm, ntop(((struct sock *)arg0)->__sk_common.skc_daddr)); }'

# Count file opens by process
bpftrace -e 'tracepoint:syscalls:sys_enter_openat { @[comm] = count(); }'

# Latency histogram of DNS lookups
bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo { @start[tid] = nsecs; }
             uretprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo /@start[tid]/ {
               @dns_latency_us = hist((nsecs - @start[tid]) / 1000);
               delete(@start[tid]);
             }'
```

### 2.2 Production-Ready bpftrace Script

```
#!/usr/bin/env bpftrace
/*
 * http_latency.bt -- Trace HTTP request latency in a Go service
 * Usage: bpftrace http_latency.bt -p $(pidof my-service)
 */

// Trace when HTTP handler starts
uprobe:/usr/local/bin/my-service:net/http.(*ServeMux).ServeHTTP
{
    @start[tid] = nsecs;
    @count++;
}

// Trace when HTTP handler returns
uretprobe:/usr/local/bin/my-service:net/http.(*ServeMux).ServeHTTP
/@start[tid]/
{
    $duration_us = (nsecs - @start[tid]) / 1000;
    @latency_us = hist($duration_us);

    // Alert on slow requests
    if ($duration_us > 1000000) {
        printf("SLOW REQUEST: %d us (tid=%d)\n", $duration_us, tid);
    }

    delete(@start[tid]);
}

// Print summary on Ctrl-C
END
{
    printf("\n--- HTTP Request Latency Summary ---\n");
    printf("Total requests: %d\n", @count);
    print(@latency_us);
}
```

### 2.3 bpftrace vs Traditional Tools

| Capability | Traditional Tool | bpftrace |
|-----------|-----------------|----------|
| CPU profiling | `perf top` | `profile:hz:99 { @[ustack] = count(); }` |
| Disk I/O | `iostat` | `tracepoint:block:block_rq_complete { @us = hist(args->nr_sector * 512); }` |
| Network connections | `ss`, `netstat` | `kprobe:tcp_connect { @[comm] = count(); }` |
| File system latency | N/A (no built-in) | `kprobe:vfs_read { @start[tid] = nsecs; }` |
| Function latency | N/A (needs APM) | `uprobe:function { @start[tid] = nsecs; }` |

---

## 3. BCC Tools

### 3.1 Essential BCC Tools for Observability

BCC (BPF Compiler Collection) provides production-ready tools:

```bash
# --- Network ---
# Trace TCP connections with latency
tcpconnect -t           # Show timestamp, PID, destination, latency

# Trace TCP retransmits (network reliability indicator)
tcpretrans              # Show retransmits with source, dest, state

# Summarize TCP round-trip time by remote host
tcprtt -i 1 -d 10      # 1-second intervals for 10 seconds

# --- Storage ---
# Trace block I/O latency
biolatency -m           # Histogram of block I/O latency in milliseconds

# Trace slow file system operations
ext4slower 1            # Show ext4 operations slower than 1ms

# --- CPU ---
# CPU profiling (flame graph input)
profile -af 60 > profile.out     # 60 seconds of stack sampling
flamegraph.pl profile.out > profile.svg

# --- Application ---
# Trace function latency for a specific process
funclatency -p $(pidof my-service) 'SSL_read' -m   # SSL read latency

# Trace memory allocations
memleak -p $(pidof my-service) --top=10             # Top 10 allocation sites

# --- DNS ---
# Trace DNS queries
gethostlatency          # Show DNS resolution latency per query
```

### 3.2 BCC Tools Mapping to Observability Signals

| Observability Need | BCC Tool | Output |
|-------------------|----------|--------|
| Network latency between services | `tcpconnect`, `tcprtt` | Per-connection latency |
| Disk I/O bottlenecks | `biolatency`, `biosnoop` | I/O latency histograms |
| DNS resolution issues | `gethostlatency` | Per-query resolution time |
| CPU hotspots | `profile` | Stack trace frequencies |
| Memory leaks | `memleak` | Allocation sites and sizes |
| File system slowness | `ext4slower`, `xfsslower` | Slow FS operations |
| TCP retransmits | `tcpretrans` | Network reliability issues |

---

## 4. Cilium Hubble

### 4.1 Architecture

Cilium uses eBPF for Kubernetes networking, and Hubble extends it for network observability:

```
┌─────────────────────────────────────────────┐
│              Hubble UI                       │
│         (Service Map + Flow Logs)            │
└──────────────────┬──────────────────────────┘
                   │ gRPC
┌──────────────────▼──────────────────────────┐
│           Hubble Relay                       │
│    (Aggregates from all nodes)               │
└──────┬──────────────────────────┬───────────┘
       │                          │
┌──────▼──────────┐     ┌────────▼────────────┐
│ Hubble Agent    │     │ Hubble Agent        │
│ (Node 1)        │     │ (Node 2)            │
│                 │     │                     │
│ Cilium Agent    │     │ Cilium Agent        │
│ ┌─────────────┐ │     │ ┌─────────────┐    │
│ │ eBPF Programs│ │     │ │ eBPF Programs│   │
│ │ (datapath)  │ │     │ │ (datapath)   │   │
│ └─────────────┘ │     │ └─────────────┘    │
└─────────────────┘     └────────────────────┘
```

### 4.2 Hubble CLI Examples

```bash
# Observe all network flows in a namespace
hubble observe --namespace production

# Observe traffic to/from a specific pod
hubble observe --to-pod production/payment-service-abc123

# Filter by HTTP status code (L7 visibility)
hubble observe --http-status 500 --namespace production

# Filter by DNS queries
hubble observe --protocol DNS --namespace production

# Show dropped packets (network policy violations)
hubble observe --verdict DROPPED --namespace production

# Trace traffic between two services
hubble observe \
  --from-label app=order-service \
  --to-label app=inventory-service \
  --protocol HTTP

# Export flows as JSON for analysis
hubble observe --output json --namespace production > flows.json
```

### 4.3 Hubble Metrics

Hubble exports Prometheus metrics from eBPF-observed network flows:

```yaml
# Cilium Helm values for Hubble metrics
hubble:
  enabled: true
  metrics:
    enabled:
      - dns
      - drop
      - tcp
      - flow
      - icmp
      - httpV2:exemplars=true;labelsContext=source_ip,source_namespace,source_workload,destination_ip,destination_namespace,destination_workload
    serviceMonitor:
      enabled: true
```

```promql
# HTTP request rate between services (from eBPF, zero instrumentation)
sum(rate(hubble_http_requests_total{
  source_workload="order-service",
  destination_workload="inventory-service"
}[5m]))

# DNS resolution failures
sum(rate(hubble_dns_responses_total{rcode!="No Error"}[5m])) by (rcode)

# Dropped packets by reason (network policy debugging)
sum(rate(hubble_drop_total[5m])) by (reason)

# TCP connection latency between services
histogram_quantile(0.99,
  sum by (le, source_workload, destination_workload) (
    rate(hubble_tcp_connect_duration_seconds_bucket[5m])
  )
)
```

---

## 5. eBPF-Based Automatic Instrumentation

### 5.1 Beyla (Grafana)

Beyla uses eBPF to automatically instrument HTTP and gRPC services without code changes:

```yaml
# Beyla configuration
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: beyla
spec:
  template:
    spec:
      containers:
        - name: beyla
          image: grafana/beyla:latest
          securityContext:
            privileged: true    # Required for eBPF
          env:
            - name: BEYLA_OPEN_PORT
              value: "8080,8443,3000"  # Ports to instrument
            - name: OTEL_EXPORTER_OTLP_ENDPOINT
              value: "http://otel-collector:4318"
            - name: BEYLA_SERVICE_NAMESPACE
              value: "production"
          volumeMounts:
            - name: cgroup
              mountPath: /sys/fs/cgroup
            - name: debug
              mountPath: /sys/kernel/debug
      volumes:
        - name: cgroup
          hostPath:
            path: /sys/fs/cgroup
        - name: debug
          hostPath:
            path: /sys/kernel/debug
```

**What Beyla captures automatically (no code changes):**
- HTTP request duration, status code, method, path
- gRPC request duration, status code, method
- SQL query duration (via uprobe on database libraries)
- Distributed trace context propagation (via header inspection)

### 5.2 Comparison: eBPF Auto-Instrumentation vs OTel Auto-Instrumentation

| Aspect | eBPF (Beyla, Pixie) | OTel Auto-Instrumentation |
|--------|-------------------|--------------------------|
| **Code changes** | None (observe from kernel) | Minimal (add agent/library) |
| **Language support** | Any language (kernel-level) | Language-specific (Python, Java, Go, etc.) |
| **Deployment** | DaemonSet or sidecar (privileged) | Library or agent per application |
| **Granularity** | HTTP/gRPC/SQL boundaries | Library boundaries + custom spans |
| **Business context** | None (cannot see application semantics) | Can inject custom attributes |
| **Overhead** | Very low (~1-2% CPU) | Low-Medium (3-5% CPU) |
| **Security** | Requires privileged mode | No special permissions |
| **Best for** | Quick wins, legacy apps, polyglot | Deep instrumentation, custom telemetry |

---

## 6. eBPF for Security Observability

### 6.1 Tetragon (Runtime Security)

Tetragon uses eBPF for security-relevant observability:

```yaml
# Tetragon tracing policy: detect suspicious file access
apiVersion: cilium.io/v1alpha1
kind: TracingPolicy
metadata:
  name: sensitive-file-access
spec:
  kprobes:
    - call: "security_file_open"
      syscall: false
      args:
        - index: 0
          type: "file"
      selectors:
        - matchArgs:
            - index: 0
              operator: "Prefix"
              values:
                - "/etc/shadow"
                - "/etc/passwd"
                - "/root/.ssh"
                - "/var/run/secrets/kubernetes.io"
      return: true
      returnArg:
        index: 0
        type: "int"
```

```bash
# Observe security events
tetra getevents --namespace production

# Output:
# process: /usr/bin/cat
# args: /etc/shadow
# pod: production/compromised-pod
# action: ALERT
```

### 6.2 Network Policy Observability

```bash
# See which network policies are being enforced
hubble observe --verdict DROPPED -o json | jq '.flow.drop_reason'

# Common drop reasons:
# - POLICY_DENIED: Cilium NetworkPolicy blocked the traffic
# - UNSUPPORTED_L3_PROTOCOL: Unknown protocol
# - CT_MAP_INSERTION_FAILED: Connection tracking table full (scale issue)
```

---

## 7. When to Use eBPF vs OpenTelemetry

### 7.1 Decision Framework

```
Need to observe?
    │
    ├── Application business logic (order created, payment processed)
    │   └── Use OpenTelemetry (custom spans, metrics, logs)
    │
    ├── HTTP/gRPC request patterns (no code access)
    │   └── Use eBPF (Beyla, Hubble)
    │
    ├── Network communication between services
    │   └── Use Cilium Hubble (eBPF)
    │
    ├── Kernel-level performance (syscalls, I/O, TCP)
    │   └── Use eBPF (bpftrace, BCC tools)
    │
    ├── Security events (file access, process execution)
    │   └── Use eBPF (Tetragon, Falco)
    │
    └── Both business context AND network/kernel details
        └── Use both: OTel for app-level, eBPF for infra-level
```

### 7.2 Complementary Usage

The best observability stacks use both:

```
┌─────────────────────────────────────────────────┐
│ Application Layer (OTel)                         │
│ - Business metrics (orders, revenue)            │
│ - Custom spans with domain attributes           │
│ - Structured logs with business context         │
├─────────────────────────────────────────────────┤
│ Service Communication Layer (eBPF / Hubble)      │
│ - HTTP/gRPC golden signals (zero instrumentation)│
│ - Service dependency map                         │
│ - Network policy enforcement                     │
├─────────────────────────────────────────────────┤
│ Kernel / Infrastructure Layer (eBPF / BCC)       │
│ - System call latency                           │
│ - TCP connection quality                        │
│ - Disk I/O patterns                             │
│ - CPU scheduling                                │
└─────────────────────────────────────────────────┘
```

---

## 8. Practical eBPF Debugging Recipes

### 8.1 "Why Is This Service Slow?"

```bash
# Step 1: Is it CPU-bound?
bpftrace -e 'profile:hz:99 /comm == "payment-svc"/ { @[ustack(5)] = count(); }' -c 'sleep 10'

# Step 2: Is it I/O-bound?
bpftrace -e 'tracepoint:syscalls:sys_exit_read /comm == "payment-svc" && args->ret > 0/ {
  @read_latency = hist(nsecs - @start[tid]);
}'

# Step 3: Is it network-bound?
tcpconnect -p $(pidof payment-svc) -t     # Show TCP connect latency
tcpretrans -p $(pidof payment-svc)         # Show retransmits

# Step 4: Is it DNS?
gethostlatency -p $(pidof payment-svc)     # Show DNS resolution time

# Step 5: Is it lock contention?
bpftrace -e 'uprobe:/lib/libpthread.so:pthread_mutex_lock /comm == "payment-svc"/ {
  @start[tid] = nsecs;
}
uretprobe:/lib/libpthread.so:pthread_mutex_lock /comm == "payment-svc" && @start[tid]/ {
  @lock_hold_us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}'
```

### 8.2 "Why Are Connections Failing?"

```bash
# Trace all TCP connection attempts and their results
bpftrace -e '
kprobe:tcp_connect {
    @conn[tid] = nsecs;
    @dest[tid] = ntop(((struct sock *)arg0)->__sk_common.skc_daddr);
}
kretprobe:tcp_connect /@conn[tid]/ {
    $ret = retval;
    $latency_ms = (nsecs - @conn[tid]) / 1000000;
    if ($ret != 0) {
        printf("FAILED: %s → %s (err=%d, %dms)\n", comm, @dest[tid], $ret, $latency_ms);
    }
    delete(@conn[tid]);
    delete(@dest[tid]);
}'

# Check for TCP reset storms
tcpretrans -l     # Show with loss type (retransmit vs timeout)
```

---

## 9. Limitations and Considerations

### 9.1 eBPF Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| **Requires Linux kernel 4.14+** | No macOS, no Windows, no old kernels | Use OTel on non-Linux platforms |
| **Requires privileged or CAP_BPF** | Security concern in shared environments | Use dedicated observability nodes |
| **No application context** | Cannot see business logic, user IDs | Combine with OTel for app-level data |
| **Stack walking limitations** | Go, Rust, JIT languages have complex stacks | Use language-specific frame pointer settings |
| **Overhead at high probe rates** | Tracing every syscall can add overhead | Sample or filter probes; avoid hot-path probes |

---

## 10. Next Steps

- [25_Continuous_Profiling.md](./25_Continuous_Profiling.md) -- CPU and memory profiling in production
- [26_Incident_Response.md](./26_Incident_Response.md) -- On-call practices and incident management

---

## Exercises

### Exercise 1: bpftrace Probe Design

Write bpftrace one-liners for each of these production debugging scenarios:

1. A service is suspected of making too many DNS lookups. Show the DNS query count per second grouped by process name.
2. A database-heavy service has latency spikes. Trace all `write()` syscalls for a specific PID and show a latency histogram.
3. You suspect a container is writing large files to disk. Track all `vfs_write` calls and show the total bytes written per process.

<details>
<summary>Show Answer</summary>

**1. DNS query rate by process:**
```bash
bpftrace -e 'uprobe:/lib/x86_64-linux-gnu/libc.so.6:getaddrinfo {
  @dns_queries[comm] = count();
}
interval:s:1 {
  print(@dns_queries);
  clear(@dns_queries);
}'
```

**2. Write syscall latency histogram for a specific PID:**
```bash
bpftrace -e '
tracepoint:syscalls:sys_enter_write /pid == 12345/ {
  @start[tid] = nsecs;
}
tracepoint:syscalls:sys_exit_write /pid == 12345 && @start[tid]/ {
  @write_latency_us = hist((nsecs - @start[tid]) / 1000);
  delete(@start[tid]);
}'
```

**3. Bytes written per process via vfs_write:**
```bash
bpftrace -e 'kretprobe:vfs_write /retval > 0/ {
  @bytes_written[comm] = sum(retval);
}
interval:s:5 {
  print(@bytes_written);
  clear(@bytes_written);
}'
```

</details>

### Exercise 2: eBPF vs OTel Decision

For each scenario, decide whether to use eBPF-based observability, OpenTelemetry, or both. Justify your answer.

1. A legacy Java application (no source code access) running on Kubernetes needs basic HTTP metrics.
2. A new Python microservice needs detailed business transaction tracing with custom attributes.
3. You need to identify which Kubernetes pods are causing excessive DNS lookups.
4. A Go service has occasional latency spikes that correlate with garbage collection.
5. You need to enforce and monitor network policies between services.

<details>
<summary>Show Answer</summary>

**1. Legacy Java app, no source code → eBPF (Beyla or Pixie)**
- No code changes possible, so OTel auto-instrumentation is not an option unless you can modify the deployment (add Java agent).
- eBPF (Beyla) can observe HTTP request patterns from the kernel without touching the application.
- If you CAN modify the deployment command line (add `-javaagent:opentelemetry-javaagent.jar`), OTel auto-instrumentation is better (richer data).

**2. New Python microservice with custom attributes → OpenTelemetry**
- Custom business attributes (order_id, customer_tier, payment_method) require application-level instrumentation.
- eBPF cannot see these application-level concepts.
- Use OTel SDK with manual spans for business-critical paths plus auto-instrumentation for HTTP/DB.

**3. Excessive DNS lookups per pod → eBPF (Cilium Hubble or bpftrace)**
- DNS is a kernel-level activity; eBPF can observe it without any application changes.
- Hubble provides per-pod DNS query metrics out of the box.
- OTel cannot observe DNS lookups (they happen in the C library, below application code).

**4. Go GC-related latency spikes → Both**
- eBPF for kernel-level observation: CPU scheduling, memory allocation patterns.
- OTel for application-level: trace the specific requests affected by GC pauses.
- Go runtime exposes GC metrics (`runtime/metrics`) that OTel can collect.
- bpftrace can trace Go GC functions directly via uprobes for precise timing.

**5. Network policy enforcement monitoring → eBPF (Cilium Hubble / Tetragon)**
- Network policies are enforced at the kernel level by Cilium's eBPF programs.
- Hubble provides verdict (ALLOWED/DROPPED) with source and destination context.
- OTel has no visibility into network policy enforcement.

</details>

---

## References

- [eBPF.io -- Official eBPF Documentation](https://ebpf.io/)
- [BPF Performance Tools (Brendan Gregg)](https://www.brendangregg.com/bpf-performance-tools-book.html)
- [bpftrace Reference Guide](https://github.com/bpftrace/bpftrace/blob/master/docs/reference_guide.md)
- [Cilium Hubble Documentation](https://docs.cilium.io/en/stable/observability/)
- [Grafana Beyla](https://grafana.com/docs/beyla/latest/)
- [Tetragon -- eBPF Security Observability](https://tetragon.io/)
