"""
Serverless Function Execution Simulation

Demonstrates serverless computing (Lambda / Cloud Functions) concepts:
- Cold start vs warm start latency and its impact on user experience
- Concurrent invocations with container reuse
- Pay-per-invocation cost modeling
- Timeout handling and retry behavior
- Memory/duration trade-offs

No cloud account required -- all behavior is simulated locally.
"""

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FunctionConfig:
    """Configuration for a serverless function.
    Key insight: memory allocation also determines CPU share -- doubling memory
    doubles CPU, which can make functions run faster and sometimes cost LESS."""
    name: str
    memory_mb: int = 128         # 128 MB to 10,240 MB (AWS Lambda)
    timeout_sec: float = 30.0    # Max execution time before forced termination
    reserved_concurrency: int = 100  # Max simultaneous executions


@dataclass
class ExecutionContainer:
    """Represents a warm container that can be reused for subsequent invocations.
    Serverless platforms keep containers alive for ~5-15 minutes after last use.
    Reusing a warm container avoids the cold start penalty."""
    container_id: str
    function_name: str
    created_at: float
    last_used: float
    warm: bool = True
    ttl_sec: float = 300.0  # Container lives for 5 min after last use

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.last_used) > self.ttl_sec


@dataclass
class InvocationResult:
    """Result of a single serverless function invocation."""
    request_id: str
    cold_start: bool
    init_duration_ms: float    # Cold start overhead (0 if warm)
    exec_duration_ms: float    # Actual function execution time
    total_duration_ms: float   # init + exec
    billed_duration_ms: float  # Rounded up to nearest 1ms (AWS) or 100ms (GCP)
    memory_used_mb: int
    cost_usd: float
    timed_out: bool = False
    error: Optional[str] = None


class ServerlessRuntime:
    """Simulates a serverless execution environment with container management."""

    # AWS Lambda pricing (approximate): $0.20 per 1M requests + duration cost
    REQUEST_COST = 0.0000002        # $0.20 per million
    DURATION_COST_PER_GB_SEC = 0.0000166667  # ~$0.0000166667 per GB-second

    def __init__(self):
        self.containers: List[ExecutionContainer] = []
        self._next_container_id = 1
        self._next_request_id = 1
        self.invocation_log: List[InvocationResult] = []

    def _find_warm_container(self, function_name: str) -> Optional[ExecutionContainer]:
        """Try to find a warm container for reuse.
        This is the core optimization in serverless -- avoiding cold starts
        by reusing existing execution environments."""
        for container in self.containers:
            if (container.function_name == function_name
                    and container.warm
                    and not container.is_expired):
                return container
        return None

    def _create_container(self, function_name: str) -> ExecutionContainer:
        """Spin up a new container -- this is the 'cold start'.
        Cold starts involve: downloading code, setting up runtime, importing
        dependencies. Can take 100ms-10s depending on language and package size."""
        container = ExecutionContainer(
            container_id=f"ctr-{self._next_container_id:04d}",
            function_name=function_name,
            created_at=time.monotonic(),
            last_used=time.monotonic(),
        )
        self._next_container_id += 1
        self.containers.append(container)
        return container

    def invoke(self, config: FunctionConfig,
               simulated_exec_ms: float = 50.0) -> InvocationResult:
        """Invoke a serverless function, simulating real execution behavior.

        The platform either reuses a warm container (fast) or creates a new
        one (cold start penalty). Either way, you only pay for what you use.
        """
        request_id = f"req-{self._next_request_id:06d}"
        self._next_request_id += 1

        # Check for warm container reuse
        container = self._find_warm_container(config.name)
        cold_start = container is None

        if cold_start:
            container = self._create_container(config.name)
            # Cold start latency depends on memory (more memory = faster init)
            # and runtime (Python/Node ~200-500ms, Java/C# ~500-3000ms)
            init_duration = random.uniform(150, 500) * (256 / config.memory_mb)
        else:
            init_duration = 0.0

        container.last_used = time.monotonic()

        # Simulate execution -- more memory means proportionally more CPU,
        # which can reduce execution time for CPU-bound workloads
        cpu_factor = 128 / config.memory_mb  # Higher memory = faster
        actual_exec = simulated_exec_ms * cpu_factor + random.uniform(-5, 10)
        actual_exec = max(1.0, actual_exec)

        # Check timeout
        total_duration = init_duration + actual_exec
        timed_out = (total_duration / 1000) > config.timeout_sec

        if timed_out:
            total_duration = config.timeout_sec * 1000
            actual_exec = total_duration - init_duration
            error = f"Task timed out after {config.timeout_sec}s"
        else:
            error = None

        # Billing: round up to nearest 1ms, minimum 1ms
        billed_ms = max(1.0, round(total_duration))

        # Cost = request cost + (memory_GB * duration_seconds * rate)
        memory_gb = config.memory_mb / 1024
        duration_sec = billed_ms / 1000
        cost = self.REQUEST_COST + (memory_gb * duration_sec * self.DURATION_COST_PER_GB_SEC)

        memory_used = int(config.memory_mb * random.uniform(0.3, 0.8))

        result = InvocationResult(
            request_id=request_id,
            cold_start=cold_start,
            init_duration_ms=round(init_duration, 1),
            exec_duration_ms=round(actual_exec, 1),
            total_duration_ms=round(total_duration, 1),
            billed_duration_ms=billed_ms,
            memory_used_mb=memory_used,
            cost_usd=cost,
            timed_out=timed_out,
            error=error,
        )
        self.invocation_log.append(result)
        return result


def demonstrate_cold_vs_warm():
    """Show the difference between cold start and warm start invocations."""
    print("=" * 75)
    print("Cold Start vs Warm Start")
    print("=" * 75)

    runtime = ServerlessRuntime()
    config = FunctionConfig("process-image", memory_mb=512, timeout_sec=30)

    for i in range(5):
        result = runtime.invoke(config, simulated_exec_ms=80)
        start_type = "COLD" if result.cold_start else "WARM"
        print(f"  Invocation {i+1} [{start_type}]: "
              f"init={result.init_duration_ms:>6.1f}ms + "
              f"exec={result.exec_duration_ms:>6.1f}ms = "
              f"total={result.total_duration_ms:>7.1f}ms  "
              f"cost=${result.cost_usd:.8f}")
    print()


def demonstrate_memory_tradeoff():
    """Show how memory allocation affects performance and cost.
    Counter-intuitive insight: more memory can be CHEAPER because the function
    runs faster, and you pay for memory * time."""
    print("=" * 75)
    print("Memory vs Performance Trade-off")
    print("=" * 75)

    memory_configs = [128, 256, 512, 1024, 2048]

    for mem in memory_configs:
        runtime = ServerlessRuntime()
        config = FunctionConfig("data-transform", memory_mb=mem)
        # Run 100 invocations to get average
        for _ in range(100):
            runtime.invoke(config, simulated_exec_ms=200)

        results = runtime.invocation_log
        avg_duration = sum(r.total_duration_ms for r in results) / len(results)
        avg_cost = sum(r.cost_usd for r in results) / len(results)
        total_cost = sum(r.cost_usd for r in results)

        print(f"  {mem:>5} MB: avg_duration={avg_duration:>8.1f}ms  "
              f"avg_cost=${avg_cost:.8f}  total_100=${total_cost:.6f}")
    print()


def simulate_concurrent_burst():
    """Simulate a burst of concurrent requests hitting the function.
    When burst exceeds warm containers, new cold starts are triggered.
    Reserved concurrency limits protect downstream dependencies."""
    print("=" * 75)
    print("Concurrent Burst Simulation (50 simultaneous requests)")
    print("=" * 75)

    runtime = ServerlessRuntime()
    config = FunctionConfig("api-handler", memory_mb=256, reserved_concurrency=30)

    burst_size = 50
    cold_count = 0
    throttled_count = 0

    for i in range(burst_size):
        if i >= config.reserved_concurrency:
            # Requests beyond reserved concurrency are throttled (HTTP 429)
            throttled_count += 1
            continue
        result = runtime.invoke(config, simulated_exec_ms=30)
        if result.cold_start:
            cold_count += 1

    warm_count = burst_size - cold_count - throttled_count
    print(f"  Total requests:  {burst_size}")
    print(f"  Cold starts:     {cold_count} (new containers provisioned)")
    print(f"  Warm reuses:     {warm_count}")
    print(f"  Throttled (429): {throttled_count} (exceeded reserved concurrency of "
          f"{config.reserved_concurrency})")
    print()


def cost_comparison():
    """Compare serverless cost vs always-on VM for varying request volumes."""
    print("=" * 75)
    print("Cost Comparison: Serverless vs Always-On VM (monthly)")
    print("=" * 75)

    vm_monthly = 0.0416 * 730  # t3.medium on-demand, ~$30.37/month

    print(f"  {'Requests/month':<20} {'Lambda Cost':>14} {'VM Cost':>14} {'Winner':>10}")
    print(f"  {'-'*60}")

    for requests in [10_000, 100_000, 1_000_000, 10_000_000, 50_000_000]:
        # Average 100ms execution, 256MB
        gb_sec = requests * 0.1 * (256 / 1024)
        lambda_cost = (requests * 0.0000002) + (gb_sec * 0.0000166667)
        winner = "Lambda" if lambda_cost < vm_monthly else "VM"
        print(f"  {requests:>15,}     ${lambda_cost:>11.2f}   ${vm_monthly:>11.2f}   {winner:>8}")

    # Key takeaway: serverless is cheaper at low/medium traffic,
    # but VMs win at high, sustained traffic
    print(f"\n  Insight: Serverless excels for sporadic/bursty workloads.")
    print(f"  For sustained high-traffic, a reserved VM is more cost-effective.")
    print()


if __name__ == "__main__":
    random.seed(42)
    demonstrate_cold_vs_warm()
    demonstrate_memory_tradeoff()
    simulate_concurrent_burst()
    cost_comparison()
