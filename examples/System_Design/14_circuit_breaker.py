"""
Circuit Breaker Pattern

Demonstrates:
- Circuit breaker state machine (Closed → Open → Half-Open)
- Failure counting and threshold
- Automatic recovery with half-open probing
- Metrics tracking

Theory:
- The circuit breaker pattern prevents cascading failures by
  stopping calls to a failing service.
- States:
  - CLOSED: Normal operation, requests pass through.
    Track failures; trip to OPEN when threshold exceeded.
  - OPEN: All requests fail immediately (fast-fail).
    After timeout, transition to HALF_OPEN.
  - HALF_OPEN: Allow limited probe requests.
    If probe succeeds, go to CLOSED. If fails, back to OPEN.
- Benefits: fail fast, reduce load on failing services,
  automatic recovery detection.

Adapted from System Design Lesson 14.
"""

import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable, Any


class State(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    rejected_calls: int = 0
    state_transitions: list[tuple[float, str, str]] = field(
        default_factory=list
    )


# Why: The circuit breaker pattern is borrowed from electrical engineering — when
# a downstream service is failing, "open the circuit" to fail fast rather than
# waiting for timeouts. This prevents cascading failures across microservices.
class CircuitBreaker:
    """Circuit breaker with configurable thresholds."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
        # Why: Requiring multiple consecutive successes in half-open state
        # prevents the breaker from closing prematurely on a single lucky
        # request while the downstream service is still flapping.
        success_threshold: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self.success_threshold = success_threshold

        self.state = State.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_failure_time = 0.0
        self.metrics = CircuitBreakerMetrics()

    def _transition(self, new_state: State, now: float) -> None:
        old = self.state.value
        self.state = new_state
        self.metrics.state_transitions.append((now, old, new_state.value))

    def call(self, func: Callable, *args: Any,
             now: float | None = None, **kwargs: Any) -> Any:
        """Execute func through the circuit breaker."""
        current_time = now if now is not None else time.monotonic()
        self.metrics.total_calls += 1

        if self.state == State.OPEN:
            # Why: The recovery timeout gives the failing service time to recover
            # before we probe it again. Without this cooldown, we would either
            # never retry (permanent failure) or retry too aggressively.
            if current_time - self.last_failure_time >= self.recovery_timeout:
                self._transition(State.HALF_OPEN, current_time)
                self.half_open_calls = 0
                self.success_count = 0
            else:
                self.metrics.rejected_calls += 1
                raise CircuitOpenError(
                    f"Circuit OPEN. Retry after "
                    f"{self.recovery_timeout - (current_time - self.last_failure_time):.1f}s"
                )

        # Why: Limiting probe calls in half-open state prevents overwhelming
        # a recovering service with too many test requests at once.
        if self.state == State.HALF_OPEN:
            if self.half_open_calls >= self.half_open_max_calls:
                self.metrics.rejected_calls += 1
                raise CircuitOpenError("Half-open: max probe calls reached")
            self.half_open_calls += 1

        # Execute the call
        try:
            result = func(*args, **kwargs)
            self._on_success(current_time)
            return result
        except Exception as e:
            self._on_failure(current_time)
            raise

    def _on_success(self, now: float) -> None:
        self.metrics.successful_calls += 1
        if self.state == State.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self._transition(State.CLOSED, now)
                self.failure_count = 0
        elif self.state == State.CLOSED:
            # Why: Reset failure count on success so that occasional transient
            # errors don't accumulate over time and trip the breaker falsely.
            self.failure_count = 0

    def _on_failure(self, now: float) -> None:
        self.metrics.failed_calls += 1
        self.last_failure_time = now

        if self.state == State.HALF_OPEN:
            self._transition(State.OPEN, now)
        elif self.state == State.CLOSED:
            self.failure_count += 1
            if self.failure_count >= self.failure_threshold:
                self._transition(State.OPEN, now)

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


class CircuitOpenError(Exception):
    pass


# ── Simulation (non-real-time) ─────────────────────────────────────────

class SimulatedService:
    """A service that can be configured to fail."""

    def __init__(self):
        self.fail_until: float = 0.0
        self.call_count = 0

    def call(self, timestamp: float = 0.0) -> str:
        self.call_count += 1
        if timestamp < self.fail_until:
            raise ConnectionError("Service unavailable")
        return "OK"


# ── Demos ──────────────────────────────────────────────────────────────

def demo_basic():
    print("=" * 60)
    print("CIRCUIT BREAKER STATE MACHINE")
    print("=" * 60)

    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=5.0,
        half_open_max_calls=2,
        success_threshold=2,
    )
    service = SimulatedService()
    service.fail_until = 8.0  # Service fails for first 8 seconds

    print(f"\n  Config: failure_threshold=3, recovery_timeout=5s")
    print(f"  Service fails until t=8s\n")
    print(f"  {'Time':>6}  {'State':<12}  {'Result':<20}  Notes")
    print(f"  {'-'*6}  {'-'*12}  {'-'*20}  {'-'*20}")

    for t in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
        t = float(t)
        state_before = cb.state.value
        try:
            result = cb.call(service.call, t, now=t)
            note = ""
            if state_before == "HALF_OPEN":
                note = f"probe success ({cb.success_count}/{cb.success_threshold})"
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'SUCCESS':<20}  {note}")
        except CircuitOpenError:
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'REJECTED (fast)':<20}  "
                  f"circuit open")
        except ConnectionError:
            note = ""
            if state_before == "CLOSED":
                note = f"failures: {cb.failure_count}/{cb.failure_threshold}"
            elif state_before == "HALF_OPEN":
                note = "probe failed → OPEN"
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'FAILED':<20}  {note}")


def demo_metrics():
    print("\n" + "=" * 60)
    print("CIRCUIT BREAKER METRICS")
    print("=" * 60)

    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=5.0)
    service = SimulatedService()
    service.fail_until = 7.0

    # Run 20 calls over 15 seconds
    for t in range(15):
        try:
            cb.call(service.call, float(t), now=float(t))
        except (CircuitOpenError, ConnectionError):
            pass

    m = cb.metrics
    print(f"\n  Total calls:      {m.total_calls}")
    print(f"  Successful:       {m.successful_calls}")
    print(f"  Failed:           {m.failed_calls}")
    print(f"  Rejected (fast):  {m.rejected_calls}")

    print(f"\n  State transitions:")
    for ts, old, new in m.state_transitions:
        print(f"    t={ts:>5.0f}s: {old} → {new}")


def demo_recovery_patterns():
    print("\n" + "=" * 60)
    print("RECOVERY PATTERNS")
    print("=" * 60)

    # Scenario: intermittent failures
    cb = CircuitBreaker(
        failure_threshold=3,
        recovery_timeout=2.0,
        success_threshold=2,
    )

    # Define failure pattern: fail at specific times
    fail_times = {0, 1, 2, 5, 10, 11}  # time → fail

    print(f"\n  Intermittent failure pattern:")
    print(f"  Fail at times: {sorted(fail_times)}\n")

    print(f"  {'Time':>6}  {'State':<12}  {'Action':<15}  {'Result'}")
    print(f"  {'-'*6}  {'-'*12}  {'-'*15}  {'-'*15}")

    for t in range(15):
        t = float(t)
        state_before = cb.state.value

        def service_call() -> str:
            if t in fail_times:
                raise ConnectionError("fail")
            return "OK"

        try:
            result = cb.call(service_call, now=t)
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'call':<15}  SUCCESS")
        except CircuitOpenError:
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'fast-fail':<15}  REJECTED")
        except ConnectionError:
            print(f"  {t:>5.0f}s  {cb.state.value:<12}  {'call':<15}  FAILED")


def demo_comparison():
    """Compare with vs without circuit breaker."""
    print("\n" + "=" * 60)
    print("WITH vs WITHOUT CIRCUIT BREAKER")
    print("=" * 60)

    fail_times = set(range(5, 50))  # Service down from t=5 to t=49

    # Without circuit breaker: every call waits for timeout
    timeout_ms = 500
    no_cb_time = 0
    no_cb_calls = 0
    for t in range(60):
        no_cb_calls += 1
        if t in fail_times:
            no_cb_time += timeout_ms  # Full timeout wait
        else:
            no_cb_time += 10  # Normal response

    # With circuit breaker
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10.0)
    cb_time = 0
    cb_calls = 0
    for t in range(60):
        cb_calls += 1
        t_float = float(t)
        try:
            def svc():
                if t in fail_times:
                    raise ConnectionError()
                return "OK"
            cb.call(svc, now=t_float)
            cb_time += 10  # Normal response
        except CircuitOpenError:
            cb_time += 1  # Fast fail
        except ConnectionError:
            cb_time += timeout_ms  # Full timeout

    print(f"\n  Scenario: service down from t=5 to t=49 (45s)")
    print(f"  60 total calls, timeout = {timeout_ms}ms\n")
    print(f"  {'Metric':<30} {'No CB':>10} {'With CB':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    print(f"  {'Total time (ms)':<30} {no_cb_time:>10} {cb_time:>10}")
    print(f"  {'Avg latency (ms)':<30} {no_cb_time/no_cb_calls:>10.0f} "
          f"{cb_time/cb_calls:>10.0f}")
    savings = (1 - cb_time / no_cb_time) * 100
    print(f"\n  Circuit breaker saves {savings:.0f}% of wasted time!")


if __name__ == "__main__":
    demo_basic()
    demo_metrics()
    demo_recovery_patterns()
    demo_comparison()
