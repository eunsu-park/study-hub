# Exercise: Job Queues
# Practice with task queues, retry logic, priority scheduling, and dead letter handling.

import time
import uuid
import threading
from typing import Callable, Optional
from dataclasses import dataclass, field


# Exercise 1: Priority Job Queue
class PriorityQueue:
    """Thread-safe priority queue (lower number = higher priority)."""

    def __init__(self):
        self._items = []  # list of (priority, insertion_order, item)
        self._counter = 0
        self._lock = threading.Lock()

    def push(self, item, priority: int = 0):
        """Add item with priority. Items with same priority are FIFO."""
        # TODO: Implement
        pass

    def pop(self):
        """Remove and return highest-priority item. Return None if empty."""
        # TODO: Implement
        pass

    def peek(self):
        """Return highest-priority item without removing. Return None if empty."""
        # TODO: Implement
        pass

    def __len__(self):
        """Return number of items in queue."""
        # TODO: Implement
        pass


# Test
# q = PriorityQueue()
# q.push("low", priority=10)
# q.push("high", priority=1)
# q.push("medium", priority=5)
# assert q.pop() == "high"
# assert q.pop() == "medium"
# assert q.pop() == "low"
# assert q.pop() is None


# Exercise 2: Retry with Exponential Backoff
def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 0.01,
    max_delay: float = 1.0,
    retryable_exceptions: tuple = (Exception,),
) -> dict:
    """Execute func with exponential backoff retry.

    Backoff formula: min(base_delay * 2^attempt, max_delay)

    Returns: {
        "success": bool,
        "result": <return value or None>,
        "attempts": int,
        "total_delay": float,  # total seconds spent waiting
        "errors": [str, ...],
    }
    """
    # TODO: Implement
    pass


# Test
# state = {"calls": 0}
# def flaky():
#     state["calls"] += 1
#     if state["calls"] < 3:
#         raise ConnectionError("unavailable")
#     return "ok"
# result = retry_with_backoff(flaky, max_retries=5, base_delay=0.001)
# assert result["success"] is True
# assert result["attempts"] == 3
# assert len(result["errors"]) == 2


# Exercise 3: Job Worker Pool
@dataclass
class Job:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = ""
    payload: dict = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[str] = None
    error: Optional[str] = None


class WorkerPool:
    """Process jobs from a queue using a pool of worker threads."""

    def __init__(self, num_workers: int = 2):
        self.num_workers = num_workers
        self._handlers = {}  # job_name -> handler_fn
        self._completed = []
        self._failed = []
        self._lock = threading.Lock()

    def register(self, name: str, handler: Callable):
        """Register a handler for a job type. handler(payload) -> result."""
        # TODO: Implement
        pass

    def process(self, jobs: list[Job]) -> dict:
        """Process all jobs using thread pool.

        Returns: {"completed": int, "failed": int, "results": [Job, ...]}
        """
        # TODO: Implement
        # 1. For each job, find handler by name
        # 2. Execute handler, catch exceptions
        # 3. Update job status and result/error
        # 4. Collect results thread-safely
        pass


# Test
# pool = WorkerPool(num_workers=2)
# pool.register("greet", lambda p: f"Hello {p['name']}")
# pool.register("fail", lambda p: (_ for _ in ()).throw(RuntimeError("boom")))
# jobs = [
#     Job(name="greet", payload={"name": "Alice"}),
#     Job(name="greet", payload={"name": "Bob"}),
#     Job(name="fail", payload={}),
# ]
# result = pool.process(jobs)
# assert result["completed"] == 2
# assert result["failed"] == 1


# Exercise 4: Dead Letter Queue
class DeadLetterQueue:
    """Capture permanently failed jobs for later inspection."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._items = []

    def add(self, job: Job, reason: str):
        """Add a failed job with failure reason. Evict oldest if at max_size."""
        # TODO: Implement
        pass

    def list(self, limit: int = 10) -> list[dict]:
        """Return most recent dead-lettered jobs.

        Each entry: {"job_id": str, "name": str, "reason": str, "payload": dict}
        """
        # TODO: Implement
        pass

    def retry(self, job_id: str) -> Optional[Job]:
        """Remove a job from DLQ and return it for reprocessing.
        Reset status to 'pending'. Return None if not found.
        """
        # TODO: Implement
        pass

    def purge(self) -> int:
        """Remove all items. Return count removed."""
        # TODO: Implement
        pass


# Test
# dlq = DeadLetterQueue(max_size=5)
# j = Job(id="abc", name="send_email", payload={"to": "x@y.com"})
# dlq.add(j, "SMTP timeout after 3 retries")
# items = dlq.list()
# assert len(items) == 1
# assert items[0]["reason"] == "SMTP timeout after 3 retries"
# retried = dlq.retry("abc")
# assert retried.status == "pending"
# assert dlq.list() == []


# Exercise 5: Periodic Scheduler
class Scheduler:
    """Run functions on fixed intervals."""

    def __init__(self):
        self._tasks = []
        self._stop = threading.Event()

    def every(self, seconds: float, func: Callable, name: str = ""):
        """Register a periodic task."""
        # TODO: Implement
        pass

    def run_for(self, duration: float):
        """Run all scheduled tasks for `duration` seconds, then stop.

        Check tasks every 0.05 seconds.
        """
        # TODO: Implement
        pass

    def get_run_counts(self) -> dict[str, int]:
        """Return {task_name: run_count} for each registered task."""
        # TODO: Implement
        pass


# Test
# s = Scheduler()
# counter = {"a": 0, "b": 0}
# s.every(0.1, lambda: counter.__setitem__("a", counter["a"] + 1), name="fast")
# s.every(0.3, lambda: counter.__setitem__("b", counter["b"] + 1), name="slow")
# s.run_for(0.5)
# assert counter["a"] >= 4  # ~5 runs in 0.5s at 0.1s interval
# assert counter["b"] >= 1  # ~1-2 runs in 0.5s at 0.3s interval


if __name__ == "__main__":
    print("Job Queues Exercise")
    print("Implement each class/function and verify with the test cases.")
