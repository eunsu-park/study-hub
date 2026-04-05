"""
Job Queues — Background Tasks, Task Queues, Scheduling
Demonstrates: in-process background workers, priority queue, retry logic,
              dead letter queue, and periodic scheduling.

Run: python 21_job_queues.py
"""

import json
import time
import uuid
import heapq
import threading
from enum import Enum
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Callable, Optional
from concurrent.futures import ThreadPoolExecutor


# --- 1. Job Status and Definition ---

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"  # exceeded max retries


@dataclass
class Job:
    id: str
    name: str
    payload: dict
    priority: int = 0  # lower = higher priority
    max_retries: int = 3
    retry_count: int = 0
    status: JobStatus = JobStatus.PENDING
    result: Optional[str] = None
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other):
        return self.priority < other.priority


# --- 2. In-Memory Job Queue ---

class JobQueue:
    """Priority-based job queue with retry and dead letter support."""

    def __init__(self, workers: int = 3):
        self._heap: list[Job] = []
        self._lock = threading.Lock()
        self._handlers: dict[str, Callable] = {}
        self._dead_letter: list[Job] = []
        self._completed: list[Job] = []
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._running = True

    def register(self, name: str, handler: Callable):
        """Register a handler function for a job type."""
        self._handlers[name] = handler

    def enqueue(self, name: str, payload: dict, priority: int = 0, max_retries: int = 3) -> str:
        job = Job(id=uuid.uuid4().hex[:8], name=name, payload=payload,
                  priority=priority, max_retries=max_retries)
        with self._lock:
            heapq.heappush(self._heap, job)
        print(f"  [ENQUEUE] {job.id} ({job.name}) priority={job.priority}")
        return job.id

    def _process_one(self) -> bool:
        with self._lock:
            if not self._heap:
                return False
            job = heapq.heappop(self._heap)

        handler = self._handlers.get(job.name)
        if not handler:
            print(f"  [ERROR] No handler for {job.name}")
            return True

        job.status = JobStatus.RUNNING
        try:
            result = handler(job.payload)
            job.status = JobStatus.COMPLETED
            job.result = str(result)
            self._completed.append(job)
            print(f"  [DONE] {job.id} ({job.name}) -> {result}")
        except Exception as e:
            job.retry_count += 1
            if job.retry_count >= job.max_retries:
                job.status = JobStatus.DEAD
                self._dead_letter.append(job)
                print(f"  [DEAD] {job.id} ({job.name}) after {job.retry_count} retries: {e}")
            else:
                job.status = JobStatus.PENDING
                with self._lock:
                    heapq.heappush(self._heap, job)
                backoff = 2 ** job.retry_count * 0.01  # exponential backoff (scaled down)
                print(f"  [RETRY] {job.id} ({job.name}) attempt {job.retry_count}, backoff={backoff:.2f}s")
                time.sleep(backoff)
        return True

    def process_all(self):
        """Process all queued jobs synchronously."""
        while self._process_one():
            pass

    @property
    def dead_letter_count(self) -> int:
        return len(self._dead_letter)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def stats(self) -> dict:
        return {
            "pending": len(self._heap),
            "completed": self.completed_count,
            "dead_letter": self.dead_letter_count,
        }


# --- 3. Periodic Scheduler ---

class PeriodicScheduler:
    """Run functions on a fixed interval (cron-like)."""

    def __init__(self):
        self._tasks: list[dict] = []
        self._stop_event = threading.Event()

    def every(self, seconds: float, func: Callable, name: str = ""):
        self._tasks.append({"func": func, "interval": seconds, "name": name or func.__name__,
                            "last_run": 0.0})

    def start(self):
        def _loop():
            while not self._stop_event.is_set():
                now = time.time()
                for task in self._tasks:
                    if now - task["last_run"] >= task["interval"]:
                        try:
                            task["func"]()
                            task["last_run"] = now
                        except Exception as e:
                            print(f"  [SCHED ERROR] {task['name']}: {e}")
                self._stop_event.wait(0.1)
        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def stop(self):
        self._stop_event.set()


# ========== Demo ==========

if __name__ == "__main__":
    queue = JobQueue(workers=2)

    # Register handlers
    queue.register("send_email", lambda p: f"Sent to {p['to']}")
    queue.register("resize_image", lambda p: f"Resized {p['file']} to {p['size']}")

    fail_count = {"n": 0}
    def flaky_handler(payload):
        fail_count["n"] += 1
        if fail_count["n"] < 3:
            raise RuntimeError("Temporary failure")
        return "Success on retry"

    queue.register("flaky_task", flaky_handler)

    # Enqueue jobs with different priorities
    print("=== Enqueueing Jobs ===")
    queue.enqueue("send_email", {"to": "alice@example.com"}, priority=1)
    queue.enqueue("resize_image", {"file": "photo.jpg", "size": "800x600"}, priority=2)
    queue.enqueue("send_email", {"to": "bob@example.com"}, priority=0)  # highest priority
    queue.enqueue("flaky_task", {"data": "test"}, priority=1, max_retries=3)

    print("\n=== Processing Jobs ===")
    queue.process_all()

    print(f"\n=== Stats: {queue.stats()} ===")

    # Periodic scheduler demo
    print("\n=== Periodic Scheduler ===")
    scheduler = PeriodicScheduler()
    tick = {"count": 0}

    def heartbeat():
        tick["count"] += 1
        print(f"  [HEARTBEAT] tick #{tick['count']}")

    scheduler.every(0.3, heartbeat, name="heartbeat")
    scheduler.start()
    time.sleep(1.0)
    scheduler.stop()
    print(f"  Heartbeat ran {tick['count']} times")
