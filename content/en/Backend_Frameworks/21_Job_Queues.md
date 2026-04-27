# 21. Job Queues

**Previous**: [Redis Caching Patterns](./20_Redis_Caching_Patterns.md)

**Difficulty**: ⭐⭐⭐

## Learning Objectives

- Understand why job queues are essential for reliable asynchronous processing
- Implement task queues with Celery (Python) backed by Redis and RabbitMQ
- Build job processors with Bull/BullMQ (Node.js) using Redis
- Manage the full job lifecycle: enqueue, process, retry, and dead-letter
- Use priority queues and delayed/scheduled jobs for complex workflows
- Scale workers with concurrency controls and resource management
- Monitor job queues with Flower (Python) and Bull Board (Node.js)
- Apply common patterns: email sending, image processing, report generation
- Design robust error handling and retry strategies

## Table of Contents

1. [Why Job Queues](#1-why-job-queues)
2. [Celery with Redis/RabbitMQ](#2-celery-with-redisrabbitmq)
3. [Bull/BullMQ with Redis](#3-bullbullmq-with-redis)
4. [Job Lifecycle](#4-job-lifecycle)
5. [Priority Queues and Delayed Jobs](#5-priority-queues-and-delayed-jobs)
6. [Worker Scaling and Concurrency](#6-worker-scaling-and-concurrency)
7. [Monitoring and Observability](#7-monitoring-and-observability)
8. [Common Patterns](#8-common-patterns)
9. [Error Handling and Retry Strategies](#9-error-handling-and-retry-strategies)
10. [Reliability Patterns](#10-reliability-patterns)
11. [Practice Exercises](#11-practice-exercises)

---

## 1. Why Job Queues

Job queues decouple time-consuming or unreliable work from the request-response cycle. Instead of making users wait, work is pushed to a background queue and processed asynchronously.

### Theory: Producer-Consumer Over a Durable Queue

The pattern: a producer puts a job into a queue, a consumer takes a job out, processes it, and acknowledges. The queue between them is *durable* — it survives crashes of either side.

#### A.1 What the queue actually buys you

Without a queue:

```
Web request → handler → send_email() (10s) → response
```

The user waits 10 seconds. If `send_email()` raises, the request fails. If the email server is down, every request that needs email fails.

With a queue:

```
Web request → handler → queue.enqueue("send_email", ...) → response (immediate)
                                ↓
                            Worker pulls job, runs send_email, ACK or retry
```

Three things change:

1. **Latency.** The user sees the response immediately; the email is "in flight" but the request is done.
2. **Reliability.** If the email server is down, the job stays in the queue and retries. The web request still succeeded.
3. **Scaling.** Email processing capacity scales independently from web request capacity. Add workers; queue drains faster.

#### A.2 The persistence requirement

For all of this to work, the queue must outlive a worker crash. If the worker pulls a job, starts processing, and crashes before completion, the job *must not* be lost. Two mechanisms achieve this:

- **Acknowledgment (ACK) semantics.** The worker pulls a job and the queue marks it "in flight" — but does not delete it. Only after the worker successfully processes and ACKs does the queue actually delete. If the worker crashes (or fails to ACK within a timeout), the queue redelivers the job to another worker.
- **Persistent storage.** Jobs survive a queue server restart. RabbitMQ writes to disk; Redis with AOF persistence does too; Kafka writes everything to disk by default.

The combination — ACK + persistence — gives you "no job is lost". But it also gives you "a job might be delivered twice" (next section).

#### A.3 Two-phase visibility

The standard queue lifecycle for one job:

```
1. enqueued        ← producer pushed it
2. running         ← worker pulled it, but hasn't ACKed
3. completed       ← worker ACKed; queue deletes
   OR
3'. failed/retry   ← worker NACKed or timed out; queue redelivers
4. dead letter     ← exhausted retries; moved to a separate queue for inspection
```

Each transition is durable. Operators can inspect "running" jobs (find stuck workers), "failed" jobs (retry counters), and the "dead letter" queue (jobs that exhausted retries). All three are critical for observability — see Lesson 17 §7.

### Problems Job Queues Solve

**User experience**: A request that sends an email, generates a PDF, or resizes an image should not block the HTTP response.

**Reliability**: If an external API is down, the job can be retried later rather than failing the user's request.

**Rate limiting**: External APIs often have rate limits. A queue lets you control throughput.

**Resource management**: CPU-intensive tasks (image processing, ML inference) can be processed on dedicated workers without impacting the web server.

**Ordering guarantees**: Some workflows require tasks to be processed in a specific order.

### Architecture Overview

```
┌──────────┐     ┌───────────┐     ┌──────────┐     ┌──────────┐
│  Web App │────▶│  Message   │────▶│  Worker  │────▶│  Result  │
│ (Producer)│     │  Broker    │     │(Consumer)│     │  Store   │
└──────────┘     │(Redis/RMQ) │     └──────────┘     │ (Redis)  │
                 └───────────┘                       └──────────┘
```

### Message Broker Comparison

| Feature | Redis | RabbitMQ | Amazon SQS |
|---|---|---|---|
| Protocol | Custom | AMQP | HTTP/SQS |
| Persistence | Optional (AOF/RDB) | Disk by default | Managed |
| Max message size | 512 MB | 128 MB (configurable) | 256 KB |
| Throughput | Very high | High | Moderate |
| Message ordering | Per-queue FIFO | Per-queue FIFO | Best-effort (FIFO available) |
| Dead-letter queue | Manual | Built-in | Built-in |
| Complexity | Low | Medium | Low (managed) |

### Task Queue Library Comparison

| Feature | Celery (Python) | BullMQ (Node.js) |
|---------|----------------|-------------------|
| Broker | Redis, RabbitMQ | Redis only |
| Result backend | Redis, DB, S3 | Redis |
| Scheduling | celery-beat | Built-in repeatable |
| Priority queues | Yes (limited) | Yes (native) |
| Rate limiting | Per-task decorator | Per-queue config |
| Dead letter queue | Manual (on_failure) | Built-in |
| Monitoring | Flower dashboard | Bull Board |

---

## 2. Celery with Redis/RabbitMQ

[Celery](https://docs.celeryq.dev/) is the de facto standard for task queues in Python. It supports Redis and RabbitMQ as message brokers.

### Theory: Distributed Queues Compared

Three categories of queue dominate the backend ecosystem. Each has a different design center.

#### C.1 Redis-backed queues (Celery, BullMQ, Sidekiq)

Use Redis lists or streams as the queue. Properties:

- **Pros:** simple to operate, low latency, often already deployed for caching.
- **Cons:** Redis persistence is limited (RDB snapshots, AOF append-only file). A crashed Redis can lose recently-acked jobs. Single-node throughput limit ~100K jobs/s.
- **Best for:** small-to-medium apps where Redis is already in the stack.

#### C.2 RabbitMQ (AMQP-based)

A purpose-built message broker. Properties:

- **Pros:** strong delivery guarantees, sophisticated routing (exchanges, bindings), per-message ACK, dead-letter exchanges built in. Mature operations tooling.
- **Cons:** another piece of infrastructure to operate. Cluster setup is non-trivial.
- **Best for:** apps that need enterprise messaging features, complex routing, or strict delivery guarantees.

The exchange/binding model lets you implement fan-out (one message to many queues), topic-based routing (regex matching), or simple direct queues — all in the same broker.

#### C.3 Kafka (log-based)

Not a traditional queue — a *distributed commit log*. Producers append messages to a *topic*; consumers read from any offset they want. Properties:

- **Pros:** massive throughput (millions of messages/s on a cluster), durable retention (keep messages for days/weeks), reprocessing (rewind to any offset), strict ordering within a partition.
- **Cons:** higher latency than Redis or RabbitMQ. Operationally complex (ZooKeeper or KRaft, partitioning, consumer groups). Overkill for low-volume workloads.
- **Best for:** event streaming, audit logs, analytics pipelines, anything where retention and replay matter.

#### C.4 Partitioning vs exchanges

The deepest architectural difference between Kafka and RabbitMQ is *how messages are distributed across consumers*.

- **RabbitMQ exchanges + queues.** A message goes to one or more queues based on routing rules. Each queue has its own consumers. Order is per-queue.
- **Kafka partitions.** A topic is split into partitions; each partition is consumed by exactly one consumer in a consumer group. Order is per-partition. Scale = add partitions + add consumers.

Kafka's partitioning gives natural horizontal scale at the cost of within-key ordering needing co-located processing. RabbitMQ's exchange model gives flexible routing but no built-in partition concept — you build it yourself with multiple queues.

#### C.5 Picking a queue

| Workload | Queue |
|----------|-------|
| Background jobs in an existing Redis-using app | Celery on Redis / BullMQ |
| Cross-team messaging with rich routing | RabbitMQ |
| High-volume event streams, retention matters | Kafka |
| AWS/GCP-native | SQS / Pub/Sub (managed equivalents of the above) |

### Setup

```bash
pip install celery[redis]
# or for RabbitMQ:
pip install celery[rabbitmq]
```

### Basic Configuration

```python
# celery_app.py
from celery import Celery

app = Celery(
    "myapp",
    broker="redis://localhost:6379/0",       # Message broker
    backend="redis://localhost:6379/1",      # Result backend
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,            # Hard limit: 5 minutes
    task_soft_time_limit=240,       # Soft limit: 4 minutes (raises SoftTimeLimitExceeded)
    worker_prefetch_multiplier=1,   # Fetch one task at a time
    worker_max_tasks_per_child=100, # Restart worker after 100 tasks (prevent memory leaks)
)
```

### Defining Tasks

```python
# tasks.py
from celery_app import app
from celery import shared_task
from celery.utils.log import get_task_logger

logger = get_task_logger(__name__)

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def send_email(self, to: str, subject: str, body: str):
    """Send an email asynchronously."""
    try:
        logger.info(f"Sending email to {to}: {subject}")
        # ... SMTP logic ...
        return {"status": "sent", "to": to}
    except ConnectionError as exc:
        logger.warning(f"Email failed, retrying: {exc}")
        raise self.retry(exc=exc)

@app.task(bind=True, max_retries=5)
def process_image(self, image_path: str, operations: list[str]):
    """Resize, crop, or convert an image."""
    try:
        logger.info(f"Processing image: {image_path}")
        # ... PIL/Pillow logic ...
        return {"status": "processed", "path": image_path}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)

@app.task
def generate_report(report_type: str, params: dict):
    """Generate a report (PDF, CSV, etc.)."""
    logger.info(f"Generating {report_type} report with params: {params}")
    # ... report generation logic ...
    return {"status": "completed", "download_url": "/reports/123.pdf"}
```

### Calling Tasks

```python
# Asynchronous call (returns AsyncResult immediately)
result = send_email.delay("user@example.com", "Welcome!", "Hello there")
print(f"Task ID: {result.id}")
print(f"Status: {result.status}")   # PENDING, STARTED, SUCCESS, FAILURE

# Wait for result (blocking)
output = result.get(timeout=30)
print(f"Result: {output}")

# apply_async for more control
result = send_email.apply_async(
    args=["user@example.com", "Welcome!", "Hello"],
    countdown=60,                    # Delay execution by 60 seconds
    expires=3600,                    # Expire if not started within 1 hour
    queue="high_priority",           # Route to specific queue
    priority=9,                      # Priority (0-9, higher = more priority)
)

# Check result without blocking
if result.ready():
    if result.successful():
        print(f"Done: {result.result}")
    else:
        print(f"Failed: {result.traceback}")
```

### Task Chains, Groups, and Chords

```python
from celery import chain, group, chord

# Chain: sequential execution (output of one feeds into next)
workflow = chain(
    fetch_data.s("https://api.example.com/data"),
    transform_data.s(),
    save_to_database.s(),
)
result = workflow.apply_async()

# Group: parallel execution
batch = group(
    process_image.s(f"image_{i}.jpg", ["resize", "compress"])
    for i in range(100)
)
result = batch.apply_async()

# Chord: group + callback after all tasks complete
workflow = chord(
    group(
        process_image.s(f"image_{i}.jpg", ["resize"])
        for i in range(10)
    ),
    create_gallery.s()  # Called when all images are processed
)
result = workflow.apply_async()
```

### Running Workers

```bash
# Start a worker
celery -A celery_app worker --loglevel=info --concurrency=4

# Start with specific queues
celery -A celery_app worker -Q high_priority,default --concurrency=8

# Start Celery Beat (periodic task scheduler)
celery -A celery_app beat --loglevel=info
```

---

## 3. Bull/BullMQ with Redis

[BullMQ](https://docs.bullmq.io/) is the standard job queue for Node.js, backed by Redis. It replaced the older Bull library.

### Setup

```bash
npm install bullmq ioredis
```

### Basic Queue and Worker

```javascript
// queue.js
import { Queue } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis({
    host: 'localhost',
    port: 6379,
    maxRetriesPerRequest: null,  // Required by BullMQ
});

export const emailQueue = new Queue('email', { connection });
export const imageQueue = new Queue('image-processing', { connection });

// Add a job
await emailQueue.add('send-welcome', {
    to: 'user@example.com',
    subject: 'Welcome!',
    body: 'Hello and welcome to our platform.',
});

// Add with options
await emailQueue.add('send-notification', {
    to: 'user@example.com',
    subject: 'Order shipped',
}, {
    attempts: 3,
    backoff: { type: 'exponential', delay: 1000 },
    removeOnComplete: { age: 3600 },  // Remove completed jobs after 1 hour
    removeOnFail: { age: 86400 },     // Remove failed jobs after 24 hours
});
```

### Worker

```javascript
// worker.js
import { Worker } from 'bullmq';
import IORedis from 'ioredis';

const connection = new IORedis({
    host: 'localhost',
    port: 6379,
    maxRetriesPerRequest: null,
});

const emailWorker = new Worker('email', async (job) => {
    console.log(`Processing job ${job.id}: ${job.name}`);
    const { to, subject, body } = job.data;

    // Update progress
    await job.updateProgress(10);

    // Simulate sending email
    await sendEmail(to, subject, body);

    await job.updateProgress(100);

    return { status: 'sent', to };
}, {
    connection,
    concurrency: 5,            // Process 5 jobs concurrently
    limiter: {
        max: 10,               // Max 10 jobs
        duration: 1000,        // Per second
    },
});

// Event listeners
emailWorker.on('completed', (job, result) => {
    console.log(`Job ${job.id} completed:`, result);
});

emailWorker.on('failed', (job, err) => {
    console.error(`Job ${job.id} failed:`, err.message);
});

emailWorker.on('progress', (job, progress) => {
    console.log(`Job ${job.id} progress: ${progress}%`);
});
```

### Flow Producer (Job Dependencies)

BullMQ supports parent-child job relationships:

```javascript
import { FlowProducer } from 'bullmq';

const flowProducer = new FlowProducer({ connection });

// Define a workflow with dependencies
const flow = await flowProducer.add({
    name: 'send-report',
    queueName: 'email',
    data: { to: 'manager@example.com' },
    children: [
        {
            name: 'generate-pdf',
            queueName: 'reports',
            data: { type: 'monthly-sales' },
        },
        {
            name: 'generate-charts',
            queueName: 'reports',
            data: { type: 'sales-charts' },
        },
    ],
});

// The parent job (send-report) waits until all children complete
```

---

## 4. Job Lifecycle

Understanding the job lifecycle is critical for building reliable systems.

### Theory: Delivery Semantics

When a queue redelivers a job, the consumer might process it twice. This is the consequence of crash recovery: there is no way to *know* whether a worker that disappeared mid-job actually completed the work. Three delivery contracts.

#### B.1 At-most-once

The producer fires and forgets; the queue makes no retry guarantees. If the worker crashes, the job is gone.

```
producer → queue → consumer (crashes) → nothing
```

Use when losing a job is acceptable: telemetry samples, "best-effort" notifications, low-value events. Never use for payments, orders, anything users see.

#### B.2 At-least-once

The default. The queue redelivers until ACKed. A job *will* succeed — but might be delivered (and processed) more than once.

```
producer → queue → consumer (crashes) → queue redelivers → consumer (succeeds, ACKs)
```

This is the right default for almost every workload. It also means **your consumer must be idempotent** — running it twice with the same input must produce the same outcome. Otherwise the user gets two emails, two charges, two records.

#### B.3 Exactly-once: not really — but practically yes

True "exactly-once" delivery is impossible in a distributed system without infinite coordination. What is achievable: **at-least-once delivery + idempotent consumers + idempotency keys**, which behaves like exactly-once from the user's perspective.

The idempotency key pattern:

```python
def send_email_job(idempotency_key, to, subject, body):
    if processed_keys.exists(idempotency_key):
        return  # already done; skip
    send_email(to, subject, body)
    processed_keys.add(idempotency_key)
```

The producer attaches a unique key (UUID) per logical operation. The consumer checks whether that key has been processed; if so, no-op. If the queue redelivers, the second worker sees the key in `processed_keys` and skips. Net result: the email is sent once, no matter how many times the queue delivered.

The idempotency-key pattern is the single most important design discipline in queue-based systems. Every consumer should be designed to be idempotent under at-least-once delivery.

#### B.4 The retry strategy

When a job fails, the queue retries with backoff:

```
attempt 1: fail, retry in 1s
attempt 2: fail, retry in 2s
attempt 3: fail, retry in 4s
...
attempt N: fail → move to dead-letter queue
```

Exponential backoff prevents a flood of retries from a struggling downstream service. After N attempts (typically 5-10), the job moves to a *dead-letter queue* (DLQ) — a separate queue for human inspection. The DLQ is critical operationally: it shows you which jobs are systematically failing without losing them.

### State Transitions

```
                    ┌─────────────┐
                    │   WAITING   │ (in queue, not yet picked up)
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │   ACTIVE    │ (being processed by a worker)
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼──────┐  ┌─▼──────┐  ┌──▼──────────┐
       │  COMPLETED   │  │ FAILED │  │   DELAYED    │
       │              │  │        │  │ (retry later)│
       └──────────────┘  └───┬────┘  └──────┬───────┘
                             │              │
                        ┌────▼────┐         │
                        │  RETRY  │─────────┘
                        └────┬────┘
                             │ (max retries exceeded)
                        ┌────▼─────────┐
                        │ DEAD-LETTER  │
                        └──────────────┘
```

### Celery Job States

```python
from celery_app import app
from celery.result import AsyncResult

def check_job_status(task_id: str) -> dict:
    result = AsyncResult(task_id, app=app)

    status = {
        "task_id": task_id,
        "status": result.status,  # PENDING, STARTED, SUCCESS, FAILURE, RETRY, REVOKED
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
    }

    if result.ready():
        if result.successful():
            status["result"] = result.result
        else:
            status["error"] = str(result.result)
            status["traceback"] = result.traceback

    return status

# Revoke (cancel) a task
def cancel_job(task_id: str):
    app.control.revoke(task_id, terminate=True, signal="SIGTERM")
```

### BullMQ Job States

```javascript
import { Queue } from 'bullmq';

const queue = new Queue('email', { connection });

async function checkJobStatus(jobId) {
    const job = await queue.getJob(jobId);
    if (!job) return { error: 'Job not found' };

    const state = await job.getState();
    return {
        id: job.id,
        name: job.name,
        state,                          // waiting, active, completed, failed, delayed
        progress: job.progress,
        data: job.data,
        returnValue: job.returnvalue,
        failedReason: job.failedReason,
        attemptsMade: job.attemptsMade,
        timestamp: job.timestamp,
        processedOn: job.processedOn,
        finishedOn: job.finishedOn,
    };
}

// Get queue statistics
async function getQueueStats() {
    const [waiting, active, completed, failed, delayed] = await Promise.all([
        queue.getWaitingCount(),
        queue.getActiveCount(),
        queue.getCompletedCount(),
        queue.getFailedCount(),
        queue.getDelayedCount(),
    ]);
    return { waiting, active, completed, failed, delayed };
}
```

---

## 5. Priority Queues and Delayed Jobs

### Celery Priority Queues

```python
# Define queue routing
app.conf.task_routes = {
    "tasks.send_email": {"queue": "high_priority"},
    "tasks.generate_report": {"queue": "low_priority"},
    "tasks.process_image": {"queue": "default"},
}

# Or route dynamically
@app.task(bind=True)
def flexible_task(self, data, priority="default"):
    pass

flexible_task.apply_async(
    args=[data],
    queue="high_priority",
    priority=9,  # 0 (lowest) to 9 (highest)
)
```

### Celery Periodic Tasks (Beat)

```python
from celery.schedules import crontab

app.conf.beat_schedule = {
    "cleanup-expired-sessions": {
        "task": "tasks.cleanup_sessions",
        "schedule": crontab(minute=0, hour="*/6"),  # Every 6 hours
    },
    "send-daily-digest": {
        "task": "tasks.send_digest",
        "schedule": crontab(minute=0, hour=8),       # Daily at 8 AM UTC
        "args": ["daily"],
    },
    "check-health": {
        "task": "tasks.health_check",
        "schedule": 30.0,                             # Every 30 seconds
    },
}
```

### BullMQ Priority and Delayed Jobs

```javascript
// Priority: lower number = higher priority
await emailQueue.add('critical-alert', { to: 'admin@example.com' }, {
    priority: 1,  // Processed before priority 2, 3, etc.
});

await emailQueue.add('newsletter', { to: 'user@example.com' }, {
    priority: 10, // Lower priority
});

// Delayed job: process after 5 minutes
await emailQueue.add('follow-up', { to: 'user@example.com' }, {
    delay: 5 * 60 * 1000,  // 5 minutes in milliseconds
});

// Scheduled job: process at a specific time
const targetTime = new Date('2026-03-15T10:00:00Z');
await emailQueue.add('scheduled-email', { to: 'user@example.com' }, {
    delay: targetTime.getTime() - Date.now(),
});

// Repeatable jobs (cron-like)
await emailQueue.add('daily-report', { type: 'sales' }, {
    repeat: {
        pattern: '0 8 * * *',  // Daily at 8 AM
        tz: 'America/New_York',
    },
});

await emailQueue.add('cleanup', {}, {
    repeat: {
        every: 30 * 60 * 1000,  // Every 30 minutes
    },
});
```

---

## 6. Worker Scaling and Concurrency

### Celery Concurrency Models

```bash
# Prefork (default): multiprocessing, good for CPU-bound tasks
celery -A celery_app worker --pool=prefork --concurrency=4

# Gevent: green threads, good for I/O-bound tasks
celery -A celery_app worker --pool=gevent --concurrency=100

# Eventlet: similar to gevent
celery -A celery_app worker --pool=eventlet --concurrency=100

# Solo: single-threaded, good for debugging
celery -A celery_app worker --pool=solo
```

### Scaling Strategy

```python
# Configure different workers for different task types
# Worker 1: High-priority, low concurrency (CPU-bound)
# celery -A celery_app worker -Q image_processing --concurrency=2 --pool=prefork

# Worker 2: Low-priority, high concurrency (I/O-bound)
# celery -A celery_app worker -Q email,notifications --concurrency=50 --pool=gevent

# Worker 3: Default queue
# celery -A celery_app worker -Q default --concurrency=8 --pool=prefork
```

### BullMQ Concurrency and Rate Limiting

```javascript
const worker = new Worker('image-processing', processImage, {
    connection,
    concurrency: 4,                  // Process 4 jobs in parallel
    limiter: {
        max: 20,                     // Max 20 jobs
        duration: 60 * 1000,         // Per minute
    },
    lockDuration: 300000,            // 5 minutes lock per job
    stalledInterval: 30000,          // Check for stalled jobs every 30s
    maxStalledCount: 2,              // Max stalled count before failing
});

// Sandboxed workers (run in separate process)
const worker = new Worker('cpu-intensive', './processor.js', {
    connection,
    concurrency: 2,
    useWorkerThreads: true,  // Use worker threads instead of child processes
});
```

### Docker Compose Scaling

```yaml
# docker-compose.yml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  web:
    build: .
    ports:
      - "8000:8000"
    depends_on:
      - redis

  worker-high:
    build: .
    command: celery -A celery_app worker -Q high_priority --concurrency=4
    depends_on:
      - redis
    deploy:
      replicas: 2

  worker-default:
    build: .
    command: celery -A celery_app worker -Q default --concurrency=8
    depends_on:
      - redis
    deploy:
      replicas: 3

  worker-low:
    build: .
    command: celery -A celery_app worker -Q low_priority --concurrency=2
    depends_on:
      - redis
    deploy:
      replicas: 1

  beat:
    build: .
    command: celery -A celery_app beat --loglevel=info
    depends_on:
      - redis
```

---

## 7. Monitoring and Observability

### Flower (Celery)

[Flower](https://flower.readthedocs.io/) is a real-time web monitor for Celery.

```bash
pip install flower
celery -A celery_app flower --port=5555 --broker=redis://localhost:6379/0
```

Flower provides:
- Real-time worker status (online, heartbeat, processed/failed count)
- Task progress and history
- Queue lengths and consumer counts
- Task rate graphs
- Remote worker control (shutdown, pool resize)

### Programmatic Monitoring (Celery)

```python
from celery_app import app

def get_worker_stats():
    """Get stats from all active workers."""
    inspector = app.control.inspect()

    return {
        "active_tasks": inspector.active(),
        "reserved_tasks": inspector.reserved(),
        "registered_tasks": inspector.registered(),
        "stats": inspector.stats(),
        "queues": inspector.active_queues(),
    }

def get_queue_lengths():
    """Get the number of messages in each queue."""
    with app.connection_or_acquire() as conn:
        return {
            "high_priority": conn.default_channel.queue_declare(
                "high_priority", passive=True
            ).message_count,
            "default": conn.default_channel.queue_declare(
                "default", passive=True
            ).message_count,
        }
```

### Bull Board (BullMQ)

[Bull Board](https://github.com/felixmosh/bull-board) provides a UI dashboard for BullMQ.

```javascript
import { createBullBoard } from '@bull-board/api';
import { BullMQAdapter } from '@bull-board/api/bullMQAdapter';
import { ExpressAdapter } from '@bull-board/express';
import express from 'express';

const serverAdapter = new ExpressAdapter();
serverAdapter.setBasePath('/admin/queues');

createBullBoard({
    queues: [
        new BullMQAdapter(emailQueue),
        new BullMQAdapter(imageQueue),
    ],
    serverAdapter,
});

const app = express();
app.use('/admin/queues', serverAdapter.getRouter());
app.listen(3000);
```

### Custom Metrics

```javascript
// Emit metrics for Prometheus/Grafana
import { Queue, QueueEvents } from 'bullmq';

const queueEvents = new QueueEvents('email', { connection });

queueEvents.on('completed', ({ jobId, returnvalue }) => {
    metrics.increment('jobs.completed', { queue: 'email' });
});

queueEvents.on('failed', ({ jobId, failedReason }) => {
    metrics.increment('jobs.failed', { queue: 'email' });
});

// Periodic stats collection
setInterval(async () => {
    const counts = await emailQueue.getJobCounts();
    metrics.gauge('queue.waiting', counts.waiting, { queue: 'email' });
    metrics.gauge('queue.active', counts.active, { queue: 'email' });
    metrics.gauge('queue.failed', counts.failed, { queue: 'email' });
}, 10000);
```

---

## 8. Common Patterns

### Email Sending Pipeline

```python
@app.task(bind=True, max_retries=3, default_retry_delay=30)
def send_transactional_email(self, template: str, recipient: str, context: dict):
    """Send a transactional email with template rendering."""
    try:
        # Render template
        html_content = render_template(template, context)

        # Send via SMTP/API
        response = email_client.send(
            to=recipient,
            subject=context.get("subject", "Notification"),
            html=html_content,
        )

        return {"message_id": response.id, "status": "sent"}
    except RateLimitError:
        # Retry with exponential backoff
        raise self.retry(countdown=60 * (2 ** self.request.retries))
    except InvalidRecipientError:
        # Don't retry for permanent failures
        return {"status": "skipped", "reason": "invalid_recipient"}

# Batch email sending
@app.task
def send_newsletter(campaign_id: int):
    """Send newsletter to all subscribers."""
    subscribers = get_subscribers(campaign_id)

    # Create a group of individual email tasks
    batch = group(
        send_transactional_email.s("newsletter", sub.email, {
            "subject": "Weekly Newsletter",
            "name": sub.name,
            "campaign_id": campaign_id,
        })
        for sub in subscribers
    )

    result = batch.apply_async()
    return {"total": len(subscribers), "group_id": result.id}
```

### Image Processing Pipeline

```python
from celery import chain

@app.task
def download_image(url: str) -> str:
    """Download image and return local path."""
    response = requests.get(url)
    path = f"/tmp/images/{uuid4()}.jpg"
    with open(path, "wb") as f:
        f.write(response.content)
    return path

@app.task
def resize_image(path: str, width: int = 800, height: int = 600) -> str:
    """Resize image to specified dimensions."""
    from PIL import Image
    img = Image.open(path)
    img = img.resize((width, height), Image.LANCZOS)
    output_path = path.replace(".jpg", f"_{width}x{height}.jpg")
    img.save(output_path, quality=85)
    return output_path

@app.task
def upload_to_s3(path: str) -> str:
    """Upload processed image to S3."""
    key = f"images/{os.path.basename(path)}"
    s3_client.upload_file(path, "my-bucket", key)
    os.remove(path)  # Cleanup local file
    return f"https://my-bucket.s3.amazonaws.com/{key}"

# Chain: download → resize → upload
def process_user_avatar(image_url: str):
    workflow = chain(
        download_image.s(image_url),
        resize_image.s(200, 200),
        upload_to_s3.s(),
    )
    return workflow.apply_async()
```

### Report Generation

```javascript
// BullMQ report generation
import { Queue, Worker } from 'bullmq';

const reportQueue = new Queue('reports', { connection });

// Producer: API endpoint
app.post('/api/reports', async (req, res) => {
    const { type, dateRange, format } = req.body;

    const job = await reportQueue.add('generate', {
        type,
        dateRange,
        format: format || 'pdf',
        requestedBy: req.user.id,
    }, {
        attempts: 2,
        backoff: { type: 'fixed', delay: 30000 },
    });

    res.json({ jobId: job.id, status: 'queued' });
});

// Status endpoint
app.get('/api/reports/:jobId', async (req, res) => {
    const job = await reportQueue.getJob(req.params.jobId);
    if (!job) return res.status(404).json({ error: 'Not found' });

    const state = await job.getState();
    res.json({
        id: job.id,
        state,
        progress: job.progress,
        result: job.returnvalue,
    });
});

// Worker
const reportWorker = new Worker('reports', async (job) => {
    const { type, dateRange, format } = job.data;

    await job.updateProgress(10);
    const data = await fetchReportData(type, dateRange);

    await job.updateProgress(50);
    const file = await generateDocument(data, format);

    await job.updateProgress(90);
    const url = await uploadToStorage(file);

    await job.updateProgress(100);
    return { downloadUrl: url, generatedAt: new Date().toISOString() };
}, { connection, concurrency: 2 });
```

---

## 9. Error Handling and Retry Strategies

### Retry Strategies

```python
# Fixed delay retry
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def fixed_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        raise self.retry(exc=exc)  # Retry after 60 seconds

# Exponential backoff
@app.task(bind=True, max_retries=5)
def exponential_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        countdown = 2 ** self.request.retries * 30  # 30s, 60s, 120s, 240s, 480s
        raise self.retry(exc=exc, countdown=countdown)

# Exponential backoff with jitter (recommended)
import random

@app.task(bind=True, max_retries=5)
def jittered_retry_task(self, data):
    try:
        process(data)
    except TransientError as exc:
        base_delay = 2 ** self.request.retries * 30
        jitter = random.uniform(0, base_delay * 0.1)
        raise self.retry(exc=exc, countdown=base_delay + jitter)
```

### BullMQ Backoff Strategies

```javascript
// Exponential backoff
await queue.add('task', data, {
    attempts: 5,
    backoff: {
        type: 'exponential',
        delay: 1000,  // 1s, 2s, 4s, 8s, 16s
    },
});

// Fixed backoff
await queue.add('task', data, {
    attempts: 3,
    backoff: {
        type: 'fixed',
        delay: 5000,  // Always 5 seconds between retries
    },
});

// Custom backoff strategy
await queue.add('task', data, {
    attempts: 5,
    backoff: {
        type: 'custom',
    },
});

// In the worker, implement custom backoff
const worker = new Worker('queue', processor, {
    connection,
    settings: {
        backoffStrategy: (attemptsMade) => {
            // Custom: 10s, 30s, 90s, 270s, 810s
            return Math.pow(3, attemptsMade) * 10000;
        },
    },
});
```

### Dead-Letter Queue Pattern

```python
# Celery dead-letter handling
@app.task(bind=True, max_retries=3)
def process_order(self, order_id: int):
    try:
        # ... process order ...
        pass
    except Exception as exc:
        if self.request.retries >= self.max_retries:
            # Max retries exceeded — send to dead-letter queue
            dead_letter_task.delay(
                original_task="process_order",
                args=[order_id],
                error=str(exc),
                retries=self.request.retries,
            )
            return {"status": "dead_lettered", "order_id": order_id}
        raise self.retry(exc=exc, countdown=60)

@app.task
def dead_letter_task(original_task: str, args: list, error: str, retries: int):
    """Store failed tasks for manual review."""
    logger.error(
        f"Dead letter: task={original_task}, args={args}, "
        f"error={error}, retries={retries}"
    )
    # Store in database for admin review
    FailedJob.objects.create(
        task_name=original_task,
        arguments=json.dumps(args),
        error_message=error,
        retry_count=retries,
    )
```

### Idempotency

Jobs may be executed more than once (at-least-once delivery). Design tasks to be idempotent:

```python
@app.task(bind=True)
def charge_payment(self, payment_id: str, amount: float):
    """Idempotent payment processing."""
    # Check if already processed (idempotency key)
    if redis_client.get(f"payment:processed:{payment_id}"):
        logger.info(f"Payment {payment_id} already processed, skipping")
        return {"status": "already_processed"}

    try:
        result = payment_gateway.charge(payment_id, amount)

        # Mark as processed with TTL
        redis_client.setex(
            f"payment:processed:{payment_id}",
            86400 * 7,  # 7 days
            json.dumps(result),
        )

        return {"status": "charged", "transaction_id": result.transaction_id}
    except PaymentError as exc:
        raise self.retry(exc=exc, countdown=30)
```

---

## 10. Reliability Patterns

### Celery Reliability Settings

Two Celery settings prevent task loss when a worker crashes mid-execution:

```python
app.conf.update(
    # acks_late: acknowledge the message AFTER the task returns, not before.
    # Default (acks_early) removes the message from the queue the moment a
    # worker picks it up. If the worker dies before finishing, the task is gone.
    task_acks_late=True,

    # reject_on_worker_lost: if a worker is killed (OOM, SIGKILL), re-queue
    # the task instead of losing it. Requires acks_late=True.
    task_reject_on_worker_lost=True,

    # Prefetch only one task at a time so late-ack doesn't hold many messages.
    worker_prefetch_multiplier=1,
)
```

These two settings together give **at-least-once delivery**. Make your tasks idempotent (see Section 9) to handle duplicate execution safely.

### Saga Pattern for Distributed Transactions

When an operation spans multiple services (e.g., charge payment → reserve stock → send email), a single database transaction cannot span service boundaries. The **Saga** pattern coordinates the steps and runs compensating actions on failure.

**Choreography-based Saga** (event-driven):

```python
@app.task(bind=True, max_retries=3)
def saga_charge_payment(self, order_id: str):
    try:
        charge_payment(order_id)
        # On success, trigger the next step
        saga_reserve_stock.delay(order_id)
    except PaymentError as exc:
        # No compensation needed — nothing was committed yet
        raise self.retry(exc=exc, countdown=30)

@app.task(bind=True, max_retries=3)
def saga_reserve_stock(self, order_id: str):
    try:
        reserve_stock(order_id)
        saga_send_confirmation.delay(order_id)
    except StockUnavailableError as exc:
        # Compensation: refund the payment that already succeeded
        saga_refund_payment.delay(order_id, reason="out_of_stock")
        raise

@app.task
def saga_refund_payment(order_id: str, reason: str):
    """Compensating transaction for saga_charge_payment."""
    refund_payment(order_id)
    notify_customer(order_id, f"Order cancelled: {reason}")
```

Each step publishes an event (or calls the next task) on success, and triggers a **compensating task** on failure to undo already-completed steps. Keep compensating tasks idempotent — they may also be retried.

---

## 11. Practice Exercises

### Exercise 1: Email Queue with Celery

Build a Celery application for sending emails:
- Define three task types: welcome email, password reset, order confirmation
- Each task type uses a different template
- Implement exponential backoff with jitter for retries
- Add a periodic task that sends a daily digest at 8 AM
- Track sent/failed counts in Redis

```python
# Starter code
from celery import Celery

app = Celery("email_service", broker="redis://localhost:6379/0")

@app.task(bind=True, max_retries=3)
def send_welcome_email(self, user_id: int, email: str):
    # TODO: Implement with retry logic
    pass

@app.task
def send_daily_digest():
    # TODO: Gather digest content and send to all subscribers
    pass

# TODO: Configure beat_schedule for periodic tasks
# TODO: Add dead-letter handling
```

### Exercise 2: Image Processing Pipeline with BullMQ

Build a Node.js image processing service:
- Queue: accept image URLs for processing
- Worker: download, resize to 3 sizes (thumbnail, medium, large), upload
- Flow: use FlowProducer for the download → resize → upload chain
- Progress: report percentage through each stage
- Dashboard: set up Bull Board on `/admin/queues`

```javascript
// Starter code
import { Queue, Worker, FlowProducer } from 'bullmq';

const imageQueue = new Queue('images', { connection });

// TODO: Define worker that handles 'download', 'resize', 'upload' job names
// TODO: Set up FlowProducer for chained processing
// TODO: Add Bull Board dashboard
// TODO: Implement progress reporting
```

### Exercise 3: Priority Task Scheduler

Design a task scheduling system:
- Three priority levels: critical (P1), normal (P2), low (P3)
- Critical tasks are processed immediately
- Normal tasks are processed with at most 10 concurrency
- Low tasks are processed only when no higher-priority tasks are waiting
- Implement a dashboard endpoint showing queue depths and processing rates
- Write a load test that submits 1000 tasks across all priorities

### Exercise 4: Reliable Order Processing

Build an order processing system with guaranteed delivery:
- Order submission endpoint adds a job to the queue
- Worker processes the order: validate stock, charge payment, update inventory
- Make the payment step idempotent (use idempotency keys)
- Implement dead-letter queue for orders that fail after 3 retries
- Build an admin endpoint to view and retry dead-lettered orders
- Add monitoring: track processing time, success rate, queue depth

```python
# Starter code
@app.task(bind=True, max_retries=3)
def process_order(self, order_id: str):
    """
    Steps:
    1. Validate stock availability
    2. Charge payment (must be idempotent)
    3. Update inventory
    4. Send confirmation email
    """
    # TODO: Implement each step
    # TODO: Handle partial failures (e.g., payment charged but inventory update fails)
    # TODO: Send to dead-letter queue on max retries
    pass
```

---

## Further Reading

- [Celery Documentation](https://docs.celeryq.dev/)
- [BullMQ Documentation](https://docs.bullmq.io/)
- [RabbitMQ Tutorials](https://www.rabbitmq.com/tutorials)
- [Flower (Celery Monitor)](https://flower.readthedocs.io/)
- [Bull Board](https://github.com/felixmosh/bull-board)
- [Enterprise Integration Patterns](https://www.enterpriseintegrationpatterns.com/)
- [Designing Data-Intensive Applications (Chapter 11: Stream Processing)](https://dataintensive.net/)

---

**Previous**: [Redis Caching Patterns](./20_Redis_Caching_Patterns.md)
