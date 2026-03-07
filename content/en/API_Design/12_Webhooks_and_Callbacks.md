# 12. Webhooks and Callbacks

**Previous**: [API Testing and Validation](./11_API_Testing_and_Validation.md) | **Next**: [API Gateway Patterns](./13_API_Gateway_Patterns.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Design webhook systems that reliably deliver event notifications to subscribers
- Implement signature verification using HMAC to ensure webhook authenticity
- Apply retry policies with exponential backoff and dead letter queues for failed deliveries
- Use idempotency keys to prevent duplicate event processing
- Secure webhook endpoints against replay attacks and unauthorized access
- Build a complete webhook producer and consumer with FastAPI

---

## Table of Contents

1. [Webhooks vs. Polling](#1-webhooks-vs-polling)
2. [Webhook Design Principles](#2-webhook-design-principles)
3. [Event Payloads](#3-event-payloads)
4. [Signature Verification (HMAC)](#4-signature-verification-hmac)
5. [Retry Policies and Delivery Guarantees](#5-retry-policies-and-delivery-guarantees)
6. [Idempotency Keys](#6-idempotency-keys)
7. [Dead Letter Queues](#7-dead-letter-queues)
8. [Webhook Security](#8-webhook-security)
9. [Building a Webhook System](#9-building-a-webhook-system)
10. [Exercises](#10-exercises)
11. [References](#11-references)

---

## 1. Webhooks vs. Polling

APIs have two models for notifying clients about changes:

**Polling**: The client periodically asks "has anything changed?"

```
Client                          Server
  |                               |
  |-- GET /orders?since=12:00 --->|  (every 30 seconds)
  |<-- 200 OK [] ---------------  |  (no changes)
  |                               |
  |-- GET /orders?since=12:00 --->|
  |<-- 200 OK [] ---------------  |  (still nothing)
  |                               |
  |-- GET /orders?since=12:00 --->|
  |<-- 200 OK [order_42] -------  |  (finally, a change!)
```

**Webhooks**: The server pushes events to the client when something happens.

```
Client                          Server
  |                               |
  |   (registers callback URL)    |
  |                               |
  |                               |-- order.created event occurs
  |<-- POST /webhook {event} -----|  (immediate push)
  |-- 200 OK ------------------->|
```

### Comparison

| Aspect | Polling | Webhooks |
|--------|---------|----------|
| Latency | Seconds to minutes | Near real-time |
| Server load | High (many empty responses) | Low (push only on events) |
| Client complexity | Simple (just HTTP GET) | Must host an endpoint |
| Reliability | Client controls retry | Server must handle failures |
| Firewall issues | None (outbound only) | Client must accept inbound |
| Data freshness | Depends on poll interval | Immediate |

---

## 2. Webhook Design Principles

### Registration

Allow consumers to register their callback URLs and specify which events they want to receive:

```python
from pydantic import BaseModel, HttpUrl
from fastapi import FastAPI, HTTPException
import secrets

app = FastAPI()


class WebhookRegistration(BaseModel):
    """Schema for registering a webhook subscription."""
    url: HttpUrl
    events: list[str]
    description: str | None = None


class WebhookSubscription(BaseModel):
    """The subscription record stored in the database."""
    id: str
    url: str
    events: list[str]
    secret: str          # Shared secret for HMAC signing
    active: bool = True
    description: str | None = None


@app.post("/webhooks", status_code=201)
async def register_webhook(registration: WebhookRegistration):
    """Register a new webhook subscription.

    Returns the subscription including the signing secret.
    The secret is only shown once — the client must store it.
    """
    # Validate that the URL is reachable (optional but recommended)
    if not await verify_endpoint_reachable(str(registration.url)):
        raise HTTPException(
            status_code=400,
            detail="Webhook URL is not reachable. Ensure the endpoint "
                   "returns 200 for POST requests.",
        )

    # Validate event types
    valid_events = {
        "order.created", "order.updated", "order.cancelled",
        "payment.succeeded", "payment.failed",
        "user.created", "user.deleted",
    }
    invalid = set(registration.events) - valid_events
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid event types: {invalid}. Valid: {valid_events}",
        )

    subscription = WebhookSubscription(
        id=f"whk_{secrets.token_hex(16)}",
        url=str(registration.url),
        events=registration.events,
        secret=secrets.token_hex(32),
    )

    await save_subscription(subscription)

    return {
        "id": subscription.id,
        "url": subscription.url,
        "events": subscription.events,
        "secret": subscription.secret,  # Only shown once
        "message": "Store this secret securely. It will not be shown again.",
    }


@app.get("/webhooks")
async def list_webhooks():
    """List all webhook subscriptions (secrets are redacted)."""
    subscriptions = await get_all_subscriptions()
    return [
        {
            "id": s.id,
            "url": s.url,
            "events": s.events,
            "active": s.active,
            # Never expose the secret in list responses
        }
        for s in subscriptions
    ]


@app.delete("/webhooks/{webhook_id}", status_code=204)
async def delete_webhook(webhook_id: str):
    """Unsubscribe a webhook."""
    await remove_subscription(webhook_id)
```

---

## 3. Event Payloads

Design event payloads to be self-contained. The receiver should not need to make additional API calls to understand the event.

### Event Envelope Structure

```python
from datetime import datetime, timezone
from pydantic import BaseModel, Field
import uuid


class WebhookEvent(BaseModel):
    """Standard envelope for all webhook events.

    Every event includes metadata that the receiver needs
    for processing, deduplication, and verification.
    """

    id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex}",
        description="Unique event identifier for deduplication",
    )
    type: str = Field(
        description="Event type (e.g., 'order.created')",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When the event occurred (UTC)",
    )
    api_version: str = Field(
        default="2025-01-01",
        description="API version that generated this event",
    )
    data: dict = Field(
        description="Event-specific payload data",
    )


# Example events
order_created_event = WebhookEvent(
    type="order.created",
    data={
        "order": {
            "id": 42,
            "customer_id": 7,
            "items": [
                {"book_id": 1, "quantity": 2, "price": 29.99},
                {"book_id": 5, "quantity": 1, "price": 45.00},
            ],
            "total": 104.98,
            "status": "pending",
            "created_at": "2025-01-15T10:30:00Z",
        }
    },
)

payment_failed_event = WebhookEvent(
    type="payment.failed",
    data={
        "payment": {
            "id": "pay_abc123",
            "order_id": 42,
            "amount": 104.98,
            "currency": "USD",
            "failure_reason": "insufficient_funds",
            "failed_at": "2025-01-15T10:31:00Z",
        }
    },
)
```

### Event Naming Convention

Use a `resource.action` pattern for event types:

```
order.created        # A new order was placed
order.updated        # An existing order was modified
order.cancelled      # An order was cancelled
payment.succeeded    # Payment was processed successfully
payment.failed       # Payment processing failed
user.created         # A new user account was created
user.deleted         # A user account was deleted
```

---

## 4. Signature Verification (HMAC)

Webhook receivers must verify that incoming requests are genuinely from the expected sender. HMAC (Hash-based Message Authentication Code) provides this guarantee.

### Producer: Signing the Payload

```python
import hashlib
import hmac
import json
import time


def sign_webhook_payload(
    payload: dict, secret: str, timestamp: int | None = None
) -> dict[str, str]:
    """Generate HMAC signature for a webhook payload.

    The signature covers both the timestamp and the payload body,
    preventing replay attacks and tampering.

    Args:
        payload: The event data to sign.
        secret: The shared secret for this subscription.
        timestamp: Unix timestamp (defaults to current time).

    Returns:
        Headers to include in the webhook delivery request.
    """
    if timestamp is None:
        timestamp = int(time.time())

    # Canonical payload: timestamp + "." + JSON body
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    message = f"{timestamp}.{body}"

    # HMAC-SHA256 signature
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-Webhook-Signature": f"sha256={signature}",
        "X-Webhook-Timestamp": str(timestamp),
        "X-Webhook-Id": payload.get("id", ""),
    }
```

### Consumer: Verifying the Signature

```python
import hashlib
import hmac
import time
from fastapi import FastAPI, Request, HTTPException, Header

app = FastAPI()

WEBHOOK_SECRET = "your-shared-secret-from-registration"
TIMESTAMP_TOLERANCE = 300  # 5 minutes


async def verify_webhook_signature(
    request: Request,
    x_webhook_signature: str = Header(...),
    x_webhook_timestamp: str = Header(...),
):
    """Verify the HMAC signature on incoming webhook requests.

    This dependency:
    1. Checks the timestamp is recent (prevents replay attacks)
    2. Recomputes the HMAC signature from the raw body
    3. Uses constant-time comparison to prevent timing attacks
    """
    # Step 1: Check timestamp freshness
    try:
        timestamp = int(x_webhook_timestamp)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp")

    age = abs(time.time() - timestamp)
    if age > TIMESTAMP_TOLERANCE:
        raise HTTPException(
            status_code=400,
            detail=f"Timestamp too old ({age:.0f}s). Max tolerance: {TIMESTAMP_TOLERANCE}s",
        )

    # Step 2: Read the raw body and compute expected signature
    body = await request.body()
    message = f"{timestamp}.{body.decode('utf-8')}"

    expected_signature = hmac.new(
        WEBHOOK_SECRET.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Step 3: Constant-time comparison (prevents timing attacks)
    received_signature = x_webhook_signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected_signature, received_signature):
        raise HTTPException(status_code=401, detail="Invalid signature")


@app.post("/webhook")
async def handle_webhook(
    request: Request,
    _=Depends(verify_webhook_signature),
):
    """Handle incoming webhook events after signature verification."""
    event = await request.json()
    event_type = event.get("type")

    if event_type == "order.created":
        await process_new_order(event["data"]["order"])
    elif event_type == "payment.failed":
        await handle_payment_failure(event["data"]["payment"])
    else:
        # Log unknown event types but do not reject them
        # (forward compatibility for new event types)
        logger.info(f"Received unknown event type: {event_type}")

    return {"status": "received"}
```

---

## 5. Retry Policies and Delivery Guarantees

Webhook delivery can fail for many reasons: network issues, receiver downtime, timeouts. A robust webhook system retries failed deliveries with exponential backoff.

### Retry with Exponential Backoff

```python
import asyncio
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
class DeliveryAttempt:
    """Record of a single delivery attempt."""
    attempt_number: int
    timestamp: datetime
    status_code: int | None = None
    error: str | None = None
    success: bool = False


@dataclass
class DeliveryResult:
    """Complete delivery result with all attempts."""
    event_id: str
    subscription_id: str
    delivered: bool
    attempts: list[DeliveryAttempt] = field(default_factory=list)


async def deliver_webhook(
    url: str,
    payload: dict,
    secret: str,
    max_retries: int = 5,
    initial_delay: float = 1.0,
) -> DeliveryResult:
    """Deliver a webhook with exponential backoff retry.

    Retry schedule (default):
      Attempt 1: immediate
      Attempt 2: 1 second delay
      Attempt 3: 2 seconds delay
      Attempt 4: 4 seconds delay
      Attempt 5: 8 seconds delay

    Total max wait: ~15 seconds (for 5 retries).
    In production, use longer delays (minutes to hours) and
    a background task queue (Celery, Dramatiq).
    """
    result = DeliveryResult(
        event_id=payload.get("id", "unknown"),
        subscription_id="",
        delivered=False,
    )

    headers = sign_webhook_payload(payload, secret)
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.post(
                    url,
                    json=payload,
                    headers=headers,
                )

                attempt_record = DeliveryAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now(timezone.utc),
                    status_code=response.status_code,
                    success=200 <= response.status_code < 300,
                )
                result.attempts.append(attempt_record)

                if attempt_record.success:
                    result.delivered = True
                    logger.info(
                        f"Webhook delivered: event={result.event_id} "
                        f"url={url} attempt={attempt}"
                    )
                    return result

                # Non-2xx response — retry for 5xx, give up for 4xx
                if 400 <= response.status_code < 500:
                    logger.error(
                        f"Webhook rejected (client error): "
                        f"event={result.event_id} status={response.status_code}"
                    )
                    return result  # Do not retry client errors

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                attempt_record = DeliveryAttempt(
                    attempt_number=attempt,
                    timestamp=datetime.now(timezone.utc),
                    error=str(e),
                )
                result.attempts.append(attempt_record)

            # Exponential backoff
            if attempt < max_retries:
                delay = initial_delay * (2 ** (attempt - 1))
                logger.warning(
                    f"Webhook delivery failed: event={result.event_id} "
                    f"attempt={attempt}/{max_retries} retry_in={delay}s"
                )
                await asyncio.sleep(delay)

    logger.error(
        f"Webhook delivery exhausted retries: event={result.event_id} url={url}"
    )
    return result
```

### Production Retry Schedule

For production systems, use a task queue with longer retry intervals:

```python
# Celery task with exponential backoff
from celery import Celery

celery_app = Celery("webhooks", broker="redis://localhost:6379/0")


@celery_app.task(
    bind=True,
    max_retries=8,
    default_retry_delay=60,  # 1 minute initial delay
)
def deliver_webhook_task(self, url: str, payload: dict, secret: str):
    """Deliver webhook with Celery retry.

    Retry schedule:
      Attempt 1: immediate
      Attempt 2: 1 minute
      Attempt 3: 2 minutes
      Attempt 4: 4 minutes
      Attempt 5: 8 minutes
      Attempt 6: 16 minutes
      Attempt 7: 32 minutes
      Attempt 8: 64 minutes (~1 hour)
    Total span: ~2 hours
    """
    try:
        response = httpx.post(url, json=payload, headers=sign_webhook_payload(payload, secret))
        response.raise_for_status()
    except Exception as exc:
        raise self.retry(
            exc=exc,
            countdown=60 * (2 ** self.request.retries),
        )
```

---

## 6. Idempotency Keys

Receivers may get the same event more than once (due to retries or network issues). Idempotency keys let the receiver detect and skip duplicates.

```python
import redis.asyncio as redis
from fastapi import FastAPI, Request, HTTPException, Depends, Header

app = FastAPI()


class IdempotencyGuard:
    """Prevents duplicate processing of webhook events.

    Uses Redis to track processed event IDs. If an event
    has already been processed, it is acknowledged (200)
    but not processed again.
    """

    def __init__(self, redis_client: redis.Redis, ttl: int = 86400):
        self.redis = redis_client
        self.ttl = ttl  # How long to remember processed events (24 hours)

    async def is_duplicate(self, event_id: str) -> bool:
        """Check if this event has already been processed."""
        key = f"webhook:processed:{event_id}"
        # SET NX returns True if the key was newly set
        was_set = await self.redis.set(key, "1", nx=True, ex=self.ttl)
        return not was_set  # If not set, it already existed (duplicate)

    async def mark_failed(self, event_id: str):
        """Remove the idempotency key if processing fails.

        This allows the event to be retried on the next delivery attempt.
        """
        key = f"webhook:processed:{event_id}"
        await self.redis.delete(key)


@app.post("/webhook")
async def handle_webhook(
    request: Request,
    x_webhook_id: str = Header(...),
    _=Depends(verify_webhook_signature),
):
    """Handle webhook with idempotency protection."""
    idempotency = IdempotencyGuard(request.app.state.redis)

    # Check for duplicate
    if await idempotency.is_duplicate(x_webhook_id):
        # Already processed — acknowledge without reprocessing
        return {"status": "duplicate", "event_id": x_webhook_id}

    event = await request.json()

    try:
        await process_event(event)
        return {"status": "processed", "event_id": x_webhook_id}
    except Exception:
        # Processing failed — allow retry
        await idempotency.mark_failed(x_webhook_id)
        raise HTTPException(
            status_code=500,
            detail="Event processing failed. Please retry.",
        )
```

---

## 7. Dead Letter Queues

When all retry attempts fail, the event should be moved to a dead letter queue (DLQ) for manual investigation rather than being silently dropped.

```python
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
import json


@dataclass
class DeadLetterEntry:
    """A failed webhook delivery stored for manual review."""
    event_id: str
    subscription_id: str
    url: str
    payload: dict
    attempts: list[dict]
    failed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: str | None = None


class DeadLetterQueue:
    """Redis-backed dead letter queue for failed webhook deliveries."""

    def __init__(self, redis_client):
        self.redis = redis_client
        self.queue_key = "webhook:dlq"

    async def add(self, entry: DeadLetterEntry):
        """Add a failed delivery to the dead letter queue."""
        await self.redis.lpush(
            self.queue_key,
            json.dumps({
                "event_id": entry.event_id,
                "subscription_id": entry.subscription_id,
                "url": entry.url,
                "payload": entry.payload,
                "attempts": entry.attempts,
                "failed_at": entry.failed_at.isoformat(),
            }),
        )

    async def list_entries(self, limit: int = 50) -> list[dict]:
        """List recent dead letter entries."""
        raw = await self.redis.lrange(self.queue_key, 0, limit - 1)
        return [json.loads(entry) for entry in raw]

    async def retry_entry(self, event_id: str):
        """Manually retry a dead letter entry."""
        entries = await self.list_entries(limit=1000)
        for entry in entries:
            if entry["event_id"] == event_id:
                # Re-enqueue for delivery
                await deliver_webhook_task.delay(
                    url=entry["url"],
                    payload=entry["payload"],
                    secret=await get_subscription_secret(entry["subscription_id"]),
                )
                return True
        return False


# Admin endpoints for DLQ management
@app.get("/admin/webhooks/dlq")
async def list_dead_letters(request: Request):
    """List failed webhook deliveries for manual review."""
    dlq = DeadLetterQueue(request.app.state.redis)
    return await dlq.list_entries()


@app.post("/admin/webhooks/dlq/{event_id}/retry")
async def retry_dead_letter(event_id: str, request: Request):
    """Manually retry a failed webhook delivery."""
    dlq = DeadLetterQueue(request.app.state.redis)
    success = await dlq.retry_entry(event_id)
    if not success:
        raise HTTPException(status_code=404, detail="Event not found in DLQ")
    return {"status": "retrying", "event_id": event_id}
```

---

## 8. Webhook Security

### Threat Model

| Threat | Attack Vector | Mitigation |
|--------|--------------|------------|
| Spoofing | Attacker sends fake webhook requests | HMAC signature verification |
| Replay | Attacker resends a previously captured request | Timestamp validation |
| Information leak | Webhook URL exposed in logs | URL validation, HTTPS only |
| SSRF | Attacker registers an internal URL | URL allowlist, block private IPs |
| DoS | Attacker overwhelms the webhook receiver | Rate limiting on webhook endpoint |

### Preventing SSRF in Webhook Registration

```python
import ipaddress
from urllib.parse import urlparse
import socket


def validate_webhook_url(url: str) -> bool:
    """Validate that a webhook URL is safe to deliver to.

    Blocks:
    - Private/internal IP addresses (SSRF prevention)
    - Localhost and loopback addresses
    - Non-HTTPS URLs (except in development)
    - Known internal domains
    """
    parsed = urlparse(url)

    # Require HTTPS
    if parsed.scheme != "https":
        raise ValueError("Webhook URL must use HTTPS")

    # Block internal hostnames
    hostname = parsed.hostname
    blocked_hosts = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
    if hostname in blocked_hosts:
        raise ValueError("Webhook URL cannot point to localhost")

    blocked_domains = {".internal", ".local", ".corp"}
    if any(hostname.endswith(d) for d in blocked_domains):
        raise ValueError("Webhook URL cannot point to internal domains")

    # Resolve DNS and check for private IPs
    try:
        resolved_ips = socket.getaddrinfo(hostname, None)
        for _, _, _, _, addr in resolved_ips:
            ip = ipaddress.ip_address(addr[0])
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                raise ValueError(
                    f"Webhook URL resolves to private IP: {ip}"
                )
    except socket.gaierror:
        raise ValueError(f"Cannot resolve hostname: {hostname}")

    return True
```

### IP Allowlisting

Publish the IP addresses your webhook requests originate from, so receivers can restrict inbound traffic:

```python
@app.get("/webhooks/ips")
async def webhook_source_ips():
    """Return the IP addresses used for webhook delivery.

    Consumers can use this to configure firewall rules
    and restrict their webhook endpoint to known sources.
    """
    return {
        "ipv4": [
            "203.0.113.10/32",
            "203.0.113.11/32",
        ],
        "ipv6": [
            "2001:db8::10/128",
            "2001:db8::11/128",
        ],
        "last_updated": "2025-01-01",
    }
```

---

## 9. Building a Webhook System

### Complete Producer

```python
import asyncio
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class WebhookDispatcher:
    """Coordinates webhook event delivery to all matching subscriptions.

    Events are dispatched asynchronously. Each subscription receives
    its own delivery attempt with independent retry logic.
    """

    def __init__(self, redis_client, db_session):
        self.redis = redis_client
        self.db = db_session
        self.dlq = DeadLetterQueue(redis_client)

    async def dispatch(self, event: WebhookEvent):
        """Dispatch an event to all matching subscriptions."""
        subscriptions = await self.db.execute(
            select(WebhookSubscriptionModel).where(
                WebhookSubscriptionModel.active == True,
                WebhookSubscriptionModel.events.contains([event.type]),
            )
        )

        tasks = []
        for sub in subscriptions.scalars():
            tasks.append(
                self._deliver_to_subscription(event, sub)
            )

        # Deliver to all subscriptions concurrently
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _deliver_to_subscription(self, event, subscription):
        """Deliver an event to a single subscription with retry."""
        payload = event.model_dump(mode="json")
        result = await deliver_webhook(
            url=subscription.url,
            payload=payload,
            secret=subscription.secret,
            max_retries=5,
        )

        # Log the delivery result
        await self._record_delivery(result, subscription)

        # Move to DLQ if all retries failed
        if not result.delivered:
            await self.dlq.add(DeadLetterEntry(
                event_id=event.id,
                subscription_id=subscription.id,
                url=subscription.url,
                payload=payload,
                attempts=[
                    {
                        "attempt": a.attempt_number,
                        "status_code": a.status_code,
                        "error": a.error,
                    }
                    for a in result.attempts
                ],
            ))

            # Disable subscription after too many consecutive failures
            await self._check_subscription_health(subscription)


# Usage: dispatch events from your business logic
@app.post("/orders", status_code=201)
async def create_order(order: OrderCreate, request: Request):
    new_order = await save_order(order)

    # Dispatch webhook event
    dispatcher = WebhookDispatcher(
        request.app.state.redis,
        request.app.state.db,
    )
    await dispatcher.dispatch(WebhookEvent(
        type="order.created",
        data={"order": new_order.dict()},
    ))

    return new_order
```

---

## 10. Exercises

### Exercise 1: Build a Webhook Consumer

Build a FastAPI application that receives webhooks from a payment provider. Implement:

- `POST /webhooks/payment` endpoint
- HMAC-SHA256 signature verification
- Timestamp validation (reject events older than 5 minutes)
- Idempotency check using Redis
- Event routing based on event type (`payment.succeeded`, `payment.failed`, `payment.refunded`)
- Proper error handling and logging

### Exercise 2: Webhook Delivery System

Build a complete webhook delivery system as a FastAPI application:

- `POST /webhooks` — Register a subscription (with URL validation, no SSRF)
- `GET /webhooks` — List subscriptions
- `DELETE /webhooks/{id}` — Remove a subscription
- `POST /events` — Trigger a test event and deliver to all matching subscriptions
- Include HMAC signing, exponential backoff retry (3 attempts), and delivery logging

### Exercise 3: Dead Letter Queue Manager

Build an admin interface for managing failed webhook deliveries:

- `GET /admin/dlq` — List all failed deliveries with pagination
- `GET /admin/dlq/{event_id}` — Get details of a specific failure (all attempts, errors)
- `POST /admin/dlq/{event_id}/retry` — Retry a single failed delivery
- `POST /admin/dlq/retry-all` — Retry all failed deliveries
- `DELETE /admin/dlq/{event_id}` — Mark as resolved and remove
- Add filters: by subscription, by event type, by date range

### Exercise 4: Webhook Testing Tool

Build a webhook testing tool similar to webhook.site:

- `POST /test-endpoints` — Create a temporary test endpoint (returns a unique URL)
- `GET /test-endpoints/{id}/events` — List all events received by the test endpoint
- Test endpoints auto-expire after 1 hour
- Support configurable response status codes (to test retry behavior)
- Store the raw request headers and body for debugging

### Exercise 5: Subscription Health Monitor

Design and implement a subscription health monitoring system that:

- Tracks the success/failure rate for each subscription over the last 24 hours
- Automatically disables subscriptions with more than 10 consecutive failures
- Sends an email notification to the subscription owner before disabling
- Provides an endpoint to re-enable a disabled subscription
- Includes a health dashboard showing all subscriptions and their delivery statistics

---

## 11. References

- [Stripe Webhooks Guide](https://stripe.com/docs/webhooks)
- [GitHub Webhooks](https://docs.github.com/en/webhooks)
- [Standard Webhooks Specification](https://www.standardwebhooks.com/)
- [HMAC (RFC 2104)](https://tools.ietf.org/html/rfc2104)
- [Svix (Webhook Infrastructure)](https://www.svix.com/)
- [webhook.site (Testing Tool)](https://webhook.site/)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Celery Documentation](https://docs.celeryq.dev/)

---

**Previous**: [API Testing and Validation](./11_API_Testing_and_Validation.md) | [Overview](./00_Overview.md) | **Next**: [API Gateway Patterns](./13_API_Gateway_Patterns.md)

**License**: CC BY-NC 4.0
