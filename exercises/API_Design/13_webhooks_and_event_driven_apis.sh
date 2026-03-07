#!/bin/bash
# Exercises for Lesson 13: Webhooks and Event-Driven APIs
# Topic: API_Design
# Solutions to practice problems from the lesson.

# === Exercise 1: Webhook Payload Signing ===
# Problem: Implement HMAC-SHA256 signing for webhook payloads and
# build a receiver that verifies the signature.
exercise_1() {
    echo "=== Exercise 1: Webhook Payload Signing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import hashlib
import hmac
import json
import time


def sign_webhook(payload: dict, secret: str, timestamp: int = None) -> dict:
    """Sign a webhook payload with HMAC-SHA256.

    The signature includes the timestamp to prevent replay attacks.
    Format: "v1={timestamp}.{payload_json}"

    Returns headers to include in the webhook delivery.
    """
    ts = timestamp or int(time.time())
    payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)

    # Sign: timestamp + "." + payload (prevents replay attacks)
    message = f"{ts}.{payload_json}"
    signature = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        "X-Webhook-Signature": f"v1={signature}",
        "X-Webhook-Timestamp": str(ts),
        "Content-Type": "application/json",
    }


def verify_webhook(
    payload_bytes: bytes,
    signature_header: str,
    timestamp_header: str,
    secret: str,
    tolerance_seconds: int = 300,
) -> bool:
    """Verify a webhook signature.

    Steps:
    1. Check timestamp is within tolerance (prevent replay attacks)
    2. Reconstruct the signed message
    3. Compute expected signature
    4. Compare with timing-safe comparison (prevent timing attacks)
    """
    # Step 1: Replay protection
    ts = int(timestamp_header)
    now = int(time.time())
    if abs(now - ts) > tolerance_seconds:
        raise ValueError(f"Timestamp too old: {abs(now - ts)}s > {tolerance_seconds}s")

    # Step 2: Reconstruct message
    message = f"{ts}.{payload_bytes.decode('utf-8')}"

    # Step 3: Compute expected signature
    expected = hmac.new(
        secret.encode("utf-8"),
        message.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    # Step 4: Extract signature from header
    if not signature_header.startswith("v1="):
        return False
    received = signature_header[3:]

    # Timing-safe comparison (prevents timing attacks)
    return hmac.compare_digest(expected, received)


# Demo
secret = "whsec_test_secret_key_12345"
payload = {"event": "order.created", "data": {"order_id": "ord_123", "total": 99.99}}

# Sender signs
headers = sign_webhook(payload, secret)
print(f"Signature: {headers['X-Webhook-Signature']}")
print(f"Timestamp: {headers['X-Webhook-Timestamp']}")

# Receiver verifies
payload_json = json.dumps(payload, separators=(",", ":"), sort_keys=True)
is_valid = verify_webhook(
    payload_json.encode(),
    headers["X-Webhook-Signature"],
    headers["X-Webhook-Timestamp"],
    secret,
)
print(f"Signature valid: {is_valid}")
SOLUTION
}

# === Exercise 2: Webhook Retry with Backoff ===
# Problem: Build a webhook delivery system with exponential backoff retries
# and dead letter queue for permanently failed deliveries.
exercise_2() {
    echo "=== Exercise 2: Webhook Retry with Backoff ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time
import httpx
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from enum import Enum


class DeliveryStatus(str, Enum):
    PENDING = "pending"
    SUCCESS = "success"
    RETRYING = "retrying"
    FAILED = "failed"       # Permanently failed → dead letter queue


@dataclass
class DeliveryAttempt:
    timestamp: str
    status_code: Optional[int] = None
    error: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class WebhookDelivery:
    id: str
    url: str
    payload: dict
    status: DeliveryStatus = DeliveryStatus.PENDING
    attempts: list[DeliveryAttempt] = field(default_factory=list)
    next_retry_at: Optional[str] = None
    max_retries: int = 5

    @property
    def attempt_count(self) -> int:
        return len(self.attempts)


# Dead letter queue for permanently failed deliveries
dead_letter_queue: list[WebhookDelivery] = []


async def deliver_with_retries(delivery: WebhookDelivery, secret: str) -> WebhookDelivery:
    """Deliver a webhook with exponential backoff.

    Retry schedule:
    - Attempt 1: immediate
    - Attempt 2: after 10 seconds
    - Attempt 3: after 30 seconds
    - Attempt 4: after 2 minutes
    - Attempt 5: after 10 minutes
    - After max retries: move to dead letter queue
    """
    backoff_schedule = [0, 10, 30, 120, 600]  # seconds

    async with httpx.AsyncClient() as client:
        for attempt_num in range(delivery.max_retries):
            # Wait before retry (skip for first attempt)
            if attempt_num > 0:
                wait = backoff_schedule[min(attempt_num, len(backoff_schedule) - 1)]
                delivery.status = DeliveryStatus.RETRYING
                delivery.next_retry_at = datetime.now(timezone.utc).isoformat()
                await asyncio.sleep(wait)  # In production, use a task queue

            start = time.monotonic()
            attempt = DeliveryAttempt(
                timestamp=datetime.now(timezone.utc).isoformat(),
            )

            try:
                response = await client.post(
                    delivery.url,
                    json=delivery.payload,
                    timeout=10.0,
                )
                attempt.status_code = response.status_code
                attempt.duration_ms = (time.monotonic() - start) * 1000

                if 200 <= response.status_code < 300:
                    delivery.status = DeliveryStatus.SUCCESS
                    delivery.attempts.append(attempt)
                    return delivery

                attempt.error = f"HTTP {response.status_code}"

            except httpx.TimeoutException:
                attempt.error = "Timeout after 10s"
                attempt.duration_ms = (time.monotonic() - start) * 1000
            except Exception as e:
                attempt.error = str(e)
                attempt.duration_ms = (time.monotonic() - start) * 1000

            delivery.attempts.append(attempt)

    # All retries exhausted → dead letter queue
    delivery.status = DeliveryStatus.FAILED
    dead_letter_queue.append(delivery)
    return delivery
SOLUTION
}

# === Exercise 3: Event Payload Design ===
# Problem: Design a consistent event payload schema that supports multiple
# event types with a shared envelope and type-specific data.
exercise_3() {
    echo "=== Exercise 3: Event Payload Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from pydantic import BaseModel, Field
from typing import Any, Literal
from datetime import datetime, timezone
from uuid import uuid4


class WebhookEvent(BaseModel):
    """Standard webhook event envelope.

    All events share the same top-level structure. The `type` field
    determines how to interpret the `data` payload.
    """
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Unique event ID for idempotency",
    )
    type: str = Field(
        ...,
        description="Event type: resource.action (e.g., order.created)",
    )
    api_version: str = Field(
        "2025-06-15",
        description="API version that generated this event",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
    )
    data: dict = Field(
        ...,
        description="Event-specific payload",
    )
    # For related resource lookups
    resource: dict = Field(
        default_factory=dict,
        description="Links to the affected resource",
    )


# Event type definitions

class OrderCreatedData(BaseModel):
    order_id: str
    customer_id: str
    total: float
    currency: str
    items: list[dict]


class OrderStatusChangedData(BaseModel):
    order_id: str
    previous_status: str
    new_status: str
    changed_by: str
    reason: str = ""


class PaymentCompletedData(BaseModel):
    payment_id: str
    order_id: str
    amount: float
    method: str  # card, bank_transfer, etc.


# Factory functions for type-safe event creation

def order_created(order: dict) -> WebhookEvent:
    return WebhookEvent(
        type="order.created",
        data=OrderCreatedData(
            order_id=order["id"],
            customer_id=order["customer_id"],
            total=order["total"],
            currency=order.get("currency", "USD"),
            items=order.get("items", []),
        ).model_dump(),
        resource={
            "type": "order",
            "id": order["id"],
            "url": f"/api/v1/orders/{order['id']}",
        },
    )


def order_status_changed(order_id: str, old: str, new: str, by: str) -> WebhookEvent:
    return WebhookEvent(
        type="order.status_changed",
        data=OrderStatusChangedData(
            order_id=order_id,
            previous_status=old,
            new_status=new,
            changed_by=by,
        ).model_dump(),
        resource={
            "type": "order",
            "id": order_id,
            "url": f"/api/v1/orders/{order_id}",
        },
    )


# Usage:
event = order_created({
    "id": "ord_123",
    "customer_id": "cust_456",
    "total": 99.99,
    "items": [{"product": "Widget", "qty": 2}],
})
print(event.model_dump_json(indent=2))
SOLUTION
}

# === Exercise 4: Idempotent Event Processing ===
# Problem: Build a webhook receiver that processes events idempotently,
# handling duplicate deliveries and out-of-order events.
exercise_4() {
    echo "=== Exercise 4: Idempotent Event Processing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from fastapi import FastAPI, Request, HTTPException, Header
from datetime import datetime, timezone
import hashlib
import hmac
import json

app = FastAPI()

# Idempotency store: event_id → processing result
processed_events: dict[str, dict] = {}

# Event sequence tracking per resource
resource_versions: dict[str, str] = {}  # resource_id → last_event_timestamp

WEBHOOK_SECRET = "whsec_test_secret_12345"


@app.post("/webhook/receive")
async def receive_webhook(
    request: Request,
    x_webhook_id: str = Header(...),
    x_webhook_signature: str = Header(...),
    x_webhook_timestamp: str = Header(...),
):
    """Process webhook events idempotently.

    Idempotency guarantees:
    1. Same event processed at most once (even if delivered multiple times)
    2. Out-of-order events do not override newer data
    3. All events are acknowledged quickly (within 5 seconds)
    """
    body = await request.body()

    # Step 1: Verify signature (reject spoofed events)
    message = f"{x_webhook_timestamp}.{body.decode()}"
    expected = hmac.new(
        WEBHOOK_SECRET.encode(),
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

    if not x_webhook_signature.startswith("v1="):
        raise HTTPException(status_code=401, detail="Invalid signature format")

    received_sig = x_webhook_signature[3:]
    if not hmac.compare_digest(expected, received_sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    # Step 2: Idempotency check — already processed?
    if x_webhook_id in processed_events:
        return {
            "status": "already_processed",
            "event_id": x_webhook_id,
            "original_result": processed_events[x_webhook_id],
        }

    # Step 3: Parse event
    event = json.loads(body)
    event_type = event.get("type", "")
    event_time = event.get("created_at", "")
    resource_id = event.get("resource", {}).get("id", "")

    # Step 4: Out-of-order detection
    if resource_id and resource_id in resource_versions:
        last_time = resource_versions[resource_id]
        if event_time < last_time:
            # Older event arrived after newer one — skip data update
            result = {
                "status": "skipped_stale",
                "reason": f"Event {event_time} is older than {last_time}",
            }
            processed_events[x_webhook_id] = result
            return result

    # Step 5: Process the event
    result = process_event(event_type, event.get("data", {}))

    # Step 6: Record processing
    processed_events[x_webhook_id] = result
    if resource_id:
        resource_versions[resource_id] = event_time

    return {"status": "processed", "event_id": x_webhook_id, "result": result}


def process_event(event_type: str, data: dict) -> dict:
    """Route events to their handlers."""
    handlers = {
        "order.created": handle_order_created,
        "order.status_changed": handle_order_status_changed,
        "payment.completed": handle_payment_completed,
    }
    handler = handlers.get(event_type)
    if handler:
        return handler(data)
    return {"action": "ignored", "reason": f"Unknown event type: {event_type}"}


def handle_order_created(data: dict) -> dict:
    return {"action": "order_recorded", "order_id": data.get("order_id")}

def handle_order_status_changed(data: dict) -> dict:
    return {"action": "status_updated", "new_status": data.get("new_status")}

def handle_payment_completed(data: dict) -> dict:
    return {"action": "payment_recorded", "payment_id": data.get("payment_id")}
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 13: Webhooks and Event-Driven APIs"
echo "=================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
