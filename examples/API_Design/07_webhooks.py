#!/usr/bin/env python3
"""Example: Webhooks — Sending and Receiving

Demonstrates a complete webhook system with:
- Webhook registration and management
- HMAC-SHA256 payload signing (sender side)
- Signature verification (receiver side)
- Retry logic with exponential backoff
- Idempotency via event IDs

Related lesson: 13_Webhooks_and_Event_Driven_APIs.md

Run:
    pip install "fastapi[standard]" httpx

    # Terminal 1: Start the webhook sender (producer API)
    uvicorn 07_webhooks:sender_app --reload --port 8000

    # Terminal 2: Start the webhook receiver (consumer)
    uvicorn 07_webhooks:receiver_app --reload --port 8001

Test:
    # Register a webhook endpoint
    http POST :8000/api/v1/webhooks url=http://localhost:8001/webhook/receive events:='["order.created","order.updated"]' secret=my-webhook-secret

    # Trigger an event (sends webhook to registered endpoints)
    http POST :8000/api/v1/orders product="Widget" quantity:=5
"""

import hashlib
import hmac
import json
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, HttpUrl

# =============================================================================
# WEBHOOK SENDER (Producer API)
# =============================================================================
# The sender is the service that generates events and delivers them to
# registered webhook endpoints. Key responsibilities:
# 1. Register/manage webhook subscriptions
# 2. Sign payloads with HMAC for authenticity
# 3. Deliver with retries and exponential backoff
# 4. Include unique event IDs for idempotency

sender_app = FastAPI(title="Webhook Sender API", version="1.0.0")


# --- Schemas ---

class WebhookRegister(BaseModel):
    """Registration request for a new webhook subscription."""
    url: str = Field(..., description="HTTPS endpoint to receive events")
    events: list[str] = Field(
        ...,
        description="Event types to subscribe to",
        examples=[["order.created", "order.updated"]],
    )
    secret: str = Field(
        ...,
        min_length=16,
        description="Shared secret for HMAC signature verification",
    )


class WebhookSubscription(BaseModel):
    id: str
    url: str
    events: list[str]
    active: bool = True
    created_at: str


class WebhookEvent(BaseModel):
    """The payload delivered to webhook endpoints."""
    id: str = Field(..., description="Unique event ID for idempotency")
    type: str = Field(..., description="Event type (e.g., order.created)")
    timestamp: str = Field(..., description="ISO 8601 event timestamp")
    data: dict = Field(..., description="Event-specific payload")


class WebhookDeliveryResult(BaseModel):
    subscription_id: str
    url: str
    status_code: Optional[int] = None
    success: bool
    attempts: int
    error: Optional[str] = None


# --- In-memory stores ---

subscriptions_db: dict[str, dict] = {}
delivery_log: list[dict] = []


# --- HMAC Signing ---

def sign_payload(payload: str, secret: str) -> str:
    """Create HMAC-SHA256 signature of the payload.

    The signature proves the webhook was sent by us (not a third party).
    The receiver computes the same signature and compares.

    HMAC prevents:
    - Payload tampering (integrity)
    - Spoofed webhooks (authenticity)
    """
    return hmac.new(
        secret.encode("utf-8"),
        payload.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


# --- Delivery with retries ---

async def deliver_webhook(
    event: WebhookEvent,
    subscription: dict,
    max_retries: int = 3,
) -> WebhookDeliveryResult:
    """Deliver a webhook event with exponential backoff retries.

    Retry strategy:
    - Attempt 1: immediate
    - Attempt 2: after 2 seconds
    - Attempt 3: after 4 seconds
    - Give up after max_retries

    Success criteria: 2xx status code within 10 seconds.
    """
    payload_json = event.model_dump_json()
    signature = sign_payload(payload_json, subscription["secret"])

    headers = {
        "Content-Type": "application/json",
        "X-Webhook-ID": event.id,
        "X-Webhook-Signature": f"sha256={signature}",
        "X-Webhook-Timestamp": event.timestamp,
        "User-Agent": "MyApp-Webhook/1.0",
    }

    result = WebhookDeliveryResult(
        subscription_id=subscription["id"],
        url=subscription["url"],
        success=False,
        attempts=0,
    )

    async with httpx.AsyncClient() as client:
        for attempt in range(max_retries):
            result.attempts = attempt + 1
            try:
                response = await client.post(
                    subscription["url"],
                    content=payload_json,
                    headers=headers,
                    timeout=10.0,
                )
                result.status_code = response.status_code

                if 200 <= response.status_code < 300:
                    result.success = True
                    break

                # Non-2xx: retry
                result.error = f"HTTP {response.status_code}"

            except httpx.TimeoutException:
                result.error = "Request timed out"
            except httpx.ConnectError:
                result.error = "Connection refused"
            except Exception as e:
                result.error = str(e)

            # Exponential backoff before retry
            if attempt < max_retries - 1:
                backoff = 2 ** attempt  # 1s, 2s, 4s
                time.sleep(backoff)

    delivery_log.append(result.model_dump())
    return result


# --- Sender Routes ---

@sender_app.post(
    "/api/v1/webhooks",
    response_model=WebhookSubscription,
    status_code=status.HTTP_201_CREATED,
    tags=["Webhooks"],
)
def register_webhook(body: WebhookRegister):
    """Register a new webhook subscription.

    The secret is used to sign payloads with HMAC-SHA256. Store the secret
    securely — if compromised, rotate it immediately.
    """
    sub_id = str(uuid4())
    subscription = {
        "id": sub_id,
        "url": body.url,
        "events": body.events,
        "secret": body.secret,  # In production, encrypt at rest
        "active": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    subscriptions_db[sub_id] = subscription
    return WebhookSubscription(**{k: v for k, v in subscription.items() if k != "secret"})


@sender_app.get("/api/v1/webhooks", response_model=list[WebhookSubscription], tags=["Webhooks"])
def list_webhooks():
    """List all webhook subscriptions (secrets are never returned)."""
    return [
        WebhookSubscription(**{k: v for k, v in s.items() if k != "secret"})
        for s in subscriptions_db.values()
    ]


@sender_app.delete("/api/v1/webhooks/{sub_id}", status_code=204, tags=["Webhooks"])
def delete_webhook(sub_id: str):
    """Unsubscribe a webhook endpoint."""
    if sub_id not in subscriptions_db:
        raise HTTPException(status_code=404, detail="Subscription not found")
    del subscriptions_db[sub_id]


@sender_app.post("/api/v1/orders", tags=["Orders"])
async def create_order(product: str, quantity: int = 1):
    """Create an order — triggers 'order.created' webhook event.

    This simulates a business action that generates a webhook event.
    All subscriptions listening for 'order.created' will be notified.
    """
    order_id = str(uuid4())
    order = {
        "order_id": order_id,
        "product": product,
        "quantity": quantity,
        "total": quantity * 29.99,
        "status": "confirmed",
    }

    # Build webhook event
    event = WebhookEvent(
        id=str(uuid4()),
        type="order.created",
        timestamp=datetime.now(timezone.utc).isoformat(),
        data=order,
    )

    # Deliver to all matching subscriptions
    results = []
    for sub in subscriptions_db.values():
        if sub["active"] and "order.created" in sub["events"]:
            result = await deliver_webhook(event, sub)
            results.append(result.model_dump())

    return {
        "order": order,
        "webhooks_sent": len(results),
        "delivery_results": results,
    }


@sender_app.get("/api/v1/webhooks/deliveries", tags=["Webhooks"])
def get_delivery_log():
    """View webhook delivery history for debugging."""
    return delivery_log[-50:]  # Last 50 deliveries


# =============================================================================
# WEBHOOK RECEIVER (Consumer)
# =============================================================================
# The receiver is the service that processes incoming webhook events.
# Key responsibilities:
# 1. Verify the HMAC signature (reject spoofed events)
# 2. Respond quickly (< 5 seconds) to avoid sender timeouts
# 3. Handle events idempotently (same event ID = same result)
# 4. Process asynchronously if work is slow

receiver_app = FastAPI(title="Webhook Receiver", version="1.0.0")

# Track processed event IDs for idempotency
processed_events: set[str] = set()
received_events: list[dict] = []

# Shared secret (in production, load from environment variable)
WEBHOOK_SECRET = "my-webhook-secret"


def verify_signature(payload: bytes, signature_header: str, secret: str) -> bool:
    """Verify the HMAC-SHA256 signature of an incoming webhook.

    IMPORTANT: Use hmac.compare_digest() for timing-safe comparison.
    A naive == comparison leaks timing information that could be exploited.
    """
    if not signature_header.startswith("sha256="):
        return False

    expected_sig = signature_header[7:]  # Remove "sha256=" prefix
    computed_sig = hmac.new(
        secret.encode("utf-8"),
        payload,
        hashlib.sha256,
    ).hexdigest()

    # Timing-safe comparison prevents timing attacks
    return hmac.compare_digest(computed_sig, expected_sig)


@receiver_app.post("/webhook/receive", tags=["Receiver"])
async def receive_webhook(
    request: Request,
    x_webhook_id: str = Header(...),
    x_webhook_signature: str = Header(...),
    x_webhook_timestamp: str = Header(...),
):
    """Receive and process a webhook event.

    Processing steps:
    1. Verify HMAC signature (reject if invalid)
    2. Check for duplicate event ID (idempotency)
    3. Process the event
    4. Return 200 quickly (offload heavy work to background)
    """
    body = await request.body()

    # Step 1: Verify signature
    if not verify_signature(body, x_webhook_signature, WEBHOOK_SECRET):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature",
        )

    # Step 2: Idempotency check
    if x_webhook_id in processed_events:
        # Already processed — return 200 (not an error, just a no-op)
        return {"status": "already_processed", "event_id": x_webhook_id}

    # Step 3: Parse and process
    payload = json.loads(body)
    event_type = payload.get("type", "unknown")

    # Record the event
    processed_events.add(x_webhook_id)
    received_events.append({
        "event_id": x_webhook_id,
        "type": event_type,
        "received_at": datetime.now(timezone.utc).isoformat(),
        "data": payload.get("data", {}),
    })

    # In production, enqueue heavy processing to a background worker
    # (e.g., Celery, Redis Queue, or asyncio task)

    # Step 4: Return 200 quickly
    return {"status": "received", "event_id": x_webhook_id, "type": event_type}


@receiver_app.get("/webhook/events", tags=["Receiver"])
def list_received_events():
    """List all received webhook events (for debugging)."""
    return received_events


# =============================================================================
# ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys
    import uvicorn

    if "--receiver" in sys.argv:
        uvicorn.run("07_webhooks:receiver_app", host="127.0.0.1", port=8001, reload=True)
    else:
        uvicorn.run("07_webhooks:sender_app", host="127.0.0.1", port=8000, reload=True)
