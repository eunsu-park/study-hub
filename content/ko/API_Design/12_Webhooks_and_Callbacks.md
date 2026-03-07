# 12. Webhook과 콜백(Webhooks and Callbacks)

**이전**: [API 테스트와 검증](./11_API_Testing_and_Validation.md) | **다음**: [API Gateway 패턴](./13_API_Gateway_Patterns.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- 구독자에게 이벤트 알림을 안정적으로 전달하는 webhook 시스템을 설계할 수 있다
- HMAC을 사용한 서명 검증을 구현하여 webhook의 진위를 보장할 수 있다
- 지수 백오프(exponential backoff)와 데드 레터 큐(dead letter queue)를 적용한 재시도 정책을 구현할 수 있다
- 멱등성 키(idempotency key)를 사용하여 중복 이벤트 처리를 방지할 수 있다
- 재생 공격(replay attack)과 비인가 접근으로부터 webhook 엔드포인트를 보호할 수 있다
- FastAPI를 사용하여 완전한 webhook 생산자와 소비자를 구축할 수 있다

---

## 목차

1. [Webhook vs. 폴링](#1-webhook-vs-폴링)
2. [Webhook 설계 원칙](#2-webhook-설계-원칙)
3. [이벤트 페이로드](#3-이벤트-페이로드)
4. [서명 검증 (HMAC)](#4-서명-검증-hmac)
5. [재시도 정책과 전달 보장](#5-재시도-정책과-전달-보장)
6. [멱등성 키](#6-멱등성-키)
7. [데드 레터 큐](#7-데드-레터-큐)
8. [Webhook 보안](#8-webhook-보안)
9. [Webhook 시스템 구축](#9-webhook-시스템-구축)
10. [연습 문제](#10-연습-문제)
11. [참고 자료](#11-참고-자료)

---

## 1. Webhook vs. 폴링

API에는 클라이언트에게 변경 사항을 알리는 두 가지 모델이 있습니다:

**폴링(Polling)**: 클라이언트가 주기적으로 "변경된 사항이 있나요?"라고 질문합니다.

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

**Webhook**: 서버가 이벤트 발생 시 클라이언트에게 푸시합니다.

```
Client                          Server
  |                               |
  |   (registers callback URL)    |
  |                               |
  |                               |-- order.created event occurs
  |<-- POST /webhook {event} -----|  (immediate push)
  |-- 200 OK ------------------->|
```

### 비교

| 항목 | 폴링 | Webhook |
|------|------|---------|
| 지연 시간 | 수초~수분 | 거의 실시간 |
| 서버 부하 | 높음 (빈 응답이 많음) | 낮음 (이벤트 발생 시에만 푸시) |
| 클라이언트 복잡도 | 단순 (HTTP GET만 사용) | 엔드포인트를 호스팅해야 함 |
| 신뢰성 | 클라이언트가 재시도를 제어 | 서버가 실패를 처리해야 함 |
| 방화벽 이슈 | 없음 (아웃바운드만) | 클라이언트가 인바운드를 수락해야 함 |
| 데이터 최신성 | 폴링 간격에 따라 다름 | 즉시 |

---

## 2. Webhook 설계 원칙

### 등록(Registration)

소비자가 콜백 URL을 등록하고 수신할 이벤트를 지정할 수 있도록 합니다:

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

## 3. 이벤트 페이로드

이벤트 페이로드는 자체 완결적(self-contained)으로 설계합니다. 수신자가 이벤트를 이해하기 위해 추가 API 호출을 할 필요가 없어야 합니다.

### 이벤트 엔벨로프 구조

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

### 이벤트 명명 규칙

이벤트 타입에는 `resource.action` 패턴을 사용합니다:

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

## 4. 서명 검증 (HMAC)

Webhook 수신자는 들어오는 요청이 실제로 예상된 발신자로부터 온 것인지 검증해야 합니다. HMAC(Hash-based Message Authentication Code)가 이 보장을 제공합니다.

### 생산자: 페이로드 서명

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

### 소비자: 서명 검증

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

## 5. 재시도 정책과 전달 보장

Webhook 전달은 네트워크 문제, 수신자 다운타임, 타임아웃 등 다양한 이유로 실패할 수 있습니다. 견고한 webhook 시스템은 지수 백오프(exponential backoff)를 사용하여 실패한 전달을 재시도합니다.

### 지수 백오프를 사용한 재시도

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

### 프로덕션 재시도 스케줄

프로덕션 시스템에서는 더 긴 재시도 간격과 태스크 큐를 사용합니다:

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

## 6. 멱등성 키

수신자는 재시도나 네트워크 문제로 인해 동일한 이벤트를 두 번 이상 받을 수 있습니다. 멱등성 키를 사용하면 수신자가 중복을 감지하고 건너뛸 수 있습니다.

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

## 7. 데드 레터 큐

모든 재시도가 실패하면, 이벤트는 조용히 버려지는 대신 수동 조사를 위해 데드 레터 큐(DLQ)로 이동해야 합니다.

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

## 8. Webhook 보안

### 위협 모델

| 위협 | 공격 벡터 | 완화 방법 |
|------|----------|----------|
| 스푸핑(Spoofing) | 공격자가 가짜 webhook 요청을 전송 | HMAC 서명 검증 |
| 재생 공격(Replay) | 공격자가 이전에 캡처한 요청을 재전송 | 타임스탬프 검증 |
| 정보 유출 | Webhook URL이 로그에 노출 | URL 검증, HTTPS만 사용 |
| SSRF | 공격자가 내부 URL을 등록 | URL 허용 목록, 사설 IP 차단 |
| DoS | 공격자가 webhook 수신자를 과부하 | webhook 엔드포인트에 속도 제한 적용 |

### Webhook 등록 시 SSRF 방지

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

### IP 허용 목록

Webhook 요청이 발신되는 IP 주소를 공개하여 수신자가 인바운드 트래픽을 제한할 수 있도록 합니다:

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

## 9. Webhook 시스템 구축

### 완전한 생산자(Producer)

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

## 10. 연습 문제

### 문제 1: Webhook 소비자 구축

결제 제공업체로부터 webhook을 수신하는 FastAPI 애플리케이션을 구축하세요. 다음을 구현하세요:

- `POST /webhooks/payment` 엔드포인트
- HMAC-SHA256 서명 검증
- 타임스탬프 검증 (5분 이상 된 이벤트 거부)
- Redis를 사용한 멱등성 검사
- 이벤트 타입에 따른 이벤트 라우팅 (`payment.succeeded`, `payment.failed`, `payment.refunded`)
- 적절한 오류 처리와 로깅

### 문제 2: Webhook 전달 시스템

완전한 webhook 전달 시스템을 FastAPI 애플리케이션으로 구축하세요:

- `POST /webhooks` — 구독 등록 (URL 검증, SSRF 방지 포함)
- `GET /webhooks` — 구독 목록 조회
- `DELETE /webhooks/{id}` — 구독 제거
- `POST /events` — 테스트 이벤트를 트리거하고 일치하는 모든 구독에 전달
- HMAC 서명, 지수 백오프 재시도 (3회), 전달 로깅 포함

### 문제 3: 데드 레터 큐 관리자

실패한 webhook 전달을 관리하기 위한 관리자 인터페이스를 구축하세요:

- `GET /admin/dlq` — 페이지네이션이 포함된 모든 실패 전달 목록
- `GET /admin/dlq/{event_id}` — 특정 실패의 상세 정보 (모든 시도, 오류)
- `POST /admin/dlq/{event_id}/retry` — 단일 실패 전달 재시도
- `POST /admin/dlq/retry-all` — 모든 실패 전달 재시도
- `DELETE /admin/dlq/{event_id}` — 해결됨으로 표시하고 제거
- 필터 추가: 구독별, 이벤트 타입별, 날짜 범위별

### 문제 4: Webhook 테스트 도구

webhook.site와 유사한 webhook 테스트 도구를 구축하세요:

- `POST /test-endpoints` — 임시 테스트 엔드포인트 생성 (고유 URL 반환)
- `GET /test-endpoints/{id}/events` — 테스트 엔드포인트가 수신한 모든 이벤트 목록
- 테스트 엔드포인트는 1시간 후 자동 만료
- 구성 가능한 응답 상태 코드 지원 (재시도 동작 테스트용)
- 디버깅을 위한 원시 요청 헤더와 본문 저장

### 문제 5: 구독 상태 모니터

다음 기능을 가진 구독 상태 모니터링 시스템을 설계하고 구현하세요:

- 지난 24시간 동안 각 구독의 성공/실패 비율 추적
- 10회 이상 연속 실패한 구독을 자동으로 비활성화
- 비활성화 전에 구독 소유자에게 이메일 알림 전송
- 비활성화된 구독을 다시 활성화하는 엔드포인트 제공
- 모든 구독과 전달 통계를 보여주는 상태 대시보드 포함

---

## 11. 참고 자료

- [Stripe Webhooks Guide](https://stripe.com/docs/webhooks)
- [GitHub Webhooks](https://docs.github.com/en/webhooks)
- [Standard Webhooks Specification](https://www.standardwebhooks.com/)
- [HMAC (RFC 2104)](https://tools.ietf.org/html/rfc2104)
- [Svix (Webhook Infrastructure)](https://www.svix.com/)
- [webhook.site (Testing Tool)](https://webhook.site/)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
- [Celery Documentation](https://docs.celeryq.dev/)

---

**이전**: [API 테스트와 검증](./11_API_Testing_and_Validation.md) | [개요](./00_Overview.md) | **다음**: [API Gateway 패턴](./13_API_Gateway_Patterns.md)

**License**: CC BY-NC 4.0
