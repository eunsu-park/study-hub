# 13. Django Advanced

**Previous**: [Django REST Framework](./12_Django_REST_Framework.md) | **Next**: [API Design Patterns](./14_API_Design_Patterns.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Implement real-time WebSocket communication using Django Channels with ASGI
2. Configure Celery with Redis for asynchronous task processing, including periodic tasks and retry logic
3. Apply Django's caching framework at multiple levels (per-view, template fragment, low-level)
4. Use Django signals and custom middleware to implement cross-cutting concerns
5. Create custom management commands for administrative automation

---

Production Django applications need more than synchronous HTTP: real-time communication, background jobs, caching, and extensibility hooks. This lesson covers the advanced features that separate a working prototype from a production system.

## Table of Contents

1. [Django Channels for WebSockets](#1-django-channels-for-websockets)
2. [Celery for Async Tasks](#2-celery-for-async-tasks)
3. [Redis as Cache and Broker](#3-redis-as-cache-and-broker)
4. [Django Signals](#4-django-signals)
5. [Custom Management Commands](#5-custom-management-commands)
6. [Caching Strategies](#6-caching-strategies)
7. [Django Middleware](#7-django-middleware)
8. [Practice Problems](#8-practice-problems)

---

## 1. Django Channels for WebSockets

Channels extends Django to handle WebSockets via ASGI.

```bash
pip install channels~=4.1 channels-redis~=4.2
```

```python
# settings.py
INSTALLED_APPS = ["daphne", ...other_apps..., "channels"]
ASGI_APPLICATION = "mysite.asgi.application"
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels_redis.core.RedisChannelLayer",
        "CONFIG": {"hosts": [("127.0.0.1", 6379)]},
    },
}
```

### WebSocket Consumer

```python
# chat/consumers.py
import json
from channels.generic.websocket import AsyncWebSocketConsumer

class ChatConsumer(AsyncWebSocketConsumer):
    async def connect(self):
        self.room = self.scope["url_route"]["kwargs"]["room"]
        self.group = f"chat_{self.room}"
        await self.channel_layer.group_add(self.group, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group, self.channel_name)

    async def receive(self, text_data):
        data = json.loads(text_data)
        await self.channel_layer.group_send(self.group, {
            "type": "chat.message",
            "message": data["message"],
            "username": self.scope["user"].username or "Anonymous",
        })

    async def chat_message(self, event):
        await self.send(text_data=json.dumps(event))
```

```python
# chat/routing.py
from django.urls import re_path
from . import consumers
websocket_urlpatterns = [
    re_path(r"ws/chat/(?P<room>\w+)/$", consumers.ChatConsumer.as_asgi()),
]
```

Client: `new WebSocket("ws://localhost/ws/chat/general/")`.

---

## 2. Celery for Async Tasks

Celery runs long tasks outside the request cycle: emails, reports, API calls.

```bash
pip install celery~=5.4 redis~=5.0
```

```python
# mysite/celery.py
import os
from celery import Celery

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "mysite.settings")
app = Celery("mysite")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()
```

```python
# settings.py
CELERY_BROKER_URL = "redis://127.0.0.1:6379/0"
CELERY_RESULT_BACKEND = "redis://127.0.0.1:6379/0"
CELERY_TASK_TIME_LIMIT = 300
```

### Defining and Calling Tasks

```python
# blog/tasks.py
from celery import shared_task

@shared_task(bind=True, max_retries=3, default_retry_delay=60)
def send_notification_email(self, user_id: int, subject: str, message: str):
    from django.contrib.auth import get_user_model
    from django.core.mail import send_mail
    try:
        user = get_user_model().objects.get(pk=user_id)
        send_mail(subject, message, "noreply@example.com", [user.email])
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60 * (2 ** self.request.retries))
```

```python
# Call from a view
send_notification_email.delay(user.id, "New comment", "Someone commented...")
send_notification_email.apply_async(args=[...], countdown=300, queue="emails")
```

### Periodic Tasks (Celery Beat)

```python
# settings.py
from celery.schedules import crontab
CELERY_BEAT_SCHEDULE = {
    "cleanup-sessions": {
        "task": "blog.tasks.cleanup_sessions",
        "schedule": crontab(hour=3, minute=0),
    },
}
```

```bash
celery -A mysite worker --loglevel=info      # Worker
celery -A mysite beat --loglevel=info        # Scheduler
```

---

## 3. Redis as Cache and Broker

Redis serves as both Celery's broker and Django's cache backend:

```python
# settings.py
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",  # Different DB than Celery
    },
}
```

```python
from django.core.cache import cache

cache.set("total_posts", 42, timeout=600)
total = cache.get("total_posts")
total = cache.get_or_set("total_posts", lambda: Post.objects.count(), 600)
cache.incr("page_views")                    # Atomic increment
cache.set_many({"k1": "v1", "k2": "v2"})   # Batch operations
```

---

## 4. Django Signals

Signals implement the observer pattern -- decoupled notification when actions occur.

```python
# blog/signals.py
from django.db.models.signals import post_save
from django.dispatch import receiver
from .models import Post

@receiver(post_save, sender=Post)
def notify_on_publish(sender, instance, created, **kwargs):
    if not created and instance.status == "published":
        from blog.tasks import send_notification_email
        for sub in instance.author.subscribers.all():
            send_notification_email.delay(sub.id, f"New: {instance.title}", "...")
```

Register in `apps.py`:

```python
class BlogConfig(AppConfig):
    name = "blog"
    def ready(self):
        import blog.signals  # noqa: F401
```

Built-in signals: `pre_save`, `post_save`, `pre_delete`, `post_delete`, `m2m_changed`, `user_logged_in`. Custom signals: `Signal()` + `send()`.

---

### Theory: The Observer Pattern: Django Signals

Django signals implement the observer pattern: a publisher emits an event without knowing who listens; subscribers register interest and run when the event fires. Built-in signals cover the entire model lifecycle, request/response cycle, and authentication events.

#### A.1 The mechanism

```python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=Order)
def update_inventory(sender, instance, created, **kwargs):
    if created:
        Inventory.objects.filter(sku=instance.sku).update(qty=F("qty") - 1)
```

When `Order.objects.create(...)` runs, Django emits `post_save` after the row is committed. Every receiver registered for `(post_save, Order)` runs synchronously, in registration order. The publisher (the `Order.save()` code path) does not know `update_inventory` exists.

#### A.2 The seven phases of the model lifecycle

| Signal | Fires when |
|--------|-----------|
| `pre_init` | Before `__init__` of a model instance |
| `post_init` | After `__init__` |
| `pre_save` | Before `INSERT`/`UPDATE` |
| `post_save` | After `INSERT`/`UPDATE` |
| `pre_delete` | Before `DELETE` |
| `post_delete` | After `DELETE` |
| `m2m_changed` | A many-to-many relationship was modified |

Plus request/response signals (`request_started`, `request_finished`, `got_request_exception`) and auth signals (`user_logged_in`, `user_logged_out`, `user_login_failed`).

#### A.3 The dangers of signals

The decoupling that signals provide is also their main hazard. Three failure modes recur:

1. **Hidden side effects.** Reading `Order.objects.create(...)` in the calling code does not reveal that inventory updates, email sends, and analytics events all happen as a consequence. New developers cannot predict the side effects.
2. **Hard-to-trace ordering bugs.** `post_save` receivers run in the order they registered, which may depend on `INSTALLED_APPS` order. Cross-app ordering is fragile.
3. **Synchronous execution.** Signal receivers run inside the request's transaction, blocking the response until they finish. A slow signal handler turns a fast endpoint into a slow one with no obvious cause.

The pragmatic guidance: prefer **explicit method calls** over signals for important business logic. Use signals for genuinely decoupled cross-cutting concerns (audit logs, analytics) — and offload anything slow to Celery (§B-style background tasks).

## 5. Custom Management Commands

Extend `manage.py` for admin tasks, cron jobs, and data operations.

```python
# blog/management/commands/seed_data.py
from django.core.management.base import BaseCommand
from blog.models import Post, Category

class Command(BaseCommand):
    help = "Seed the database with sample data"

    def add_arguments(self, parser):
        parser.add_argument("--posts", type=int, default=50)
        parser.add_argument("--clear", action="store_true")

    def handle(self, *args, **options):
        if options["clear"]:
            Post.objects.all().delete()
            self.stdout.write("Cleared existing posts.")

        categories = ["Python", "Django", "JavaScript", "DevOps"]
        for name in categories:
            Category.objects.get_or_create(name=name, defaults={"slug": name.lower()})

        # ... create posts ...
        self.stdout.write(self.style.SUCCESS(f"Created {options['posts']} posts."))
```

```bash
python manage.py seed_data --posts 200 --clear
```

---

## 6. Caching Strategies

### Theory: The Cache Hierarchy

Caching trades stale data for lower latency and lower load on the database. Django's cache framework abstracts over multiple backends, each with a different position on the speed/capacity/consistency triangle.

#### B.1 The backend taxonomy

| Backend | Where lives | Latency | Capacity | Shared across processes |
|---------|-------------|---------|----------|-------------------------|
| `LocMemCache` | Worker process RAM | nanoseconds | small (RAM) | No (per-worker) |
| `MemcachedCache` | Out-of-process daemon | ~0.5 ms | RAM-bounded | Yes |
| `RedisCache` | Out-of-process daemon | ~0.5 ms | RAM-bounded | Yes |
| `DatabaseCache` | Database table | ~5 ms | disk | Yes |
| `FileBasedCache` | Filesystem | ~5 ms | disk | Yes (per-host) |

`LocMemCache` is fastest but each Gunicorn worker has its own copy. Setting a value on worker A is invisible to worker B. Useful for read-mostly per-process data.

`Redis` and `Memcached` are the production defaults: shared across all workers and hosts, low-latency, eviction policies built in.

#### B.2 The cache-aside pattern

The standard read flow:

```python
def get_post(post_id):
    key = f"post:{post_id}"
    post = cache.get(key)
    if post is None:                       # cache miss
        post = Post.objects.get(pk=post_id)
        cache.set(key, post, timeout=300)  # populate for 5 minutes
    return post
```

Three guarantees this gives:

1. The cache *only* serves data from the database; it does not own data.
2. A miss is auto-healing: the next read populates it.
3. The TTL bounds staleness: data is at most 300 seconds old.

The classic invalidation problem returns when `Post` is updated. Two strategies:

- **TTL only.** Accept staleness up to the TTL; do not actively invalidate. Simple and resilient.
- **Active invalidation.** On `Post.save()`, `cache.delete(f"post:{post.id}")`. More accurate but adds coupling and risks missed invalidations on cross-process updates.

#### B.3 Cache stampede and the dogpile

When a hot key expires under heavy traffic, hundreds of requests all miss simultaneously and all rush to recompute. The "dogpile" overloads the database — exactly when you needed the cache most. Defenses:

- **Probabilistic early refresh.** Refresh slightly before TTL based on a random check.
- **Locking.** First miss takes a Redis lock and recomputes; others briefly wait or serve stale.
- **Request coalescing** at the cache framework level — pending lookups for the same key share one backend call.

This pattern recurs in Lesson 20 (Redis Caching Patterns) — it is a general distributed-cache problem, not Django-specific.

#### B.4 The Django levels of cache integration

Django exposes four granularities:

- **Per-site cache.** Middleware caches every page response with no code change.
- **Per-view cache.** `@cache_page(60 * 15)` decorator on a view.
- **Template fragment cache.** `{% cache 600 sidebar %}...{% endcache %}` in templates.
- **Low-level API.** `cache.get / cache.set` for arbitrary keys (the §B.2 cache-aside).

The right level depends on what changes and on what cadence. Per-view is great for marketing pages; low-level is the only choice for partial-object caching.

### Per-View Cache

```python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # 15 minutes
def post_list(request):
    return render(request, "blog/post_list.html", {"posts": Post.objects.all()})
```

### Template Fragment Cache

```html
{% load cache %}
{% cache 300 sidebar %}
    <div class="sidebar">{% for c in categories %}...{% endfor %}</div>
{% endcache %}
{% cache 300 user_dash request.user.id %}  <!-- Per-user cache -->
    ...
{% endcache %}
```

### Low-Level Cache in Views

```python
def post_detail(request, pk: int):
    cache_key = f"post_{pk}"
    data = cache.get(cache_key)
    if data is None:
        post = get_object_or_404(Post, pk=pk)
        data = {"post": post, "comments": list(post.comments.all())}
        cache.set(cache_key, data, timeout=600)
    return render(request, "blog/post_detail.html", data)
```

### Cache Invalidation via Signals

```python
@receiver([post_save, post_delete], sender=Post)
def invalidate_cache(sender, instance, **kwargs):
    cache.delete(f"post_{instance.pk}")
    cache.delete("post_list_page_1")
```

---

## 7. Django Middleware

Middleware hooks into every request/response cycle. Execution: Request flows through middleware 1-2-3 to the view, then response flows back 3-2-1.

```python
# mysite/middleware.py
import time, uuid, logging
from django.http import JsonResponse

logger = logging.getLogger(__name__)

class RequestTimingMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        start = time.perf_counter()
        response = self.get_response(request)
        ms = (time.perf_counter() - start) * 1000
        response["X-Request-Duration-Ms"] = f"{ms:.2f}"
        return response

class RequestIDMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        response = self.get_response(request)
        response["X-Request-ID"] = request.request_id
        return response

class MaintenanceModeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        from django.core.cache import cache
        if cache.get("maintenance_mode") and not request.user.is_staff:
            return JsonResponse({"error": "Under maintenance"}, status=503)
        return self.get_response(request)
```

Register in `MIDDLEWARE` list in `settings.py`. Order matters -- place timing early, maintenance after auth.

---

### Theory: The Middleware Chain in Production

Lesson 10 §C.2 introduced Django middleware. Production workloads add three concerns to the chain.

#### C.1 Async middleware (Django 4+)

A middleware can be sync, async, or both:

```python
class MyMiddleware:
    sync_capable = True
    async_capable = True

    def __init__(self, get_response):
        self.get_response = get_response
        self._is_coroutine = asyncio.iscoroutinefunction(get_response)

    def __call__(self, request):
        if self._is_coroutine:
            return self._async_call(request)
        return self._sync_call(request)
```

When a request travels through the chain, sync middleware around async middleware causes Django to spin up a thread (and vice versa) — there is a per-boundary context-switch cost. Best practice: keep the chain consistently sync or consistently async for performance, mixing only at well-defined boundaries.

#### C.2 State injection: `request.foo` patterns

Middleware that adds attributes to `request` (`request.user`, `request.session`, `request.tenant`, `request.trace_id`) lets every view downstream use those values without explicit Depends-style injection. The cost is loss of type information — the view does not know what attributes are guaranteed to exist.

The fix in modern Django: type-annotated middleware that uses `Generic[T]`-style request typing, or moving toward DRF's pattern where state is explicit per-request via `request.user` from `authentication_classes`.

#### C.3 Response post-processing: compression, security headers, ID echoing

The middleware response phase (Lesson 10 §C.2, after `get_response` returns) is where production adds:

- `GZipMiddleware` — compresses responses larger than the threshold.
- `SecurityMiddleware` — strict-transport-security, content-type-options, frame-options.
- Custom middleware that echoes a `X-Request-ID` to clients for log correlation.
- Trace-context propagation (Lesson 17 — distributed tracing).

Each runs in reverse order. The outermost-registered middleware sees the final, fully-processed response.

## 8. Practice Problems

### Problem 1: Real-Time Notifications

Build a notification system with Channels: personal WebSocket channel per user, real-time push when comments are posted, database-backed `Notification` model, and a REST endpoint to mark as read.

### Problem 2: Task Pipeline

Design a Celery pipeline for image uploads: validate -> generate thumbnails (3 sizes) -> optimize -> update database. Use `chain()`, retry logic, and a dead-letter task for failures.

### Problem 3: Multi-Level Caching

Implement caching for a blog homepage: full-page cache for anonymous users (5 min), fragment + low-level caches for logged-in users, signal-based invalidation on post changes, and a cache-warming management command.

### Problem 4: Audit Trail

Create an audit system: middleware logging all requests, signals logging model changes with before/after values, a `audit_report` management command, and a DRF endpoint to query logs.

### Problem 5: Management Commands

Build: `healthcheck` (DB, Redis, Celery, disk), `export_data` (JSON/CSV with date and status filters), and `cleanup` (expired sessions, old logs, orphaned files) with `--dry-run`.

---

## References

- [Django Channels Documentation](https://channels.readthedocs.io/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [Django Caching Framework](https://docs.djangoproject.com/en/5.1/topics/cache/)
- [Django Signals](https://docs.djangoproject.com/en/5.1/topics/signals/)
- [Django Middleware](https://docs.djangoproject.com/en/5.1/topics/http/middleware/)

---

**Previous**: [Django REST Framework](./12_Django_REST_Framework.md) | **Next**: [API Design Patterns](./14_API_Design_Patterns.md)
