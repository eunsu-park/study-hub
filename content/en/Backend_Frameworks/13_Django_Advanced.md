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
