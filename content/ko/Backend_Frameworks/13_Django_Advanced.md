# 13. Django 고급

**이전**: [Django REST Framework](./12_Django_REST_Framework.md) | **다음**: [API 설계 패턴](./14_API_Design_Patterns.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ASGI를 사용하는 Django Channels로 실시간 WebSocket 통신을 구현한다
2. 주기적 작업(periodic task)과 재시도(retry) 로직을 포함하여 비동기 작업 처리를 위해 Redis와 함께 Celery를 설정한다
3. 뷰별(per-view), 템플릿 프래그먼트(template fragment), 저수준(low-level) 등 여러 수준에서 Django의 캐시 프레임워크를 적용한다
4. Django 시그널(signal)과 커스텀 미들웨어(middleware)를 사용해 횡단 관심사(cross-cutting concerns)를 구현한다
5. 관리 자동화를 위한 커스텀 관리 명령(management command)을 만든다

---

프로덕션 Django 애플리케이션은 동기 HTTP 이상의 것을 필요로 합니다: 실시간 통신, 백그라운드 작업, 캐싱, 확장성 훅(hook). 이 레슨은 작동하는 프로토타입과 프로덕션 시스템을 구분하는 고급 기능을 다룹니다.

## 목차

1. [WebSocket을 위한 Django Channels](#1-websocket을-위한-django-channels)
2. [비동기 작업을 위한 Celery](#2-비동기-작업을-위한-celery)
3. [캐시와 브로커로서의 Redis](#3-캐시와-브로커로서의-redis)
4. [Django 시그널](#4-django-시그널)
5. [커스텀 관리 명령](#5-커스텀-관리-명령)
6. [캐싱 전략](#6-캐싱-전략)
7. [Django 미들웨어](#7-django-미들웨어)
8. [연습 문제](#8-연습-문제)

---

## 1. WebSocket을 위한 Django Channels

Channels는 ASGI를 통해 Django가 WebSocket을 처리할 수 있도록 확장합니다.

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

### WebSocket 컨슈머(Consumer)

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

클라이언트: `new WebSocket("ws://localhost/ws/chat/general/")`.

---

## 2. 비동기 작업을 위한 Celery

Celery는 이메일 발송, 보고서 생성, API 호출 등 시간이 오래 걸리는 작업을 요청 사이클 밖에서 실행합니다.

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

### 태스크 정의와 호출

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
# 뷰에서 호출
send_notification_email.delay(user.id, "New comment", "Someone commented...")
send_notification_email.apply_async(args=[...], countdown=300, queue="emails")
```

### 주기적 작업 (Celery Beat)

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
celery -A mysite worker --loglevel=info      # 워커
celery -A mysite beat --loglevel=info        # 스케줄러
```

---

## 3. 캐시와 브로커로서의 Redis

Redis는 Celery의 브로커이자 Django의 캐시 백엔드로 동시에 사용됩니다:

```python
# settings.py
CACHES = {
    "default": {
        "BACKEND": "django.core.cache.backends.redis.RedisCache",
        "LOCATION": "redis://127.0.0.1:6379/1",  # Celery와 다른 DB 번호
    },
}
```

```python
from django.core.cache import cache

cache.set("total_posts", 42, timeout=600)
total = cache.get("total_posts")
total = cache.get_or_set("total_posts", lambda: Post.objects.count(), 600)
cache.incr("page_views")                    # 원자적 증가
cache.set_many({"k1": "v1", "k2": "v2"})   # 일괄 작업
```

---

## 4. Django 시그널

시그널(Signal)은 옵저버(observer) 패턴을 구현합니다 -- 액션이 발생할 때 분리된 방식으로 알림을 전달합니다.

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

`apps.py`에 등록:

```python
class BlogConfig(AppConfig):
    name = "blog"
    def ready(self):
        import blog.signals  # noqa: F401
```

내장 시그널: `pre_save`, `post_save`, `pre_delete`, `post_delete`, `m2m_changed`, `user_logged_in`. 커스텀 시그널: `Signal()` + `send()`.

---

## 5. 커스텀 관리 명령

관리 작업, 크론 작업, 데이터 작업을 위해 `manage.py`를 확장합니다.

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

        # ... 게시물 생성 ...
        self.stdout.write(self.style.SUCCESS(f"Created {options['posts']} posts."))
```

```bash
python manage.py seed_data --posts 200 --clear
```

---

## 6. 캐싱 전략

### 뷰별 캐시(Per-View Cache)

```python
from django.views.decorators.cache import cache_page

@cache_page(60 * 15)  # 15분
def post_list(request):
    return render(request, "blog/post_list.html", {"posts": Post.objects.all()})
```

### 템플릿 프래그먼트 캐시(Template Fragment Cache)

```html
{% load cache %}
{% cache 300 sidebar %}
    <div class="sidebar">{% for c in categories %}...{% endfor %}</div>
{% endcache %}
{% cache 300 user_dash request.user.id %}  <!-- 사용자별 캐시 -->
    ...
{% endcache %}
```

### 뷰에서의 저수준 캐시(Low-Level Cache)

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

### 시그널을 통한 캐시 무효화

```python
@receiver([post_save, post_delete], sender=Post)
def invalidate_cache(sender, instance, **kwargs):
    cache.delete(f"post_{instance.pk}")
    cache.delete("post_list_page_1")
```

---

## 7. Django 미들웨어

미들웨어(Middleware)는 모든 요청/응답 사이클에 훅(hook)으로 연결됩니다. 실행 순서: 요청은 미들웨어 1-2-3을 통해 뷰로 흘러가고, 응답은 3-2-1 순으로 돌아옵니다.

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

`settings.py`의 `MIDDLEWARE` 목록에 등록합니다. 순서가 중요합니다 -- 타이밍 미들웨어는 앞에, 점검 모드(maintenance) 미들웨어는 인증 이후에 배치하세요.

---

## 8. 연습 문제

### 문제 1: 실시간 알림

Channels를 사용해 알림 시스템을 구축하세요: 사용자별 개인 WebSocket 채널, 댓글 게시 시 실시간 푸시, 데이터베이스에 저장하는 `Notification` 모델, 읽음 처리를 위한 REST 엔드포인트.

### 문제 2: 작업 파이프라인

이미지 업로드를 위한 Celery 파이프라인을 설계하세요: 유효성 검사 → 썸네일 생성 (3가지 크기) → 최적화 → 데이터베이스 업데이트. `chain()`, 재시도 로직, 실패 시 데드레터(dead-letter) 태스크를 사용하세요.

### 문제 3: 다중 레벨 캐싱

블로그 홈페이지를 위한 캐싱을 구현하세요: 익명 사용자를 위한 전체 페이지 캐시(5분), 로그인 사용자를 위한 프래그먼트 및 저수준 캐시, 게시물 변경 시 시그널 기반 무효화, 캐시 워밍(cache-warming) 관리 명령.

### 문제 4: 감사 추적

감사 시스템을 만드세요: 모든 요청을 로깅하는 미들웨어, 변경 전후 값을 함께 기록하는 시그널, `audit_report` 관리 명령, 로그를 조회하는 DRF 엔드포인트.

### 문제 5: 관리 명령

다음을 구축하세요: `healthcheck`(DB, Redis, Celery, 디스크), `export_data`(날짜 및 상태 필터를 포함한 JSON/CSV), `cleanup`(만료된 세션, 오래된 로그, 고아 파일) `--dry-run` 옵션 포함.

---

## 참고 자료

- [Django Channels 공식 문서](https://channels.readthedocs.io/)
- [Celery 공식 문서](https://docs.celeryq.dev/)
- [Django 캐싱 프레임워크](https://docs.djangoproject.com/en/5.1/topics/cache/)
- [Django 시그널](https://docs.djangoproject.com/en/5.1/topics/signals/)
- [Django 미들웨어](https://docs.djangoproject.com/en/5.1/topics/http/middleware/)

---

**이전**: [Django REST Framework](./12_Django_REST_Framework.md) | **다음**: [API 설계 패턴](./14_API_Design_Patterns.md)
