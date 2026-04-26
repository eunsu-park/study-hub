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

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. Django 시그널 밑에 깔린 옵저버 패턴, 캐시 계층(LocMem/Memcached/Redis/database)과 그 일관성 트레이드오프, 그리고 다시 살펴보는 미들웨어 체인 합성을 다룹니다.

1. [WebSocket을 위한 Django Channels](#1-websocket을-위한-django-channels)
2. [비동기 작업을 위한 Celery](#2-비동기-작업을-위한-celery)
3. [캐시와 브로커로서의 Redis](#3-캐시와-브로커로서의-redis)
4. [Django 시그널](#4-django-시그널)
5. [커스텀 관리 명령](#5-커스텀-관리-명령)
6. [캐싱 전략](#6-캐싱-전략)
7. [Django 미들웨어](#7-django-미들웨어)
8. [연습 문제](#8-연습-문제)

---

## 이론과 원리

"고급" Django 기능들은 무관한 도구 8개가 아니라 세 개의 직교 패턴 주위에 모입니다. 각 패턴을 분리해서 이해하면, 이 영역에 들어오는 새 기능이 어디에 맞을지 예측할 수 있게 됩니다.

- **(A) 옵저버 패턴: 정전적 예시로서의 Django 시그널** — 한 프로세스 안에서의 분리된 "publish/subscribe", 그리고 그 위험들.
- **(B) 캐시 계층** — 다른 지연시간, 용량, 일관성 모델을 가진 여러 계층, 그리고 그것들을 엮는 cache-aside 패턴.
- **(C) 미들웨어 체인 — 고급 워크로드가 거기에 더하는 것** — 비동기 미들웨어, 상태 주입, 프로덕션의 응답 후처리.

### A. 옵저버 패턴: Django 시그널

Django 시그널은 옵저버 패턴을 구현합니다. 발행자가 누가 듣는지 모른 채 이벤트를 emit하고, 구독자가 관심을 등록해 이벤트가 발화하면 실행됩니다. 내장 시그널은 모델 전체 생명주기, 요청/응답 사이클, 인증 이벤트를 다룹니다.

#### A.1 메커니즘

```python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=Order)
def update_inventory(sender, instance, created, **kwargs):
    if created:
        Inventory.objects.filter(sku=instance.sku).update(qty=F("qty") - 1)
```

`Order.objects.create(...)`가 실행되면 Django는 행이 commit된 후 `post_save`를 emit합니다. `(post_save, Order)`에 등록된 모든 receiver가 등록 순서로 동기적으로 실행됩니다. 발행자(`Order.save()` 코드 경로)는 `update_inventory`의 존재를 모릅니다.

#### A.2 모델 생명주기의 7단계

| 시그널 | 발화 시점 |
|--------|-----------|
| `pre_init` | 모델 인스턴스의 `__init__` 이전 |
| `post_init` | `__init__` 이후 |
| `pre_save` | `INSERT`/`UPDATE` 이전 |
| `post_save` | `INSERT`/`UPDATE` 이후 |
| `pre_delete` | `DELETE` 이전 |
| `post_delete` | `DELETE` 이후 |
| `m2m_changed` | 다대다 관계가 수정됨 |

추가로 요청/응답 시그널(`request_started`, `request_finished`, `got_request_exception`)과 인증 시그널(`user_logged_in`, `user_logged_out`, `user_login_failed`)이 있습니다.

#### A.3 시그널의 위험

시그널이 제공하는 분리는 그 주된 위험이기도 합니다. 세 가지 실패 모드가 반복됩니다.

1. **숨겨진 부수 효과.** 호출 코드에서 `Order.objects.create(...)`를 읽어서는 그 결과로 재고 업데이트, 이메일 송신, analytics 이벤트가 모두 일어난다는 것을 알 수 없습니다. 새 개발자는 부수 효과를 예측할 수 없습니다.
2. **추적하기 힘든 순서 버그.** `post_save` receiver는 등록 순서로 실행되며, 이는 `INSTALLED_APPS` 순서에 의존할 수 있습니다. 앱 간 순서는 깨지기 쉽습니다.
3. **동기 실행.** 시그널 receiver는 요청의 트랜잭션 안에서 실행되어, 끝날 때까지 응답을 블로킹합니다. 느린 시그널 핸들러가 빠른 엔드포인트를 명백한 원인 없이 느리게 만듭니다.

실용적 가이드: 중요한 비즈니스 로직은 시그널보다 **명시적 메서드 호출**을 선호하세요. 진짜로 분리된 횡단 관심사(audit log, analytics)에 시그널을 사용하고 — 느린 것은 모두 Celery(§B 스타일 백그라운드 태스크)에 떠넘기세요.

### B. 캐시 계층

캐싱은 stale 데이터를 더 낮은 지연시간과 데이터베이스에 더 낮은 부하와 거래합니다. Django의 cache framework는 여러 백엔드 위에 추상화되어 있으며, 각 백엔드가 속도/용량/일관성 삼각형의 다른 위치에 있습니다.

#### B.1 백엔드 분류

| 백엔드 | 사는 곳 | 지연시간 | 용량 | 프로세스 간 공유 |
|---------|-------------|---------|----------|-------------------------|
| `LocMemCache` | 워커 프로세스 RAM | 나노초 | 작음(RAM) | 아니오(워커별) |
| `MemcachedCache` | 프로세스 외 데몬 | ~0.5 ms | RAM 한계 | 예 |
| `RedisCache` | 프로세스 외 데몬 | ~0.5 ms | RAM 한계 | 예 |
| `DatabaseCache` | 데이터베이스 테이블 | ~5 ms | 디스크 | 예 |
| `FileBasedCache` | 파일시스템 | ~5 ms | 디스크 | 예(호스트별) |

`LocMemCache`는 가장 빠르지만 각 Gunicorn 워커가 자체 사본을 가집니다. 워커 A에 값을 설정해도 워커 B에는 보이지 않습니다. 읽기 위주의 프로세스별 데이터에 유용합니다.

`Redis`와 `Memcached`는 프로덕션 기본값입니다. 모든 워커와 호스트 간에 공유되고, 지연시간이 낮으며, 축출 정책이 내장되어 있습니다.

#### B.2 Cache-aside 패턴

표준 읽기 흐름:

```python
def get_post(post_id):
    key = f"post:{post_id}"
    post = cache.get(key)
    if post is None:                       # 캐시 miss
        post = Post.objects.get(pk=post_id)
        cache.set(key, post, timeout=300)  # 5분 동안 채움
    return post
```

이것이 주는 세 가지 보장:

1. 캐시는 *오로지* 데이터베이스의 데이터를 제공합니다. 데이터를 소유하지 않습니다.
2. Miss는 자동 치유됩니다. 다음 읽기가 그것을 채웁니다.
3. TTL이 staleness를 제한합니다. 데이터는 최대 300초 됩니다.

`Post`가 업데이트되면 고전적인 무효화 문제가 돌아옵니다. 두 전략:

- **TTL만.** TTL까지의 staleness를 받아들이고, 능동적으로 무효화하지 않습니다. 단순하고 견고합니다.
- **능동적 무효화.** `Post.save()`에서 `cache.delete(f"post:{post.id}")`. 더 정확하지만 결합을 추가하고 프로세스 간 업데이트에서 무효화 누락 위험이 있습니다.

#### B.3 캐시 스탬피드와 dogpile

뜨거운 키가 무거운 트래픽 아래에서 만료되면 수백 개의 요청이 모두 동시에 miss하고 모두 다시 계산하려고 달려듭니다. "dogpile"이 데이터베이스를 과부하시킵니다 — 정확히 캐시가 가장 필요했던 순간에. 방어책:

- **확률적 조기 갱신.** 무작위 검사에 기반해 TTL 약간 전에 갱신.
- **Locking.** 첫 miss가 Redis 락을 잡고 다시 계산하고, 다른 것들은 잠시 대기하거나 stale을 제공.
- **Request coalescing**을 캐시 프레임워크 수준에서 — 같은 키에 대한 보류 중 lookup이 하나의 백엔드 호출을 공유합니다.

이 패턴은 레슨 20(Redis Caching Patterns)에서 반복됩니다 — Django 특유의 문제가 아니라 일반적인 분산 캐시 문제입니다.

#### B.4 Django의 캐시 통합 수준

Django는 네 가지 세분성을 노출합니다.

- **사이트 전체 캐시.** 미들웨어가 코드 변경 없이 모든 페이지 응답을 캐시합니다.
- **뷰별 캐시.** 뷰에 `@cache_page(60 * 15)` 데코레이터.
- **템플릿 fragment 캐시.** 템플릿에서 `{% cache 600 sidebar %}...{% endcache %}`.
- **저수준 API.** 임의 키에 대한 `cache.get / cache.set`(§B.2 cache-aside).

올바른 수준은 무엇이 어떤 빈도로 바뀌느냐에 달려 있습니다. 뷰별은 마케팅 페이지에 좋고, 저수준은 부분 객체 캐싱에 유일한 선택입니다.

### C. 프로덕션의 미들웨어 체인

레슨 10 §C.2가 Django 미들웨어를 도입했습니다. 프로덕션 워크로드는 체인에 세 가지 관심사를 더합니다.

#### C.1 비동기 미들웨어 (Django 4+)

미들웨어는 동기, 비동기, 또는 둘 다일 수 있습니다.

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

요청이 체인을 지날 때 비동기 미들웨어 주위의 동기 미들웨어는 Django가 스레드를 띄우게 만듭니다(반대도 마찬가지) — 경계마다 컨텍스트 스위치 비용이 있습니다. 모범 사례: 성능을 위해 체인을 일관되게 동기 또는 일관되게 비동기로 유지하고, 잘 정의된 경계에서만 섞으세요.

#### C.2 상태 주입: `request.foo` 패턴

`request`에 속성(`request.user`, `request.session`, `request.tenant`, `request.trace_id`)을 추가하는 미들웨어는 다운스트림 모든 뷰가 명시적 Depends 스타일 주입 없이 그 값들을 쓸 수 있게 해줍니다. 비용은 타입 정보의 손실입니다 — 뷰는 어떤 속성이 존재한다고 보장되는지 모릅니다.

현대 Django의 해결책: `Generic[T]` 스타일 request 타이핑을 사용하는 타입 어노테이션 미들웨어, 또는 상태가 `authentication_classes`의 `request.user`를 통해 요청별로 명시적인 DRF 패턴으로의 이동.

#### C.3 응답 후처리: 압축, 보안 헤더, ID 에코

미들웨어 응답 단계(레슨 10 §C.2, `get_response` 반환 이후)는 프로덕션이 다음을 더하는 곳입니다.

- `GZipMiddleware` — 임계값보다 큰 응답을 압축.
- `SecurityMiddleware` — strict-transport-security, content-type-options, frame-options.
- 로그 상관관계를 위해 클라이언트에 `X-Request-ID`를 에코하는 커스텀 미들웨어.
- Trace 컨텍스트 전파(레슨 17 — 분산 트레이싱).

각각이 역순으로 실행됩니다. 가장 바깥에 등록된 미들웨어가 최종, 완전히 처리된 응답을 봅니다.

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (Channels for WebSockets)은 ASGI(레슨 02 §A) 위의 Django입니다 — 비동기 미들웨어(§C.1)와 장기 연결을 처리하는 consumer 클래스.
- §2 (Celery)는 §A.3의 "느린 시그널 핸들러"의 올바른 집입니다 — 동기 receiver 대신 명시적 out-of-band 태스크.
- §3 (캐시와 브로커로서의 Redis)는 캐싱(§B.2)과 §2를 뒷받침하는 Celery 브로커 모두에 대한 §B.1 백엔드 선택입니다.
- §4 (Django 시그널)은 §A 옵저버 패턴을 구체적인 코드로, §A.3의 위험을 명시적으로 풀어 갑니다.
- §5 (관리 명령)은 웹 요청에 대한 CLI 짝입니다 — 같은 Django 앱, HTTP 계층 없음.
- §6 (캐싱 전략)은 §B.4 통합 수준을 예제와 함께 따라갑니다.
- §7 (Django 미들웨어)은 §C.1–§C.3 미들웨어 패턴을 구체적인 프로덕션 예제로 확장합니다.

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
