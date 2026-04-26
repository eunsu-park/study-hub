# 10. Django 기초

**이전**: [Express 테스트](./09_Express_Testing.md) | **다음**: [Django ORM](./11_Django_ORM.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. Django의 "배터리 포함(batteries included)" 철학을 설명하고, 마이크로 프레임워크와의 차이를 비교한다
2. Django 프로젝트와 앱을 생성하고, 각 생성 파일의 목적을 설명한다
3. MTV(Model-Template-View) 패턴을 설명하고, 전통적인 MVC와 대응시킨다
4. `path()`, `include()`, 경로 변환기(path converter)를 사용해 URL 라우팅을 설정한다
5. 일반적인 HTTP 작업을 위한 함수 기반 뷰(FBV)와 클래스 기반 뷰(CBV)를 모두 구현한다

---

Django는 빠른 개발과 깔끔하고 실용적인 설계를 장려하는 고수준 Python 웹 프레임워크입니다. FastAPI가 최소한의 핵심만 제공하는 반면, Django는 ORM, 인증, 관리자 인터페이스(admin interface), 폼 처리 등을 모두 기본으로 제공하며 처음부터 함께 동작합니다. Django 5.x는 생성 필드(generated fields), 관리자(admin)에서의 패싯 필터(facet filters), 간소화된 폼 렌더링(form rendering)을 추가했습니다.

## 목차

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. Django를 실행하는 WSGI 인터페이스, MVC와 비교한 MTV 패턴, 그리고 미들웨어·URL 해석·뷰·응답을 거치는 요청 생명주기를 다룹니다.

1. [Django 철학](#1-django-철학)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [MTV 패턴](#3-mtv-패턴)
4. [URL 라우팅](#4-url-라우팅)
5. [뷰](#5-뷰)
6. [모델 기초](#6-모델-기초)
7. [Django 관리자 인터페이스](#7-django-관리자-인터페이스)
8. [설정과 구성](#8-설정과-구성)
9. [연습 문제](#9-연습-문제)

---

## 이론과 원리

Django는 FastAPI보다 15년 더 오래되었고, async Python 자체보다 앞섭니다. 그 설계 선택들 — 동기 WSGI, 독특한 MTV 명명, 관습 중심의 프로젝트 구조 — 은 무엇에 반응한 것이고 무엇을 가능하게 했는지 알아야만 말이 됩니다. 세 개념이 거의 모든 후속 결정을 다룹니다.

- **(A) WSGI: Django가 세워진 동기 서버 인터페이스** — 그리고 Django 4+가 어떻게 모델을 깨지 않고 ASGI를 추가했는지.
- **(B) MTV vs MVC** — 다른 이름의 같은 아이디어, 그리고 Django가 왜 이 이름을 골랐는지.
- **(C) 요청 생명주기** — 미들웨어 → URL 해석기 → 뷰 → 응답, 매 단계에 hook이 있음.

### A. WSGI: 동기 서버 인터페이스

WSGI(Web Server Gateway Interface, PEP 3333)는 Python 웹 앱이 서버(Gunicorn, uWSGI, mod_wsgi)와 대화하는 계약입니다. 2003년부터 ~2018년까지 Python 웹의 *그* 표준이었으며, Django는 그 위에 정확히 세워졌습니다.

#### A.1 WSGI callable

WSGI 앱은 동기 callable 하나입니다.

```python
def application(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"Hello"]
```

- `environ`은 요청을 담은 dict입니다: 메서드, 경로, 헤더, 파일 같은 객체로서의 본문.
- `start_response`는 바이트를 yield하기 *전에* 상태와 헤더로 정확히 한 번 호출하는 콜백입니다.
- 반환값은 바이트의 iterable입니다 — 응답 본문.

그게 다입니다. 이벤트 루프도, 스트리밍 프리미티브도, WebSocket도 없습니다. 단순함이 WSGI를 보편화시켰고, 같은 단순함이 Python 웹 생태계가 다음 10년 동안 ASGI를 필요로 하게 만들었습니다.

#### A.2 이것이 Django 런타임 모델에 시사하는 것

Django의 요청 처리기는 WSGI callable입니다. 각 요청은 한 Gunicorn 워커 프로세스에서 끝까지 실행됩니다. 동시성은 N개의 워커 프로세스(보통 `2 × CPU + 1`)를 돌리는 데서 옵니다 — 각각이 한 번에 한 요청을 처리합니다. 워커가 `db.execute(...)`에서 블로킹되어 있는 동안, 그 워커에서는 다른 요청이 진전하지 못합니다.

이는 레슨 01 §C.1의 **요청당 스레드** 모델입니다. 처리량은 워커 수에 선형이고, RAM에 의해 제한됩니다. 4-CPU 머신은 Gunicorn 워커 9개를 돌려 동시 요청 9개를 처리할 수 있습니다.

이 단순함의 대가는: 요청 간에 메모리 내 공유 상태가 없고(워커가 별도 프로세스), 모든 블로킹 호출이 그 전체 기간 동안 워커 하나를 묶어둔다는 것입니다. Django가 겨냥하는 전형적인 CRUD/admin 앱에서는 보통 둘 다 괜찮습니다.

#### A.3 비동기 Django (4.x+)

Django 4는 ASGI 지원을 도입했습니다 — 핸들러가 `async def`일 수 있고, 미들웨어가 비동기일 수 있습니다. ORM도 부분적으로 비동기입니다(`aget`, `acreate` 등). 그러나 프레임워크는 *근본적으로* 동기 모양으로 남아 있습니다. `django.contrib.*` 대부분이 여전히 동기이고, 비동기 ORM 작업은 내부적으로 종종 스레드 풀에 넘깁니다.

실용적 입장: 특정 고동시성 엔드포인트(채팅, SSE, 느린 upstream 호출)에서 비동기 Django를 사용하고, 나머지는 동기 워커를 유지하세요. 혼합은 지원되며 흔합니다.

### B. MTV vs MVC: 같은 패턴, 다른 이름

대부분의 웹 프레임워크는 **MVC**(Model-View-Controller)를 사용합니다. Django 문서는 **MTV**(Model-Template-View)라고 부릅니다. 다른 패턴이 아닙니다 — 이름 두 개가 바뀐 같은 아이디어입니다.

#### B.1 매핑

| MVC 용어 | MTV 용어 | 하는 일 |
|----------|----------|--------------|
| Model | Model | 데이터베이스 스키마와 비즈니스 로직 |
| View | Template | 렌더링 계층 (HTML, JSON) |
| Controller | View | 요청을 처리하고 무엇을 렌더링할지 결정하는 함수 |

그래서 Django의 "view"는 다른 프레임워크가 "controller"라고 부르는 것이고, Django의 "template"은 그들이 "view"라고 부르는 것입니다. 이름 충돌 때문에만 혼란스러울 뿐, 데이터 흐름은 동일합니다.

```
HTTP 요청 → URL 라우터 → View (controller) → Model (데이터베이스)
                                    ↓                ↑
                                 Template (렌더링)    ┘
                                    ↓
                              HTTP 응답
```

#### B.2 Django가 이 이름을 고른 이유

Django 저자들의 주장: 어떤 프레임워크에서든 controller(URL → 핸들러 디스패치)는 프레임워크 자신이 소유합니다. 사용자가 작성하는 것은 *view* — 모델 데이터를 응답으로 바꾸는 코드입니다. 그러면 "template"이라는 이름이 자체 관심사인 렌더링 계층(치환 마커가 있는 HTML 파일, view 코드와 분리 가능)을 잘 포착합니다.

이 이름 변경이 명료화인지 세금인지는 배경에 따라 다릅니다. 어느 쪽이든, "view"라고 쓰인 Django 문서를 읽을 때 도움이 된다면 머릿속에서 "controller"로 읽으세요.

#### B.3 실전에서의 MTV 분리

Django의 표준 프로젝트 구조는 MTV를 반영합니다.

```
myapp/
├── models.py       # M — 데이터베이스 스키마, querysets, 비즈니스 불변식
├── views.py        # V (= controller) — URL 라우트의 핸들러
├── templates/      # T — HTML 템플릿, context dict로 렌더링
├── urls.py         # 라우팅, MTV의 일부가 아님
├── forms.py        # 폼 렌더링과 검증, 종종 M과 T를 걸침
└── admin.py        # 모델에서 파생된 자동 생성 admin UI
```

API 전용 Django 앱(DRF, 레슨 12)에서는 템플릿이 JSON serializer로 압축됩니다. 패턴은 그대로 적용되고, 렌더링 계층이 다른 출력 형식을 만들 뿐입니다.

### C. 요청 생명주기

모든 요청은 Django를 거치는 같은 길을 따라갑니다. 단계 순서를 알면 각 hook이 어디에 맞는지, 어디서 어떤 데이터가 사용 가능한지가 설명됩니다.

#### C.1 전체 체인

```
WSGI 서버 (Gunicorn)
    ↓
WSGIHandler — Django의 WSGI 앱
    ↓
미들웨어 (요청 단계, 위 → 아래)
    ↓
URL 해석기 — URL 패턴을 view callable과 매칭
    ↓
View — 작성한 함수/클래스
    ↓
ORM 쿼리 — lazy QuerySet, 결과가 필요할 때 발화
    ↓
Template 렌더 — context dict + template = HTML/JSON
    ↓
HttpResponse 객체
    ↓
미들웨어 (응답 단계, 아래 → 위)
    ↓
WSGI 서버가 클라이언트에 바이트 반환
```

#### C.2 감싸는 계층으로서의 미들웨어

Django 미들웨어는 Express 미들웨어와 구조적으로 비슷하지만, 다른 모양입니다. 각 미들웨어는 `__init__(get_response)`와 `__call__(request)`를 가진 클래스입니다. `__call__` 본문은 FastAPI 미들웨어(레슨 03 §C)와 같은 양파 구조입니다.

```python
class MyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # 뷰 이전
        response = self.get_response(request)  # 다음 미들웨어 / 뷰로 통과
        # 뷰 이후
        return response
```

`MIDDLEWARE` 설정의 순서가 요청 순서이며, 응답 단계는 역순으로 실행됩니다. 표준 순서(위 → 아래): `SecurityMiddleware`, `SessionMiddleware`, `AuthenticationMiddleware`, `CsrfViewMiddleware`, `ClickjackingMiddleware`. 각각이 `request`에 속성(`request.user`, `request.session`)을 추가하거나 응답을 수정합니다.

#### C.3 URL 해석: `urls.py` 트리

Django의 URL 해석기는 `urlpatterns` 리스트의 트리를 따라갑니다. 루트 `urls.py`가 `include()`로 앱 수준 `urls.py`를 포함합니다. 각 패턴은 `path(...)`(문자열 기반) 또는 `re_path(...)`(정규표현식 기반)입니다. 첫 매치가 이깁니다. 매치되는 패턴이 없으면 Django는 `404`를 반환합니다.

경로 변환기(`<int:pk>`, `<slug:name>`, `<uuid:id>`)는 URL 세그먼트를 캡처해 타입 변환과 함께 kwargs로 뷰에 전달합니다.

#### C.4 뷰 계약

뷰는 `(request, *args, **kwargs) -> HttpResponse` 시그니처의 모든 callable입니다. 함수 기반 뷰(FBV)는 평범한 함수입니다. 클래스 기반 뷰(CBV)는 HTTP 메서드 핸들러(`get`, `post`, `put`, ...)를 가진 클래스입니다. Django의 `as_view()` 팩토리가 CBV 클래스를 view callable로 바꾸어, 올바른 메서드 핸들러로 디스패치합니다.

둘 다 `HttpResponse`(또는 `JsonResponse`, `StreamingHttpResponse`, `HttpResponseRedirect` 같은 서브클래스)를 만듭니다. 그 객체가 미들웨어 응답 단계가 감싸고 WSGI 서버가 반환하는 것입니다.

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (Django 철학)은 §A의 WSGI 유산과 §C의 미들웨어 중심 생명주기에서 흘러나오는 설계 선택을 명명합니다.
- §2 (프로젝트 구조)는 §B.3에 매핑되는 디렉터리 레이아웃입니다 — MTV 계층별 파일과 라우팅 파일.
- §3 (MTV 패턴)은 §B.1을 명시적으로 풀이합니다.
- §4 (URL 라우팅)은 §C.3을 구체적인 코드로 구현합니다: `path()`, `include()`, 경로 변환기.
- §5 (뷰)는 §C.4 뷰 계약입니다 — "HttpResponse를 반환하는 callable"의 FBV와 CBV 변종.
- §6 (모델 기초)은 MTV의 M을 도입합니다. 다음 레슨에서 ORM 내부로 들어갑니다.
- §7 (Admin 인터페이스)는 모델에서 파생된 §B 스타일 자동 생성 렌더링 계층입니다.
- §8 (설정과 구성)은 §C.2 미들웨어 리스트와 §A의 WSGI/ASGI 배포를 구성합니다.

---

## 1. Django 철학

Django는 **"배터리 포함(batteries included)"** 원칙을 따릅니다:

| 기능 | Django | FastAPI / Express |
|------|--------|-------------------|
| ORM | 내장 | SQLAlchemy / Prisma |
| 인증(Authentication) | 내장 (`django.contrib.auth`) | 직접 선택 |
| 관리자 패널(Admin panel) | 내장 | 직접 구현 또는 써드파티 설치 |
| 마이그레이션(Migrations) | 내장 (`manage.py migrate`) | Alembic / Prisma Migrate |

핵심 원칙: **DRY**(한 번만 정의), **암묵적 방식보다 명시적 방식**(매직 대신 Python 설정 파일), **설정보다 관례**(합리적인 기본값, 필요 시 재정의).

---

## 2. 프로젝트 구조

```bash
pip install django~=5.1
django-admin startproject mysite && cd mysite

# 프로젝트 레이아웃
mysite/
    manage.py           # CLI 진입점
    mysite/
        settings.py     # 설정
        urls.py         # 루트 URL 라우팅
        asgi.py         # ASGI 진입점
        wsgi.py         # WSGI 진입점
```

특정 기능을 위한 앱 생성:

```bash
python manage.py startapp blog

# blog/
#     admin.py    models.py    views.py
#     apps.py     tests.py     migrations/
```

`settings.py`에 등록:

```python
INSTALLED_APPS = [
    "django.contrib.admin", "django.contrib.auth",
    "django.contrib.contenttypes", "django.contrib.sessions",
    "django.contrib.messages", "django.contrib.staticfiles",
    "blog.apps.BlogConfig",  # 내 앱
]
```

---

## 3. MTV 패턴

Django는 **Model-Template-View**를 사용하며, 전통적인 MVC에 다음과 같이 대응합니다:

| MTV (Django) | MVC | 책임 |
|-------------|-----|------|
| **Model** | Model | 데이터 + 데이터베이스 상호작용 |
| **Template** | View | HTML 표현 |
| **View** | Controller | 요청 처리 + 로직 |

URL 라우터가 요청을 뷰(View)에 전달하고, 뷰는 모델(Model)을 조회하여 템플릿(Template)을 렌더링합니다.

### Django 요청 수명 주기(Request Lifecycle)

```
Browser Request
      │
      ▼
┌──────────────┐
│  URL Router   │  urls.py — pattern matching
│  (urlconf)    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Middleware   │  Auth, CORS, session, CSRF
│  (in order)  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    View      │  Function or class-based
│              │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Model /    │  ORM queries
│   Database   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Template /  │  HTML or JSON serialization
│  Serializer  │
└──────┬───────┘
       │
       ▼
  HTTP Response
```

---

## 4. URL 라우팅

```python
# mysite/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("blog/", include("blog.urls")),
]
```

```python
# blog/urls.py
from django.urls import path
from . import views

app_name = "blog"
urlpatterns = [
    path("", views.post_list, name="post_list"),
    path("<int:pk>/", views.post_detail, name="post_detail"),
    path("category/<slug:slug>/", views.category_detail, name="category_detail"),
]
```

경로 변환기(path converter): `str`, `int`, `slug`, `uuid`, `path`. URL을 하드코딩하는 대신 `reverse("blog:post_detail", kwargs={"pk": 42})`를 사용해 URL을 조회합니다.

---

## 5. 뷰

### 함수 기반 뷰(Function-Based Views)

```python
from django.shortcuts import render, get_object_or_404
from .models import Post

def post_list(request):
    posts = Post.objects.filter(status="published").order_by("-published_at")
    return render(request, "blog/post_list.html", {"posts": posts})

def post_detail(request, pk: int):
    post = get_object_or_404(Post, pk=pk, status="published")
    return render(request, "blog/post_detail.html", {"post": post})
```

### 클래스 기반 뷰(Class-Based Views)

```python
from django.views.generic import ListView, DetailView, CreateView
from django.urls import reverse_lazy

class PostListView(ListView):
    model = Post
    template_name = "blog/post_list.html"
    context_object_name = "posts"
    paginate_by = 10

    def get_queryset(self):
        return Post.objects.filter(status="published").order_by("-published_at")

class PostCreateView(CreateView):
    model = Post
    fields = ["title", "body", "status"]
    success_url = reverse_lazy("blog:post_list")

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
```

단순하거나 커스텀 로직에는 FBV를, 재사용 가능한 제네릭 뷰를 활용하는 표준 CRUD에는 CBV를 사용합니다.

---

## 6. 모델 기초

```python
from django.db import models
from django.conf import settings

class Post(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        PUBLISHED = "published", "Published"

    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique_for_date="published_at")
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    body = models.TextField()
    status = models.CharField(max_length=10, choices=Status, default=Status.DRAFT)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    published_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-published_at"]
        indexes = [models.Index(fields=["-published_at"])]

    def __str__(self) -> str:
        return self.title
```

주요 필드: `CharField`, `TextField`, `IntegerField`, `BooleanField`, `DateTimeField`, `ForeignKey`, `ManyToManyField`, `SlugField`, `JSONField`.

```bash
python manage.py makemigrations blog   # 마이그레이션 생성
python manage.py migrate               # 데이터베이스에 적용
```

---

## 7. Django 관리자 인터페이스

```python
# blog/admin.py
from django.contrib import admin
from .models import Post

@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ["title", "author", "status", "published_at"]
    list_filter = ["status", "created_at", "author"]
    search_fields = ["title", "body"]
    prepopulated_fields = {"slug": ("title",)}
    date_hierarchy = "published_at"
    show_facets = admin.ShowFacets.ALWAYS  # Django 5.x
```

`python manage.py createsuperuser`로 슈퍼유저를 생성한 뒤 `/admin/`에서 접속합니다.

---

## 8. 설정과 구성

`settings.py`의 주요 설정:

```python
SECRET_KEY = "change-in-production"   # 실제 시크릿 키를 커밋하지 마세요
DEBUG = True                          # 프로덕션에서는 False
ALLOWED_HOSTS = ["localhost"]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",  # 프로덕션은 PostgreSQL
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
TIME_ZONE = "UTC"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
```

다중 환경 설정에는 `settings/base.py`, `settings/dev.py`, `settings/prod.py`로 분리하고 `DJANGO_SETTINGS_MODULE`을 적절히 설정합니다.

---

## 9. 연습 문제

### 문제 1: 프로젝트 설정

`catalog` 앱이 포함된 `bookstore` 프로젝트를 생성하세요. `title`, `author`, `isbn`(고유), `price`(DecimalField), `published_date` 필드를 가진 `Book` 모델을 정의하고, 검색과 필터링이 가능하도록 관리자(admin)에 등록하세요.

### 문제 2: URL 라우팅

`recipes` 앱의 URL을 설계하세요: 목록(`/recipes/`), 상세(`/recipes/<id>/`), 카테고리별(`/recipes/category/<slug>/`), 검색(`/recipes/search/?q=`). 네임스페이스(namespacing)를 사용하세요.

### 문제 3: FBV와 CBV 비교

`post_detail`을 FBV 방식과 CBV 방식으로 모두 구현하세요. 미발행 게시물에는 404를 반환하고, 댓글을 템플릿에 전달하며, `view_count` 필드를 증가시켜야 합니다.

### 문제 4: 모델 설계

전자상거래를 위한 `Product`, `Order`, `OrderItem` 모델을 설계하세요. 적절한 관계(relationships), 주문 상태(order status) 선택지(choices), Meta 옵션, 인덱스(indexes)를 포함하세요.

---

## 참고 자료

- [Django 공식 문서 (5.x)](https://docs.djangoproject.com/en/5.1/)
- [Django 설계 철학](https://docs.djangoproject.com/en/5.1/misc/design-philosophies/)
- [Two Scoops of Django](https://www.feldroy.com/books/two-scoops-of-django-5-0) by Feldroy
- [Classy Class-Based Views](https://ccbv.co.uk/)

---

**이전**: [Express 테스트](./09_Express_Testing.md) | **다음**: [Django ORM](./11_Django_ORM.md)
