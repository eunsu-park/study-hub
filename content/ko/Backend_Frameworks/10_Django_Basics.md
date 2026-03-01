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
