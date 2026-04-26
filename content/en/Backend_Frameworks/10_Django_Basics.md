# 10. Django Basics

**Previous**: [Express Testing](./09_Express_Testing.md) | **Next**: [Django ORM](./11_Django_ORM.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain Django's "batteries included" philosophy and how it contrasts with micro-frameworks
2. Create a Django project and app, and describe the purpose of each generated file
3. Describe the MTV (Model-Template-View) pattern and map it to traditional MVC
4. Configure URL routing with `path()`, `include()`, and path converters
5. Implement both function-based views and class-based views for common HTTP operations

---

Django is a high-level Python web framework that encourages rapid development and clean, pragmatic design. Where FastAPI gives you a minimal core, Django ships with an ORM, authentication, admin interface, form handling, and more -- all working together out of the box. Django 5.x brings generated fields, facet filters in the admin, and simplified form rendering.

## Table of Contents

Before the framework reference, read [**Theory & Principles**](#theory--principles) — the WSGI interface that runs Django, the MTV pattern compared to MVC, and the request lifecycle as it travels through middleware, URL resolution, view, and response.

1. [Django Philosophy](#1-django-philosophy)
2. [Project Structure](#2-project-structure)
3. [MTV Pattern](#3-mtv-pattern)
4. [URL Routing](#4-url-routing)
5. [Views](#5-views)
6. [Models Basics](#6-models-basics)
7. [Django Admin Interface](#7-django-admin-interface)
8. [Settings and Configuration](#8-settings-and-configuration)
9. [Practice Problems](#9-practice-problems)

---

## Theory & Principles

Django is older than FastAPI by 15 years and predates async Python entirely. Its design choices — synchronous WSGI, the unique MTV naming, the convention-heavy project structure — only make sense once you know what they reacted against and what they enabled. Three concepts cover almost every later decision.

- **(A) WSGI: the synchronous server interface Django was built on** — and how Django 4+ added ASGI without breaking the model.
- **(B) MTV vs MVC** — the same idea with different names, and why Django picked these.
- **(C) The request lifecycle** — middleware → URL resolver → view → response, with hooks at every stage.

### A. WSGI: The Synchronous Server Interface

WSGI (Web Server Gateway Interface, PEP 3333) is the contract that lets a Python web app talk to a server (Gunicorn, uWSGI, mod_wsgi). It is *the* Python web standard from 2003 to ~2018, and Django was built squarely on it.

#### A.1 The WSGI callable

A WSGI app is one synchronous callable:

```python
def application(environ, start_response):
    start_response("200 OK", [("Content-Type", "text/plain")])
    return [b"Hello"]
```

- `environ` is a dict containing the request: method, path, headers, the body as a file-like object.
- `start_response` is a callback the app calls *exactly once* with status and headers, before yielding bytes.
- The return value is an iterable of bytes — the response body.

That is it. No event loop, no streaming primitives, no WebSocket. The simplicity is what made WSGI ubiquitous; it is also what made the Python web ecosystem need ASGI for the next decade.

#### A.2 What this implies for Django's runtime model

Django's request handler is a WSGI callable. Each request runs to completion on one Gunicorn worker process. Concurrency comes from running N worker processes (typically `2 × CPU + 1`), each handling one request at a time. While a worker is blocked on `db.execute(...)`, no other request makes progress on that worker.

This is the **thread-per-request** model from Lesson 01 §C.1. Throughput is linear in worker count, bounded by RAM. A 4-CPU box might run 9 Gunicorn workers, handling 9 simultaneous requests.

The trade for this simplicity is: no shared in-memory state across requests (workers are separate processes), and any blocking call ties up a whole worker for its full duration. Both are usually fine for the typical CRUD/admin app Django targets.

#### A.3 Async Django (4.x+)

Django 4 introduced ASGI support — handlers can be `async def`, and middleware can be async. The ORM is partially async (with `aget`, `acreate`, etc.). But the framework remains *fundamentally* sync-shaped: most of `django.contrib.*` is still synchronous, and async ORM operations under the hood often hand off to a thread pool.

The pragmatic position: use async Django when you have specific high-concurrency endpoints (chat, SSE, slow upstream calls); stick with sync workers for everything else. Mixing is supported and common.

### B. MTV vs MVC: Same Pattern, Different Names

Most web frameworks use **MVC** (Model-View-Controller). Django's docs call it **MTV** (Model-Template-View). It is not a different pattern — it is the same idea with two names swapped.

#### B.1 The mapping

| MVC term | MTV term | What it does |
|----------|----------|--------------|
| Model | Model | Database schema and business logic |
| View | Template | Rendering layer (HTML, JSON) |
| Controller | View | The function that handles a request and decides what to render |

So the Django "view" is what other frameworks call the "controller", and the Django "template" is what they call the "view". Confusing only because of the name collision; the data flow is identical:

```
HTTP request → URL router → View (controller) → Model (database)
                                    ↓                ↑
                                 Template (renders)  ┘
                                    ↓
                              HTTP response
```

#### B.2 Why Django picked these names

The Django authors argue: in any framework, the framework itself owns the controller (the URL → handler dispatch). What you write is the *view* — the code that turns model data into a response. The "template" name then captures the rendering layer, which is its own concern (HTML files with substitution markers, separable from the view code).

Whether the rename is a clarification or a tax depends on your background. Either way, when you read Django docs that say "view", read "controller in your head" if it helps.

#### B.3 The MTV separation in practice

Django's standard project structure mirrors MTV:

```
myapp/
├── models.py       # M — database schema, querysets, business invariants
├── views.py        # V (= controller) — handlers for URL routes
├── templates/      # T — HTML templates, rendered with context dicts
├── urls.py         # routing, not part of MTV
├── forms.py        # form rendering and validation, often spans M and T
└── admin.py        # auto-generated admin UI, derived from models
```

For an API-only Django app (DRF, Lesson 12), templates collapse to JSON serializers. The pattern still applies; the rendering layer just produces a different output format.

### C. The Request Lifecycle

Every request traverses the same path through Django. Knowing the order of stages explains where each hook fits and what data is available where.

#### C.1 The full chain

```
WSGI server (Gunicorn)
    ↓
WSGIHandler — Django's WSGI app
    ↓
Middleware (request phase, top → bottom)
    ↓
URL resolver — match URL pattern to view callable
    ↓
View — the function/class you wrote
    ↓
ORM queries — lazy QuerySets, fired when the result is needed
    ↓
Template render — context dict + template = HTML/JSON
    ↓
HttpResponse object
    ↓
Middleware (response phase, bottom → top)
    ↓
WSGI server returns bytes to client
```

#### C.2 Middleware as the wrapping layer

Django middleware is structurally similar to Express middleware, but with a different shape. Each middleware is a class with `__init__(get_response)` and `__call__(request)`. The `__call__` body has the same onion structure as FastAPI's middleware (Lesson 03 §C):

```python
class MyMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # before view
        response = self.get_response(request)  # passes through to next middleware / view
        # after view
        return response
```

The order in `MIDDLEWARE` setting is the request order; response phase runs in reverse. Standard order (top → bottom): `SecurityMiddleware`, `SessionMiddleware`, `AuthenticationMiddleware`, `CsrfViewMiddleware`, `ClickjackingMiddleware`. Each adds attributes to `request` (`request.user`, `request.session`) or modifies the response.

#### C.3 URL resolution: the `urls.py` tree

Django's URL resolver walks a tree of `urlpatterns` lists. The root `urls.py` includes app-level `urls.py`s with `include()`. Each pattern is a `path(...)` (string-based) or `re_path(...)` (regex-based). The first match wins; if no pattern matches, Django returns `404`.

Path converters (`<int:pk>`, `<slug:name>`, `<uuid:id>`) capture URL segments and pass them as kwargs to the view, with type conversion built in.

#### C.4 The view contract

A view is any callable with signature `(request, *args, **kwargs) -> HttpResponse`. Function-based views (FBVs) are plain functions. Class-based views (CBVs) are classes with HTTP method handlers (`get`, `post`, `put`, ...). Django's `as_view()` factory turns a CBV class into a view callable, dispatching to the right method handler.

Both produce an `HttpResponse` (or subclass like `JsonResponse`, `StreamingHttpResponse`, `HttpResponseRedirect`). That object is what middleware's response phase wraps and the WSGI server returns.

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (Django philosophy) names the design choices that flow from §A's WSGI heritage and §C's middleware-heavy lifecycle.
- §2 (Project structure) is the directory layout that maps to §B.3 — files for each MTV layer plus the routing files.
- §3 (MTV pattern) is the §B.1 explicit walkthrough.
- §4 (URL routing) is §C.3 in concrete code: `path()`, `include()`, path converters.
- §5 (Views) is the §C.4 view contract — both FBV and CBV variants of "callable returning `HttpResponse`".
- §6 (Models basics) introduces the M of MTV; the next lesson dives into the ORM internals.
- §7 (Admin interface) is a §B-style auto-generated rendering layer derived from models.
- §8 (Settings and configuration) configures the §C.2 middleware list and the WSGI/ASGI deployment from §A.

---

## 1. Django Philosophy

Django follows the **"batteries included"** principle:

| Feature | Django | FastAPI / Express |
|---------|--------|-------------------|
| ORM | Built-in | SQLAlchemy / Prisma |
| Authentication | Built-in (`django.contrib.auth`) | Choose your own |
| Admin panel | Built-in | Build or install third-party |
| Migrations | Built-in (`manage.py migrate`) | Alembic / Prisma Migrate |

Core principles: **DRY** (define things once), **explicit over implicit** (Python config files, not magic), **convention over configuration** (sensible defaults, override as needed).

---

## 2. Project Structure

```bash
pip install django~=5.1
django-admin startproject mysite && cd mysite

# Project layout
mysite/
    manage.py           # CLI entry point
    mysite/
        settings.py     # Configuration
        urls.py         # Root URL routing
        asgi.py         # ASGI entry point
        wsgi.py         # WSGI entry point
```

Create an app for a specific feature:

```bash
python manage.py startapp blog

# blog/
#     admin.py    models.py    views.py
#     apps.py     tests.py     migrations/
```

Register it in `settings.py`:

```python
INSTALLED_APPS = [
    "django.contrib.admin", "django.contrib.auth",
    "django.contrib.contenttypes", "django.contrib.sessions",
    "django.contrib.messages", "django.contrib.staticfiles",
    "blog.apps.BlogConfig",  # Your app
]
```

---

## 3. MTV Pattern

Django uses **Model-Template-View**, which maps to traditional MVC:

| MTV (Django) | MVC | Responsibility |
|-------------|-----|----------------|
| **Model** | Model | Data + database interaction |
| **Template** | View | HTML presentation |
| **View** | Controller | Request handling + logic |

The URL router dispatches requests to Views, which query Models and render Templates.

### Django Request Lifecycle

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

## 4. URL Routing

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

Path converters: `str`, `int`, `slug`, `uuid`, `path`. Use `reverse("blog:post_detail", kwargs={"pk": 42})` for URL lookups instead of hardcoding.

---

## 5. Views

### Function-Based Views

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

### Class-Based Views

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

Use FBVs for simple/custom logic; CBVs for standard CRUD with reusable generic views.

---

## 6. Models Basics

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

Common fields: `CharField`, `TextField`, `IntegerField`, `BooleanField`, `DateTimeField`, `ForeignKey`, `ManyToManyField`, `SlugField`, `JSONField`.

```bash
python manage.py makemigrations blog   # Generate migration
python manage.py migrate               # Apply to database
```

---

## 7. Django Admin Interface

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

Create a superuser with `python manage.py createsuperuser`, then visit `/admin/`.

---

## 8. Settings and Configuration

Key settings in `settings.py`:

```python
SECRET_KEY = "change-in-production"   # Never commit real secrets
DEBUG = True                          # False in production
ALLOWED_HOSTS = ["localhost"]
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",  # PostgreSQL for production
        "NAME": BASE_DIR / "db.sqlite3",
    }
}
TIME_ZONE = "UTC"
USE_TZ = True
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
```

For multi-environment setups, split into `settings/base.py`, `settings/dev.py`, `settings/prod.py` and set `DJANGO_SETTINGS_MODULE` accordingly.

---

## 9. Practice Problems

### Problem 1: Project Setup

Create a `bookstore` project with a `catalog` app. Define a `Book` model with `title`, `author`, `isbn` (unique), `price` (DecimalField), and `published_date`. Register it in the admin with search and filtering.

### Problem 2: URL Routing

Design URLs for a `recipes` app: list (`/recipes/`), detail (`/recipes/<id>/`), by category (`/recipes/category/<slug>/`), and search (`/recipes/search/?q=`). Use namespacing.

### Problem 3: FBV vs CBV

Implement `post_detail` in both FBV and CBV style. It should return 404 for unpublished posts, pass comments to the template, and increment a `view_count` field.

### Problem 4: Model Design

Design `Product`, `Order`, and `OrderItem` models for e-commerce. Include proper relationships, choices for order status, Meta options, and indexes.

---

## References

- [Django Documentation (5.x)](https://docs.djangoproject.com/en/5.1/)
- [Django Design Philosophies](https://docs.djangoproject.com/en/5.1/misc/design-philosophies/)
- [Two Scoops of Django](https://www.feldroy.com/books/two-scoops-of-django-5-0) by Feldroy
- [Classy Class-Based Views](https://ccbv.co.uk/)

---

**Previous**: [Express Testing](./09_Express_Testing.md) | **Next**: [Django ORM](./11_Django_ORM.md)
