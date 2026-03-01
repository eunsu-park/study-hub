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
