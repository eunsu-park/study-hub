"""
Django Basics — Models, Views, and URL Configuration
Demonstrates: Model definition, function-based views, class-based views.

This is a standalone demonstration file — in a real Django project,
these would be split across models.py, views.py, urls.py, serializers.py.

Setup:
    pip install django
    django-admin startproject myproject
    python manage.py startapp blog
"""

# === models.py ===

"""
from django.db import models
from django.contrib.auth.models import User


class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(unique=True)

    class Meta:
        verbose_name_plural = "categories"
        ordering = ["name"]

    def __str__(self):
        return self.name


class Post(models.Model):
    class Status(models.TextChoices):
        DRAFT = "draft", "Draft"
        PUBLISHED = "published", "Published"

    title = models.CharField(max_length=200)
    slug = models.SlugField(unique=True)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name="posts")
    category = models.ForeignKey(
        Category, on_delete=models.SET_NULL, null=True, related_name="posts"
    )
    status = models.CharField(
        max_length=10, choices=Status.choices, default=Status.DRAFT
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["-created_at"]),
            models.Index(fields=["slug"]),
        ]

    def __str__(self):
        return self.title
"""


# === views.py (Function-Based Views) ===

"""
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from .models import Post


@require_http_methods(["GET"])
def post_list(request):
    posts = Post.objects.filter(status=Post.Status.PUBLISHED)

    # Pagination
    page = int(request.GET.get("page", 1))
    per_page = int(request.GET.get("per_page", 10))
    offset = (page - 1) * per_page

    data = list(
        posts[offset : offset + per_page].values(
            "id", "title", "slug", "created_at", "author__username"
        )
    )
    return JsonResponse({"data": data, "page": page})


@require_http_methods(["GET"])
def post_detail(request, slug):
    post = get_object_or_404(Post, slug=slug, status=Post.Status.PUBLISHED)
    return JsonResponse({
        "id": post.id,
        "title": post.title,
        "content": post.content,
        "author": post.author.username,
        "created_at": post.created_at.isoformat(),
    })
"""


# === views.py (Class-Based Views) ===

"""
from django.views.generic import ListView, DetailView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Post


class PostListView(ListView):
    model = Post
    template_name = "blog/post_list.html"
    context_object_name = "posts"
    paginate_by = 10

    def get_queryset(self):
        return Post.objects.filter(status=Post.Status.PUBLISHED)


class PostDetailView(DetailView):
    model = Post
    template_name = "blog/post_detail.html"
    slug_url_kwarg = "slug"

    def get_queryset(self):
        return Post.objects.filter(status=Post.Status.PUBLISHED)


class PostCreateView(LoginRequiredMixin, CreateView):
    model = Post
    fields = ["title", "slug", "content", "category"]
    template_name = "blog/post_form.html"

    def form_valid(self, form):
        form.instance.author = self.request.user
        return super().form_valid(form)
"""


# === urls.py ===

"""
from django.urls import path
from . import views

app_name = "blog"

urlpatterns = [
    path("", views.PostListView.as_view(), name="post_list"),
    path("<slug:slug>/", views.PostDetailView.as_view(), name="post_detail"),
    path("create/", views.PostCreateView.as_view(), name="post_create"),

    # API endpoints (function-based)
    path("api/posts/", views.post_list, name="api_post_list"),
    path("api/posts/<slug:slug>/", views.post_detail, name="api_post_detail"),
]
"""


# === admin.py ===

"""
from django.contrib import admin
from .models import Post, Category


@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ["name", "slug"]
    prepopulated_fields = {"slug": ("name",)}


@admin.register(Post)
class PostAdmin(admin.ModelAdmin):
    list_display = ["title", "author", "status", "created_at"]
    list_filter = ["status", "category", "created_at"]
    search_fields = ["title", "content"]
    prepopulated_fields = {"slug": ("title",)}
    date_hierarchy = "created_at"
"""


# --- Standalone Demo ---

if __name__ == "__main__":
    print("Django Models, Views, and URLs Demo")
    print("=" * 50)
    print()
    print("This file contains Django code organized as:")
    print("  1. models.py  — Post, Category models with Meta options")
    print("  2. views.py   — FBV (function-based) + CBV (class-based)")
    print("  3. urls.py    — URL routing with app_name namespace")
    print("  4. admin.py   — Admin customization")
    print()
    print("To run in a real Django project:")
    print("  django-admin startproject myproject")
    print("  python manage.py startapp blog")
    print("  # Copy the code sections into their respective files")
    print("  python manage.py makemigrations && python manage.py migrate")
    print("  python manage.py runserver")
