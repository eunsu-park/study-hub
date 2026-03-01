"""
Django REST Framework — Serializers, ViewSets, Router
Demonstrates: DRF API patterns for building RESTful APIs.

Setup:
    pip install djangorestframework
    # Add 'rest_framework' to INSTALLED_APPS
"""

# === serializers.py ===

"""
from rest_framework import serializers
from .models import Post, Category


class CategorySerializer(serializers.ModelSerializer):
    post_count = serializers.IntegerField(read_only=True)

    class Meta:
        model = Category
        fields = ["id", "name", "slug", "post_count"]


class PostListSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField()
    category_name = serializers.CharField(source="category.name", read_only=True)

    class Meta:
        model = Post
        fields = ["id", "title", "slug", "author", "category_name", "created_at"]


class PostDetailSerializer(serializers.ModelSerializer):
    author = serializers.StringRelatedField()
    category = CategorySerializer(read_only=True)
    category_id = serializers.PrimaryKeyRelatedField(
        queryset=Category.objects.all(),
        source="category",
        write_only=True,
    )

    class Meta:
        model = Post
        fields = [
            "id", "title", "slug", "content", "author",
            "category", "category_id", "status", "created_at", "updated_at",
        ]
        read_only_fields = ["author"]

    def validate_title(self, value):
        if len(value) < 5:
            raise serializers.ValidationError("Title must be at least 5 characters.")
        return value
"""


# === views.py (ViewSet) ===

"""
from rest_framework import viewsets, permissions, filters, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django_filters.rest_framework import DjangoFilterBackend
from django.db.models import Count
from .models import Post, Category
from .serializers import (
    PostListSerializer, PostDetailSerializer, CategorySerializer,
)


class PostViewSet(viewsets.ModelViewSet):
    \"\"\"
    CRUD API for blog posts.

    list:   GET    /api/posts/
    create: POST   /api/posts/
    read:   GET    /api/posts/{slug}/
    update: PUT    /api/posts/{slug}/
    patch:  PATCH  /api/posts/{slug}/
    delete: DELETE /api/posts/{slug}/
    \"\"\"
    queryset = Post.objects.select_related("author", "category")
    lookup_field = "slug"
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    filter_backends = [DjangoFilterBackend, filters.SearchFilter, filters.OrderingFilter]
    filterset_fields = ["status", "category__slug"]
    search_fields = ["title", "content"]
    ordering_fields = ["created_at", "title"]
    ordering = ["-created_at"]

    def get_serializer_class(self):
        if self.action == "list":
            return PostListSerializer
        return PostDetailSerializer

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

    @action(detail=False, methods=["get"])
    def published(self, request):
        \"\"\"GET /api/posts/published/ — only published posts\"\"\"
        posts = self.get_queryset().filter(status="published")
        page = self.paginate_queryset(posts)
        serializer = PostListSerializer(page, many=True)
        return self.get_paginated_response(serializer.data)

    @action(detail=True, methods=["post"])
    def publish(self, request, slug=None):
        \"\"\"POST /api/posts/{slug}/publish/ — publish a draft\"\"\"
        post = self.get_object()
        if post.author != request.user:
            return Response(
                {"error": "Only the author can publish"},
                status=status.HTTP_403_FORBIDDEN,
            )
        post.status = "published"
        post.save()
        return Response(PostDetailSerializer(post).data)


class CategoryViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Category.objects.annotate(post_count=Count("posts"))
    serializer_class = CategorySerializer
    lookup_field = "slug"
"""


# === urls.py ===

"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from . import views

router = DefaultRouter()
router.register("posts", views.PostViewSet)
router.register("categories", views.CategoryViewSet)

urlpatterns = [
    path("api/", include(router.urls)),
]
"""


# === settings.py (DRF configuration) ===

"""
REST_FRAMEWORK = {
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.TokenAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
    "DEFAULT_THROTTLE_CLASSES": [
        "rest_framework.throttling.AnonRateThrottle",
        "rest_framework.throttling.UserRateThrottle",
    ],
    "DEFAULT_THROTTLE_RATES": {
        "anon": "100/hour",
        "user": "1000/hour",
    },
}
"""


# --- Standalone Demo ---

if __name__ == "__main__":
    print("Django REST Framework API Patterns")
    print("=" * 50)
    print()
    print("Key DRF patterns demonstrated:")
    print("  1. ModelSerializer with nested, related, and custom fields")
    print("  2. ModelViewSet with full CRUD + custom actions")
    print("  3. DefaultRouter for automatic URL generation")
    print("  4. Filtering, searching, and ordering backends")
    print("  5. Permission and throttle configuration")
    print()
    print("ViewSet generates these endpoints:")
    print("  GET    /api/posts/           → list")
    print("  POST   /api/posts/           → create")
    print("  GET    /api/posts/{slug}/    → retrieve")
    print("  PUT    /api/posts/{slug}/    → update")
    print("  PATCH  /api/posts/{slug}/    → partial_update")
    print("  DELETE /api/posts/{slug}/    → destroy")
    print("  GET    /api/posts/published/ → custom action")
    print("  POST   /api/posts/{slug}/publish/ → custom action")
