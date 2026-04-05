# 12. Django REST Framework

**Previous**: [Django ORM](./11_Django_ORM.md) | **Next**: [Django Advanced](./13_Django_Advanced.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Install and configure DRF, and explain how it extends Django's request/response cycle
2. Build serializers with custom validation, nested representations, and computed fields
3. Implement API endpoints using APIView, generic views, and ViewSets with routers
4. Configure authentication (Token, JWT) and permission classes to secure endpoints
5. Apply pagination, filtering, and search to produce production-quality list endpoints

---

Django REST Framework (DRF) is the standard for building RESTful APIs with Django. It adds serialization, content negotiation, authentication, permissions, throttling, and a browsable API. Like Django itself, most features work with minimal configuration yet remain deeply customizable.

## Table of Contents

1. [DRF Setup](#1-drf-setup)
2. [Serializers](#2-serializers)
3. [Views: APIView to ViewSets](#3-views-apiview-to-viewsets)
4. [Routers](#4-routers)
5. [Authentication](#5-authentication)
6. [Permissions](#6-permissions)
7. [Pagination](#7-pagination)
8. [Filtering and Search](#8-filtering-and-search)
9. [Practice Problems](#9-practice-problems)

---

## 1. DRF Setup

```bash
pip install djangorestframework~=3.15 django-filter~=24.3 djangorestframework-simplejwt~=5.3
```

```python
# settings.py
INSTALLED_APPS = [... "rest_framework", "django_filters"]

REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": [
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ],
    "DEFAULT_PERMISSION_CLASSES": [
        "rest_framework.permissions.IsAuthenticatedOrReadOnly",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 20,
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
}
```

DRF provides a browsable HTML API out of the box -- visit any endpoint in a browser.

---

## 2. Serializers

Serializers convert model instances to JSON and validate incoming data (like Pydantic in FastAPI).

### ModelSerializer

```python
from rest_framework import serializers
from .models import Post, Category

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name", "slug"]

class PostSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)              # Nested read
    category_id = serializers.PrimaryKeyRelatedField(          # ID for write
        queryset=Category.objects.all(), source="category", write_only=True
    )
    author_name = serializers.CharField(source="author.get_full_name", read_only=True)
    comment_count = serializers.SerializerMethodField()

    class Meta:
        model = Post
        fields = ["id", "title", "slug", "body", "author", "author_name",
                  "category", "category_id", "status", "comment_count",
                  "published_at", "created_at"]
        read_only_fields = ["author", "slug", "created_at"]

    def get_comment_count(self, obj) -> int:
        return obj.comments.count()
```

### Custom Validation

```python
class PostSerializer(serializers.ModelSerializer):
    # ...
    def validate_title(self, value: str) -> str:
        if len(value) < 5:
            raise serializers.ValidationError("Title must be at least 5 characters.")
        return value

    def validate(self, attrs: dict) -> dict:
        if attrs.get("status") == "published" and not attrs.get("body"):
            raise serializers.ValidationError({"body": "Published posts need content."})
        return attrs

    def create(self, validated_data: dict) -> Post:
        validated_data["author"] = self.context["request"].user
        return super().create(validated_data)
```

---

## 3. Views: APIView to ViewSets

DRF provides a spectrum from low-level to high-level:

### APIView (Full Control)

```python
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class PostListAPIView(APIView):
    def get(self, request):
        posts = Post.objects.filter(status="published")
        serializer = PostSerializer(posts, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = PostSerializer(data=request.data, context={"request": request})
        serializer.is_valid(raise_exception=True)
        serializer.save()
        return Response(serializer.data, status=status.HTTP_201_CREATED)
```

### Generic Views (Less Boilerplate)

```python
from rest_framework import generics

class PostListCreateView(generics.ListCreateAPIView):
    queryset = Post.objects.filter(status="published")
    serializer_class = PostSerializer

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

class PostDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Post.objects.filter(status="published")
    serializer_class = PostSerializer
```

### ViewSets (Maximum DRY)

```python
from rest_framework import viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.select_related("author", "category")
    serializer_class = PostSerializer

    def get_queryset(self):
        qs = super().get_queryset()
        return qs.filter(status="published") if self.action == "list" else qs

    def perform_create(self, serializer):
        serializer.save(author=self.request.user)

    @action(detail=True, methods=["post"])
    def publish(self, request, pk=None):
        post = self.get_object()
        post.status = "published"
        post.save(update_fields=["status"])
        return Response({"status": "published"})
```

---

## 4. Routers

Routers generate URL patterns automatically for ViewSets:

```python
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r"posts", PostViewSet, basename="post")

urlpatterns = [path("api/", include(router.urls))]
```

This generates: `GET/POST /api/posts/`, `GET/PUT/PATCH/DELETE /api/posts/{pk}/`, plus custom actions like `POST /api/posts/{pk}/publish/`.

---

## 5. Authentication

### JWT (SimpleJWT)

```python
# settings.py
from datetime import timedelta
SIMPLE_JWT = {
    "ACCESS_TOKEN_LIFETIME": timedelta(minutes=30),
    "REFRESH_TOKEN_LIFETIME": timedelta(days=7),
    "ROTATE_REFRESH_TOKENS": True,
}

# urls.py
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView
urlpatterns = [
    path("api/token/", TokenObtainPairView.as_view()),
    path("api/token/refresh/", TokenRefreshView.as_view()),
]
```

```bash
# Obtain tokens
curl -X POST /api/token/ -d '{"username":"alice","password":"secret"}'
# {"access": "eyJ...", "refresh": "eyJ..."}

# Use access token
curl /api/posts/ -H "Authorization: Bearer eyJ..."
```

### Token Authentication

Add `rest_framework.authtoken` to `INSTALLED_APPS`, run `migrate`, then use `Authorization: Token <key>` header.

---

## 6. Permissions

```python
from rest_framework import permissions

class IsAuthorOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.author == request.user

class PostViewSet(viewsets.ModelViewSet):
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsAuthorOrReadOnly]

    def get_permissions(self):
        if self.action == "destroy":
            return [permissions.IsAdminUser()]
        return super().get_permissions()
```

Built-in: `AllowAny`, `IsAuthenticated`, `IsAdminUser`, `IsAuthenticatedOrReadOnly`.

---

## 7. Pagination

Three styles:

```python
from rest_framework.pagination import PageNumberPagination, CursorPagination

class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100

class TimelinePagination(CursorPagination):
    page_size = 20
    ordering = "-published_at"
```

**PageNumber**: Simple, allows jumping to pages. **Cursor**: Consistent ordering, no duplicate/missing items during inserts. **LimitOffset**: SQL-style, good for arbitrary offsets.

---

## 8. Filtering and Search

### django-filter

```python
import django_filters
from .models import Post

class PostFilter(django_filters.FilterSet):
    title = django_filters.CharFilter(lookup_expr="icontains")
    published_after = django_filters.DateTimeFilter(field_name="published_at", lookup_expr="gte")
    category = django_filters.CharFilter(field_name="category__slug")

    class Meta:
        model = Post
        fields = ["status", "author"]

class PostViewSet(viewsets.ModelViewSet):
    filterset_class = PostFilter
    search_fields = ["title", "body", "author__username"]
    ordering_fields = ["published_at", "view_count"]
```

```
GET /api/posts/?status=published&category=python&search=orm&ordering=-view_count
```

---

## 9. Practice Problems

### Problem 1: Library API

Build a DRF API for `Book`, `Author`, `Borrow` models. Include nested serializers, filtering by genre/availability, a custom `POST /api/books/{pk}/borrow/` action, and role-based permissions (only staff can add books).

### Problem 2: Serializer Validation

Create an `OrderSerializer` that validates: items list is non-empty, quantities are positive, total does not exceed user's credit limit, and shipping address is required when delivery method is "shipping".

### Problem 3: JWT Auth Flow

Implement registration, login (JWT tokens), token refresh, protected profile endpoint, and password change requiring old password.

### Problem 4: Advanced Filtering

Build product search with: text search across name/description, price range, multiple category slugs, minimum average rating, and ordering by price/rating/relevance.

---

## References

- [DRF Documentation](https://www.django-rest-framework.org/)
- [DRF Serializers Guide](https://www.django-rest-framework.org/api-guide/serializers/)
- [SimpleJWT Documentation](https://django-rest-framework-simplejwt.readthedocs.io/)
- [django-filter Documentation](https://django-filter.readthedocs.io/)
- [Classy DRF](https://www.cdrf.co/)

---

**Previous**: [Django ORM](./11_Django_ORM.md) | **Next**: [Django Advanced](./13_Django_Advanced.md)
