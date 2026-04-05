# 12. Django REST Framework

**이전**: [Django ORM](./11_Django_ORM.md) | **다음**: [Django 고급](./13_Django_Advanced.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. DRF를 설치하고 설정하며, Django의 요청/응답 사이클(request/response cycle)을 어떻게 확장하는지 설명한다
2. 커스텀 유효성 검사(validation), 중첩 표현(nested representation), 계산 필드(computed field)를 포함한 직렬화기(serializer)를 구축한다
3. APIView, 제네릭 뷰(generic view), 라우터(router)를 사용하는 ViewSet으로 API 엔드포인트를 구현한다
4. 엔드포인트 보안을 위해 인증(authentication)(Token, JWT)과 권한(permission) 클래스를 설정한다
5. 프로덕션 수준의 목록 엔드포인트를 위해 페이지네이션(pagination), 필터링(filtering), 검색(search)을 적용한다

---

Django REST Framework(DRF)는 Django로 RESTful API를 구축하는 표준 라이브러리입니다. 직렬화(serialization), 콘텐츠 협상(content negotiation), 인증(authentication), 권한(permissions), 쓰로틀링(throttling), 브라우저블 API(browsable API)를 추가합니다. Django 자체와 마찬가지로, 대부분의 기능은 최소한의 설정으로 동작하면서도 깊은 수준의 커스터마이징이 가능합니다.

## 목차

1. [DRF 설정](#1-drf-설정)
2. [직렬화기](#2-직렬화기)
3. [뷰: APIView에서 ViewSet까지](#3-뷰-apiview에서-viewset까지)
4. [라우터](#4-라우터)
5. [인증](#5-인증)
6. [권한](#6-권한)
7. [페이지네이션](#7-페이지네이션)
8. [필터링과 검색](#8-필터링과-검색)
9. [연습 문제](#9-연습-문제)

---

## 1. DRF 설정

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

DRF는 브라우저에서 바로 사용할 수 있는 HTML 브라우저블 API(browsable API)를 기본으로 제공합니다.

---

## 2. 직렬화기

직렬화기(Serializer)는 모델 인스턴스를 JSON으로 변환하고 들어오는 데이터를 유효성 검사합니다(FastAPI의 Pydantic과 유사).

### ModelSerializer

```python
from rest_framework import serializers
from .models import Post, Category

class CategorySerializer(serializers.ModelSerializer):
    class Meta:
        model = Category
        fields = ["id", "name", "slug"]

class PostSerializer(serializers.ModelSerializer):
    category = CategorySerializer(read_only=True)              # 중첩 읽기
    category_id = serializers.PrimaryKeyRelatedField(          # 쓰기용 ID
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

### 커스텀 유효성 검사

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

## 3. 뷰: APIView에서 ViewSet까지

DRF는 저수준부터 고수준까지 다양한 스펙트럼을 제공합니다:

### APIView (완전한 제어)

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

### 제네릭 뷰 (보일러플레이트 최소화)

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

### ViewSet (최대 DRY)

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

## 4. 라우터

라우터(router)는 ViewSet에 대한 URL 패턴을 자동으로 생성합니다:

```python
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r"posts", PostViewSet, basename="post")

urlpatterns = [path("api/", include(router.urls))]
```

이렇게 하면 `GET/POST /api/posts/`, `GET/PUT/PATCH/DELETE /api/posts/{pk}/`, 그리고 `POST /api/posts/{pk}/publish/`와 같은 커스텀 액션(action)이 생성됩니다.

---

## 5. 인증

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
# 토큰 발급
curl -X POST /api/token/ -d '{"username":"alice","password":"secret"}'
# {"access": "eyJ...", "refresh": "eyJ..."}

# 액세스 토큰 사용
curl /api/posts/ -H "Authorization: Bearer eyJ..."
```

### 토큰 인증(Token Authentication)

`INSTALLED_APPS`에 `rest_framework.authtoken`을 추가하고 `migrate`를 실행한 뒤, `Authorization: Token <key>` 헤더를 사용합니다.

---

## 6. 권한

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

내장 권한: `AllowAny`, `IsAuthenticated`, `IsAdminUser`, `IsAuthenticatedOrReadOnly`.

---

## 7. 페이지네이션

세 가지 방식:

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

**PageNumber**: 단순하며 임의 페이지로 이동 가능. **Cursor**: 삽입 중에도 일관된 순서 유지, 중복/누락 없음. **LimitOffset**: SQL 방식, 임의 오프셋에 적합.

---

## 8. 필터링과 검색

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

## 9. 연습 문제

### 문제 1: 도서관 API

`Book`, `Author`, `Borrow` 모델을 위한 DRF API를 구축하세요. 중첩 직렬화기(nested serializer), 장르/대여 가능 여부 기준 필터링, 커스텀 `POST /api/books/{pk}/borrow/` 액션, 역할 기반 권한(staff만 도서 추가 가능)을 포함하세요.

### 문제 2: 직렬화기 유효성 검사

다음을 검사하는 `OrderSerializer`를 만드세요: 아이템 목록이 비어있지 않음, 수량이 양수임, 총액이 사용자의 신용 한도를 초과하지 않음, 배송 방법이 "shipping"인 경우 배송 주소 필수.

### 문제 3: JWT 인증 흐름

회원가입, 로그인(JWT 토큰), 토큰 갱신, 보호된 프로필 엔드포인트, 이전 비밀번호 확인이 필요한 비밀번호 변경을 구현하세요.

### 문제 4: 고급 필터링

다음을 포함하는 상품 검색을 구축하세요: 이름/설명에 대한 텍스트 검색, 가격 범위, 여러 카테고리 슬러그(slug), 최소 평균 평점, 가격/평점/관련도 기준 정렬.

---

## 참고 자료

- [DRF 공식 문서](https://www.django-rest-framework.org/)
- [DRF 직렬화기 가이드](https://www.django-rest-framework.org/api-guide/serializers/)
- [SimpleJWT 문서](https://django-rest-framework-simplejwt.readthedocs.io/)
- [django-filter 문서](https://django-filter.readthedocs.io/)
- [Classy DRF](https://www.cdrf.co/)

---

**이전**: [Django ORM](./11_Django_ORM.md) | **다음**: [Django 고급](./13_Django_Advanced.md)
