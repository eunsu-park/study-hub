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

### 이론: 직렬화기: 두 방향, 하나의 클래스

DRF 직렬화기는 *외부* 표현(JSON 모양)과 *내부* 표현(Python 객체, 보통 Django 모델 인스턴스) 둘 다를 아는 클래스입니다. 방향에 따라 두 파이프라인을 돌립니다.

#### A.1 직렬화 방향 (인스턴스 → JSON)

뷰가 인스턴스를 가지고 응답을 만들어야 할 때:

```python
serializer = PostSerializer(post)
data = serializer.data  # JSONRenderer에 바로 줄 OrderedDict
```

파이프라인:

1. 선언된 각 필드에 대해 인스턴스 속성에 그 필드의 `to_representation(value)` 메서드를 호출합니다.
2. 합성 필드(중첩 직렬화기, `SerializerMethodField`)는 재귀합니다.
3. 결과는 원시 타입(str, int, list, dict)의 `OrderedDict`입니다.
4. 렌더러가 JSON 바이트로 변환합니다.

이는 한 방향입니다: 인스턴스에서 값을 빼내고, 필드 수준 변환기를 거쳐, 출력을 조립합니다.

#### A.2 역직렬화 방향 (JSON → 검증된 인스턴스)

뷰가 요청 본문을 받을 때:

```python
serializer = PostSerializer(data=request.data)
serializer.is_valid(raise_exception=True)  # 검증 실행
post = serializer.save()                    # 인스턴스 생성 또는 업데이트
```

파이프라인이 더 깁니다.

1. **초기 파싱.** 렌더러의 역(기본 JSONParser)이 실행되어 바이트를 Python dict로 만듭니다.
2. **필드 수준 검증(`to_internal_value`).** 선언된 각 필드가 입력을 검증합니다 — 타입 검사, 최대 길이, 외래 키 존재 — 그리고 변환된 Python 값을 반환하거나 `ValidationError`를 던집니다.
3. **필드 수준 `validate_<field>(value)` hook.** 선택적 필드별 커스텀 검증.
4. **객체 수준 `validate(attrs)` hook.** 필드 간 검증. 모든 필드 값을 받고, 검증된 dict를 반환하거나 throw합니다.
5. **`save()`가 `create(validated_data)` 또는 `update(instance, validated_data)`로 디스패치합니다.** 보통 여기서 `Model.objects.create(**validated_data)`가 실행됩니다.

2-4 단계가 DRF가 정확한 구조화된 오류(그저 "bad request"가 아니라)를 반환할 수 있는 이유입니다. 각 필드가 자체 오류 메시지를 모읍니다.

#### A.3 ModelSerializer: 선언적 단축

`ModelSerializer`는 Django 모델을 인트로스펙트해 자동으로 필드 선언을 생성합니다.

```python
class PostSerializer(serializers.ModelSerializer):
    class Meta:
        model = Post
        fields = ["id", "title", "content", "created_at"]
```

필드는 모델의 컬럼 메타데이터에서 파생되고, validator는 컬럼 제약(`max_length`, `unique` 등)에서 파생되며, 기본 `create()`와 `update()`는 `Model.objects.create(...)`와 `instance.save()`를 호출합니다. 비기본인 것만 override합니다.

레슨 02 §B와 같은 평행: FastAPI에서 같은 Pydantic 모델이 검증·직렬화·OpenAPI 문서를 구동하듯, DRF에서 같은 ModelSerializer가 검증·직렬화·API 스키마를 구동합니다.

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

### 이론: ViewSet + Router: 관습 기반 리소스 매핑

평범한 Django(레슨 10 §C.4)에서는 모든 URL이 명시적으로 매핑됩니다. DRF는 관습 계층을 더합니다. HTTP 동작에 매핑되는 메서드 이름의 클래스를 정의하고, Router에 등록하면 URL 패턴이 생성됩니다.

#### B.1 동작 매핑

`ModelViewSet`은 6개의 표준 메서드를 노출하며, 각각이 HTTP 메서드/경로에 묶여 있습니다.

| 동작 | HTTP 메서드 | URL 패턴 |
|--------|-------------|-------------|
| `list` | GET | `/posts/` |
| `create` | POST | `/posts/` |
| `retrieve` | GET | `/posts/{pk}/` |
| `update` | PUT | `/posts/{pk}/` |
| `partial_update` | PATCH | `/posts/{pk}/` |
| `destroy` | DELETE | `/posts/{pk}/` |

다섯 줄 코드로 완전한 CRUD 엔드포인트 세트를 얻습니다.

```python
class PostViewSet(viewsets.ModelViewSet):
    queryset = Post.objects.all()
    serializer_class = PostSerializer
```

여섯 메서드 모두의 기본 구현이 `ModelViewSet`에서 상속됩니다. 커스터마이즈가 필요한 것만 override합니다.

#### B.2 Router가 URL 배선을 한다

```python
router = DefaultRouter()
router.register(r"posts", PostViewSet)
urlpatterns = router.urls
```

`router.urls`는 명시적 `path()` 라인 6줄을 직접 쓰는 것과 동등한 `path(...)` 항목 리스트입니다. URL 패턴을 0개 작성했고, 레슨 01 §B.3의 완전한 REST 리소스 모양을 얻었습니다.

#### B.3 커스텀 동작: `@action` 데코레이터

비표준 엔드포인트(예: `POST /posts/{pk}/publish/`)에는 `@action` 데코레이터가 추가 메서드를 선언합니다.

```python
class PostViewSet(viewsets.ModelViewSet):
    ...
    @action(detail=True, methods=["post"])
    def publish(self, request, pk=None):
        post = self.get_object()
        post.publish()
        return Response({"status": "published"})
```

`detail=True`는 `/posts/{pk}/publish/`에 바인딩하고, `detail=False`는 `/posts/publish/`에 바인딩합니다. Router가 자동으로 픽업합니다.

#### B.4 이것이 중요한 이유: REST 모양이 기본값이 된다

DRF 없이는 모든 Django REST API가 "그것의 URL은 무엇이어야 하지?"에 대한 한 개발자의 해석입니다. DRF + ViewSet + Router로는 답이 프레임워크에 의해 고정됩니다. 표준 6개 동작이 표준 URL의 표준 HTTP 메서드에 매핑됩니다. 팀에 합류하는 새 개발자가 모델 리스트만 보고도 URL 표면의 90%를 예측할 수 있습니다.

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

### 이론: 플러그 가능한 백엔드 프로토콜

DRF의 모든 횡단 관심사 — 인증, 권한, throttle, 페이지네이션, 필터 — 는 같은 플러그인 패턴을 따릅니다: 잘 알려진 메서드 하나를 가진 클래스를 리스트에 등록하면, 뷰가 적절한 시점에 호출합니다.

#### C.1 모양

```python
class MyAuthentication(BaseAuthentication):
    def authenticate(self, request):
        # 성공 시 (user, auth_token) 반환, 자격 증명 없으면 None, 잘못된 자격 증명이면 throw
        ...

class MyPermission(BasePermission):
    def has_permission(self, request, view):
        return True or False  # 뷰에 대해 전역
    def has_object_permission(self, request, view, obj):
        return True or False  # 객체별 검사

class MyThrottle(BaseThrottle):
    def allow_request(self, request, view):
        return True or False
```

각 백엔드는 (구조적 타이핑 의미에서) *프로토콜*입니다. 올바른 메서드를 구현하면 DRF가 그것을 유효한 구현으로 취급합니다.

#### C.2 실행 순서

한 요청에 대해 DRF의 `APIView.initial(request)`가 실행됩니다.

1. **인증.** `authentication_classes` 리스트를 따라갑니다. 처음 `(user, auth)`를 반환하는 것이 이기고 `request.user`와 `request.auth`를 설정합니다. 매치된 것이 없으면 `request.user = AnonymousUser`.
2. **권한.** `permission_classes`를 따라가며 각각에 `has_permission(request, view)`를 호출합니다. 어느 하나라도 False → `403 Forbidden`(익명이면 `401`).
3. **Throttle.** `throttle_classes`를 따라가며 `allow_request`를 호출합니다. 어느 하나라도 False → `429 Too Many Requests`.

그 후 뷰의 동작 메서드(`list`, `retrieve` 등)가 실행됩니다. 객체 수준 뷰의 경우 `get_object()`가 추가로 인스턴스를 가져온 뒤 `has_object_permission`을 검사합니다.

#### C.3 출력에도 같은 패턴: 페이지네이션, 필터링

출력 측은 입력을 거울처럼 따릅니다. `pagination_class`는 `paginate_queryset(queryset, request, view)`와 `get_paginated_response(data)`를 제공합니다. `filter_backends`는 `filter_queryset(request, queryset, view)`를 제공합니다. 각각이 파이프라인의 알려진 지점에서 호출되는 알려진 메서드를 가진 클래스입니다.

이 균일성의 이득: 커스텀 인증, 권한, throttle, paginator, 필터를 작성하는 것이 같은 연습입니다 — 프로토콜을 구현하고, 클래스를 등록하고, 끝. "DRF를 어떻게 확장하지?"의 인지 부담이 "그 백엔드의 메서드 이름이 뭐였더라?"로 줄어듭니다.

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
