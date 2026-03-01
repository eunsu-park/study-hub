# 11. Django ORM

**이전**: [Django 기초](./10_Django_Basics.md) | **다음**: [Django REST Framework](./12_Django_REST_Framework.md)

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 체이닝(chaining), `values()`, 지연 평가(lazy evaluation)를 포함한 쿼리셋(QuerySet) API를 사용해 복잡한 쿼리를 작성한다
2. 필드 조회(field lookup), F 객체, Q 객체를 사용해 서브쿼리나 복잡한 WHERE 절이 필요한 쿼리를 표현한다
3. `select_related()`와 `prefetch_related()`를 사용해 N+1 쿼리 문제를 식별하고 해결한다
4. 집계(aggregation) 및 어노테이션(annotation) 함수를 적용해 분석용 쿼리를 작성한다
5. 커스텀 매니저(manager)와 쿼리셋(QuerySet)을 구현해 재사용 가능한 쿼리 로직을 캡슐화한다

---

Django의 ORM은 **액티브 레코드(Active Record)** 패턴을 따릅니다: 모델 인스턴스가 스스로 저장, 삭제, 조회하는 방법을 알고 있습니다. 핵심 추상화는 `QuerySet`으로, 데이터베이스 쿼리를 지연(lazy)되고, 체이닝(chainable) 가능하며, 불변(immutable)적으로 표현합니다.

## 목차

1. [쿼리셋 기초](#1-쿼리셋-기초)
2. [필드 조회](#2-필드-조회)
3. [F 객체와 Q 객체](#3-f-객체와-q-객체)
4. [집계와 어노테이션](#4-집계와-어노테이션)
5. [N+1 문제 해결](#5-n1-문제-해결)
6. [Raw SQL과 데이터베이스 함수](#6-raw-sql과-데이터베이스-함수)
7. [커스텀 매니저와 쿼리셋](#7-커스텀-매니저와-쿼리셋)
8. [연습 문제](#8-연습-문제)

---

## 1. 쿼리셋 기초

쿼리셋(QuerySet)은 **지연(lazy)** 방식으로 동작합니다 -- 순회하거나, 슬라이싱하거나, `list()`를 호출하거나, 평가(evaluate)될 때까지 SQL이 실행되지 않습니다.

```python
from blog.models import Post

published = Post.objects.filter(status="published")     # SQL 미실행
recent = published.order_by("-published_at")[:10]       # 여전히 SQL 미실행
for post in recent:                                     # 여기서 SQL 실행
    print(post.title)
```

주요 연산:

```python
Post.objects.all()                         # 모든 행
Post.objects.filter(status="published")    # WHERE status='published'
Post.objects.exclude(status="draft")       # WHERE NOT status='draft'
Post.objects.get(pk=1)                     # 단일 객체 (0개 또는 2개 이상이면 예외 발생)
Post.objects.count()                       # SELECT COUNT(*)
Post.objects.exists()                      # 효율적인 존재 여부 확인
```

쿼리셋은 불변(immutable)입니다 -- 각 메서드는 새로운 쿼리셋을 반환하므로 유창하게 체이닝(chaining)할 수 있습니다:

```python
results = (
    Post.objects
    .filter(status="published", category__name="Python")
    .exclude(author__username="bot")
    .order_by("-published_at")[:10]
)
```

전체 모델 인스턴스 대신 딕셔너리를 원하면 `values()`, 튜플을 원하면 `values_list()`를 사용합니다:

```python
Post.objects.values("id", "title", "author__username")
Post.objects.values_list("title", flat=True).distinct()
```

---

## 2. 필드 조회

필드 조회(field lookup)는 `field__lookup` 구문(이중 밑줄)을 사용합니다:

```python
Post.objects.filter(title__icontains="django")          # LIKE '%django%' (대소문자 무시)
Post.objects.filter(published_at__year=2025)             # 연도 추출
Post.objects.filter(published_at__gte=datetime(2025,1,1))  # >= 비교
Post.objects.filter(status__in=["published", "archived"])   # IN 절
Post.objects.filter(category__isnull=True)               # IS NULL
Post.objects.filter(
    published_at__range=(start_date, end_date)           # BETWEEN
)
```

주요 조회: `exact`, `iexact`, `contains`, `icontains`, `startswith`, `endswith`, `gt`, `gte`, `lt`, `lte`, `in`, `range`, `isnull`, `year`, `month`, `day`.

### 관계 탐색

이중 밑줄 구문은 ForeignKey와 ManyToMany 관계를 탐색합니다:

```python
Post.objects.filter(author__username="alice")            # FK 탐색
Post.objects.filter(author__profile__country="US")       # 다단계 탐색
Category.objects.filter(posts__status="published").distinct()  # 역방향 FK
```

---

## 3. F 객체와 Q 객체

### F 객체: 모델 필드 참조

`F()`는 SQL에서 필드 값을 참조하여 필드 간 비교와 원자적(atomic) 업데이트를 가능하게 합니다:

```python
from django.db.models import F
from datetime import timedelta

# 필드 비교: 발행 후 7일 이상 경과 후 업데이트된 게시물
Post.objects.filter(updated_at__gt=F("published_at") + timedelta(days=7))

# 원자적 증가 (경쟁 조건 없음)
Post.objects.filter(pk=1).update(view_count=F("view_count") + 1)
```

### Q 객체: 불리언 논리

여러 `filter()` 인자는 AND로 결합됩니다. `Q()`를 사용하면 OR와 NOT을 추가할 수 있습니다:

```python
from django.db.models import Q

# OR
Post.objects.filter(Q(status="published") | Q(author=request.user))

# NOT
Post.objects.filter(~Q(status="draft"))

# 복잡한 중첩: (published AND python) OR (draft AND mine)
Post.objects.filter(
    (Q(status="published") & Q(category__name="Python"))
    | (Q(status="draft") & Q(author=request.user))
)
```

Q 객체는 `filter()`에서 키워드 인자보다 앞에 위치해야 합니다.

---

## 4. 집계와 어노테이션

**집계(Aggregation)**는 단일 값을 계산하고, **어노테이션(annotation)**은 각 객체에 계산된 값을 추가합니다.

```python
from django.db.models import Count, Sum, Avg, Max

# 집계: 단일 결과 딕셔너리
Post.objects.aggregate(total=Count("id"), avg_views=Avg("view_count"))
# {"total": 42, "avg_views": 156.3}

# 어노테이션: 객체별 계산 필드
posts = Post.objects.annotate(
    comment_count=Count("comments")
).order_by("-comment_count")
```

### values() + annotate()를 사용한 GROUP BY

```python
Post.objects.values("category__name").annotate(
    count=Count("id"), avg_views=Avg("view_count")
).order_by("-count")
```

### 조건부 어노테이션

```python
from django.db.models import Case, When, Value, CharField

Post.objects.annotate(
    popularity=Case(
        When(view_count__gte=1000, then=Value("viral")),
        When(view_count__gte=100, then=Value("popular")),
        default=Value("normal"),
        output_field=CharField(),
    )
)
```

---

## 5. N+1 문제 해결

N+1 문제: N개의 객체를 로드한 뒤 각 객체에서 관련 필드에 접근하면 N번의 추가 쿼리가 발생합니다.

### select_related (ForeignKey / OneToOne)

SQL JOIN을 수행하여 관련 객체를 단일 쿼리로 가져옵니다:

```python
# N+1 대신 쿼리 1번
posts = Post.objects.select_related("author", "category").all()
for post in posts:
    print(post.author.username)  # 추가 쿼리 없음
```

### prefetch_related (ManyToMany / 역방향 FK)

관계별로 별도의 쿼리를 실행하고 Python에서 조인합니다:

```python
# N+1 대신 쿼리 3번
posts = Post.objects.prefetch_related("tags", "comments").all()
```

### 커스텀 Prefetch

```python
from django.db.models import Prefetch

posts = Post.objects.prefetch_related(
    Prefetch(
        "comments",
        queryset=Comment.objects.filter(is_approved=True).order_by("-created_at"),
        to_attr="approved_comments",
    )
)
```

FK/OneToOne에는 `select_related`(단일 JOIN), M2M/역방향 FK에는 `prefetch_related`(별도 쿼리)를 사용합니다.

---

## 6. Raw SQL과 데이터베이스 함수

### Raw 쿼리

```python
posts = Post.objects.raw("SELECT * FROM blog_post WHERE status = %s", ["published"])

from django.db import connection
with connection.cursor() as cursor:
    cursor.execute("SELECT category_id, COUNT(*) FROM blog_post GROUP BY 1")
    rows = cursor.fetchall()
```

### 데이터베이스 함수

```python
from django.db.models.functions import Lower, Coalesce, TruncMonth
from django.db.models import Value, Subquery, OuterRef

Post.objects.annotate(title_lower=Lower("title"))
Post.objects.annotate(display_cat=Coalesce("category__name", Value("Uncategorized")))

# 서브쿼리: 게시물별 최신 댓글 텍스트
newest = Comment.objects.filter(post=OuterRef("pk")).order_by("-created_at")
Post.objects.annotate(latest_comment=Subquery(newest.values("text")[:1]))
```

---

## 7. 커스텀 매니저와 쿼리셋

재사용 가능한 쿼리 패턴을 커스텀 쿼리셋(QuerySet)으로 캡슐화합니다:

```python
class PostQuerySet(models.QuerySet):
    def published(self):
        return self.filter(status="published")

    def by_author(self, user):
        return self.filter(author=user)

    def popular(self, min_views: int = 100):
        return self.filter(view_count__gte=min_views)

    def with_comment_count(self):
        return self.annotate(comment_count=Count("comments"))

class Post(models.Model):
    # ... 필드 ...
    objects = PostQuerySet.as_manager()
```

체이닝 사용:

```python
trending = (
    Post.objects.published().popular(50).with_comment_count()
    .order_by("-comment_count")[:10]
)
```

---

## 8. 연습 문제

### 문제 1: 쿼리셋 연습

다음을 위한 쿼리셋을 작성하세요: (a) 최근 30일 내 발행된 게시물을 조회 수 기준으로 정렬, (b) 제목에 "Django"가 포함되고 댓글이 하나 이상 있는 게시물, (c) 발행된 게시물 수 기준 상위 5개 카테고리, (d) `updated_at`이 `published_at`보다 7일 이상 늦은 게시물.

### 문제 2: N+1 최적화

다음 N+1 뷰를 수정하세요:

```python
def dashboard(request):
    posts = Post.objects.filter(author=request.user)
    data = []
    for post in posts:
        data.append({
            "title": post.title,
            "category": post.category.name,
            "tags": [t.name for t in post.tags.all()],
            "comment_count": post.comments.count(),
        })
    return JsonResponse({"posts": data})
```

### 문제 3: 커스텀 쿼리셋

체이닝 가능한 메서드를 가진 `ProductQuerySet`을 만드세요: `available()`(재고 > 0이고 활성), `in_price_range(min, max)`, `with_avg_rating()`, `bestsellers(limit=10)`.

### 문제 4: 복잡한 집계

월별 매출 보고서 쿼리를 작성하세요: 월별 총 매출, 평균 주문 금액, 고유 고객 수, 월별 베스트셀러 상품(힌트: Subquery 사용).

---

## 참고 자료

- [Django QuerySet API 레퍼런스](https://docs.djangoproject.com/en/5.1/ref/models/querysets/)
- [Django 집계 가이드](https://docs.djangoproject.com/en/5.1/topics/db/aggregation/)
- [Django 데이터베이스 함수](https://docs.djangoproject.com/en/5.1/ref/models/database-functions/)
- [django-debug-toolbar](https://django-debug-toolbar.readthedocs.io/) -- SQL 쿼리 검사 도구

---

**이전**: [Django 기초](./10_Django_Basics.md) | **다음**: [Django REST Framework](./12_Django_REST_Framework.md)
