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
8. [대량 연산과 트랜잭션](#8-대량-연산과-트랜잭션)
9. [연습 문제](#9-연습-문제)

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

### 이론: 지연 평가: 지연 SQL로서의 QuerySet

QuerySet은 모델 인스턴스의 *리스트가 아닙니다*. 쿼리를 *기술하는* 객체입니다 — 무언가가 실행을 강제하기 전까지 SQL은 프로세스를 떠나지 않습니다.

#### A.1 만들어지는 것 vs 실행되는 것

```python
qs = Post.objects.filter(published=True).order_by("-created_at")[:10]
# 아직 SQL 없음. qs는 다음을 기술하는 QuerySet 객체:
#   SELECT ... FROM blog_post WHERE published = TRUE ORDER BY created_at DESC LIMIT 10
```

Python에서 일어나는 일:

1. `Post.objects`가 기본 `Manager`를 반환하고, 그것이 신선한 `QuerySet`을 반환합니다.
2. `.filter(...)`가 그 WHERE 절이 내부 쿼리 트리에 추가된 *새* QuerySet을 반환합니다.
3. `.order_by(...)`가 ORDER BY가 추가된 *또 다른* 새 QuerySet을 반환합니다.
4. `[:10]`이 LIMIT이 있는 *또 또 다른* QuerySet을 반환합니다.

각 QuerySet은 불변입니다. 체이닝이 새 것을 만듭니다. 내부적으로 각각 `Query` 객체를 보유합니다 — Django의 SQL 문 트리 표현입니다. 아직 라운드트립은 없습니다.

#### A.2 평가를 강제하는 트리거들

QuerySet은 다음 중 하나가 일어나는 순간 평가됩니다(즉, SQL을 발행합니다):

- **순회** — `for post in qs:`가 쿼리를 실행하고 결과를 돕니다.
- **step이 있는 슬라이싱** 또는 **bool 변환** — `if qs:`가 쿼리를 합니다.
- **`list(qs)`, `len(qs)`, `bool(qs)`** — 명시적 materialization.
- **`qs.count()`, `qs.exists()`, `qs.first()`, `qs.last()`** — 최적화된 쿼리를 실행합니다.
- **Pickling, repr, JSON 직렬화** — 결국 순회로 끝납니다.

평가되면 결과는 QuerySet에 캐시됩니다 — 다시 순회해도 쿼리를 다시 돌리지 않습니다. 그러나 평가 후 `qs.filter(...)`를 체이닝하면 캐시가 빈 새 QuerySet이 만들어지고, 새 쿼리는 다시 실행됩니다.

실용적 결과: 같은 queryset을 템플릿에 전달하고 그 위에서 다른 메서드를 호출하면 우연히 데이터베이스를 여러 번 치기 쉽습니다. 해결책은 `list(qs)`를 한 번 한 뒤 materialized 리스트 위에서 작업하는 것입니다.

#### A.3 성능 함의

지연 QuerySet은 아무것도 보내기 전에 Python에서 쿼리를 합성하게 해줍니다. 이것이 기능입니다. 서비스가 베이스 queryset을 만들고 평가 전에 여러 계층의 필터를 통과시킬 수 있는데, 모든 필터가 하나의 SQL 문으로 합쳐집니다. 동시에 발에 쏘는 총입니다. 평가가 조용히 일어나는 순간(템플릿 순회에서, 로깅 문에서)이 "빠른" 함수가 느려지는 순간입니다.

표준 규율: 각 QuerySet이 정확히 어디서 평가되는지를 알 것. 개발에서는 `django-debug-toolbar`를 사용하고, 프로덕션에서는 `connection.queries`나 `pg_stat_statements`를 통해 느린 쿼리를 로깅하세요.

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

| 접근법 | 쿼리 수 | 예시 |
|--------|---------|------|
| 단순(Naive) | N+1 | posts 1번 + 각 post.author마다 N번 |
| select_related | 1 (JOIN) | JOIN을 포함한 단일 쿼리 |
| prefetch_related | 2 | posts 1번 + 모든 author 1번 |

### 이론: Django의 N+1 문제

레슨 04 §C.1과 레슨 08 §A.3과 같은 N+1 함정이지만, Django의 기제는 다릅니다.

#### B.1 모양

```python
posts = Post.objects.all()            # 쿼리 1개
for post in posts:
    print(post.author.username)       # FK당 post 하나마다 쿼리 1개
                                      # → 총 N+1
```

각 `post.author` 접근이 관련된 User 행을 lazy하게 로드합니다. post 100개면 쿼리 101개입니다.

#### B.2 두 가지 방어

Django는 두 개의 구별되는 로더를 제공합니다.

| 메서드 | 메커니즘 | 사용처 |
|--------|-----------|---------|
| `select_related("author")` | SQL `INNER JOIN`(nullable FK는 `LEFT JOIN`), 단일 쿼리 | 다대일(FK), 일대일 |
| `prefetch_related("comments")` | 부모 쿼리 1개, 자식에 대한 추가 `WHERE parent_id IN (...)` 쿼리 1개, Python에서 join | 일대다(역 FK), 다대다 |

`select_related`는 모든 forward foreign key에 대한 올바른 호출입니다 — 관련된 행이 같은 쿼리에 도착합니다. `prefetch_related`는 역 관계와 다대다에 대한 올바른 호출입니다 — Django는 부모 join의 한 행에 다중 행 자식 집합을 표현할 수 없으므로 대신 두 쿼리를 합니다.

```python
posts = (
    Post.objects
        .select_related("author")           # FK → JOIN
        .prefetch_related("comments")       # 역 FK → IN 쿼리
)
for post in posts:
    print(post.author.username)             # 추가 쿼리 없음
    for c in post.comments.all():           # post당 추가 쿼리 없음
        print(c.body)
```

총 두 쿼리, 101개가 아니라.

#### B.3 체이닝된 prefetch

`prefetch_related`는 계층을 가로질러 체이닝됩니다. `prefetch_related("comments__author")`는 추가 쿼리 각 1개로 comments와 각 comment의 author를 가져옵니다. 복잡한 그래프에서는 `Prefetch(...)`로 안쪽 queryset을 커스터마이즈할 수 있습니다(예: 발행된 comment만).

원리는 일반화됩니다: **쿼리가 필요로 하는 관계를 명시적으로 진술하라, 그러면 Django가 최소 라운드트립 수로 압축한다**. Lazy 접근은 편의이지 전략이 아닙니다.

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

### 이론: QuerySet에서 SQL로: 컴파일 파이프라인

`.filter()`, `.annotate()`, `.values()` 체인은 트리를 짓는 DSL입니다. 평가가 트리거되면 Django가 그 트리를 SQL로 컴파일합니다.

#### C.1 컴파일 단계

1. **Build**: 체이닝된 각 메서드가 내부 `Query` 객체에 추가하거나 수정합니다.
2. **Compile**: 평가 시점에 Django의 `SQLCompiler`가 `Query` 트리를 따라 다음을 만듭니다:
   - SELECT 절(가져올 컬럼),
   - FROM 절(모델의 테이블과 join된 테이블들),
   - WHERE 절(`filter()`와 `Q()`에서 번역됨),
   - GROUP BY / HAVING (`annotate()` + `aggregate()`에서),
   - ORDER BY (`order_by()`에서),
   - LIMIT/OFFSET (슬라이싱에서).
3. **Parameterize**: 리터럴이 bind placeholder가 됩니다(psycopg는 `%s`, sqlite는 `?`). 레슨 08 §B의 같은 prepared-statement 방어.
4. **Execute**: 매개변수화된 SQL이 데이터베이스 백엔드를 통해 실행됩니다.
5. **Hydrate**: 행이 모델 인스턴스(또는 `values()`/`values_list()`의 경우 dict/tuple)가 됩니다.

컴파일러는 결정론적입니다 — 같은 체인이 매번 같은 SQL을 만듭니다. `qs.query`(또는 `str(qs.query)`)를 검사하면 Django가 실행할 SQL을 보여줍니다.

#### C.2 `values()`와 `values_list()`: hydration 건너뛰기

행을 완전한 모델 인스턴스로 hydrate하는 것은 공짜가 아닙니다. 모델 메서드가 필요 없는 집계/리포팅 쿼리에서는 `values()`가 dict를, `values_list()`가 tuple을 반환합니다 — 모델 인스턴스화 없음. 더 빠르고 더 적은 메모리:

```python
Post.objects.values("status").annotate(n=Count("id"))
# [{'status': 'draft', 'n': 12}, {'status': 'published', 'n': 84}]
```

#### C.3 F 객체: 서버 측 계산

`F("price") * 2`는 컬럼을 Python으로 읽지 않고 SQL 수준에서 참조합니다. 비교:

```python
# 나쁨: 읽고, Python에서 계산하고, 다시 쓰기
for product in Product.objects.all():
    product.price *= 2
    product.save()

# 좋음: 서버에서 단일 UPDATE
Product.objects.update(price=F("price") * 2)
```

첫 번째 버전은 N+1에 N개 쓰기. 두 번째는 라운드트립 1개이며 race-free입니다 — 다른 트랜잭션이 다른 가격을 끼워 넣을 수 있는 read-then-write 창이 없습니다.

#### C.4 Q 객체: WHERE 절 합성

`Q(status="published") | Q(featured=True)`는 OR을 만듭니다. `~Q(...)`는 NOT입니다. 이들은 컴파일러가 emit하는 WHERE 트리로 합성되어, 평범한 `.filter(**kwargs)`(암묵적 AND)로는 표현할 수 없는 로직을 표현하게 해줍니다.

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

## 8. 대량 연산과 트랜잭션

쓰기 작업이 많은 워크로드의 경우, 행별 저장을 피하고 Django의 대량 API를 활용한다.

### bulk_create와 bulk_update

```python
# bulk_create: 여러 행을 단일 INSERT로 처리
posts = [Post(title=f"Post {i}", body="...", author_id=1) for i in range(1000)]
Post.objects.bulk_create(posts, batch_size=200)

# bulk_update: 여러 행을 단일 UPDATE로 처리 (먼저 조회해야 함)
posts = list(Post.objects.filter(status="draft"))
for post in posts:
    post.status = "published"
Post.objects.bulk_update(posts, ["status"], batch_size=200)
```

`bulk_create`는 `save()` 시그널과 행별 유효성 검사를 건너뛴다; 호출 전에 명시적으로 유효성을 검사해야 한다.

### 원자적 트랜잭션(Atomic Transactions)

관련 쓰기 작업을 `atomic()`으로 감싸면 실패 시 모든 변경사항이 롤백된다:

```python
from django.db import transaction

@transaction.atomic
def transfer_points(from_user_id, to_user_id, amount):
    sender = User.objects.select_for_update().get(pk=from_user_id)
    receiver = User.objects.select_for_update().get(pk=to_user_id)
    if sender.points < amount:
        raise ValueError("Insufficient points")
    sender.points -= amount
    receiver.points += amount
    sender.save(update_fields=["points"])
    receiver.save(update_fields=["points"])
```

`select_for_update()`는 행 수준 잠금(row-level lock)을 획득하여 동시 수정을 방지한다.

### 대용량 쿼리셋을 위한 iterator()

수백만 행을 메모리에 로드하면 워커가 충돌한다. `iterator()`를 사용하여 청크(chunk) 단위로 스트리밍한다:

```python
# iterator() 없이: 모든 행을 한 번에 로드
for post in Post.objects.all():          # 대규모에서 OOM 위험
    process(post)

# iterator() 사용: DB 왕복당 chunk_size 행씩 가져옴
for post in Post.objects.all().iterator(chunk_size=500):
    process(post)
```

`iterator()`와 `prefetch_related()`를 함께 사용하지 말 것 — 프리패치(prefetching)는 서버 측 커서와 호환되지 않는다.

---

## 9. 연습 문제

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
