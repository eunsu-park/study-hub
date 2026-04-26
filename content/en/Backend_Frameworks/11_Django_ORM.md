# 11. Django ORM

**Previous**: [Django Basics](./10_Django_Basics.md) | **Next**: [Django REST Framework](./12_Django_REST_Framework.md)

## Learning Objectives

After completing this lesson, you will be able to:

1. Construct complex queries using the QuerySet API including chaining, `values()`, and lazy evaluation
2. Use field lookups, F objects, and Q objects to express queries that would require subqueries or complex WHERE clauses in raw SQL
3. Identify and resolve the N+1 query problem using `select_related()` and `prefetch_related()`
4. Apply aggregation and annotation functions to produce analytics-style queries
5. Implement custom managers and QuerySets to encapsulate reusable query logic

---

Django's ORM follows the **Active Record** pattern: model instances know how to save, delete, and query themselves. The core abstraction is the `QuerySet` -- a lazy, chainable, immutable representation of a database query.

## Table of Contents

Before the framework reference, read [**Theory & Principles**](#theory--principles) — what "lazy QuerySet" actually means at the SQL/Python boundary, the deferred-evaluation rules that decide when SQL fires, and the N+1 problem expressed as a SQL execution count.

1. [QuerySet Fundamentals](#1-queryset-fundamentals)
2. [Field Lookups](#2-field-lookups)
3. [F and Q Objects](#3-f-and-q-objects)
4. [Aggregation and Annotation](#4-aggregation-and-annotation)
5. [Solving the N+1 Problem](#5-solving-the-n1-problem)
6. [Raw SQL and Database Functions](#6-raw-sql-and-database-functions)
7. [Custom Managers and QuerySets](#7-custom-managers-and-querysets)
8. [Bulk Operations and Transactions](#8-bulk-operations-and-transactions)
9. [Practice Problems](#9-practice-problems)

---

## Theory & Principles

The Django ORM looks like normal Python code, but it is hiding a deferred SQL pipeline behind every `.filter()` call. Three concepts explain everything from "why is my query slow?" to "why am I getting `RelatedManager` instead of a list?".

- **(A) Lazy evaluation: the QuerySet as deferred SQL** — what is built in Python, what is executed in the database, and exactly when the boundary is crossed.
- **(B) The N+1 problem in Django form** — the same trap from Lesson 04 §C and Lesson 08 §A.3, with Django-specific defenses.
- **(C) The query plan: from QuerySet to SQL** — how chaining, lookups, and annotations compose into one SQL statement (or several).

### A. Lazy Evaluation: QuerySet as Deferred SQL

A QuerySet is *not* a list of model instances. It is an object that *describes* a query — until something forces it to execute, no SQL leaves your process.

#### A.1 What is built vs what is executed

```python
qs = Post.objects.filter(published=True).order_by("-created_at")[:10]
# Still no SQL. qs is a QuerySet object describing:
#   SELECT ... FROM blog_post WHERE published = TRUE ORDER BY created_at DESC LIMIT 10
```

What happens in Python:

1. `Post.objects` returns the default `Manager`, which returns a fresh `QuerySet`.
2. `.filter(...)` returns a *new* QuerySet with that WHERE clause appended to its internal query tree.
3. `.order_by(...)` returns *another* new QuerySet with ORDER BY added.
4. `[:10]` returns *yet another* QuerySet with LIMIT.

Each QuerySet is immutable; chaining produces new ones. Internally, each holds a `Query` object — Django's tree representation of a SQL statement. No round-trip yet.

#### A.2 The triggers that force evaluation

A QuerySet evaluates (i.e., issues SQL) the moment one of these happens:

- **Iteration** — `for post in qs:` runs the query and walks results.
- **Slicing with a step** or **bool conversion** — `if qs:` does the query.
- **`list(qs)`, `len(qs)`, `bool(qs)`** — explicit materialization.
- **`qs.count()`, `qs.exists()`, `qs.first()`, `qs.last()`** — execute optimized queries.
- **Pickling, repr, JSON serialization** — these end up iterating.

Once evaluated, the result is cached on the QuerySet — re-iterating does not re-run the query. But chaining `qs.filter(...)` after evaluation creates a new QuerySet whose cache is empty; the new query will execute again.

The practical consequence: it is easy to accidentally hit the database many times in a template by passing the same queryset and calling different methods on it. The fix is `list(qs)` once, then operate on the materialized list.

#### A.3 The performance implication

Lazy QuerySets let you compose queries in Python before sending anything. That is a feature: a service can build a base queryset and pass it through several layers of filters before evaluation, with all filters merged into one SQL statement. It is also a footgun: the moment evaluation happens silently (in a template iteration, in a logging statement) is the moment a "fast" function turns slow.

The standard discipline: know exactly where each QuerySet evaluates. Use `django-debug-toolbar` in development; in production, log slow queries via `connection.queries` or `pg_stat_statements`.

### B. The N+1 Problem in Django

Same N+1 trap as Lesson 04 §C.1 and Lesson 08 §A.3, but Django's mechanics are different.

#### B.1 The shape

```python
posts = Post.objects.all()            # 1 query
for post in posts:
    print(post.author.username)       # 1 query per post for FK
                                      # → N+1 total
```

Each `post.author` access lazily loads the related User row. With 100 posts that is 101 queries.

#### B.2 The two defenses

Django provides two distinct loaders:

| Method | Mechanism | Use for |
|--------|-----------|---------|
| `select_related("author")` | SQL `INNER JOIN` (or `LEFT JOIN` for nullable FK), single query | many-to-one (FK), one-to-one |
| `prefetch_related("comments")` | One query for the parents, one extra query `WHERE parent_id IN (...)` for children, joined in Python | one-to-many (reverse FK), many-to-many |

`select_related` is the right call for any forward foreign key — the related row arrives in the same query. `prefetch_related` is the right call for reverse relations and many-to-many — Django can't represent a multi-row child set in a single row of the parent's join, so it does two queries instead.

```python
posts = (
    Post.objects
        .select_related("author")           # FK → JOIN
        .prefetch_related("comments")       # reverse FK → IN query
)
for post in posts:
    print(post.author.username)             # no extra query
    for c in post.comments.all():           # no extra query per post
        print(c.body)
```

Two queries total instead of 101.

#### B.3 Chained prefetches

`prefetch_related` chains across levels: `prefetch_related("comments__author")` fetches comments and each comment's author with one extra query each. For complex graphs, `Prefetch(...)` lets you customize the inner queryset (e.g., only published comments).

The principle generalizes: **state explicitly what relations a query needs, and Django collapses them into the minimum number of round-trips**. Lazy access is a convenience, not a strategy.

### C. From QuerySet to SQL: The Compilation Pipeline

The `.filter()`, `.annotate()`, `.values()` chain is a tree-building DSL. When evaluation triggers, Django compiles that tree into SQL.

#### C.1 The compile stages

1. **Build**: each chained method appends to or modifies an internal `Query` object.
2. **Compile**: at evaluation time, Django's `SQLCompiler` walks the `Query` tree and produces:
   - the SELECT clause (columns to fetch),
   - the FROM clause (the model's table plus any joined tables),
   - the WHERE clause (translated from `filter()` and `Q()`),
   - the GROUP BY / HAVING (from `annotate()` + `aggregate()`),
   - the ORDER BY (from `order_by()`),
   - the LIMIT/OFFSET (from slicing).
3. **Parameterize**: literals become bind placeholders (`%s` for psycopg, `?` for sqlite). Same prepared-statement defense from Lesson 08 §B.
4. **Execute**: the parameterized SQL runs through the database backend.
5. **Hydrate**: rows become model instances (or dicts/tuples for `values()`/`values_list()`).

The compiler is deterministic — the same chain produces the same SQL every time. Inspecting `qs.query` (or `str(qs.query)`) shows the SQL Django will run.

#### C.2 `values()` and `values_list()`: skip hydration

Hydrating a row into a full model instance is not free. For aggregation/reporting queries that don't need model methods, `values()` returns dicts and `values_list()` returns tuples — no model instantiation. Faster and lower memory:

```python
Post.objects.values("status").annotate(n=Count("id"))
# [{'status': 'draft', 'n': 12}, {'status': 'published', 'n': 84}]
```

#### C.3 F objects: server-side computation

`F("price") * 2` references the column at SQL level rather than reading it into Python. Compare:

```python
# Bad: read, compute in Python, write back
for product in Product.objects.all():
    product.price *= 2
    product.save()

# Good: single UPDATE on the server
Product.objects.update(price=F("price") * 2)
```

The first version is N+1 plus N writes. The second is one round-trip and is also race-free — no read-then-write window where another transaction could insert a different price.

#### C.4 Q objects: composing WHERE clauses

`Q(status="published") | Q(featured=True)` builds an OR. `~Q(...)` is NOT. These compose into the WHERE tree the compiler emits, letting you express logic that plain `.filter(**kwargs)` cannot (which is implicitly AND).

### From Theory to the Code Below

Each section that follows operationalizes one piece of this framework:

- §1 (QuerySet fundamentals) explores §A.1's lazy chaining and §A.2's evaluation triggers in concrete code.
- §2 (Field lookups) is the syntax (`__gte`, `__contains`, `__in`) that adds WHERE conditions to the §C.1 query tree.
- §3 (F and Q objects) is §C.3 (server-side compute) and §C.4 (composable WHERE).
- §4 (Aggregation and annotation) generates GROUP BY queries — §C.1 stage 2 with `Sum`, `Avg`, `Count`.
- §5 (Solving N+1) is §B.2's `select_related` and `prefetch_related` in detail.
- §6 (Raw SQL) is the escape hatch when the §C compiler cannot express what you need.
- §7 (Custom managers/QuerySets) lets you encapsulate common chains as named methods, so calling code stays readable.
- §8 (Bulk operations and transactions) is the §C.3 server-side update pattern plus the transaction wrapper from Lesson 08 §C.

---

## 1. QuerySet Fundamentals

QuerySets are **lazy** -- no SQL executes until you iterate, slice, call `list()`, or evaluate them.

```python
from blog.models import Post

published = Post.objects.filter(status="published")     # No SQL yet
recent = published.order_by("-published_at")[:10]       # Still no SQL
for post in recent:                                     # SQL executes here
    print(post.title)
```

Key operations:

```python
Post.objects.all()                         # All rows
Post.objects.filter(status="published")    # WHERE status='published'
Post.objects.exclude(status="draft")       # WHERE NOT status='draft'
Post.objects.get(pk=1)                     # Single object (raises on 0 or 2+)
Post.objects.count()                       # SELECT COUNT(*)
Post.objects.exists()                      # Efficient existence check
```

QuerySets are immutable -- each method returns a new QuerySet, enabling fluent chaining:

```python
results = (
    Post.objects
    .filter(status="published", category__name="Python")
    .exclude(author__username="bot")
    .order_by("-published_at")[:10]
)
```

Use `values()` for dictionaries or `values_list()` for tuples instead of full model instances:

```python
Post.objects.values("id", "title", "author__username")
Post.objects.values_list("title", flat=True).distinct()
```

---

## 2. Field Lookups

Field lookups use `field__lookup` syntax (double underscore):

```python
Post.objects.filter(title__icontains="django")          # LIKE '%django%' (case-insensitive)
Post.objects.filter(published_at__year=2025)             # Extract year
Post.objects.filter(published_at__gte=datetime(2025,1,1))  # >= comparison
Post.objects.filter(status__in=["published", "archived"])   # IN clause
Post.objects.filter(category__isnull=True)               # IS NULL
Post.objects.filter(
    published_at__range=(start_date, end_date)           # BETWEEN
)
```

Common lookups: `exact`, `iexact`, `contains`, `icontains`, `startswith`, `endswith`, `gt`, `gte`, `lt`, `lte`, `in`, `range`, `isnull`, `year`, `month`, `day`.

### Spanning Relationships

The double-underscore syntax traverses ForeignKey and ManyToMany relationships:

```python
Post.objects.filter(author__username="alice")            # FK traversal
Post.objects.filter(author__profile__country="US")       # Multi-level
Category.objects.filter(posts__status="published").distinct()  # Reverse FK
```

---

## 3. F and Q Objects

### F Objects: Reference Model Fields

`F()` references a field's value in SQL, enabling field-to-field comparisons and atomic updates:

```python
from django.db.models import F
from datetime import timedelta

# Field comparison: updated more than 7 days after publishing
Post.objects.filter(updated_at__gt=F("published_at") + timedelta(days=7))

# Atomic increment (no race condition)
Post.objects.filter(pk=1).update(view_count=F("view_count") + 1)
```

### Q Objects: Boolean Logic

Multiple `filter()` args use AND. `Q()` adds OR and NOT:

```python
from django.db.models import Q

# OR
Post.objects.filter(Q(status="published") | Q(author=request.user))

# NOT
Post.objects.filter(~Q(status="draft"))

# Complex nesting: (published AND python) OR (draft AND mine)
Post.objects.filter(
    (Q(status="published") & Q(category__name="Python"))
    | (Q(status="draft") & Q(author=request.user))
)
```

Q objects must appear before keyword arguments in `filter()`.

---

## 4. Aggregation and Annotation

**Aggregation** computes a single value; **annotation** attaches a computed value to each object.

```python
from django.db.models import Count, Sum, Avg, Max

# Aggregate: single result dictionary
Post.objects.aggregate(total=Count("id"), avg_views=Avg("view_count"))
# {"total": 42, "avg_views": 156.3}

# Annotate: per-object computed field
posts = Post.objects.annotate(
    comment_count=Count("comments")
).order_by("-comment_count")
```

### GROUP BY with values() + annotate()

```python
Post.objects.values("category__name").annotate(
    count=Count("id"), avg_views=Avg("view_count")
).order_by("-count")
```

### Conditional Annotation

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

## 5. Solving the N+1 Problem

The N+1 problem: loading N objects then accessing a related field on each triggers N extra queries.

| Approach | Queries | Example |
|----------|---------|---------|
| Naive | N+1 | 1 for posts + N for each post.author |
| select_related | 1 (JOIN) | Single query with JOIN |
| prefetch_related | 2 | 1 for posts + 1 for all authors |

### select_related (ForeignKey / OneToOne)

Performs a SQL JOIN -- fetches related objects in one query:

```python
# 1 query instead of N+1
posts = Post.objects.select_related("author", "category").all()
for post in posts:
    print(post.author.username)  # No extra query
```

### prefetch_related (ManyToMany / Reverse FK)

Runs a separate query per relationship, joins in Python:

```python
# 3 queries instead of N+1
posts = Post.objects.prefetch_related("tags", "comments").all()
```

### Custom Prefetch

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

Use `select_related` for FK/OneToOne (single JOIN), `prefetch_related` for M2M/reverse FK (separate query).

---

## 6. Raw SQL and Database Functions

### Raw Queries

```python
posts = Post.objects.raw("SELECT * FROM blog_post WHERE status = %s", ["published"])

from django.db import connection
with connection.cursor() as cursor:
    cursor.execute("SELECT category_id, COUNT(*) FROM blog_post GROUP BY 1")
    rows = cursor.fetchall()
```

### Database Functions

```python
from django.db.models.functions import Lower, Coalesce, TruncMonth
from django.db.models import Value, Subquery, OuterRef

Post.objects.annotate(title_lower=Lower("title"))
Post.objects.annotate(display_cat=Coalesce("category__name", Value("Uncategorized")))

# Subquery: latest comment text per post
newest = Comment.objects.filter(post=OuterRef("pk")).order_by("-created_at")
Post.objects.annotate(latest_comment=Subquery(newest.values("text")[:1]))
```

---

## 7. Custom Managers and QuerySets

Encapsulate reusable query patterns in custom QuerySets:

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
    # ... fields ...
    objects = PostQuerySet.as_manager()
```

Chainable usage:

```python
trending = (
    Post.objects.published().popular(50).with_comment_count()
    .order_by("-comment_count")[:10]
)
```

---

## 8. Bulk Operations and Transactions

For write-heavy workloads, avoid per-row saves and use Django's bulk APIs.

### bulk_create and bulk_update

```python
# bulk_create: single INSERT for many rows
posts = [Post(title=f"Post {i}", body="...", author_id=1) for i in range(1000)]
Post.objects.bulk_create(posts, batch_size=200)

# bulk_update: single UPDATE for many rows (must fetch first)
posts = list(Post.objects.filter(status="draft"))
for post in posts:
    post.status = "published"
Post.objects.bulk_update(posts, ["status"], batch_size=200)
```

`bulk_create` skips `save()` signals and per-row validation; validate explicitly before calling it.

### Atomic Transactions

Wrap related writes in `atomic()` so a failure rolls back all changes:

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

`select_for_update()` acquires a row-level lock, preventing concurrent modifications.

### iterator() for Large QuerySets

Loading millions of rows into memory crashes workers. Use `iterator()` to stream in chunks:

```python
# Without iterator(): loads all rows at once
for post in Post.objects.all():          # OOM risk at scale
    process(post)

# With iterator(): fetches chunk_size rows per DB round-trip
for post in Post.objects.all().iterator(chunk_size=500):
    process(post)
```

Do not combine `iterator()` with `prefetch_related()` — prefetching is incompatible with server-side cursors.

---

## 9. Practice Problems

### Problem 1: QuerySet Exercises

Write QuerySets for: (a) published posts from the last 30 days ordered by views, (b) posts with title containing "Django" that have at least one comment, (c) top 5 categories by published post count, (d) posts where `updated_at` is 7+ days after `published_at`.

### Problem 2: N+1 Optimization

Fix this N+1 view:

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

### Problem 3: Custom QuerySet

Create a `ProductQuerySet` with chainable methods: `available()` (stock > 0 and active), `in_price_range(min, max)`, `with_avg_rating()`, and `bestsellers(limit=10)`.

### Problem 4: Complex Aggregation

Write a monthly sales report query: total revenue per month, average order value, unique customers, and best-selling product per month (hint: Subquery).

---

## References

- [Django QuerySet API Reference](https://docs.djangoproject.com/en/5.1/ref/models/querysets/)
- [Django Aggregation Guide](https://docs.djangoproject.com/en/5.1/topics/db/aggregation/)
- [Django Database Functions](https://docs.djangoproject.com/en/5.1/ref/models/database-functions/)
- [django-debug-toolbar](https://django-debug-toolbar.readthedocs.io/) -- Inspect SQL queries

---

**Previous**: [Django Basics](./10_Django_Basics.md) | **Next**: [Django REST Framework](./12_Django_REST_Framework.md)
