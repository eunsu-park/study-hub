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

1. [QuerySet Fundamentals](#1-queryset-fundamentals)
2. [Field Lookups](#2-field-lookups)
3. [F and Q Objects](#3-f-and-q-objects)
4. [Aggregation and Annotation](#4-aggregation-and-annotation)
5. [Solving the N+1 Problem](#5-solving-the-n1-problem)
6. [Raw SQL and Database Functions](#6-raw-sql-and-database-functions)
7. [Custom Managers and QuerySets](#7-custom-managers-and-querysets)
8. [Practice Problems](#8-practice-problems)

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

## 8. Practice Problems

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
