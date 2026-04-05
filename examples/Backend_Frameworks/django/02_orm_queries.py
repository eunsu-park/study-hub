"""
Django ORM — QuerySet API, F/Q Objects, select_related/prefetch_related
Demonstrates: Django ORM query patterns for efficient database access.

This file shows query patterns as standalone examples.
In a real project, these would be in views.py, services.py, or management commands.
"""

# === QuerySet Basics ===

"""
# All published posts
posts = Post.objects.filter(status="published")

# Chaining filters (AND)
recent_posts = Post.objects.filter(
    status="published",
    created_at__gte=datetime(2024, 1, 1),
).order_by("-created_at")

# Get single object (raises DoesNotExist if not found)
post = Post.objects.get(slug="my-post")

# Get or create
category, created = Category.objects.get_or_create(
    name="Python",
    defaults={"slug": "python"},
)

# Exclude
non_draft = Post.objects.exclude(status="draft")

# Values and values_list
titles = Post.objects.values_list("title", flat=True)
post_data = Post.objects.values("title", "author__username")

# Slicing (LIMIT / OFFSET)
first_five = Post.objects.all()[:5]
page_two = Post.objects.all()[10:20]
"""

# === Field Lookups ===

"""
# Comparison
Post.objects.filter(created_at__year=2024)
Post.objects.filter(title__contains="Python")         # LIKE '%Python%'
Post.objects.filter(title__icontains="python")         # Case-insensitive
Post.objects.filter(title__startswith="How")
Post.objects.filter(id__in=[1, 2, 3])
Post.objects.filter(category__isnull=True)
Post.objects.filter(created_at__range=(start, end))

# Related field lookups (double underscore)
Post.objects.filter(author__username="alice")
Post.objects.filter(category__name="Python")
"""

# === F and Q Objects ===

"""
from django.db.models import F, Q

# F objects: reference field values in queries
# Find posts where updated_at > created_at
Post.objects.filter(updated_at__gt=F("created_at"))

# Increment a counter without race conditions
Post.objects.filter(id=1).update(view_count=F("view_count") + 1)

# Q objects: complex queries with OR / NOT
# Posts by alice OR in Python category
Post.objects.filter(
    Q(author__username="alice") | Q(category__name="Python")
)

# NOT draft AND (by alice OR recent)
Post.objects.filter(
    ~Q(status="draft"),
    Q(author__username="alice") | Q(created_at__year=2024),
)
"""

# === Aggregation ===

"""
from django.db.models import Count, Avg, Max, Min, Sum

# Single aggregate
Post.objects.aggregate(
    total=Count("id"),
    avg_length=Avg("content__length"),
)
# Returns: {"total": 42, "avg_length": 1500.5}

# Annotate (per-row aggregation)
categories = Category.objects.annotate(
    post_count=Count("posts"),
).filter(post_count__gt=0).order_by("-post_count")

# Group by author
from django.db.models import Count
author_stats = Post.objects.values("author__username").annotate(
    total=Count("id"),
).order_by("-total")
"""

# === N+1 Problem and Solutions ===

"""
# BAD: N+1 queries (1 for posts + N for each author)
posts = Post.objects.all()
for post in posts:
    print(post.author.username)  # Each access hits the DB!

# GOOD: select_related (JOIN, for ForeignKey/OneToOne)
posts = Post.objects.select_related("author", "category").all()
for post in posts:
    print(post.author.username)  # No extra queries — already loaded

# GOOD: prefetch_related (separate query, for ManyToMany/reverse FK)
categories = Category.objects.prefetch_related("posts").all()
for cat in categories:
    print(cat.posts.count())  # Uses prefetched data

# Custom prefetch with filtering
from django.db.models import Prefetch
categories = Category.objects.prefetch_related(
    Prefetch(
        "posts",
        queryset=Post.objects.filter(status="published").order_by("-created_at"),
        to_attr="published_posts",
    )
)
"""

# === Raw SQL (escape hatch) ===

"""
# Raw queries (parameterized — safe from SQL injection)
posts = Post.objects.raw(
    "SELECT * FROM blog_post WHERE title LIKE %s", ["%Python%"]
)

# Database cursor for non-model queries
from django.db import connection
with connection.cursor() as cursor:
    cursor.execute("SELECT COUNT(*) FROM blog_post WHERE status = %s", ["published"])
    count = cursor.fetchone()[0]
"""


# --- Standalone Demo ---

if __name__ == "__main__":
    print("Django ORM Query Patterns Demo")
    print("=" * 50)
    print()
    print("Key patterns demonstrated:")
    print("  1. QuerySet chaining (filter, exclude, order_by)")
    print("  2. Field lookups (__contains, __gt, __in, etc.)")
    print("  3. F objects (field references in queries)")
    print("  4. Q objects (complex OR/NOT queries)")
    print("  5. Aggregation (Count, Avg, annotate)")
    print("  6. select_related / prefetch_related (N+1 fix)")
    print("  7. Raw SQL (parameterized)")
    print()
    print("These patterns work with any Django model.")
    print("See the lesson content for detailed explanations.")
