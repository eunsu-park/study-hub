# Exercise: Django Models and ORM
# Practice with model design, QuerySets, and F/Q objects.

# Exercise 1: Design a Library Database
# Create models for: Library, Book, Author, Borrowing
# Requirements:
# - Book has many-to-many with Author
# - Book belongs to one Library
# - Borrowing tracks who borrowed which book and when
# - Use proper field types, indexes, and constraints

"""
from django.db import models

class Author(models.Model):
    # TODO: Define fields (name, bio, birth_date)
    pass

class Library(models.Model):
    # TODO: Define fields (name, address, city)
    pass

class Book(models.Model):
    # TODO: Define fields (title, isbn, published_date, authors M2M, library FK)
    pass

class Borrowing(models.Model):
    # TODO: Define fields (book FK, borrower_name, borrowed_at, returned_at nullable)
    pass
"""


# Exercise 2: QuerySet Practice
# Write these queries (pseudo-code — would run in Django shell)

"""
# a) Find all books by an author named "Alice" published after 2020
# TODO: Book.objects.filter(...)

# b) Find libraries that have more than 100 books
# TODO: Library.objects.annotate(...).filter(...)

# c) Find books that are currently borrowed (returned_at is null)
# TODO: Book.objects.filter(...)

# d) Count books per library, ordered by count descending
# TODO: Library.objects.annotate(...).order_by(...)

# e) Find the most recent borrowing for each book
# TODO: Use Subquery and OuterRef
"""


# Exercise 3: F and Q Objects
# Write complex queries using F and Q

"""
from django.db.models import F, Q

# a) Books where title length > author count (conceptual)
# TODO: Book.objects.annotate(...).filter(...)

# b) Books that are either:
#    - Published before 2000 AND by author "Classic Author"
#    - OR published after 2020
# TODO: Book.objects.filter(Q(...) | Q(...))

# c) Update all borrowings: if returned_at is null and borrowed_at
#    was more than 30 days ago, mark as overdue (add an overdue field)
# TODO: Borrowing.objects.filter(...).update(...)
"""


# Exercise 4: Optimize N+1
# Fix the N+1 problem in this code:

"""
# BAD: N+1 queries
def get_library_report():
    libraries = Library.objects.all()
    report = []
    for lib in libraries:
        books = lib.book_set.all()
        for book in books:
            authors = book.authors.all()
            report.append({
                "library": lib.name,
                "book": book.title,
                "authors": [a.name for a in authors],
            })
    return report

# TODO: Rewrite using select_related and/or prefetch_related
# to minimize database queries.
"""


# Exercise 5: Custom Manager and QuerySet
# Create a custom manager for Book that provides:
# - Book.objects.available() — not currently borrowed
# - Book.objects.by_author(name) — filter by author name
# - Book.objects.recent(days=30) — published in last N days

"""
class BookQuerySet(models.QuerySet):
    # TODO: Implement available(), by_author(), recent()
    pass

class BookManager(models.Manager):
    def get_queryset(self):
        return BookQuerySet(self.model, using=self._db)

    # TODO: Delegate methods
"""


if __name__ == "__main__":
    print("Django ORM Exercise")
    print("These exercises are meant to be implemented in a Django project.")
    print("Create the models, run migrations, then test in Django shell.")
