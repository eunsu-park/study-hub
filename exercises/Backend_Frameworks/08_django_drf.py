# Exercise: Django REST Framework
# Practice with serializers, viewsets, and API design.

# Exercise 1: Nested Serializer
# Create serializers for a blog with nested comments.
"""
class CommentSerializer(serializers.ModelSerializer):
    # TODO: Fields: id, author, content, created_at
    pass

class PostSerializer(serializers.ModelSerializer):
    comments = CommentSerializer(many=True, read_only=True)
    comment_count = serializers.SerializerMethodField()

    # TODO: Implement get_comment_count
    # TODO: Custom validate_title (min 5 chars)
    pass
"""


# Exercise 2: ViewSet with Custom Actions
# Create a PostViewSet with:
# - Standard CRUD
# - POST /posts/{id}/publish/ — change status to published
# - POST /posts/{id}/archive/ — change status to archived
# - GET /posts/stats/ — return count by status

"""
class PostViewSet(viewsets.ModelViewSet):
    # TODO: Implement with custom actions using @action decorator
    pass
"""


# Exercise 3: Permission Classes
# Create custom permission classes:
# - IsAuthorOrReadOnly: only the author can edit/delete
# - IsVerifiedUser: only users with is_verified=True can create

"""
from rest_framework.permissions import BasePermission

class IsAuthorOrReadOnly(BasePermission):
    # TODO: Implement has_object_permission
    pass

class IsVerifiedUser(BasePermission):
    # TODO: Implement has_permission
    pass
"""


# Exercise 4: Pagination
# Implement cursor-based pagination for posts (ordered by created_at).
"""
from rest_framework.pagination import CursorPagination

class PostCursorPagination(CursorPagination):
    # TODO: Configure page_size, ordering, cursor_query_param
    pass
"""


# Exercise 5: Filter and Search
# Add filtering capabilities:
# GET /posts?status=published&author=alice&search=python&ordering=-created_at
"""
# TODO: Configure filterset_fields, search_fields, ordering_fields
# on the ViewSet
"""


if __name__ == "__main__":
    print("Django REST Framework Exercise")
    print("Implement in a Django project with DRF installed.")
