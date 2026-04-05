#!/usr/bin/env python3
"""Example: Factory Boy for Test Data Generation

Demonstrates factory_boy patterns: basic factories, sequences, lazy attributes,
subfactories, traits, and integration with pytest for generating realistic
test data without manual fixture boilerplate.
Related lesson: 11_Test_Data_Management.md
"""

# =============================================================================
# WHY FACTORY BOY?
# Creating test data manually is tedious and brittle:
#   user = User(name="Alice", email="alice@test.com", age=30, role="admin", ...)
#
# Factory Boy provides:
#   1. Sensible defaults — create objects with minimal specification
#   2. Unique values — auto-incrementing sequences prevent collisions
#   3. Composability — subfactories build object graphs automatically
#   4. Traits — predefined configurations for common scenarios
#   5. Lazy attributes — computed fields that depend on other fields
# =============================================================================

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timedelta
from enum import Enum

try:
    import factory
    from factory import fuzzy
    FACTORY_AVAILABLE = True
except ImportError:
    FACTORY_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FACTORY_AVAILABLE,
    reason="factory_boy not installed (pip install factory-boy)"
)


# =============================================================================
# DOMAIN MODELS
# =============================================================================
# These are the models we want to generate test data for.

class UserRole(Enum):
    VIEWER = "viewer"
    EDITOR = "editor"
    ADMIN = "admin"


@dataclass
class Address:
    street: str
    city: str
    country: str
    zip_code: str


@dataclass
class User:
    id: int
    username: str
    email: str
    full_name: str
    role: UserRole = UserRole.VIEWER
    is_active: bool = True
    address: Optional[Address] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class BlogPost:
    id: int
    title: str
    content: str
    author: User
    tags: List[str] = field(default_factory=list)
    published: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    view_count: int = 0


@dataclass
class Comment:
    id: int
    post: BlogPost
    author: User
    text: str
    created_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# 1. BASIC FACTORY
# =============================================================================

class AddressFactory(factory.Factory):
    """Factory for generating Address instances.
    factory.Factory works with any class (dataclasses, attrs, plain classes).
    For Django models, use factory.django.DjangoModelFactory instead."""

    class Meta:
        model = Address

    # factory.Sequence generates unique values: street_1, street_2, ...
    # This prevents collisions when creating multiple addresses in one test.
    street = factory.Sequence(lambda n: f"{100 + n} Main Street")
    city = factory.Faker("city")              # Faker generates realistic data
    country = factory.Faker("country")
    zip_code = factory.Faker("zipcode")


class UserFactory(factory.Factory):
    """Factory for User model with sensible defaults."""

    class Meta:
        model = User

    # Sequence ensures unique IDs and usernames across all factory calls
    id = factory.Sequence(lambda n: n + 1)
    username = factory.Sequence(lambda n: f"user_{n}")

    # LazyAttribute computes a value based on OTHER fields.
    # This is better than Faker because it creates consistent relationships
    # between fields (email matches username).
    email = factory.LazyAttribute(lambda obj: f"{obj.username}@example.com")
    full_name = factory.Faker("name")
    role = UserRole.VIEWER
    is_active = True

    # SubFactory: automatically create an Address when creating a User.
    # This builds the entire object graph without manual setup.
    address = factory.SubFactory(AddressFactory)

    created_at = factory.LazyFunction(datetime.utcnow)

    # ==========================================================================
    # TRAITS — Predefined configurations for common scenarios
    # ==========================================================================
    class Params:
        admin = factory.Trait(
            role=UserRole.ADMIN,
            username=factory.Sequence(lambda n: f"admin_{n}"),
        )
        inactive = factory.Trait(
            is_active=False,
        )
        without_address = factory.Trait(
            address=None,
        )


class BlogPostFactory(factory.Factory):
    """Factory for BlogPost with nested User creation."""

    class Meta:
        model = BlogPost

    id = factory.Sequence(lambda n: n + 1)
    title = factory.Sequence(lambda n: f"Blog Post {n}: Interesting Topic")
    content = factory.Faker("paragraph", nb_sentences=5)
    author = factory.SubFactory(UserFactory)
    tags = factory.LazyFunction(lambda: ["python", "testing"])
    published = False
    created_at = factory.LazyFunction(datetime.utcnow)
    view_count = 0

    class Params:
        popular = factory.Trait(
            published=True,
            view_count=factory.fuzzy.FuzzyInteger(1000, 10000),
        )
        draft = factory.Trait(
            published=False,
            title=factory.Sequence(lambda n: f"[DRAFT] Post {n}"),
        )


class CommentFactory(factory.Factory):
    """Factory for Comment — demonstrates multi-level SubFactory."""

    class Meta:
        model = Comment

    id = factory.Sequence(lambda n: n + 1)
    post = factory.SubFactory(BlogPostFactory)
    author = factory.SubFactory(UserFactory)
    text = factory.Faker("sentence")
    created_at = factory.LazyFunction(datetime.utcnow)


# =============================================================================
# TESTS
# =============================================================================

class TestBasicFactory:
    """Demonstrate basic factory usage."""

    def test_create_default_user(self):
        """Create a user with all default values — zero configuration."""
        user = UserFactory()
        assert user.username.startswith("user_")
        assert "@example.com" in user.email
        assert user.role == UserRole.VIEWER
        assert user.is_active is True
        assert user.address is not None  # SubFactory auto-creates Address

    def test_create_with_overrides(self):
        """Override specific fields — only specify what matters for the test.
        This keeps tests focused and readable."""
        user = UserFactory(
            username="alice",
            role=UserRole.ADMIN,
        )
        assert user.username == "alice"
        assert user.email == "alice@example.com"  # LazyAttribute recomputes
        assert user.role == UserRole.ADMIN

    def test_create_multiple(self):
        """factory.create_batch generates multiple instances efficiently."""
        users = UserFactory.create_batch(5)
        assert len(users) == 5
        # All usernames are unique thanks to Sequence
        usernames = [u.username for u in users]
        assert len(set(usernames)) == 5

    def test_unique_ids(self):
        """Sequences ensure uniqueness across all factory calls in a test."""
        user1 = UserFactory()
        user2 = UserFactory()
        assert user1.id != user2.id
        assert user1.username != user2.username


class TestTraits:
    """Demonstrate trait usage for common scenarios."""

    def test_admin_trait(self):
        """Traits bundle multiple field overrides under a readable name."""
        admin = UserFactory(admin=True)
        assert admin.role == UserRole.ADMIN
        assert admin.username.startswith("admin_")

    def test_inactive_trait(self):
        user = UserFactory(inactive=True)
        assert user.is_active is False

    def test_combined_traits(self):
        """Traits can be combined for complex scenarios."""
        user = UserFactory(admin=True, inactive=True)
        assert user.role == UserRole.ADMIN
        assert user.is_active is False

    def test_without_address(self):
        """Trait can nullify SubFactory fields."""
        user = UserFactory(without_address=True)
        assert user.address is None


class TestSubFactories:
    """Demonstrate nested object creation."""

    def test_post_creates_author(self):
        """SubFactory automatically creates the nested User."""
        post = BlogPostFactory()
        assert isinstance(post.author, User)
        assert post.author.address is not None

    def test_post_with_specific_author(self):
        """Override the SubFactory to use a specific instance."""
        alice = UserFactory(username="alice")
        post = BlogPostFactory(author=alice)
        assert post.author.username == "alice"

    def test_comment_creates_full_graph(self):
        """CommentFactory creates Comment -> Post -> Author -> Address.
        The entire object graph is built automatically."""
        comment = CommentFactory()
        assert isinstance(comment.post, BlogPost)
        assert isinstance(comment.post.author, User)
        assert isinstance(comment.author, User)
        # Comment author and post author are different users
        assert comment.author.id != comment.post.author.id

    def test_override_nested_fields(self):
        """Override fields deep in the object graph using __ syntax."""
        post = BlogPostFactory(author__username="deep_override")
        assert post.author.username == "deep_override"
        assert post.author.email == "deep_override@example.com"


class TestBlogPostTraits:
    """Test BlogPost-specific traits."""

    def test_popular_post(self):
        post = BlogPostFactory(popular=True)
        assert post.published is True
        assert post.view_count >= 1000

    def test_draft_post(self):
        post = BlogPostFactory(draft=True)
        assert post.published is False
        assert "[DRAFT]" in post.title


class TestFactoryWithPytest:
    """Integration patterns with pytest fixtures."""

    @pytest.fixture
    def admin_user(self):
        """Fixture that uses factory for consistent test data."""
        return UserFactory(admin=True, username="test_admin")

    @pytest.fixture
    def published_posts(self):
        """Create a batch of published posts for list/filter tests."""
        return BlogPostFactory.create_batch(10, published=True)

    def test_admin_permissions(self, admin_user):
        """Test reads clearly: admin_user is self-documenting."""
        assert admin_user.role == UserRole.ADMIN

    def test_published_post_count(self, published_posts):
        """Batch creation makes list operation tests easy."""
        assert len(published_posts) == 10
        assert all(p.published for p in published_posts)

    def test_mixed_scenario(self):
        """Combine factories for complex test scenarios."""
        alice = UserFactory(username="alice", admin=True)
        bob = UserFactory(username="bob")

        # Alice writes a popular post
        post = BlogPostFactory(author=alice, popular=True)

        # Bob comments on it
        comment = CommentFactory(post=post, author=bob, text="Great post!")

        assert comment.post.author.username == "alice"
        assert comment.author.username == "bob"
        assert comment.text == "Great post!"
        assert post.view_count >= 1000


# =============================================================================
# BEST PRACTICES
# =============================================================================

class TestBestPractices:
    def test_only_specify_relevant_fields(self):
        """GOOD: Only override fields relevant to the test.
        Default values for irrelevant fields reduce noise."""
        # Testing email format — only username matters
        user = UserFactory(username="special_user")
        assert user.email == "special_user@example.com"
        # Don't assert on full_name, address, etc. — they're irrelevant

    def test_use_traits_over_manual_overrides(self):
        """GOOD: Traits are self-documenting and reusable.
        BAD: UserFactory(role=UserRole.ADMIN, is_active=False)
        GOOD: UserFactory(admin=True, inactive=True)"""
        user = UserFactory(admin=True, inactive=True)
        assert user.role == UserRole.ADMIN

    def test_factory_per_model(self):
        """GOOD: One factory per model, kept near the model or in conftest.py.
        This prevents factory definitions from being scattered across tests."""
        # All factories are defined at module level, not inside tests
        user = UserFactory()
        post = BlogPostFactory(author=user)
        assert post.author is user


# =============================================================================
# RUNNING THIS FILE
# =============================================================================
# pip install factory-boy faker pytest
# pytest 08_factory_boy_example.py -v

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
