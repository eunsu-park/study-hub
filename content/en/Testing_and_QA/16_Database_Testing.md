# Lesson 16: Database Testing

**Previous**: [Testing Async Code](./15_Testing_Async_Code.md) | **Next**: [Testing Legacy Code](./17_Testing_Legacy_Code.md)

---

Database code is notoriously difficult to test well. It sits at the boundary between your application logic and an external system with its own rules — SQL dialects, transaction semantics, constraint enforcement, migration ordering. Many teams either skip database testing entirely (and suffer production data corruption) or write brittle tests that are slow, tightly coupled to schema details, and break with every migration. This lesson shows how to test database interactions effectively using SQLAlchemy, factory_boy, and Alembic, balancing speed, realism, and maintainability.

**Difficulty**: ⭐⭐⭐⭐

**Prerequisites**:
- Python testing with pytest (Lessons 02–04)
- Basic SQL and relational database concepts
- Familiarity with SQLAlchemy ORM (basic models, sessions, queries)
- Understanding of database transactions

## Learning Objectives

After completing this lesson, you will be able to:

1. Choose between in-memory SQLite and a real test database for different testing scenarios
2. Implement transaction rollback per test for fast, isolated database tests
3. Use factory_boy to create realistic test data without boilerplate
4. Generate realistic fake data with Faker
5. Test database migrations with Alembic
6. Write data integrity tests that verify constraints and referential integrity

---

## 1. Database Testing Strategies

There are three main strategies for testing database code, each with different tradeoffs:

| Strategy | Speed | Realism | Isolation | Best For |
|---|---|---|---|---|
| In-memory SQLite | Very fast | Low | High | Unit tests, simple queries |
| Transaction rollback | Fast | High | High | Integration tests |
| Dedicated test database | Moderate | Highest | Moderate | Full integration, migrations |

### 1.1 In-Memory SQLite

Replace your production database with an in-memory SQLite instance for testing:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from myapp.models import Base


# Production: PostgreSQL
# PROD_ENGINE = create_engine("postgresql://user:pass@localhost/mydb")

# Testing: In-memory SQLite
TEST_ENGINE = create_engine("sqlite:///:memory:")


@pytest.fixture(scope="function")
def db_session():
    """Create tables, provide a session, then tear down."""
    Base.metadata.create_all(TEST_ENGINE)
    Session = sessionmaker(bind=TEST_ENGINE)
    session = Session()

    yield session

    session.close()
    Base.metadata.drop_all(TEST_ENGINE)
```

**Advantages**: Extremely fast, no external dependencies, perfect isolation.

**Limitations**: SQLite does not support all PostgreSQL features (arrays, JSON operators, CTEs with certain syntax, `ENUM` types, advanced constraint syntax). Tests may pass on SQLite but fail on PostgreSQL.

### 1.2 When SQLite Differs from PostgreSQL

```python
from sqlalchemy import Column, Integer, String, Enum
from sqlalchemy.dialects.postgresql import ARRAY, JSONB

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    tags = Column(ARRAY(String))           # PostgreSQL-only
    metadata_ = Column(JSONB)              # PostgreSQL JSONB
    role = Column(Enum("admin", "user"))   # Different behavior

# These columns won't work with SQLite.
# Strategy: use a real PostgreSQL test database for models with
# PostgreSQL-specific features.
```

---

## 2. Transaction Rollback Per Test

The most practical strategy for integration tests: start a transaction before each test and roll it back after. The database state is never actually changed, so tests are fast and isolated.

### 2.1 Implementation with SQLAlchemy

```python
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from myapp.models import Base


@pytest.fixture(scope="session")
def engine():
    """Create the test database engine once per test session."""
    engine = create_engine("postgresql://test:test@localhost/testdb")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)


@pytest.fixture(scope="function")
def db_session(engine):
    """
    Provide a transactional database session that rolls back after each test.

    This uses a nested transaction pattern:
    1. Outer connection with a real transaction
    2. Session binds to this connection
    3. After the test, the transaction is rolled back
    """
    connection = engine.connect()
    transaction = connection.begin()

    # Bind a session to the connection (not the engine)
    Session = sessionmaker(bind=connection)
    session = Session()

    # If the application code calls session.commit(), we need to
    # intercept and restart the nested transaction
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, trans):
        if trans.nested and not trans._parent.nested:
            session.begin_nested()

    session.begin_nested()  # Start a SAVEPOINT

    yield session

    session.close()
    transaction.rollback()  # Roll back the outer transaction
    connection.close()
```

### 2.2 Using the Transactional Fixture

```python
from myapp.models import User
from myapp.services import UserService


def test_create_user(db_session):
    service = UserService(db_session)
    user = service.create_user("alice", "alice@example.com")

    assert user.id is not None
    assert user.name == "alice"

    # This user is visible within the test
    found = db_session.query(User).filter_by(email="alice@example.com").first()
    assert found is not None


def test_database_is_clean(db_session):
    """This test runs after the previous one — database is clean."""
    users = db_session.query(User).all()
    assert len(users) == 0  # Rollback ensured isolation
```

---

## 3. Test Data Factories with factory_boy

Building test data by hand is tedious and couples tests to the model's constructor signature. [factory_boy](https://factoryboy.readthedocs.io/) generates test objects with sensible defaults and customizable overrides.

### 3.1 Installation

```bash
pip install factory_boy
```

### 3.2 Defining Factories

```python
import factory
from factory.alchemy import SQLAlchemyModelFactory

from myapp.models import User, Post, Comment


class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = None  # Set per-test via fixture

    name = factory.Faker("name")
    email = factory.LazyAttribute(lambda obj: f"{obj.name.lower().replace(' ', '.')}@example.com")
    age = factory.Faker("random_int", min=18, max=80)
    is_active = True
    created_at = factory.Faker("date_time_this_year")


class PostFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Post
        sqlalchemy_session = None

    title = factory.Faker("sentence", nb_words=6)
    body = factory.Faker("paragraph", nb_sentences=5)
    author = factory.SubFactory(UserFactory)  # Creates a User automatically
    published = True


class CommentFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Comment
        sqlalchemy_session = None

    text = factory.Faker("sentence")
    post = factory.SubFactory(PostFactory)
    author = factory.SubFactory(UserFactory)
```

### 3.3 Wiring Factories to the Test Session

```python
@pytest.fixture(autouse=True)
def set_factory_session(db_session):
    """Wire all factories to the current test session."""
    UserFactory._meta.sqlalchemy_session = db_session
    PostFactory._meta.sqlalchemy_session = db_session
    CommentFactory._meta.sqlalchemy_session = db_session
    yield
    # Cleanup is handled by db_session rollback
```

### 3.4 Using Factories in Tests

```python
def test_user_has_default_values():
    user = UserFactory()
    assert user.name is not None
    assert "@" in user.email
    assert user.is_active is True


def test_user_with_specific_name():
    user = UserFactory(name="Bob Smith")
    assert user.name == "Bob Smith"
    assert user.email == "bob.smith@example.com"


def test_post_creates_author_automatically():
    post = PostFactory()
    assert post.author is not None
    assert post.author.id is not None


def test_post_with_specific_author():
    author = UserFactory(name="Alice")
    post = PostFactory(author=author)
    assert post.author.name == "Alice"


def test_batch_creation():
    users = UserFactory.create_batch(10)
    assert len(users) == 10
    assert len(set(u.email for u in users)) == 10  # All unique
```

### 3.5 Factory Traits and Sequences

```python
class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = None

    name = factory.Faker("name")
    email = factory.Sequence(lambda n: f"user{n}@example.com")  # Guaranteed unique
    is_active = True
    role = "user"

    class Params:
        admin = factory.Trait(
            role="admin",
            email=factory.LazyAttribute(lambda obj: f"admin.{obj.name.lower()}@example.com")
        )
        inactive = factory.Trait(is_active=False)


# Usage
admin = UserFactory(admin=True)
assert admin.role == "admin"

inactive = UserFactory(inactive=True)
assert inactive.is_active is False

inactive_admin = UserFactory(admin=True, inactive=True)
assert inactive_admin.role == "admin"
assert inactive_admin.is_active is False
```

---

## 4. Generating Realistic Data with Faker

[Faker](https://faker.readthedocs.io/) generates realistic-looking fake data for tests. factory_boy uses it internally, but you can also use it directly.

### 4.1 Basic Usage

```python
from faker import Faker

fake = Faker()

# Personal data
fake.name()          # "John Smith"
fake.email()         # "john.smith@example.com"
fake.phone_number()  # "+1-555-123-4567"
fake.address()       # "123 Main St, Springfield, IL 62704"

# Internet data
fake.url()           # "https://example.com/path"
fake.ipv4()          # "192.168.1.100"
fake.user_agent()    # "Mozilla/5.0 ..."

# Business data
fake.company()       # "Smith LLC"
fake.job()           # "Software Engineer"
fake.credit_card_number()  # "4111111111111111"

# Text
fake.sentence()      # "The quick brown fox jumps."
fake.paragraph()     # Several sentences
fake.text(200)       # 200 characters of text
```

### 4.2 Deterministic Data with Seeds

```python
Faker.seed(12345)
fake = Faker()

# Always produces the same sequence
name1 = fake.name()  # Always "John Smith" (with this seed)
name2 = fake.name()  # Always "Jane Doe" (with this seed)
```

### 4.3 Localized Data

```python
# Generate data in specific locales
fake_jp = Faker("ja_JP")
fake_jp.name()     # "田中 太郎"
fake_jp.address()  # Japanese address

fake_de = Faker("de_DE")
fake_de.name()     # "Hans Mueller"
```

---

## 5. Testing Database Migrations

Database schema changes (migrations) are among the highest-risk operations in software. A migration that works in development can fail in production due to data dependencies, column sizes, or constraint violations.

### 5.1 Alembic Migration Testing Strategy

```python
# tests/test_migrations.py
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect


@pytest.fixture(scope="session")
def alembic_config():
    config = Config("alembic.ini")
    config.set_main_option(
        "sqlalchemy.url",
        "postgresql://test:test@localhost/test_migrations"
    )
    return config


@pytest.fixture(scope="session")
def migration_engine():
    engine = create_engine("postgresql://test:test@localhost/test_migrations")
    yield engine
    engine.dispose()


def test_migrations_run_to_head(alembic_config, migration_engine):
    """Verify all migrations can be applied from scratch."""
    # Start fresh
    from myapp.models import Base
    Base.metadata.drop_all(migration_engine)

    # Run all migrations
    command.upgrade(alembic_config, "head")

    # Verify expected tables exist
    inspector = inspect(migration_engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "posts" in tables
    assert "alembic_version" in tables


def test_migrations_are_reversible(alembic_config):
    """Verify all migrations can be rolled back."""
    command.upgrade(alembic_config, "head")
    command.downgrade(alembic_config, "base")
    command.upgrade(alembic_config, "head")  # Should work again
```

### 5.2 Testing Individual Migrations

```python
def test_add_email_column_migration(alembic_config, migration_engine):
    """Test a specific migration that adds an email column."""
    # Apply migrations up to the one before our target
    command.upgrade(alembic_config, "abc123")  # revision before

    # Insert test data (represents existing production data)
    with migration_engine.connect() as conn:
        conn.execute(
            "INSERT INTO users (name) VALUES ('alice'), ('bob')"
        )
        conn.commit()

    # Apply the migration under test
    command.upgrade(alembic_config, "def456")  # our migration

    # Verify the migration handled existing data correctly
    with migration_engine.connect() as conn:
        result = conn.execute("SELECT email FROM users")
        emails = [row.email for row in result]
        # Email should be NULL for existing rows (or default value)
        assert all(e is None for e in emails)

    inspector = inspect(migration_engine)
    columns = {c["name"] for c in inspector.get_columns("users")}
    assert "email" in columns
```

### 5.3 Testing Data Migrations

```python
def test_data_migration_normalizes_phone_numbers(alembic_config, migration_engine):
    """Test that a data migration correctly transforms existing data."""
    command.upgrade(alembic_config, "before_phone_migration")

    # Insert data in the old format
    with migration_engine.connect() as conn:
        conn.execute("""
            INSERT INTO contacts (name, phone) VALUES
            ('alice', '555-123-4567'),
            ('bob', '(555) 987-6543'),
            ('carol', '+1 555 111 2222')
        """)
        conn.commit()

    # Run the data migration
    command.upgrade(alembic_config, "phone_migration")

    # Verify normalization
    with migration_engine.connect() as conn:
        result = conn.execute("SELECT phone FROM contacts ORDER BY name")
        phones = [row.phone for row in result]
        assert phones == ["+15551234567", "+15559876543", "+15551112222"]
```

---

## 6. Data Integrity Tests

Data integrity tests verify that your database constraints, triggers, and application logic maintain data consistency.

### 6.1 Testing Constraints

```python
import pytest
from sqlalchemy.exc import IntegrityError


def test_unique_email_constraint(db_session):
    """Verify that duplicate emails are rejected."""
    UserFactory(email="alice@example.com")
    db_session.flush()

    with pytest.raises(IntegrityError):
        UserFactory(email="alice@example.com")
        db_session.flush()


def test_not_null_constraint(db_session):
    """Verify that name cannot be NULL."""
    with pytest.raises(IntegrityError):
        user = User(name=None, email="test@example.com")
        db_session.add(user)
        db_session.flush()


def test_foreign_key_constraint(db_session):
    """Verify that posts must reference an existing user."""
    with pytest.raises(IntegrityError):
        post = Post(title="Test", body="Body", author_id=99999)
        db_session.add(post)
        db_session.flush()
```

### 6.2 Testing Cascade Behavior

```python
def test_deleting_user_cascades_to_posts(db_session):
    """Verify that deleting a user also deletes their posts."""
    author = UserFactory()
    PostFactory.create_batch(3, author=author)
    db_session.flush()

    author_id = author.id
    db_session.delete(author)
    db_session.flush()

    remaining_posts = db_session.query(Post).filter_by(author_id=author_id).all()
    assert len(remaining_posts) == 0


def test_deleting_user_nullifies_comments(db_session):
    """Verify that deleting a user sets comment.author_id to NULL."""
    commenter = UserFactory()
    comment = CommentFactory(author=commenter)
    db_session.flush()

    comment_id = comment.id
    db_session.delete(commenter)
    db_session.flush()

    updated_comment = db_session.query(Comment).get(comment_id)
    assert updated_comment is not None  # Comment still exists
    assert updated_comment.author_id is None  # But author is nullified
```

### 6.3 Testing Check Constraints

```python
def test_age_must_be_positive(db_session):
    """Verify the CHECK constraint on age."""
    with pytest.raises(IntegrityError):
        UserFactory(age=-1)
        db_session.flush()


def test_price_must_be_non_negative(db_session):
    """Verify the CHECK constraint on price."""
    with pytest.raises(IntegrityError):
        product = Product(name="Widget", price=-10.00)
        db_session.add(product)
        db_session.flush()
```

---

## 7. Testing Query Performance

```python
import time

import pytest


@pytest.mark.slow
def test_user_search_is_fast(db_session):
    """Verify that user search completes within acceptable time."""
    # Create a realistic amount of data
    UserFactory.create_batch(1000)
    db_session.flush()

    start = time.perf_counter()
    results = db_session.query(User).filter(
        User.name.ilike("%smith%")
    ).all()
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"Search took {elapsed:.3f}s (limit: 0.5s)"


def test_query_uses_index(db_session, engine):
    """Verify that a query uses the expected index."""
    from sqlalchemy import text

    result = db_session.execute(text(
        "EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'test@example.com'"
    ))
    plan = "\n".join(row[0] for row in result)

    assert "Index Scan" in plan or "Index Only Scan" in plan, (
        f"Expected index scan, but got:\n{plan}"
    )
```

---

## 8. Best Practices Summary

| Practice | Why |
|---|---|
| Use transaction rollback | Fast, isolated, realistic |
| Use factories, not raw constructors | Maintainable, readable test data |
| Test constraints explicitly | Catch missing constraints early |
| Test migrations forward and backward | Prevent broken deployments |
| Seed with Faker for realism | Find encoding, length, and format bugs |
| Keep SQLite for simple tests | Maximum speed for logic that does not use DB features |
| Use a real DB for integration tests | Catch dialect-specific issues |

---

## Exercises

1. **Transaction Rollback Fixture**: Implement the full transaction rollback fixture for a SQLAlchemy project. Write three tests that create data and verify that each test starts with a clean database.

2. **Factory Design**: Create factories for a blog application with `User`, `Post`, `Tag`, and `Comment` models. Include traits for `admin`, `draft_post`, and `spam_comment`. Write tests demonstrating each trait.

3. **Migration Testing**: Write a test that applies all Alembic migrations from base to head, verifies the schema matches the expected state, then downgrades back to base.

4. **Constraint Coverage**: For a given set of database models, identify all constraints (unique, not-null, foreign key, check) and write a test for each one, verifying that the constraint is actually enforced.

5. **Performance Baseline**: Create a factory that generates 10,000 users, then write benchmark tests for three common queries (search by email, list by creation date, count by role). Set maximum acceptable execution times.

---

**License**: CC BY-NC 4.0
