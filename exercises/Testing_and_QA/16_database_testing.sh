#!/bin/bash
# Exercises for Lesson 16: Database Testing
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Transaction Rollback Fixture ===
# Problem: Implement the full transaction rollback fixture for a
# SQLAlchemy project. Write three tests that create data and verify
# that each test starts with a clean database.
exercise_1() {
    echo "=== Exercise 1: Transaction Rollback Fixture ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from sqlalchemy import create_engine, Column, Integer, String, event
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False)

# --- Fixtures ---

@pytest.fixture(scope="session")
def engine():
    """Create an in-memory SQLite engine for the test session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)

@pytest.fixture(scope="function")
def db_session(engine):
    """
    Transactional fixture: each test runs inside a transaction
    that is rolled back after the test completes.
    """
    connection = engine.connect()
    transaction = connection.begin()

    Session = sessionmaker(bind=connection)
    session = Session()

    # Restart nested transactions when the app calls session.commit()
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, trans):
        if trans.nested and not trans._parent.nested:
            session.begin_nested()

    session.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

# --- Tests ---

def test_create_user(db_session):
    """Insert a user — visible within this test."""
    user = User(name="Alice", email="alice@example.com")
    db_session.add(user)
    db_session.flush()

    found = db_session.query(User).filter_by(email="alice@example.com").first()
    assert found is not None
    assert found.name == "Alice"

def test_database_is_clean(db_session):
    """Rollback ensures no data from previous test."""
    users = db_session.query(User).all()
    assert len(users) == 0

def test_create_different_user(db_session):
    """Another insertion — also isolated."""
    user = User(name="Bob", email="bob@example.com")
    db_session.add(user)
    db_session.flush()

    count = db_session.query(User).count()
    assert count == 1  # Only Bob, no Alice
SOLUTION
}

# === Exercise 2: Factory Design ===
# Problem: Create factories for a blog application with User, Post,
# Tag, and Comment models. Include traits for admin, draft_post,
# and spam_comment. Write tests demonstrating each trait.
exercise_2() {
    echo "=== Exercise 2: Factory Design ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
import factory
from factory.alchemy import SQLAlchemyModelFactory
from sqlalchemy import (
    create_engine, Column, Integer, String, Text,
    Boolean, ForeignKey, Table
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

Base = declarative_base()

# --- Models ---

post_tags = Table(
    "post_tags", Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id")),
    Column("tag_id", Integer, ForeignKey("tags.id")),
)

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True)
    role = Column(String(20), default="user")
    posts = relationship("Post", back_populates="author")

class Post(Base):
    __tablename__ = "posts"
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False)
    body = Column(Text)
    published = Column(Boolean, default=True)
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")
    tags = relationship("Tag", secondary=post_tags)

class Tag(Base):
    __tablename__ = "tags"
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)

class Comment(Base):
    __tablename__ = "comments"
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    is_spam = Column(Boolean, default=False)
    post_id = Column(Integer, ForeignKey("posts.id"))
    author_id = Column(Integer, ForeignKey("users.id"))

# --- Factories ---

class UserFactory(SQLAlchemyModelFactory):
    class Meta:
        model = User
        sqlalchemy_session = None

    name = factory.Faker("name")
    email = factory.Sequence(lambda n: f"user{n}@example.com")
    role = "user"

    class Params:
        admin = factory.Trait(role="admin")

class PostFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Post
        sqlalchemy_session = None

    title = factory.Faker("sentence", nb_words=6)
    body = factory.Faker("paragraph")
    published = True
    author = factory.SubFactory(UserFactory)

    class Params:
        draft = factory.Trait(published=False)

class TagFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Tag
        sqlalchemy_session = None

    name = factory.Sequence(lambda n: f"tag-{n}")

class CommentFactory(SQLAlchemyModelFactory):
    class Meta:
        model = Comment
        sqlalchemy_session = None

    text = factory.Faker("sentence")
    is_spam = False
    post = factory.SubFactory(PostFactory)
    author = factory.SubFactory(UserFactory)

    class Params:
        spam = factory.Trait(
            is_spam=True,
            text=factory.LazyFunction(lambda: "Buy cheap stuff at http://spam.example")
        )

# --- Tests ---

@pytest.fixture(scope="session")
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng

@pytest.fixture(autouse=True)
def session(engine):
    Session = sessionmaker(bind=engine)
    session = Session()
    for F in [UserFactory, PostFactory, TagFactory, CommentFactory]:
        F._meta.sqlalchemy_session = session
    yield session
    session.rollback()
    session.close()

def test_admin_trait():
    admin = UserFactory(admin=True)
    assert admin.role == "admin"

def test_default_user_is_not_admin():
    user = UserFactory()
    assert user.role == "user"

def test_draft_post_trait():
    post = PostFactory(draft=True)
    assert post.published is False

def test_published_post_default():
    post = PostFactory()
    assert post.published is True

def test_spam_comment_trait():
    comment = CommentFactory(spam=True)
    assert comment.is_spam is True
    assert "spam" in comment.text.lower() or "http" in comment.text.lower()

def test_comment_creates_post_and_author():
    comment = CommentFactory()
    assert comment.post is not None
    assert comment.author is not None
    assert comment.is_spam is False
SOLUTION
}

# === Exercise 3: Migration Testing ===
# Problem: Write a test that applies all Alembic migrations from base
# to head, verifies the schema matches the expected state, then
# downgrades back to base.
exercise_3() {
    echo "=== Exercise 3: Migration Testing ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

@pytest.fixture(scope="module")
def alembic_config(tmp_path_factory):
    """Create an Alembic config pointing to a test database."""
    db_path = tmp_path_factory.mktemp("db") / "test.db"
    config = Config("alembic.ini")
    config.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return config

@pytest.fixture(scope="module")
def migration_engine(alembic_config):
    url = alembic_config.get_main_option("sqlalchemy.url")
    engine = create_engine(url)
    yield engine
    engine.dispose()

def test_upgrade_to_head(alembic_config, migration_engine):
    """Apply all migrations from base to head."""
    command.upgrade(alembic_config, "head")

    inspector = inspect(migration_engine)
    tables = inspector.get_table_names()

    # Verify core tables exist
    assert "alembic_version" in tables
    assert "users" in tables
    assert "posts" in tables

    # Verify columns on the users table
    columns = {c["name"] for c in inspector.get_columns("users")}
    assert "id" in columns
    assert "name" in columns
    assert "email" in columns

def test_downgrade_to_base(alembic_config, migration_engine):
    """Roll back all migrations to base."""
    command.downgrade(alembic_config, "base")

    inspector = inspect(migration_engine)
    tables = inspector.get_table_names()

    # Only alembic_version may remain (or nothing)
    assert "users" not in tables
    assert "posts" not in tables

def test_upgrade_again_after_downgrade(alembic_config, migration_engine):
    """Re-apply migrations — proves migrations are fully reversible."""
    command.upgrade(alembic_config, "head")

    inspector = inspect(migration_engine)
    tables = inspector.get_table_names()
    assert "users" in tables
    assert "posts" in tables
SOLUTION
}

# === Exercise 4: Constraint Coverage ===
# Problem: For a given set of database models, identify all constraints
# (unique, not-null, foreign key, check) and write a test for each one,
# verifying that the constraint is actually enforced.
exercise_4() {
    echo "=== Exercise 4: Constraint Coverage ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    ForeignKey, CheckConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from sqlalchemy.exc import IntegrityError

Base = declarative_base()

class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("price >= 0", name="ck_price_non_negative"),
        CheckConstraint("stock >= 0", name="ck_stock_non_negative"),
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    sku = Column(String(50), unique=True, nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, default=0)

class OrderItem(Base):
    __tablename__ = "order_items"

    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    quantity = Column(Integer, nullable=False)

@pytest.fixture(scope="module")
def engine():
    engine = create_engine("sqlite:///:memory:")
    # Enable FK enforcement for SQLite
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_fk_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    Base.metadata.create_all(engine)
    yield engine

@pytest.fixture
def session(engine):
    Session = sessionmaker(bind=engine)
    s = Session()
    yield s
    s.rollback()
    s.close()

# --- NOT NULL constraints ---

def test_product_name_not_null(session):
    with pytest.raises(IntegrityError):
        p = Product(name=None, sku="SKU001", price=9.99)
        session.add(p)
        session.flush()

def test_product_sku_not_null(session):
    with pytest.raises(IntegrityError):
        p = Product(name="Widget", sku=None, price=9.99)
        session.add(p)
        session.flush()

# --- UNIQUE constraint ---

def test_duplicate_sku_rejected(session):
    p1 = Product(name="Widget", sku="UNIQUE001", price=9.99)
    session.add(p1)
    session.flush()

    with pytest.raises(IntegrityError):
        p2 = Product(name="Gadget", sku="UNIQUE001", price=19.99)
        session.add(p2)
        session.flush()

# --- FOREIGN KEY constraint ---

def test_order_item_requires_valid_product(session):
    with pytest.raises(IntegrityError):
        item = OrderItem(product_id=99999, quantity=1)
        session.add(item)
        session.flush()

# --- CHECK constraints ---
# Note: SQLite enforces CHECK constraints only if compiled with
# SQLITE_ENABLE_CHECK_CONSTRAINTS or version >= 3.25. These tests
# show the pattern; use PostgreSQL in CI for full enforcement.

def test_price_must_be_non_negative(session):
    """CHECK constraint: price >= 0."""
    with pytest.raises(IntegrityError):
        p = Product(name="Free?", sku="NEG001", price=-5.00)
        session.add(p)
        session.flush()
SOLUTION
}

# === Exercise 5: Performance Baseline ===
# Problem: Create a factory that generates 10,000 users, then write
# benchmark tests for three common queries. Set maximum acceptable
# execution times.
exercise_5() {
    echo "=== Exercise 5: Performance Baseline ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import time

import pytest
from sqlalchemy import create_engine, Column, Integer, String, func
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    email = Column(String(200), unique=True, nullable=False)
    role = Column(String(20), default="user")

@pytest.fixture(scope="module")
def engine():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return engine

@pytest.fixture(scope="module")
def seeded_session(engine):
    """Insert 10,000 users once for the entire module."""
    Session = sessionmaker(bind=engine)
    session = Session()

    users = []
    for i in range(10_000):
        role = "admin" if i % 100 == 0 else "user"
        name = f"User {i}" if i % 7 != 0 else f"Smith {i}"
        users.append(User(name=name, email=f"user{i}@example.com", role=role))

    session.bulk_save_objects(users)
    session.commit()
    yield session
    session.close()

@pytest.mark.slow
def test_search_by_email_performance(seeded_session):
    """Benchmark: lookup by email (expected to use unique index)."""
    start = time.perf_counter()
    user = seeded_session.query(User).filter_by(
        email="user5000@example.com"
    ).first()
    elapsed = time.perf_counter() - start

    assert user is not None
    assert elapsed < 0.1, f"Email lookup took {elapsed:.4f}s (limit: 0.1s)"

@pytest.mark.slow
def test_list_by_name_pattern_performance(seeded_session):
    """Benchmark: LIKE query on name column."""
    start = time.perf_counter()
    results = seeded_session.query(User).filter(
        User.name.like("Smith%")
    ).all()
    elapsed = time.perf_counter() - start

    assert len(results) > 0
    assert elapsed < 0.5, f"Name search took {elapsed:.4f}s (limit: 0.5s)"

@pytest.mark.slow
def test_count_by_role_performance(seeded_session):
    """Benchmark: GROUP BY role count."""
    start = time.perf_counter()
    counts = seeded_session.query(
        User.role, func.count(User.id)
    ).group_by(User.role).all()
    elapsed = time.perf_counter() - start

    role_map = dict(counts)
    assert role_map["admin"] == 100   # Every 100th user
    assert role_map["user"] == 9900
    assert elapsed < 0.5, f"Role count took {elapsed:.4f}s (limit: 0.5s)"
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 16: Database Testing"
echo "====================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
echo ""
exercise_5
