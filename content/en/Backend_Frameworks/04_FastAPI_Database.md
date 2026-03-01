# 04. FastAPI Database Integration

**Previous**: [FastAPI Advanced](./03_FastAPI_Advanced.md) | **Next**: [FastAPI Testing](./05_FastAPI_Testing.md)

**Difficulty**: ⭐⭐⭐

---

## Learning Objectives

- Configure SQLAlchemy 2.0 with async support for use with FastAPI
- Define database models using the modern `mapped_column` declarative style
- Implement dependency-injected database sessions with proper lifecycle management
- Build async CRUD operations with relationships and eager/lazy loading strategies
- Set up and run Alembic migrations for schema evolution in production

---

## Table of Contents

1. [SQLAlchemy 2.0 with Async Support](#1-sqlalchemy-20-with-async-support)
2. [Defining Models with mapped_column](#2-defining-models-with-mapped_column)
3. [Database Session Management](#3-database-session-management)
4. [Async CRUD Operations](#4-async-crud-operations)
5. [Relationships](#5-relationships)
6. [Alembic for Migrations](#6-alembic-for-migrations)
7. [Connection Pooling](#7-connection-pooling)
8. [Practice Problems](#8-practice-problems)
9. [References](#9-references)

---

## 1. SQLAlchemy 2.0 with Async Support

SQLAlchemy 2.0 introduced a new API that embraces Python type hints and native async/await. Combined with FastAPI, it provides a fully asynchronous database stack.

### Installation

```bash
# Core SQLAlchemy + async PostgreSQL driver
pip install sqlalchemy[asyncio] asyncpg

# For SQLite async (good for development/testing)
pip install aiosqlite

# Alembic for migrations
pip install alembic
```

### Architecture

```
┌──────────────────────────────────────────┐
│  FastAPI Endpoint                         │
│  async def create_user(db: AsyncSession)  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  AsyncSession                             │
│  - execute(select(...))                   │
│  - add(model_instance)                    │
│  - commit() / rollback()                  │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  AsyncEngine (connection pool)            │
│  - Manages pool of database connections   │
│  - Handles reconnection on failure        │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  Database (PostgreSQL / SQLite)            │
└──────────────────────────────────────────┘
```

### Engine and Session Factory Setup

```python
# app/database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# async engine uses an async driver (asyncpg for PostgreSQL, aiosqlite for SQLite)
# echo=True logs all SQL statements -- useful for debugging, disable in production
DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/mydb"

engine = create_async_engine(
    DATABASE_URL,
    echo=True,       # Log SQL queries (disable in production)
    pool_size=5,     # Number of persistent connections
    max_overflow=10, # Additional connections when pool is exhausted
)

# Session factory: creates new AsyncSession instances
# expire_on_commit=False prevents attribute access errors after commit
# (without this, accessing user.name after commit would trigger a lazy load,
# which fails in async because it requires a synchronous DB call)
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

---

## 2. Defining Models with mapped_column

SQLAlchemy 2.0 uses `Mapped` and `mapped_column` for type-safe model definitions. This replaces the older `Column()` syntax.

```python
# app/models.py
from datetime import datetime
from typing import Optional

from sqlalchemy import String, Text, ForeignKey, func
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)

class Base(DeclarativeBase):
    """Base class for all models. DeclarativeBase replaces the older
    declarative_base() function and supports type annotations natively."""
    pass

class User(Base):
    __tablename__ = "users"

    # Mapped[int] declares both the Python type and the column type
    # mapped_column() configures column-level details
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,  # Index for faster lookups by username
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(String(255))

    # Optional fields use Mapped[Optional[...]] or Mapped[... | None]
    full_name: Mapped[str | None] = mapped_column(String(100), default=None)
    bio: Mapped[str | None] = mapped_column(Text, default=None)
    is_active: Mapped[bool] = mapped_column(default=True)

    # Server-side defaults: the DB generates these values
    # func.now() translates to NOW() in PostgreSQL, CURRENT_TIMESTAMP in SQLite
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),  # Auto-update on row modification
    )

    # Relationship to posts (one-to-many)
    # back_populates creates the bidirectional link
    posts: Mapped[list["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",  # Delete posts when user is deleted
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, username='{self.username}')>"


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(200))
    content: Mapped[str] = mapped_column(Text)
    is_published: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    # Foreign key: links each post to its author
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # Relationship back to user
    author: Mapped["User"] = relationship(back_populates="posts")

    # Many-to-many relationship with tags (via association table)
    tags: Mapped[list["Tag"]] = relationship(
        secondary="post_tags",
        back_populates="posts",
    )
```

### Pydantic Schemas (Separate from ORM Models)

```python
# app/schemas.py
from datetime import datetime
from pydantic import BaseModel, Field

# --- User Schemas ---
class UserCreate(BaseModel):
    """Input schema: what the client sends to create a user."""
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None

class UserResponse(BaseModel):
    """Output schema: what we send back to the client.
    Note: hashed_password is intentionally excluded."""
    id: int
    username: str
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime

    # Allows creating UserResponse directly from an ORM User object
    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    """Partial update schema: all fields optional."""
    full_name: str | None = None
    bio: str | None = None
    email: str | None = Field(default=None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# --- Post Schemas ---
class PostCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    content: str = Field(min_length=1)
    is_published: bool = False

class PostResponse(BaseModel):
    id: int
    title: str
    content: str
    is_published: bool
    created_at: datetime
    author_id: int

    model_config = {"from_attributes": True}
```

---

## 3. Database Session Management

The database session should be created per-request and closed afterward. FastAPI's dependency injection with yield dependencies handles this cleanly.

```python
# app/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from .database import async_session_factory

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session for each request.
    The session is committed on success or rolled back on exception.

    This is a 'yield dependency' -- FastAPI runs the code before yield
    when the request starts, and the code after yield when it ends."""
    async with async_session_factory() as session:
        try:
            yield session
            # If the endpoint completes without error, commit the transaction
            await session.commit()
        except Exception:
            # On any exception, roll back to prevent partial writes
            await session.rollback()
            raise
```

### Using the Session in Endpoints

```python
# app/routers/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from ..dependencies import get_db
from ..schemas import UserCreate, UserResponse

router = APIRouter(prefix="/api/users", tags=["users"])

@router.post("/", response_model=UserResponse, status_code=201)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),  # Injected session
):
    """The db session is created by get_db, used here,
    and automatically closed/committed after the response."""
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(new_user)
    # Flush pushes the INSERT to the DB so we get the generated id,
    # but doesn't commit the transaction yet (get_db handles that)
    await db.flush()
    return new_user
```

---

## 4. Async CRUD Operations

### Create

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import User, Post
from .schemas import UserCreate, PostCreate

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """Create a new user. db.add() stages the object;
    flush() sends the INSERT to get the auto-generated ID."""
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(user)
    await db.flush()     # Generates the ID without committing
    await db.refresh(user)  # Reload server-generated fields (created_at)
    return user
```

### Read (Single and List)

```python
async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    """Fetch a single user by primary key.
    SQLAlchemy 2.0 uses select() instead of query()."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()

async def get_user_by_username(db: AsyncSession, username: str) -> User | None:
    result = await db.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()

async def list_users(
    db: AsyncSession,
    skip: int = 0,
    limit: int = 10,
    is_active: bool | None = None,
) -> list[User]:
    """List users with pagination and optional filtering.
    Build the query dynamically based on provided filters."""
    query = select(User).offset(skip).limit(limit).order_by(User.created_at.desc())

    if is_active is not None:
        query = query.where(User.is_active == is_active)

    result = await db.execute(query)
    return list(result.scalars().all())
```

### Update

```python
async def update_user(
    db: AsyncSession,
    user_id: int,
    update_data: dict,
) -> User | None:
    """Partial update: only modify fields present in update_data.
    exclude_unset=True on the Pydantic model gives us only the
    fields the client explicitly sent."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return None

    # Only update fields that were explicitly provided
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.flush()
    await db.refresh(user)
    return user
```

### Delete

```python
async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """Delete a user. Returns True if the user existed and was deleted."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False
    await db.delete(user)
    await db.flush()
    return True
```

### Putting CRUD in Endpoints

```python
@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.patch("/{user_id}", response_model=UserResponse)
async def update_user_endpoint(
    user_id: int,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    # exclude_unset=True ensures we only update fields the client sent
    # This differentiates "field not sent" from "field set to None"
    update_dict = user_data.model_dump(exclude_unset=True)
    user = await update_user(db, user_id, update_dict)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## 5. Relationships

### One-to-Many (User has many Posts)

Already defined in the models above. Here is how to query with relationships:

```python
from sqlalchemy.orm import selectinload

async def get_user_with_posts(db: AsyncSession, user_id: int) -> User | None:
    """Eager-load posts to avoid N+1 query problem.
    Without selectinload, accessing user.posts would trigger
    a lazy load -- which fails in async contexts."""
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.posts))  # Load posts in a single query
    )
    return result.scalar_one_or_none()
```

### Many-to-Many (Posts and Tags)

```python
# app/models.py (continued)
from sqlalchemy import Table, Column, Integer, ForeignKey

# Association table: no ORM model needed, just a table
post_tags = Table(
    "post_tags",
    Base.metadata,
    Column("post_id", Integer, ForeignKey("posts.id"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id"), primary_key=True),
)

class Tag(Base):
    __tablename__ = "tags"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(50), unique=True)

    posts: Mapped[list["Post"]] = relationship(
        secondary=post_tags,
        back_populates="tags",
    )
```

### Querying Many-to-Many

```python
async def get_posts_by_tag(db: AsyncSession, tag_name: str) -> list[Post]:
    """Find all posts with a specific tag.
    The join traverses: Post → post_tags → Tag."""
    result = await db.execute(
        select(Post)
        .join(Post.tags)
        .where(Tag.name == tag_name)
        .options(selectinload(Post.tags))  # Also load all tags for each post
    )
    return list(result.scalars().unique().all())

async def add_tag_to_post(db: AsyncSession, post_id: int, tag_name: str):
    """Add a tag to a post. Creates the tag if it doesn't exist."""
    # Get or create the tag
    result = await db.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if not tag:
        tag = Tag(name=tag_name)
        db.add(tag)
        await db.flush()

    # Load the post with its tags
    result = await db.execute(
        select(Post)
        .where(Post.id == post_id)
        .options(selectinload(Post.tags))
    )
    post = result.scalar_one_or_none()
    if not post:
        raise ValueError(f"Post {post_id} not found")

    # Append the tag (SQLAlchemy handles the association table)
    if tag not in post.tags:
        post.tags.append(tag)
        await db.flush()
```

### Loading Strategies

| Strategy | Method | SQL Generated | Use When |
|----------|--------|--------------|----------|
| Lazy (default) | Access attribute | Separate query per access | Sync only, rarely used in async |
| Select-in | `selectinload()` | 1 extra SELECT with IN clause | Default choice for async |
| Joined | `joinedload()` | Single JOIN query | One-to-one or small one-to-many |
| Subquery | `subqueryload()` | 1 extra subquery | Large collections |

---

## 6. Alembic for Migrations

Alembic tracks database schema changes (migrations) so you can evolve your schema safely in production.

### Setup

```bash
# Initialize Alembic in your project
alembic init alembic

# This creates:
# alembic/
#   env.py          # Configuration
#   versions/       # Migration scripts
# alembic.ini       # Connection string and settings
```

### Configure Alembic for Async

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.models import Base          # Import your Base
from app.database import DATABASE_URL  # Import your DB URL

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Alembic compares this metadata against the database to detect changes
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """Generate SQL scripts without connecting to the database."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """Run migrations using async engine."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()

def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

### Creating and Running Migrations

```bash
# Auto-generate a migration by comparing models to current DB schema
alembic revision --autogenerate -m "create users and posts tables"

# Apply all pending migrations
alembic upgrade head

# Rollback the last migration
alembic downgrade -1

# View migration history
alembic history --verbose

# Show current database version
alembic current
```

### Generated Migration Example

```python
# alembic/versions/001_create_users_and_posts.py
"""create users and posts tables

Revision ID: a1b2c3d4e5f6
"""
from alembic import op
import sqlalchemy as sa

revision = "a1b2c3d4e5f6"
down_revision = None

def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("username", sa.String(50), unique=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(100)),
        sa.Column("is_active", sa.Boolean(), server_default="true"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    # Create index after table creation for cleaner migrations
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])

def downgrade() -> None:
    op.drop_index("ix_users_email")
    op.drop_index("ix_users_username")
    op.drop_table("users")
```

---

## 7. Connection Pooling

Connection pooling reuses database connections instead of creating a new one per request. This dramatically reduces connection overhead.

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/mydb",

    # Pool size: number of connections kept open permanently
    # Start with 5, increase based on load testing
    pool_size=5,

    # Overflow: temporary connections created when pool is full
    # These are closed after use, not returned to the pool
    max_overflow=10,

    # Recycle connections after 30 minutes to prevent stale connections
    # PostgreSQL's default idle timeout is often longer, but load balancers
    # (like PgBouncer or AWS RDS Proxy) may close idle connections sooner
    pool_recycle=1800,

    # Test connection health before handing it to a request
    # Adds a tiny overhead but prevents "connection reset" errors
    pool_pre_ping=True,

    # How long to wait for a connection from the pool
    # Raise an error instead of waiting forever
    pool_timeout=30,
)
```

### Pool Size Guidelines

```
Total connections = pool_size + max_overflow = 5 + 10 = 15 per worker

With 4 Uvicorn workers: 15 * 4 = 60 total connections

PostgreSQL default max_connections = 100
Leave headroom for admin connections and migrations

Rule of thumb: total connections < 80% of max_connections
```

### Monitoring Pool Health

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/debug/pool")
async def pool_status():
    """Expose connection pool metrics for monitoring.
    Only enable this in development or behind admin auth."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),  # Currently in use
        "checked_in": pool.checkedin(),    # Available in pool
        "overflow": pool.overflow(),       # Temporary connections
    }
```

---

## 8. Practice Problems

### Problem 1: Blog API with Full CRUD

Build a complete blog API with the following:
- `User` model: id, username, email, created_at
- `Post` model: id, title, content, is_published, author_id, created_at
- Endpoints: create user, create post (linked to user), list posts (with pagination and author filter), get single post with author info, update post, delete post
- Use `selectinload` to include author data in post responses
- Use separate Pydantic schemas for create, update, and response

### Problem 2: Database Session Testing

Write a `get_db` dependency that:
1. Creates a new `AsyncSession` per request
2. Commits on success, rolls back on exception
3. Logs the transaction duration
4. Can be overridden in tests to use a test database

Then write a test that overrides `get_db` to use an in-memory SQLite database.

### Problem 3: Many-to-Many with CRUD

Extend the blog API with:
- `Tag` model and `post_tags` association table
- `POST /posts/{post_id}/tags` -- add a tag to a post
- `DELETE /posts/{post_id}/tags/{tag_id}` -- remove a tag
- `GET /tags/{tag_name}/posts` -- list all posts with a specific tag
- Ensure tags are unique (reuse existing tags, don't create duplicates)

### Problem 4: Alembic Migration Workflow

Starting from an empty database:
1. Create initial migration with User and Post tables
2. Create a second migration that adds a `bio` column to User and a `view_count` column (default 0) to Post
3. Create a third migration that adds an index on `Post.created_at`
4. Practice upgrading to head, downgrading one step, and upgrading again

Document the commands you run at each step.

### Problem 5: Connection Pool Tuning

You have an API that:
- Receives 500 requests/second
- Each request takes ~50ms of database time
- Runs 4 Uvicorn workers
- PostgreSQL `max_connections` is set to 200

Calculate:
1. How many concurrent DB connections are needed per worker?
2. What `pool_size` and `max_overflow` values would you set?
3. What happens if the pool is exhausted? How would you detect and handle it?
4. Would you recommend PgBouncer in front of PostgreSQL? Why or why not?

---

## 9. References

- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/en/20/)
- [SQLAlchemy Async Documentation](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [FastAPI SQL Databases Guide](https://fastapi.tiangolo.com/tutorial/sql-databases/)
- [asyncpg Documentation](https://magicstack.github.io/asyncpg/)
- [Pydantic model_config](https://docs.pydantic.dev/latest/concepts/config/)

---

**Previous**: [FastAPI Advanced](./03_FastAPI_Advanced.md) | **Next**: [FastAPI Testing](./05_FastAPI_Testing.md)
