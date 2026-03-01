# 18. Project: REST API

**Previous**: [Observability](./17_Observability.md)

**Difficulty**: ⭐⭐⭐⭐

## Learning Objectives

- Build a complete REST API from scratch using FastAPI, SQLAlchemy, and Alembic
- Implement CRUD operations with proper validation, error handling, and status codes
- Add JWT authentication with user registration, login, and protected endpoints
- Apply pagination, filtering, and sorting patterns to list endpoints
- Containerize and test the application for production deployment

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Project Setup](#2-project-setup)
3. [Data Models](#3-data-models)
4. [CRUD Endpoints](#4-crud-endpoints)
5. [JWT Authentication](#5-jwt-authentication)
6. [Pagination and Filtering](#6-pagination-and-filtering)
7. [Testing with pytest](#7-testing-with-pytest)
8. [Docker Deployment](#8-docker-deployment)
9. [Extension Ideas](#9-extension-ideas)

---

## 1. Project Overview

In this capstone project, you will build a **blog API** that brings together the patterns covered throughout this course. The API supports user registration and authentication, blog post management, and commenting --- all with proper pagination, filtering, and error handling.

### Features

- User registration and JWT authentication (access + refresh tokens)
- Blog post CRUD (create, read, update, delete) with ownership enforcement
- Commenting on posts
- Pagination (cursor-based), filtering (by author, status, tag), and sorting
- Input validation with Pydantic
- Database migrations with Alembic
- Comprehensive test suite
- Docker containerization

### Final Project Structure

```
blog-api/
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry point
│   ├── config.py           # Settings (Pydantic BaseSettings)
│   ├── database.py         # SQLAlchemy engine and session
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── post.py
│   │   └── comment.py
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   ├── post.py
│   │   ├── comment.py
│   │   └── pagination.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── users.py
│   │   ├── posts.py
│   │   └── comments.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   └── post.py
│   └── dependencies.py     # Shared FastAPI dependencies
├── alembic/
│   ├── env.py
│   └── versions/
├── tests/
│   ├── conftest.py
│   ├── test_auth.py
│   ├── test_posts.py
│   └── test_comments.py
├── alembic.ini
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

---

## 2. Project Setup

### Requirements

```
# requirements.txt
fastapi==0.115.0
uvicorn[standard]==0.32.0
sqlalchemy[asyncio]==2.0.36
asyncpg==0.30.0
alembic==1.14.0
pydantic-settings==2.6.0
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.12
httpx==0.28.0
pytest==8.3.0
pytest-asyncio==0.24.0
```

### Configuration

```python
# app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Blog API"
    debug: bool = False

    # Database
    database_url: str = "postgresql+asyncpg://blog:secret@localhost:5432/blogdb"

    # Authentication
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Pagination defaults
    default_page_size: int = 20
    max_page_size: int = 100

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

### Database Setup

```python
# app/database.py
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    """Dependency that provides a database session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### Application Entry Point

```python
# app/main.py
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import engine, Base
from app.routers import auth, users, posts, comments

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/users", tags=["Users"])
app.include_router(posts.router, prefix="/posts", tags=["Posts"])
app.include_router(comments.router, prefix="/posts", tags=["Comments"])


@app.get("/health")
async def health_check():
    return {"status": "ok"}
```

### Initialize Alembic

```bash
# Initialize Alembic for async SQLAlchemy
alembic init -t async alembic
```

```python
# alembic/env.py (key modifications)
from app.database import Base
from app.models import user, post, comment  # Import all models

target_metadata = Base.metadata

# ... rest of the async env.py template
```

```bash
# Create and run first migration
alembic revision --autogenerate -m "initial tables"
alembic upgrade head
```

---

## 3. Data Models

### User Model

```python
# app/models/user.py
from datetime import datetime, timezone
from sqlalchemy import String, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    posts: Mapped[list["Post"]] = relationship(back_populates="author", cascade="all, delete-orphan")
    comments: Mapped[list["Comment"]] = relationship(back_populates="author", cascade="all, delete-orphan")
```

### Post Model

```python
# app/models/post.py
from datetime import datetime, timezone
from sqlalchemy import String, Text, ForeignKey, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column, relationship
import enum

from app.database import Base


class PostStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class Post(Base):
    __tablename__ = "posts"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    slug: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    status: Mapped[PostStatus] = mapped_column(
        Enum(PostStatus), default=PostStatus.DRAFT
    )
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    author: Mapped["User"] = relationship(back_populates="posts")
    comments: Mapped[list["Comment"]] = relationship(
        back_populates="post", cascade="all, delete-orphan"
    )
```

### Comment Model

```python
# app/models/comment.py
from datetime import datetime, timezone
from sqlalchemy import Text, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Comment(Base):
    __tablename__ = "comments"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    content: Mapped[str] = mapped_column(Text)
    post_id: Mapped[int] = mapped_column(ForeignKey("posts.id"), index=True)
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    post: Mapped["Post"] = relationship(back_populates="comments")
    author: Mapped["User"] = relationship(back_populates="comments")
```

### Pydantic Schemas

```python
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# app/schemas/post.py
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from app.models.post import PostStatus


class PostCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    status: PostStatus = PostStatus.DRAFT


class PostUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    content: Optional[str] = Field(None, min_length=1)
    status: Optional[PostStatus] = None


class PostResponse(BaseModel):
    id: int
    title: str
    slug: str
    content: str
    status: PostStatus
    author: "UserResponse"
    comment_count: int = 0
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# app/schemas/comment.py
class CommentCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=5000)


class CommentResponse(BaseModel):
    id: int
    content: str
    author: "UserResponse"
    created_at: datetime

    model_config = {"from_attributes": True}
```

---

## 4. CRUD Endpoints

### Post Router

```python
# app/routers/posts.py
import re
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from typing import Optional

from app.database import get_db
from app.models.post import Post, PostStatus
from app.models.comment import Comment
from app.schemas.post import PostCreate, PostUpdate, PostResponse
from app.dependencies import get_current_user

router = APIRouter()


def generate_slug(title: str) -> str:
    """Convert a title to a URL-friendly slug."""
    slug = title.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


@router.post("/", status_code=status.HTTP_201_CREATED, response_model=PostResponse)
async def create_post(
    post_data: PostCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    slug = generate_slug(post_data.title)

    # Ensure slug uniqueness
    existing = await db.execute(select(Post).where(Post.slug == slug))
    if existing.scalar_one_or_none():
        slug = f"{slug}-{current_user.id}"

    post = Post(
        title=post_data.title,
        slug=slug,
        content=post_data.content,
        status=post_data.status,
        author_id=current_user.id,
    )
    db.add(post)
    await db.flush()
    await db.refresh(post, attribute_names=["author", "comments"])
    return post


@router.get("/{post_id}", response_model=PostResponse)
async def get_post(post_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.author), selectinload(Post.comments))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Post with id {post_id} not found",
        )
    return post


@router.put("/{post_id}", response_model=PostResponse)
async def update_post(
    post_id: int,
    post_data: PostUpdate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(
        select(Post)
        .options(selectinload(Post.author), selectinload(Post.comments))
        .where(Post.id == post_id)
    )
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to edit this post")

    # Apply partial update: only update fields that were explicitly set
    update_data = post_data.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(post, field, value)

    if "title" in update_data:
        post.slug = generate_slug(update_data["title"])

    await db.flush()
    await db.refresh(post)
    return post


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_post(
    post_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(select(Post).where(Post.id == post_id))
    post = result.scalar_one_or_none()

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    if post.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this post")

    await db.delete(post)
```

### Comment Router

```python
# app/routers/comments.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.models.post import Post
from app.models.comment import Comment
from app.schemas.comment import CommentCreate, CommentResponse
from app.dependencies import get_current_user

router = APIRouter()


@router.post(
    "/{post_id}/comments",
    status_code=status.HTTP_201_CREATED,
    response_model=CommentResponse,
)
async def create_comment(
    post_id: int,
    comment_data: CommentCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # Verify post exists
    result = await db.execute(select(Post).where(Post.id == post_id))
    if not result.scalar_one_or_none():
        raise HTTPException(status_code=404, detail="Post not found")

    comment = Comment(
        content=comment_data.content,
        post_id=post_id,
        author_id=current_user.id,
    )
    db.add(comment)
    await db.flush()
    await db.refresh(comment, attribute_names=["author"])
    return comment


@router.get("/{post_id}/comments", response_model=list[CommentResponse])
async def list_comments(
    post_id: int,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Comment)
        .options(selectinload(Comment.author))
        .where(Comment.post_id == post_id)
        .order_by(Comment.created_at.desc())
    )
    return result.scalars().all()


@router.delete(
    "/{post_id}/comments/{comment_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_comment(
    post_id: int,
    comment_id: int,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    result = await db.execute(
        select(Comment).where(
            Comment.id == comment_id, Comment.post_id == post_id
        )
    )
    comment = result.scalar_one_or_none()
    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")
    if comment.author_id != current_user.id:
        raise HTTPException(status_code=403, detail="Not authorized")
    await db.delete(comment)
```

---

## 5. JWT Authentication

### Auth Service

```python
# app/services/auth.py
from datetime import datetime, timedelta, timezone
from jose import jwt, JWTError
from passlib.context import CryptContext

from app.config import get_settings

settings = get_settings()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": str(user_id), "exp": expire, "type": "access"}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_refresh_token(user_id: int) -> str:
    expire = datetime.now(timezone.utc) + timedelta(
        days=settings.refresh_token_expire_days
    )
    payload = {"sub": str(user_id), "exp": expire, "type": "refresh"}
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises JWTError on failure."""
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
```

### Auth Dependencies

```python
# app/dependencies.py
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.services.auth import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = decode_token(token)
        user_id = payload.get("sub")
        token_type = payload.get("type")
        if user_id is None or token_type != "access":
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise credentials_exception
    return user
```

### Auth Router

```python
# app/routers/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse
from app.services.auth import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
)

router = APIRouter()


@router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    # Check for existing username or email
    result = await db.execute(
        select(User).where(
            (User.username == user_data.username) | (User.email == user_data.email)
        )
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already registered",
        )

    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)
    return user


@router.post("/login")
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    # Find user by username
    result = await db.execute(
        select(User).where(User.username == form_data.username)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return {
        "access_token": create_access_token(user.id),
        "refresh_token": create_refresh_token(user.id),
        "token_type": "bearer",
    }


@router.post("/refresh")
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    try:
        payload = decode_token(refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid token type")
        user_id = int(payload["sub"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")

    return {
        "access_token": create_access_token(user.id),
        "token_type": "bearer",
    }
```

---

## 6. Pagination and Filtering

### Cursor-Based Pagination Schema

```python
# app/schemas/pagination.py
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional
import base64
import json

T = TypeVar("T")


class CursorPage(BaseModel, Generic[T]):
    """Generic paginated response with cursor-based pagination."""
    items: list[T]
    next_cursor: Optional[str] = None
    has_more: bool = False
    total: int = 0


def encode_cursor(post_id: int, created_at: str) -> str:
    """Encode pagination cursor as base64 JSON."""
    data = {"id": post_id, "created_at": created_at}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """Decode pagination cursor from base64 JSON."""
    data = base64.urlsafe_b64decode(cursor.encode())
    return json.loads(data)
```

### Paginated Post Listing

```python
# In app/routers/posts.py — add to existing router

@router.get("/", response_model=CursorPage[PostResponse])
async def list_posts(
    db: AsyncSession = Depends(get_db),
    cursor: Optional[str] = None,
    limit: int = Query(default=20, ge=1, le=100),
    status_filter: Optional[PostStatus] = Query(None, alias="status"),
    author_id: Optional[int] = None,
    sort: str = Query(default="-created_at"),
    search: Optional[str] = None,
):
    """List posts with cursor-based pagination, filtering, and sorting."""
    # Base query with eager loading
    query = select(Post).options(
        selectinload(Post.author),
        selectinload(Post.comments),
    )

    # Apply filters
    if status_filter:
        query = query.where(Post.status == status_filter)
    else:
        # By default, only show published posts to non-authors
        query = query.where(Post.status == PostStatus.PUBLISHED)

    if author_id:
        query = query.where(Post.author_id == author_id)

    if search:
        query = query.where(Post.title.ilike(f"%{search}%"))

    # Apply cursor (for pagination continuity)
    if cursor:
        cursor_data = decode_cursor(cursor)
        # For descending order: get posts older than the cursor
        query = query.where(
            (Post.created_at < cursor_data["created_at"])
            | (
                (Post.created_at == cursor_data["created_at"])
                & (Post.id < cursor_data["id"])
            )
        )

    # Apply sorting
    if sort.startswith("-"):
        field = sort[1:]
        query = query.order_by(
            getattr(Post, field).desc(), Post.id.desc()
        )
    else:
        query = query.order_by(
            getattr(Post, sort).asc(), Post.id.asc()
        )

    # Fetch one extra to determine if there are more pages
    query = query.limit(limit + 1)

    result = await db.execute(query)
    posts = list(result.scalars().all())

    # Determine if there are more results
    has_more = len(posts) > limit
    if has_more:
        posts = posts[:limit]

    # Generate next cursor
    next_cursor = None
    if has_more and posts:
        last_post = posts[-1]
        next_cursor = encode_cursor(
            last_post.id,
            last_post.created_at.isoformat(),
        )

    # Get total count (for display purposes)
    count_query = select(func.count(Post.id))
    if status_filter:
        count_query = count_query.where(Post.status == status_filter)
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    return CursorPage(
        items=posts,
        next_cursor=next_cursor,
        has_more=has_more,
        total=total,
    )
```

---

## 7. Testing with pytest

### Test Configuration

```python
# tests/conftest.py
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

from app.main import app
from app.database import Base, get_db
from app.services.auth import hash_password, create_access_token
from app.models.user import User

# Use an in-memory SQLite for tests, or a separate test database
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture
async def db_session():
    """Create tables and provide a clean session for each test."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(db_session):
    """HTTP client with test database injected."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session):
    """Create a test user and return it with an access token."""
    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hash_password("testpassword123"),
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.refresh(user)

    token = create_access_token(user.id)
    return {"user": user, "token": token}
```

### Auth Tests

```python
# tests/test_auth.py
import pytest


@pytest.mark.asyncio
async def test_register_user(client):
    response = await client.post("/auth/register", json={
        "username": "newuser",
        "email": "new@example.com",
        "password": "securepassword123",
    })
    assert response.status_code == 201
    data = response.json()
    assert data["username"] == "newuser"
    assert data["email"] == "new@example.com"
    assert "hashed_password" not in data  # Ensure password is not exposed


@pytest.mark.asyncio
async def test_register_duplicate_username(client):
    # Register first user
    await client.post("/auth/register", json={
        "username": "duplicate",
        "email": "first@example.com",
        "password": "password12345678",
    })
    # Attempt duplicate
    response = await client.post("/auth/register", json={
        "username": "duplicate",
        "email": "second@example.com",
        "password": "password12345678",
    })
    assert response.status_code == 409


@pytest.mark.asyncio
async def test_login_success(client, test_user):
    response = await client.post("/auth/login", data={
        "username": "testuser",
        "password": "testpassword123",
    })
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_login_wrong_password(client, test_user):
    response = await client.post("/auth/login", data={
        "username": "testuser",
        "password": "wrongpassword",
    })
    assert response.status_code == 401
```

### Post Tests

```python
# tests/test_posts.py
import pytest


@pytest.mark.asyncio
async def test_create_post(client, test_user):
    token = test_user["token"]
    response = await client.post(
        "/posts/",
        json={
            "title": "My First Post",
            "content": "This is the content of my first post.",
            "status": "published",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 201
    data = response.json()
    assert data["title"] == "My First Post"
    assert data["slug"] == "my-first-post"
    assert data["status"] == "published"


@pytest.mark.asyncio
async def test_create_post_unauthorized(client):
    response = await client.post("/posts/", json={
        "title": "Unauthorized Post",
        "content": "Should fail.",
    })
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_post(client, test_user):
    token = test_user["token"]
    # Create a post first
    create_response = await client.post(
        "/posts/",
        json={
            "title": "Readable Post",
            "content": "Content here.",
            "status": "published",
        },
        headers={"Authorization": f"Bearer {token}"},
    )
    post_id = create_response.json()["id"]

    # Fetch it (no auth required for reading published posts)
    response = await client.get(f"/posts/{post_id}")
    assert response.status_code == 200
    assert response.json()["title"] == "Readable Post"


@pytest.mark.asyncio
async def test_update_post_by_owner(client, test_user):
    token = test_user["token"]
    create_response = await client.post(
        "/posts/",
        json={"title": "Original Title", "content": "Original content."},
        headers={"Authorization": f"Bearer {token}"},
    )
    post_id = create_response.json()["id"]

    response = await client.put(
        f"/posts/{post_id}",
        json={"title": "Updated Title"},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    assert response.json()["title"] == "Updated Title"


@pytest.mark.asyncio
async def test_delete_post(client, test_user):
    token = test_user["token"]
    create_response = await client.post(
        "/posts/",
        json={"title": "To Delete", "content": "Will be deleted."},
        headers={"Authorization": f"Bearer {token}"},
    )
    post_id = create_response.json()["id"]

    response = await client.delete(
        f"/posts/{post_id}",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 204

    # Verify it is gone
    get_response = await client.get(f"/posts/{post_id}")
    assert get_response.status_code == 404


@pytest.mark.asyncio
async def test_list_posts_pagination(client, test_user):
    token = test_user["token"]
    # Create 5 posts
    for i in range(5):
        await client.post(
            "/posts/",
            json={
                "title": f"Post {i}",
                "content": f"Content {i}",
                "status": "published",
            },
            headers={"Authorization": f"Bearer {token}"},
        )

    # Fetch first page (limit 2)
    response = await client.get("/posts/?limit=2")
    assert response.status_code == 200
    data = response.json()
    assert len(data["items"]) == 2
    assert data["has_more"] is True
    assert data["next_cursor"] is not None

    # Fetch second page using cursor
    response2 = await client.get(f"/posts/?limit=2&cursor={data['next_cursor']}")
    data2 = response2.json()
    assert len(data2["items"]) == 2
    assert data2["has_more"] is True
```

---

## 8. Docker Deployment

### Dockerfile

```dockerfile
FROM python:3.12-slim AS builder

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.12-slim AS runtime

RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash appuser

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY --chown=appuser:appuser . .

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["gunicorn", "app.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000"]
```

### Docker Compose

```yaml
# docker-compose.yml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://blog:secret@db:5432/blogdb
      - SECRET_KEY=${SECRET_KEY:-change-me-in-production}
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: blogdb
      POSTGRES_USER: blog
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U blog -d blogdb"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
```

### Running the Project

```bash
# Start services
docker compose up -d

# Run database migrations
docker compose exec app alembic upgrade head

# View logs
docker compose logs -f app

# Run tests
docker compose exec app pytest tests/ -v

# Access the API docs
# Open http://localhost:8000/docs in your browser
```

---

## 9. Extension Ideas

Once the core project is working, consider extending it with these features. Each one reinforces a concept from the course.

**1. Tags and Categories**
- Add a `Tag` model with a many-to-many relationship to `Post`
- Implement filtering by tag: `GET /posts?tag=python`
- This practices relationship modeling and query optimization

**2. Image Uploads**
- Add a `POST /posts/{id}/images` endpoint that accepts file uploads
- Store files in local storage or S3-compatible object storage
- Return the image URL in the post response
- This practices file handling and external service integration

**3. Full-Text Search**
- Add PostgreSQL full-text search on post titles and content
- Create a `GET /search?q=fastapi` endpoint with relevance ranking
- This practices database features beyond basic CRUD

**4. Rate Limiting**
- Add rate limiting middleware using `slowapi`
- Different limits for authenticated vs. unauthenticated users
- Return proper `429` responses with `Retry-After` headers
- This practices the API design patterns from Lesson 14

**5. WebSocket Notifications**
- Add a WebSocket endpoint for real-time comment notifications
- When a new comment is added to a post, notify the post author
- This practices async programming and real-time communication

**6. Caching Layer**
- Add Redis caching for frequently accessed posts
- Implement cache invalidation on post update/delete
- Add ETags for conditional requests
- This practices performance optimization

**7. Observability**
- Add structured logging with structlog
- Instrument with Prometheus metrics
- Add OpenTelemetry tracing
- This practices the observability patterns from Lesson 17

**8. CI/CD Pipeline**
- Create a GitHub Actions workflow that runs tests, builds the Docker image, and pushes to a container registry
- Add linting (ruff) and type checking (mypy)
- This practices production deployment from Lesson 16

---

## References

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [python-jose Documentation](https://python-jose.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)

---

**Previous**: [Observability](./17_Observability.md)
