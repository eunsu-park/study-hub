# 18. 프로젝트: REST API

**이전**: [관찰 가능성](./17_Observability.md)

**난이도**: ⭐⭐⭐⭐

## 학습 목표

- FastAPI, SQLAlchemy, Alembic을 사용하여 완전한 REST API를 처음부터 구축한다
- 적절한 검증(validation), 오류 처리, 상태 코드를 갖춘 CRUD 작업을 구현한다
- 사용자 등록, 로그인, 보호된 엔드포인트를 포함한 JWT 인증을 추가한다
- 목록 엔드포인트에 페이지네이션(pagination), 필터링(filtering), 정렬(sorting) 패턴을 적용한다
- 프로덕션 배포를 위해 애플리케이션을 컨테이너화하고 테스트한다

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 설정](#2-프로젝트-설정)
3. [데이터 모델](#3-데이터-모델)
4. [CRUD 엔드포인트](#4-crud-엔드포인트)
5. [JWT 인증](#5-jwt-인증)
6. [페이지네이션과 필터링](#6-페이지네이션과-필터링)
7. [pytest를 이용한 테스트](#7-pytest를-이용한-테스트)
8. [Docker 배포](#8-docker-배포)
9. [확장 아이디어](#9-확장-아이디어)

---

## 1. 프로젝트 개요

이 캡스톤(capstone) 프로젝트에서는 강좌 전반에 걸쳐 다룬 패턴들을 통합한 **블로그 API**를 구축한다. API는 사용자 등록 및 인증, 블로그 게시물 관리, 댓글 기능을 지원하며, 적절한 페이지네이션, 필터링, 오류 처리를 갖춘다.

### 기능

- 사용자 등록과 JWT 인증 (액세스 + 리프레시 토큰)
- 소유권 강제를 포함한 블로그 게시물 CRUD (생성, 조회, 수정, 삭제)
- 게시물 댓글
- 페이지네이션(커서 기반), 필터링(작성자, 상태, 태그별), 정렬
- Pydantic을 이용한 입력 검증
- Alembic을 이용한 데이터베이스 마이그레이션
- 포괄적인 테스트 스위트
- Docker 컨테이너화

### 최종 프로젝트 구조

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

## 2. 프로젝트 설정

### 요구 사항

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

### 설정

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

### 데이터베이스 설정

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
    """요청당 데이터베이스 세션을 제공하는 의존성(Dependency)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
```

### 애플리케이션 진입점

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

### Alembic 초기화

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

## 3. 데이터 모델

### User 모델

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

### Post 모델

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

### Comment 모델

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

### Pydantic 스키마

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

## 4. CRUD 엔드포인트

### Post 라우터

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
    """제목을 URL 친화적인 슬러그(slug)로 변환한다."""
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

### Comment 라우터

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

## 5. JWT 인증

### Auth 서비스

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
    """JWT 토큰을 디코딩하고 검증한다. 실패 시 JWTError를 발생시킨다."""
    return jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
```

### Auth 의존성

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

### Auth 라우터

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

## 6. 페이지네이션과 필터링

### 커서 기반 페이지네이션 스키마

```python
# app/schemas/pagination.py
from pydantic import BaseModel, Field
from typing import Generic, TypeVar, Optional
import base64
import json

T = TypeVar("T")


class CursorPage(BaseModel, Generic[T]):
    """커서 기반 페이지네이션(cursor-based pagination)을 사용하는 제네릭 페이지 응답."""
    items: list[T]
    next_cursor: Optional[str] = None
    has_more: bool = False
    total: int = 0


def encode_cursor(post_id: int, created_at: str) -> str:
    """페이지네이션 커서를 base64 JSON으로 인코딩한다."""
    data = {"id": post_id, "created_at": created_at}
    return base64.urlsafe_b64encode(json.dumps(data).encode()).decode()


def decode_cursor(cursor: str) -> dict:
    """페이지네이션 커서를 base64 JSON에서 디코딩한다."""
    data = base64.urlsafe_b64decode(cursor.encode())
    return json.loads(data)
```

### 페이지네이션이 적용된 게시물 목록

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
    """커서 기반 페이지네이션, 필터링, 정렬을 사용하여 게시물을 목록으로 반환한다."""
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

## 7. pytest를 이용한 테스트

### 테스트 설정

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
    """테이블을 생성하고 각 테스트에 대해 깨끗한 세션을 제공한다."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client(db_session):
    """테스트 데이터베이스가 주입된 HTTP 클라이언트."""
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def test_user(db_session):
    """테스트 사용자를 생성하고 액세스 토큰과 함께 반환한다."""
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

### Auth 테스트

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

### Post 테스트

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

## 8. Docker 배포

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

### 프로젝트 실행

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

## 9. 확장 아이디어

핵심 프로젝트가 작동하면 다음 기능으로 확장을 고려해보라. 각 기능은 강좌의 개념을 강화한다.

**1. 태그와 카테고리**
- `Post`와 다대다(many-to-many) 관계를 갖는 `Tag` 모델 추가
- 태그별 필터링 구현: `GET /posts?tag=python`
- 관계 모델링과 쿼리 최적화를 연습한다

**2. 이미지 업로드**
- 파일 업로드를 받는 `POST /posts/{id}/images` 엔드포인트 추가
- 로컬 저장소 또는 S3 호환 객체 저장소에 파일 저장
- 게시물 응답에서 이미지 URL 반환
- 파일 처리와 외부 서비스 통합을 연습한다

**3. 전문 검색(Full-Text Search)**
- 게시물 제목과 내용에 PostgreSQL 전문 검색 추가
- 관련성 순위를 갖춘 `GET /search?q=fastapi` 엔드포인트 생성
- 기본 CRUD 이상의 데이터베이스 기능을 연습한다

**4. 속도 제한(Rate Limiting)**
- `slowapi`를 사용한 속도 제한 미들웨어 추가
- 인증된 사용자와 비인증 사용자에 대해 다른 제한 적용
- `Retry-After` 헤더를 포함한 적절한 `429` 응답 반환
- 레슨 14의 API 설계 패턴을 연습한다

**5. WebSocket 알림**
- 실시간 댓글 알림을 위한 WebSocket 엔드포인트 추가
- 게시물에 새 댓글이 추가되면 게시물 작성자에게 알림
- 비동기 프로그래밍과 실시간 통신을 연습한다

**6. 캐싱 레이어**
- 자주 접근하는 게시물에 Redis 캐싱 추가
- 게시물 수정/삭제 시 캐시 무효화(cache invalidation) 구현
- 조건부 요청을 위한 ETag 추가
- 성능 최적화를 연습한다

**7. 관찰 가능성**
- structlog를 사용한 구조화 로깅 추가
- Prometheus 메트릭으로 계측
- OpenTelemetry 추적 추가
- 레슨 17의 관찰 가능성 패턴을 연습한다

**8. CI/CD 파이프라인**
- 테스트 실행, Docker 이미지 빌드, 컨테이너 레지스트리에 푸시하는 GitHub Actions 워크플로우 생성
- 린팅(ruff)과 타입 체킹(mypy) 추가
- 레슨 16의 프로덕션 배포를 연습한다

---

## 참고 자료

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [SQLAlchemy 2.0 Documentation](https://docs.sqlalchemy.org/)
- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [python-jose Documentation](https://python-jose.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [Docker Documentation](https://docs.docker.com/)

---

**이전**: [관찰 가능성](./17_Observability.md)
