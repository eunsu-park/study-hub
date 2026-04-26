# 04. FastAPI 데이터베이스 연동(FastAPI Database Integration)

**이전**: [FastAPI 고급](./03_FastAPI_Advanced.md) | **다음**: [FastAPI 테스팅](./05_FastAPI_Testing.md)

**난이도**: ⭐⭐⭐

---

## 학습 목표

- FastAPI와 함께 사용하기 위해 비동기 지원이 있는 SQLAlchemy 2.0을 구성할 수 있다
- 현대적인 `mapped_column` 선언형 스타일을 사용하여 데이터베이스 모델을 정의할 수 있다
- 적절한 생명주기 관리와 함께 의존성 주입 방식의 데이터베이스 세션을 구현할 수 있다
- 관계(relationship)와 즉시/지연 로딩 전략을 사용한 비동기 CRUD 작업을 구축할 수 있다
- 프로덕션에서 스키마 진화를 위한 Alembic 마이그레이션을 설정하고 실행할 수 있다

---

## 목차

프레임워크 참조에 들어가기 전에, [**이론과 원리**](#이론과-원리) 섹션을 먼저 읽어보세요. 연결 풀(connection pool) 알고리즘, 작업 단위(unit of work)로서의 SQLAlchemy 세션, 그리고 N+1 문제를 일으키는 lazy vs eager 로딩 트레이드오프를 다룹니다.

1. [비동기 지원이 있는 SQLAlchemy 2.0](#1-비동기-지원이-있는-sqlalchemy-20)
2. [mapped_column으로 모델 정의](#2-mapped_column으로-모델-정의)
3. [데이터베이스 세션 관리](#3-데이터베이스-세션-관리)
4. [비동기 CRUD 작업](#4-비동기-crud-작업)
5. [관계(Relationships)](#5-관계relationships)
6. [마이그레이션을 위한 Alembic](#6-마이그레이션을-위한-alembic)
7. [연결 풀링(Connection Pooling)](#7-연결-풀링connection-pooling)
8. [연습 문제](#8-연습-문제)
9. [참고 자료](#9-참고-자료)

---

## 이론과 원리

웹 앱의 데이터베이스 스택은 초보자가 흔히 하나로 뭉뜽그리는 세 개의 중첩된 관심사로 이루어집니다. 이를 미리 분리하면 이후의 모든 결정 — 풀 크기, 세션 범위, eager vs lazy 로딩 — 이 설명됩니다.

- **(A) 연결 풀(Connection pool)** — 요청 사이에서 공유되는, 데이터베이스로의 유한한 TCP 연결 집합.
- **(B) 세션 / 작업 단위(Unit of work)** — identity map이 있는 메모리 내 트랜잭션 컨텍스트.
- **(C) 로딩 전략(Loading strategies)** — 관계가 실제로 데이터베이스를 치는 시점, 그리고 N+1 함정.

### A. 연결 풀링: 유한한 자원

데이터베이스 연결은 공짜가 아닙니다. PostgreSQL 백엔드 연결은 메가바이트 단위 RAM을 가진 fork된 프로세스입니다. MySQL 연결은 OS 스레드입니다. 연결을 여는 데는 TCP 핸드셰이크에 인증까지 필요합니다. 요청마다 새 연결을 만드는 웹 앱은 앱 서버를 포화시키기 한참 전에 데이터베이스를 포화시킵니다.

해결책은 **연결 풀**입니다. 미리 열린 작은 연결 집합을 살아 있게 유지하고, 각 요청에 하나씩 빌려주고, 끝나면 돌려받습니다.

#### A.1 SQLAlchemy의 QueuePool

기본 풀 구현은 `QueuePool`입니다. 다음을 유지합니다.

- `pool_size`개의 연결을 항상 살려둡니다(기본 5).
- `max_overflow`개의 추가 연결을 임시로 허용합니다(기본 10).
- `pool_timeout` — 요청이 빈 연결을 기다리는 최대 초(기본 30).

`session.execute(...)`가 실행되면 풀에서 연결을 `acquire()`합니다. 세션이 닫히면 `release()`합니다. `pool_size + max_overflow`개가 모두 사용 중이면 다음 호출자는 최대 `pool_timeout`초 동안 블로킹된 후 `TimeoutError`를 던집니다.

외워두어야 할 산수: **데이터베이스에 가는 총 연결 수 = (pool_size + max_overflow) × 앱 프로세스 수 × 앱 인스턴스 수**. 파드당 Gunicorn 워커 4개, 파드 3개, `pool_size=5, max_overflow=10`이면 데이터베이스는 한 서비스로부터 최대 `15 × 4 × 3 = 180`개의 연결을 봅니다. 많은 프로덕션 장애가 `FATAL: too many connections`로 시작합니다. 의도적으로 상한을 정하세요.

#### A.2 비동기 환경에서의 풀 크기

레슨 03의 FastAPI 비동기 모델에서는 워커 *하나*가 수백 개의 진행 중 요청을 가질 수 있습니다. 풀 크기는 *동시 진행 중인 요청* 수가 아니라 *동시에 실행 중인 SQL 문* 수의 상한입니다. 연결을 이미 반납한(SQL 문 사이에 있는) 요청은 카운트되지 않습니다.

어림짐작: `pool_size ≈ 동시 SQL 피크 = 평균 쿼리 지연시간 × 피크 QPS`. 100 QPS에서 50ms 쿼리는 워커당 약 5개의 동시 연결이 필요합니다.

#### A.3 `pool_pre_ping`이 중요한 이유

풀 안에서 몇 시간 idle 상태였던 연결은 조용히 죽어 있을 수 있습니다 — 데이터베이스가 재시작했거나, 로드 밸런서가 TCP를 끊었거나, 네트워크가 깜빡했거나. 그 연결을 받는 다음 요청은 `OperationalError: server closed connection`을 봅니다.

`pool_pre_ping=True`는 연결을 빌려주기 전에 싸구려 `SELECT 1`을 실행합니다. 지연 비용은 한 라운드트립이고, 이득은 죽은 연결이 조용히 감지되어 교체된다는 점입니다. 프로덕션에서는 거의 항상 켜는 게 좋습니다.

### B. 세션: 작업 단위

SQLAlchemy `Session`(또는 `AsyncSession`)은 연결의 *얇은 래퍼가 아닙니다*. 메모리 내 **작업 단위(unit of work)** — commit하기 전까지 로드되거나 수정된 모든 객체를 추적하는 트랜잭션 컨텍스트입니다.

#### B.1 Identity map

Identity map은 `(클래스, 기본키)`를 키로 하는 dict입니다. `User(id=42)`를 처음 로드할 때 SQLAlchemy는 객체를 만들어 이 map에 저장합니다. 같은 세션 안에서 row id=42를 반환하는 — 관계를 통한 것을 포함한 — 후속 모든 쿼리는 새로 만든 객체가 아니라 *같은* Python 객체를 돌려줍니다. 이를 통해 보장되는 것:

- 객체 동치(equality)가 데이터베이스 identity와 일치합니다.
- 한 참조의 변경이 모든 참조에서 보입니다.
- 반복 쿼리가 중복 객체를 만들어 메모리를 낭비하지 않습니다.

결과: 세션은 짧게 살아야 합니다. 오래 사는 세션은 한 번이라도 건드린 모든 객체를 누적하여 메모리 누수처럼 동작합니다. 표준 범위는 **HTTP 요청 1개 = 세션 1개**이며, 이는 정확히 레슨 03 §A.3의 FastAPI `yield` 기반 의존성이 강제하는 것입니다.

#### B.2 작업 단위 commit

세션 안에서 `add(obj)`, 속성 수정, `delete(obj)`를 합니다 — 그러나 아직 SQL이 가지 않습니다. SQLAlchemy는 변경을 모은 뒤 `commit()`에서

1. 의존성 순서로(부모를 자식보다 먼저) `INSERT`/`UPDATE`/`DELETE` 문을 flush합니다.
2. 데이터베이스 트랜잭션을 닫는 `COMMIT` 한 번을 발행합니다.
3. 로드된 객체를 만료시켜 다음 접근 시 신선한 데이터를 다시 가져오게 합니다.

요청 핸들러가 `commit()` 이전에 예외를 던지면 세션의 `rollback()`이 모든 미정 변경을 폐기합니다. 그래서 FastAPI 의존성 패턴 `yield session; finally: session.close()`만으로 트랜잭션 안전성에 충분합니다 — commit이 없으면 쓰기도 없습니다, 끝.

#### B.3 Async 세션: 같은 모델, await로

`AsyncSession`은 같은 작업 단위지만, 네트워크를 건드리는 모든 메서드 — `execute()`, `commit()`, `flush()`, `refresh()` — 가 `await`해야 하는 코루틴입니다. Identity map은 그대로이고, 작업 단위 의미도 그대로입니다. 변경되는 것은 세션이 *정확히 한 태스크에서만 동시에* 사용되어야 한다는 점입니다 — 동시 `asyncio.gather` 호출 사이에서 하나를 공유하는 것은 정의되지 않은 동작입니다.

### C. 로딩 전략과 N+1 문제

세 번째 관심사는 관계 속성이 실제로 SQL을 발화하는 시점입니다. 이를 잘못 이해하면 "단순한" 루프가 수백 개의 쿼리가 됩니다.

#### C.1 Lazy 로딩: 기본값의 함정

기본적으로 `user.posts` 같은 관계는 *lazy*입니다. SQLAlchemy는 속성에 처음 접근할 때 별도의 `SELECT * FROM posts WHERE user_id = ?`를 발화합니다. 편리하긴 한데, 이렇게 쓰면

```python
users = await session.scalars(select(User).limit(100))
for user in users:
    print(user.posts)  # 사용자당 추가 쿼리 1개
```

이것이 **N+1**입니다. 사용자 100명을 위한 쿼리 1개에, 그들의 posts를 위한 N=100개 쿼리. 한두 번이면 끝날 일에 101번의 라운드트립.

비동기는 이를 더 나쁘게 만듭니다. SQLAlchemy의 비동기 관계 접근은 마법처럼 암묵 쿼리를 발화할 수 없습니다 — 대신 `MissingGreenlet`을 던집니다. 그래서 `AsyncSession`에서 N+1은 시끄럽게 실패하는데, 역설적으로 이는 기능입니다. 로딩 전략을 강제로 선택하게 만듭니다.

#### C.2 Eager 로딩 전략

관련된 행을 미리 가져오는 두 가지 주된 전략:

| 전략 | SQL 패턴 | 적합한 경우 |
|----------|-------------|----------|
| `selectinload(User.posts)` | 사용자 쿼리 1개, posts에 대한 `WHERE user_id IN (...)` 쿼리 1개 | 일대다; 큰 부모 집합 |
| `joinedload(User.posts)` | `LEFT OUTER JOIN posts ON ...`이 있는 단일 SQL | 다대일; 작은 결과 집합 |

`selectinload`는 부모 쿼리를 단순하게 유지하고 행 중복을 피합니다. `joinedload`는 라운드트립이 한 번이지만, 양쪽이 모두 크면 폭발할 수 있는 카르테시안 곱 모양의 결과 집합을 만듭니다.

규칙: 일대다나 다대다는 `selectinload`를, 다대일(예: `Post`와 그 단일 `Author` 로드)은 `joinedload`를 선호하세요.

#### C.3 더 깊은 원리: 로드할 것을 선언하라

N+1 문제는 lazy 접근에 의존하지 않는 순간 사라집니다. 모든 `select()`는 필요한 모든 관계를 선언해야 합니다.

```python
stmt = (
    select(User)
    .options(selectinload(User.posts).selectinload(Post.comments))
    .where(User.active == True)
)
```

쿼리는 자기 기술적이 되고(가져오는 것이 무엇인지 명시적), 예측 가능해집니다(결과 수와 무관하게 고정된 라운드트립 수).

### 이론에서 아래 코드로

뒤에 나오는 각 절은 이 틀의 한 조각을 구체화합니다.

- §1 (SQLAlchemy 2.0 + async)은 §A의 연결 풀을 소유하는 엔진을 설정합니다.
- §2 (`mapped_column` 모델)은 행을 선언합니다. §B.1의 identity map은 여기서 선언된 기본키로 키를 만듭니다.
- §3 (Depends를 통한 세션 관리)는 레슨 03 §A.3의 `yield` 기반 의존성을 통해 배관된 §B 작업 단위입니다.
- §4 (비동기 CRUD)는 §B.2 commit 사이클을 구동하는 `add` / `execute` / `commit`입니다.
- §5 (관계)는 N+1 함정을 무찌르기 위해 §C.2의 `selectinload`/`joinedload`를 도입합니다.
- §6 (Alembic)은 `mapped_column` 모델이 기술하는 스키마를 진화시키는 오프라인 도구입니다.
- §7 (연결 풀링)은 §A.1 `QueuePool` 파라미터의 명시적 구성입니다.

---

## 1. 비동기 지원이 있는 SQLAlchemy 2.0

SQLAlchemy 2.0은 Python 타입 힌트와 네이티브 async/await를 수용하는 새로운 API를 도입했습니다. FastAPI와 결합하면 완전한 비동기 데이터베이스 스택을 제공합니다.

### 설치

```bash
# 핵심 SQLAlchemy + 비동기 PostgreSQL 드라이버
pip install sqlalchemy[asyncio] asyncpg

# SQLite 비동기 (개발/테스트에 적합)
pip install aiosqlite

# 마이그레이션을 위한 Alembic
pip install alembic
```

### 아키텍처

```
┌──────────────────────────────────────────┐
│  FastAPI 엔드포인트                        │
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
│  AsyncEngine (연결 풀)                    │
│  - 데이터베이스 연결 풀 관리              │
│  - 실패 시 재연결 처리                    │
└──────────────┬───────────────────────────┘
               │
               ▼
┌──────────────────────────────────────────┐
│  데이터베이스 (PostgreSQL / SQLite)        │
└──────────────────────────────────────────┘
```

### 엔진과 세션 팩토리 설정

```python
# app/database.py
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# 비동기 엔진은 비동기 드라이버를 사용합니다 (PostgreSQL은 asyncpg, SQLite는 aiosqlite)
# echo=True는 모든 SQL 문을 로깅합니다 -- 디버깅에 유용하고, 프로덕션에서는 비활성화
DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/mydb"

engine = create_async_engine(
    DATABASE_URL,
    echo=True,       # SQL 쿼리 로깅 (프로덕션에서는 비활성화)
    pool_size=5,     # 유지할 영구 연결 수
    max_overflow=10, # 풀이 고갈됐을 때 추가 연결 수
)

# 세션 팩토리: 새 AsyncSession 인스턴스를 생성합니다
# expire_on_commit=False는 커밋 후 속성 접근 오류를 방지합니다
# (이 설정 없이는 커밋 후 user.name에 접근하면 지연 로드가 트리거되는데,
# 비동기에서는 동기 DB 호출이 필요하기 때문에 실패합니다)
async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)
```

---

## 2. mapped_column으로 모델 정의

SQLAlchemy 2.0은 타입 안전한 모델 정의를 위해 `Mapped`와 `mapped_column`을 사용합니다. 이는 이전의 `Column()` 구문을 대체합니다.

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
    """모든 모델의 기반 클래스. DeclarativeBase는 이전의
    declarative_base() 함수를 대체하며 타입 어노테이션을 기본으로 지원합니다."""
    pass

class User(Base):
    __tablename__ = "users"

    # Mapped[int]는 Python 타입과 컬럼 타입을 모두 선언합니다
    # mapped_column()은 컬럼 레벨 세부 사항을 구성합니다
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(
        String(50),
        unique=True,
        index=True,  # 사용자명으로 빠른 조회를 위한 인덱스
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(String(255))

    # 선택적 필드는 Mapped[Optional[...]] 또는 Mapped[... | None]을 사용
    full_name: Mapped[str | None] = mapped_column(String(100), default=None)
    bio: Mapped[str | None] = mapped_column(Text, default=None)
    is_active: Mapped[bool] = mapped_column(default=True)

    # 서버 사이드 기본값: DB가 이 값들을 생성합니다
    # func.now()는 PostgreSQL에서는 NOW(), SQLite에서는 CURRENT_TIMESTAMP로 변환됩니다
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        onupdate=func.now(),  # 행 수정 시 자동 업데이트
    )

    # 게시물과의 관계 (일대다)
    # back_populates는 양방향 링크를 생성합니다
    posts: Mapped[list["Post"]] = relationship(
        back_populates="author",
        cascade="all, delete-orphan",  # 사용자 삭제 시 게시물도 삭제
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

    # 외래 키: 각 게시물을 작성자에 연결
    author_id: Mapped[int] = mapped_column(ForeignKey("users.id"))

    # 사용자와의 역방향 관계
    author: Mapped["User"] = relationship(back_populates="posts")

    # 태그와의 다대다 관계 (연결 테이블을 통해)
    tags: Mapped[list["Tag"]] = relationship(
        secondary="post_tags",
        back_populates="posts",
    )
```

### Pydantic 스키마 (ORM 모델과 분리)

```python
# app/schemas.py
from datetime import datetime
from pydantic import BaseModel, Field

# --- 사용자 스키마 ---
class UserCreate(BaseModel):
    """입력 스키마: 클라이언트가 사용자 생성을 위해 보내는 데이터."""
    username: str = Field(min_length=3, max_length=50)
    email: str = Field(pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None

class UserResponse(BaseModel):
    """출력 스키마: 클라이언트에 반환하는 데이터.
    참고: hashed_password는 의도적으로 제외됩니다."""
    id: int
    username: str
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime

    # ORM User 객체에서 직접 UserResponse를 생성할 수 있도록 허용
    model_config = {"from_attributes": True}

class UserUpdate(BaseModel):
    """부분 업데이트 스키마: 모든 필드가 선택 사항."""
    full_name: str | None = None
    bio: str | None = None
    email: str | None = Field(default=None, pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")

# --- 게시물 스키마 ---
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

## 3. 데이터베이스 세션 관리

데이터베이스 세션은 요청마다 생성되고 그 후에 닫혀야 합니다. yield 의존성을 사용한 FastAPI의 의존성 주입이 이를 깔끔하게 처리합니다.

```python
# app/dependencies.py
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from .database import async_session_factory

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """각 요청에 대한 데이터베이스 세션을 yield합니다.
    성공 시 세션이 커밋되고, 예외 시 롤백됩니다.

    이것은 'yield 의존성'입니다 -- FastAPI는 요청이 시작될 때
    yield 전의 코드를 실행하고, 요청이 끝날 때 yield 후의 코드를 실행합니다."""
    async with async_session_factory() as session:
        try:
            yield session
            # 엔드포인트가 오류 없이 완료되면 트랜잭션을 커밋합니다
            await session.commit()
        except Exception:
            # 예외 발생 시 부분 쓰기를 방지하기 위해 롤백합니다
            await session.rollback()
            raise
```

### 엔드포인트에서 세션 사용

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
    db: AsyncSession = Depends(get_db),  # 주입된 세션
):
    """db 세션은 get_db가 생성하고, 여기서 사용되며,
    응답 후에 자동으로 닫히거나 커밋됩니다."""
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(new_user)
    # flush는 생성된 id를 얻기 위해 INSERT를 DB에 밀어 넣지만,
    # 트랜잭션을 커밋하지는 않습니다 (get_db가 처리함)
    await db.flush()
    return new_user
```

---

## 4. 비동기 CRUD 작업

### 생성(Create)

```python
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from .models import User, Post
from .schemas import UserCreate, PostCreate

async def create_user(db: AsyncSession, user_data: UserCreate) -> User:
    """새 사용자를 생성합니다. db.add()는 객체를 스테이징하고,
    flush()는 자동 생성 ID를 얻기 위해 INSERT를 보냅니다."""
    user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hash_password(user_data.password),
        full_name=user_data.full_name,
    )
    db.add(user)
    await db.flush()     # 커밋 없이 ID 생성
    await db.refresh(user)  # 서버 생성 필드 재로드 (created_at)
    return user
```

### 읽기(Read, 단일 및 목록)

```python
async def get_user_by_id(db: AsyncSession, user_id: int) -> User | None:
    """기본 키로 단일 사용자를 가져옵니다.
    SQLAlchemy 2.0은 query() 대신 select()를 사용합니다."""
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
    """페이지네이션과 선택적 필터링으로 사용자를 나열합니다.
    제공된 필터에 따라 쿼리를 동적으로 구성합니다."""
    query = select(User).offset(skip).limit(limit).order_by(User.created_at.desc())

    if is_active is not None:
        query = query.where(User.is_active == is_active)

    result = await db.execute(query)
    return list(result.scalars().all())
```

### 수정(Update)

```python
async def update_user(
    db: AsyncSession,
    user_id: int,
    update_data: dict,
) -> User | None:
    """부분 업데이트: update_data에 있는 필드만 수정합니다.
    Pydantic 모델의 exclude_unset=True는 클라이언트가 명시적으로
    보낸 필드만 제공합니다."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return None

    # 명시적으로 제공된 필드만 업데이트
    for field, value in update_data.items():
        setattr(user, field, value)

    await db.flush()
    await db.refresh(user)
    return user
```

### 삭제(Delete)

```python
async def delete_user(db: AsyncSession, user_id: int) -> bool:
    """사용자를 삭제합니다. 사용자가 존재하고 삭제되면 True를 반환합니다."""
    user = await get_user_by_id(db, user_id)
    if not user:
        return False
    await db.delete(user)
    await db.flush()
    return True
```

### 엔드포인트에서 CRUD 사용

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
    # exclude_unset=True는 클라이언트가 보낸 필드만 업데이트하도록 보장합니다
    # 이것은 "필드가 전송되지 않음"과 "필드가 None으로 설정됨"을 구분합니다
    update_dict = user_data.model_dump(exclude_unset=True)
    user = await update_user(db, user_id, update_dict)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## 5. 관계(Relationships)

### 일대다(User가 많은 Post를 가짐)

이미 위의 모델에 정의되어 있습니다. 관계와 함께 쿼리하는 방법은 다음과 같습니다:

```python
from sqlalchemy.orm import selectinload

async def get_user_with_posts(db: AsyncSession, user_id: int) -> User | None:
    """N+1 쿼리 문제를 방지하기 위해 게시물을 즉시 로드합니다.
    selectinload 없이는 user.posts에 접근하면 지연 로드가 트리거되는데,
    이는 비동기 컨텍스트에서 실패합니다."""
    result = await db.execute(
        select(User)
        .where(User.id == user_id)
        .options(selectinload(User.posts))  # 단일 쿼리로 게시물 로드
    )
    return result.scalar_one_or_none()
```

### 다대다(게시물과 태그)

```python
# app/models.py (계속)
from sqlalchemy import Table, Column, Integer, ForeignKey

# 연결 테이블: ORM 모델 불필요, 테이블만 필요
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

### 다대다 쿼리

```python
async def get_posts_by_tag(db: AsyncSession, tag_name: str) -> list[Post]:
    """특정 태그가 있는 모든 게시물을 찾습니다.
    조인은 다음을 탐색합니다: Post → post_tags → Tag."""
    result = await db.execute(
        select(Post)
        .join(Post.tags)
        .where(Tag.name == tag_name)
        .options(selectinload(Post.tags))  # 각 게시물의 모든 태그도 로드
    )
    return list(result.scalars().unique().all())

async def add_tag_to_post(db: AsyncSession, post_id: int, tag_name: str):
    """게시물에 태그를 추가합니다. 태그가 없으면 생성합니다."""
    # 태그 조회 또는 생성
    result = await db.execute(select(Tag).where(Tag.name == tag_name))
    tag = result.scalar_one_or_none()
    if not tag:
        tag = Tag(name=tag_name)
        db.add(tag)
        await db.flush()

    # 태그와 함께 게시물 로드
    result = await db.execute(
        select(Post)
        .where(Post.id == post_id)
        .options(selectinload(Post.tags))
    )
    post = result.scalar_one_or_none()
    if not post:
        raise ValueError(f"Post {post_id} not found")

    # 태그 추가 (SQLAlchemy가 연결 테이블을 처리)
    if tag not in post.tags:
        post.tags.append(tag)
        await db.flush()
```

### 로딩 전략

| 전략 | 메서드 | 생성되는 SQL | 사용 시기 |
|----------|--------|--------------|----------|
| 지연 (기본) | 속성 접근 | 접근마다 별도 쿼리 | 동기만 가능, 비동기에서 거의 사용 안 함 |
| Select-in | `selectinload()` | IN 절이 있는 추가 SELECT 1개 | 비동기의 기본 선택 |
| Joined | `joinedload()` | 단일 JOIN 쿼리 | 일대일 또는 소규모 일대다 |
| Subquery | `subqueryload()` | 추가 서브쿼리 1개 | 대규모 컬렉션 |

---

## 6. 마이그레이션을 위한 Alembic

Alembic은 데이터베이스 스키마 변경(마이그레이션)을 추적하여 프로덕션에서 스키마를 안전하게 발전시킬 수 있게 합니다.

### 설정

```bash
# 프로젝트에 Alembic 초기화
alembic init alembic

# 이것이 생성됩니다:
# alembic/
#   env.py          # 구성
#   versions/       # 마이그레이션 스크립트
# alembic.ini       # 연결 문자열과 설정
```

### 비동기를 위한 Alembic 구성

```python
# alembic/env.py
import asyncio
from logging.config import fileConfig
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config
from alembic import context

from app.models import Base          # Base 가져오기
from app.database import DATABASE_URL  # DB URL 가져오기

config = context.config
config.set_main_option("sqlalchemy.url", DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Alembic은 이 메타데이터를 데이터베이스와 비교하여 변경 사항을 감지합니다
target_metadata = Base.metadata

def run_migrations_offline() -> None:
    """데이터베이스에 연결하지 않고 SQL 스크립트를 생성합니다."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()

def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()

async def run_async_migrations() -> None:
    """비동기 엔진을 사용하여 마이그레이션을 실행합니다."""
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

### 마이그레이션 생성 및 실행

```bash
# 모델을 현재 DB 스키마와 비교하여 마이그레이션 자동 생성
alembic revision --autogenerate -m "create users and posts tables"

# 모든 대기 중인 마이그레이션 적용
alembic upgrade head

# 마지막 마이그레이션 롤백
alembic downgrade -1

# 마이그레이션 기록 보기
alembic history --verbose

# 현재 데이터베이스 버전 표시
alembic current
```

### 생성된 마이그레이션 예제

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
    # 더 깔끔한 마이그레이션을 위해 테이블 생성 후 인덱스 생성
    op.create_index("ix_users_username", "users", ["username"])
    op.create_index("ix_users_email", "users", ["email"])

def downgrade() -> None:
    op.drop_index("ix_users_email")
    op.drop_index("ix_users_username")
    op.drop_table("users")
```

---

## 7. 연결 풀링(Connection Pooling)

연결 풀링(Connection pooling)은 요청마다 새 연결을 생성하는 대신 데이터베이스 연결을 재사용합니다. 이는 연결 오버헤드를 극적으로 줄입니다.

```python
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://user:pass@localhost:5432/mydb",

    # 풀 크기: 영구적으로 열어 두는 연결 수
    # 5부터 시작하고 부하 테스트에 따라 증가
    pool_size=5,

    # 오버플로우: 풀이 가득 찼을 때 생성되는 임시 연결
    # 사용 후 닫히고 풀로 돌아오지 않음
    max_overflow=10,

    # 오래된 연결을 방지하기 위해 30분 후 연결 재활용
    # PostgreSQL의 기본 유휴 타임아웃은 종종 더 길지만,
    # 로드 밸런서(PgBouncer, AWS RDS Proxy 등)는 유휴 연결을 더 일찍 닫을 수 있음
    pool_recycle=1800,

    # 요청에 전달하기 전에 연결 상태 테스트
    # 약간의 오버헤드를 추가하지만 "연결 재설정" 오류를 방지
    pool_pre_ping=True,

    # 풀에서 연결을 기다리는 시간
    # 무한정 기다리는 대신 오류 발생
    pool_timeout=30,
)
```

### 풀 크기 가이드라인

```
총 연결 수 = pool_size + max_overflow = 5 + 10 = 워커당 15

Uvicorn 워커 4개: 15 * 4 = 60개의 총 연결

PostgreSQL 기본 max_connections = 100
관리 연결과 마이그레이션을 위한 여유 공간 확보

경험 법칙: 총 연결 수 < max_connections의 80%
```

### 풀 상태 모니터링

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/debug/pool")
async def pool_status():
    """모니터링을 위한 연결 풀 메트릭을 노출합니다.
    개발 환경이나 관리자 인증 뒤에서만 활성화하세요."""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_out": pool.checkedout(),  # 현재 사용 중
        "checked_in": pool.checkedin(),    # 풀에서 사용 가능
        "overflow": pool.overflow(),       # 임시 연결
    }
```

---

## 8. 연습 문제

### 문제 1: 완전한 CRUD를 갖춘 블로그 API

다음을 포함하는 완전한 블로그 API를 구축하세요:
- `User` 모델: id, username, email, created_at
- `Post` 모델: id, title, content, is_published, author_id, created_at
- 엔드포인트: 사용자 생성, 게시물 생성 (사용자에 연결), 게시물 목록 (페이지네이션과 작성자 필터 포함), 작성자 정보가 있는 단일 게시물 조회, 게시물 수정, 게시물 삭제
- 게시물 응답에 작성자 데이터를 포함하기 위해 `selectinload` 사용
- 생성, 수정, 응답에 별도의 Pydantic 스키마 사용

### 문제 2: 데이터베이스 세션 테스팅

다음을 수행하는 `get_db` 의존성을 작성하세요:
1. 요청마다 새 `AsyncSession` 생성
2. 성공 시 커밋, 예외 시 롤백
3. 트랜잭션 기간 로깅
4. 테스트에서 테스트 데이터베이스를 사용하도록 재정의 가능

그런 다음 `get_db`를 재정의하여 인메모리 SQLite 데이터베이스를 사용하는 테스트를 작성하세요.

### 문제 3: CRUD가 있는 다대다

블로그 API를 다음으로 확장하세요:
- `Tag` 모델과 `post_tags` 연결 테이블
- `POST /posts/{post_id}/tags` -- 게시물에 태그 추가
- `DELETE /posts/{post_id}/tags/{tag_id}` -- 태그 제거
- `GET /tags/{tag_name}/posts` -- 특정 태그가 있는 모든 게시물 나열
- 태그가 고유한지 확인 (기존 태그를 재사용하고 중복 생성 방지)

### 문제 4: Alembic 마이그레이션 워크플로우

빈 데이터베이스에서 시작하여:
1. User와 Post 테이블로 초기 마이그레이션 생성
2. User에 `bio` 컬럼을 추가하고 Post에 `view_count` 컬럼(기본값 0)을 추가하는 두 번째 마이그레이션 생성
3. `Post.created_at`에 인덱스를 추가하는 세 번째 마이그레이션 생성
4. head로 업그레이드, 한 단계 다운그레이드, 다시 업그레이드 연습

각 단계에서 실행하는 명령어를 문서화하세요.

### 문제 5: 연결 풀 튜닝

다음 특성을 가진 API가 있습니다:
- 초당 500개의 요청 수신
- 각 요청은 ~50ms의 데이터베이스 시간이 걸림
- Uvicorn 워커 4개로 실행
- PostgreSQL `max_connections`는 200으로 설정

다음을 계산하세요:
1. 워커당 필요한 동시 DB 연결 수는?
2. `pool_size`와 `max_overflow` 값은 어떻게 설정하겠습니까?
3. 풀이 고갈되면 어떻게 됩니까? 어떻게 감지하고 처리하겠습니까?
4. PostgreSQL 앞에 PgBouncer를 권장합니까? 이유는?

---

## 9. 참고 자료

- [SQLAlchemy 2.0 문서](https://docs.sqlalchemy.org/en/20/)
- [SQLAlchemy 비동기 문서](https://docs.sqlalchemy.org/en/20/orm/extensions/asyncio.html)
- [Alembic 튜토리얼](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [FastAPI SQL 데이터베이스 가이드](https://fastapi.tiangolo.com/tutorial/sql-databases/)
- [asyncpg 문서](https://magicstack.github.io/asyncpg/)
- [Pydantic model_config](https://docs.pydantic.dev/latest/concepts/config/)

---

**이전**: [FastAPI 고급](./03_FastAPI_Advanced.md) | **다음**: [FastAPI 테스팅](./05_FastAPI_Testing.md)
