# 레슨 16: 데이터베이스 테스트

**이전**: [Testing Async Code](./15_Testing_Async_Code.md) | **다음**: [Testing Legacy Code](./17_Testing_Legacy_Code.md)

---

데이터베이스 코드는 올바르게 테스트하기가 매우 어렵습니다. 애플리케이션 로직과 고유한 규칙을 가진 외부 시스템 -- SQL 방언, 트랜잭션 의미론, 제약 조건 적용, 마이그레이션 순서 -- 의 경계에 위치합니다. 많은 팀이 데이터베이스 테스트를 아예 건너뛰거나(프로덕션 데이터 손상으로 고생), 느리고 스키마 세부 사항에 밀접하게 결합되어 매 마이그레이션마다 깨지는 취약한 테스트를 작성합니다. 이 레슨은 SQLAlchemy, factory_boy, Alembic을 사용하여 속도, 현실성, 유지보수성의 균형을 잡으며 데이터베이스 상호작용을 효과적으로 테스트하는 방법을 보여줍니다.

**난이도**: ⭐⭐⭐⭐

**사전 요구사항**:
- pytest를 사용한 Python 테스트 (레슨 02-04)
- 기본 SQL 및 관계형 데이터베이스 개념
- SQLAlchemy ORM에 대한 익숙함 (기본 모델, 세션, 쿼리)
- 데이터베이스 트랜잭션에 대한 이해

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다양한 테스트 시나리오에 대해 인메모리 SQLite와 실제 테스트 데이터베이스 중 적절한 것을 선택할 수 있다
2. 빠르고 격리된 데이터베이스 테스트를 위한 테스트별 트랜잭션 롤백을 구현할 수 있다
3. factory_boy를 사용하여 보일러플레이트 없이 현실적인 테스트 데이터를 생성할 수 있다
4. Faker를 사용하여 현실적인 가짜 데이터를 생성할 수 있다
5. Alembic을 사용하여 데이터베이스 마이그레이션을 테스트할 수 있다
6. 제약 조건과 참조 무결성을 검증하는 데이터 무결성 테스트를 작성할 수 있다

---

## 1. 데이터베이스 테스트 전략

데이터베이스 코드를 테스트하는 세 가지 주요 전략이 있으며, 각각 다른 트레이드오프를 가집니다:

| 전략 | 속도 | 현실성 | 격리성 | 적합한 용도 |
|---|---|---|---|---|
| 인메모리 SQLite | 매우 빠름 | 낮음 | 높음 | 단위 테스트, 단순 쿼리 |
| 트랜잭션 롤백 | 빠름 | 높음 | 높음 | 통합 테스트 |
| 전용 테스트 데이터베이스 | 보통 | 가장 높음 | 보통 | 전체 통합, 마이그레이션 |

### 1.1 인메모리 SQLite

프로덕션 데이터베이스를 테스트용 인메모리 SQLite 인스턴스로 대체합니다:

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

**장점**: 극도로 빠르고, 외부 의존성이 없으며, 완벽한 격리성을 제공합니다.

**한계**: SQLite는 모든 PostgreSQL 기능(배열, JSON 연산자, 특정 구문의 CTE, `ENUM` 타입, 고급 제약 조건 구문)을 지원하지 않습니다. SQLite에서 통과하는 테스트가 PostgreSQL에서 실패할 수 있습니다.

### 1.2 SQLite와 PostgreSQL의 차이점

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

## 2. 테스트별 트랜잭션 롤백

통합 테스트를 위한 가장 실용적인 전략: 각 테스트 전에 트랜잭션을 시작하고 이후에 롤백합니다. 데이터베이스 상태가 실제로 변경되지 않으므로 테스트가 빠르고 격리됩니다.

### 2.1 SQLAlchemy 구현

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

### 2.2 트랜잭션 Fixture 사용

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

## 3. factory_boy를 사용한 테스트 데이터 팩토리

수작업으로 테스트 데이터를 빌드하는 것은 지루하고 모델의 생성자 시그니처에 결합됩니다. [factory_boy](https://factoryboy.readthedocs.io/)는 합리적인 기본값과 커스터마이즈 가능한 오버라이드로 테스트 객체를 생성합니다.

### 3.1 설치

```bash
pip install factory_boy
```

### 3.2 팩토리 정의

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

### 3.3 팩토리를 테스트 세션에 연결

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

### 3.4 테스트에서 팩토리 사용

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

### 3.5 팩토리 Trait과 Sequence

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

## 4. Faker를 사용한 현실적인 데이터 생성

[Faker](https://faker.readthedocs.io/)는 테스트를 위한 현실적인 가짜 데이터를 생성합니다. factory_boy가 내부적으로 사용하지만, 직접 사용할 수도 있습니다.

### 4.1 기본 사용법

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

### 4.2 시드를 사용한 결정적 데이터

```python
Faker.seed(12345)
fake = Faker()

# Always produces the same sequence
name1 = fake.name()  # Always "John Smith" (with this seed)
name2 = fake.name()  # Always "Jane Doe" (with this seed)
```

### 4.3 로컬라이즈된 데이터

```python
# Generate data in specific locales
fake_jp = Faker("ja_JP")
fake_jp.name()     # "田中 太郎"
fake_jp.address()  # Japanese address

fake_de = Faker("de_DE")
fake_de.name()     # "Hans Mueller"
```

---

## 5. 데이터베이스 마이그레이션 테스트

데이터베이스 스키마 변경(마이그레이션)은 소프트웨어에서 가장 위험도가 높은 작업 중 하나입니다. 개발 환경에서 작동하는 마이그레이션이 데이터 의존성, 컬럼 크기, 제약 조건 위반 등으로 인해 프로덕션에서 실패할 수 있습니다.

### 5.1 Alembic 마이그레이션 테스트 전략

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

### 5.2 개별 마이그레이션 테스트

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

### 5.3 데이터 마이그레이션 테스트

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

## 6. 데이터 무결성 테스트

데이터 무결성 테스트는 데이터베이스 제약 조건, 트리거, 애플리케이션 로직이 데이터 일관성을 유지하는지 검증합니다.

### 6.1 제약 조건 테스트

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

### 6.2 캐스케이드 동작 테스트

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

### 6.3 CHECK 제약 조건 테스트

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

## 7. 쿼리 성능 테스트

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

## 8. 모범 사례 요약

| 실천 사항 | 이유 |
|---|---|
| 트랜잭션 롤백 사용 | 빠르고, 격리되고, 현실적 |
| 원시 생성자가 아닌 팩토리 사용 | 유지보수 가능하고, 읽기 쉬운 테스트 데이터 |
| 제약 조건을 명시적으로 테스트 | 누락된 제약 조건을 조기에 발견 |
| 마이그레이션을 순방향과 역방향으로 테스트 | 배포 실패 방지 |
| 현실성을 위해 Faker로 시드 | 인코딩, 길이, 형식 관련 버그 발견 |
| 단순 테스트에는 SQLite 유지 | DB 기능을 사용하지 않는 로직에 최대 속도 |
| 통합 테스트에는 실제 DB 사용 | 방언별 이슈 발견 |

---

## 연습 문제

1. **트랜잭션 롤백 Fixture**: SQLAlchemy 프로젝트를 위한 완전한 트랜잭션 롤백 fixture를 구현하십시오. 데이터를 생성하는 세 개의 테스트를 작성하고, 각 테스트가 깨끗한 데이터베이스에서 시작하는지 검증하십시오.

2. **팩토리 설계**: `User`, `Post`, `Tag`, `Comment` 모델을 가진 블로그 애플리케이션을 위한 팩토리를 생성하십시오. `admin`, `draft_post`, `spam_comment` trait을 포함하십시오. 각 trait을 보여주는 테스트를 작성하십시오.

3. **마이그레이션 테스트**: base에서 head까지 모든 Alembic 마이그레이션을 적용하고, 스키마가 예상 상태와 일치하는지 검증한 후, base로 다시 다운그레이드하는 테스트를 작성하십시오.

4. **제약 조건 커버리지**: 주어진 데이터베이스 모델 세트에 대해 모든 제약 조건(unique, not-null, foreign key, check)을 식별하고, 각 제약 조건이 실제로 적용되는지 검증하는 테스트를 작성하십시오.

5. **성능 기준선**: 10,000명의 사용자를 생성하는 팩토리를 만든 다음, 세 가지 일반적인 쿼리(이메일 검색, 생성일 기준 목록, 역할별 카운트)에 대한 벤치마크 테스트를 작성하십시오. 최대 허용 실행 시간을 설정하십시오.

---

**License**: CC BY-NC 4.0
