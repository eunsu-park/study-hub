# 통합 테스트 (Integration Testing)

**이전**: [테스트 주도 개발](./06_Test_Driven_Development.md) | **다음**: [API 테스트](./08_API_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 컴포넌트 간 상호작용을 검증하는 통합 테스트를 설계할 수 있다
2. 테스트 데이터베이스와 함께 SQLAlchemy를 사용하여 데이터베이스 상호작용을 테스트할 수 있다
3. testcontainers를 사용하여 Docker 기반 테스트 환경을 구성할 수 있다
4. 빠르고 격리된 DB 테스트를 위한 트랜잭션 롤백 패턴을 적용할 수 있다
5. 통합 테스트 스위트 전반에 걸쳐 테스트 데이터를 효과적으로 관리할 수 있다

---

## 통합 테스트란 무엇인가?

단위 테스트는 컴포넌트를 격리하여 검증합니다. 통합 테스트는 컴포넌트들이 *함께* 작동하는지 검증합니다. "통합"은 다음과 같은 관계를 의미할 수 있습니다:

- 코드와 데이터베이스
- 애플리케이션 내 두 모듈
- 서비스와 외부 API
- 애플리케이션과 메시지 큐

통합 테스트는 단위 테스트보다 비용이 높지만(더 느리고, 인프라가 필요함), 단위 테스트로는 발견할 수 없는 범주의 버그를 잡아냅니다: 연결 버그, 직렬화 문제, SQL 오류, 설정 실수 등이 그것입니다.

```
Unit test:     [Function] ─── assertions
Integration:   [Module A] ──→ [Module B] ──→ [Database] ─── assertions
```

### 통합 테스트를 작성해야 하는 경우

- 데이터베이스 쿼리 및 ORM 상호작용
- 모듈 간 워크플로우 (서비스가 리포지토리를 호출하고 리포지토리가 모델을 호출)
- 설정 로딩 및 환경 변수 파싱
- 파일 I/O 작업
- 캐시 무효화 로직
- 메시지 직렬화/역직렬화

---

## 데이터베이스 상호작용 테스트

### SQLAlchemy로 테스트 데이터베이스 구성하기

Python 애플리케이션에서 가장 일반적인 통합 테스트 시나리오는 데이터베이스 상호작용입니다. 빠른 테스트를 위해 인메모리 SQLite 데이터베이스를 사용하거나, 프로덕션과 동일한 환경의 테스트를 위해 Docker화된 PostgreSQL을 사용합니다.

```python
# models.py
from sqlalchemy import Column, Integer, String, Float, Boolean
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    in_stock = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Product(name={self.name!r}, price={self.price})>"
```

```python
# repository.py
from sqlalchemy.orm import Session
from models import Product


class ProductRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, product: Product) -> Product:
        self.session.add(product)
        self.session.flush()  # Assigns ID without committing
        return product

    def get_by_id(self, product_id: int) -> Product | None:
        return self.session.get(Product, product_id)

    def find_by_name(self, name: str) -> list[Product]:
        return (
            self.session.query(Product)
            .filter(Product.name.ilike(f"%{name}%"))
            .all()
        )

    def find_in_stock(self) -> list[Product]:
        return (
            self.session.query(Product)
            .filter(Product.in_stock == True)
            .all()
        )

    def update_price(self, product_id: int, new_price: float) -> Product:
        product = self.get_by_id(product_id)
        if product is None:
            raise ValueError(f"Product {product_id} not found")
        if new_price < 0:
            raise ValueError("Price cannot be negative")
        product.price = new_price
        self.session.flush()
        return product

    def delete(self, product_id: int) -> None:
        product = self.get_by_id(product_id)
        if product:
            self.session.delete(product)
            self.session.flush()
```

### 데이터베이스 테스트를 위한 테스트 픽스처

```python
# conftest.py
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base
from repository import ProductRepository


@pytest.fixture(scope="module")
def engine():
    """Create an in-memory SQLite database engine."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    """Provide a transactional database session that rolls back after each test."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture
def repo(session):
    """Provide a ProductRepository with a clean session."""
    return ProductRepository(session)


@pytest.fixture
def sample_products(repo):
    """Seed the database with sample products."""
    from models import Product
    products = [
        Product(name="Laptop", price=999.99, in_stock=True),
        Product(name="Mouse", price=29.99, in_stock=True),
        Product(name="Keyboard", price=79.99, in_stock=False),
        Product(name="Monitor", price=349.99, in_stock=True),
    ]
    for p in products:
        repo.add(p)
    return products
```

### 데이터베이스 통합 테스트 작성하기

```python
# test_repository.py
import pytest
from models import Product


class TestProductRepository:
    def test_add_product(self, repo):
        product = Product(name="Webcam", price=59.99)
        result = repo.add(product)
        assert result.id is not None
        assert result.name == "Webcam"

    def test_get_by_id(self, repo):
        product = repo.add(Product(name="Headset", price=89.99))
        found = repo.get_by_id(product.id)
        assert found is not None
        assert found.name == "Headset"

    def test_get_nonexistent_returns_none(self, repo):
        assert repo.get_by_id(9999) is None

    def test_find_by_name_case_insensitive(self, repo, sample_products):
        results = repo.find_by_name("laptop")
        assert len(results) == 1
        assert results[0].name == "Laptop"

    def test_find_by_name_partial_match(self, repo, sample_products):
        results = repo.find_by_name("board")
        assert len(results) == 1
        assert results[0].name == "Keyboard"

    def test_find_in_stock(self, repo, sample_products):
        results = repo.find_in_stock()
        assert len(results) == 3  # Laptop, Mouse, Monitor
        assert all(p.in_stock for p in results)

    def test_update_price(self, repo):
        product = repo.add(Product(name="Cable", price=9.99))
        updated = repo.update_price(product.id, 14.99)
        assert updated.price == 14.99

    def test_update_price_negative_raises(self, repo):
        product = repo.add(Product(name="Cable", price=9.99))
        with pytest.raises(ValueError, match="negative"):
            repo.update_price(product.id, -5.00)

    def test_update_nonexistent_raises(self, repo):
        with pytest.raises(ValueError, match="not found"):
            repo.update_price(9999, 10.00)

    def test_delete_product(self, repo):
        product = repo.add(Product(name="Temp", price=1.00))
        repo.delete(product.id)
        assert repo.get_by_id(product.id) is None


class TestTransactionIsolation:
    """테스트 간 상태가 누출되지 않는지 검증합니다."""

    def test_add_product_a(self, repo):
        repo.add(Product(name="Product A", price=10.00))
        results = repo.find_by_name("Product A")
        assert len(results) == 1

    def test_product_a_does_not_persist(self, repo):
        """이 테스트는 test_add_product_a 이후에 실행됩니다.
        이전 테스트가 롤백되었기 때문에 Product A가 존재하지 않아야 합니다."""
        results = repo.find_by_name("Product A")
        assert len(results) == 0
```

---

## 트랜잭션 롤백 패턴

트랜잭션 롤백 패턴은 빠르고 격리된 데이터베이스 테스트의 핵심입니다. 각 테스트마다 테이블을 생성하고 삭제하는 대신 다음과 같이 합니다:

1. 각 테스트 전에 트랜잭션을 시작합니다
2. 해당 트랜잭션 내에서 테스트가 모든 작업을 수행합니다
3. 테스트 후 트랜잭션을 롤백합니다

이 방식은 데이터가 실제로 디스크에 기록되지 않거나(인메모리 데이터베이스의 경우) 커밋되지 않기 때문에(실제 데이터베이스의 경우) 빠릅니다.

```python
@pytest.fixture
def session(engine):
    """Transaction rollback pattern."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    yield session

    # Teardown: undo everything the test did
    session.close()
    transaction.rollback()
    connection.close()
```

### 중첩 트랜잭션

애플리케이션 코드가 `session.commit()`을 호출하는 경우, 외부 트랜잭션이 여전히 롤백할 수 있도록 중첩 트랜잭션(세이브포인트)이 필요합니다:

```python
from sqlalchemy import event

@pytest.fixture
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)

    # When the application code calls session.commit(),
    # restart a nested transaction instead of actually committing
    @event.listens_for(session, "after_transaction_end")
    def restart_savepoint(session, transaction):
        if transaction.nested and not transaction._parent.nested:
            session.begin_nested()

    session.begin_nested()

    yield session

    session.close()
    transaction.rollback()
    connection.close()
```

---

## Testcontainers: Docker 기반 테스트 환경

SQLite는 테스트에 편리하지만, PostgreSQL(또는 MySQL)과 미묘하게 다릅니다: 타입 강제, 잠금 동작, JSON 지원, 전문 검색 등이 그렇습니다. 프로덕션과 동일한 환경의 테스트를 위해 `testcontainers`를 사용하여 Docker에서 실제 데이터베이스를 구동합니다.

```bash
pip install testcontainers[postgres]
```

```python
# conftest.py
import pytest
from testcontainers.postgres import PostgresContainer
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from models import Base


@pytest.fixture(scope="session")
def postgres():
    """Start a PostgreSQL container for the entire test session."""
    with PostgresContainer("postgres:16-alpine") as postgres:
        yield postgres


@pytest.fixture(scope="session")
def engine(postgres):
    """Create an engine connected to the test PostgreSQL container."""
    engine = create_engine(postgres.get_connection_url())
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    """Transactional session with rollback."""
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
```

### 다른 서비스를 위한 Testcontainers

```python
from testcontainers.redis import RedisContainer
from testcontainers.mongodb import MongoDbContainer

@pytest.fixture(scope="session")
def redis_container():
    with RedisContainer() as redis:
        yield redis

@pytest.fixture
def redis_client(redis_container):
    import redis
    client = redis.from_url(redis_container.get_connection_url())
    yield client
    client.flushall()  # Clean up between tests
```

---

## 테스트 데이터 관리

### 픽스처보다 팩토리 사용하기

복잡한 데이터 모델의 경우, 팩토리 함수가 정적 픽스처보다 더 유연합니다:

```python
# factories.py
from models import Product, Order, OrderItem
from datetime import datetime, timedelta
import random


def create_product(
    name: str = "Test Product",
    price: float = 9.99,
    in_stock: bool = True,
    **kwargs,
) -> Product:
    return Product(name=name, price=price, in_stock=in_stock, **kwargs)


def create_order(
    customer_name: str = "Test Customer",
    items: list[dict] | None = None,
    created_at: datetime | None = None,
) -> Order:
    order = Order(
        customer_name=customer_name,
        created_at=created_at or datetime.now(),
    )
    if items:
        for item_data in items:
            order.items.append(OrderItem(**item_data))
    return order


def create_bulk_products(count: int = 10) -> list[Product]:
    """Create many products for performance testing."""
    return [
        create_product(
            name=f"Product {i}",
            price=round(random.uniform(1.0, 100.0), 2),
        )
        for i in range(count)
    ]
```

```python
# test_orders.py
from factories import create_product, create_order


def test_order_total(repo):
    laptop = repo.add(create_product(name="Laptop", price=999.99))
    mouse = repo.add(create_product(name="Mouse", price=29.99))

    order = create_order(
        customer_name="Alice",
        items=[
            {"product_id": laptop.id, "quantity": 1},
            {"product_id": mouse.id, "quantity": 2},
        ],
    )
    repo.add_order(order)

    assert order.total() == pytest.approx(1059.97)
```

### 통합 테스트 스위트를 위한 데이터베이스 시딩

대규모 통합 테스트 스위트의 경우, 데이터베이스를 한 번만 시딩하고 대부분의 테스트에서 읽기 전용 쿼리를 사용합니다:

```python
@pytest.fixture(scope="session")
def seeded_db(engine):
    """Seed database once for the entire test session."""
    session = Session(engine)

    # Bulk insert reference data
    categories = [Category(name=n) for n in ["Electronics", "Books", "Clothing"]]
    session.add_all(categories)

    # Bulk insert products
    products = create_bulk_products(100)
    session.add_all(products)

    session.commit()
    yield engine
    # Session-scoped: cleaned up when all tests finish
```

---

## 모듈 간 워크플로우 테스트

통합 테스트는 각 단계가 개별적으로 단위 테스트되었더라도, 다단계 워크플로우가 올바른 종합 결과를 생산하는지 검증하는 데 유용합니다.

```python
# service.py
class OrderService:
    def __init__(self, product_repo, order_repo, inventory_service, email_service):
        self.product_repo = product_repo
        self.order_repo = order_repo
        self.inventory = inventory_service
        self.email = email_service

    def place_order(self, customer_email, product_id, quantity):
        product = self.product_repo.get_by_id(product_id)
        if product is None:
            raise ValueError("Product not found")

        if not self.inventory.check_stock(product_id, quantity):
            raise ValueError("Insufficient stock")

        order = Order(
            customer_email=customer_email,
            product_id=product_id,
            quantity=quantity,
            total=product.price * quantity,
        )
        self.order_repo.add(order)
        self.inventory.reserve(product_id, quantity)
        self.email.send_confirmation(customer_email, order)
        return order
```

```python
# test_order_workflow.py
from unittest.mock import Mock


def test_place_order_full_workflow(session):
    """Integration test: real repos + mock external services."""
    # Real database repositories
    product_repo = ProductRepository(session)
    order_repo = OrderRepository(session)

    # Mock external services (not testing email/inventory integration here)
    inventory = Mock()
    inventory.check_stock.return_value = True
    email = Mock()

    # Seed data
    product = product_repo.add(Product(name="Widget", price=25.00))

    # Execute workflow
    service = OrderService(product_repo, order_repo, inventory, email)
    order = service.place_order("alice@test.com", product.id, 3)

    # Verify database state
    saved_order = order_repo.get_by_id(order.id)
    assert saved_order is not None
    assert saved_order.total == 75.00
    assert saved_order.quantity == 3

    # Verify external service calls
    inventory.reserve.assert_called_once_with(product.id, 3)
    email.send_confirmation.assert_called_once()
```

이 테스트는 데이터베이스 계층에 대한 통합 테스트(실제 리포지토리, 실제 세션)이면서 외부 서비스는 모킹(mocking)합니다. 이는 일반적이고 실용적인 접근 방식입니다: 제어할 수 있는 통합 경계는 테스트하고, 제어할 수 없는 경계는 모킹합니다.

---

## 통합 테스트 모범 사례

### 1. 통합 테스트와 단위 테스트 분리하기

```
tests/
├── unit/                  # Fast, no I/O
│   ├── test_calculator.py
│   └── test_validator.py
├── integration/           # Slower, requires DB
│   ├── conftest.py
│   ├── test_repository.py
│   └── test_workflow.py
```

```bash
# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Or use marks
pytest -m "not integration"
```

### 2. 통합 테스트를 집중적으로 유지하기

테스트 함수 하나당 하나의 통합 지점을 테스트합니다. 전체 애플리케이션을 하나의 테스트로 검증하는 것은 피합니다.

### 3. 현실적인 데이터 사용하기

통합 테스트는 현실적인 데이터 규모와 형태를 사용해야 합니다. 데이터베이스에 행이 하나만 있는 테스트로는 페이지네이션 버그나 N+1 쿼리 문제를 잡을 수 없습니다.

### 4. 테스트 후 정리하기

트랜잭션 롤백 패턴, `tmp_path` 픽스처, 또는 teardown에서의 명시적 정리를 사용합니다. 테스트 간 데이터 누출은 불안정한 통합 테스트의 가장 흔한 원인입니다.

---

## 연습 문제

1. **리포지토리 통합 테스트**: `add`, `get_by_email`, `update_name`, `delete` 메서드를 가진 `UserRepository`를 생성합니다. 인메모리 SQLite와 SQLAlchemy를 사용하여 통합 테스트를 작성합니다. 테스트가 서로 간섭하지 않도록 트랜잭션 롤백 패턴을 사용합니다.

2. **Testcontainers 설정**: `testcontainers`를 사용하여 PostgreSQL 컨테이너를 시작하는 `conftest.py`를 작성합니다. 간단한 테이블을 생성하고, 데이터를 삽입하고, 테스트에서 쿼리합니다. 세션이 끝난 후 컨테이너가 정리되는지 확인합니다.

3. **모듈 간 워크플로우**: 미니 전자상거래 워크플로우를 설계합니다: `CartService.add_item()` -> `CartService.checkout()` -> `OrderRepository.save()`. 실제 리포지토리를 사용하되 결제 게이트웨이는 모킹하는 통합 테스트를 작성합니다. 데이터베이스 상태와 결제 게이트웨이 호출 모두를 검증합니다.

---

**License**: CC BY-NC 4.0
