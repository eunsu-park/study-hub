# Integration Testing

**Previous**: [Test-Driven Development](./06_Test_Driven_Development.md) | **Next**: [API Testing](./08_API_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Design integration tests that verify component interactions
2. Test database interactions using SQLAlchemy with test databases
3. Use testcontainers to spin up Docker-based test environments
4. Apply the transaction rollback pattern for fast, isolated DB tests
5. Manage test data effectively across integration test suites

---

## What Are Integration Tests?

Unit tests verify components in isolation. Integration tests verify that components work *together*. The "integration" can be between:

- Your code and a database
- Two modules in your application
- Your service and an external API
- Your application and a message queue

Integration tests are more expensive than unit tests (slower, require infrastructure) but catch a category of bugs that unit tests cannot: wiring bugs, serialization issues, SQL errors, and configuration mistakes.

```
Unit test:     [Function] ─── assertions
Integration:   [Module A] ──→ [Module B] ──→ [Database] ─── assertions
```

### When to Write Integration Tests

- Database queries and ORM interactions
- Cross-module workflows (service calls repository calls model)
- Configuration loading and environment variable parsing
- File I/O operations
- Cache invalidation logic
- Message serialization/deserialization

---

## Testing Database Interactions

### Setting Up a Test Database with SQLAlchemy

The most common integration testing scenario in Python applications is database interaction. Use an in-memory SQLite database for fast tests, or a Dockerized PostgreSQL for production-parity tests.

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

### Test Fixtures for Database Tests

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

### Writing Database Integration Tests

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
    """Verify that tests do not leak state to each other."""

    def test_add_product_a(self, repo):
        repo.add(Product(name="Product A", price=10.00))
        results = repo.find_by_name("Product A")
        assert len(results) == 1

    def test_product_a_does_not_persist(self, repo):
        """This test runs after test_add_product_a.
        Product A should NOT exist because the previous test rolled back."""
        results = repo.find_by_name("Product A")
        assert len(results) == 0
```

---

## The Transaction Rollback Pattern

The transaction rollback pattern is the key to fast, isolated database tests. Instead of creating and dropping tables for each test, you:

1. Begin a transaction before each test
2. Let the test perform all its operations within that transaction
3. Roll back the transaction after the test

This approach is fast because no data is ever actually written to disk (with in-memory databases) or committed (with real databases).

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

### Nested Transactions

If your application code calls `session.commit()`, you need nested transactions (savepoints) so the outer transaction can still roll back:

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

## Testcontainers: Docker-Based Test Environments

SQLite is convenient for testing, but it differs from PostgreSQL (or MySQL) in subtle ways: type enforcement, locking behavior, JSON support, full-text search. For production-parity tests, use `testcontainers` to spin up real databases in Docker.

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

### Testcontainers for Other Services

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

## Test Data Management

### Factories Over Fixtures

For complex data models, factory functions are more flexible than static fixtures:

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

### Database Seeding for Integration Suites

For large integration test suites, pre-seed the database once and use read-only queries in most tests:

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

## Testing Cross-Module Workflows

Integration tests are valuable for verifying that a multi-step workflow produces the correct end-to-end result, even when each step is individually unit tested.

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

This test is an integration test for the database layer (real repos, real session) while mocking external services. This is a common and practical approach: test the integration boundaries you control, mock the ones you do not.

---

## Best Practices for Integration Tests

### 1. Separate Integration Tests from Unit Tests

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

### 2. Keep Integration Tests Focused

Test one integration point per test function. Do not write a single test that exercises the entire application.

### 3. Use Realistic Data

Integration tests should use realistic data volumes and shapes. A test with one row in the database will not catch pagination bugs or N+1 query problems.

### 4. Clean Up After Tests

Use the transaction rollback pattern, `tmp_path` fixtures, or explicit cleanup in teardown. Leaking test data across tests is the most common source of flaky integration tests.

---

## Exercises

1. **Repository integration tests**: Create a `UserRepository` with methods `add`, `get_by_email`, `update_name`, and `delete`. Write integration tests using SQLAlchemy with in-memory SQLite. Use the transaction rollback pattern so tests do not interfere with each other.

2. **Testcontainers setup**: Write a `conftest.py` that starts a PostgreSQL container using `testcontainers`. Create a simple table, insert data, and query it in a test. Verify that the container is cleaned up after the session.

3. **Cross-module workflow**: Design a mini e-commerce workflow: `CartService.add_item()` -> `CartService.checkout()` -> `OrderRepository.save()`. Write an integration test that uses real repositories but mocks the payment gateway. Verify both the database state and the payment gateway call.

---

**License**: CC BY-NC 4.0
