#!/bin/bash
# Exercises for Lesson 14: Test Architecture and Patterns
# Topic: Testing_and_QA
# Solutions to practice problems from the lesson.

# === Exercise 1: Test Double Classification ===
# Problem: Given five test scenarios, identify which test double type
# (dummy, stub, spy, mock, fake) is most appropriate and implement it.
exercise_1() {
    echo "=== Exercise 1: Test Double Classification ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from unittest.mock import Mock
from dataclasses import dataclass, field


# --- Scenario 1: Dummy ---
# "Test that UserService creates a user. A logger is required
# but irrelevant to the test."

class DummyLogger:
    """Fills the logger parameter without doing anything."""
    def info(self, msg): pass
    def error(self, msg): pass
    def debug(self, msg): pass


def test_user_creation_with_dummy_logger():
    dummy = DummyLogger()
    service = UserService(logger=dummy)
    user = service.create("alice", "alice@example.com")
    assert user.name == "alice"


# --- Scenario 2: Stub ---
# "Test that OrderService calculates tax correctly when the
# tax rate is 8%."

class StubTaxProvider:
    """Returns a fixed tax rate regardless of region."""
    def get_rate(self, region: str) -> float:
        return 0.08  # Always 8%


def test_order_tax_calculation_with_stub():
    stub = StubTaxProvider()
    order = OrderService(tax_provider=stub)
    total = order.calculate_total(subtotal=100.00, region="CA")
    assert total == 108.00  # 100 + 8% tax


# --- Scenario 3: Spy ---
# "Test that RegistrationService sends exactly one welcome email
# with the correct recipient."

class SpyEmailSender:
    """Records all emails sent for later inspection."""
    def __init__(self):
        self.sent_emails = []

    def send(self, to: str, subject: str, body: str):
        self.sent_emails.append({"to": to, "subject": subject, "body": body})


def test_registration_sends_welcome_email_with_spy():
    spy = SpyEmailSender()
    service = RegistrationService(email_sender=spy)
    service.register("bob@example.com", "password123")

    assert len(spy.sent_emails) == 1
    assert spy.sent_emails[0]["to"] == "bob@example.com"
    assert "welcome" in spy.sent_emails[0]["subject"].lower()


# --- Scenario 4: Mock ---
# "Test that PaymentService charges the gateway with the correct
# amount and currency."

def test_payment_charges_correctly_with_mock():
    mock_gateway = Mock()
    mock_gateway.charge.return_value = {"status": "success", "tx_id": "tx_123"}

    service = PaymentService(gateway=mock_gateway)
    result = service.process_payment(amount=49.99, card_token="tok_visa")

    mock_gateway.charge.assert_called_once_with(
        amount=49.99,
        token="tok_visa",
        currency="USD"
    )
    assert result["tx_id"] == "tx_123"


# --- Scenario 5: Fake ---
# "Test a complete user workflow (create, find, update, delete)
# without a real database."

@dataclass
class User:
    id: int = None
    name: str = ""
    email: str = ""


class FakeUserRepository:
    """In-memory implementation of the user repository."""
    def __init__(self):
        self._users: dict[int, User] = {}
        self._next_id = 1

    def save(self, user: User) -> User:
        if user.id is None:
            user.id = self._next_id
            self._next_id += 1
        self._users[user.id] = user
        return user

    def find_by_id(self, user_id: int) -> User | None:
        return self._users.get(user_id)

    def find_by_email(self, email: str) -> User | None:
        return next((u for u in self._users.values() if u.email == email), None)

    def delete(self, user_id: int) -> bool:
        return self._users.pop(user_id, None) is not None


def test_full_user_workflow_with_fake():
    repo = FakeUserRepository()

    # Create
    user = repo.save(User(name="alice", email="alice@example.com"))
    assert user.id == 1

    # Find
    found = repo.find_by_email("alice@example.com")
    assert found.name == "alice"

    # Update
    user.name = "alice_updated"
    repo.save(user)
    assert repo.find_by_id(1).name == "alice_updated"

    # Delete
    assert repo.delete(1) is True
    assert repo.find_by_id(1) is None
SOLUTION
}

# === Exercise 2: Builder Pattern ===
# Problem: Create a ProductBuilder and an OrderBuilder for an
# e-commerce domain. OrderBuilder should accept ProductBuilder
# instances. Write three tests using the builders.
exercise_2() {
    echo "=== Exercise 2: Builder Pattern ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Product:
    id: int
    name: str
    price: float
    category: str
    in_stock: bool


@dataclass
class OrderItem:
    product: Product
    quantity: int


@dataclass
class Order:
    id: int
    items: list[OrderItem]
    customer_email: str
    discount_pct: float
    created_at: datetime

    @property
    def subtotal(self) -> float:
        return sum(item.product.price * item.quantity for item in self.items)

    @property
    def total(self) -> float:
        return round(self.subtotal * (1 - self.discount_pct / 100), 2)


class ProductBuilder:
    """Builds Product objects with sensible defaults."""

    def __init__(self):
        self._id = 1
        self._name = "Default Product"
        self._price = 10.00
        self._category = "general"
        self._in_stock = True

    def with_id(self, id_val):
        self._id = id_val
        return self

    def with_name(self, name):
        self._name = name
        return self

    def with_price(self, price):
        self._price = price
        return self

    def with_category(self, category):
        self._category = category
        return self

    def out_of_stock(self):
        self._in_stock = False
        return self

    def build(self) -> Product:
        return Product(
            id=self._id,
            name=self._name,
            price=self._price,
            category=self._category,
            in_stock=self._in_stock,
        )


class OrderBuilder:
    """Builds Order objects, accepting ProductBuilders for items."""

    def __init__(self):
        self._id = 1
        self._items = []
        self._customer_email = "customer@example.com"
        self._discount_pct = 0.0
        self._created_at = datetime(2024, 1, 1)

    def with_item(self, product_builder: ProductBuilder, quantity: int = 1):
        product = product_builder.build()
        self._items.append(OrderItem(product=product, quantity=quantity))
        return self

    def with_discount(self, percent: float):
        self._discount_pct = percent
        return self

    def with_customer(self, email: str):
        self._customer_email = email
        return self

    def build(self) -> Order:
        return Order(
            id=self._id,
            items=self._items,
            customer_email=self._customer_email,
            discount_pct=self._discount_pct,
            created_at=self._created_at,
        )


# --- Test 1: Order total without discount ---
def test_order_total_no_discount():
    order = (
        OrderBuilder()
        .with_item(ProductBuilder().with_price(25.00), quantity=2)
        .with_item(ProductBuilder().with_price(10.00), quantity=1)
        .build()
    )
    assert order.subtotal == 60.00
    assert order.total == 60.00  # No discount


# --- Test 2: Order total with percentage discount ---
def test_order_total_with_discount():
    order = (
        OrderBuilder()
        .with_item(ProductBuilder().with_price(100.00), quantity=1)
        .with_discount(15)
        .build()
    )
    assert order.total == 85.00  # 100 - 15%


# --- Test 3: Order with multiple product categories ---
def test_order_contains_mixed_categories():
    order = (
        OrderBuilder()
        .with_item(
            ProductBuilder().with_name("Laptop").with_category("electronics").with_price(999.99),
            quantity=1
        )
        .with_item(
            ProductBuilder().with_name("Mouse Pad").with_category("accessories").with_price(12.99),
            quantity=2
        )
        .build()
    )
    categories = {item.product.category for item in order.items}
    assert categories == {"electronics", "accessories"}
    assert len(order.items) == 2
SOLUTION
}

# === Exercise 3: Page Object Refactoring ===
# Problem: Take API tests that make direct HTTP calls and refactor
# them to use the Page Object pattern. Show before and after.
exercise_3() {
    echo "=== Exercise 3: Page Object Refactoring ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
# ========================
# BEFORE: Direct HTTP calls scattered across tests
# ========================

def test_create_user_before(client):
    response = client.post("/api/v1/users", json={
        "name": "alice", "email": "alice@example.com"
    })
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "alice"

    # Fetch the user
    get_response = client.get(f"/api/v1/users/{data['id']}")
    assert get_response.status_code == 200
    assert get_response.json()["email"] == "alice@example.com"

def test_list_users_before(client):
    # Create two users
    client.post("/api/v1/users", json={"name": "alice", "email": "a@example.com"})
    client.post("/api/v1/users", json={"name": "bob", "email": "b@example.com"})

    response = client.get("/api/v1/users", params={"page": 1, "per_page": 10})
    assert response.status_code == 200
    assert len(response.json()["items"]) >= 2


# ========================
# AFTER: Page Object pattern
# ========================

class UserAPI:
    """Page Object encapsulating the User REST API."""

    def __init__(self, client, base_url="/api/v1/users"):
        self.client = client
        self.base_url = base_url

    def create(self, name: str, email: str) -> dict:
        response = self.client.post(self.base_url, json={
            "name": name, "email": email
        })
        assert response.status_code == 201, (
            f"Create user failed: {response.status_code} {response.text}"
        )
        return response.json()

    def get(self, user_id: int) -> dict | None:
        response = self.client.get(f"{self.base_url}/{user_id}")
        if response.status_code == 404:
            return None
        assert response.status_code == 200
        return response.json()

    def list_all(self, page: int = 1, per_page: int = 20) -> dict:
        response = self.client.get(
            self.base_url, params={"page": page, "per_page": per_page}
        )
        assert response.status_code == 200
        return response.json()

    def delete(self, user_id: int) -> bool:
        response = self.client.delete(f"{self.base_url}/{user_id}")
        return response.status_code == 204


import pytest

@pytest.fixture
def user_api(client):
    return UserAPI(client)


# Refactored tests — clean, focused on behavior
def test_create_and_retrieve_user(user_api):
    created = user_api.create("alice", "alice@example.com")
    retrieved = user_api.get(created["id"])
    assert retrieved["name"] == "alice"
    assert retrieved["email"] == "alice@example.com"


def test_list_includes_created_users(user_api):
    user_api.create("alice", "a@example.com")
    user_api.create("bob", "b@example.com")
    result = user_api.list_all(page=1, per_page=10)
    assert len(result["items"]) >= 2


def test_delete_user_removes_from_list(user_api):
    created = user_api.create("temp", "temp@example.com")
    assert user_api.delete(created["id"]) is True
    assert user_api.get(created["id"]) is None
SOLUTION
}

# === Exercise 4: AAA Discipline Refactoring ===
# Problem: Identify tests that violate the single-Act rule and
# refactor them into properly structured Arrange-Act-Assert tests.
exercise_4() {
    echo "=== Exercise 4: AAA Discipline Refactoring ==="
    echo ""
    echo "Solution:"
    cat << 'SOLUTION'
import pytest
from dataclasses import dataclass, field


@dataclass
class ShoppingCart:
    items: list = field(default_factory=list)
    discount_code: str = None

    def add_item(self, name: str, price: float, quantity: int = 1):
        self.items.append({"name": name, "price": price, "quantity": quantity})

    def remove_item(self, name: str):
        self.items = [i for i in self.items if i["name"] != name]

    def apply_discount(self, code: str):
        self.discount_code = code

    @property
    def subtotal(self) -> float:
        return sum(i["price"] * i["quantity"] for i in self.items)

    @property
    def total(self) -> float:
        discount = 0.10 if self.discount_code == "SAVE10" else 0.0
        return round(self.subtotal * (1 - discount), 2)

    @property
    def item_count(self) -> int:
        return sum(i["quantity"] for i in self.items)


# ========================
# BAD: Multiple Acts — tests too many things at once
# ========================

def test_shopping_cart_lifecycle_BAD():
    cart = ShoppingCart()

    cart.add_item("Widget", 10.00, 2)        # Act 1
    assert cart.subtotal == 20.00
    assert cart.item_count == 2

    cart.add_item("Gadget", 25.00, 1)         # Act 2
    assert cart.subtotal == 45.00

    cart.apply_discount("SAVE10")             # Act 3
    assert cart.total == 40.50

    cart.remove_item("Widget")                # Act 4
    assert cart.subtotal == 25.00
    assert cart.item_count == 1


# ========================
# GOOD: One Act per test, clear AAA structure
# ========================

@pytest.fixture
def cart():
    return ShoppingCart()


def test_add_item_updates_subtotal(cart):
    # Arrange — cart is empty (from fixture)

    # Act
    cart.add_item("Widget", 10.00, 2)

    # Assert
    assert cart.subtotal == 20.00
    assert cart.item_count == 2


def test_add_multiple_items_accumulates_subtotal(cart):
    # Arrange
    cart.add_item("Widget", 10.00, 2)

    # Act
    cart.add_item("Gadget", 25.00, 1)

    # Assert
    assert cart.subtotal == 45.00
    assert cart.item_count == 3


def test_discount_code_reduces_total(cart):
    # Arrange
    cart.add_item("Widget", 10.00, 2)
    cart.add_item("Gadget", 25.00, 1)

    # Act
    cart.apply_discount("SAVE10")

    # Assert
    assert cart.total == 40.50  # 45.00 - 10%


def test_remove_item_updates_subtotal(cart):
    # Arrange
    cart.add_item("Widget", 10.00, 2)
    cart.add_item("Gadget", 25.00, 1)

    # Act
    cart.remove_item("Widget")

    # Assert
    assert cart.subtotal == 25.00
    assert cart.item_count == 1


def test_invalid_discount_code_has_no_effect(cart):
    # Arrange
    cart.add_item("Widget", 10.00, 1)

    # Act
    cart.apply_discount("INVALID")

    # Assert
    assert cart.total == cart.subtotal  # No discount applied
SOLUTION
}

# Run all exercises
echo "Exercise solutions for Lesson 14: Test Architecture and Patterns"
echo "================================================================="
exercise_1
echo ""
exercise_2
echo ""
exercise_3
echo ""
exercise_4
