"""
Example 14: OOP Best Practices
Topic: Object-Oriented Programming

Demonstrates anti-pattern identification, refactoring from procedural to OOP,
dependency injection for testability, and knowing when NOT to use OOP.
"""

from dataclasses import dataclass, field


# =============================================================================
# ANTI-PATTERN: God Class -> Refactored
# =============================================================================

# GOOD: Focused classes with single responsibility

class ProductCatalog:
    """Manages product data only."""

    def __init__(self):
        self._products = {}

    def add(self, product_id, name, price):
        self._products[product_id] = {"name": name, "price": price}

    def get(self, product_id):
        return self._products.get(product_id)

    def list_all(self):
        return list(self._products.values())


class PricingService:
    """Handles pricing logic only."""

    def __init__(self, tax_rate=0.08):
        self.tax_rate = tax_rate

    def calculate_total(self, items):
        subtotal = sum(item["price"] * item["qty"] for item in items)
        tax = subtotal * self.tax_rate
        return {"subtotal": subtotal, "tax": tax, "total": subtotal + tax}


class OrderProcessor:
    """Coordinates order processing via composition."""

    def __init__(self, catalog, pricing, notifier=None):
        self._catalog = catalog
        self._pricing = pricing
        self._notifier = notifier

    def process_order(self, order_items):
        # Resolve products
        resolved = []
        for item in order_items:
            product = self._catalog.get(item["id"])
            if not product:
                raise ValueError(f"Product {item['id']} not found")
            resolved.append({"name": product["name"], "price": product["price"],
                             "qty": item["qty"]})

        # Calculate pricing
        totals = self._pricing.calculate_total(resolved)

        # Notify (optional)
        if self._notifier:
            self._notifier.send(f"Order processed: ${totals['total']:.2f}")

        return {"items": resolved, **totals}


# =============================================================================
# DEPENDENCY INJECTION FOR TESTABILITY
# =============================================================================

class FakeNotifier:
    """Test double for notification service."""

    def __init__(self):
        self.messages = []

    def send(self, message):
        self.messages.append(message)


# =============================================================================
# REFACTORING: Procedural -> OOP
# =============================================================================

@dataclass
class Task:
    """Clean data class for a task."""
    title: str
    priority: int = 3  # 1=high, 5=low
    done: bool = False

    def complete(self):
        self.done = True

    def __repr__(self):
        status = "done" if self.done else "pending"
        return f"Task({self.title!r}, P{self.priority}, {status})"


class TaskManager:
    """Manages a collection of tasks."""

    def __init__(self):
        self._tasks = []

    def add(self, title, priority=3):
        task = Task(title, priority)
        self._tasks.append(task)
        return task

    def complete(self, title):
        task = self._find(title)
        if task:
            task.complete()
        return task

    def pending(self):
        return [t for t in self._tasks if not t.done]

    def by_priority(self):
        return sorted(self.pending(), key=lambda t: t.priority)

    def _find(self, title):
        for t in self._tasks:
            if t.title == title:
                return t
        return None

    def summary(self):
        total = len(self._tasks)
        done = sum(1 for t in self._tasks if t.done)
        return f"{done}/{total} tasks completed"


# =============================================================================
# WHEN NOT TO USE OOP
# =============================================================================

def celsius_to_fahrenheit(c):
    """Simple function — no need for a class."""
    return c * 9 / 5 + 32


def word_count(text):
    """Stateless transformation — function is fine."""
    words = text.lower().split()
    counts = {}
    for w in words:
        counts[w] = counts.get(w, 0) + 1
    return counts


if __name__ == "__main__":
    # SRP + Composition
    print("=== Clean Architecture (SRP + DI) ===")
    catalog = ProductCatalog()
    catalog.add("P1", "Laptop", 999)
    catalog.add("P2", "Mouse", 49)
    catalog.add("P3", "Keyboard", 79)

    pricing = PricingService(tax_rate=0.08)
    notifier = FakeNotifier()

    processor = OrderProcessor(catalog, pricing, notifier)
    result = processor.process_order([
        {"id": "P1", "qty": 1},
        {"id": "P2", "qty": 2},
    ])

    print(f"Items: {[i['name'] for i in result['items']]}")
    print(f"Subtotal: ${result['subtotal']:.2f}")
    print(f"Tax: ${result['tax']:.2f}")
    print(f"Total: ${result['total']:.2f}")
    print(f"Notifications: {notifier.messages}")

    # Task Manager
    print("\n=== Refactored Task Manager ===")
    tm = TaskManager()
    tm.add("Write tests", 1)
    tm.add("Update docs", 3)
    tm.add("Fix bug", 2)
    tm.add("Code review", 2)

    tm.complete("Fix bug")

    print("By priority:")
    for task in tm.by_priority():
        print(f"  {task}")
    print(tm.summary())

    # When NOT to use OOP
    print("\n=== Functions Are Fine Too ===")
    print(f"100C = {celsius_to_fahrenheit(100):.1f}F")

    text = "the quick brown fox jumps over the lazy dog the fox"
    counts = word_count(text)
    top3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    print(f"Top 3 words: {top3}")
