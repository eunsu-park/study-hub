"""
Example 11: Design Patterns Intro
Topic: Object-Oriented Programming

Demonstrates Singleton, Factory, Observer, and Strategy patterns.
"""

from abc import ABC, abstractmethod


# =============================================================================
# SINGLETON
# =============================================================================

class ConfigManager:
    """Singleton configuration manager."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {}
        return cls._instance

    def set(self, key, value):
        self._config[key] = value

    def get(self, key, default=None):
        return self._config.get(key, default)

    def __repr__(self):
        return f"ConfigManager({self._config})"


# =============================================================================
# FACTORY
# =============================================================================

class Notification(ABC):
    @abstractmethod
    def send(self, recipient, message) -> str:
        pass

class EmailNotification(Notification):
    def send(self, recipient, message):
        return f"Email to {recipient}: {message}"

class SMSNotification(Notification):
    def send(self, recipient, message):
        return f"SMS to {recipient}: {message}"

class PushNotification(Notification):
    def send(self, recipient, message):
        return f"Push to {recipient}: {message}"


class NotificationFactory:
    """Factory with registry for extensibility."""

    _registry = {
        "email": EmailNotification,
        "sms": SMSNotification,
        "push": PushNotification,
    }

    @classmethod
    def create(cls, channel: str) -> Notification:
        notif_class = cls._registry.get(channel)
        if not notif_class:
            raise ValueError(f"Unknown channel: {channel}")
        return notif_class()

    @classmethod
    def register(cls, name, notif_class):
        cls._registry[name] = notif_class


# =============================================================================
# OBSERVER
# =============================================================================

class EventBus:
    """Simple event bus (Observer pattern)."""

    def __init__(self):
        self._listeners = {}

    def on(self, event, callback):
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event, callback):
        if event in self._listeners:
            self._listeners[event].remove(callback)

    def emit(self, event, **data):
        for callback in self._listeners.get(event, []):
            callback(**data)


# =============================================================================
# STRATEGY
# =============================================================================

class PricingStrategy(ABC):
    @abstractmethod
    def calculate(self, base_price: float) -> float:
        pass

class RegularPricing(PricingStrategy):
    def calculate(self, base_price):
        return base_price

class MemberPricing(PricingStrategy):
    def calculate(self, base_price):
        return base_price * 0.9  # 10% off

class VIPPricing(PricingStrategy):
    def calculate(self, base_price):
        return base_price * 0.75  # 25% off

class Order:
    def __init__(self, items, pricing: PricingStrategy = None):
        self.items = items
        self.pricing = pricing or RegularPricing()

    def total(self):
        subtotal = sum(price for _, price in self.items)
        return self.pricing.calculate(subtotal)


if __name__ == "__main__":
    # Singleton
    print("=== Singleton ===")
    c1 = ConfigManager()
    c1.set("debug", True)
    c1.set("port", 8080)

    c2 = ConfigManager()
    print(f"Same instance? {c1 is c2}")
    print(f"Config: {c2}")

    # Factory
    print("\n=== Factory ===")
    for channel in ["email", "sms", "push"]:
        notif = NotificationFactory.create(channel)
        print(notif.send("alice", "Hello!"))

    # Observer
    print("\n=== Observer ===")
    bus = EventBus()

    def on_login(user, **_):
        print(f"  [LOG] {user} logged in")

    def on_login_email(user, email, **_):
        print(f"  [EMAIL] Welcome back, {user} ({email})")

    bus.on("login", on_login)
    bus.on("login", on_login_email)
    bus.emit("login", user="Alice", email="alice@mail.com")

    # Strategy
    print("\n=== Strategy ===")
    items = [("Laptop", 999), ("Mouse", 49), ("Keyboard", 79)]

    for strategy in [RegularPricing(), MemberPricing(), VIPPricing()]:
        order = Order(items, strategy)
        print(f"{strategy.__class__.__name__}: ${order.total():.2f}")
