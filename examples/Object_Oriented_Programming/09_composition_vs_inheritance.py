"""
Example 09: Composition vs Inheritance
Topic: Object-Oriented Programming

Demonstrates composition with delegation, strategy pattern,
and refactoring from inheritance to composition.
"""

from abc import ABC, abstractmethod


# =============================================================================
# COMPOSITION: CAR WITH COMPONENTS
# =============================================================================

class Engine:
    """Engine component."""

    def __init__(self, horsepower, fuel_type="gasoline"):
        self.horsepower = horsepower
        self.fuel_type = fuel_type
        self.running = False

    def start(self):
        self.running = True
        return f"Engine ({self.horsepower}hp {self.fuel_type}) started"

    def stop(self):
        self.running = False
        return "Engine stopped"


class GPS:
    """GPS navigation component."""

    def __init__(self):
        self.destination = None

    def navigate(self, destination):
        self.destination = destination
        return f"Navigating to {destination}"

    def clear(self):
        self.destination = None


class Car:
    """Car composed of components (not inheriting from them)."""

    def __init__(self, make, model, horsepower):
        self.make = make
        self.model = model
        self.engine = Engine(horsepower)
        self.gps = GPS()
        self.mileage = 0

    def start(self):
        return self.engine.start()

    def drive(self, destination, distance):
        if not self.engine.running:
            self.start()
        self.mileage += distance
        nav = self.gps.navigate(destination)
        return f"Driving to {destination} ({distance} mi). {nav}"

    def __repr__(self):
        return f"Car({self.make} {self.model}, {self.mileage} mi)"


# =============================================================================
# STRATEGY PATTERN
# =============================================================================

class DiscountStrategy(ABC):
    """Abstract discount strategy."""

    @abstractmethod
    def calculate(self, price: float) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class NoDiscount(DiscountStrategy):
    @property
    def name(self):
        return "No Discount"

    def calculate(self, price):
        return price


class PercentageDiscount(DiscountStrategy):
    def __init__(self, percent):
        self.percent = percent

    @property
    def name(self):
        return f"{self.percent}% Off"

    def calculate(self, price):
        return price * (1 - self.percent / 100)


class BuyOneGetOneFree(DiscountStrategy):
    @property
    def name(self):
        return "Buy 1 Get 1 Free"

    def calculate(self, price):
        return price / 2


class ShoppingCart:
    """Cart with pluggable discount strategy."""

    def __init__(self, discount: DiscountStrategy = None):
        self.items = []
        self._discount = discount or NoDiscount()

    @property
    def discount(self):
        return self._discount

    @discount.setter
    def discount(self, strategy):
        self._discount = strategy

    def add_item(self, name, price):
        self.items.append({"name": name, "price": price})

    def total(self):
        subtotal = sum(item["price"] for item in self.items)
        return self._discount.calculate(subtotal)

    def checkout(self):
        subtotal = sum(item["price"] for item in self.items)
        final = self.total()
        savings = subtotal - final
        lines = [f"Cart ({self._discount.name}):"]
        for item in self.items:
            lines.append(f"  {item['name']}: ${item['price']:.2f}")
        lines.append(f"  Subtotal: ${subtotal:.2f}")
        if savings > 0:
            lines.append(f"  Savings: -${savings:.2f}")
        lines.append(f"  Total: ${final:.2f}")
        return "\n".join(lines)


# =============================================================================
# NOTIFICATION SYSTEM (COMPOSITION)
# =============================================================================

class EmailSender:
    def send(self, to, message):
        return f"Email -> {to}: {message}"


class SMSSender:
    def send(self, to, message):
        return f"SMS -> {to}: {message}"


class SlackSender:
    def send(self, to, message):
        return f"Slack -> #{to}: {message}"


class NotificationService:
    """Notification service composed of multiple senders."""

    def __init__(self):
        self._channels = {}

    def add_channel(self, name, sender):
        self._channels[name] = sender
        return self

    def notify(self, channel, recipient, message):
        sender = self._channels.get(channel)
        if not sender:
            raise ValueError(f"Unknown channel: {channel}")
        return sender.send(recipient, message)

    def broadcast(self, recipient, message):
        return [s.send(recipient, message) for s in self._channels.values()]


if __name__ == "__main__":
    # Composition: Car
    print("=== Composition: Car ===")
    car = Car("Toyota", "Camry", 203)
    print(car.start())
    print(car.drive("New York", 250))
    print(car)

    # Strategy pattern
    print("\n=== Strategy Pattern: Shopping Cart ===")
    cart = ShoppingCart()
    cart.add_item("Laptop", 999)
    cart.add_item("Mouse", 49)
    cart.add_item("Keyboard", 79)

    print(cart.checkout())

    print()
    cart.discount = PercentageDiscount(20)
    print(cart.checkout())

    print()
    cart.discount = BuyOneGetOneFree()
    print(cart.checkout())

    # Notification system
    print("\n=== Notification Service (Composition) ===")
    notifier = NotificationService()
    notifier.add_channel("email", EmailSender())
    notifier.add_channel("sms", SMSSender())
    notifier.add_channel("slack", SlackSender())

    print(notifier.notify("email", "alice@mail.com", "Hello!"))
    print("\nBroadcast:")
    for msg in notifier.broadcast("alice", "Server is down!"):
        print(f"  {msg}")
