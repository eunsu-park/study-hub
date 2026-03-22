"""
Exercises for Lesson 07: Design Patterns
Topic: Programming

Solutions to practice problems from the lesson.
"""
from abc import ABC, abstractmethod


# === Exercise 1: Identify Patterns ===
# Problem: Identify which design pattern(s) are used in each scenario.

def exercise_1():
    """Solution: Map scenarios to their design patterns."""

    patterns = {
        "1. Java's InputStream hierarchy (BufferedInputStream, GZIPInputStream)": {
            "pattern": "Decorator Pattern",
            "reasoning": "Each wrapper adds behavior (buffering, compression) to the base "
                         "stream without modifying it. They implement the same interface and "
                         "can be composed: new BufferedInputStream(new GZIPInputStream(fileStream))",
        },
        "2. Spring Framework's dependency injection": {
            "pattern": "Dependency Injection (a form of Inversion of Control)",
            "reasoning": "Objects don't create their dependencies; the framework injects them. "
                         "This decouples components and makes testing easier with mock objects.",
        },
        "3. GUI event listeners (button.onClick)": {
            "pattern": "Observer Pattern",
            "reasoning": "The button (subject) maintains a list of listeners (observers) and "
                         "notifies them when clicked. Listeners can be added/removed dynamically.",
        },
        "4. Python's @property decorator": {
            "pattern": "Decorator Pattern (language-level)",
            "reasoning": "The @property decorator wraps a method to make it behave like an "
                         "attribute access. It adds getter/setter behavior without changing "
                         "the underlying method.",
        },
        "5. SQL query builders": {
            "pattern": "Builder Pattern",
            "reasoning": "Query builders construct complex SQL queries step by step via "
                         "method chaining: query.select('*').from('users').where('age > 18'). "
                         "Separates construction from representation.",
        },
    }

    for scenario, info in patterns.items():
        print(f"  {scenario}")
        print(f"    Pattern: {info['pattern']}")
        print(f"    Why: {info['reasoning']}")
        print()


# === Exercise 2: Implement Observer ===
# Problem: Weather station that notifies multiple displays.

def exercise_2():
    """Solution: Observer pattern for a weather station system."""

    # Subject interface: maintains list of observers, notifies on change
    class WeatherStation:
        """Subject that tracks temperature and notifies display observers."""

        def __init__(self):
            self._observers = []
            self._temperature = 0.0

        def add_observer(self, observer):
            """Subscribe a display to temperature updates."""
            self._observers.append(observer)

        def remove_observer(self, observer):
            """Unsubscribe a display from updates."""
            self._observers.remove(observer)

        def set_temperature(self, temp):
            """Update temperature and notify all observers."""
            self._temperature = temp
            self._notify_observers()

        def _notify_observers(self):
            for observer in self._observers:
                observer.update(self._temperature)

    # Observer interface
    class Display(ABC):
        @abstractmethod
        def update(self, temperature):
            pass

    # Concrete observers: each displays data differently
    class CurrentConditionsDisplay(Display):
        """Shows the current temperature reading."""
        def update(self, temperature):
            print(f"    [Current] Temperature: {temperature:.1f}C")

    class StatisticsDisplay(Display):
        """Tracks min, max, and average temperature."""
        def __init__(self):
            self._readings = []

        def update(self, temperature):
            self._readings.append(temperature)
            avg = sum(self._readings) / len(self._readings)
            print(f"    [Stats] Avg: {avg:.1f}C, "
                  f"Min: {min(self._readings):.1f}C, "
                  f"Max: {max(self._readings):.1f}C")

    class ForecastDisplay(Display):
        """Simple forecast based on temperature trend."""
        def __init__(self):
            self._last_temp = None

        def update(self, temperature):
            if self._last_temp is None:
                forecast = "Not enough data"
            elif temperature > self._last_temp:
                forecast = "Warming trend - expect higher temps"
            elif temperature < self._last_temp:
                forecast = "Cooling trend - expect lower temps"
            else:
                forecast = "Steady conditions"
            self._last_temp = temperature
            print(f"    [Forecast] {forecast}")

    # Demonstration
    station = WeatherStation()

    current = CurrentConditionsDisplay()
    stats = StatisticsDisplay()
    forecast = ForecastDisplay()

    station.add_observer(current)
    station.add_observer(stats)
    station.add_observer(forecast)

    print("  Weather update: 25.0C")
    station.set_temperature(25.0)
    print("\n  Weather update: 28.5C")
    station.set_temperature(28.5)
    print("\n  Weather update: 22.0C")
    station.set_temperature(22.0)

    # Dynamic removal
    print("\n  Removing forecast display...")
    station.remove_observer(forecast)
    print("  Weather update: 20.0C")
    station.set_temperature(20.0)


# === Exercise 3: Refactor with Strategy ===
# Problem: Replace if/elif shipping calculation with Strategy pattern.

def exercise_3():
    """Solution: Strategy pattern for shipping cost calculation."""

    # Strategy interface: each shipping method defines its own pricing
    class ShippingStrategy(ABC):
        @abstractmethod
        def calculate(self, weight):
            pass

        @abstractmethod
        def name(self):
            pass

    class StandardShipping(ShippingStrategy):
        def calculate(self, weight):
            return weight * 0.5

        def name(self):
            return "Standard"

    class ExpressShipping(ShippingStrategy):
        def calculate(self, weight):
            return weight * 1.5

        def name(self):
            return "Express"

    class OvernightShipping(ShippingStrategy):
        def calculate(self, weight):
            return weight * 3.0

        def name(self):
            return "Overnight"

    # Context: Order uses the injected strategy
    class Order:
        """Order that delegates shipping cost to a strategy (OCP-compliant)."""

        def __init__(self, weight, shipping_strategy):
            self.weight = weight
            self._shipping = shipping_strategy

        def calculate_shipping(self):
            return self._shipping.calculate(self.weight)

        def set_shipping_strategy(self, strategy):
            """Can change strategy at runtime."""
            self._shipping = strategy

    # Demonstration
    weight = 10.0
    strategies = [StandardShipping(), ExpressShipping(), OvernightShipping()]

    print(f"  Package weight: {weight} kg")
    for strategy in strategies:
        order = Order(weight, strategy)
        cost = order.calculate_shipping()
        print(f"    {strategy.name()}: ${cost:.2f}")

    # Adding a new strategy requires NO changes to Order class (OCP)
    class FreeShipping(ShippingStrategy):
        def calculate(self, weight):
            return 0.0

        def name(self):
            return "Free (Promo)"

    order = Order(weight, FreeShipping())
    print(f"    {FreeShipping().name()}: ${order.calculate_shipping():.2f}")
    print("  (New strategy added without modifying existing code)")


# === Exercise 4: Build a Command System ===
# Problem: Smart home automation using the Command pattern with undo.

def exercise_4():
    """Solution: Command pattern for smart home with undo support."""

    # Command interface
    class Command(ABC):
        @abstractmethod
        def execute(self):
            pass

        @abstractmethod
        def undo(self):
            pass

    # Receivers (actual devices)
    class Light:
        def __init__(self, room):
            self.room = room
            self.is_on = False

        def turn_on(self):
            self.is_on = True
            return f"    {self.room} light ON"

        def turn_off(self):
            self.is_on = False
            return f"    {self.room} light OFF"

    class Thermostat:
        def __init__(self):
            self.temperature = 22

        def set_temp(self, temp):
            old = self.temperature
            self.temperature = temp
            return f"    Thermostat: {old}C -> {temp}C"

    class DoorLock:
        def __init__(self):
            self.is_locked = False

        def lock(self):
            self.is_locked = True
            return f"    Door LOCKED"

        def unlock(self):
            self.is_locked = False
            return f"    Door UNLOCKED"

    # Concrete commands
    class LightOnCommand(Command):
        def __init__(self, light):
            self._light = light

        def execute(self):
            return self._light.turn_on()

        def undo(self):
            return self._light.turn_off()

    class LightOffCommand(Command):
        def __init__(self, light):
            self._light = light

        def execute(self):
            return self._light.turn_off()

        def undo(self):
            return self._light.turn_on()

    class SetTemperatureCommand(Command):
        def __init__(self, thermostat, new_temp):
            self._thermostat = thermostat
            self._new_temp = new_temp
            self._old_temp = thermostat.temperature

        def execute(self):
            return self._thermostat.set_temp(self._new_temp)

        def undo(self):
            return self._thermostat.set_temp(self._old_temp)

    class LockDoorCommand(Command):
        def __init__(self, lock):
            self._lock = lock

        def execute(self):
            return self._lock.lock()

        def undo(self):
            return self._lock.unlock()

    # Macro command: groups multiple commands
    class MacroCommand(Command):
        """Execute multiple commands as a single unit (with batch undo)."""
        def __init__(self, name, commands):
            self._name = name
            self._commands = commands

        def execute(self):
            results = [f"    [Macro: {self._name}]"]
            for cmd in self._commands:
                results.append(cmd.execute())
            return "\n".join(results)

        def undo(self):
            results = [f"    [Undo Macro: {self._name}]"]
            # Undo in reverse order
            for cmd in reversed(self._commands):
                results.append(cmd.undo())
            return "\n".join(results)

    # Invoker with history for undo
    class RemoteControl:
        def __init__(self):
            self._history = []

        def execute(self, command):
            result = command.execute()
            self._history.append(command)
            print(result)

        def undo_last(self):
            if self._history:
                command = self._history.pop()
                result = command.undo()
                print(result)
            else:
                print("    Nothing to undo")

    # Demonstration
    living_light = Light("Living Room")
    bedroom_light = Light("Bedroom")
    thermostat = Thermostat()
    front_door = DoorLock()

    remote = RemoteControl()

    print("  Individual commands:")
    remote.execute(LightOnCommand(living_light))
    remote.execute(SetTemperatureCommand(thermostat, 18))

    print("\n  Undo last command:")
    remote.undo_last()

    print("\n  Macro: 'Leaving Home'")
    leaving_home = MacroCommand("Leaving Home", [
        LightOffCommand(living_light),
        LightOffCommand(bedroom_light),
        SetTemperatureCommand(thermostat, 16),
        LockDoorCommand(front_door),
    ])
    remote.execute(leaving_home)

    print("\n  Undo macro (came back home):")
    remote.undo_last()


# === Exercise 5: When NOT to Use Patterns ===
# Problem: Explain why patterns would be overkill in simple scenarios.

def exercise_5():
    """Solution: Identify when design patterns are unnecessary."""

    scenarios = {
        "1. Script that reads a file and prints contents": {
            "overkill_pattern": "Factory, Strategy, or any structural pattern",
            "reason": "A 5-line script doesn't need abstraction layers. "
                      "Adding patterns would triple the code for zero benefit. "
                      "Simple sequential code (open, read, print) is perfectly clear.",
            "better_approach": "Just use with open('file') as f: print(f.read())",
        },
        "2. Simple calculator with +/-/*/div": {
            "overkill_pattern": "Command pattern or Strategy for each operation",
            "reason": "Four fixed operations that won't change. A dictionary or "
                      "if/elif is simpler and more readable. Patterns make sense "
                      "only if operations frequently change or are user-extensible.",
            "better_approach": "ops = {'+': add, '-': sub, '*': mul, '/': div}",
        },
        "3. Todo list app with 3 screens": {
            "overkill_pattern": "Full MVC/MVVM with Observer, Mediator, Repository layers",
            "reason": "Three screens with simple CRUD don't need architectural patterns "
                      "designed for 50+ screen applications. Over-engineering slows "
                      "development without improving maintainability.",
            "better_approach": "Simple functions or a single model class with direct rendering",
        },
        "4. Config file with 5 settings": {
            "overkill_pattern": "Singleton, Builder, or Abstract Factory for config",
            "reason": "Five settings can be a dictionary or dataclass. A Singleton "
                      "for config adds complexity (thread safety, testing difficulty) "
                      "without benefit when there's only one consumer.",
            "better_approach": "config = {'key1': 'val1', ...} or a simple dataclass",
        },
    }

    for scenario, info in scenarios.items():
        print(f"  {scenario}")
        print(f"    Overkill: {info['overkill_pattern']}")
        print(f"    Why: {info['reason']}")
        print(f"    Better: {info['better_approach']}")
        print()

    print("  Rule of thumb: Use patterns when they solve a real problem")
    print("  you're currently facing, not preemptively 'just in case'.")


if __name__ == "__main__":
    print("=== Exercise 1: Identify Patterns ===")
    exercise_1()
    print("\n=== Exercise 2: Implement Observer ===")
    exercise_2()
    print("\n=== Exercise 3: Refactor with Strategy ===")
    exercise_3()
    print("\n=== Exercise 4: Build a Command System ===")
    exercise_4()
    print("\n=== Exercise 5: When NOT to Use Patterns ===")
    exercise_5()
    print("\nAll exercises completed!")
