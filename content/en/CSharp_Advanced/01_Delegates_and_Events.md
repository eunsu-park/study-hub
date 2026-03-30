# Delegates and Events

**Previous**: [Overview](./00_Overview.md) | **Next**: [Lambda Expressions and Closures](./02_Lambda_and_Closures.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and instantiate delegates to reference methods
2. Combine and remove delegates using multicast delegate chaining
3. Use the built-in delegate types `Action`, `Func`, and `Predicate`
4. Declare events with `EventHandler<T>` and raise them safely
5. Design custom `EventArgs` subclasses for domain-specific events
6. Implement the publisher-subscriber pattern in C#
7. Invoke events in a thread-safe manner using `?.Invoke()`

---

Delegates are the foundation of callback-based programming in C#. They allow you to treat methods as first-class values — storing them in variables, passing them as arguments, and composing them into chains. Events build on top of delegates to provide a safe, encapsulated notification mechanism that is central to UI frameworks, message buses, and reactive architectures throughout the .NET ecosystem.

## 1. Delegate Fundamentals

A delegate is a type that represents a reference to a method with a specific parameter list and return type. Think of it as a type-safe function pointer.

### 1.1 Declaring a Delegate Type

A delegate declaration defines a new type. Any method whose signature matches the delegate can be assigned to an instance of that delegate.

```csharp
// Declare a delegate type that takes two ints and returns an int
public delegate int MathOperation(int a, int b);

// Declare a delegate type that takes a string and returns void
public delegate void Logger(string message);

// Declare a delegate type with no parameters and no return
public delegate void SimpleCallback();
```

### 1.2 Instantiating a Delegate

You can create a delegate instance by passing a matching method name to the constructor or by assigning a method group directly.

```csharp
public class Calculator
{
    public static int Add(int a, int b) => a + b;
    public static int Multiply(int a, int b) => a * b;
    public int Subtract(int a, int b) => a - b; // instance method
}

// Constructor syntax
MathOperation op1 = new MathOperation(Calculator.Add);
Console.WriteLine(op1(3, 4)); // 7

// Method group conversion (preferred — shorter)
MathOperation op2 = Calculator.Multiply;
Console.WriteLine(op2(3, 4)); // 12

// Instance method delegate
var calc = new Calculator();
MathOperation op3 = calc.Subtract;
Console.WriteLine(op3(10, 4)); // 6
```

### 1.3 Invoking a Delegate

You can invoke a delegate just like a regular method call. Alternatively, use the `Invoke` method explicitly.

```csharp
MathOperation op = Calculator.Add;

// Direct invocation
int result1 = op(5, 3); // 8

// Explicit Invoke
int result2 = op.Invoke(5, 3); // 8
```

### 1.4 Delegate Variance

Delegates support covariance on return types and contravariance on parameter types. This means a delegate can reference a method whose return type is more derived or whose parameter types are less derived.

```csharp
public delegate Animal AnimalFactory();
public delegate void ProcessDog(Dog dog);

public class Animal { }
public class Dog : Animal { }

public static Dog CreateDog() => new Dog();
public static void ProcessAnimal(Animal animal) { }

// Covariance: Dog is more derived than Animal
AnimalFactory factory = CreateDog;  // OK — return type covariance

// Contravariance: Animal is less derived than Dog
ProcessDog processor = ProcessAnimal; // OK — parameter contravariance
```

## 2. Multicast Delegates

All delegates in C# are multicast — they can hold references to more than one method. When a multicast delegate is invoked, all methods in its invocation list are called in order.

### 2.1 Combining Delegates with += and -=

```csharp
public delegate void Notifier(string message);

public static void EmailNotify(string msg) =>
    Console.WriteLine($"[EMAIL] {msg}");

public static void SmsNotify(string msg) =>
    Console.WriteLine($"[SMS] {msg}");

public static void SlackNotify(string msg) =>
    Console.WriteLine($"[SLACK] {msg}");

// Combine delegates
Notifier notify = EmailNotify;
notify += SmsNotify;
notify += SlackNotify;

notify("Server is down!");
// Output:
// [EMAIL] Server is down!
// [SMS] Server is down!
// [SLACK] Server is down!

// Remove a delegate
notify -= SmsNotify;
notify("Server recovered.");
// Output:
// [EMAIL] Server recovered.
// [SLACK] Server recovered.
```

### 2.2 Return Values with Multicast Delegates

When a multicast delegate has a non-void return type, only the return value of the **last** method in the invocation list is returned. If you need all return values, iterate the invocation list manually.

```csharp
public delegate int Scorer(string input);

public static int LengthScore(string s) => s.Length;
public static int VowelScore(string s) => s.Count(c => "aeiouAEIOU".Contains(c));

Scorer scorer = LengthScore;
scorer += VowelScore;

// Only VowelScore's return value is captured
int result = scorer("Hello"); // 2 (vowels: e, o)

// To get all results, iterate the invocation list
foreach (Scorer s in scorer.GetInvocationList().Cast<Scorer>())
{
    Console.WriteLine(s("Hello")); // 5, then 2
}
```

### 2.3 Delegate.Combine and Delegate.Remove

The `+=` and `-=` operators are syntactic sugar for `Delegate.Combine` and `Delegate.Remove`.

```csharp
Notifier a = EmailNotify;
Notifier b = SmsNotify;

// These are equivalent:
Notifier combined1 = a + b;
Notifier combined2 = (Notifier)Delegate.Combine(a, b);

// Removal:
Notifier reduced1 = combined1 - b;
Notifier reduced2 = (Notifier)Delegate.Remove(combined2, b);
```

## 3. Built-in Delegate Types

The .NET Base Class Library provides generic delegate types that cover the vast majority of use cases, so you rarely need to declare custom delegate types.

### 3.1 Action\<T\> — Void Return

`Action` wraps a method that returns `void`. It comes in variants from zero to sixteen type parameters.

```csharp
// No parameters
Action greet = () => Console.WriteLine("Hello!");
greet();

// One parameter
Action<string> log = message => Console.WriteLine($"[LOG] {message}");
log("Application started");

// Two parameters
Action<string, int> repeat = (text, count) =>
{
    for (int i = 0; i < count; i++)
        Console.Write(text);
    Console.WriteLine();
};
repeat("Ha", 3); // HaHaHa

// As a method parameter — strategy pattern
public static void ProcessItems<T>(IEnumerable<T> items, Action<T> processor)
{
    foreach (var item in items)
        processor(item);
}

ProcessItems(new[] { 1, 2, 3 }, n => Console.WriteLine(n * 10));
// 10, 20, 30
```

### 3.2 Func\<T, TResult\> — With Return Value

`Func` wraps a method that returns a value. The last type parameter is always the return type.

```csharp
// No input, returns string
Func<string> getName = () => "Alice";
Console.WriteLine(getName()); // Alice

// int -> bool
Func<int, bool> isEven = n => n % 2 == 0;
Console.WriteLine(isEven(4)); // True

// (int, int) -> int
Func<int, int, int> add = (a, b) => a + b;
Console.WriteLine(add(3, 7)); // 10

// Func as a factory / lazy initializer
public static T CreateOrDefault<T>(Func<T> factory, bool shouldCreate)
{
    return shouldCreate ? factory() : default!;
}

var result = CreateOrDefault(() => new List<int> { 1, 2, 3 }, true);
Console.WriteLine(result.Count); // 3
```

### 3.3 Predicate\<T\> — Boolean Test

`Predicate<T>` is equivalent to `Func<T, bool>` but is used specifically in older collection APIs like `List<T>.Find`, `List<T>.RemoveAll`, etc.

```csharp
Predicate<int> isPositive = n => n > 0;

var numbers = new List<int> { -3, -1, 0, 2, 5, -7, 8 };

// List<T>.FindAll uses Predicate<T>
List<int> positives = numbers.FindAll(isPositive);
Console.WriteLine(string.Join(", ", positives)); // 2, 5, 8

// List<T>.RemoveAll uses Predicate<T>
int removed = numbers.RemoveAll(n => n < 0);
Console.WriteLine(removed); // 3
Console.WriteLine(string.Join(", ", numbers)); // 0, 2, 5, 8
```

## 4. Anonymous Methods

Before lambda expressions (C# 3.0), C# 2.0 introduced anonymous methods using the `delegate` keyword. They are largely superseded by lambdas but still appear in legacy code.

### 4.1 Anonymous Method Syntax

```csharp
// Anonymous method with delegate keyword
Func<int, int, int> multiply = delegate (int a, int b)
{
    return a * b;
};
Console.WriteLine(multiply(4, 5)); // 20

// Anonymous method that ignores parameters
// (delegate without parameter list discards all arguments)
EventHandler handler = delegate
{
    Console.WriteLine("Something happened, but I don't care about the details.");
};
handler(null, EventArgs.Empty);
```

### 4.2 When to Prefer Anonymous Methods

In modern C#, there is almost no reason to use `delegate` anonymous methods over lambdas. The only edge case is when you genuinely want to ignore all parameters:

```csharp
// Lambda requires you to specify discards or parameters
button.Click += (_, _) => HandleClick();

// Anonymous method can skip them entirely
button.Click += delegate { HandleClick(); };
```

## 5. Events and EventHandler

Events provide an encapsulation layer on top of delegates. While a delegate field can be invoked or reassigned by any code that has access, an event restricts external code to only `+=` and `-=` operations.

### 5.1 Declaring and Raising Events

```csharp
public class TemperatureSensor
{
    // Event declaration using EventHandler<T>
    public event EventHandler<TemperatureChangedEventArgs>? TemperatureChanged;

    private double _temperature;

    public double Temperature
    {
        get => _temperature;
        set
        {
            double oldTemp = _temperature;
            _temperature = value;
            OnTemperatureChanged(new TemperatureChangedEventArgs(oldTemp, value));
        }
    }

    // Protected virtual method for raising the event (standard pattern)
    protected virtual void OnTemperatureChanged(TemperatureChangedEventArgs e)
    {
        TemperatureChanged?.Invoke(this, e);
    }
}
```

### 5.2 Custom EventArgs

```csharp
public class TemperatureChangedEventArgs : EventArgs
{
    public double OldTemperature { get; }
    public double NewTemperature { get; }
    public double Delta => NewTemperature - OldTemperature;

    public TemperatureChangedEventArgs(double oldTemp, double newTemp)
    {
        OldTemperature = oldTemp;
        NewTemperature = newTemp;
    }
}
```

### 5.3 Subscribing to Events

```csharp
var sensor = new TemperatureSensor();

// Subscribe with a method group
sensor.TemperatureChanged += OnTemperatureChanged;

// Subscribe with a lambda
sensor.TemperatureChanged += (sender, e) =>
{
    if (e.Delta > 5)
        Console.WriteLine($"ALERT: Temperature spike of {e.Delta:F1}°C!");
};

sensor.Temperature = 20.0;
sensor.Temperature = 28.5; // triggers alert (delta = 8.5)

static void OnTemperatureChanged(object? sender, TemperatureChangedEventArgs e)
{
    Console.WriteLine($"Temperature: {e.OldTemperature:F1} -> {e.NewTemperature:F1}");
}
```

## 6. Event Declaration Patterns

### 6.1 Simple Event with EventHandler

For events that carry no extra data, use the non-generic `EventHandler`.

```csharp
public class Button
{
    public event EventHandler? Clicked;

    public void SimulateClick()
    {
        Console.WriteLine("Button clicked.");
        Clicked?.Invoke(this, EventArgs.Empty);
    }
}

var btn = new Button();
btn.Clicked += (sender, e) => Console.WriteLine("Handler 1 executed");
btn.Clicked += (sender, e) => Console.WriteLine("Handler 2 executed");
btn.SimulateClick();
// Button clicked.
// Handler 1 executed
// Handler 2 executed
```

### 6.2 Custom Event Accessors (add/remove)

You can provide explicit `add` and `remove` accessors for fine-grained control over event subscription, such as logging or thread synchronization.

```csharp
public class SecurePublisher
{
    private EventHandler? _completed;
    private readonly object _lock = new();

    public event EventHandler Completed
    {
        add
        {
            lock (_lock)
            {
                Console.WriteLine($"Subscriber added: {value.Method.Name}");
                _completed += value;
            }
        }
        remove
        {
            lock (_lock)
            {
                Console.WriteLine($"Subscriber removed: {value.Method.Name}");
                _completed -= value;
            }
        }
    }

    public void RaiseCompleted()
    {
        EventHandler? handler;
        lock (_lock)
        {
            handler = _completed;
        }
        handler?.Invoke(this, EventArgs.Empty);
    }
}
```

### 6.3 Events vs Delegates — Access Control

The key difference between events and delegate fields is encapsulation:

```csharp
public class WithDelegateField
{
    public Action<string>? OnMessage; // Public delegate field
}

public class WithEvent
{
    public event Action<string>? OnMessage; // Event

    public void Send(string msg) => OnMessage?.Invoke(msg);
}

var df = new WithDelegateField();
df.OnMessage = msg => Console.WriteLine(msg); // OK — full assignment
df.OnMessage("test");        // OK — external invocation
df.OnMessage = null;         // OK — external can wipe all subscribers

var ev = new WithEvent();
ev.OnMessage += msg => Console.WriteLine(msg); // OK — subscribe
// ev.OnMessage("test");     // ERROR — cannot invoke from outside
// ev.OnMessage = null;      // ERROR — cannot assign from outside
ev.Send("test");             // OK — must go through the class's own method
```

## 7. Thread-Safe Event Invocation

### 7.1 The Null-Conditional Pattern

The classic pitfall with events is a race condition: between the null check and the invocation, another thread could unsubscribe the last handler. The null-conditional operator `?.` solves this by capturing the delegate reference atomically.

```csharp
// WRONG — race condition possible
if (TemperatureChanged != null)       // Another thread does -= here
    TemperatureChanged(this, args);   // NullReferenceException!

// CORRECT — null-conditional operator
TemperatureChanged?.Invoke(this, args);

// Also correct — explicit local copy
var handler = TemperatureChanged;
handler?.Invoke(this, args);
```

### 7.2 Volatile Delegate Reads

For classes that see heavy concurrent subscription and invocation, you can use `Volatile.Read` to ensure the delegate reference is fresh:

```csharp
protected virtual void OnDataReceived(DataReceivedEventArgs e)
{
    var handler = Volatile.Read(ref _dataReceived);
    handler?.Invoke(this, e);
}
```

## 8. Publisher-Subscriber Pattern

The publisher-subscriber (pub-sub) pattern decouples event producers from consumers. C# events are a natural implementation of this pattern.

### 8.1 A Complete Pub-Sub Example: Stock Ticker

```csharp
// Event arguments
public class StockPriceChangedEventArgs : EventArgs
{
    public string Symbol { get; }
    public decimal OldPrice { get; }
    public decimal NewPrice { get; }
    public decimal ChangePercent => OldPrice == 0 ? 0 :
        Math.Round((NewPrice - OldPrice) / OldPrice * 100, 2);

    public StockPriceChangedEventArgs(string symbol, decimal oldPrice, decimal newPrice)
    {
        Symbol = symbol;
        OldPrice = oldPrice;
        NewPrice = newPrice;
    }
}

// Publisher
public class StockTicker
{
    private readonly Dictionary<string, decimal> _prices = new();
    public event EventHandler<StockPriceChangedEventArgs>? PriceChanged;

    public void UpdatePrice(string symbol, decimal newPrice)
    {
        _prices.TryGetValue(symbol, out decimal oldPrice);
        _prices[symbol] = newPrice;

        if (oldPrice != newPrice)
        {
            PriceChanged?.Invoke(this,
                new StockPriceChangedEventArgs(symbol, oldPrice, newPrice));
        }
    }
}

// Subscriber 1: Console logger
public class PriceLogger
{
    public void Subscribe(StockTicker ticker)
    {
        ticker.PriceChanged += OnPriceChanged;
    }

    private void OnPriceChanged(object? sender, StockPriceChangedEventArgs e)
    {
        Console.WriteLine(
            $"[LOG] {e.Symbol}: ${e.OldPrice} -> ${e.NewPrice} ({e.ChangePercent:+0.00;-0.00}%)");
    }
}

// Subscriber 2: Alert system
public class PriceAlert
{
    private readonly decimal _thresholdPercent;

    public PriceAlert(decimal thresholdPercent) => _thresholdPercent = thresholdPercent;

    public void Subscribe(StockTicker ticker)
    {
        ticker.PriceChanged += (_, e) =>
        {
            if (Math.Abs(e.ChangePercent) >= _thresholdPercent)
            {
                Console.WriteLine(
                    $"*** ALERT: {e.Symbol} moved {e.ChangePercent:+0.00;-0.00}% ***");
            }
        };
    }
}

// Usage
var ticker = new StockTicker();
var logger = new PriceLogger();
var alert = new PriceAlert(thresholdPercent: 3.0m);

logger.Subscribe(ticker);
alert.Subscribe(ticker);

ticker.UpdatePrice("MSFT", 350.00m);
ticker.UpdatePrice("MSFT", 365.00m); // +4.29% — triggers alert
ticker.UpdatePrice("AAPL", 180.00m);
ticker.UpdatePrice("AAPL", 178.50m); // -0.83% — no alert
```

## 9. Practical Example: Event-Driven Notification System

Let's build a more complete event-driven notification system that demonstrates multiple event types, unsubscription, and weak event-like patterns.

### 9.1 The Domain Model

```csharp
public enum OrderStatus { Created, Processing, Shipped, Delivered, Cancelled }

public class OrderStatusChangedEventArgs : EventArgs
{
    public int OrderId { get; }
    public OrderStatus OldStatus { get; }
    public OrderStatus NewStatus { get; }
    public DateTime Timestamp { get; }

    public OrderStatusChangedEventArgs(int orderId, OrderStatus oldStatus, OrderStatus newStatus)
    {
        OrderId = orderId;
        OldStatus = oldStatus;
        NewStatus = newStatus;
        Timestamp = DateTime.UtcNow;
    }
}
```

### 9.2 The Publisher (Order)

```csharp
public class Order
{
    public int Id { get; }
    public string CustomerEmail { get; }

    private OrderStatus _status = OrderStatus.Created;
    public OrderStatus Status
    {
        get => _status;
        private set
        {
            if (_status != value)
            {
                var old = _status;
                _status = value;
                OnStatusChanged(new OrderStatusChangedEventArgs(Id, old, value));
            }
        }
    }

    public event EventHandler<OrderStatusChangedEventArgs>? StatusChanged;

    public Order(int id, string customerEmail)
    {
        Id = id;
        CustomerEmail = customerEmail;
    }

    public void Process() => Status = OrderStatus.Processing;
    public void Ship() => Status = OrderStatus.Shipped;
    public void Deliver() => Status = OrderStatus.Delivered;
    public void Cancel() => Status = OrderStatus.Cancelled;

    protected virtual void OnStatusChanged(OrderStatusChangedEventArgs e)
    {
        StatusChanged?.Invoke(this, e);
    }
}
```

### 9.3 Subscribers

```csharp
public class EmailNotificationService
{
    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;
    public void Unsubscribe(Order order) => order.StatusChanged -= HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        if (sender is Order order)
        {
            Console.WriteLine(
                $"[EMAIL -> {order.CustomerEmail}] " +
                $"Order #{e.OrderId}: {e.OldStatus} -> {e.NewStatus}");
        }
    }
}

public class InventoryService
{
    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        switch (e.NewStatus)
        {
            case OrderStatus.Cancelled:
                Console.WriteLine($"[INVENTORY] Restocking items for order #{e.OrderId}");
                break;
            case OrderStatus.Shipped:
                Console.WriteLine($"[INVENTORY] Items shipped for order #{e.OrderId}");
                break;
        }
    }
}

public class AuditLog
{
    private readonly List<string> _entries = new();

    public void Subscribe(Order order) => order.StatusChanged += HandleStatusChange;

    private void HandleStatusChange(object? sender, OrderStatusChangedEventArgs e)
    {
        string entry = $"[{e.Timestamp:u}] Order #{e.OrderId}: {e.OldStatus} -> {e.NewStatus}";
        _entries.Add(entry);
        Console.WriteLine($"[AUDIT] {entry}");
    }

    public IReadOnlyList<string> GetEntries() => _entries.AsReadOnly();
}
```

### 9.4 Running the System

```csharp
// Create order and services
var order = new Order(1001, "alice@example.com");
var emailService = new EmailNotificationService();
var inventoryService = new InventoryService();
var auditLog = new AuditLog();

// Subscribe all services
emailService.Subscribe(order);
inventoryService.Subscribe(order);
auditLog.Subscribe(order);

// Process order lifecycle
order.Process();   // Created -> Processing
order.Ship();      // Processing -> Shipped
order.Deliver();   // Shipped -> Delivered

Console.WriteLine($"\nAudit log has {auditLog.GetEntries().Count} entries.");

// Create another order and cancel it
var order2 = new Order(1002, "bob@example.com");
emailService.Subscribe(order2);
inventoryService.Subscribe(order2);
auditLog.Subscribe(order2);

order2.Process();
order2.Cancel(); // triggers inventory restock
```

## 10. Practice Problems

1. **Custom Delegate Chain**: Declare a delegate `StringTransform` that takes a `string` and returns a `string`. Create a chain of three transforms (trim whitespace, convert to lowercase, replace spaces with hyphens). Apply the chain to `"  Hello Beautiful World  "` and print each intermediate result by iterating the invocation list.

2. **Generic Event Aggregator**: Implement an `EventAggregator` class with methods `Subscribe<TEvent>(Action<TEvent> handler)` and `Publish<TEvent>(TEvent eventData)`. Subscribers should receive events only of the type they subscribed to. Test with at least two different event types.

3. **Unsubscription and Memory Leaks**: Create a `Timer` class that raises a `Tick` event every second (use `Task.Delay` in a loop). Subscribe a handler, observe ticks, then unsubscribe and demonstrate that ticks stop being processed. Discuss what would happen if you never unsubscribed.

4. **Cancelable Events**: Design an event system where subscribers can cancel an operation. Create a `FileDownloader` class with a `Downloading` event that uses a custom `DownloadingEventArgs` (which includes a `Cancel` property). Before downloading, raise the event. If any subscriber sets `Cancel = true`, abort the download.

5. **Delegate Performance Comparison**: Write a benchmark that compares the invocation cost of: (a) a direct method call, (b) a `Func<int, int>` delegate, and (c) an interface method call (`ITransform.Apply`). Run each 10 million times and print the elapsed time. What do you observe?
