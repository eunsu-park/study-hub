// Lesson 01: Delegates and Events
// Run: dotnet run

using System;
using System.Collections.Generic;

// ============================================================
// 1. Basic Delegate Declaration and Usage
// ============================================================

// A delegate is a type-safe function pointer
delegate int MathOperation(int a, int b);

MathOperation add = (a, b) => a + b;
MathOperation multiply = (a, b) => a * b;

Console.WriteLine("=== Basic Delegates ===");
Console.WriteLine($"Add(3, 4) = {add(3, 4)}");
Console.WriteLine($"Multiply(3, 4) = {multiply(3, 4)}");

// Delegates can be reassigned
MathOperation current = add;
Console.WriteLine($"Current(5, 6) = {current(5, 6)}");
current = multiply;
Console.WriteLine($"Current(5, 6) = {current(5, 6)}");

// ============================================================
// 2. Action and Func — Built-in Generic Delegates
// ============================================================

Console.WriteLine("\n=== Action and Func ===");

// Action<T> — delegate that returns void
Action<string> greet = name => Console.WriteLine($"Hello, {name}!");
greet("Alice");

Action<string, int> repeatGreet = (name, times) =>
{
    for (int i = 0; i < times; i++)
        Console.WriteLine($"  Hi, {name}! ({i + 1})");
};
repeatGreet("Bob", 3);

// Func<T, TResult> — delegate that returns a value
Func<int, int, int> subtract = (a, b) => a - b;
Func<double, double> square = x => x * x;
Func<string, int> wordCount = s => s.Split(' ').Length;

Console.WriteLine($"Subtract(10, 3) = {subtract(10, 3)}");
Console.WriteLine($"Square(4.5) = {square(4.5)}");
Console.WriteLine($"WordCount(\"hello world foo\") = {wordCount("hello world foo")}");

// Predicate<T> — a Func<T, bool> shortcut
Predicate<int> isEven = n => n % 2 == 0;
Console.WriteLine($"IsEven(7) = {isEven(7)}");

// ============================================================
// 3. Multicast Delegates
// ============================================================

Console.WriteLine("\n=== Multicast Delegates ===");

// Multiple methods can be attached to a single delegate
Action<string> logger = msg => Console.WriteLine($"  [Console] {msg}");
logger += msg => Console.WriteLine($"  [File]    {msg}");
logger += msg => Console.WriteLine($"  [Network] {msg}");

// Invoking calls all subscribed methods in order
logger("Application started");

// Remove a handler (must reference the same method instance)
Action<string> networkLog = msg => Console.WriteLine($"  [Network] {msg}");
Action<string> pipeline = msg => Console.WriteLine($"  [A] {msg}");
pipeline += msg => Console.WriteLine($"  [B] {msg}");
pipeline += networkLog;

Console.WriteLine("Before removal:");
pipeline("test");

pipeline -= networkLog;
Console.WriteLine("After removal:");
pipeline("test");

// ============================================================
// 4. Events — Encapsulated Delegate Invocation
// ============================================================

Console.WriteLine("\n=== Events ===");

// Events restrict external code to += and -= only (no direct invocation)
var thermostat = new Thermostat();

// Subscribe to the event
thermostat.TemperatureChanged += (sender, e) =>
    Console.WriteLine($"  Alert: Temperature is now {e.Temperature}°C (from {sender?.GetType().Name})");

thermostat.TemperatureChanged += (sender, e) =>
{
    if (e.Temperature > 30)
        Console.WriteLine("  WARNING: Temperature exceeds 30°C!");
};

thermostat.CurrentTemperature = 22;
thermostat.CurrentTemperature = 35;
thermostat.CurrentTemperature = 18;

// ============================================================
// 5. Custom EventArgs
// ============================================================

Console.WriteLine("\n=== Custom EventArgs ===");

var store = new OrderProcessor();

store.OrderPlaced += (sender, e) =>
    Console.WriteLine($"  Order #{e.OrderId} placed: {e.Item} x{e.Quantity} (${e.TotalPrice:F2})");

store.OrderPlaced += (sender, e) =>
{
    if (e.TotalPrice > 100)
        Console.WriteLine($"  VIP order detected! Order #{e.OrderId}");
};

store.PlaceOrder("Laptop", 1, 999.99m);
store.PlaceOrder("Mouse", 3, 25.50m);

// ============================================================
// 6. Delegate as Strategy Pattern
// ============================================================

Console.WriteLine("\n=== Delegate as Strategy ===");

var numbers = new List<int> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

// Pass different strategies via Func delegates
List<int> FilterList(List<int> source, Func<int, bool> predicate)
{
    var result = new List<int>();
    foreach (var item in source)
        if (predicate(item))
            result.Add(item);
    return result;
}

var evens = FilterList(numbers, n => n % 2 == 0);
var greaterThan5 = FilterList(numbers, n => n > 5);
var primes = FilterList(numbers, n =>
{
    if (n < 2) return false;
    for (int i = 2; i * i <= n; i++)
        if (n % i == 0) return false;
    return true;
});

Console.WriteLine($"Evens: {string.Join(", ", evens)}");
Console.WriteLine($"Greater than 5: {string.Join(", ", greaterThan5)}");
Console.WriteLine($"Primes: {string.Join(", ", primes)}");

// ============================================================
// Supporting Types
// ============================================================

// Custom EventArgs for temperature changes
public class TemperatureChangedEventArgs : EventArgs
{
    public double Temperature { get; }
    public TemperatureChangedEventArgs(double temperature) => Temperature = temperature;
}

// Thermostat class using EventHandler<T>
public class Thermostat
{
    private double _temperature;

    // Declare the event using EventHandler<TEventArgs>
    public event EventHandler<TemperatureChangedEventArgs>? TemperatureChanged;

    public double CurrentTemperature
    {
        get => _temperature;
        set
        {
            _temperature = value;
            // Raise the event safely using the ?. pattern
            TemperatureChanged?.Invoke(this, new TemperatureChangedEventArgs(value));
        }
    }
}

// Custom EventArgs for orders
public class OrderEventArgs : EventArgs
{
    public int OrderId { get; }
    public string Item { get; }
    public int Quantity { get; }
    public decimal TotalPrice { get; }

    public OrderEventArgs(int orderId, string item, int quantity, decimal totalPrice)
    {
        OrderId = orderId;
        Item = item;
        Quantity = quantity;
        TotalPrice = totalPrice;
    }
}

// Order processor with an event
public class OrderProcessor
{
    private int _nextOrderId = 1000;

    public event EventHandler<OrderEventArgs>? OrderPlaced;

    public void PlaceOrder(string item, int quantity, decimal unitPrice)
    {
        var orderId = _nextOrderId++;
        var total = quantity * unitPrice;

        // Raise the event
        OrderPlaced?.Invoke(this, new OrderEventArgs(orderId, item, quantity, total));
    }
}
