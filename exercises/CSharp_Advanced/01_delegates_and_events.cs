/*
 * Exercises for Lesson 01: Delegates and Events
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;

// ---------------------------------------------------------------------------
// Exercise 1: Delegate chaining — build a text-processing pipeline
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Delegate Chaining ===");

    Func<string, string> trim = s => s.Trim();
    Func<string, string> lower = s => s.ToLower();
    Func<string, string> removeDoubleSpaces = s =>
    {
        while (s.Contains("  ")) s = s.Replace("  ", " ");
        return s;
    };

    // Chain delegates manually
    Func<string, string> pipeline = input =>
        removeDoubleSpaces(lower(trim(input)));

    string raw = "   Hello   WORLD   from   C#   ";
    string result = pipeline(raw);
    Console.WriteLine($"Input : '{raw}'");
    Console.WriteLine($"Output: '{result}'");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Multicast delegate — invoke multiple handlers
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Multicast Delegate ===");

    Action<string> log = message => Console.WriteLine($"[LOG]   {message}");
    log += message => Console.WriteLine($"[TRACE] {message}");
    log += message => Console.WriteLine($"[AUDIT] {message}");

    log("User logged in");
    Console.WriteLine($"Delegate invocation list count: {log.GetInvocationList().Length}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Event publisher / subscriber — stock price alert
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Event Publisher/Subscriber ===");

    var monitor = new StockPriceMonitor("MSFT", threshold: 300m);

    // Subscriber 1: console alert
    monitor.PriceThresholdReached += (sender, args) =>
        Console.WriteLine($"  ALERT: {args.Symbol} hit {args.Price:C} at {args.Timestamp:T}");

    // Subscriber 2: logging
    monitor.PriceThresholdReached += (sender, args) =>
        Console.WriteLine($"  LOG  : threshold breached for {args.Symbol}");

    monitor.UpdatePrice(280m);
    monitor.UpdatePrice(305m); // triggers event
    monitor.UpdatePrice(310m); // triggers again
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Predicate delegate — generic filter method
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Predicate Delegate Filter ===");

    List<int> numbers = new() { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

    Predicate<int> isEven = n => n % 2 == 0;
    Predicate<int> isGreaterThanFive = n => n > 5;

    var evens = FilterList(numbers, isEven);
    var greaterThanFive = FilterList(numbers, isGreaterThanFive);

    Console.WriteLine($"Evens: {string.Join(", ", evens)}");
    Console.WriteLine($"> 5  : {string.Join(", ", greaterThanFive)}");
    Console.WriteLine();
}

List<T> FilterList<T>(List<T> source, Predicate<T> predicate)
{
    var result = new List<T>();
    foreach (var item in source)
        if (predicate(item)) result.Add(item);
    return result;
}

// ---------------------------------------------------------------------------
// Exercise 5: Custom event args — order processing pipeline
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Custom EventArgs — Order Pipeline ===");

    var processor = new OrderProcessor();

    processor.OrderReceived += (s, e) =>
        Console.WriteLine($"  Received : Order#{e.OrderId} — {e.Item} x{e.Quantity}");
    processor.OrderValidated += (s, e) =>
        Console.WriteLine($"  Validated: Order#{e.OrderId}");
    processor.OrderShipped += (s, e) =>
        Console.WriteLine($"  Shipped  : Order#{e.OrderId}");

    processor.ProcessOrder(1001, "Keyboard", 2);
    processor.ProcessOrder(1002, "Monitor", 1);
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

class StockPriceEventArgs : EventArgs
{
    public string Symbol { get; init; } = "";
    public decimal Price { get; init; }
    public DateTime Timestamp { get; init; }
}

class StockPriceMonitor
{
    private readonly string _symbol;
    private readonly decimal _threshold;
    public event EventHandler<StockPriceEventArgs>? PriceThresholdReached;

    public StockPriceMonitor(string symbol, decimal threshold)
    {
        _symbol = symbol;
        _threshold = threshold;
    }

    public void UpdatePrice(decimal newPrice)
    {
        Console.WriteLine($"  Price update: {_symbol} = {newPrice:C}");
        if (newPrice >= _threshold)
            PriceThresholdReached?.Invoke(this,
                new StockPriceEventArgs { Symbol = _symbol, Price = newPrice, Timestamp = DateTime.Now });
    }
}

class OrderEventArgs : EventArgs
{
    public int OrderId { get; init; }
    public string Item { get; init; } = "";
    public int Quantity { get; init; }
}

class OrderProcessor
{
    public event EventHandler<OrderEventArgs>? OrderReceived;
    public event EventHandler<OrderEventArgs>? OrderValidated;
    public event EventHandler<OrderEventArgs>? OrderShipped;

    public void ProcessOrder(int id, string item, int qty)
    {
        var args = new OrderEventArgs { OrderId = id, Item = item, Quantity = qty };
        OrderReceived?.Invoke(this, args);
        OrderValidated?.Invoke(this, args);
        OrderShipped?.Invoke(this, args);
    }
}
