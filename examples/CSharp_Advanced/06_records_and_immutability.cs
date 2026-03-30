// Lesson 06: Records and Immutability
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Collections.Immutable;

// ============================================================
// 1. Record Basics — Value Equality
// ============================================================

Console.WriteLine("=== Record Basics ===");

// Record classes provide value-based equality by default
var p1 = new Person("Alice", 30);
var p2 = new Person("Alice", 30);
var p3 = new Person("Bob", 25);

Console.WriteLine($"p1 == p2: {p1 == p2}");   // True — value equality
Console.WriteLine($"p1 == p3: {p1 == p3}");   // False
Console.WriteLine($"p1.GetHashCode() == p2.GetHashCode(): {p1.GetHashCode() == p2.GetHashCode()}");

// Records auto-generate a readable ToString()
Console.WriteLine($"p1.ToString(): {p1}");

// ============================================================
// 2. Positional Records (Shorthand Syntax)
// ============================================================

Console.WriteLine("\n=== Positional Records ===");

// Positional record — compiler generates init-only properties and Deconstruct
var coord = new Coordinate(47.6, -122.3);
Console.WriteLine($"Coordinate: {coord}");

// Deconstruct
var (lat, lon) = coord;
Console.WriteLine($"Deconstructed: lat={lat}, lon={lon}");

// Properties are init-only — cannot be modified after construction
// coord.Latitude = 0; // Error: init-only property

// ============================================================
// 3. With-Expressions (Non-Destructive Mutation)
// ============================================================

Console.WriteLine("\n=== With-Expressions ===");

var original = new Person("Alice", 30);

// Create a copy with modified properties
var older = original with { Age = 31 };
var renamed = original with { Name = "Alicia" };
var completelyNew = original with { Name = "Bob", Age = 25 };

Console.WriteLine($"Original:      {original}");
Console.WriteLine($"Older:         {older}");
Console.WriteLine($"Renamed:       {renamed}");
Console.WriteLine($"CompletelyNew: {completelyNew}");

// Original is unchanged (immutability preserved)
Console.WriteLine($"Original still: {original}");

// ============================================================
// 4. Record with Additional Members
// ============================================================

Console.WriteLine("\n=== Records with Custom Members ===");

var temp = new Temperature(100, TemperatureUnit.Celsius);
Console.WriteLine($"Temperature: {temp}");
Console.WriteLine($"In Fahrenheit: {temp.ToFahrenheit():F1}°F");
Console.WriteLine($"Is boiling: {temp.IsBoiling}");

var cold = temp with { Value = -10 };
Console.WriteLine($"Cold: {cold}, IsBoiling: {cold.IsBoiling}");

// ============================================================
// 5. Record Structs (C# 10) — Value-Type Records
// ============================================================

Console.WriteLine("\n=== Record Structs ===");

// Record struct: value type with value equality (stack-allocated)
var v1 = new Vector2D(3.0, 4.0);
var v2 = new Vector2D(3.0, 4.0);

Console.WriteLine($"v1: {v1}");
Console.WriteLine($"v1 == v2: {v1 == v2}");
Console.WriteLine($"Magnitude: {v1.Magnitude:F2}");

// With-expression works on record structs too
var v3 = v1 with { X = 5.0 };
Console.WriteLine($"v3 (modified X): {v3}");

// ============================================================
// 6. Record Inheritance
// ============================================================

Console.WriteLine("\n=== Record Inheritance ===");

// Records support inheritance (record classes only, not record structs)
Shape shape1 = new Circle("Red", 5.0);
Shape shape2 = new RectangleShape("Blue", 4.0, 6.0);
Shape shape3 = new Circle("Red", 5.0);

Console.WriteLine($"shape1: {shape1}");
Console.WriteLine($"shape2: {shape2}");
Console.WriteLine($"shape1 == shape3: {shape1 == shape3}"); // True — same type and values

// With-expression preserves derived type
var bigCircle = (Circle)shape1 with { Radius = 10.0 };
Console.WriteLine($"bigCircle: {bigCircle}");

// Equality considers the runtime type
Shape circleAsShape = new Circle("Red", 5.0);
Shape rectAsShape = new RectangleShape("Red", 5.0, 5.0);
Console.WriteLine($"Different types equal? {circleAsShape == rectAsShape}"); // False

// ============================================================
// 7. Immutable Collections
// ============================================================

Console.WriteLine("\n=== Immutable Collections ===");

// ImmutableList — returns a new list on every modification
var list = ImmutableList.Create(1, 2, 3);
var list2 = list.Add(4);
var list3 = list2.Remove(2);

Console.WriteLine($"Original: [{string.Join(", ", list)}]");
Console.WriteLine($"After Add(4): [{string.Join(", ", list2)}]");
Console.WriteLine($"After Remove(2): [{string.Join(", ", list3)}]");

// ImmutableDictionary
var dict = ImmutableDictionary<string, int>.Empty
    .Add("a", 1)
    .Add("b", 2)
    .SetItem("a", 10); // Replace value

Console.WriteLine($"Dict: a={dict["a"]}, b={dict["b"]}");

// Builder pattern for batch modifications (more efficient)
var builder = ImmutableList.CreateBuilder<string>();
builder.Add("alpha");
builder.Add("beta");
builder.Add("gamma");
ImmutableList<string> final = builder.ToImmutable();
Console.WriteLine($"Built: [{string.Join(", ", final)}]");

// ============================================================
// 8. Practical: Immutable Domain Model
// ============================================================

Console.WriteLine("\n=== Immutable Domain Model ===");

var order = new Order(
    Id: 1001,
    Customer: "Alice",
    Items: ImmutableList.Create(
        new OrderItem("Widget", 2, 9.99m),
        new OrderItem("Gadget", 1, 24.99m)
    ),
    Status: OrderStatus.Pending
);

Console.WriteLine($"Order: {order.Id}, Total: ${order.Total:F2}, Status: {order.Status}");

// Evolve state with with-expressions and immutable collection operations
var confirmedOrder = order with
{
    Status = OrderStatus.Confirmed,
    Items = order.Items.Add(new OrderItem("Bonus item", 1, 0.00m))
};

Console.WriteLine($"Confirmed: {confirmedOrder.Items.Count} items, Total: ${confirmedOrder.Total:F2}");
Console.WriteLine($"Original unchanged: {order.Items.Count} items, Status: {order.Status}");

// ============================================================
// Supporting Types
// ============================================================

record Person(string Name, int Age);

record Coordinate(double Latitude, double Longitude);

record Temperature(double Value, TemperatureUnit Unit)
{
    public bool IsBoiling => Unit == TemperatureUnit.Celsius ? Value >= 100 : Value >= 212;

    public double ToFahrenheit() => Unit switch
    {
        TemperatureUnit.Celsius    => Value * 9.0 / 5.0 + 32,
        TemperatureUnit.Fahrenheit => Value,
        _ => throw new NotSupportedException()
    };
}

enum TemperatureUnit { Celsius, Fahrenheit }

record struct Vector2D(double X, double Y)
{
    public double Magnitude => Math.Sqrt(X * X + Y * Y);
}

// Record inheritance hierarchy
abstract record Shape(string Color);
record Circle(string Color, double Radius) : Shape(Color);
record RectangleShape(string Color, double Width, double Height) : Shape(Color);

// Immutable domain model
record OrderItem(string Product, int Quantity, decimal UnitPrice)
{
    public decimal Subtotal => Quantity * UnitPrice;
}

record Order(int Id, string Customer, ImmutableList<OrderItem> Items, OrderStatus Status)
{
    public decimal Total => Items.Sum(i => i.Subtotal);
}

enum OrderStatus { Pending, Confirmed, Shipped, Delivered }
