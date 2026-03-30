// Lesson 04: Pattern Matching
// Run: dotnet run

using System;
using System.Collections.Generic;

// ============================================================
// 1. Type Patterns
// ============================================================

Console.WriteLine("=== Type Patterns ===");

object[] items = { 42, "hello", 3.14, true, null!, new int[] { 1, 2, 3 } };

foreach (var item in items)
{
    // 'is' with type pattern and variable declaration
    string description = item switch
    {
        int n    => $"Integer: {n}",
        string s => $"String: \"{s}\" (length {s.Length})",
        double d => $"Double: {d:F2}",
        bool b   => $"Boolean: {b}",
        int[] a  => $"Int array with {a.Length} elements",
        null     => "Null value",
        _        => $"Unknown type: {item.GetType().Name}"
    };
    Console.WriteLine($"  {description}");
}

// ============================================================
// 2. Property Patterns
// ============================================================

Console.WriteLine("\n=== Property Patterns ===");

var people = new Person[]
{
    new("Alice", 30, new Address("Seattle", "USA")),
    new("Bob", 25, new Address("London", "UK")),
    new("Charlie", 35, new Address("Tokyo", "Japan")),
    new("Diana", 28, new Address("New York", "USA")),
};

foreach (var person in people)
{
    // Property pattern — match on property values
    var greeting = person switch
    {
        { Age: < 26 }                                => $"{person.Name} is young",
        { Address.Country: "USA", Age: >= 30 }       => $"{person.Name} is a senior US resident",
        { Address: { City: "Tokyo" } }               => $"{person.Name} lives in Tokyo",
        _                                            => $"{person.Name} — no special category"
    };
    Console.WriteLine($"  {greeting}");
}

// ============================================================
// 3. Positional Patterns (with Deconstruct)
// ============================================================

Console.WriteLine("\n=== Positional Patterns ===");

var points = new Point[]
{
    new(0, 0), new(1, 0), new(0, 5), new(3, 4), new(-2, -3)
};

foreach (var point in points)
{
    // Positional pattern uses Deconstruct
    var quadrant = point switch
    {
        (0, 0)           => "Origin",
        ( > 0, > 0)      => "Quadrant I",
        ( < 0, > 0)      => "Quadrant II",
        ( < 0, < 0)      => "Quadrant III",
        ( > 0, < 0)      => "Quadrant IV",
        (0, _)           => "On Y-axis",
        (_, 0)           => "On X-axis",
    };
    Console.WriteLine($"  ({point.X}, {point.Y}) -> {quadrant}");
}

// ============================================================
// 4. Relational and Logical Patterns
// ============================================================

Console.WriteLine("\n=== Relational and Logical Patterns ===");

int[] scores = { 95, 82, 67, 45, 30, 100, 55 };

foreach (var score in scores)
{
    // Relational patterns with 'and', 'or', 'not'
    var grade = score switch
    {
        >= 90 and <= 100 => "A",
        >= 80 and < 90   => "B",
        >= 70 and < 80   => "C",
        >= 60 and < 70   => "D",
        >= 0 and < 60    => "F",
        _                => "Invalid"
    };
    Console.WriteLine($"  Score {score} -> Grade {grade}");
}

// 'not' pattern
object value = "test";
if (value is not null and not int)
    Console.WriteLine($"\n  value is neither null nor int: {value}");

// ============================================================
// 5. List Patterns (C# 11)
// ============================================================

Console.WriteLine("\n=== List Patterns (C# 11) ===");

int[][] arrays =
{
    new[] { 1 },
    new[] { 1, 2 },
    new[] { 1, 2, 3 },
    new[] { 0, 42, 99 },
    new[] { 1, 2, 3, 4, 5 },
    Array.Empty<int>(),
};

foreach (var arr in arrays)
{
    var result = arr switch
    {
        []              => "Empty",
        [var single]    => $"Single element: {single}",
        [1, 2]          => "Exactly [1, 2]",
        [1, 2, 3]       => "Exactly [1, 2, 3]",
        [0, .., var last] => $"Starts with 0, ends with {last}",
        [_, _, .. var rest] => $"2+ elements, rest has {rest.Length} items",
    };
    Console.WriteLine($"  [{string.Join(", ", arr)}] -> {result}");
}

// Slice pattern with discard
int[] data = { 10, 20, 30, 40, 50 };
if (data is [var first, .. var middle, var last])
    Console.WriteLine($"\n  First={first}, Middle=[{string.Join(",", middle)}], Last={last}");

// ============================================================
// 6. Switch Expression with Complex Patterns
// ============================================================

Console.WriteLine("\n=== Complex Pattern Example: Shape Area ===");

Shape[] shapes =
{
    new Circle(5),
    new Rectangle(4, 6),
    new Triangle(3, 8),
    new Circle(0),
};

foreach (var shape in shapes)
{
    var (description, area) = shape switch
    {
        Circle { Radius: 0 }     => ("Degenerate circle", 0.0),
        Circle c                  => ($"Circle r={c.Radius}", Math.PI * c.Radius * c.Radius),
        Rectangle { W: var w, H: var h } when w == h
                                  => ($"Square side={w}", w * h),
        Rectangle r               => ($"Rectangle {r.W}x{r.H}", r.W * r.H),
        Triangle { Base: var b, Height: var h }
                                  => ($"Triangle b={b} h={h}", 0.5 * b * h),
        _                         => ("Unknown", 0.0)
    };
    Console.WriteLine($"  {description} -> area = {area:F2}");
}

// ============================================================
// 7. Pattern Matching in if/when
// ============================================================

Console.WriteLine("\n=== Patterns in if Statements ===");

object input = 42;

// Combining 'is' with property/relational patterns
if (input is int number and > 0 and < 100)
    Console.WriteLine($"  {number} is a positive integer under 100");

// When clause in switch
var status = (int code) => code switch
{
    >= 200 and < 300 => "Success",
    >= 300 and < 400 => "Redirect",
    >= 400 and < 500 => "Client Error",
    >= 500           => "Server Error",
    _                => "Unknown"
};

int[] codes = { 200, 301, 404, 500, 102 };
foreach (var code in codes)
    Console.WriteLine($"  HTTP {code}: {status(code)}");

// ============================================================
// Supporting Types
// ============================================================

record Person(string Name, int Age, Address Address);
record Address(string City, string Country);

record Point(int X, int Y);

abstract record Shape;
record Circle(double Radius) : Shape;
record Rectangle(double W, double H) : Shape;
record Triangle(double Base, double Height) : Shape;
