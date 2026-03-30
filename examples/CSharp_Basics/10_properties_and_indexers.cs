// Lesson 10: Properties and Indexers
// Run: dotnet run

using System;
using System.Collections.Generic;

// =============================================================================
// AUTO-IMPLEMENTED PROPERTIES
// =============================================================================
Console.WriteLine("=== Auto-Implemented Properties ===");

var product = new Product
{
    Name = "Widget",
    Price = 19.99m,
    Category = "Electronics"
};

Console.WriteLine($"Product: {product.Name}, ${product.Price}, {product.Category}");

// Read-only auto property (set only in constructor)
Console.WriteLine($"ID: {product.Id}");

// =============================================================================
// PROPERTIES WITH BACKING FIELDS
// =============================================================================
Console.WriteLine("\n=== Properties with Validation ===");

var person = new Person("Alice", 25);
Console.WriteLine(person);

person.Age = 30;
Console.WriteLine($"After birthday: {person}");

// Validation in setter
try
{
    person.Age = -5;  // Will throw
}
catch (ArgumentException ex)
{
    Console.WriteLine($"Validation error: {ex.Message}");
}

try
{
    person.Name = "";  // Will throw
}
catch (ArgumentException ex)
{
    Console.WriteLine($"Validation error: {ex.Message}");
}

// =============================================================================
// COMPUTED (READ-ONLY) PROPERTIES
// =============================================================================
Console.WriteLine("\n=== Computed Properties ===");

var circle = new Circle(5.0);
Console.WriteLine($"Circle radius: {circle.Radius}");
Console.WriteLine($"Area: {circle.Area:F4}");
Console.WriteLine($"Circumference: {circle.Circumference:F4}");
Console.WriteLine($"Diameter: {circle.Diameter:F4}");

// =============================================================================
// INIT-ONLY PROPERTIES (C# 9+)
// =============================================================================
Console.WriteLine("\n=== Init-Only Properties ===");

var config = new AppConfig
{
    AppName = "MyApp",
    Version = "2.0",
    MaxConnections = 100
};

Console.WriteLine($"Config: {config.AppName} v{config.Version}, max={config.MaxConnections}");

// config.AppName = "Other"; // Compile error: init-only property

// =============================================================================
// REQUIRED PROPERTIES (C# 11+)
// =============================================================================
Console.WriteLine("\n=== Required Properties ===");

var user = new UserProfile
{
    Username = "alice",    // Required — must be set
    Email = "a@test.com", // Required — must be set
    Bio = "Developer"     // Optional
};

Console.WriteLine($"User: {user.Username}, {user.Email}, Bio: {user.Bio ?? "(none)"}");

// var invalid = new UserProfile { Username = "bob" };
// Compile error: required member 'Email' must be set

// =============================================================================
// EXPRESSION-BODIED PROPERTY MEMBERS
// =============================================================================
Console.WriteLine("\n=== Expression-Bodied Properties ===");

var temp = new TemperatureConverter(100);
Console.WriteLine($"Celsius: {temp.Celsius}°C");
Console.WriteLine($"Fahrenheit: {temp.Fahrenheit:F2}°F");
Console.WriteLine($"Kelvin: {temp.Kelvin:F2}K");
Console.WriteLine($"Summary: {temp.Summary}");

// =============================================================================
// INDEXERS
// =============================================================================
Console.WriteLine("\n=== Indexers ===");

// Integer indexer
var sentence = new WordCollection("The quick brown fox jumps over the lazy dog");
Console.WriteLine($"Word count: {sentence.Count}");
Console.WriteLine($"sentence[0]: {sentence[0]}");
Console.WriteLine($"sentence[3]: {sentence[3]}");
Console.WriteLine($"sentence[^1]: {sentence[^1]}"); // Last word

// String indexer (dictionary-like)
var settings = new Settings();
settings["theme"] = "dark";
settings["language"] = "en";
settings["font_size"] = "14";

Console.WriteLine($"\nSettings:");
Console.WriteLine($"  theme: {settings["theme"]}");
Console.WriteLine($"  language: {settings["language"]}");
Console.WriteLine($"  missing: {settings["missing"]}"); // Returns default

// Multi-dimensional indexer
var matrix = new Matrix(3, 3);
matrix[0, 0] = 1; matrix[0, 1] = 2; matrix[0, 2] = 3;
matrix[1, 0] = 4; matrix[1, 1] = 5; matrix[1, 2] = 6;
matrix[2, 0] = 7; matrix[2, 1] = 8; matrix[2, 2] = 9;

Console.WriteLine($"\nMatrix:");
for (int r = 0; r < 3; r++)
{
    Console.Write("  ");
    for (int c = 0; c < 3; c++)
    {
        Console.Write($"{matrix[r, c],4}");
    }
    Console.WriteLine();
}
Console.WriteLine($"matrix[1,1] = {matrix[1, 1]}");

// =============================================================================
// STATIC PROPERTIES
// =============================================================================
Console.WriteLine("\n=== Static Properties ===");

Console.WriteLine($"Counter.Value before: {Counter.Value}");
Counter.Increment();
Counter.Increment();
Counter.Increment();
Console.WriteLine($"Counter.Value after 3 increments: {Counter.Value}");
Counter.Reset();
Console.WriteLine($"Counter.Value after reset: {Counter.Value}");

// =============================================================================
// CLASS DEFINITIONS
// =============================================================================

class Product
{
    // Auto-implemented properties
    public string Name { get; set; } = "Unknown";
    public decimal Price { get; set; }
    public string Category { get; set; } = "General";

    // Read-only auto property — only set in constructor/initializer
    public Guid Id { get; } = Guid.NewGuid();
}

class Person
{
    // Backing fields for validation
    private string _name = null!;
    private int _age;

    public Person(string name, int age)
    {
        Name = name;  // Uses property setter with validation
        Age = age;
    }

    // Property with validation in setter
    public string Name
    {
        get => _name;
        set
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("Name cannot be empty.");
            _name = value;
        }
    }

    public int Age
    {
        get => _age;
        set
        {
            if (value < 0 || value > 150)
                throw new ArgumentException("Age must be between 0 and 150.");
            _age = value;
        }
    }

    public override string ToString() => $"Person({Name}, age {Age})";
}

class Circle
{
    public double Radius { get; }

    public Circle(double radius)
    {
        Radius = radius > 0 ? radius : throw new ArgumentException("Radius must be positive.");
    }

    // Computed read-only properties (expression-bodied)
    public double Area => Math.PI * Radius * Radius;
    public double Circumference => 2 * Math.PI * Radius;
    public double Diameter => 2 * Radius;
}

/// <summary>
/// Init-only properties: settable during object initialization but immutable after.
/// </summary>
class AppConfig
{
    public string AppName { get; init; } = "Default";
    public string Version { get; init; } = "1.0";
    public int MaxConnections { get; init; } = 10;
}

/// <summary>
/// Required properties (C# 11+): must be set during initialization.
/// </summary>
class UserProfile
{
    public required string Username { get; init; }
    public required string Email { get; init; }
    public string? Bio { get; set; }
}

/// <summary>
/// Expression-bodied property demonstrations.
/// </summary>
class TemperatureConverter
{
    public double Celsius { get; }

    public TemperatureConverter(double celsius) => Celsius = celsius;

    // Expression-bodied read-only properties
    public double Fahrenheit => Celsius * 9.0 / 5.0 + 32;
    public double Kelvin => Celsius + 273.15;
    public string Summary => $"{Celsius}°C / {Fahrenheit:F1}°F / {Kelvin:F1}K";
}

/// <summary>
/// Indexer with int index — behaves like array access.
/// </summary>
class WordCollection
{
    private readonly string[] _words;

    public WordCollection(string sentence)
    {
        _words = sentence.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    public int Count => _words.Length;

    // Integer indexer
    public string this[int index] => _words[index];

    // Index from end
    public string this[Index index] => _words[index];
}

/// <summary>
/// Indexer with string key — dictionary-like access.
/// </summary>
class Settings
{
    private readonly Dictionary<string, string> _data = new();

    public string this[string key]
    {
        get => _data.TryGetValue(key, out string? val) ? val : "(not set)";
        set => _data[key] = value;
    }
}

/// <summary>
/// Multi-dimensional indexer.
/// </summary>
class Matrix
{
    private readonly double[,] _data;

    public Matrix(int rows, int cols)
    {
        _data = new double[rows, cols];
    }

    public double this[int row, int col]
    {
        get => _data[row, col];
        set => _data[row, col] = value;
    }
}

/// <summary>
/// Static properties.
/// </summary>
static class Counter
{
    public static int Value { get; private set; }

    public static void Increment() => Value++;
    public static void Reset() => Value = 0;
}
