// Lesson 07: Enums and Structs
// Run: dotnet run

using System;

// =============================================================================
// BASIC ENUMS
// =============================================================================
Console.WriteLine("=== Basic Enums ===");

// Using an enum
Season currentSeason = Season.Summer;
Console.WriteLine($"Current season: {currentSeason}");
Console.WriteLine($"Enum value (int): {(int)currentSeason}");

// Iterate all enum values
Console.Write("All seasons: ");
foreach (Season s in Enum.GetValues<Season>())
{
    Console.Write($"{s}({(int)s}) ");
}
Console.WriteLine();

// Parse from string
Season parsed = Enum.Parse<Season>("Winter");
Console.WriteLine($"Parsed: {parsed}");

// TryParse
if (Enum.TryParse<Season>("Autumn", out Season result))
{
    Console.WriteLine($"TryParse: {result}");
}

// Check if value is defined
Console.WriteLine($"IsDefined(3): {Enum.IsDefined(typeof(Season), 3)}");
Console.WriteLine($"IsDefined(99): {Enum.IsDefined(typeof(Season), 99)}");

// =============================================================================
// ENUM WITH CUSTOM VALUES
// =============================================================================
Console.WriteLine("\n=== Enum with Custom Values ===");

HttpStatus status = HttpStatus.NotFound;
Console.WriteLine($"Status: {status} = {(int)status}");

// Switch on enum
string message = status switch
{
    HttpStatus.OK => "Success",
    HttpStatus.BadRequest => "Client error",
    HttpStatus.NotFound => "Resource not found",
    HttpStatus.InternalServerError => "Server error",
    _ => "Unknown status"
};
Console.WriteLine($"Message: {message}");

// =============================================================================
// FLAGS ENUM (bitwise combinable)
// =============================================================================
Console.WriteLine("\n=== [Flags] Enum ===");

// Combine permissions with bitwise OR
Permission userPerms = Permission.Read | Permission.Write;
Console.WriteLine($"User permissions: {userPerms}");
Console.WriteLine($"User value: {(int)userPerms}");

Permission adminPerms = Permission.All;
Console.WriteLine($"Admin permissions: {adminPerms}");
Console.WriteLine($"Admin value: {(int)adminPerms}");

// Check for a specific flag
bool canWrite = userPerms.HasFlag(Permission.Write);
bool canDelete = userPerms.HasFlag(Permission.Delete);
Console.WriteLine($"User can write: {canWrite}");
Console.WriteLine($"User can delete: {canDelete}");

// Add a permission
userPerms |= Permission.Execute;
Console.WriteLine($"After adding Execute: {userPerms}");

// Remove a permission
userPerms &= ~Permission.Write;
Console.WriteLine($"After removing Write: {userPerms}");

// Toggle a permission
userPerms ^= Permission.Delete;
Console.WriteLine($"After toggling Delete: {userPerms}");

// =============================================================================
// BASIC STRUCTS
// =============================================================================
Console.WriteLine("\n=== Basic Structs ===");

// Structs are value types (stored on the stack)
Point2D p1 = new Point2D(3.0, 4.0);
Point2D p2 = p1; // Copy, not reference
p2.X = 10.0;     // Modifying p2 does NOT affect p1

Console.WriteLine($"p1: {p1}");
Console.WriteLine($"p2: {p2}");
Console.WriteLine($"p1 distance from origin: {p1.DistanceFromOrigin():F4}");

// Default struct (all fields zero-initialized)
Point2D origin = default;
Console.WriteLine($"default Point2D: {origin}");

// =============================================================================
// READONLY STRUCT
// =============================================================================
Console.WriteLine("\n=== Readonly Struct ===");

// Readonly structs guarantee immutability — all fields must be readonly
Temperature temp = new Temperature(98.6, TemperatureUnit.Fahrenheit);
Console.WriteLine($"Temperature: {temp}");
Console.WriteLine($"In Celsius: {temp.ToCelsius():F2}°C");
Console.WriteLine($"In Fahrenheit: {temp.ToFahrenheit():F2}°F");

// temp.Value = 100; // Compile error: readonly struct fields cannot be modified

// =============================================================================
// STRUCT WITH CONSTRUCTOR OVERLOADS
// =============================================================================
Console.WriteLine("\n=== Struct with Multiple Constructors ===");

Color red = new Color(255, 0, 0);
Color fromHex = Color.FromHex("#00FF00");
Console.WriteLine($"Red: {red}");
Console.WriteLine($"FromHex: {fromHex}");
Console.WriteLine($"Are equal: {red.Equals(fromHex)}");

// =============================================================================
// TUPLES (lightweight struct-like containers)
// =============================================================================
Console.WriteLine("\n=== Tuples ===");

// Unnamed tuple
(int, string, bool) record = (1, "Alice", true);
Console.WriteLine($"Tuple: ({record.Item1}, {record.Item2}, {record.Item3})");

// Named tuple
(string Name, int Age, string City) person = ("Bob", 30, "NYC");
Console.WriteLine($"Person: {person.Name}, {person.Age}, {person.City}");

// Tuple deconstruction
var (name, age, city) = person;
Console.WriteLine($"Deconstructed: {name}, {age}, {city}");

// Tuple equality (C# 7.3+)
var t1 = (1, "hello");
var t2 = (1, "hello");
Console.WriteLine($"\nTuple equality: {t1 == t2}");

// Tuple as return value
var (min, max) = MinMax(new[] { 5, 2, 8, 1, 9 });
Console.WriteLine($"MinMax: min={min}, max={max}");

// Swap using tuples
int a = 10, b = 20;
(a, b) = (b, a);
Console.WriteLine($"Swapped: a={a}, b={b}");

// =============================================================================
// VALUE TYPE VS REFERENCE TYPE BEHAVIOR
// =============================================================================
Console.WriteLine("\n=== Value vs Reference Semantics ===");

// Struct (value type): copying creates independent copies
Point2D original = new Point2D(1, 2);
Point2D clone = original;
clone.X = 99;
Console.WriteLine($"Struct original: {original}");
Console.WriteLine($"Struct clone:    {clone}");
Console.WriteLine("  -> Modifying clone did NOT affect original (value semantics)");

// Array (reference type): copying shares the reference
int[] arrOriginal = { 1, 2, 3 };
int[] arrRef = arrOriginal;
arrRef[0] = 99;
Console.WriteLine($"\nArray original: [{string.Join(", ", arrOriginal)}]");
Console.WriteLine($"Array ref:      [{string.Join(", ", arrRef)}]");
Console.WriteLine("  -> Modifying arrRef DID affect original (reference semantics)");

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

// Basic enum (default underlying type is int, values start at 0)
enum Season
{
    Spring = 0,
    Summer = 1,
    Autumn = 2,
    Winter = 3
}

// Enum with custom integer values
enum HttpStatus
{
    OK = 200,
    BadRequest = 400,
    NotFound = 404,
    InternalServerError = 500
}

// Flags enum — each value is a power of 2 for bitwise combination
[Flags]
enum Permission
{
    None    = 0,
    Read    = 1,     // 0001
    Write   = 2,     // 0010
    Execute = 4,     // 0100
    Delete  = 8,     // 1000
    All     = Read | Write | Execute | Delete  // 1111 = 15
}

enum TemperatureUnit { Celsius, Fahrenheit }

// Mutable struct
struct Point2D
{
    public double X;
    public double Y;

    public Point2D(double x, double y)
    {
        X = x;
        Y = y;
    }

    public double DistanceFromOrigin() => Math.Sqrt(X * X + Y * Y);

    public override string ToString() => $"({X}, {Y})";
}

// Readonly struct — guarantees immutability
readonly struct Temperature
{
    public readonly double Value;
    public readonly TemperatureUnit Unit;

    public Temperature(double value, TemperatureUnit unit)
    {
        Value = value;
        Unit = unit;
    }

    public double ToCelsius() => Unit == TemperatureUnit.Celsius
        ? Value
        : (Value - 32) * 5.0 / 9.0;

    public double ToFahrenheit() => Unit == TemperatureUnit.Fahrenheit
        ? Value
        : Value * 9.0 / 5.0 + 32;

    public override string ToString() => $"{Value}°{(Unit == TemperatureUnit.Celsius ? "C" : "F")}";
}

// Struct with static factory method
struct Color
{
    public byte R, G, B;

    public Color(byte r, byte g, byte b)
    {
        R = r; G = g; B = b;
    }

    public static Color FromHex(string hex)
    {
        hex = hex.TrimStart('#');
        byte r = Convert.ToByte(hex[..2], 16);
        byte g = Convert.ToByte(hex[2..4], 16);
        byte b = Convert.ToByte(hex[4..6], 16);
        return new Color(r, g, b);
    }

    public override string ToString() => $"RGB({R}, {G}, {B})";
}

// Helper method
static (int Min, int Max) MinMax(int[] arr)
{
    int min = arr[0], max = arr[0];
    foreach (int n in arr)
    {
        if (n < min) min = n;
        if (n > max) max = n;
    }
    return (min, max);
}
