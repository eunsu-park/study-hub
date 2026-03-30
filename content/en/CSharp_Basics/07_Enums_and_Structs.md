# Enums and Structs

**Previous**: [Arrays and Strings](./06_Arrays_and_Strings.md) | **Next**: [Collections](./08_Collections.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and use enumerations with custom underlying types
2. Apply the `[Flags]` attribute for bitwise enum operations
3. Parse and iterate enum values at runtime
4. Define structs and understand their value semantics
5. Use `readonly struct` for immutable value types
6. Distinguish when to use struct vs class
7. Work with record structs and tuples as lightweight data carriers

---

Enums and structs are two fundamental value types in C#. Enums give meaningful names to sets of related constants, making code more readable and less error-prone. Structs provide lightweight, stack-allocated containers for small data groupings. Together, they form the backbone of value-type programming in C#.

## 1. Enum Declaration and Usage

An enum (enumeration) defines a set of named integral constants.

### 1.1 Basic Enum Declaration

```csharp
// Basic enum declaration
enum Season
{
    Spring,   // 0
    Summer,   // 1
    Autumn,   // 2
    Winter    // 3
}

// Usage
Season current = Season.Summer;
Console.WriteLine(current);        // "Summer"
Console.WriteLine((int)current);   // 1

// Comparison
if (current == Season.Summer)
{
    Console.WriteLine("Time for vacation!");
}

// Switch on enum
string description = current switch
{
    Season.Spring => "Flowers bloom",
    Season.Summer => "Sun is high",
    Season.Autumn => "Leaves fall",
    Season.Winter => "Snow falls",
    _ => "Unknown season"
};
```

### 1.2 Custom Values

By default, enum members start at 0 and increment by 1. You can assign explicit values:

```csharp
enum HttpStatusCode
{
    OK = 200,
    Created = 201,
    NoContent = 204,
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    InternalServerError = 500
}

HttpStatusCode status = HttpStatusCode.NotFound;
int code = (int)status; // 404

// Enum members can reference other members
enum Priority
{
    Low = 1,
    Medium = 5,
    High = 10,
    Critical = High + 10  // 20
}
```

### 1.3 Enum in Methods

```csharp
enum Direction { North, South, East, West }

static (int dx, int dy) GetDelta(Direction dir)
{
    return dir switch
    {
        Direction.North => (0, 1),
        Direction.South => (0, -1),
        Direction.East  => (1, 0),
        Direction.West  => (-1, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(dir))
    };
}

// Usage
var (dx, dy) = GetDelta(Direction.North);
Console.WriteLine($"dx={dx}, dy={dy}"); // dx=0, dy=1
```

## 2. Enum Underlying Types

By default, enums use `int` as their underlying type. You can specify a different integral type.

### 2.1 Specifying the Underlying Type

```csharp
// Use byte to save memory (0-255 range)
enum Color : byte
{
    Red = 1,
    Green = 2,
    Blue = 3
}

// Use long for large values
enum FileSize : long
{
    Kilobyte = 1024L,
    Megabyte = 1024L * 1024,
    Gigabyte = 1024L * 1024 * 1024,
    Terabyte = 1024L * 1024 * 1024 * 1024
}

// Supported underlying types: byte, sbyte, short, ushort, int, uint, long, ulong
Console.WriteLine(sizeof(Color));    // 1 byte
Console.WriteLine((long)FileSize.Terabyte); // 1099511627776
```

### 2.2 Casting Between Enum and Integer

```csharp
enum Planet { Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune }

// Enum to int
int earthIndex = (int)Planet.Earth; // 2

// Int to enum
Planet planet = (Planet)4; // Jupiter

// Be careful: invalid values are allowed at runtime
Planet invalid = (Planet)99; // No compile error, but logically invalid
Console.WriteLine(invalid);  // "99"

// Validate with Enum.IsDefined
bool isValid = Enum.IsDefined(typeof(Planet), 4);  // true
bool isInvalid = Enum.IsDefined(typeof(Planet), 99); // false
```

## 3. Flags Enums for Bitwise Operations

The `[Flags]` attribute enables combining enum values using bitwise operators, useful for representing sets of options.

### 3.1 Declaring Flags Enums

```csharp
[Flags]
enum FilePermissions
{
    None    = 0,       // 0b0000
    Read    = 1,       // 0b0001
    Write   = 2,       // 0b0010
    Execute = 4,       // 0b0100
    All     = Read | Write | Execute  // 0b0111 = 7
}

// Combine flags with bitwise OR
FilePermissions userPerms = FilePermissions.Read | FilePermissions.Write;
Console.WriteLine(userPerms); // "Read, Write"

// Check if a flag is set with HasFlag
bool canRead = userPerms.HasFlag(FilePermissions.Read);   // true
bool canExec = userPerms.HasFlag(FilePermissions.Execute); // false

// Alternative: check with bitwise AND (faster)
bool canWrite = (userPerms & FilePermissions.Write) != 0; // true
```

### 3.2 Manipulating Flags

```csharp
[Flags]
enum DaysOfWeek
{
    None      = 0,
    Monday    = 1,
    Tuesday   = 2,
    Wednesday = 4,
    Thursday  = 8,
    Friday    = 16,
    Saturday  = 32,
    Sunday    = 64,
    Weekdays  = Monday | Tuesday | Wednesday | Thursday | Friday,
    Weekend   = Saturday | Sunday,
    All       = Weekdays | Weekend
}

DaysOfWeek schedule = DaysOfWeek.Monday | DaysOfWeek.Wednesday | DaysOfWeek.Friday;

// Add a flag
schedule |= DaysOfWeek.Tuesday;

// Remove a flag
schedule &= ~DaysOfWeek.Monday;

// Toggle a flag
schedule ^= DaysOfWeek.Friday;

// Check membership
bool worksWednesday = schedule.HasFlag(DaysOfWeek.Wednesday); // true

// Check if schedule contains any weekday
bool hasWeekday = (schedule & DaysOfWeek.Weekdays) != 0; // true

Console.WriteLine(schedule); // "Tuesday, Wednesday"
```

### 3.3 Flags Best Practices

```csharp
// Always assign powers of 2 (or use bit shift for clarity)
[Flags]
enum TextStyle
{
    None          = 0,
    Bold          = 1 << 0,  // 1
    Italic        = 1 << 1,  // 2
    Underline     = 1 << 2,  // 4
    Strikethrough = 1 << 3,  // 8
    Highlight     = 1 << 4,  // 16
    BoldItalic    = Bold | Italic  // Convenience combination
}

TextStyle style = TextStyle.Bold | TextStyle.Underline;
Console.WriteLine(style); // "Bold, Underline"
```

## 4. Enum Utility Methods

The `System.Enum` class provides static methods for working with enums at runtime.

### 4.1 Parse and TryParse

```csharp
enum Color { Red, Green, Blue }

// Parse from string (case-sensitive by default)
Color parsed = (Color)Enum.Parse(typeof(Color), "Green");
Console.WriteLine(parsed); // Green

// Generic parse (C# 7+)
Color parsed2 = Enum.Parse<Color>("Blue");

// Case-insensitive parse
Color parsed3 = Enum.Parse<Color>("red", ignoreCase: true);

// TryParse (safe, no exception)
if (Enum.TryParse<Color>("Green", out Color result))
{
    Console.WriteLine($"Parsed: {result}"); // Parsed: Green
}

if (!Enum.TryParse<Color>("Yellow", out Color _))
{
    Console.WriteLine("Yellow is not a valid Color");
}
```

### 4.2 GetValues and GetNames

```csharp
enum Fruit { Apple, Banana, Cherry, Date, Elderberry }

// Get all values
Fruit[] allFruits = Enum.GetValues<Fruit>();
foreach (Fruit f in allFruits)
{
    Console.WriteLine($"{f} = {(int)f}");
}
// Apple = 0, Banana = 1, Cherry = 2, Date = 3, Elderberry = 4

// Get all names as strings
string[] names = Enum.GetNames<Fruit>();
Console.WriteLine(string.Join(", ", names));
// "Apple, Banana, Cherry, Date, Elderberry"

// Build a lookup dictionary
Dictionary<string, Fruit> lookup = new();
foreach (Fruit f in Enum.GetValues<Fruit>())
{
    lookup[f.ToString()] = f;
}
```

### 4.3 Convert Enum to Display Strings

```csharp
enum OrderStatus
{
    Pending,
    InProgress,
    Shipped,
    Delivered,
    Cancelled
}

// Simple approach: ToString with manual formatting
static string ToDisplayName(OrderStatus status)
{
    return status switch
    {
        OrderStatus.InProgress => "In Progress",
        _ => status.ToString()
    };
}

// Iterate all values for a dropdown menu
static List<(int Value, string Label)> GetDropdownOptions<T>() where T : struct, Enum
{
    var options = new List<(int, string)>();
    foreach (T value in Enum.GetValues<T>())
    {
        options.Add((Convert.ToInt32(value), value.ToString()));
    }
    return options;
}
```

## 5. Struct Declaration

A struct is a value type that can contain fields, methods, properties, and constructors. Structs are allocated on the stack (when used as local variables) and are copied by value.

### 5.1 Basic Struct

```csharp
struct Point
{
    public double X;
    public double Y;

    // Constructor
    public Point(double x, double y)
    {
        X = x;
        Y = y;
    }

    // Method
    public double DistanceTo(Point other)
    {
        double dx = X - other.X;
        double dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    // Override ToString
    public override string ToString() => $"({X}, {Y})";
}

// Usage
Point p1 = new Point(3, 4);
Point p2 = new Point(0, 0);
double dist = p1.DistanceTo(p2); // 5.0
Console.WriteLine(p1);           // "(3, 4)"

// Default constructor (all fields set to default values)
Point origin = new Point(); // X=0, Y=0
Point same = default;       // Also X=0, Y=0
```

### 5.2 Struct with Properties

```csharp
struct Rectangle
{
    public double Width { get; set; }
    public double Height { get; set; }

    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }

    // Computed properties
    public double Area => Width * Height;
    public double Perimeter => 2 * (Width + Height);
    public bool IsSquare => Width == Height;

    public override string ToString()
        => $"Rectangle({Width} x {Height}, Area={Area})";
}

Rectangle rect = new Rectangle(5, 3);
Console.WriteLine(rect.Area);      // 15
Console.WriteLine(rect.Perimeter); // 16
Console.WriteLine(rect.IsSquare);  // false
```

## 6. Value Semantics vs Reference Semantics

The most important distinction between structs and classes is how they behave when assigned or passed to methods.

### 6.1 Copy Behavior

```csharp
// Struct (value type): assignment copies the data
struct PointStruct
{
    public int X, Y;
}

PointStruct a = new PointStruct { X = 1, Y = 2 };
PointStruct b = a;  // b is a copy of a
b.X = 99;

Console.WriteLine(a.X); // 1 (unchanged)
Console.WriteLine(b.X); // 99

// Class (reference type): assignment copies the reference
class PointClass
{
    public int X, Y;
}

PointClass c = new PointClass { X = 1, Y = 2 };
PointClass d = c;  // d points to the same object as c
d.X = 99;

Console.WriteLine(c.X); // 99 (changed!)
Console.WriteLine(d.X); // 99
```

### 6.2 Method Parameter Behavior

```csharp
struct ValuePoint { public int X, Y; }

static void ModifyStruct(ValuePoint p)
{
    p.X = 999; // Modifies a local copy; caller's value unchanged
}

static void ModifyStructByRef(ref ValuePoint p)
{
    p.X = 999; // Modifies the caller's value directly
}

ValuePoint pt = new ValuePoint { X = 1, Y = 2 };

ModifyStruct(pt);
Console.WriteLine(pt.X); // 1 (unchanged)

ModifyStructByRef(ref pt);
Console.WriteLine(pt.X); // 999 (changed)
```

### 6.3 Equality

```csharp
struct Coordinate
{
    public int X, Y;
}

Coordinate a = new Coordinate { X = 5, Y = 10 };
Coordinate b = new Coordinate { X = 5, Y = 10 };

// Structs use value equality by default (via reflection, slow)
Console.WriteLine(a.Equals(b)); // true

// For performance, override Equals and GetHashCode
struct BetterCoordinate : IEquatable<BetterCoordinate>
{
    public int X, Y;

    public bool Equals(BetterCoordinate other)
        => X == other.X && Y == other.Y;

    public override bool Equals(object? obj)
        => obj is BetterCoordinate other && Equals(other);

    public override int GetHashCode()
        => HashCode.Combine(X, Y);

    public static bool operator ==(BetterCoordinate left, BetterCoordinate right)
        => left.Equals(right);

    public static bool operator !=(BetterCoordinate left, BetterCoordinate right)
        => !left.Equals(right);
}
```

## 7. Readonly Struct

A `readonly struct` guarantees that no instance member can modify the struct's state after construction.

### 7.1 Declaration

```csharp
readonly struct Vector3
{
    public double X { get; }
    public double Y { get; }
    public double Z { get; }

    public Vector3(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public double Magnitude
        => Math.Sqrt(X * X + Y * Y + Z * Z);

    public Vector3 Normalized()
    {
        double mag = Magnitude;
        return new Vector3(X / mag, Y / mag, Z / mag);
    }

    // Operator overloading
    public static Vector3 operator +(Vector3 a, Vector3 b)
        => new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

    public static Vector3 operator *(Vector3 v, double scalar)
        => new Vector3(v.X * scalar, v.Y * scalar, v.Z * scalar);

    public static double Dot(Vector3 a, Vector3 b)
        => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    public override string ToString() => $"<{X}, {Y}, {Z}>";
}

Vector3 v1 = new Vector3(1, 2, 3);
Vector3 v2 = new Vector3(4, 5, 6);
Vector3 sum = v1 + v2;                    // <5, 7, 9>
double dot = Vector3.Dot(v1, v2);         // 32
Vector3 scaled = v1 * 2.0;               // <2, 4, 6>
Console.WriteLine(v1.Normalized());        // <0.267..., 0.534..., 0.801...>
```

### 7.2 Readonly Members in Non-Readonly Structs

Even if the entire struct is not `readonly`, individual members can be marked `readonly`:

```csharp
struct MutablePoint
{
    public double X;
    public double Y;

    // This method promises not to modify the struct
    public readonly double DistanceFromOrigin()
        => Math.Sqrt(X * X + Y * Y);

    // This method modifies state, so it cannot be readonly
    public void Reset()
    {
        X = 0;
        Y = 0;
    }
}
```

## 8. Struct vs Class: When to Use Which

### 8.1 Decision Guidelines

| Use **struct** when... | Use **class** when... |
|------------------------|----------------------|
| Data is small (< ~16 bytes) | Data is large or complex |
| Represents a single value (like a coordinate) | Represents an entity with identity |
| Instances are short-lived | Instances are long-lived or shared |
| No need for inheritance | Need inheritance or polymorphism |
| Immutability is desired | Mutable state with shared references |
| Frequently allocated (avoid GC pressure) | Need null semantics |

### 8.2 Examples in the .NET Framework

```csharp
// These are structs in .NET (small, value-like)
DateTime birthday = new DateTime(1990, 5, 15);
TimeSpan duration = TimeSpan.FromHours(2.5);
Guid id = Guid.NewGuid();
decimal price = 29.99m;
KeyValuePair<string, int> pair = new("Alice", 30);

// These are classes in .NET (complex, identity-based)
string name = "Hello";         // Immutable reference type
List<int> list = new();        // Mutable, shared
Exception ex = new Exception(); // Inheritance hierarchy
Stream stream = File.Open("f.txt", FileMode.Open); // Resource management
```

### 8.3 Boxing Warning

When a struct is assigned to an `object` or interface variable, it gets boxed (copied to the heap):

```csharp
struct SmallData { public int Value; }

SmallData data = new SmallData { Value = 42 };

// Boxing: copies struct to heap, wraps in object
object boxed = data;

// Unboxing: copies back from heap
SmallData unboxed = (SmallData)boxed;

// Modifying unboxed does not affect boxed
unboxed.Value = 99;
Console.WriteLine(((SmallData)boxed).Value); // Still 42

// Avoid boxing in hot paths (use generics instead of object)
```

## 9. Record Struct

C# 10 introduced `record struct` for concise, value-type data carriers with built-in equality, `ToString`, and deconstruction.

### 9.1 Positional Record Struct

```csharp
// Concise syntax: compiler generates properties, Equals, GetHashCode, ToString
record struct Point(double X, double Y);

Point p1 = new Point(3, 4);
Point p2 = new Point(3, 4);

Console.WriteLine(p1);          // "Point { X = 3, Y = 4 }"
Console.WriteLine(p1 == p2);    // true (value equality)

// Deconstruction
var (x, y) = p1;
Console.WriteLine($"x={x}, y={y}"); // x=3, y=4

// Mutable by default (unlike record class)
p1.X = 10;
Console.WriteLine(p1); // "Point { X = 10, Y = 4 }"
```

### 9.2 Readonly Record Struct

```csharp
// Immutable record struct
readonly record struct Color(byte R, byte G, byte B)
{
    // Additional computed property
    public string Hex => $"#{R:X2}{G:X2}{B:X2}";
}

Color red = new Color(255, 0, 0);
Console.WriteLine(red);      // "Color { R = 255, G = 0, B = 0 }"
Console.WriteLine(red.Hex);  // "#FF0000"

// with expression creates a copy with modifications
Color darkRed = red with { R = 139 };
Console.WriteLine(darkRed.Hex); // "#8B0000"
```

### 9.3 Record Struct vs Struct vs Record Class

```csharp
// Regular struct: manual Equals/GetHashCode/ToString
struct ManualPoint
{
    public double X, Y;
    // Must override Equals, GetHashCode, ToString manually
}

// Record struct: auto-generated Equals/GetHashCode/ToString, value type
record struct AutoPoint(double X, double Y);

// Record class: auto-generated, reference type, immutable by default
record class RefPoint(double X, double Y);

// Key differences:
// - record struct is a VALUE type (stack allocated, copied on assignment)
// - record class is a REFERENCE type (heap allocated, shared on assignment)
// - record struct is mutable by default; record class is immutable by default
```

## 10. Tuples

Tuples provide a lightweight way to group multiple values without defining a full type.

### 10.1 ValueTuple Basics

```csharp
// Create a tuple
(string Name, int Age) person = ("Alice", 30);
Console.WriteLine(person.Name); // "Alice"
Console.WriteLine(person.Age);  // 30

// Tuple with var
var point = (X: 3.0, Y: 4.0);
Console.WriteLine($"({point.X}, {point.Y})");

// Unnamed tuple (accessed by Item1, Item2, etc.)
var unnamed = (42, "hello", true);
Console.WriteLine(unnamed.Item1); // 42
Console.WriteLine(unnamed.Item2); // "hello"
```

### 10.2 Tuples in Methods

```csharp
// Return multiple values from a method
static (double Min, double Max, double Average) GetStats(int[] numbers)
{
    double min = numbers.Min();
    double max = numbers.Max();
    double avg = numbers.Average();
    return (min, max, avg);
}

int[] data = { 5, 3, 8, 1, 9 };
var stats = GetStats(data);
Console.WriteLine($"Min={stats.Min}, Max={stats.Max}, Avg={stats.Average}");
// Min=1, Max=9, Avg=5.2

// Deconstruct into individual variables
var (min, max, avg) = GetStats(data);
Console.WriteLine($"Range: {min} to {max}");
```

### 10.3 Tuple Equality and Deconstruction

```csharp
// Tuples support value equality
var a = (1, "hello");
var b = (1, "hello");
Console.WriteLine(a == b); // true

// Deconstruction with discard
var (name, _, age) = ("Alice", "Middle", 30);
// Ignores the second element

// Tuple assignment (swap)
int x = 1, y = 2;
(x, y) = (y, x);
Console.WriteLine($"x={x}, y={y}"); // x=2, y=1

// Nested tuples
var nested = ((1, 2), (3, 4));
var ((a1, a2), (b1, b2)) = nested;
```

### 10.4 When to Use Tuples vs Named Types

```csharp
// Good use: internal helper method returning a few values
static (bool Success, string Message) Validate(string input)
{
    if (string.IsNullOrWhiteSpace(input))
        return (false, "Input cannot be empty");
    if (input.Length > 100)
        return (false, "Input too long");
    return (true, "Valid");
}

// Consider a named type for public APIs or complex data
// (Better readability and documentation)
record struct ValidationResult(bool Success, string Message);

static ValidationResult ValidateBetter(string input)
{
    if (string.IsNullOrWhiteSpace(input))
        return new ValidationResult(false, "Input cannot be empty");
    return new ValidationResult(true, "Valid");
}
```

## 11. Practice Problems

1. **Traffic Light Enum**: Define an enum `TrafficLight` with values `Red`, `Yellow`, and `Green`. Write a method `TrafficLight NextLight(TrafficLight current)` that returns the next state in the cycle (Red -> Green -> Yellow -> Red). Also write a method that returns the recommended wait time in seconds for each light color.

2. **Permission System**: Create a `[Flags]` enum `Permission` with values `None`, `Read`, `Write`, `Execute`, `Delete`, and `Admin` (combination of all). Write methods `void GrantPermission(ref Permission current, Permission toGrant)`, `void RevokePermission(ref Permission current, Permission toRevoke)`, and `bool HasPermission(Permission current, Permission required)`.

3. **Immutable Money Struct**: Create a `readonly struct Money` with properties `decimal Amount` and `string Currency`. Implement operator overloading for `+`, `-`, and `*` (scaling by a number). Throw `InvalidOperationException` if currencies do not match for addition or subtraction. Override `ToString` to display as `"$123.45 USD"`.

4. **Color Converter**: Create a `readonly record struct RgbColor(byte R, byte G, byte B)` with methods to convert to and from a hex string (`"#FF0000"` for red). Also add a method `RgbColor Mix(RgbColor other)` that averages the two colors component-wise.

5. **Enum Parser Menu**: Write a generic method `T PromptForEnum<T>(string prompt) where T : struct, Enum` that prints all possible values of an enum, asks the user to enter a name, and returns the parsed enum value. Use `Enum.GetNames<T>()` and `Enum.TryParse<T>()`. Handle invalid input by re-prompting.
