# Variables and Types

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Operators and Expressions](./03_Operators_and_Expressions.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between value types and reference types
2. Declare and initialize variables using explicit types and `var`
3. Use all common numeric types and understand their ranges
4. Work with `const` and `readonly` for immutable values
5. Handle nullable value types with the `?` suffix
6. Perform implicit and explicit type conversions safely
7. Use string literals, verbatim strings, raw strings, and interpolation

---

Every program works with data, and every piece of data in C# has a type. The C# type system is one of the richest among mainstream programming languages, offering a clear distinction between value types (stored directly on the stack) and reference types (stored on the heap with a reference on the stack). Understanding this distinction is fundamental to writing correct, efficient C# code.

## 1. The C# Type System Overview

C# is a **statically typed** language: every variable, parameter, and expression has a type known at compile time. The type system is organized into two main categories:

```
                    C# Types
                       │
            ┌──────────┴──────────┐
        Value Types          Reference Types
            │                      │
    ┌───────┼───────┐       ┌──────┼──────┐
  Simple  Struct  Enum    Class  Interface  Delegate
    │                       │
  int, double,           string, object,
  bool, char, ...        arrays, ...
```

### 1.1 Value Types vs Reference Types

The key difference is how they are stored in memory:

```csharp
// Value type: data is stored directly in the variable
int x = 42;
int y = x;    // y gets a COPY of the value
y = 100;
Console.WriteLine(x);  // 42 — x is unchanged
Console.WriteLine(y);  // 100

// Reference type: variable stores a reference (pointer) to the data
int[] a = { 1, 2, 3 };
int[] b = a;          // b gets a COPY of the REFERENCE (points to same data)
b[0] = 99;
Console.WriteLine(a[0]);  // 99 — a[0] changed because a and b point to the same array!
```

Memory layout:

```
Value Types (Stack):        Reference Types (Stack + Heap):

Stack:                      Stack:           Heap:
┌────────┐                  ┌────────┐      ┌─────────────┐
│ x: 42  │                  │ a: ──────────▶│ { 1, 2, 3 } │
├────────┤                  ├────────┤      └─────────────┘
│ y: 100 │                  │ b: ──────────▶  (same object)
└────────┘                  └────────┘
```

## 2. Integer Types

C# provides several integer types that differ in size and whether they support negative numbers:

```csharp
// Signed integers (can be negative)
sbyte  smallestSigned = -128;           // 8-bit:  -128 to 127
short  shortValue     = -32_768;        // 16-bit: -32,768 to 32,767
int    intValue       = -2_147_483_648; // 32-bit: ~-2.1 billion to ~2.1 billion
long   longValue      = -9_000_000_000; // 64-bit: ~-9.2 quintillion to ~9.2 quintillion

// Unsigned integers (non-negative only)
byte   byteValue      = 255;            // 8-bit:  0 to 255
ushort ushortValue     = 65_535;         // 16-bit: 0 to 65,535
uint   uintValue       = 4_294_967_295; // 32-bit: 0 to ~4.3 billion
ulong  ulongValue      = 18_000_000_000_000_000_000; // 64-bit: 0 to ~18.4 quintillion

// nint and nuint — platform-dependent (32-bit or 64-bit)
nint  nativeInt  = 42;   // Same size as IntPtr
nuint nativeUint = 42;   // Same size as UIntPtr
```

Note the use of underscores (`_`) as digit separators to improve readability. They have no effect on the value.

### 2.1 Integer Literals

```csharp
// Decimal (base 10)
int decimal_ = 42;

// Hexadecimal (base 16) — prefix 0x
int hex = 0x2A;           // 42

// Binary (base 2) — prefix 0b
int binary = 0b0010_1010; // 42

// Long literal — suffix L
long bigNumber = 3_000_000_000L;

// Unsigned literal — suffix U
uint positive = 42U;

// Unsigned long — suffix UL
ulong veryBig = 42UL;

Console.WriteLine($"decimal={decimal_}, hex={hex}, binary={binary}");
// Output: decimal=42, hex=42, binary=42
```

### 2.2 Checking Type Ranges

```csharp
Console.WriteLine($"int:    {int.MinValue:N0} to {int.MaxValue:N0}");
Console.WriteLine($"long:   {long.MinValue:N0} to {long.MaxValue:N0}");
Console.WriteLine($"byte:   {byte.MinValue} to {byte.MaxValue}");
Console.WriteLine($"short:  {short.MinValue:N0} to {short.MaxValue:N0}");

// Output:
// int:    -2,147,483,648 to 2,147,483,647
// long:   -9,223,372,036,854,775,808 to 9,223,372,036,854,775,807
// byte:   0 to 255
// short:  -32,768 to 32,767
```

## 3. Floating-Point and Decimal Types

For numbers with fractional parts, C# offers three types with different precision and performance characteristics:

```csharp
// float: 32-bit, ~6-7 significant digits, suffix F
float  pi_f = 3.14159265f;
Console.WriteLine($"float:   {pi_f}");  // 3.1415927 (limited precision)

// double: 64-bit, ~15-17 significant digits (default for literals)
double pi_d = 3.141592653589793;
Console.WriteLine($"double:  {pi_d}");  // 3.141592653589793

// decimal: 128-bit, 28-29 significant digits, suffix M
// Best for financial calculations — no binary floating-point rounding
decimal price = 19.99m;
decimal tax   = 0.08m;
decimal total = price + (price * tax);
Console.WriteLine($"decimal: {total}");  // 21.5892 (exact)
```

### 3.1 Floating-Point Precision Pitfalls

```csharp
// The classic floating-point surprise
double a = 0.1 + 0.2;
Console.WriteLine(a == 0.3);         // False!
Console.WriteLine($"{a:R}");         // 0.30000000000000004

// decimal does not have this problem for base-10 fractions
decimal b = 0.1m + 0.2m;
Console.WriteLine(b == 0.3m);        // True
Console.WriteLine(b);                // 0.3
```

### 3.2 Special Floating-Point Values

```csharp
double posInf = double.PositiveInfinity;
double negInf = double.NegativeInfinity;
double nan    = double.NaN;

Console.WriteLine(1.0 / 0.0);                    // Infinity
Console.WriteLine(-1.0 / 0.0);                   // -Infinity
Console.WriteLine(0.0 / 0.0);                    // NaN
Console.WriteLine(double.IsNaN(nan));             // True
Console.WriteLine(double.IsInfinity(posInf));     // True

// NaN is not equal to anything, including itself
Console.WriteLine(nan == nan);                    // False
Console.WriteLine(double.NaN == double.NaN);      // False
```

## 4. Boolean and Character Types

### 4.1 bool

The `bool` type holds either `true` or `false`. Unlike C or C++, integers cannot be implicitly converted to `bool`:

```csharp
bool isReady = true;
bool isComplete = false;

// Booleans from expressions
bool isAdult = 21 >= 18;        // true
bool isEqual = (3 + 4) == 7;    // true

// Cannot do this (unlike C/C++):
// if (1) { }  // Compile error: Cannot implicitly convert type 'int' to 'bool'

// Must be explicit:
if (isReady)
{
    Console.WriteLine("Ready!");
}
```

### 4.2 char

The `char` type represents a single Unicode character (UTF-16 code unit, 16 bits):

```csharp
char letter = 'A';
char digit  = '7';
char symbol = '€';
char korean = '한';
char emoji  = '☺';  // Basic emoji (BMP)

// Character escape sequences
char newline  = '\n';   // Line feed
char tab      = '\t';   // Tab
char backslash = '\\';  // Backslash
char quote    = '\'';   // Single quote
char nullChar = '\0';   // Null character

// Unicode escape
char omega = '\u03A9';  // Ω
Console.WriteLine(omega); // Ω

// char is actually a 16-bit unsigned integer
Console.WriteLine((int)letter);    // 65
Console.WriteLine((char)65);       // A
Console.WriteLine((int)'0');       // 48

// Character methods
Console.WriteLine(char.IsLetter('A'));     // True
Console.WriteLine(char.IsDigit('7'));      // True
Console.WriteLine(char.IsUpper('A'));      // True
Console.WriteLine(char.ToLower('A'));      // a
Console.WriteLine(char.IsWhiteSpace(' ')); // True
```

## 5. Reference Types: string, object, dynamic

### 5.1 string

`string` is a reference type, but it behaves much like a value type because strings are **immutable** — once created, a string cannot be changed:

```csharp
string greeting = "Hello";
string name = "World";
string message = greeting + ", " + name + "!";
Console.WriteLine(message);  // Hello, World!

// Strings are immutable — "changing" a string creates a new one
string original = "Hello";
string modified = original.Replace("H", "J");
Console.WriteLine(original);  // Hello (unchanged)
Console.WriteLine(modified);  // Jello (new string)

// String comparison
string a = "hello";
string b = "Hello";
Console.WriteLine(a == b);                                        // False (case-sensitive)
Console.WriteLine(a.Equals(b, StringComparison.OrdinalIgnoreCase)); // True

// Common string methods
string text = "  Hello, World!  ";
Console.WriteLine(text.Trim());              // "Hello, World!"
Console.WriteLine(text.ToUpper());           // "  HELLO, WORLD!  "
Console.WriteLine(text.Contains("World"));   // True
Console.WriteLine(text.IndexOf("World"));    // 9
Console.WriteLine(text.Substring(8, 5));     // "orld!"
Console.WriteLine(text.Length);              // 17
```

### 5.2 object

`object` is the base type of all types in C#. Every type (both value and reference) ultimately inherits from `object`:

```csharp
object obj1 = 42;           // int boxed into object
object obj2 = "Hello";      // string assigned to object
object obj3 = 3.14;         // double boxed into object
object obj4 = true;         // bool boxed into object

Console.WriteLine(obj1.GetType());  // System.Int32
Console.WriteLine(obj2.GetType());  // System.String

// Unboxing — extract the value (must cast to correct type)
int number = (int)obj1;
Console.WriteLine(number);  // 42

// ToString() — every object has this method
Console.WriteLine(obj1.ToString());  // "42"
Console.WriteLine(obj3.ToString());  // "3.14"
```

### 5.3 dynamic

The `dynamic` type bypasses compile-time type checking. The type is resolved at runtime:

```csharp
dynamic value = 42;
Console.WriteLine(value.GetType());  // System.Int32

value = "Now I'm a string";
Console.WriteLine(value.GetType());  // System.String

value = new[] { 1, 2, 3 };
Console.WriteLine(value.Length);     // 3

// This compiles but throws RuntimeBinderException at runtime:
// dynamic d = "hello";
// int x = d * 2;  // Runtime error: cannot apply * to string and int
```

Use `dynamic` sparingly — it sacrifices compile-time safety. It is primarily used for interoperability (COM, dynamic languages, reflection).

## 6. var and Type Inference

The `var` keyword lets the compiler infer the type from the right-hand side of an assignment:

```csharp
var count = 42;              // int (inferred)
var pi = 3.14159;            // double (inferred)
var name = "Alice";          // string (inferred)
var items = new List<int>(); // List<int> (inferred)
var flag = true;             // bool (inferred)

// var is still statically typed — you cannot change the type
// count = "hello";  // Compile error: Cannot implicitly convert string to int

// var requires initialization
// var x;  // Compile error: Implicitly-typed variables must be initialized
```

### 6.1 When to Use var

```csharp
// USE var when the type is obvious from the right side
var names = new List<string>();                     // Obviously List<string>
var lookup = new Dictionary<string, List<int>>();    // Avoids repeating long type
var stream = File.OpenRead("data.txt");              // Obviously FileStream

// AVOID var when the type is not obvious
var result = Calculate(x, y);  // What type is result? Not clear.
double result = Calculate(x, y);  // Better — type is explicit

// var is required for anonymous types
var anon = new { Name = "Alice", Age = 30 };
Console.WriteLine($"{anon.Name} is {anon.Age}");
```

## 7. const and readonly

### 7.1 const — Compile-Time Constants

`const` declares a value that must be known at compile time and can never change:

```csharp
const double Pi = 3.14159265358979;
const int MaxRetries = 3;
const string AppName = "MyApp";

// const values are baked into the compiled code
// They cannot be modified:
// Pi = 3.0;  // Compile error

// const can only be used with primitive types, string, and null
// const DateTime now = DateTime.Now;  // Compile error: not a constant expression

// Constants in a class
class Config
{
    public const int MaxConnections = 100;
    public const string Version = "1.0.0";
    public const double Gravity = 9.81;
}

// Accessed without an instance (they are implicitly static)
Console.WriteLine(Config.MaxConnections);  // 100
```

### 7.2 readonly — Runtime Constants

For values that should be set once but cannot be determined at compile time, use `readonly` (covered more in the Classes lesson, but here is a preview):

```csharp
class AppSettings
{
    public readonly DateTime StartTime;
    public readonly string MachineName;

    public AppSettings()
    {
        StartTime = DateTime.Now;          // Set at runtime
        MachineName = Environment.MachineName;
    }
}
```

### 7.3 const vs readonly

```csharp
// const: compile-time, implicitly static, limited types
// readonly: runtime, per-instance (or static), any type

class Example
{
    public const int CompileTimeValue = 42;           // Must know at compile time
    public readonly int RuntimeValue;                  // Can set in constructor
    public static readonly int SharedValue = GetValue(); // Computed once at startup

    public Example(int value)
    {
        RuntimeValue = value;
    }

    private static int GetValue() => Environment.ProcessorCount;
}
```

## 8. Nullable Value Types

Value types normally cannot be `null`. Adding `?` makes them nullable:

```csharp
// Regular int cannot be null
int count = 0;
// count = null;  // Compile error

// Nullable int can be null
int? maybeCount = null;
int? definiteCount = 42;

Console.WriteLine(maybeCount.HasValue);     // False
Console.WriteLine(definiteCount.HasValue);  // True
Console.WriteLine(definiteCount.Value);     // 42

// Accessing .Value when null throws InvalidOperationException
// Console.WriteLine(maybeCount.Value);  // Runtime error!

// Safe access with null-coalescing operator
int safeCount = maybeCount ?? 0;  // Use 0 if null
Console.WriteLine(safeCount);     // 0

// GetValueOrDefault
Console.WriteLine(maybeCount.GetValueOrDefault());     // 0
Console.WriteLine(maybeCount.GetValueOrDefault(-1));   // -1
Console.WriteLine(definiteCount.GetValueOrDefault());  // 42

// Nullable in practice: database results, optional configuration
double? temperature = ReadSensor();   // might return null if sensor fails
bool? userConsent = null;             // tri-state: true, false, or unknown
```

### 8.1 Nullable Reference Types

C# 8+ introduced nullable reference types (enabled with `<Nullable>enable</Nullable>` in `.csproj`):

```csharp
// With nullable reference types enabled:
string name = "Alice";   // Non-nullable — compiler warns if you assign null
string? nickname = null;  // Explicitly nullable — null is allowed

// Compiler warnings help prevent NullReferenceException
// string bad = null;  // Warning: Converting null literal or possible null value

// Null-conditional operator
int? length = nickname?.Length;  // null if nickname is null, otherwise Length

// Null-forgiving operator (suppress warning when you know it's not null)
string definitelyNotNull = nickname!;  // Tells compiler "trust me, this isn't null"
```

## 9. Type Conversions

### 9.1 Implicit Conversions (Widening)

Implicit conversions happen automatically when there is no risk of data loss:

```csharp
// Smaller type to larger type — always safe
byte b = 42;
int i = b;       // byte → int (implicit)
long l = i;      // int → long (implicit)
float f = l;     // long → float (implicit, but may lose precision!)
double d = f;    // float → double (implicit)

// int to decimal is implicit (no precision loss)
decimal dec = i;  // int → decimal (implicit)

Console.WriteLine($"byte={b}, int={i}, long={l}, float={f}, double={d}, decimal={dec}");
```

### 9.2 Explicit Conversions (Narrowing / Casting)

When data loss is possible, you must use an explicit cast:

```csharp
double pi = 3.14159;
int truncated = (int)pi;           // 3 (fractional part lost)
Console.WriteLine(truncated);

long big = 3_000_000_000;
int overflow = (int)big;           // -1294967296 (overflow! wraps around)
Console.WriteLine(overflow);

float precise = 1.23456789f;
int rounded = (int)precise;        // 1 (truncated, not rounded)
Console.WriteLine(rounded);

// Casting between numeric types
double temperature = 98.6;
int tempInt = (int)temperature;    // 98
float tempFloat = (float)temperature;  // 98.6
```

### 9.3 The Convert Class

The `Convert` class provides methods that handle rounding and null:

```csharp
// Convert rounds instead of truncating
double value = 3.7;
int converted = Convert.ToInt32(value);  // 4 (banker's rounding)
int casted = (int)value;                 // 3 (truncated)
Console.WriteLine($"Convert: {converted}, Cast: {casted}");

// Converting strings to numbers
string numberStr = "42";
int number = Convert.ToInt32(numberStr);
double dbl = Convert.ToDouble("3.14");
bool flag = Convert.ToBoolean("true");

Console.WriteLine($"int={number}, double={dbl}, bool={flag}");

// Convert handles null gracefully
string? nullStr = null;
int fromNull = Convert.ToInt32(nullStr);  // 0 (not an exception)
```

### 9.4 Parsing Strings

```csharp
// Parse — throws FormatException on invalid input
int parsed = int.Parse("42");
double dParsed = double.Parse("3.14");

// TryParse — returns bool, never throws (preferred)
if (int.TryParse("42", out int result))
{
    Console.WriteLine($"Parsed successfully: {result}");
}

if (int.TryParse("not a number", out int failed))
{
    Console.WriteLine($"Success: {failed}");
}
else
{
    Console.WriteLine("Failed to parse");  // This runs
}

// TryParse with different types
bool validDouble = double.TryParse("3.14", out double dResult);
bool validBool = bool.TryParse("true", out bool bResult);
bool validDate = DateTime.TryParse("2024-01-15", out DateTime dtResult);

Console.WriteLine($"double={dResult}, bool={bResult}, date={dtResult:yyyy-MM-dd}");
```

## 10. Default Values and Initialization

Every type in C# has a default value:

```csharp
// Default values for common types
Console.WriteLine(default(int));       // 0
Console.WriteLine(default(double));    // 0
Console.WriteLine(default(bool));      // False
Console.WriteLine(default(char));      // '\0' (null character)
Console.WriteLine(default(string));    // (null — empty line)
Console.WriteLine(default(int?));      // (null — empty line)

// Using default keyword with var
int x = default;           // 0
string s = default!;       // null (with null-forgiving operator)
bool b = default;          // false

// Fields in a class get default values automatically
class Defaults
{
    public int Number;         // 0
    public double Fraction;    // 0.0
    public bool Flag;          // false
    public string? Text;       // null
    public int? Optional;      // null
}

var d = new Defaults();
Console.WriteLine($"Number={d.Number}, Flag={d.Flag}, Text={d.Text ?? "null"}");
// Output: Number=0, Flag=False, Text=null
```

## 11. String Literals In Depth

C# provides several ways to write string literals:

```csharp
// 1. Regular string literal
string regular = "Hello\nWorld";  // \n is a newline

// 2. Verbatim string literal — @ prefix
// Escape sequences are NOT processed (except "" for a quote)
string path = @"C:\Users\Alice\Documents";    // No need to escape backslashes
string multiLine = @"Line 1
Line 2
Line 3";
string withQuote = @"She said ""hello""";

// 3. String interpolation — $ prefix
string name = "Alice";
int age = 30;
string intro = $"My name is {name} and I am {age} years old.";
Console.WriteLine(intro);

// 4. Combining verbatim + interpolation — $@ or @$
string filePath = $@"C:\Users\{name}\Documents";
Console.WriteLine(filePath);  // C:\Users\Alice\Documents

// 5. Raw string literals (C# 11+) — triple (or more) quotes
string rawJson = """
    {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "coding"]
    }
    """;
Console.WriteLine(rawJson);

// 6. Raw interpolated string
int count = 5;
string rawInterpolated = $"""
    The count is {count}.
    To use braces in output: {{like this}}.
    """;
Console.WriteLine(rawInterpolated);

// 7. Interpolation with format specifiers
double price = 49.99;
DateTime now = DateTime.Now;
Console.WriteLine($"Price: {price:C}");           // Price: $49.99 (currency)
Console.WriteLine($"Price: {price:F4}");           // Price: 49.9900 (4 decimals)
Console.WriteLine($"Date: {now:yyyy-MM-dd}");      // Date: 2024-01-15
Console.WriteLine($"Hex: {255:X}");                // Hex: FF
Console.WriteLine($"Padded: {42,10}");             // Padded:         42 (right-aligned)
Console.WriteLine($"Padded: {42,-10}|");           // Padded: 42        | (left-aligned)
```

## 12. Practice Problems

1. **Type Size Table**: Write a program that prints a table showing each numeric type's name, size in bytes (using `sizeof` for value types), minimum value, and maximum value. Include `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`, `double`, and `decimal`.

2. **Floating-Point Comparison**: Write a program that demonstrates the floating-point imprecision of `double` (e.g., `0.1 + 0.2 != 0.3`). Then show the same calculation using `decimal` and verify it produces the expected result. Include at least three different examples of floating-point surprises.

3. **Type Conversion Explorer**: Write a program that takes a string input from the user (via `Console.ReadLine()`), then attempts to parse it as `int`, `double`, `bool`, and `DateTime` using `TryParse`. Print which conversions succeeded and the resulting values.

4. **String Manipulation**: Write a program that takes a full name as input (e.g., "John Michael Smith") and prints: (a) the number of characters, (b) the name in uppercase, (c) the name reversed, (d) each word capitalized separately, and (e) the initials (e.g., "JMS"). Use only `string` methods — no `StringBuilder` yet.

5. **Nullable Calculator**: Write a calculator that uses `int?` for its operands. If the user enters an empty string for either operand, treat it as `null`. Handle null propagation: if either operand is `null`, the result should be `null`. Print "Result: N/A" for null results and the actual value otherwise.

---

**Previous**: [Getting Started](./01_Getting_Started.md) | **Next**: [Operators and Expressions](./03_Operators_and_Expressions.md)
