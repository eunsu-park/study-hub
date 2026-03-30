// Lesson 02: Variables and Types
// Run: dotnet run

using System;

// =============================================================================
// VALUE TYPES — stored on the stack, hold data directly
// =============================================================================
Console.WriteLine("=== Value Types ===");

// Integer types
sbyte  sb = -128;               // 8-bit signed   (-128 to 127)
byte   b  = 255;                // 8-bit unsigned  (0 to 255)
short  s  = -32_768;            // 16-bit signed
ushort us = 65_535;             // 16-bit unsigned
int    i  = 2_147_483_647;     // 32-bit signed (most common)
uint   ui = 4_294_967_295u;    // 32-bit unsigned
long   l  = 9_223_372_036_854_775_807L;  // 64-bit signed
ulong  ul = 18_446_744_073_709_551_615UL; // 64-bit unsigned

Console.WriteLine($"int range: {int.MinValue} to {int.MaxValue}");
Console.WriteLine($"long range: {long.MinValue} to {long.MaxValue}");

// Floating-point types
float  f = 3.14f;      // 32-bit, ~6-7 digits precision
double d = 3.14159265; // 64-bit, ~15-16 digits precision (default)
decimal m = 3.1415926535897932m; // 128-bit, ~28-29 digits (financial)

Console.WriteLine($"\nfloat:   {f} (precision ~7 digits)");
Console.WriteLine($"double:  {d} (precision ~15 digits)");
Console.WriteLine($"decimal: {m} (precision ~28 digits)");

// Boolean and char
bool isActive = true;
char letter = 'A';
char unicode = '\u0041'; // Also 'A'
Console.WriteLine($"\nbool: {isActive}, char: {letter}, unicode: {unicode}");

// =============================================================================
// REFERENCE TYPES — stored on the heap, variables hold references
// =============================================================================
Console.WriteLine("\n=== Reference Types ===");

string greeting = "Hello, C#!";
object obj = 42;                // Boxing: value type wrapped in object
dynamic dyn = "I can change";  // Type resolved at runtime
dyn = 42;                      // No compile error with dynamic

Console.WriteLine($"string: {greeting}");
Console.WriteLine($"object: {obj} (type: {obj.GetType()})");
Console.WriteLine($"dynamic: {dyn} (type: {dyn.GetType()})");

// =============================================================================
// VAR — implicit typing (compiler infers the type)
// =============================================================================
Console.WriteLine("\n=== Implicit Typing (var) ===");

var count = 10;            // Compiler infers int
var message = "Hello";     // Compiler infers string
var ratio = 3.14;          // Compiler infers double
var items = new int[] { 1, 2, 3 }; // Compiler infers int[]

Console.WriteLine($"var count: {count} (type: {count.GetType().Name})");
Console.WriteLine($"var message: {message} (type: {message.GetType().Name})");
Console.WriteLine($"var ratio: {ratio} (type: {ratio.GetType().Name})");
// Note: var requires initialization; you cannot write: var x;

// =============================================================================
// CONST and READONLY
// =============================================================================
Console.WriteLine("\n=== Constants ===");

const double Pi = 3.14159265358979;
const string AppName = "CSharp Basics";
// const values must be known at compile time
Console.WriteLine($"Pi = {Pi}");
Console.WriteLine($"App = {AppName}");

// readonly is for fields in classes (shown later); const is for local/static constants

// =============================================================================
// NULLABLE VALUE TYPES
// =============================================================================
Console.WriteLine("\n=== Nullable Types ===");

int? nullableInt = null;       // Nullable<int>
double? nullableDouble = 3.14;

Console.WriteLine($"nullableInt has value: {nullableInt.HasValue}");
Console.WriteLine($"nullableDouble value: {nullableDouble.Value}");

// GetValueOrDefault provides a fallback
int result = nullableInt.GetValueOrDefault(0);
Console.WriteLine($"GetValueOrDefault: {result}");

// Nullable reference types (C# 8+ with #nullable enable)
string? nullableName = null;  // Allowed: explicitly nullable reference
string nonNullName = "Alice"; // Should not be null (compiler warning if assigned null)
Console.WriteLine($"nullableName is null: {nullableName is null}");
Console.WriteLine($"nonNullName: {nonNullName}");

// =============================================================================
// TYPE CONVERSIONS
// =============================================================================
Console.WriteLine("\n=== Type Conversions ===");

// Implicit conversion (widening — no data loss)
int intVal = 42;
long longVal = intVal;     // int -> long: safe
double doubleVal = intVal; // int -> double: safe
Console.WriteLine($"Implicit: int {intVal} -> long {longVal} -> double {doubleVal}");

// Explicit conversion (narrowing — potential data loss)
double bigDouble = 9999.99;
int truncated = (int)bigDouble;  // Cast: truncates decimal part
Console.WriteLine($"Explicit cast: {bigDouble} -> {truncated}");

// Convert class
string numberStr = "123";
int parsed = Convert.ToInt32(numberStr);
bool flag = Convert.ToBoolean(1);
Console.WriteLine($"Convert: \"{numberStr}\" -> {parsed}, 1 -> {flag}");

// Parse and TryParse
int.TryParse("456", out int parsedValue);
Console.WriteLine($"TryParse: \"456\" -> {parsedValue}");

bool success = int.TryParse("not_a_number", out int failed);
Console.WriteLine($"TryParse: \"not_a_number\" -> success={success}, value={failed}");

// =============================================================================
// STRINGS
// =============================================================================
Console.WriteLine("\n=== String Features ===");

// String interpolation
string name = "Alice";
int age = 30;
Console.WriteLine($"{name} is {age} years old.");

// Verbatim strings (no escape processing)
string path = @"C:\Users\Alice\Documents";
Console.WriteLine($"Path: {path}");

// Raw string literals (C# 11+)
string json = """
    {
        "name": "Alice",
        "age": 30
    }
    """;
Console.WriteLine($"Raw string literal:\n{json}");

// String methods
string text = "  Hello, World!  ";
Console.WriteLine($"Trim: '{text.Trim()}'");
Console.WriteLine($"Upper: '{text.Trim().ToUpper()}'");
Console.WriteLine($"Contains 'World': {text.Contains("World")}");
Console.WriteLine($"Substring(8, 5): '{text.Trim().Substring(7, 5)}'");
Console.WriteLine($"Replace: '{text.Trim().Replace("World", "C#")}'");

// String comparison
Console.WriteLine($"Equals (ordinal): {"hello".Equals("Hello", StringComparison.OrdinalIgnoreCase)}");

// =============================================================================
// DEFAULT VALUES
// =============================================================================
Console.WriteLine("\n=== Default Values ===");
Console.WriteLine($"default(int):    {default(int)}");
Console.WriteLine($"default(bool):   {default(bool)}");
Console.WriteLine($"default(double): {default(double)}");
Console.WriteLine($"default(string): {default(string) ?? "(null)"}");
Console.WriteLine($"default(char):   '{default(char)}' (\\0)");
