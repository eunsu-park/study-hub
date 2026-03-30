// Lesson 03: Operators and Expressions
// Run: dotnet run

using System;

// =============================================================================
// ARITHMETIC OPERATORS
// =============================================================================
Console.WriteLine("=== Arithmetic Operators ===");

int a = 17, b = 5;
Console.WriteLine($"a = {a}, b = {b}");
Console.WriteLine($"a + b = {a + b}");   // Addition: 22
Console.WriteLine($"a - b = {a - b}");   // Subtraction: 12
Console.WriteLine($"a * b = {a * b}");   // Multiplication: 85
Console.WriteLine($"a / b = {a / b}");   // Integer division: 3 (truncated)
Console.WriteLine($"a % b = {a % b}");   // Modulus (remainder): 2

// Division with doubles gives fractional result
double da = 17.0, db = 5.0;
Console.WriteLine($"\n17.0 / 5.0 = {da / db}");  // 3.4

// Increment and decrement
int x = 10;
Console.WriteLine($"\nx = {x}");
Console.WriteLine($"x++ (post): {x++}, now x = {x}"); // Returns 10, then x becomes 11
Console.WriteLine($"++x (pre):  {++x}, now x = {x}"); // x becomes 12, returns 12
Console.WriteLine($"x-- (post): {x--}, now x = {x}"); // Returns 12, then x becomes 11
Console.WriteLine($"--x (pre):  {--x}, now x = {x}"); // x becomes 10, returns 10

// =============================================================================
// COMPARISON OPERATORS
// =============================================================================
Console.WriteLine("\n=== Comparison Operators ===");

int p = 10, q = 20;
Console.WriteLine($"p = {p}, q = {q}");
Console.WriteLine($"p == q: {p == q}");  // false
Console.WriteLine($"p != q: {p != q}");  // true
Console.WriteLine($"p < q:  {p < q}");   // true
Console.WriteLine($"p > q:  {p > q}");   // false
Console.WriteLine($"p <= q: {p <= q}");  // true
Console.WriteLine($"p >= q: {p >= q}");  // false

// Reference equality vs value equality for strings
string s1 = "hello";
string s2 = "hello";
string s3 = new string("hello");
Console.WriteLine($"\nString equality:");
Console.WriteLine($"s1 == s2: {s1 == s2}");                       // true (value comparison)
Console.WriteLine($"ReferenceEquals(s1, s2): {ReferenceEquals(s1, s2)}"); // true (interned)
Console.WriteLine($"s1 == s3: {s1 == s3}");                       // true (value comparison)
Console.WriteLine($"ReferenceEquals(s1, s3): {ReferenceEquals(s1, s3)}"); // may be false

// =============================================================================
// LOGICAL OPERATORS
// =============================================================================
Console.WriteLine("\n=== Logical Operators ===");

bool t = true, f = false;
Console.WriteLine($"true && false: {t && f}");  // AND: false
Console.WriteLine($"true || false: {t || f}");  // OR:  true
Console.WriteLine($"!true:         {!t}");       // NOT: false

// Short-circuit evaluation
int divisor = 0;
bool safe = divisor != 0 && (100 / divisor > 5);
Console.WriteLine($"\nShort-circuit: divisor=0, safe={safe}");
// The division is never evaluated because divisor != 0 is false

// =============================================================================
// BITWISE OPERATORS
// =============================================================================
Console.WriteLine("\n=== Bitwise Operators ===");

int m = 0b_1100;  // 12
int n = 0b_1010;  // 10
Console.WriteLine($"m = {m} (0b{Convert.ToString(m, 2).PadLeft(4, '0')})");
Console.WriteLine($"n = {n} (0b{Convert.ToString(n, 2).PadLeft(4, '0')})");
Console.WriteLine($"m & n  = {m & n}  (AND:  0b{Convert.ToString(m & n, 2).PadLeft(4, '0')})");
Console.WriteLine($"m | n  = {m | n} (OR:   0b{Convert.ToString(m | n, 2).PadLeft(4, '0')})");
Console.WriteLine($"m ^ n  = {m ^ n}  (XOR:  0b{Convert.ToString(m ^ n, 2).PadLeft(4, '0')})");
Console.WriteLine($"~m     = {~m} (NOT)");
Console.WriteLine($"m << 1 = {m << 1} (Left shift)");
Console.WriteLine($"m >> 1 = {m >> 1}  (Right shift)");

// =============================================================================
// NULL-COALESCING AND NULL-CONDITIONAL OPERATORS
// =============================================================================
Console.WriteLine("\n=== Null-Coalescing Operators ===");

// ?? — returns left operand if non-null, otherwise right operand
string? input = null;
string displayName = input ?? "Anonymous";
Console.WriteLine($"input ?? \"Anonymous\": {displayName}");

// ??= — assigns right operand only if left is null
input ??= "Default User";
Console.WriteLine($"input ??= \"Default User\": {input}");

// ?. — null-conditional (safe navigation)
string? text = null;
int? length = text?.Length;  // null, not NullReferenceException
Console.WriteLine($"null?.Length: {length?.ToString() ?? "(null)"}");

text = "Hello";
length = text?.Length;
Console.WriteLine($"\"Hello\"?.Length: {length}");

// Chaining null-conditional
int?[] nullableArray = null;
int? firstElement = nullableArray?[0];
Console.WriteLine($"nullableArray?[0]: {firstElement?.ToString() ?? "(null)"}");

// =============================================================================
// TERNARY (CONDITIONAL) OPERATOR
// =============================================================================
Console.WriteLine("\n=== Ternary Operator ===");

int age = 20;
string status = age >= 18 ? "Adult" : "Minor";
Console.WriteLine($"Age {age}: {status}");

// Nested ternary (use sparingly — prefer if/else for readability)
int score = 85;
string grade = score >= 90 ? "A"
             : score >= 80 ? "B"
             : score >= 70 ? "C"
             : "F";
Console.WriteLine($"Score {score}: Grade {grade}");

// =============================================================================
// CHECKED AND UNCHECKED ARITHMETIC
// =============================================================================
Console.WriteLine("\n=== Checked / Unchecked Arithmetic ===");

// Unchecked (default): overflow wraps around silently
int maxInt = int.MaxValue;
unchecked
{
    int overflow = maxInt + 1;
    Console.WriteLine($"Unchecked: {maxInt} + 1 = {overflow}");  // Wraps to MinValue
}

// Checked: overflow throws OverflowException
try
{
    int overflow = checked(maxInt + 1);
    Console.WriteLine($"Checked: {overflow}");
}
catch (OverflowException ex)
{
    Console.WriteLine($"Checked: OverflowException — {ex.Message}");
}

// =============================================================================
// COMPOUND ASSIGNMENT OPERATORS
// =============================================================================
Console.WriteLine("\n=== Compound Assignment ===");

int v = 100;
Console.WriteLine($"v = {v}");
v += 10; Console.WriteLine($"v += 10 -> {v}");
v -= 5;  Console.WriteLine($"v -= 5  -> {v}");
v *= 2;  Console.WriteLine($"v *= 2  -> {v}");
v /= 3;  Console.WriteLine($"v /= 3  -> {v}");
v %= 13; Console.WriteLine($"v %%= 13 -> {v}");

// =============================================================================
// TYPEOF, SIZEOF, NAMEOF
// =============================================================================
Console.WriteLine("\n=== typeof, sizeof, nameof ===");

Console.WriteLine($"typeof(int):    {typeof(int)}");
Console.WriteLine($"typeof(string): {typeof(string)}");
Console.WriteLine($"sizeof(int):    {sizeof(int)} bytes");
Console.WriteLine($"sizeof(double): {sizeof(double)} bytes");
Console.WriteLine($"sizeof(char):   {sizeof(char)} bytes");

string myVariable = "test";
Console.WriteLine($"nameof(myVariable): {nameof(myVariable)}");
Console.WriteLine($"nameof(Console):    {nameof(Console)}");
