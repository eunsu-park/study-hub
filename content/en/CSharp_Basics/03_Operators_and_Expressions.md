# Operators and Expressions

**Previous**: [Variables and Types](./02_Variables_and_Types.md) | **Next**: [Control Flow](./04_Control_Flow.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use arithmetic operators for mathematical calculations
2. Compare values with relational operators
3. Combine conditions with logical operators
4. Manipulate individual bits with bitwise operators
5. Simplify assignments with compound assignment operators
6. Handle null values safely with null-coalescing and null-conditional operators
7. Write concise conditional expressions with the ternary operator
8. Apply correct operator precedence in complex expressions
9. Detect and prevent arithmetic overflow with checked/unchecked contexts

---

Operators are symbols that tell the compiler to perform specific operations on one, two, or three operands. An **expression** is a combination of operands and operators that evaluates to a single value. C# has a rich set of operators that you will use in virtually every program you write.

## 1. Arithmetic Operators

Arithmetic operators perform mathematical calculations on numeric operands.

### 1.1 Basic Arithmetic

```csharp
int a = 17;
int b = 5;

Console.WriteLine($"a + b = {a + b}");   // 22  (addition)
Console.WriteLine($"a - b = {a - b}");   // 12  (subtraction)
Console.WriteLine($"a * b = {a * b}");   // 85  (multiplication)
Console.WriteLine($"a / b = {a / b}");   // 3   (integer division — truncated)
Console.WriteLine($"a % b = {a % b}");   // 2   (modulus — remainder)
```

### 1.2 Integer Division vs Floating-Point Division

A critical distinction: when both operands are integers, `/` performs integer division (truncates the fractional part). When at least one operand is a floating-point type, true division occurs:

```csharp
// Integer division — fractional part is discarded
Console.WriteLine(7 / 2);       // 3 (not 3.5)
Console.WriteLine(-7 / 2);      // -3 (truncates toward zero)

// Floating-point division — preserves fractional part
Console.WriteLine(7.0 / 2);     // 3.5
Console.WriteLine(7 / 2.0);     // 3.5
Console.WriteLine((double)7 / 2); // 3.5 (cast one operand)

// Division by zero
// Console.WriteLine(7 / 0);    // Compile error (integer division by constant 0)
Console.WriteLine(7.0 / 0);     // Infinity
Console.WriteLine(-7.0 / 0);    // -Infinity
Console.WriteLine(0.0 / 0);     // NaN
```

### 1.3 The Modulus Operator

The modulus (remainder) operator `%` is useful for many common tasks:

```csharp
// Check if a number is even or odd
for (int i = 1; i <= 10; i++)
{
    string parity = (i % 2 == 0) ? "even" : "odd";
    Console.WriteLine($"{i} is {parity}");
}

// Wrap around (circular indexing)
string[] days = { "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun" };
for (int i = 0; i < 14; i++)
{
    Console.WriteLine($"Day {i}: {days[i % 7]}");
}

// Extract digits
int number = 12345;
while (number > 0)
{
    int digit = number % 10;
    Console.Write($"{digit} ");  // 5 4 3 2 1
    number /= 10;
}
Console.WriteLine();
```

### 1.4 Increment and Decrement

```csharp
int x = 5;

// Prefix: increment/decrement BEFORE using the value
Console.WriteLine(++x);  // 6 (x is now 6)
Console.WriteLine(--x);  // 5 (x is now 5)

// Postfix: use the value, THEN increment/decrement
Console.WriteLine(x++);  // 5 (prints 5, then x becomes 6)
Console.WriteLine(x--);  // 6 (prints 6, then x becomes 5)
Console.WriteLine(x);    // 5

// Common usage in loops
for (int i = 0; i < 5; i++)  // i++ is postfix, but result is not used
{
    Console.Write($"{i} ");  // 0 1 2 3 4
}
Console.WriteLine();
```

### 1.5 Unary Plus and Minus

```csharp
int positive = +5;   // 5 (unary plus — rarely used)
int negative = -5;   // -5 (unary minus — negation)
int negated = -negative;  // 5 (double negation)
Console.WriteLine($"{positive}, {negative}, {negated}");
```

## 2. Comparison (Relational) Operators

Comparison operators evaluate to a `bool` value (`true` or `false`):

```csharp
int x = 10;
int y = 20;

Console.WriteLine($"x == y: {x == y}");   // False (equal to)
Console.WriteLine($"x != y: {x != y}");   // True  (not equal to)
Console.WriteLine($"x < y:  {x < y}");    // True  (less than)
Console.WriteLine($"x > y:  {x > y}");    // False (greater than)
Console.WriteLine($"x <= y: {x <= y}");   // True  (less than or equal)
Console.WriteLine($"x >= y: {x >= y}");   // False (greater than or equal)
```

### 2.1 Comparing Different Types

```csharp
// Numeric comparisons work across types (implicit conversion)
Console.WriteLine(42 == 42.0);       // True (int compared to double)
Console.WriteLine(42 == 42L);        // True (int compared to long)

// String comparison
string a = "hello";
string b = "hello";
string c = "Hello";
Console.WriteLine(a == b);           // True (value equality for strings)
Console.WriteLine(a == c);           // False (case-sensitive)

// Object reference comparison
object obj1 = new object();
object obj2 = new object();
object obj3 = obj1;
Console.WriteLine(obj1 == obj2);     // False (different objects)
Console.WriteLine(obj1 == obj3);     // True (same reference)

// Null comparison
string? name = null;
Console.WriteLine(name == null);     // True
Console.WriteLine(name is null);     // True (pattern matching — preferred)
```

### 2.2 Chained Comparisons

Unlike Python, C# does not support chained comparisons like `1 < x < 10`. You must use logical operators:

```csharp
int value = 5;

// This does NOT work as expected:
// bool inRange = 1 < value < 10;  // Compile error

// Use logical AND instead:
bool inRange = value > 1 && value < 10;
Console.WriteLine($"{value} in range (1, 10): {inRange}");  // True
```

## 3. Logical Operators

Logical operators combine or modify boolean expressions:

```csharp
bool a = true;
bool b = false;

// Logical AND — true only if BOTH are true
Console.WriteLine($"a && b: {a && b}");   // False
Console.WriteLine($"a && a: {a && a}");   // True

// Logical OR — true if EITHER is true
Console.WriteLine($"a || b: {a || b}");   // True
Console.WriteLine($"b || b: {b || b}");   // False

// Logical NOT — inverts the value
Console.WriteLine($"!a: {!a}");           // False
Console.WriteLine($"!b: {!b}");           // True

// Logical XOR — true if exactly one is true
Console.WriteLine($"a ^ b: {a ^ b}");    // True
Console.WriteLine($"a ^ a: {a ^ a}");    // False
```

### 3.1 Short-Circuit Evaluation

`&&` and `||` use **short-circuit evaluation**: the right operand is not evaluated if the left operand already determines the result.

```csharp
// Short-circuit AND: if left is false, right is not evaluated
int[] numbers = null!;
// Without short-circuit, this would throw NullReferenceException:
if (numbers != null && numbers.Length > 0)
{
    Console.WriteLine($"First: {numbers[0]}");
}

// Short-circuit OR: if left is true, right is not evaluated
string? input = null;
string result = input ?? "default";  // Not exactly ||, but similar concept

// Demonstrating short-circuit
int counter = 0;
bool Increment() { counter++; return true; }

bool test1 = false && Increment();   // Increment() is NOT called
Console.WriteLine($"counter = {counter}");  // 0

bool test2 = true || Increment();    // Increment() is NOT called
Console.WriteLine($"counter = {counter}");  // 0

bool test3 = true && Increment();    // Increment() IS called
Console.WriteLine($"counter = {counter}");  // 1
```

### 3.2 Non-Short-Circuit Operators

The `&` and `|` operators (without doubling) always evaluate both sides:

```csharp
int x = 0;
bool alwaysEval = false & (++x > 0);  // x IS incremented even though left is false
Console.WriteLine(x);  // 1

// Use & and | when you need side effects from both operands
// (rare in practice — prefer && and ||)
```

## 4. Bitwise Operators

Bitwise operators work on the individual bits of integer types:

```csharp
int a = 0b_1100;  // 12 in decimal
int b = 0b_1010;  // 10 in decimal

// AND — 1 only if both bits are 1
Console.WriteLine($"a & b  = {a & b}");    // 8  (0b_1000)

// OR — 1 if either bit is 1
Console.WriteLine($"a | b  = {a | b}");    // 14 (0b_1110)

// XOR — 1 if bits are different
Console.WriteLine($"a ^ b  = {a ^ b}");    // 6  (0b_0110)

// NOT — inverts all bits
Console.WriteLine($"~a     = {~a}");       // -13 (flips all 32 bits)

// Left shift — multiply by 2^n
Console.WriteLine($"a << 1 = {a << 1}");   // 24 (0b_11000)
Console.WriteLine($"a << 2 = {a << 2}");   // 48 (0b_110000)

// Right shift — divide by 2^n
Console.WriteLine($"a >> 1 = {a >> 1}");   // 6  (0b_0110)
Console.WriteLine($"a >> 2 = {a >> 2}");   // 3  (0b_0011)

// Unsigned right shift (C# 11+) — always fills with zeros
Console.WriteLine($"a >>> 1 = {a >>> 1}"); // 6
```

### 4.1 Common Bit Manipulation Patterns

```csharp
// Check if a number is even (last bit is 0)
int num = 42;
bool isEven = (num & 1) == 0;
Console.WriteLine($"{num} is even: {isEven}");  // True

// Set a specific bit
int flags = 0;
flags |= (1 << 3);  // Set bit 3
Console.WriteLine($"After setting bit 3: {flags}");  // 8 (0b_1000)

// Clear a specific bit
flags &= ~(1 << 3);  // Clear bit 3
Console.WriteLine($"After clearing bit 3: {flags}");  // 0

// Toggle a specific bit
flags = 0b_1010;
flags ^= (1 << 1);  // Toggle bit 1
Console.WriteLine($"After toggling bit 1: {Convert.ToString(flags, 2)}");  // 1000

// Check if a specific bit is set
int value = 0b_1010;
bool bit1Set = (value & (1 << 1)) != 0;
bool bit2Set = (value & (1 << 2)) != 0;
Console.WriteLine($"Bit 1 set: {bit1Set}");  // True
Console.WriteLine($"Bit 2 set: {bit2Set}");  // False

// Count set bits (population count)
int PopCount(int n)
{
    int count = 0;
    while (n != 0)
    {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
Console.WriteLine($"PopCount(0b_1011) = {PopCount(0b_1011)}");  // 3
```

### 4.2 Practical Example: RGB Color Manipulation

```csharp
// Pack RGB values into a single int (0xAARRGGBB format)
int PackColor(byte r, byte g, byte b, byte a = 255)
{
    return (a << 24) | (r << 16) | (g << 8) | b;
}

// Unpack individual channels
byte GetRed(int color)   => (byte)((color >> 16) & 0xFF);
byte GetGreen(int color) => (byte)((color >> 8) & 0xFF);
byte GetBlue(int color)  => (byte)(color & 0xFF);
byte GetAlpha(int color) => (byte)((color >> 24) & 0xFF);

int purple = PackColor(128, 0, 128);
Console.WriteLine($"Color: 0x{purple:X8}");
Console.WriteLine($"R={GetRed(purple)}, G={GetGreen(purple)}, B={GetBlue(purple)}");
// Output: Color: 0xFF800080
// Output: R=128, G=0, B=128
```

## 5. Assignment Operators

### 5.1 Simple Assignment

```csharp
int x = 10;       // Assign 10 to x
string name = "Alice";
```

### 5.2 Compound Assignment

Compound assignment operators combine an operation with assignment:

```csharp
int x = 10;

x += 5;    // x = x + 5;   → 15
x -= 3;    // x = x - 3;   → 12
x *= 2;    // x = x * 2;   → 24
x /= 4;    // x = x / 4;   → 6
x %= 4;    // x = x % 4;   → 2

// Bitwise compound assignment
int flags = 0;
flags |= 0b_0100;   // Set bits:    flags = 0b_0100
flags &= 0b_1100;   // Clear bits:  flags = 0b_0100
flags ^= 0b_0110;   // Toggle bits: flags = 0b_0010
flags <<= 2;         // Shift left:  flags = 0b_1000
flags >>= 1;         // Shift right: flags = 0b_0100

Console.WriteLine($"flags = {Convert.ToString(flags, 2)}");
```

### 5.3 Null-Coalescing Assignment (??=)

```csharp
string? name = null;

// Assign only if currently null
name ??= "Default Name";
Console.WriteLine(name);  // "Default Name"

// Does not assign because name is no longer null
name ??= "Other Name";
Console.WriteLine(name);  // "Default Name" (unchanged)

// Useful for lazy initialization
List<int>? cache = null;
// ... later in code ...
cache ??= new List<int>();  // Only creates list if cache is null
cache.Add(42);
```

## 6. Null-Coalescing and Null-Conditional Operators

These operators are essential for safe null handling in C#.

### 6.1 Null-Coalescing Operator (??)

Returns the left operand if it is not null; otherwise returns the right operand:

```csharp
string? input = null;
string result = input ?? "default";
Console.WriteLine(result);  // "default"

input = "hello";
result = input ?? "default";
Console.WriteLine(result);  // "hello"

// Chaining multiple fallbacks
string? primary = null;
string? secondary = null;
string? tertiary = "found!";
string value = primary ?? secondary ?? tertiary ?? "last resort";
Console.WriteLine(value);  // "found!"

// With value types
int? maybeNumber = null;
int number = maybeNumber ?? -1;
Console.WriteLine(number);  // -1
```

### 6.2 Null-Conditional Operator (?.)

Accesses a member only if the object is not null; otherwise returns null:

```csharp
string? name = null;

// Without null-conditional — would throw NullReferenceException
// int length = name.Length;

// With null-conditional — returns null instead of throwing
int? length = name?.Length;
Console.WriteLine(length);  // (null)

// Chain multiple null-conditional accesses
string?[] names = { "Alice", null, "Charlie" };
Console.WriteLine(names[1]?.ToUpper()?.Substring(0, 3));  // (null — no exception)
Console.WriteLine(names[0]?.ToUpper()?.Substring(0, 3));  // ALI

// Combine with null-coalescing
string display = name?.ToUpper() ?? "N/A";
Console.WriteLine(display);  // "N/A"

// Null-conditional with indexer
int[]? numbers = null;
int? first = numbers?[0];  // null (no exception)
Console.WriteLine(first);

// Null-conditional with method calls
string? text = "Hello, World";
bool? contains = text?.Contains("World");
Console.WriteLine(contains);  // True
```

## 7. Ternary (Conditional) Operator

The ternary operator `?:` is a concise alternative to simple `if`/`else`:

```csharp
// Syntax: condition ? valueIfTrue : valueIfFalse

int age = 20;
string status = age >= 18 ? "adult" : "minor";
Console.WriteLine(status);  // "adult"

// Equivalent if/else (more verbose):
// string status;
// if (age >= 18) status = "adult";
// else status = "minor";

// Nested ternary (use sparingly — can reduce readability)
int score = 85;
string grade = score >= 90 ? "A"
             : score >= 80 ? "B"
             : score >= 70 ? "C"
             : score >= 60 ? "D"
             : "F";
Console.WriteLine($"Score {score}: Grade {grade}");  // Score 85: Grade B

// In string interpolation
int count = 5;
Console.WriteLine($"You have {count} item{(count != 1 ? "s" : "")}.");
// Output: You have 5 items.

// With method calls
int x = -5;
int absolute = x >= 0 ? x : -x;
Console.WriteLine($"|{x}| = {absolute}");  // |-5| = 5
```

## 8. Operator Precedence

Operators are evaluated in a specific order. Higher precedence means the operator binds more tightly:

```
Precedence   Operators                          Associativity
───────────  ─────────────────────────────────  ─────────────
1 (highest)  x.y  x?.y  x?[i]  f(x)  a[i]    Left to right
             x++  x--  new  typeof  sizeof
2            +x  -x  !x  ~x  ++x  --x         Right to left
             (T)x  await
3            x * y   x / y   x % y             Left to right
4            x + y   x - y                     Left to right
5            x << y  x >> y  x >>> y           Left to right
6            x < y   x > y   x <= y  x >= y   Left to right
             is  as
7            x == y  x != y                    Left to right
8            x & y                             Left to right
9            x ^ y                             Left to right
10           x | y                             Left to right
11           x && y                            Left to right
12           x || y                            Left to right
13           x ?? y                            Left to right
14           c ? t : f                         Right to left
15 (lowest)  =  +=  -=  *=  /=  %=            Right to left
             &=  |=  ^=  <<=  >>=  ??=
```

### 8.1 Precedence Examples

```csharp
// Multiplication before addition
int result1 = 2 + 3 * 4;
Console.WriteLine(result1);  // 14 (not 20)

// Comparison before logical
bool result2 = 5 > 3 && 2 < 4;
Console.WriteLine(result2);  // True (evaluated as (5 > 3) && (2 < 4))

// Parentheses override precedence
int result3 = (2 + 3) * 4;
Console.WriteLine(result3);  // 20

// A common mistake
int a = 5, b = 3;
// bool wrong = a & b == 0;      // Evaluates as a & (b == 0) — probably not intended!
bool correct = (a & b) == 0;     // Correct: apply & first, then compare

// Assignment is right-associative
int x, y, z;
x = y = z = 10;  // z=10, then y=10, then x=10
Console.WriteLine($"x={x}, y={y}, z={z}");  // x=10, y=10, z=10
```

### 8.2 Best Practice: Use Parentheses

When in doubt, add parentheses. They make intent clear and prevent subtle bugs:

```csharp
// Ambiguous without parentheses
int result = a + b * c - d / e;

// Clear with parentheses
int result_clear = a + (b * c) - (d / e);

// Even if you know the precedence, parentheses help readers
bool isValid = (age >= 18) && (age <= 65) && (hasLicense == true);
```

## 9. Checked and Unchecked Arithmetic

By default, integer arithmetic in C# wraps around on overflow without throwing an exception. The `checked` and `unchecked` keywords control this behavior.

### 9.1 Default Behavior (Unchecked)

```csharp
int max = int.MaxValue;  // 2,147,483,647
int overflow = max + 1;
Console.WriteLine(overflow);  // -2,147,483,648 (wrapped around silently!)

byte b = 255;
b++;
Console.WriteLine(b);  // 0 (wrapped around)
```

### 9.2 Checked Context

```csharp
// checked block — throws OverflowException on overflow
try
{
    checked
    {
        int max = int.MaxValue;
        int overflow = max + 1;  // Throws OverflowException!
    }
}
catch (OverflowException ex)
{
    Console.WriteLine($"Overflow detected: {ex.Message}");
}

// checked expression — for a single operation
try
{
    int result = checked(int.MaxValue + 1);
}
catch (OverflowException)
{
    Console.WriteLine("Overflow in single expression!");
}
```

### 9.3 Unchecked Context

```csharp
// Explicitly unchecked (same as default, but makes intent clear)
unchecked
{
    int max = int.MaxValue;
    int overflow = max + 1;
    Console.WriteLine(overflow);  // -2,147,483,648 (no exception)
}

// Useful when overflow is intentional (e.g., hash code computation)
unchecked
{
    int hash = 17;
    hash = hash * 31 + "hello".GetHashCode();
    hash = hash * 31 + 42.GetHashCode();
    Console.WriteLine($"Hash: {hash}");
}
```

### 9.4 Project-Wide Checked Arithmetic

You can enable checked arithmetic for your entire project in the `.csproj`:

```xml
<PropertyGroup>
  <CheckForOverflowUnderflow>true</CheckForOverflowUnderflow>
</PropertyGroup>
```

## 10. Miscellaneous Operators

### 10.1 typeof, sizeof, nameof

```csharp
// typeof — gets the System.Type object
Type intType = typeof(int);
Console.WriteLine(intType.FullName);  // System.Int32

// sizeof — size in bytes (only for unmanaged types in unsafe or known types)
Console.WriteLine(sizeof(int));       // 4
Console.WriteLine(sizeof(double));    // 8
Console.WriteLine(sizeof(char));      // 2
Console.WriteLine(sizeof(bool));      // 1
Console.WriteLine(sizeof(long));      // 8

// nameof — gets the name of a variable, type, or member as a string
string variableName = "test";
Console.WriteLine(nameof(variableName));      // "variableName"
Console.WriteLine(nameof(Console.WriteLine)); // "WriteLine"
Console.WriteLine(nameof(String));            // "String"
```

### 10.2 The is Operator (Type Testing)

```csharp
object value = 42;

if (value is int)
{
    Console.WriteLine("It's an integer!");
}

// is with pattern variable (C# 7+)
if (value is int number)
{
    Console.WriteLine($"The integer is {number}");
}

// is with constant patterns
if (value is not null)
{
    Console.WriteLine($"Value is not null: {value}");
}
```

### 10.3 The as Operator (Safe Cast)

```csharp
object obj = "Hello, World!";

// as — returns null instead of throwing on failed cast
string? str = obj as string;
if (str != null)
{
    Console.WriteLine($"String: {str}");
}

// as only works with reference types and nullable value types
// int num = obj as int;  // Compile error: int is not nullable
int? num = obj as int?;    // OK: returns null (obj is a string, not an int)
Console.WriteLine(num);    // (null)
```

## 11. Practice Problems

1. **Expression Evaluator**: Without running the code, predict the output of each expression. Then verify by running:
   ```csharp
   Console.WriteLine(5 + 3 * 2);
   Console.WriteLine((5 + 3) * 2);
   Console.WriteLine(10 / 3 + 10 % 3);
   Console.WriteLine(true || false && false);
   Console.WriteLine((true || false) && false);
   ```

2. **Bit Flag System**: Create a permission system using bitwise operators. Define flags for `Read = 1`, `Write = 2`, `Execute = 4`, `Admin = 8`. Write methods to: (a) grant a permission, (b) revoke a permission, (c) check if a permission is set, and (d) display all active permissions. Demonstrate combining and checking multiple permissions.

3. **Null-Safe Chain**: Given the following class structure, write a chain of null-conditional operators to safely access deeply nested properties:
   ```csharp
   class Company { public Department? MainDepartment; }
   class Department { public Employee? Lead; }
   class Employee { public Address? HomeAddress; }
   class Address { public string? City; }
   ```
   Print the city or "Unknown" if any part of the chain is null.

4. **Overflow Detective**: Write a program that demonstrates overflow for `byte`, `short`, `int`, and `long`. For each type, show: (a) the maximum value, (b) the result of adding 1 in an unchecked context, and (c) that a `checked` context throws `OverflowException`.

5. **Grade Calculator**: Write a program that reads a numeric score (0-100) from the user and converts it to a letter grade using only the ternary operator (no `if` statements). Handle invalid input (non-numeric, out of range) using `TryParse` and null-coalescing operators.

---

**Previous**: [Variables and Types](./02_Variables_and_Types.md) | **Next**: [Control Flow](./04_Control_Flow.md)
