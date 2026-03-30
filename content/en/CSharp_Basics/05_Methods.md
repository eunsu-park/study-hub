# Methods

**Previous**: [Control Flow](./04_Control_Flow.md) | **Next**: [Arrays and Strings](./06_Arrays_and_Strings.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and call methods with various parameter types
2. Use `ref`, `out`, `in`, and `params` parameter modifiers
3. Define optional parameters and use named arguments
4. Overload methods with different parameter signatures
5. Write concise expression-bodied methods
6. Create and use local functions
7. Implement recursive algorithms
8. Distinguish between static and instance methods
9. Return multiple values with tuple return types

---

A **method** is a named block of code that performs a specific task. Methods are the primary way to organize and reuse code in C#. They take input (parameters), perform operations, and optionally return a result. Well-designed methods make programs easier to read, test, and maintain.

## 1. Method Declaration and Calling

### 1.1 Basic Method Structure

```csharp
// Method declaration
// <access modifier> <return type> <name>(<parameters>)
// {
//     <body>
//     return <value>; // if non-void
// }

void SayHello()
{
    Console.WriteLine("Hello!");
}

int Add(int a, int b)
{
    return a + b;
}

string FormatName(string first, string last)
{
    return $"{last}, {first}";
}

// Calling methods
SayHello();                         // Hello!
int sum = Add(3, 5);                // 8
string name = FormatName("John", "Doe");  // "Doe, John"
Console.WriteLine($"Sum: {sum}, Name: {name}");
```

### 1.2 Return Types

```csharp
// void — no return value
void PrintLine(string text)
{
    Console.WriteLine($">>> {text}");
}

// Specific return type
double CalculateArea(double radius)
{
    return Math.PI * radius * radius;
}

// Boolean return
bool IsEven(int number)
{
    return number % 2 == 0;
}

// Early return for guard clauses
double SafeDivide(double a, double b)
{
    if (b == 0)
    {
        Console.WriteLine("Warning: division by zero");
        return 0;  // Early return
    }
    return a / b;
}

Console.WriteLine($"Area: {CalculateArea(5):F2}");    // Area: 78.54
Console.WriteLine($"IsEven(7): {IsEven(7)}");          // IsEven(7): False
Console.WriteLine($"10 / 3 = {SafeDivide(10, 3):F2}"); // 10 / 3 = 3.33
```

### 1.3 Methods as Expressions in Other Contexts

```csharp
// Methods can be used anywhere an expression of their return type is expected
int max = Math.Max(Add(1, 2), Add(3, 4));
Console.WriteLine(max);  // 7

// Chaining
string result = FormatName("Jane", "Smith").ToUpper();
Console.WriteLine(result);  // SMITH, JANE

// In string interpolation
Console.WriteLine($"Circle area: {CalculateArea(3):F4}");
```

## 2. Parameter Passing: Value, ref, out, in

Understanding how parameters are passed is crucial. By default, C# passes parameters **by value**, meaning the method receives a copy.

### 2.1 Pass by Value (Default)

```csharp
void TryToChange(int x)
{
    x = 999;  // Only changes the local copy
    Console.WriteLine($"Inside method: x = {x}");
}

int number = 42;
TryToChange(number);
Console.WriteLine($"After method: number = {number}");
// Output:
// Inside method: x = 999
// After method: number = 42  (unchanged!)
```

### 2.2 Pass by Reference: ref

The `ref` keyword passes a reference to the variable, allowing the method to modify the original:

```csharp
void Swap(ref int a, ref int b)
{
    int temp = a;
    a = b;
    b = temp;
}

int x = 10, y = 20;
Console.WriteLine($"Before: x={x}, y={y}");  // Before: x=10, y=20
Swap(ref x, ref y);
Console.WriteLine($"After:  x={x}, y={y}");  // After:  x=20, y=10

// ref parameters MUST be initialized before passing
// int uninitialized;
// Swap(ref uninitialized, ref y);  // Compile error: use of unassigned variable

// Practical example: increment and return
void IncrementBy(ref int value, int amount)
{
    value += amount;
}

int counter = 0;
IncrementBy(ref counter, 5);
IncrementBy(ref counter, 3);
Console.WriteLine($"Counter: {counter}");  // 8
```

### 2.3 Output Parameters: out

The `out` keyword is similar to `ref` but is specifically for output. The variable does not need to be initialized before passing, and the method **must** assign it before returning:

```csharp
// Classic out usage: returning multiple values
bool TryDivide(int dividend, int divisor, out int quotient, out int remainder)
{
    if (divisor == 0)
    {
        quotient = 0;
        remainder = 0;
        return false;
    }
    quotient = dividend / divisor;
    remainder = dividend % divisor;
    return true;
}

if (TryDivide(17, 5, out int q, out int r))
{
    Console.WriteLine($"17 / 5 = {q} remainder {r}");  // 17 / 5 = 3 remainder 2
}

// Inline out variable declaration (C# 7+)
if (int.TryParse("123", out int parsed))
{
    Console.WriteLine($"Parsed: {parsed}");  // Parsed: 123
}

// Discard out values you don't need
if (int.TryParse("456", out _))
{
    Console.WriteLine("It's a valid number");
}

// out parameters with complex data
void GetMinMax(int[] array, out int min, out int max)
{
    min = int.MaxValue;
    max = int.MinValue;
    foreach (int val in array)
    {
        if (val < min) min = val;
        if (val > max) max = val;
    }
}

int[] data = { 5, 2, 8, 1, 9, 3 };
GetMinMax(data, out int minimum, out int maximum);
Console.WriteLine($"Min: {minimum}, Max: {maximum}");  // Min: 1, Max: 9
```

### 2.4 Read-Only Reference: in

The `in` keyword passes by reference but prevents the method from modifying the value. It is an optimization for large structs:

```csharp
// in prevents modification — useful for large structs
double CalculateDistance(in (double X, double Y) p1, in (double X, double Y) p2)
{
    // p1.X = 0;  // Compile error: cannot modify 'in' parameter
    double dx = p2.X - p1.X;
    double dy = p2.Y - p1.Y;
    return Math.Sqrt(dx * dx + dy * dy);
}

var point1 = (X: 0.0, Y: 0.0);
var point2 = (X: 3.0, Y: 4.0);
double dist = CalculateDistance(in point1, in point2);
Console.WriteLine($"Distance: {dist}");  // Distance: 5

// The 'in' keyword is optional at the call site
double dist2 = CalculateDistance(point1, point2);  // Also works
```

### 2.5 Comparison Table

```csharp
// Summary of parameter modifiers:
// ──────────────────────────────────────────────────────
// Modifier  | Direction  | Must init before? | Can modify?
// ──────────────────────────────────────────────────────
// (none)    | In         | Yes               | Copy only
// ref       | In/Out     | Yes               | Yes
// out       | Out        | No                | Must assign
// in        | In         | Yes               | No
// ──────────────────────────────────────────────────────
```

## 3. params: Variable Number of Arguments

The `params` keyword allows a method to accept any number of arguments of a specified type:

```csharp
// params must be the last parameter and must be an array type
int Sum(params int[] numbers)
{
    int total = 0;
    foreach (int n in numbers)
    {
        total += n;
    }
    return total;
}

// Call with any number of arguments
Console.WriteLine(Sum());              // 0
Console.WriteLine(Sum(1));             // 1
Console.WriteLine(Sum(1, 2, 3));       // 6
Console.WriteLine(Sum(1, 2, 3, 4, 5)); // 15

// Can also pass an array directly
int[] values = { 10, 20, 30 };
Console.WriteLine(Sum(values));        // 60

// params with other parameters
string JoinWithSeparator(string separator, params string[] items)
{
    return string.Join(separator, items);
}

Console.WriteLine(JoinWithSeparator(", ", "apple", "banana", "cherry"));
// Output: apple, banana, cherry

// Practical example: logging with variable arguments
void Log(string level, string message, params object[] args)
{
    string formatted = string.Format(message, args);
    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{level}] {formatted}");
}

Log("INFO", "User {0} logged in from {1}", "Alice", "192.168.1.1");
Log("ERROR", "Failed to process {0} items", 42);
```

## 4. Optional Parameters and Named Arguments

### 4.1 Optional Parameters

Parameters with default values are optional — the caller can omit them:

```csharp
void PrintMessage(string message, int repeat = 1, bool uppercase = false)
{
    string output = uppercase ? message.ToUpper() : message;
    for (int i = 0; i < repeat; i++)
    {
        Console.WriteLine(output);
    }
}

PrintMessage("Hello");                         // Hello (once, lowercase)
PrintMessage("Hello", 3);                      // Hello Hello Hello
PrintMessage("Hello", 2, true);               // HELLO HELLO
PrintMessage("Hello", uppercase: true);        // HELLO (using named argument to skip 'repeat')

// Optional parameters must come after required parameters
string CreateGreeting(string name, string prefix = "Mr.", string suffix = "")
{
    return $"{prefix} {name}{(suffix.Length > 0 ? $", {suffix}" : "")}";
}

Console.WriteLine(CreateGreeting("Smith"));                    // Mr. Smith
Console.WriteLine(CreateGreeting("Smith", "Dr."));             // Dr. Smith
Console.WriteLine(CreateGreeting("Smith", "Dr.", "PhD"));      // Dr. Smith, PhD
```

### 4.2 Named Arguments

Named arguments let you specify parameters by name, in any order:

```csharp
void CreateUser(string name, int age, string email, string role = "user")
{
    Console.WriteLine($"Name: {name}, Age: {age}, Email: {email}, Role: {role}");
}

// Positional (must be in order)
CreateUser("Alice", 30, "alice@example.com");

// Named (can be in any order)
CreateUser(email: "bob@example.com", name: "Bob", age: 25);

// Mixed (positional first, then named)
CreateUser("Charlie", age: 35, email: "charlie@example.com", role: "admin");

// Named arguments are especially useful with many optional parameters
void Configure(
    string host = "localhost",
    int port = 8080,
    bool ssl = false,
    int timeout = 30,
    int maxRetries = 3)
{
    Console.WriteLine($"Host: {host}:{port}, SSL: {ssl}, Timeout: {timeout}s, Retries: {maxRetries}");
}

Configure(port: 443, ssl: true);  // Only specify what differs from defaults
Configure(maxRetries: 5, timeout: 60);
```

## 5. Method Overloading

Method overloading allows multiple methods with the same name but different parameter lists:

```csharp
// Same name, different parameter types
int Multiply(int a, int b)
{
    Console.WriteLine("int * int");
    return a * b;
}

double Multiply(double a, double b)
{
    Console.WriteLine("double * double");
    return a * b;
}

string Multiply(string text, int count)
{
    Console.WriteLine("string * int");
    return string.Concat(Enumerable.Repeat(text, count));
}

Console.WriteLine(Multiply(3, 4));          // int * int → 12
Console.WriteLine(Multiply(3.5, 2.0));      // double * double → 7
Console.WriteLine(Multiply("Ha", 3));       // string * int → HaHaHa

// Same name, different number of parameters
double Average(double a, double b)
{
    return (a + b) / 2;
}

double Average(double a, double b, double c)
{
    return (a + b + c) / 3;
}

double Average(params double[] values)
{
    return values.Length > 0 ? values.Average() : 0;
}

Console.WriteLine(Average(10.0, 20.0));          // 2-param version: 15
Console.WriteLine(Average(10.0, 20.0, 30.0));    // 3-param version: 20
Console.WriteLine(Average(1.0, 2.0, 3.0, 4.0)); // params version: 2.5
```

### 5.1 Overloading Rules

```csharp
// Overloading is based on parameter LIST, not return type
// These are valid overloads:
void Process(int x) { }
void Process(string x) { }
void Process(int x, int y) { }

// This is NOT a valid overload (same parameter list, different return type):
// int Process(int x) { return x; }  // Compile error: already defined

// ref, out, and in count as different signatures from each other
void Transform(ref int x) { x *= 2; }
void Transform(out int x) { x = 42; }
// But ref and out cannot overload each other:
// void Transform(ref int x) { }  // Compile error if Transform(out int x) exists
```

## 6. Expression-Bodied Methods

For simple methods that consist of a single expression, use the `=>` arrow syntax:

```csharp
// Traditional method body
int Square(int x)
{
    return x * x;
}

// Expression-bodied equivalent (shorter)
int SquareExpr(int x) => x * x;

// More examples
double CircleArea(double r) => Math.PI * r * r;
bool IsPositive(int n) => n > 0;
string Greet(string name) => $"Hello, {name}!";
int Max(int a, int b) => a > b ? a : b;
void PrintStars(int count) => Console.WriteLine(new string('*', count));

// Using expression-bodied methods
Console.WriteLine(SquareExpr(7));          // 49
Console.WriteLine(CircleArea(5));          // 78.539...
Console.WriteLine(IsPositive(-3));         // False
Console.WriteLine(Greet("World"));         // Hello, World!
Console.WriteLine(Max(10, 20));            // 20
PrintStars(10);                             // **********
```

## 7. Local Functions

Local functions are methods defined inside other methods. They are useful for helper logic that only makes sense in one context:

```csharp
// Local function for validation
void ProcessOrder(string productId, int quantity)
{
    // Local function — not visible outside ProcessOrder
    bool IsValid()
    {
        if (string.IsNullOrEmpty(productId)) return false;
        if (quantity <= 0) return false;
        if (quantity > 1000) return false;
        return true;
    }

    if (!IsValid())
    {
        Console.WriteLine("Invalid order.");
        return;
    }

    Console.WriteLine($"Processing: {quantity}x {productId}");
}

ProcessOrder("SKU-123", 5);    // Processing: 5x SKU-123
ProcessOrder("", 5);            // Invalid order.
ProcessOrder("SKU-123", -1);   // Invalid order.

// Local functions can capture variables from the enclosing scope
int[] FilterAndSum(int[] numbers, int threshold)
{
    var filtered = new List<int>();
    int sum = 0;

    void Accumulate(int value)
    {
        // Accesses 'threshold' from enclosing scope
        if (value > threshold)
        {
            filtered.Add(value);
            sum += value;
        }
    }

    foreach (int n in numbers)
    {
        Accumulate(n);
    }

    Console.WriteLine($"Sum of values > {threshold}: {sum}");
    return filtered.ToArray();
}

int[] result = FilterAndSum(new[] { 1, 5, 3, 8, 2, 7, 4 }, 4);
Console.WriteLine($"Filtered: [{string.Join(", ", result)}]");
// Output: Sum of values > 4: 20
// Output: Filtered: [5, 8, 7]

// Static local functions (C# 8+) — cannot capture enclosing variables
int Calculate(int x, int y)
{
    return AddAndDouble(x, y);

    // static local function — must receive all data as parameters
    static int AddAndDouble(int a, int b) => (a + b) * 2;
}

Console.WriteLine(Calculate(3, 4));  // 14
```

## 8. Recursion

A recursive method calls itself. Every recursive method needs a **base case** (stopping condition) to prevent infinite recursion.

### 8.1 Classic Recursion Examples

```csharp
// Factorial: n! = n * (n-1) * ... * 1
long Factorial(int n)
{
    if (n <= 1) return 1;       // Base case
    return n * Factorial(n - 1); // Recursive case
}

Console.WriteLine($"5! = {Factorial(5)}");    // 120
Console.WriteLine($"10! = {Factorial(10)}");  // 3628800

// Fibonacci: F(n) = F(n-1) + F(n-2)
int Fibonacci(int n)
{
    if (n <= 0) return 0;       // Base case
    if (n == 1) return 1;       // Base case
    return Fibonacci(n - 1) + Fibonacci(n - 2);  // Recursive case
}

for (int i = 0; i <= 10; i++)
{
    Console.Write($"{Fibonacci(i)} ");
}
Console.WriteLine();
// Output: 0 1 1 2 3 5 8 13 21 34 55
```

### 8.2 Power and GCD

```csharp
// Power: base^exponent
double Power(double baseVal, int exponent)
{
    if (exponent == 0) return 1;
    if (exponent < 0) return 1.0 / Power(baseVal, -exponent);
    return baseVal * Power(baseVal, exponent - 1);
}

Console.WriteLine($"2^10 = {Power(2, 10)}");    // 1024
Console.WriteLine($"3^-2 = {Power(3, -2):F4}"); // 0.1111

// GCD using Euclidean algorithm
int Gcd(int a, int b)
{
    if (b == 0) return a;
    return Gcd(b, a % b);
}

Console.WriteLine($"GCD(48, 18) = {Gcd(48, 18)}");  // 6
Console.WriteLine($"GCD(100, 75) = {Gcd(100, 75)}"); // 25
```

### 8.3 Binary Search (Recursive)

```csharp
int BinarySearch(int[] sorted, int target, int low, int high)
{
    if (low > high) return -1;  // Base case: not found

    int mid = low + (high - low) / 2;

    if (sorted[mid] == target) return mid;
    if (sorted[mid] < target) return BinarySearch(sorted, target, mid + 1, high);
    return BinarySearch(sorted, target, low, mid - 1);
}

int[] array = { 2, 5, 8, 12, 16, 23, 38, 56, 72, 91 };
int index = BinarySearch(array, 23, 0, array.Length - 1);
Console.WriteLine($"Found 23 at index {index}");  // Found 23 at index 5
```

### 8.4 Recursion vs Iteration

```csharp
// Recursive sum (simple but may stack overflow for large n)
long SumRecursive(int n)
{
    if (n <= 0) return 0;
    return n + SumRecursive(n - 1);
}

// Iterative sum (no risk of stack overflow)
long SumIterative(int n)
{
    long total = 0;
    for (int i = 1; i <= n; i++)
        total += i;
    return total;
}

// Mathematical formula (best performance)
long SumFormula(int n) => (long)n * (n + 1) / 2;

Console.WriteLine(SumRecursive(100));  // 5050
Console.WriteLine(SumIterative(100));  // 5050
Console.WriteLine(SumFormula(100));    // 5050
```

## 9. Static vs Instance Methods (Preview)

This is a preview — classes are covered in depth in Lesson 9.

```csharp
class Calculator
{
    // Instance field
    private int _memory = 0;

    // Instance method — requires an object (accesses instance data)
    public void Store(int value)
    {
        _memory = value;
    }

    public int Recall()
    {
        return _memory;
    }

    // Static method — does NOT require an object (no instance data)
    public static int Add(int a, int b)
    {
        return a + b;
    }

    public static double SquareRoot(double x)
    {
        return Math.Sqrt(x);
    }
}

// Static methods: called on the class itself
int sum = Calculator.Add(3, 5);
double sqrt = Calculator.SquareRoot(16);
Console.WriteLine($"Add: {sum}, Sqrt: {sqrt}");

// Instance methods: called on an object
var calc = new Calculator();
calc.Store(42);
Console.WriteLine($"Recalled: {calc.Recall()}");  // 42
```

## 10. Tuple Return Types

Tuples allow methods to return multiple values without defining a custom class:

```csharp
// Unnamed tuple
(int, int) Divide(int dividend, int divisor)
{
    return (dividend / divisor, dividend % divisor);
}

var result = Divide(17, 5);
Console.WriteLine($"Quotient: {result.Item1}, Remainder: {result.Item2}");

// Named tuple (much better readability)
(int Quotient, int Remainder) DivideNamed(int dividend, int divisor)
{
    return (dividend / divisor, dividend % divisor);
}

var namedResult = DivideNamed(17, 5);
Console.WriteLine($"Quotient: {namedResult.Quotient}, Remainder: {namedResult.Remainder}");

// Deconstruction — extract tuple elements into separate variables
var (q, r) = DivideNamed(17, 5);
Console.WriteLine($"q={q}, r={r}");

// More complex example: statistics
(double Mean, double Min, double Max, int Count) GetStats(params double[] values)
{
    if (values.Length == 0)
        return (0, 0, 0, 0);

    double sum = 0, min = double.MaxValue, max = double.MinValue;
    foreach (double v in values)
    {
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    return (sum / values.Length, min, max, values.Length);
}

var stats = GetStats(4.5, 2.1, 8.3, 1.7, 6.9);
Console.WriteLine($"Mean: {stats.Mean:F2}, Min: {stats.Min}, Max: {stats.Max}, Count: {stats.Count}");
// Output: Mean: 4.70, Min: 1.7, Max: 8.3, Count: 5

// Discarding unwanted tuple elements
var (mean, _, _, count) = GetStats(1, 2, 3, 4, 5);
Console.WriteLine($"Mean of {count} values: {mean}");

// Tuples vs out parameters
// Tuples: cleaner syntax, named fields, easy deconstruction
// out: more traditional, works with TryParse pattern
```

## 11. Practice Problems

1. **Temperature Converter**: Write three overloaded methods called `Convert`: one that converts Celsius to Fahrenheit (taking a `double`), one that converts Fahrenheit to Celsius (taking a `double` and a `bool` flag `toFahrenheit = true`), and one that converts an array of temperatures. Use the formulas: F = C * 9/5 + 32 and C = (F - 32) * 5/9.

2. **String Utilities**: Write the following methods using expression-bodied syntax where possible: (a) `Reverse(string s)` — returns the reversed string, (b) `IsPalindrome(string s)` — checks if a string reads the same forwards and backwards (case-insensitive), (c) `CountVowels(string s)` — returns the number of vowels, (d) `Truncate(string s, int maxLength, string ellipsis = "...")` — truncates with optional ellipsis.

3. **Recursive Tower of Hanoi**: Implement the Tower of Hanoi puzzle recursively. The method should take the number of disks and the names of three pegs (source, auxiliary, destination). Print each move. Verify that 3 disks require 7 moves and 4 disks require 15 moves.

4. **Statistics with Tuples**: Write a method `Analyze(params int[] numbers)` that returns a named tuple containing: `(int Sum, double Average, int Min, int Max, int Range, double Variance)`. Use local functions for computing variance. Test with at least three different datasets.

5. **Ref/Out Swap and Parse**: (a) Write a generic-like method using `ref` that swaps two `string` variables. (b) Write a method `TryParsePoint(string input, out double x, out double y)` that parses strings like `"3.5, 4.2"` into two coordinates, returning `true` on success. Handle invalid formats gracefully.

---

**Previous**: [Control Flow](./04_Control_Flow.md) | **Next**: [Arrays and Strings](./06_Arrays_and_Strings.md)
