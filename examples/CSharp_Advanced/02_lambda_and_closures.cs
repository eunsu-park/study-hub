// Lesson 02: Lambda Expressions and Closures
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Linq;

// ============================================================
// 1. Lambda Syntax Basics
// ============================================================

Console.WriteLine("=== Lambda Syntax ===");

// Expression lambda — single expression, implicit return
Func<int, int> doubleIt = x => x * 2;
Console.WriteLine($"doubleIt(5) = {doubleIt(5)}");

// Statement lambda — multiple statements in braces
Func<int, int, string> compare = (a, b) =>
{
    if (a > b) return $"{a} is greater";
    if (a < b) return $"{b} is greater";
    return "equal";
};
Console.WriteLine($"compare(3, 7) = {compare(3, 7)}");

// Zero-parameter lambda
Func<DateTime> now = () => DateTime.Now;
Console.WriteLine($"now() = {now()}");

// Discard parameter (unused parameter convention)
Action<int> ignoreArg = _ => Console.WriteLine("  Argument ignored");
ignoreArg(42);

// Explicitly typed parameters (useful when compiler cannot infer)
Func<int, double, double> divide = (int a, double b) => a / b;
Console.WriteLine($"divide(10, 3.0) = {divide(10, 3.0):F4}");

// ============================================================
// 2. Closures — Capturing Outer Variables
// ============================================================

Console.WriteLine("\n=== Closures ===");

// A closure captures variables from the enclosing scope
int multiplier = 3;
Func<int, int> tripler = x => x * multiplier;
Console.WriteLine($"tripler(10) = {tripler(10)}");

// The captured variable is shared, not copied
multiplier = 5;
Console.WriteLine($"After changing multiplier to 5: tripler(10) = {tripler(10)}");

// Counter example — closure over a mutable variable
int counter = 0;
Action increment = () => counter++;
Func<int> getCount = () => counter;

increment();
increment();
increment();
Console.WriteLine($"Counter after 3 increments: {getCount()}");

// ============================================================
// 3. Closure Pitfall — Loop Variable Capture
// ============================================================

Console.WriteLine("\n=== Loop Variable Capture ===");

// Common pitfall: all lambdas capture the same loop variable
var actions = new List<Func<int>>();

// Correct in modern C# — each iteration gets its own variable copy
for (int i = 0; i < 5; i++)
{
    // In C# 5+, the loop variable is captured per-iteration
    actions.Add(() => i);
}

Console.Write("Loop capture (C# 5+ correct): ");
foreach (var action in actions)
    Console.Write($"{action()} ");
Console.WriteLine();

// Demonstrating the classic workaround (still useful to understand)
var actionsManual = new List<Func<int>>();
for (int i = 0; i < 5; i++)
{
    int localCopy = i; // Explicit local copy
    actionsManual.Add(() => localCopy);
}

Console.Write("Explicit local copy:          ");
foreach (var action in actionsManual)
    Console.Write($"{action()} ");
Console.WriteLine();

// ============================================================
// 4. Static Lambdas (C# 9+)
// ============================================================

Console.WriteLine("\n=== Static Lambdas ===");

// Static lambdas prevent accidental variable capture
// This causes a compile error if you try to capture outer variables:
// int x = 10;
// Func<int, int> bad = static (n) => n + x; // Error!

// Static lambda — guaranteed no capture, potential performance benefit
Func<int, int, int> staticAdd = static (a, b) => a + b;
Console.WriteLine($"staticAdd(3, 4) = {staticAdd(3, 4)}");

// Useful in hot paths where allocation from closure must be avoided
var numbers = Enumerable.Range(1, 10);
int sum = numbers.Aggregate(0, static (acc, n) => acc + n);
Console.WriteLine($"Sum of 1..10 using static lambda: {sum}");

// ============================================================
// 5. Local Functions vs Lambdas
// ============================================================

Console.WriteLine("\n=== Local Functions ===");

// Local function — declared inside a method, supports recursion efficiently
int Factorial(int n)
{
    if (n <= 1) return 1;
    return n * Factorial(n - 1);
}
Console.WriteLine($"Factorial(6) = {Factorial(6)}");

// Static local function — cannot capture enclosing variables
static int Add(int a, int b) => a + b;
Console.WriteLine($"static Add(10, 20) = {Add(10, 20)}");

// Local function with closure (non-static)
string prefix = "Result";
string FormatResult(int value) => $"{prefix}: {value}";
Console.WriteLine(FormatResult(42));

// Local functions are preferred over lambdas when:
// - Recursion is needed (no delegate allocation overhead)
// - The function is complex and benefits from a name
// - You want to avoid heap allocation from closure capture

// Fibonacci with local function and tuple return
(int value, int calls) Fibonacci(int n)
{
    int callCount = 0;

    int Fib(int k)
    {
        callCount++;
        if (k <= 1) return k;
        return Fib(k - 1) + Fib(k - 2);
    }

    return (Fib(n), callCount);
}

var (fib10, calls) = Fibonacci(10);
Console.WriteLine($"Fibonacci(10) = {fib10} (computed in {calls} recursive calls)");

// ============================================================
// 6. Natural Delegate Types (C# 10+)
// ============================================================

Console.WriteLine("\n=== Natural Delegate Types (C# 10+) ===");

// The compiler infers the delegate type from the lambda
var square = (int x) => x * x;           // Inferred as Func<int, int>
var greet = (string name) => $"Hi, {name}!"; // Inferred as Func<string, string>
var print = (string msg) => { Console.WriteLine($"  {msg}"); }; // Inferred as Action<string>

Console.WriteLine($"square(9) = {square(9)}");
Console.WriteLine($"greet(\"World\") = {greet("World")}");
print("This is an Action<string>");

// Default parameter values in lambdas (C# 12)
// var greetDefault = (string name = "World") => $"Hello, {name}!";
// Console.WriteLine(greetDefault());       // "Hello, World!"
// Console.WriteLine(greetDefault("Alice"));// "Hello, Alice!"

// ============================================================
// 7. Practical: Building a Pipeline with Lambdas
// ============================================================

Console.WriteLine("\n=== Lambda Pipeline ===");

// Compose a data processing pipeline using Func delegates
var data = new List<string>
{
    "  Alice  ", "BOB", "  charlie", "DIANA  ", "  eve  "
};

// Define transformation steps
Func<string, string> trim = s => s.Trim();
Func<string, string> lower = s => s.ToLower();
Func<string, string> capitalize = s =>
    string.IsNullOrEmpty(s) ? s : char.ToUpper(s[0]) + s[1..];

// Compose the pipeline
Func<string, string> Compose(params Func<string, string>[] steps)
{
    return input =>
    {
        var result = input;
        foreach (var step in steps)
            result = step(result);
        return result;
    };
}

var normalize = Compose(trim, lower, capitalize);

Console.WriteLine("Normalized names:");
foreach (var name in data)
    Console.WriteLine($"  \"{name}\" -> \"{normalize(name)}\"");
