# Lambda Expressions and Closures

**Previous**: [Delegates and Events](./01_Delegates_and_Events.md) | **Next**: [LINQ](./03_LINQ.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write lambda expressions using both expression and statement syntax
2. Use lambdas with `Func<T>`, `Action<T>`, and `Predicate<T>` delegates
3. Understand how closures capture variables and identify common pitfalls
4. Apply static lambdas to avoid unintended captures
5. Leverage natural function types and method group conversions
6. Compare local functions and lambdas to choose the right tool
7. Build processing pipelines using functional composition with lambdas

---

Lambda expressions are concise, inline functions that can be assigned to delegate variables, passed as arguments, or used anywhere a delegate or expression tree is expected. Combined with closures — the ability to capture variables from the enclosing scope — lambdas are the backbone of LINQ, event handling, and functional-style programming in C#.

## 1. Lambda Expression Syntax

### 1.1 Expression Lambdas

An expression lambda has a single expression as its body. The result of the expression is the return value.

```csharp
// Single parameter — parentheses optional
Func<int, int> square = x => x * x;
Console.WriteLine(square(5)); // 25

// Multiple parameters — parentheses required
Func<int, int, int> add = (a, b) => a + b;
Console.WriteLine(add(3, 7)); // 10

// No parameters
Func<DateTime> now = () => DateTime.UtcNow;
Console.WriteLine(now());

// Explicit parameter types (useful when compiler can't infer)
Func<int, string> toHex = (int n) => $"0x{n:X}";
Console.WriteLine(toHex(255)); // 0xFF
```

### 1.2 Statement Lambdas

A statement lambda has a block body enclosed in braces. It can contain multiple statements and requires an explicit `return` if it has a return type.

```csharp
Func<int, int, int> safeDivide = (a, b) =>
{
    if (b == 0)
    {
        Console.WriteLine("Warning: division by zero, returning 0");
        return 0;
    }
    return a / b;
};

Console.WriteLine(safeDivide(10, 3));  // 3
Console.WriteLine(safeDivide(10, 0));  // Warning... then 0
```

### 1.3 Discards in Lambda Parameters

When you don't need certain parameters, use discards (`_`) to signal intent:

```csharp
// Event handler where we don't need sender or args
button.Click += (_, _) => Console.WriteLine("Clicked!");

// Projection where index is irrelevant
Func<string, int, string> ignoreIndex = (name, _) => name.ToUpper();
```

### 1.4 Lambda Return Type Specification (C# 10+)

In cases where the return type is ambiguous, you can specify it explicitly:

```csharp
// Without explicit return type — compiler can't decide between Func<bool> and Func<object>
// var choose = () => true ? 1 : null; // Error in some contexts

// Explicit return type
var choose = int? () => true ? 1 : null;
Console.WriteLine(choose()); // 1

// Useful for method overload resolution
var parse = int (string s) => int.Parse(s);
```

## 2. Lambdas with Built-in Delegate Types

### 2.1 Action\<T\> — Side-Effect Lambdas

```csharp
Action<string> greet = name => Console.WriteLine($"Hello, {name}!");
greet("Alice"); // Hello, Alice!

Action<List<int>> shuffleAndPrint = list =>
{
    var rng = new Random();
    int n = list.Count;
    while (n > 1)
    {
        int k = rng.Next(n--);
        (list[n], list[k]) = (list[k], list[n]);
    }
    Console.WriteLine(string.Join(", ", list));
};

shuffleAndPrint(new List<int> { 1, 2, 3, 4, 5 });
```

### 2.2 Func\<T, TResult\> — Transform Lambdas

```csharp
Func<string, int> wordCount = text =>
    text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

Console.WriteLine(wordCount("The quick brown fox")); // 4

// Chaining Funcs
Func<string, string> trim = s => s.Trim();
Func<string, string> lower = s => s.ToLowerInvariant();
Func<string, string> slugify = s => lower(trim(s)).Replace(' ', '-');

Console.WriteLine(slugify("  Hello World  ")); // hello-world
```

### 2.3 Predicate\<T\> — Filter Lambdas

```csharp
Predicate<string> isEmail = s =>
    s.Contains('@') && s.Contains('.') && s.IndexOf('@') < s.LastIndexOf('.');

var addresses = new List<string>
{
    "alice@example.com", "not-an-email", "bob@test.org", "bad@", "@.com"
};

List<string> validEmails = addresses.FindAll(isEmail);
Console.WriteLine(string.Join(", ", validEmails));
// alice@example.com, bob@test.org
```

### 2.4 Comparison\<T\> — Sort Lambdas

```csharp
var people = new List<(string Name, int Age)>
{
    ("Alice", 30), ("Bob", 25), ("Charlie", 35), ("Diana", 28)
};

// Sort by age ascending
people.Sort((a, b) => a.Age.CompareTo(b.Age));
Console.WriteLine(string.Join(", ", people.Select(p => $"{p.Name}({p.Age})")));
// Bob(25), Diana(28), Alice(30), Charlie(35)

// Sort by name descending
people.Sort((a, b) => string.Compare(b.Name, a.Name, StringComparison.Ordinal));
Console.WriteLine(string.Join(", ", people.Select(p => p.Name)));
// Diana, Charlie, Bob, Alice
```

## 3. Closures and Variable Capture

A closure is a lambda that references variables from its enclosing scope. The lambda "captures" those variables and keeps them alive even after the enclosing method returns.

### 3.1 Basic Closure

```csharp
public static Func<int, int> CreateMultiplier(int factor)
{
    // 'factor' is captured by the lambda
    return x => x * factor;
}

var triple = CreateMultiplier(3);
var tenTimes = CreateMultiplier(10);

Console.WriteLine(triple(5));    // 15
Console.WriteLine(tenTimes(5));  // 50
```

### 3.2 How Closures Work Internally

The compiler generates a hidden class to hold captured variables. The lambda becomes a method on that class.

```csharp
// What you write:
int counter = 0;
Action increment = () => counter++;

// What the compiler roughly generates:
// class <>c__DisplayClass
// {
//     public int counter;
//     public void <Main>b__0() => counter++;
// }
// var display = new <>c__DisplayClass { counter = 0 };
// Action increment = display.<Main>b__0;
```

### 3.3 Mutable Captures

Captured variables are shared by reference, not copied. Changes to the variable inside the lambda affect the outer scope and vice versa.

```csharp
int count = 0;

Action increment = () => count++;
Action print = () => Console.WriteLine($"Count: {count}");

increment();
increment();
increment();
print(); // Count: 3

count = 100;
print(); // Count: 100
```

### 3.4 Closure over a Loop Variable — The Classic Pitfall

```csharp
// BUG: all actions capture the SAME variable 'i'
var actions = new List<Action>();
for (int i = 0; i < 5; i++)
{
    actions.Add(() => Console.Write(i + " "));
}
foreach (var action in actions)
    action();
// Output: 5 5 5 5 5  (NOT 0 1 2 3 4!)

// FIX: create a local copy inside the loop
var fixedActions = new List<Action>();
for (int i = 0; i < 5; i++)
{
    int captured = i; // local copy — each iteration gets its own variable
    fixedActions.Add(() => Console.Write(captured + " "));
}
foreach (var action in fixedActions)
    action();
// Output: 0 1 2 3 4

// NOTE: foreach does NOT have this problem since C# 5
var foreachActions = new List<Action>();
foreach (var name in new[] { "Alice", "Bob", "Charlie" })
{
    foreachActions.Add(() => Console.Write(name + " "));
}
foreach (var action in foreachActions)
    action();
// Output: Alice Bob Charlie (correct)
```

### 3.5 Closure Lifetime and Memory

Captured variables live as long as the delegate that references them. This can cause unexpected memory retention:

```csharp
public static Func<int> CreateCounter()
{
    var largeData = new byte[10_000_000]; // 10 MB
    int count = 0;

    // 'largeData' is captured even though we only use 'count'
    // The entire display class (containing both) stays alive
    return () =>
    {
        // If we reference largeData here, it will never be GC'd
        // as long as this delegate exists
        return ++count;
    };
}

// BETTER: avoid capturing unnecessary variables
public static Func<int> CreateCounterFixed()
{
    int count = 0;
    return () => ++count;
    // largeData is not in scope, not captured, can be GC'd
}
```

## 4. Static Lambdas

C# 9 introduced `static` lambdas that are guaranteed not to capture any variables from the enclosing scope. The compiler produces an error if the lambda body references any instance or local variable.

### 4.1 Syntax and Purpose

```csharp
// Regular lambda — captures nothing, but compiler doesn't enforce it
Func<int, int> regular = x => x * 2;

// Static lambda — compiler enforces that nothing is captured
Func<int, int> @static = static x => x * 2;

// This would be a compile error:
int factor = 3;
// Func<int, int> bad = static x => x * factor; // Error: cannot capture 'factor'
```

### 4.2 Performance Benefits

Static lambdas can avoid allocating a closure object, and the compiler can cache the delegate instance since it has no state.

```csharp
// Non-static: might allocate a closure even if it captures nothing visible
var numbers = Enumerable.Range(1, 100);

// Static: guaranteed no closure allocation
var evens = numbers.Where(static x => x % 2 == 0);
var doubled = evens.Select(static x => x * 2);

Console.WriteLine(doubled.Sum()); // 5100
```

### 4.3 Static Lambdas with Static Local Variables

You can use static lambdas together with const values or static members:

```csharp
const int Threshold = 100;

Func<int, bool> isAboveThreshold = static x => x > Threshold; // OK — const is inlined
Console.WriteLine(isAboveThreshold(150)); // True
```

## 5. Natural Function Types and Method Group Conversions

### 5.1 Natural Function Types (C# 10+)

Starting in C# 10, the compiler can infer a delegate type for lambdas and method groups when assigned to `var`.

```csharp
// Before C# 10 — must specify delegate type
Func<int, int, int> add1 = (a, b) => a + b;

// C# 10+ — compiler infers Func<int, int, int>
var add2 = (int a, int b) => a + b;

// Parameter types must be explicit for var inference
// var bad = (a, b) => a + b; // Error — types not inferrable

// Method group with natural type
var writeLine = Console.WriteLine; // infers Action<string> (or one of the overloads)
```

### 5.2 Method Group Conversions

A method group (method name without parentheses) can be converted to a compatible delegate type.

```csharp
// Method group as argument
var numbers = new List<int> { 3, 1, 4, 1, 5 };
numbers.Sort(int.CompareTo); // Error — this doesn't work as you'd expect

// Correct method group usage
numbers.ForEach(Console.WriteLine); // Action<int> -> Console.WriteLine(int)

// Method group to delegate variable
Func<string, bool> isNullOrEmpty = string.IsNullOrEmpty;
Console.WriteLine(isNullOrEmpty("")); // True

// Instance method group
var sb = new System.Text.StringBuilder();
Action<string> append = sb.Append; // Error — Append returns StringBuilder, not void

// This works — using a compatible overload
Func<string, System.Text.StringBuilder> appendFunc = sb.Append;
appendFunc("Hello");
appendFunc(" World");
Console.WriteLine(sb); // Hello World
```

## 6. Expression-Bodied Members

Expression-bodied syntax uses the `=>` arrow for concise single-expression member definitions. While not lambdas themselves, they use the same arrow syntax.

### 6.1 Methods, Properties, and Indexers

```csharp
public class Circle
{
    public double Radius { get; }

    // Expression-bodied constructor
    public Circle(double radius) => Radius = radius;

    // Expression-bodied readonly property
    public double Diameter => Radius * 2;

    // Expression-bodied method
    public double Area() => Math.PI * Radius * Radius;

    // Expression-bodied property with get and set
    private string _name = "";
    public string Name
    {
        get => _name;
        set => _name = value ?? throw new ArgumentNullException(nameof(value));
    }

    // Expression-bodied indexer
    private readonly double[] _points = new double[10];
    public double this[int i] => _points[i];

    // Expression-bodied finalizer
    ~Circle() => Console.WriteLine("Circle finalized");

    // Expression-bodied ToString
    public override string ToString() => $"Circle(r={Radius})";
}
```

### 6.2 When to Use Expression Bodies

Expression bodies are best for simple, single-expression members. Once logic requires multiple statements, switch to a block body for readability.

```csharp
// Good — simple and clear
public bool IsValid => Name.Length > 0 && Age >= 0;

// Debatable — getting complex
public string Display => $"{Name} ({Age}) - {(IsValid ? "OK" : "Invalid")}";

// Too complex — use a block body instead
public string DetailedReport()
{
    var sb = new System.Text.StringBuilder();
    sb.AppendLine($"Name: {Name}");
    sb.AppendLine($"Age: {Age}");
    sb.AppendLine($"Valid: {IsValid}");
    return sb.ToString();
}
```

## 7. Local Functions vs Lambdas

C# 7 introduced local functions — named methods defined inside other methods. They overlap significantly with lambdas but have important differences.

### 7.1 Syntax Comparison

```csharp
public void DemonstrateComparison()
{
    // Lambda assigned to a delegate variable
    Func<int, int> squareLambda = x => x * x;

    // Local function — no delegate allocation
    int SquareLocal(int x) => x * x;

    Console.WriteLine(squareLambda(5)); // 25
    Console.WriteLine(SquareLocal(5));  // 25
}
```

### 7.2 Key Differences

```csharp
public static void Differences()
{
    // 1. Local functions can be recursive without extra allocation
    int Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);
    Console.WriteLine(Factorial(10)); // 3628800

    // Lambda recursion requires assigning to variable first
    Func<int, int> factLambda = null!;
    factLambda = n => n <= 1 ? 1 : n * factLambda(n - 1);
    Console.WriteLine(factLambda(10)); // 3628800

    // 2. Local functions support generics
    T Identity<T>(T value) => value;
    Console.WriteLine(Identity(42));
    Console.WriteLine(Identity("hello"));
    // Lambdas cannot be generic

    // 3. Local functions can use 'ref', 'in', 'out' parameters
    void Swap(ref int a, ref int b) => (a, b) = (b, a);
    int x = 1, y = 2;
    Swap(ref x, ref y);
    Console.WriteLine($"{x}, {y}"); // 2, 1

    // 4. Local functions can be iterators
    IEnumerable<int> Range(int start, int count)
    {
        for (int i = 0; i < count; i++)
            yield return start + i;
    }
    Console.WriteLine(string.Join(", ", Range(5, 3))); // 5, 6, 7

    // 5. Local functions can be async without Task
    async IAsyncEnumerable<int> AsyncRange(int count)
    {
        for (int i = 0; i < count; i++)
        {
            await Task.Delay(10);
            yield return i;
        }
    }
}
```

### 7.3 Static Local Functions

Static local functions (C# 8) cannot capture any variables from the enclosing method, preventing accidental closures:

```csharp
public static double CalculateDistance(double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    return Hypotenuse(dx, dy);

    // Static local function — all data passed as parameters
    static double Hypotenuse(double a, double b)
        => Math.Sqrt(a * a + b * b);

    // This would fail:
    // static double Bad() => dx + dy; // Error: cannot capture dx
}
```

### 7.4 Decision Guide

| Feature | Lambda | Local Function |
|---------|--------|----------------|
| Passed as delegate argument | Preferred | Must wrap in delegate |
| Recursive | Awkward | Natural |
| Generic | Not possible | Supported |
| ref/out parameters | Not possible | Supported |
| Iterator (yield) | Not possible | Supported |
| Performance (no allocation) | May allocate closure | Can be allocation-free |
| Async enumerable | Not possible | Supported |
| Can be static | Yes (C# 9) | Yes (C# 8) |

## 8. Practical Example: Building a Pipeline with Lambdas

Let's build a text-processing pipeline using functional composition.

### 8.1 The Pipeline Builder

```csharp
public class Pipeline<T>
{
    private readonly List<Func<T, T>> _steps = new();

    public Pipeline<T> AddStep(Func<T, T> step)
    {
        _steps.Add(step);
        return this; // fluent API
    }

    public Pipeline<T> AddConditionalStep(Func<T, bool> predicate, Func<T, T> step)
    {
        _steps.Add(value => predicate(value) ? step(value) : value);
        return this;
    }

    public T Execute(T input)
    {
        T result = input;
        foreach (var step in _steps)
        {
            result = step(result);
        }
        return result;
    }

    // Compose all steps into a single function
    public Func<T, T> Compile()
    {
        var steps = _steps.ToList(); // snapshot
        return input =>
        {
            T result = input;
            foreach (var step in steps)
                result = step(result);
            return result;
        };
    }
}
```

### 8.2 Text Processing Pipeline

```csharp
var textPipeline = new Pipeline<string>()
    .AddStep(s => s.Trim())
    .AddStep(s => s.ToLowerInvariant())
    .AddStep(s => System.Text.RegularExpressions.Regex.Replace(s, @"\s+", " "))
    .AddStep(s => s.Replace(' ', '-'))
    .AddConditionalStep(
        s => s.Length > 50,
        s => s[..50] + "...");

string slug = textPipeline.Execute("   Hello   Beautiful    World   ");
Console.WriteLine(slug); // hello-beautiful-world

// Compile for repeated use
Func<string, string> slugify = textPipeline.Compile();
Console.WriteLine(slugify("  Another   Test  ")); // another-test
```

### 8.3 Numeric Pipeline with Logging

```csharp
var mathPipeline = new Pipeline<double>()
    .AddStep(x => { Console.Write($"  Input: {x}"); return x; })
    .AddStep(x => { double r = Math.Abs(x); Console.Write($" -> Abs: {r}"); return r; })
    .AddStep(x => { double r = Math.Round(x, 2); Console.Write($" -> Round: {r}"); return r; })
    .AddStep(x => { double r = Math.Sqrt(x); Console.Write($" -> Sqrt: {r:F4}"); return r; })
    .AddStep(x => { Console.WriteLine(); return x; });

double result = mathPipeline.Execute(-17.456);
Console.WriteLine($"Final result: {result:F4}");
// Input: -17.456 -> Abs: 17.456 -> Round: 17.46 -> Sqrt: 4.1785
// Final result: 4.1785
```

### 8.4 Function Composition Utilities

```csharp
public static class FuncExtensions
{
    /// <summary>Compose two functions: (f, g) => x => g(f(x))</summary>
    public static Func<T, TResult2> Then<T, TResult1, TResult2>(
        this Func<T, TResult1> first,
        Func<TResult1, TResult2> second)
    {
        return x => second(first(x));
    }

    /// <summary>Apply a function N times</summary>
    public static Func<T, T> Repeat<T>(this Func<T, T> f, int times)
    {
        return x =>
        {
            T result = x;
            for (int i = 0; i < times; i++)
                result = f(result);
            return result;
        };
    }

    /// <summary>Memoize a pure function</summary>
    public static Func<T, TResult> Memoize<T, TResult>(this Func<T, TResult> f)
        where T : notnull
    {
        var cache = new Dictionary<T, TResult>();
        return x =>
        {
            if (!cache.TryGetValue(x, out var result))
            {
                result = f(x);
                cache[x] = result;
            }
            return result;
        };
    }
}

// Usage
Func<string, string> trim = s => s.Trim();
Func<string, string> lower = s => s.ToLower();
Func<string, int> length = s => s.Length;

// Compose: trim -> lower -> length
Func<string, int> pipeline = trim.Then(lower).Then(length);
Console.WriteLine(pipeline("  Hello World  ")); // 11

// Repeat a function
Func<int, int> doubleIt = x => x * 2;
Func<int, int> eightTimes = doubleIt.Repeat(3); // 2^3 = 8x
Console.WriteLine(eightTimes(5)); // 40

// Memoize an expensive function
Func<int, long> fibonacci = null!;
fibonacci = n => n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2);
var memoFib = fibonacci.Memoize();
// Note: only outer calls are memoized; for deep memoization
// the recursive calls must also go through the memoized version.
```

## 9. Practice Problems

1. **Closure Counter Factory**: Write a function `CreateCounter(int start, int step)` that returns three delegates: `Func<int> next` (returns current and advances), `Func<int> peek` (returns current without advancing), and `Action reset` (resets to start). All three should share the same captured state. Demonstrate that calling `next()` repeatedly produces an arithmetic sequence.

2. **Loop Capture Debugging**: Given the following code, explain why it prints `10` ten times. Then fix it using (a) a local variable copy, (b) a `foreach` loop, and (c) a static lambda approach where the index is passed as a parameter instead of captured.
   ```csharp
   var funcs = new List<Func<int>>();
   for (int i = 0; i < 10; i++)
       funcs.Add(() => i);
   funcs.ForEach(f => Console.Write(f() + " "));
   ```

3. **Generic Pipeline Builder**: Extend the `Pipeline<T>` class from Section 8 to support an `AddStep<TIntermediate>(Func<T, TIntermediate>, Func<TIntermediate, T>)` method that temporarily transforms the value to an intermediate type, applies some operation, and transforms back. Use this to build a string pipeline that converts to char array, reverses it, and converts back.

4. **Memoization with Expiry**: Implement a `MemoizeWithExpiry<T, TResult>(this Func<T, TResult> f, TimeSpan ttl)` extension method. Cached results should expire after `ttl` has elapsed. Use `Stopwatch` or `DateTime` for timing. Test with a function that simulates an expensive computation using `Thread.Sleep`.

5. **Lambda vs Local Function Benchmark**: Write code that calls a lambda and a local function each one million times in a tight loop. Use `System.Diagnostics.Stopwatch` to measure the elapsed time. Vary the scenario: (a) no captures, (b) capturing one int variable, (c) capturing a large object. Report your findings on when each approach is preferable.
