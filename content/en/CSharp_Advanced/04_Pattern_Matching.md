# Pattern Matching

**Previous**: [LINQ](./03_LINQ.md) | **Next**: [Nullable Reference Types](./05_Nullable_Reference_Types.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use type patterns to safely cast and test types in one expression
2. Apply property patterns to match on object structure
3. Deconstruct objects with positional patterns
4. Write relational and logical patterns to express ranges and conditions
5. Use list patterns to match sequences by shape
6. Combine patterns in switch expressions for concise, exhaustive branching
7. Build a practical expression evaluator using nested pattern matching

---

Pattern matching in C# lets you test a value against a shape — its type, structure, properties, or position — and extract data in a single, declarative expression. Introduced gradually from C# 7 through C# 12, pattern matching has evolved into a comprehensive feature that can replace many verbose `if-else` chains and `switch` statements with clear, concise, and compiler-verified code.

## 1. Type Patterns

Type patterns test whether a value is of a given type and optionally bind it to a variable.

### 1.1 The `is` Expression with Type Pattern

```csharp
object value = "Hello, World!";

// Type pattern with variable binding
if (value is string text)
{
    Console.WriteLine($"String of length {text.Length}"); // String of length 13
}

// Negated type pattern
if (value is not int)
{
    Console.WriteLine("Not an integer");
}

// Type pattern in a method
static string Describe(object obj)
{
    if (obj is int n)
        return $"Integer: {n}";
    if (obj is double d)
        return $"Double: {d:F2}";
    if (obj is string s)
        return $"String: \"{s}\"";
    if (obj is null)
        return "Null";
    return $"Unknown: {obj.GetType().Name}";
}

Console.WriteLine(Describe(42));        // Integer: 42
Console.WriteLine(Describe(3.14));      // Double: 3.14
Console.WriteLine(Describe("test"));    // String: "test"
Console.WriteLine(Describe(null!));     // Null
```

### 1.2 Type Pattern in Switch

```csharp
static double CalculateArea(object shape) => shape switch
{
    Circle c => Math.PI * c.Radius * c.Radius,
    Rectangle r => r.Width * r.Height,
    Triangle t => 0.5 * t.Base * t.Height,
    _ => throw new ArgumentException($"Unknown shape: {shape.GetType().Name}")
};

record Circle(double Radius);
record Rectangle(double Width, double Height);
record Triangle(double Base, double Height);

Console.WriteLine(CalculateArea(new Circle(5)));          // 78.54
Console.WriteLine(CalculateArea(new Rectangle(4, 6)));    // 24
Console.WriteLine(CalculateArea(new Triangle(3, 8)));     // 12
```

## 2. Constant Patterns

Constant patterns match a value against a compile-time constant, including `null`.

```csharp
static string ClassifyHttpStatus(int statusCode) => statusCode switch
{
    200 => "OK",
    201 => "Created",
    204 => "No Content",
    301 => "Moved Permanently",
    400 => "Bad Request",
    401 => "Unauthorized",
    403 => "Forbidden",
    404 => "Not Found",
    500 => "Internal Server Error",
    503 => "Service Unavailable",
    _ => $"Unknown ({statusCode})"
};

Console.WriteLine(ClassifyHttpStatus(200)); // OK
Console.WriteLine(ClassifyHttpStatus(404)); // Not Found
Console.WriteLine(ClassifyHttpStatus(418)); // Unknown (418)

// Null constant pattern
static string SafeToUpper(string? input) => input switch
{
    null => "(null)",
    "" => "(empty)",
    _ => input.ToUpper()
};
```

## 3. Property Patterns

Property patterns match on the values of an object's properties or fields. They use curly braces `{ }` to specify property conditions.

### 3.1 Basic Property Patterns

```csharp
record Address(string City, string State, string ZipCode);
record Person(string Name, int Age, Address Address);

static string DescribePerson(Person p) => p switch
{
    { Age: < 13 } => "Child",
    { Age: < 18 } => "Teenager",
    { Age: >= 65 } => "Senior",
    { Name: "Alice" } => "It's Alice!",
    _ => "Adult"
};

var alice = new Person("Alice", 30, new Address("Seattle", "WA", "98101"));
Console.WriteLine(DescribePerson(alice)); // It's Alice!

var kid = new Person("Timmy", 8, new Address("Portland", "OR", "97201"));
Console.WriteLine(DescribePerson(kid)); // Child
```

### 3.2 Nested Property Patterns

```csharp
static decimal CalculateShipping(Person customer) => customer switch
{
    { Address: { State: "WA" } } => 0.00m,       // Free shipping in Washington
    { Address: { State: "OR" or "CA" } } => 5.99m, // West coast
    { Address: { State: "NY" or "NJ" } } => 7.99m, // East coast
    { Age: >= 65 } => 2.99m,                        // Senior discount
    _ => 9.99m                                       // Standard
};

Console.WriteLine(CalculateShipping(alice)); // 0.00 (WA)

// Shorthand nested access (C# 10+) — dot notation
static string GetRegion(Person p) => p switch
{
    { Address.State: "WA" or "OR" or "CA" } => "West Coast",
    { Address.State: "NY" or "NJ" or "CT" } => "East Coast",
    { Address.State: "TX" or "FL" } => "South",
    _ => "Other"
};
```

### 3.3 Property Pattern with Variable Binding

```csharp
static string FormatAddress(Person p) => p switch
{
    { Name: var name, Address: { City: var city, State: "WA" } }
        => $"{name} lives in {city}, Washington",
    { Name: var name, Address: { City: var city, State: var state } }
        => $"{name} lives in {city}, {state}",
};

Console.WriteLine(FormatAddress(alice)); // Alice lives in Seattle, Washington
```

## 4. Positional Patterns

Positional patterns use deconstruction to match on the components of a type that has a `Deconstruct` method (or is a positional record/tuple).

### 4.1 Tuple Patterns

```csharp
static string ClassifyPoint(int x, int y) => (x, y) switch
{
    (0, 0) => "Origin",
    (> 0, > 0) => "Quadrant I",
    (< 0, > 0) => "Quadrant II",
    (< 0, < 0) => "Quadrant III",
    (> 0, < 0) => "Quadrant IV",
    (0, _) => "Y-axis",
    (_, 0) => "X-axis"
};

Console.WriteLine(ClassifyPoint(3, 4));   // Quadrant I
Console.WriteLine(ClassifyPoint(-1, 5));  // Quadrant II
Console.WriteLine(ClassifyPoint(0, 0));   // Origin
Console.WriteLine(ClassifyPoint(0, -7));  // Y-axis
```

### 4.2 Positional Records

```csharp
record Point3D(double X, double Y, double Z);

static string Classify(Point3D p) => p switch
{
    (0, 0, 0) => "Origin",
    (_, 0, 0) => "On X-axis",
    (0, _, 0) => "On Y-axis",
    (0, 0, _) => "On Z-axis",
    (var x, var y, 0) => $"XY-plane at ({x}, {y})",
    (var x, var y, var z) => $"3D point ({x}, {y}, {z})"
};

Console.WriteLine(Classify(new Point3D(0, 0, 0)));   // Origin
Console.WriteLine(Classify(new Point3D(5, 0, 0)));   // On X-axis
Console.WriteLine(Classify(new Point3D(3, 4, 0)));   // XY-plane at (3, 4)
Console.WriteLine(Classify(new Point3D(1, 2, 3)));   // 3D point (1, 2, 3)
```

### 4.3 Custom Deconstruct Methods

```csharp
public class Temperature
{
    public double Celsius { get; }
    public Temperature(double celsius) => Celsius = celsius;

    public void Deconstruct(out double celsius, out double fahrenheit)
    {
        celsius = Celsius;
        fahrenheit = Celsius * 9 / 5 + 32;
    }
}

static string DescribeTemp(Temperature t) => t switch
{
    ( < -40, _) => "Extreme cold",
    ( < 0, _) => "Below freezing",
    ( < 20, _) => "Cool",
    ( < 30, _) => "Comfortable",
    ( < 40, _) => "Hot",
    _ => "Extreme heat"
};

Console.WriteLine(DescribeTemp(new Temperature(-50)));  // Extreme cold
Console.WriteLine(DescribeTemp(new Temperature(22)));   // Comfortable
Console.WriteLine(DescribeTemp(new Temperature(45)));   // Extreme heat
```

## 5. Relational Patterns

Relational patterns compare a value against a constant using `<`, `>`, `<=`, or `>=`.

```csharp
static string GradeFromScore(int score) => score switch
{
    >= 97 => "A+",
    >= 93 => "A",
    >= 90 => "A-",
    >= 87 => "B+",
    >= 83 => "B",
    >= 80 => "B-",
    >= 77 => "C+",
    >= 73 => "C",
    >= 70 => "C-",
    >= 67 => "D+",
    >= 63 => "D",
    >= 60 => "D-",
    _ => "F"
};

Console.WriteLine(GradeFromScore(95)); // A
Console.WriteLine(GradeFromScore(82)); // B-
Console.WriteLine(GradeFromScore(55)); // F
```

## 6. Logical Patterns

Logical patterns combine other patterns using `and`, `or`, and `not`.

### 6.1 And / Or Patterns

```csharp
static string ClassifyTemperature(double temp) => temp switch
{
    >= -10 and < 0 => "Cold",
    >= 0 and < 15 => "Cool",
    >= 15 and < 25 => "Comfortable",
    >= 25 and < 35 => "Warm",
    >= 35 and < 45 => "Hot",
    _ => "Extreme"
};

Console.WriteLine(ClassifyTemperature(22));  // Comfortable
Console.WriteLine(ClassifyTemperature(-5));  // Cold
Console.WriteLine(ClassifyTemperature(50));  // Extreme

// Or pattern — match multiple values
static bool IsWeekend(DayOfWeek day) => day is DayOfWeek.Saturday or DayOfWeek.Sunday;

static bool IsVowel(char c) => char.ToLower(c) is 'a' or 'e' or 'i' or 'o' or 'u';
```

### 6.2 Not Pattern

```csharp
// Cleaner than !=
if (value is not null)
{
    Console.WriteLine(value);
}

// Not combined with type pattern
if (shape is not Circle)
{
    Console.WriteLine("Not a circle");
}

// Not in switch
static string Validate(int age) => age switch
{
    not (>= 0 and <= 150) => "Invalid age",
    < 18 => "Minor",
    >= 65 => "Senior",
    _ => "Adult"
};
```

### 6.3 Complex Logical Combinations

```csharp
static string ClassifyCharacter(char c) => c switch
{
    >= 'a' and <= 'z' => "lowercase letter",
    >= 'A' and <= 'Z' => "uppercase letter",
    >= '0' and <= '9' => "digit",
    ' ' or '\t' or '\n' or '\r' => "whitespace",
    '.' or ',' or ';' or ':' or '!' or '?' => "punctuation",
    _ => "other"
};

Console.WriteLine(ClassifyCharacter('g')); // lowercase letter
Console.WriteLine(ClassifyCharacter('5')); // digit
Console.WriteLine(ClassifyCharacter('!')); // punctuation
```

## 7. List Patterns (C# 11)

List patterns match sequences (arrays, lists, spans) by their shape — specific elements, length, and slices.

### 7.1 Basic List Patterns

```csharp
int[] numbers = { 1, 2, 3, 4, 5 };

// Exact match
bool isOneTwoThree = numbers is [1, 2, 3]; // false (5 elements, not 3)

// Match any 5-element array starting with 1 and ending with 5
bool matchShape = numbers is [1, _, _, _, 5]; // true

// Discard pattern for "any element"
bool startsWithOne = numbers is [1, ..]; // true

// Slice pattern (..) matches zero or more elements
bool endsWithFive = numbers is [.., 5]; // true

// Empty array
int[] empty = Array.Empty<int>();
bool isEmpty = empty is []; // true
```

### 7.2 Variable Binding in List Patterns

```csharp
static string DescribeArray(int[] arr) => arr switch
{
    [] => "Empty",
    [var single] => $"Single: {single}",
    [var first, var second] => $"Pair: {first}, {second}",
    [var first, .., var last] => $"Array from {first} to {last} ({arr.Length} elements)",
};

Console.WriteLine(DescribeArray(Array.Empty<int>()));     // Empty
Console.WriteLine(DescribeArray(new[] { 42 }));           // Single: 42
Console.WriteLine(DescribeArray(new[] { 1, 2 }));         // Pair: 1, 2
Console.WriteLine(DescribeArray(new[] { 1, 2, 3, 4, 5 })); // Array from 1 to 5 (5 elements)
```

### 7.3 Nested Patterns in Lists

```csharp
static string AnalyzeSequence(int[] seq) => seq switch
{
    [> 0, > 0, > 0] => "Three positive numbers",
    [< 0, .., < 0] => "Starts and ends negative",
    [0, ..] => "Starts with zero",
    [_, _, _, ..] when seq.All(x => x % 2 == 0) => "3+ even numbers",
    _ => "Other"
};

Console.WriteLine(AnalyzeSequence(new[] { 1, 2, 3 }));      // Three positive numbers
Console.WriteLine(AnalyzeSequence(new[] { -1, 5, -3 }));     // Starts and ends negative
Console.WriteLine(AnalyzeSequence(new[] { 0, 7, 8 }));       // Starts with zero
```

### 7.4 Slice Patterns with Variable Capture

```csharp
// Capture the slice into a variable (C# 11)
static string ProcessCommand(string[] args) => args switch
{
    ["help"] => "Showing help...",
    ["version"] => "v1.0.0",
    ["add", var item] => $"Adding: {item}",
    ["remove", var item] => $"Removing: {item}",
    ["search", .. var terms] => $"Searching for: {string.Join(" ", terms)}",
    [var cmd, ..] => $"Unknown command: {cmd}",
    [] => "No command specified"
};

Console.WriteLine(ProcessCommand(new[] { "help" }));
// Showing help...
Console.WriteLine(ProcessCommand(new[] { "add", "milk" }));
// Adding: milk
Console.WriteLine(ProcessCommand(new[] { "search", "C#", "patterns" }));
// Searching for: C# patterns
```

## 8. Var and Discard Patterns

### 8.1 Var Pattern

The `var` pattern always matches and binds the value to a new variable. It is useful inside complex patterns where you want to capture an intermediate value.

```csharp
static string AnalyzeOrder(decimal amount, string country) => (amount, country) switch
{
    ( > 1000, "US") => "Large US order — free shipping",
    ( > 500, "US") => "Medium US order — reduced shipping",
    (var a, "US") when a > 0 => $"Small US order (${a})",
    (var a, var c) when a > 0 => $"International order to {c} (${a})",
    _ => "Invalid order"
};
```

### 8.2 Discard Pattern

The `_` discard pattern matches anything and discards the value. It serves as a "catch-all" or placeholder.

```csharp
// In switch expression — default arm
var result = input switch
{
    1 => "one",
    2 => "two",
    _ => "other"  // discard — matches everything
};

// In positional pattern — ignore a component
static bool IsOnAxis(Point3D p) => p is (_, 0, 0) or (0, _, 0) or (0, 0, _);
```

## 9. Switch Expressions — Putting It All Together

Switch expressions are the most powerful way to use patterns. They are expressions (not statements), which means they return a value.

### 9.1 Exhaustiveness

The compiler checks that switch expressions are exhaustive — all possible inputs are handled. A warning is produced if coverage is incomplete.

```csharp
// Enum — compiler knows all possible values
enum Season { Spring, Summer, Autumn, Winter }

static string Describe(Season s) => s switch
{
    Season.Spring => "Flowers blooming",
    Season.Summer => "Sun shining",
    Season.Autumn => "Leaves falling",
    Season.Winter => "Snow falling",
    // No _ needed — all enum values covered
};
```

### 9.2 Guards with `when`

Add a `when` clause to refine a pattern with an arbitrary boolean condition.

```csharp
record Order(string Product, int Quantity, decimal UnitPrice)
{
    public decimal Total => Quantity * UnitPrice;
}

static string ClassifyOrder(Order order) => order switch
{
    { Quantity: <= 0 } => "Invalid: zero or negative quantity",
    { UnitPrice: <= 0 } => "Invalid: zero or negative price",
    { Total: > 10_000 } when order.Product.StartsWith("PREMIUM")
        => "VIP premium order — assign dedicated rep",
    { Total: > 10_000 } => "Large order — notify sales",
    { Total: > 1_000 } => "Medium order",
    _ => "Standard order"
};
```

### 9.3 Nested Pattern Combinations

```csharp
record Customer(string Name, string Tier, Address Address);

static decimal CalculateDiscount(Customer c, Order o) => (c, o) switch
{
    ({ Tier: "Gold" }, { Total: > 5000 }) => 0.20m,
    ({ Tier: "Gold" }, _) => 0.15m,
    ({ Tier: "Silver" }, { Total: > 5000 }) => 0.12m,
    ({ Tier: "Silver" }, _) => 0.08m,
    ({ Address.State: "WA" }, { Total: > 1000 }) => 0.05m, // local loyalty
    _ => 0.0m
};
```

## 10. Practical Example: Expression Evaluator

Let's build a simple arithmetic expression evaluator using pattern matching on a recursive data structure.

### 10.1 The Expression Tree

```csharp
public abstract record Expr;
public record Num(double Value) : Expr;
public record Add(Expr Left, Expr Right) : Expr;
public record Sub(Expr Left, Expr Right) : Expr;
public record Mul(Expr Left, Expr Right) : Expr;
public record Div(Expr Left, Expr Right) : Expr;
public record Neg(Expr Operand) : Expr;
public record Var(string Name) : Expr;
```

### 10.2 The Evaluator

```csharp
public static class ExprEvaluator
{
    public static double Evaluate(Expr expr, Dictionary<string, double>? vars = null) =>
        expr switch
        {
            Num(var v) => v,
            Add(var l, var r) => Evaluate(l, vars) + Evaluate(r, vars),
            Sub(var l, var r) => Evaluate(l, vars) - Evaluate(r, vars),
            Mul(var l, var r) => Evaluate(l, vars) * Evaluate(r, vars),
            Div(var l, Num(0)) => throw new DivideByZeroException(),
            Div(var l, var r) => Evaluate(l, vars) / Evaluate(r, vars),
            Neg(var operand) => -Evaluate(operand, vars),
            Var(var name) when vars?.ContainsKey(name) == true => vars[name],
            Var(var name) => throw new ArgumentException($"Undefined variable: {name}"),
            _ => throw new ArgumentException($"Unknown expression: {expr}")
        };

    public static string PrettyPrint(Expr expr) => expr switch
    {
        Num(var v) => v.ToString("G"),
        Var(var name) => name,
        Neg(var op) => $"-({PrettyPrint(op)})",
        Add(var l, var r) => $"({PrettyPrint(l)} + {PrettyPrint(r)})",
        Sub(var l, var r) => $"({PrettyPrint(l)} - {PrettyPrint(r)})",
        Mul(var l, var r) => $"({PrettyPrint(l)} * {PrettyPrint(r)})",
        Div(var l, var r) => $"({PrettyPrint(l)} / {PrettyPrint(r)})",
        _ => "?"
    };

    // Simplifier using pattern matching
    public static Expr Simplify(Expr expr) => expr switch
    {
        // x + 0 = x, 0 + x = x
        Add(var x, Num(0)) => Simplify(x),
        Add(Num(0), var x) => Simplify(x),

        // x * 1 = x, 1 * x = x
        Mul(var x, Num(1)) => Simplify(x),
        Mul(Num(1), var x) => Simplify(x),

        // x * 0 = 0, 0 * x = 0
        Mul(_, Num(0)) => new Num(0),
        Mul(Num(0), _) => new Num(0),

        // x - 0 = x
        Sub(var x, Num(0)) => Simplify(x),

        // --x = x
        Neg(Neg(var x)) => Simplify(x),

        // Constant folding
        Add(Num(var a), Num(var b)) => new Num(a + b),
        Sub(Num(var a), Num(var b)) => new Num(a - b),
        Mul(Num(var a), Num(var b)) => new Num(a * b),
        Div(Num(var a), Num(var b)) when b != 0 => new Num(a / b),

        // Recurse into subexpressions
        Add(var l, var r) => new Add(Simplify(l), Simplify(r)),
        Sub(var l, var r) => new Sub(Simplify(l), Simplify(r)),
        Mul(var l, var r) => new Mul(Simplify(l), Simplify(r)),
        Div(var l, var r) => new Div(Simplify(l), Simplify(r)),
        Neg(var op) => new Neg(Simplify(op)),

        _ => expr
    };
}
```

### 10.3 Using the Evaluator

```csharp
// Build: (3 + x) * (2 - 1)
Expr expr = new Mul(
    new Add(new Num(3), new Var("x")),
    new Sub(new Num(2), new Num(1))
);

Console.WriteLine(ExprEvaluator.PrettyPrint(expr));
// ((3 + x) * (2 - 1))

var vars = new Dictionary<string, double> { ["x"] = 7 };
Console.WriteLine(ExprEvaluator.Evaluate(expr, vars)); // 10

// Simplify: (2 - 1) becomes Num(1), then (3 + x) * 1 becomes (3 + x)
var simplified = ExprEvaluator.Simplify(expr);
Console.WriteLine(ExprEvaluator.PrettyPrint(simplified)); // (3 + x)

// Another simplification: 0 + (5 * 1) -> 5
Expr expr2 = new Add(new Num(0), new Mul(new Num(5), new Num(1)));
var s2 = ExprEvaluator.Simplify(expr2);
Console.WriteLine(ExprEvaluator.PrettyPrint(s2)); // 5
```

## 11. Practice Problems

1. **FizzBuzz with Patterns**: Rewrite FizzBuzz (1 to 100) using a single switch expression on a tuple `(n % 3, n % 5)` with constant patterns. Print "Fizz", "Buzz", "FizzBuzz", or the number.

2. **JSON-like Value Classifier**: Define a discriminated union of JSON value types: `JsonNull`, `JsonBool(bool Value)`, `JsonNumber(double Value)`, `JsonString(string Value)`, `JsonArray(List<JsonValue> Items)`, `JsonObject(Dictionary<string, JsonValue> Properties)`. Write a `Describe(JsonValue v)` method using pattern matching that returns a human-readable description including nested depth for arrays and objects.

3. **Command Parser with List Patterns**: Parse a `string[]` command line using list patterns. Support: `["git", "commit", "-m", var message]`, `["git", "push", var remote, var branch]`, `["git", "log", "--oneline", "-n", var count]`, and `["git", .. var rest]` as catch-all. Return a record describing each parsed command.

4. **Temperature Converter**: Write a method that accepts a `(double Value, string Unit)` tuple and converts between Celsius, Fahrenheit, and Kelvin using positional + constant patterns. Handle invalid units with a clear error message.

5. **Binary Tree Depth**: Define a recursive `record Tree` with `Leaf(int Value)` and `Branch(Tree Left, Tree Right)`. Write `MaxDepth(Tree t)` and `Sum(Tree t)` using pattern matching. Then write `Mirror(Tree t)` that returns a mirror image of the tree.
