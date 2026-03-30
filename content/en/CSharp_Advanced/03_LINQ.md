# LINQ

**Previous**: [Lambda Expressions and Closures](./02_Lambda_and_Closures.md) | **Next**: [Pattern Matching](./04_Pattern_Matching.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write LINQ queries using both query syntax and method syntax
2. Explain deferred execution and know when to materialize queries
3. Apply filtering, projection, ordering, grouping, and joining operators
4. Use aggregate, set, and element operators effectively
5. Flatten nested collections with `SelectMany`
6. Write custom LINQ extension methods
7. Reason about LINQ performance and avoid common pitfalls

---

LINQ (Language Integrated Query) is one of C#'s most powerful features. It provides a unified, declarative syntax for querying and transforming data from any source — in-memory collections, databases, XML, JSON, and more. Rather than writing imperative loops to filter, sort, and group data, LINQ lets you describe *what* you want in a readable, composable pipeline.

## 1. LINQ Overview and Philosophy

### 1.1 What Is LINQ?

LINQ integrates query capabilities directly into the C# language. At its core, LINQ operates on any type that implements `IEnumerable<T>` (for in-memory data) or `IQueryable<T>` (for external data sources).

```csharp
// Imperative approach — HOW to get the result
var results = new List<string>();
foreach (var name in names)
{
    if (name.Length > 3)
    {
        results.Add(name.ToUpper());
    }
}
results.Sort();

// LINQ approach — WHAT result we want
var results = names
    .Where(name => name.Length > 3)
    .Select(name => name.ToUpper())
    .OrderBy(name => name)
    .ToList();
```

### 1.2 LINQ Providers

LINQ is not limited to in-memory collections. Different providers translate LINQ expressions to different targets:

```csharp
// LINQ to Objects — in-memory IEnumerable<T>
var inMemory = numbers.Where(n => n > 10).ToList();

// LINQ to Entities (EF Core) — translates to SQL
var fromDb = dbContext.Products
    .Where(p => p.Price > 10)
    .OrderBy(p => p.Name)
    .ToListAsync();

// LINQ to XML
var elements = xdoc.Descendants("item")
    .Where(e => (int)e.Attribute("qty")! > 5)
    .Select(e => e.Value);
```

## 2. Query Syntax

Query syntax uses SQL-like keywords built into the C# language. The compiler transforms them into method calls.

### 2.1 Basic Query Syntax

```csharp
int[] numbers = { 5, 12, 8, 3, 17, 1, 25, 9, 14 };

// Basic query: filter + order + project
var query = from n in numbers
            where n > 5
            orderby n descending
            select n * 2;

foreach (var item in query)
    Console.Write($"{item} "); // 50 34 28 24 18 16
Console.WriteLine();
```

### 2.2 Query with Multiple Sources

```csharp
string[] colors = { "red", "green", "blue" };
string[] sizes = { "S", "M", "L" };

// Cross join (Cartesian product)
var combinations = from color in colors
                   from size in sizes
                   select $"{color}-{size}";

Console.WriteLine(string.Join(", ", combinations));
// red-S, red-M, red-L, green-S, green-M, green-L, blue-S, blue-M, blue-L
```

### 2.3 Let Clause for Intermediate Values

```csharp
string[] words = { "Hello", "Beautiful", "World", "LINQ", "Rocks" };

var query = from w in words
            let lower = w.ToLower()
            let len = w.Length
            where len > 4
            orderby len descending
            select new { Word = lower, Length = len };

foreach (var item in query)
    Console.WriteLine($"{item.Word} ({item.Length})");
// beautiful (9)
// hello (5)
// world (5)
// rocks (5)
```

### 2.4 Group By

```csharp
var students = new[]
{
    new { Name = "Alice", Grade = "A" },
    new { Name = "Bob", Grade = "B" },
    new { Name = "Charlie", Grade = "A" },
    new { Name = "Diana", Grade = "C" },
    new { Name = "Eve", Grade = "B" },
    new { Name = "Frank", Grade = "A" },
};

var grouped = from s in students
              group s by s.Grade into gradeGroup
              orderby gradeGroup.Key
              select new
              {
                  Grade = gradeGroup.Key,
                  Count = gradeGroup.Count(),
                  Students = string.Join(", ", gradeGroup.Select(g => g.Name))
              };

foreach (var g in grouped)
    Console.WriteLine($"Grade {g.Grade}: {g.Count} students ({g.Students})");
// Grade A: 3 students (Alice, Charlie, Frank)
// Grade B: 2 students (Bob, Eve)
// Grade C: 1 students (Diana)
```

### 2.5 Join

```csharp
var departments = new[]
{
    new { Id = 1, Name = "Engineering" },
    new { Id = 2, Name = "Marketing" },
    new { Id = 3, Name = "Finance" },
};

var employees = new[]
{
    new { Name = "Alice", DeptId = 1 },
    new { Name = "Bob", DeptId = 2 },
    new { Name = "Charlie", DeptId = 1 },
    new { Name = "Diana", DeptId = 3 },
    new { Name = "Eve", DeptId = 2 },
};

var query = from e in employees
            join d in departments on e.DeptId equals d.Id
            select new { e.Name, Department = d.Name };

foreach (var item in query)
    Console.WriteLine($"{item.Name} -> {item.Department}");
// Alice -> Engineering
// Bob -> Marketing
// Charlie -> Engineering
// Diana -> Finance
// Eve -> Marketing
```

## 3. Method Syntax

Method syntax (also called fluent syntax) calls LINQ extension methods directly. It is more flexible than query syntax and supports all LINQ operators.

### 3.1 Basic Method Syntax

```csharp
int[] numbers = { 5, 12, 8, 3, 17, 1, 25, 9, 14 };

var result = numbers
    .Where(n => n > 5)
    .OrderByDescending(n => n)
    .Select(n => n * 2);

Console.WriteLine(string.Join(", ", result)); // 50, 34, 28, 24, 18, 16
```

### 3.2 Method Syntax Equivalents

```csharp
// GroupBy
var grouped = employees
    .GroupBy(e => e.DeptId)
    .Select(g => new { DeptId = g.Key, Count = g.Count() });

// Join
var joined = employees
    .Join(
        departments,
        e => e.DeptId,       // outer key
        d => d.Id,           // inner key
        (e, d) => new { e.Name, Department = d.Name }); // result

// GroupJoin (left outer join equivalent)
var deptWithEmployees = departments
    .GroupJoin(
        employees,
        d => d.Id,
        e => e.DeptId,
        (dept, emps) => new
        {
            Department = dept.Name,
            Employees = emps.Select(e => e.Name).ToList()
        });

foreach (var d in deptWithEmployees)
    Console.WriteLine($"{d.Department}: {string.Join(", ", d.Employees)}");
```

## 4. Query vs Method Syntax Comparison

Most queries can be written in either style. The compiler transforms query syntax into method calls — they produce identical IL.

```csharp
var data = new[]
{
    new { Name = "Widget", Category = "A", Price = 25.0 },
    new { Name = "Gadget", Category = "B", Price = 45.0 },
    new { Name = "Doohickey", Category = "A", Price = 15.0 },
    new { Name = "Thingamajig", Category = "B", Price = 80.0 },
    new { Name = "Whatchamacallit", Category = "A", Price = 35.0 },
};

// Query syntax
var q1 = from item in data
         where item.Price > 20
         group item by item.Category into catGroup
         select new { Category = catGroup.Key, Avg = catGroup.Average(x => x.Price) };

// Method syntax — identical result
var q2 = data
    .Where(item => item.Price > 20)
    .GroupBy(item => item.Category)
    .Select(g => new { Category = g.Key, Avg = g.Average(x => x.Price) });

// Guidelines:
// - Query syntax is often cleaner for joins and group operations
// - Method syntax is required for operators without query keywords (Take, Skip, Distinct, etc.)
// - Mix both when it improves readability
```

## 5. Deferred Execution and Materialization

One of the most important concepts in LINQ is deferred execution. A LINQ query is not executed when it is defined — it is executed when you iterate over it.

### 5.1 Demonstrating Deferred Execution

```csharp
var numbers = new List<int> { 1, 2, 3, 4, 5 };

// Define the query — no execution yet
var query = numbers.Where(n => n > 2);

// Modify the source AFTER defining the query
numbers.Add(6);
numbers.Add(7);

// NOW execute by iterating — includes 6 and 7
Console.WriteLine(string.Join(", ", query)); // 3, 4, 5, 6, 7
```

### 5.2 Side Effects Reveal Deferred Execution

```csharp
var names = new[] { "Alice", "Bob", "Charlie" };

var query = names.Where(n =>
{
    Console.WriteLine($"  Evaluating: {n}");
    return n.Length > 3;
});

Console.WriteLine("Query defined. No evaluation yet.");
Console.WriteLine("Now iterating:");

foreach (var name in query) // evaluation happens here
    Console.WriteLine($"  Result: {name}");

// Output:
// Query defined. No evaluation yet.
// Now iterating:
//   Evaluating: Alice
//   Result: Alice
//   Evaluating: Bob
//   Evaluating: Charlie
//   Result: Charlie
```

### 5.3 Materialization Operators

Materialization operators force immediate execution and store results in memory.

```csharp
var numbers = Enumerable.Range(1, 10);

// These execute immediately and store results
List<int> list = numbers.Where(n => n > 5).ToList();
int[] array = numbers.Where(n => n > 5).ToArray();
Dictionary<int, int> dict = numbers.ToDictionary(n => n, n => n * n);
HashSet<int> set = numbers.ToHashSet();

// Count, Sum, etc. also trigger immediate execution
int count = numbers.Count(n => n > 5); // 5
int sum = numbers.Sum(); // 55
```

### 5.4 Multiple Enumeration Warning

Because deferred queries re-execute on each iteration, enumerating a query multiple times runs the pipeline multiple times. This can cause bugs or performance issues.

```csharp
IEnumerable<int> ExpensiveQuery()
{
    Console.WriteLine("  [Executing expensive query]");
    return Enumerable.Range(1, 5).Select(x => x * x);
}

var query = ExpensiveQuery();

// Enumerating twice runs the query twice
Console.WriteLine("First: " + string.Join(", ", query));
Console.WriteLine("Second: " + string.Join(", ", query));
// Output:
//   [Executing expensive query]
// First: 1, 4, 9, 16, 25
//   [Executing expensive query]
// Second: 1, 4, 9, 16, 25

// FIX: materialize if you need to enumerate more than once
var materialized = ExpensiveQuery().ToList();
Console.WriteLine("First: " + string.Join(", ", materialized));  // no re-execution
Console.WriteLine("Second: " + string.Join(", ", materialized)); // no re-execution
```

## 6. Common Operators

### 6.1 Filtering

```csharp
var people = new[]
{
    new { Name = "Alice", Age = 30 },
    new { Name = "Bob", Age = 25 },
    new { Name = "Charlie", Age = 35 },
    new { Name = "Diana", Age = 28 },
    new { Name = "Eve", Age = 30 },
};

// Where — filter by predicate
var adults = people.Where(p => p.Age >= 30);

// OfType — filter by type
object[] mixed = { 1, "hello", 2, "world", 3.14, 4 };
var integers = mixed.OfType<int>(); // 1, 2, 4

// Distinct
int[] dupes = { 1, 2, 2, 3, 3, 3, 4 };
var unique = dupes.Distinct(); // 1, 2, 3, 4

// DistinctBy (C# 10 / .NET 6+)
var uniqueByAge = people.DistinctBy(p => p.Age);
```

### 6.2 Projection

```csharp
var numbers = Enumerable.Range(1, 5);

// Select — transform each element
var squares = numbers.Select(n => n * n); // 1, 4, 9, 16, 25

// Select with index
var indexed = numbers.Select((n, i) => $"[{i}]={n}");
// [0]=1, [1]=2, [2]=3, [3]=4, [4]=5

// Anonymous type projection
var projected = people.Select(p => new { p.Name, AgeGroup = p.Age >= 30 ? "Senior" : "Junior" });
```

### 6.3 Ordering

```csharp
// OrderBy / OrderByDescending
var byAge = people.OrderBy(p => p.Age);
var byAgeDesc = people.OrderByDescending(p => p.Age);

// ThenBy — secondary sort
var sorted = people
    .OrderBy(p => p.Age)
    .ThenBy(p => p.Name);
// (Bob,25), (Diana,28), (Alice,30), (Eve,30), (Charlie,35)

// Reverse
var reversed = people.Reverse();
```

### 6.4 Grouping

```csharp
var products = new[]
{
    new { Name = "Laptop", Category = "Electronics", Price = 999m },
    new { Name = "Phone", Category = "Electronics", Price = 699m },
    new { Name = "Desk", Category = "Furniture", Price = 350m },
    new { Name = "Chair", Category = "Furniture", Price = 250m },
    new { Name = "Tablet", Category = "Electronics", Price = 449m },
};

var groups = products.GroupBy(p => p.Category);

foreach (var group in groups)
{
    Console.WriteLine($"{group.Key}:");
    foreach (var product in group)
        Console.WriteLine($"  {product.Name}: ${product.Price}");
    Console.WriteLine($"  Average: ${group.Average(p => p.Price):F2}");
}
// Electronics:
//   Laptop: $999
//   Phone: $699
//   Tablet: $449
//   Average: $715.67
// Furniture:
//   Desk: $350
//   Chair: $250
//   Average: $300.00
```

## 7. Aggregate Operators

```csharp
var numbers = new[] { 10, 20, 30, 40, 50 };

Console.WriteLine(numbers.Count());            // 5
Console.WriteLine(numbers.Sum());              // 150
Console.WriteLine(numbers.Average());          // 30
Console.WriteLine(numbers.Min());              // 10
Console.WriteLine(numbers.Max());              // 50
Console.WriteLine(numbers.MinBy(n => Math.Abs(n - 25))); // 20 (closest to 25)
Console.WriteLine(numbers.MaxBy(n => Math.Abs(n - 25))); // 50 (farthest from 25)

// Aggregate — custom accumulation (fold/reduce)
// Sum via Aggregate:
int sum = numbers.Aggregate((acc, n) => acc + n); // 150

// Aggregate with seed — build a sentence
string sentence = numbers.Aggregate(
    "Numbers:",                           // seed
    (acc, n) => $"{acc} {n}",             // accumulator
    result => result + ".");               // result selector
Console.WriteLine(sentence); // Numbers: 10 20 30 40 50.

// Running product
long product = new[] { 2, 3, 4, 5 }.Aggregate(1L, (acc, n) => acc * n);
Console.WriteLine(product); // 120
```

## 8. Set Operators

```csharp
var a = new[] { 1, 2, 3, 4, 5 };
var b = new[] { 3, 4, 5, 6, 7 };

// Distinct — unique elements
Console.WriteLine(string.Join(", ", a.Concat(b).Distinct()));
// 1, 2, 3, 4, 5, 6, 7

// Union — set union (distinct elements from both)
Console.WriteLine(string.Join(", ", a.Union(b)));
// 1, 2, 3, 4, 5, 6, 7

// Intersect — elements in both
Console.WriteLine(string.Join(", ", a.Intersect(b)));
// 3, 4, 5

// Except — elements in a but not in b
Console.WriteLine(string.Join(", ", a.Except(b)));
// 1, 2

// ExceptBy / IntersectBy / UnionBy (.NET 6+)
var inventory = new[] { ("Widget", 5), ("Gadget", 10), ("Doohickey", 3) };
var discontinued = new[] { "Gadget", "Thingamajig" };

var active = inventory.ExceptBy(discontinued, item => item.Item1);
foreach (var (name, qty) in active)
    Console.WriteLine($"{name}: {qty}");
// Widget: 5
// Doohickey: 3
```

## 9. Element Operators

```csharp
var names = new[] { "Alice", "Bob", "Charlie", "Diana" };

// First / FirstOrDefault
Console.WriteLine(names.First());                       // Alice
Console.WriteLine(names.First(n => n.StartsWith("C"))); // Charlie
Console.WriteLine(names.FirstOrDefault(n => n.StartsWith("Z")) ?? "none"); // none

// Last / LastOrDefault
Console.WriteLine(names.Last()); // Diana

// Single — exactly one element (throws if 0 or >1)
var single = new[] { 42 };
Console.WriteLine(single.Single()); // 42
// names.Single(); // throws — more than one element

// SingleOrDefault — 0 or 1 element
Console.WriteLine(names.SingleOrDefault(n => n == "Bob")); // Bob

// ElementAt / ElementAtOrDefault
Console.WriteLine(names.ElementAt(2));              // Charlie
Console.WriteLine(names.ElementAtOrDefault(99));    // null (default for string)

// C# 12 — Index/Range support
Console.WriteLine(names.ElementAt(^1)); // Diana (last element)
```

## 10. SelectMany — Flattening Nested Collections

`SelectMany` is one of the most powerful LINQ operators. It projects each element to a collection and then flattens all collections into a single sequence.

### 10.1 Basic Flattening

```csharp
var sentences = new[]
{
    "The quick brown fox",
    "jumps over the lazy dog",
    "and runs away fast"
};

// Split each sentence into words and flatten
var words = sentences.SelectMany(s => s.Split(' '));
Console.WriteLine(string.Join(", ", words));
// The, quick, brown, fox, jumps, over, the, lazy, dog, and, runs, away, fast
```

### 10.2 Flattening Nested Objects

```csharp
var departments = new[]
{
    new { Name = "Engineering", Members = new[] { "Alice", "Bob" } },
    new { Name = "Marketing", Members = new[] { "Charlie", "Diana", "Eve" } },
    new { Name = "Finance", Members = new[] { "Frank" } },
};

// Flat list of all members
var allMembers = departments.SelectMany(d => d.Members);
Console.WriteLine(string.Join(", ", allMembers));
// Alice, Bob, Charlie, Diana, Eve, Frank

// With result selector — include department info
var memberDetails = departments.SelectMany(
    d => d.Members,
    (dept, member) => new { dept.Name, Member = member });

foreach (var m in memberDetails)
    Console.WriteLine($"{m.Member} works in {m.Name}");
```

### 10.3 SelectMany with Index

```csharp
var matrix = new[]
{
    new[] { 1, 2, 3 },
    new[] { 4, 5, 6 },
    new[] { 7, 8, 9 },
};

var cells = matrix.SelectMany(
    (row, rowIdx) => row.Select((val, colIdx) => new { Row = rowIdx, Col = colIdx, Value = val }));

foreach (var cell in cells)
    Console.Write($"[{cell.Row},{cell.Col}]={cell.Value} ");
// [0,0]=1 [0,1]=2 [0,2]=3 [1,0]=4 [1,1]=5 [1,2]=6 [2,0]=7 [2,1]=8 [2,2]=9
```

## 11. Writing Custom LINQ Extension Methods

LINQ's power comes from its composability. You can write your own extension methods that participate seamlessly in LINQ pipelines.

### 11.1 Custom Filter: WhereNot

```csharp
public static class LinqExtensions
{
    /// <summary>Inverse of Where — keeps elements that do NOT match the predicate.</summary>
    public static IEnumerable<T> WhereNot<T>(
        this IEnumerable<T> source, Func<T, bool> predicate)
    {
        foreach (var item in source)
        {
            if (!predicate(item))
                yield return item;
        }
    }

    /// <summary>Batch elements into chunks of the given size.</summary>
    public static IEnumerable<IReadOnlyList<T>> Batch<T>(
        this IEnumerable<T> source, int batchSize)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        var batch = new List<T>(batchSize);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == batchSize)
            {
                yield return batch.AsReadOnly();
                batch = new List<T>(batchSize);
            }
        }
        if (batch.Count > 0)
            yield return batch.AsReadOnly();
    }

    /// <summary>Interleave two sequences element by element.</summary>
    public static IEnumerable<T> Interleave<T>(
        this IEnumerable<T> first, IEnumerable<T> second)
    {
        using var e1 = first.GetEnumerator();
        using var e2 = second.GetEnumerator();

        while (true)
        {
            bool has1 = e1.MoveNext();
            bool has2 = e2.MoveNext();

            if (has1) yield return e1.Current;
            if (has2) yield return e2.Current;
            if (!has1 && !has2) yield break;
        }
    }

    /// <summary>Return elements with their running total.</summary>
    public static IEnumerable<(T Item, decimal RunningTotal)> WithRunningTotal<T>(
        this IEnumerable<T> source, Func<T, decimal> selector)
    {
        decimal total = 0;
        foreach (var item in source)
        {
            total += selector(item);
            yield return (item, total);
        }
    }
}
```

### 11.2 Using Custom Extensions

```csharp
var numbers = Enumerable.Range(1, 20);

// WhereNot
var notDivisibleBy3 = numbers.WhereNot(n => n % 3 == 0);
Console.WriteLine(string.Join(", ", notDivisibleBy3));
// 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20

// Batch (note: .NET 6+ has Chunk which does the same thing)
foreach (var batch in numbers.Batch(6))
    Console.WriteLine(string.Join(", ", batch));
// 1, 2, 3, 4, 5, 6
// 7, 8, 9, 10, 11, 12
// 13, 14, 15, 16, 17, 18
// 19, 20

// Interleave
var odds = new[] { 1, 3, 5, 7 };
var evens = new[] { 2, 4, 6, 8, 10 };
Console.WriteLine(string.Join(", ", odds.Interleave(evens)));
// 1, 2, 3, 4, 5, 6, 7, 8, 10

// Running total
var orders = new[]
{
    new { Item = "Widget", Price = 9.99m },
    new { Item = "Gadget", Price = 24.99m },
    new { Item = "Doohickey", Price = 4.49m },
};

foreach (var (order, total) in orders.WithRunningTotal(o => o.Price))
    Console.WriteLine($"{order.Item}: ${order.Price} (running: ${total})");
// Widget: $9.99 (running: $9.99)
// Gadget: $24.99 (running: $34.98)
// Doohickey: $4.49 (running: $39.47)
```

## 12. Performance Considerations

### 12.1 LINQ Overhead

LINQ introduces some overhead compared to hand-written loops: delegate invocation, enumerator allocation, and potential closure allocations. For hot paths, consider whether the readability benefit justifies the cost.

```csharp
// LINQ — clear but allocates enumerators and delegates
int sum1 = numbers.Where(n => n % 2 == 0).Sum();

// Manual loop — no allocations, faster in tight loops
int sum2 = 0;
foreach (var n in numbers)
{
    if (n % 2 == 0)
        sum2 += n;
}

// Span-based — zero allocation for arrays
int sum3 = 0;
ReadOnlySpan<int> span = numbers;
foreach (var n in span)
{
    if (n % 2 == 0)
        sum3 += n;
}
```

### 12.2 Common Performance Pitfalls

```csharp
var items = Enumerable.Range(1, 1_000_000);

// BAD: Count() iterates the entire sequence, then Any() iterates again
if (items.Where(x => x > 500_000).Count() > 0) { } // O(n)

// GOOD: Any() short-circuits on the first match
if (items.Any(x => x > 500_000)) { } // O(1) in best case

// BAD: OrderBy then First — sorts everything then takes one
var min1 = items.OrderBy(x => x).First(); // O(n log n)

// GOOD: Min does a single pass
var min2 = items.Min(); // O(n)

// BAD: multiple enumerations of a deferred query
var query = items.Where(x => x % 7 == 0);
Console.WriteLine(query.Count());   // enumerates
Console.WriteLine(query.Sum());     // enumerates again

// GOOD: materialize once
var list = items.Where(x => x % 7 == 0).ToList();
Console.WriteLine(list.Count);
Console.WriteLine(list.Sum());
```

## 13. Practice Problems

1. **Student Report Card**: Given a list of students, each with a name and a list of (Subject, Score) tuples, use LINQ to: (a) find the top 3 students by average score, (b) find the subject with the highest average across all students, (c) group students by grade letter (A: 90+, B: 80+, C: 70+, D: 60+, F: below 60) based on their average score.

2. **Word Frequency Counter**: Given a paragraph of text, use LINQ to split it into words, normalize to lowercase, remove punctuation, and produce a `Dictionary<string, int>` of word frequencies sorted by frequency descending. Print the top 10 most common words.

3. **Flatten and Transform**: Given a `List<List<(string Key, int Value)>>`, use `SelectMany` to flatten it, then group by Key, and compute the sum and average of Values for each Key. Return the results as a list of anonymous objects sorted by sum descending.

4. **Custom LINQ Operator — Pairwise**: Implement `Pairwise<T>(this IEnumerable<T> source)` that yields consecutive pairs as `(T First, T Second)`. For `[1,2,3,4,5]`, it should yield `(1,2), (2,3), (3,4), (4,5)`. Use it to find the maximum difference between consecutive elements in a sorted array.

5. **Deferred Execution Puzzle**: Write a LINQ query that filters a list. After defining the query (but before iterating), add new items to the list and remove existing ones. Predict and verify what the query returns. Then modify the code to use `ToList()` immediately and show how the behavior changes.
