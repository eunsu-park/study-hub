/*
 * Exercises for Lesson 03: LINQ
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Basic LINQ queries — filtering and projection
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Basic LINQ Queries ===");

    var products = new List<Product>
    {
        new("Laptop", "Electronics", 999.99m, 15),
        new("Mouse", "Electronics", 29.99m, 150),
        new("Desk", "Furniture", 349.00m, 30),
        new("Chair", "Furniture", 249.00m, 45),
        new("Keyboard", "Electronics", 79.99m, 80),
        new("Monitor", "Electronics", 449.00m, 25),
        new("Lamp", "Furniture", 59.99m, 60),
    };

    // Query syntax: electronics over $50
    var expensiveElectronics =
        from p in products
        where p.Category == "Electronics" && p.Price > 50m
        orderby p.Price descending
        select new { p.Name, p.Price };

    Console.WriteLine("  Expensive Electronics:");
    foreach (var item in expensiveElectronics)
        Console.WriteLine($"    {item.Name} — {item.Price:C}");

    // Method syntax: total inventory value per category
    var categoryTotals = products
        .GroupBy(p => p.Category)
        .Select(g => new { Category = g.Key, TotalValue = g.Sum(p => p.Price * p.Stock) })
        .OrderByDescending(x => x.TotalValue);

    Console.WriteLine("  Category Inventory Values:");
    foreach (var cat in categoryTotals)
        Console.WriteLine($"    {cat.Category}: {cat.TotalValue:C}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: GroupBy and Aggregate
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: GroupBy and Aggregate ===");

    var scores = new List<StudentScore>
    {
        new("Alice", "Math", 92), new("Alice", "Science", 88),
        new("Bob", "Math", 76), new("Bob", "Science", 95),
        new("Charlie", "Math", 85), new("Charlie", "Science", 79),
        new("Alice", "English", 90), new("Bob", "English", 82),
        new("Charlie", "English", 91),
    };

    var studentAverages = scores
        .GroupBy(s => s.Name)
        .Select(g => new
        {
            Student = g.Key,
            Average = g.Average(s => s.Score),
            Best = g.OrderByDescending(s => s.Score).First().Subject
        })
        .OrderByDescending(x => x.Average);

    foreach (var s in studentAverages)
        Console.WriteLine($"  {s.Student}: avg={s.Average:F1}, best={s.Best}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Join — correlate two data sources
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: LINQ Join ===");

    var departments = new List<(int Id, string Name)>
    {
        (1, "Engineering"), (2, "Marketing"), (3, "Sales")
    };

    var employees = new List<(string Name, int DeptId, decimal Salary)>
    {
        ("Alice", 1, 120_000m), ("Bob", 1, 110_000m),
        ("Charlie", 2, 90_000m), ("Dave", 3, 85_000m),
        ("Eve", 2, 95_000m),
    };

    var joined =
        from e in employees
        join d in departments on e.DeptId equals d.Id
        group e by d.Name into deptGroup
        select new
        {
            Department = deptGroup.Key,
            HeadCount = deptGroup.Count(),
            AvgSalary = deptGroup.Average(e => e.Salary)
        };

    foreach (var d in joined)
        Console.WriteLine($"  {d.Department}: {d.HeadCount} employees, avg {d.AvgSalary:C0}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Custom LINQ extension method — Batch
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Custom Extension — Batch ===");

    var numbers = Enumerable.Range(1, 11);
    var batches = numbers.Batch(3);

    foreach (var batch in batches)
        Console.WriteLine($"  [{string.Join(", ", batch)}]");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Deferred execution demonstration
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Deferred Execution ===");

    var source = new List<int> { 1, 2, 3, 4, 5 };
    var query = source.Where(n =>
    {
        Console.WriteLine($"    Evaluating {n}");
        return n > 2;
    });

    Console.WriteLine("  Query defined — nothing evaluated yet.");
    Console.WriteLine("  Now iterating:");
    foreach (var item in query)
        Console.WriteLine($"  -> {item}");

    Console.WriteLine("  Modifying source and re-iterating:");
    source.Add(10);
    foreach (var item in query)
        Console.WriteLine($"  -> {item}");
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

record Product(string Name, string Category, decimal Price, int Stock);
record StudentScore(string Name, string Subject, int Score);

static class LinqExtensions
{
    public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
    {
        var batch = new List<T>(size);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == size)
            {
                yield return batch;
                batch = new List<T>(size);
            }
        }
        if (batch.Count > 0) yield return batch;
    }
}
