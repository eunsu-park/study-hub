/*
 * Exercises for Lesson 06: Records and Immutability
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Record design — define a domain model with records
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Record Design ===");

    var product = new Product("SKU-001", "Laptop", 999.99m, new[] { "electronics", "computing" });
    Console.WriteLine($"  {product}");

    // Positional record deconstruction
    var (sku, name, price, _) = product;
    Console.WriteLine($"  Deconstructed: sku={sku}, name={name}, price={price:C}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: With-expressions — immutable updates
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: With-Expressions ===");

    var original = new Employee("E001", "Alice", "Engineering", 95_000m);
    Console.WriteLine($"  Original : {original}");

    var promoted = original with { Department = "Management", Salary = 120_000m };
    Console.WriteLine($"  Promoted : {promoted}");

    var clone = original with { };
    Console.WriteLine($"  Clone eq : {ReferenceEquals(original, clone)} (ref), {original == clone} (value)");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Value equality — records vs classes
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Value Equality ===");

    var r1 = new Point(3, 4);
    var r2 = new Point(3, 4);
    var r3 = new Point(5, 6);

    Console.WriteLine($"  r1 == r2: {r1 == r2}");   // true
    Console.WriteLine($"  r1 == r3: {r1 == r3}");   // false
    Console.WriteLine($"  r1.GetHashCode() == r2.GetHashCode(): {r1.GetHashCode() == r2.GetHashCode()}");

    // Use in HashSet
    var set = new HashSet<Point> { r1, r2, r3 };
    Console.WriteLine($"  HashSet count (r1,r2,r3): {set.Count}"); // 2
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Record struct vs record class
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Record Struct vs Record Class ===");

    var colorClass = new ColorRecord(255, 128, 0);
    var colorStruct = new ColorRecordStruct(255, 128, 0);

    Console.WriteLine($"  Record class : {colorClass}");
    Console.WriteLine($"  Record struct: {colorStruct}");

    // Record struct is value type
    var copy = colorStruct;
    Console.WriteLine($"  Struct copy == original: {copy == colorStruct}");
    Console.WriteLine($"  Class is reference type: {colorClass.GetType().IsClass}");
    Console.WriteLine($"  Struct is value type   : {colorStruct.GetType().IsValueType}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Immutable collection — build an event log
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Immutable Event Log ===");

    var log = new EventLog(ImmutableList<LogEntry>.Empty);

    log = log.Append(new LogEntry(DateTime.Now, "INFO", "Application started"));
    log = log.Append(new LogEntry(DateTime.Now, "WARN", "Low memory"));
    log = log.Append(new LogEntry(DateTime.Now, "ERROR", "Connection failed"));

    Console.WriteLine($"  Total entries: {log.Entries.Count}");
    foreach (var entry in log.Entries)
        Console.WriteLine($"    [{entry.Level}] {entry.Message}");

    var filtered = log.FilterByLevel("ERROR");
    Console.WriteLine($"  Error entries: {filtered.Entries.Count}");
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

record Product(string Sku, string Name, decimal Price, string[] Tags);

record Employee(string Id, string Name, string Department, decimal Salary);

record Point(double X, double Y);

record ColorRecord(byte R, byte G, byte B);

record struct ColorRecordStruct(byte R, byte G, byte B);

record LogEntry(DateTime Timestamp, string Level, string Message);

record EventLog(ImmutableList<LogEntry> Entries)
{
    public EventLog Append(LogEntry entry) =>
        this with { Entries = Entries.Add(entry) };

    public EventLog FilterByLevel(string level) =>
        this with { Entries = Entries.Where(e => e.Level == level).ToImmutableList() };
}
