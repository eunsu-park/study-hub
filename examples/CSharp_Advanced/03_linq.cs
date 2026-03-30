// Lesson 03: LINQ (Language Integrated Query)
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Linq;

// ============================================================
// Sample Data
// ============================================================

var students = new List<Student>
{
    new("Alice",   "CS",      3.8, 3),
    new("Bob",     "CS",      3.2, 2),
    new("Charlie", "Math",    3.9, 4),
    new("Diana",   "CS",      3.5, 3),
    new("Eve",     "Math",    2.8, 1),
    new("Frank",   "Physics", 3.7, 4),
    new("Grace",   "Physics", 3.1, 2),
    new("Hank",    "Math",    3.6, 3),
};

// ============================================================
// 1. Query Syntax vs Method Syntax
// ============================================================

Console.WriteLine("=== Query Syntax vs Method Syntax ===");

// Query syntax (SQL-like)
var queryCs = from s in students
              where s.Department == "CS"
              orderby s.GPA descending
              select s.Name;

Console.WriteLine("CS students (query syntax): " + string.Join(", ", queryCs));

// Equivalent method syntax (fluent API)
var methodCs = students
    .Where(s => s.Department == "CS")
    .OrderByDescending(s => s.GPA)
    .Select(s => s.Name);

Console.WriteLine("CS students (method syntax): " + string.Join(", ", methodCs));

// ============================================================
// 2. Deferred Execution
// ============================================================

Console.WriteLine("\n=== Deferred Execution ===");

var mutableList = new List<int> { 1, 2, 3, 4, 5 };

// This query is NOT executed yet — it is just a definition
var evens = mutableList.Where(n => n % 2 == 0);

// Modify the source BEFORE enumeration
mutableList.Add(6);
mutableList.Add(8);

// Now the query runs — includes 6 and 8
Console.WriteLine($"Evens (deferred): {string.Join(", ", evens)}");

// Force immediate execution with ToList/ToArray
var snapshot = mutableList.Where(n => n > 3).ToList();
mutableList.Add(100);
Console.WriteLine($"Snapshot (immediate): {string.Join(", ", snapshot)}");
// 100 is NOT included because the query was already materialized

// ============================================================
// 3. Filtering and Projection
// ============================================================

Console.WriteLine("\n=== Filtering and Projection ===");

// Where — filter
var seniorStudents = students.Where(s => s.Year >= 3);
Console.WriteLine($"Seniors: {string.Join(", ", seniorStudents.Select(s => s.Name))}");

// Select — project into anonymous type
var summaries = students.Select(s => new { s.Name, Honor = s.GPA >= 3.5 ? "Honor" : "Regular" });
foreach (var s in summaries)
    Console.WriteLine($"  {s.Name}: {s.Honor}");

// SelectMany — flatten nested collections
var departments = new[]
{
    new { Name = "CS", Courses = new[] { "Algorithms", "OS", "DB" } },
    new { Name = "Math", Courses = new[] { "Calculus", "Algebra" } },
};
var allCourses = departments.SelectMany(d => d.Courses);
Console.WriteLine($"All courses: {string.Join(", ", allCourses)}");

// ============================================================
// 4. Ordering
// ============================================================

Console.WriteLine("\n=== Ordering ===");

var ordered = students
    .OrderBy(s => s.Department)
    .ThenByDescending(s => s.GPA);

foreach (var s in ordered)
    Console.WriteLine($"  {s.Department,-8} {s.Name,-8} GPA={s.GPA}");

// ============================================================
// 5. Grouping
// ============================================================

Console.WriteLine("\n=== Grouping ===");

// GroupBy returns IGrouping<TKey, TElement>
var byDept = students.GroupBy(s => s.Department);

foreach (var group in byDept)
{
    var avgGpa = group.Average(s => s.GPA);
    Console.WriteLine($"  {group.Key}: {group.Count()} students, avg GPA = {avgGpa:F2}");
    foreach (var s in group)
        Console.WriteLine($"    - {s.Name} ({s.GPA})");
}

// ============================================================
// 6. Joins
// ============================================================

Console.WriteLine("\n=== Joins ===");

var advisors = new[]
{
    new { Department = "CS", Advisor = "Prof. Turing" },
    new { Department = "Math", Advisor = "Prof. Euler" },
    new { Department = "Physics", Advisor = "Prof. Newton" },
};

// Inner join
var withAdvisor = students.Join(
    advisors,
    s => s.Department,
    a => a.Department,
    (s, a) => new { s.Name, s.Department, a.Advisor }
);

foreach (var item in withAdvisor.Take(5))
    Console.WriteLine($"  {item.Name} ({item.Department}) -> {item.Advisor}");

// ============================================================
// 7. Aggregation Operators
// ============================================================

Console.WriteLine("\n=== Aggregation ===");

Console.WriteLine($"Count: {students.Count}");
Console.WriteLine($"Max GPA: {students.Max(s => s.GPA)}");
Console.WriteLine($"Min GPA: {students.Min(s => s.GPA)}");
Console.WriteLine($"Average GPA: {students.Average(s => s.GPA):F2}");
Console.WriteLine($"Sum of years: {students.Sum(s => s.Year)}");

// Aggregate — custom reduction
var allNames = students.Aggregate("", (acc, s) => acc + (acc == "" ? "" : ", ") + s.Name);
Console.WriteLine($"All names: {allNames}");

// ============================================================
// 8. Element Operators
// ============================================================

Console.WriteLine("\n=== Element Operators ===");

var first = students.First(s => s.Department == "Math");
Console.WriteLine($"First Math student: {first.Name}");

var lastOrDefault = students.LastOrDefault(s => s.Department == "Biology");
Console.WriteLine($"Last Biology student: {lastOrDefault?.Name ?? "(none)"}");

var single = students.Single(s => s.Name == "Alice");
Console.WriteLine($"Single 'Alice': {single.Name}, GPA={single.GPA}");

bool anyHonor = students.Any(s => s.GPA >= 3.8);
bool allPassing = students.All(s => s.GPA >= 2.0);
Console.WriteLine($"Any honor (>=3.8): {anyHonor}");
Console.WriteLine($"All passing (>=2.0): {allPassing}");

// ============================================================
// 9. Set Operations
// ============================================================

Console.WriteLine("\n=== Set Operations ===");

var csStudents = students.Where(s => s.Department == "CS").Select(s => s.Name);
var seniorNames = students.Where(s => s.Year >= 3).Select(s => s.Name);

Console.WriteLine($"CS: {string.Join(", ", csStudents)}");
Console.WriteLine($"Seniors: {string.Join(", ", seniorNames)}");
Console.WriteLine($"Union: {string.Join(", ", csStudents.Union(seniorNames))}");
Console.WriteLine($"Intersect: {string.Join(", ", csStudents.Intersect(seniorNames))}");
Console.WriteLine($"CS Except Seniors: {string.Join(", ", csStudents.Except(seniorNames))}");

// ============================================================
// 10. Chunk and Range (C# 10+/LINQ additions)
// ============================================================

Console.WriteLine("\n=== Chunk ===");

var chunks = Enumerable.Range(1, 10).Chunk(3);
foreach (var chunk in chunks)
    Console.WriteLine($"  [{string.Join(", ", chunk)}]");

// ============================================================
// Record type for sample data
// ============================================================

record Student(string Name, string Department, double GPA, int Year);
