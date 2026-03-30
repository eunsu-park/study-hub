// Lesson 08: Collections and LINQ Basics
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Linq;

// =============================================================================
// LIST<T> — dynamic array
// =============================================================================
Console.WriteLine("=== List<T> ===");

// Creation
List<string> fruits = new() { "Apple", "Banana", "Cherry" };
var numbers = new List<int> { 10, 20, 30, 40, 50 };

// Add / Insert / Remove
fruits.Add("Date");
fruits.Insert(1, "Avocado");
fruits.Remove("Cherry");
Console.WriteLine($"Fruits: [{string.Join(", ", fruits)}]");
Console.WriteLine($"Count: {fruits.Count}, Capacity: {fruits.Capacity}");

// Access and search
Console.WriteLine($"First: {fruits[0]}, Last: {fruits[^1]}");
Console.WriteLine($"Contains 'Banana': {fruits.Contains("Banana")}");
Console.WriteLine($"IndexOf 'Date': {fruits.IndexOf("Date")}");

// Sort and reverse
fruits.Sort();
Console.WriteLine($"Sorted: [{string.Join(", ", fruits)}]");

// FindAll / Find
var withA = fruits.FindAll(f => f.StartsWith("A"));
Console.WriteLine($"Starting with 'A': [{string.Join(", ", withA)}]");

// ForEach
Console.Write("ForEach: ");
fruits.ForEach(f => Console.Write($"{f} "));
Console.WriteLine();

// =============================================================================
// DICTIONARY<TKey, TValue> — key-value pairs
// =============================================================================
Console.WriteLine("\n=== Dictionary<TKey, TValue> ===");

var scores = new Dictionary<string, int>
{
    ["Alice"] = 95,
    ["Bob"] = 87,
    ["Charlie"] = 92
};

// Add and access
scores["Diana"] = 88;
Console.WriteLine($"Alice's score: {scores["Alice"]}");
Console.WriteLine($"Count: {scores.Count}");

// Safe access with TryGetValue
if (scores.TryGetValue("Bob", out int bobScore))
{
    Console.WriteLine($"Bob's score: {bobScore}");
}

if (!scores.TryGetValue("Eve", out _))
{
    Console.WriteLine("Eve not found in dictionary.");
}

// Iterate
Console.WriteLine("All scores:");
foreach (KeyValuePair<string, int> kvp in scores)
{
    Console.WriteLine($"  {kvp.Key}: {kvp.Value}");
}

// Keys and Values collections
Console.WriteLine($"Keys: [{string.Join(", ", scores.Keys)}]");
Console.WriteLine($"ContainsKey 'Charlie': {scores.ContainsKey("Charlie")}");
Console.WriteLine($"ContainsValue 88: {scores.ContainsValue(88)}");

// Remove
scores.Remove("Charlie");
Console.WriteLine($"After removing Charlie: [{string.Join(", ", scores.Keys)}]");

// =============================================================================
// HASHSET<T> — unique elements, fast lookup
// =============================================================================
Console.WriteLine("\n=== HashSet<T> ===");

var setA = new HashSet<int> { 1, 2, 3, 4, 5 };
var setB = new HashSet<int> { 3, 4, 5, 6, 7 };

Console.WriteLine($"Set A: {{{string.Join(", ", setA)}}}");
Console.WriteLine($"Set B: {{{string.Join(", ", setB)}}}");

// Add returns false if element already exists
bool added = setA.Add(3);
Console.WriteLine($"Add 3 to A (already exists): {added}");

// Set operations (modify setA, so work on copies)
var union = new HashSet<int>(setA);
union.UnionWith(setB);
Console.WriteLine($"A ∪ B: {{{string.Join(", ", union)}}}");

var intersection = new HashSet<int>(setA);
intersection.IntersectWith(setB);
Console.WriteLine($"A ∩ B: {{{string.Join(", ", intersection)}}}");

var difference = new HashSet<int>(setA);
difference.ExceptWith(setB);
Console.WriteLine($"A - B: {{{string.Join(", ", difference)}}}");

var symmetric = new HashSet<int>(setA);
symmetric.SymmetricExceptWith(setB);
Console.WriteLine($"A △ B: {{{string.Join(", ", symmetric)}}}");

Console.WriteLine($"A is subset of union: {setA.IsSubsetOf(union)}");

// =============================================================================
// QUEUE<T> — FIFO (First In, First Out)
// =============================================================================
Console.WriteLine("\n=== Queue<T> (FIFO) ===");

var queue = new Queue<string>();
queue.Enqueue("Task 1");
queue.Enqueue("Task 2");
queue.Enqueue("Task 3");
queue.Enqueue("Task 4");

Console.WriteLine($"Queue: [{string.Join(", ", queue)}]");
Console.WriteLine($"Peek: {queue.Peek()}");      // View front without removing
Console.WriteLine($"Dequeue: {queue.Dequeue()}"); // Remove from front
Console.WriteLine($"Dequeue: {queue.Dequeue()}");
Console.WriteLine($"Remaining: [{string.Join(", ", queue)}]");
Console.WriteLine($"Count: {queue.Count}");

// =============================================================================
// STACK<T> — LIFO (Last In, First Out)
// =============================================================================
Console.WriteLine("\n=== Stack<T> (LIFO) ===");

var stack = new Stack<string>();
stack.Push("Page 1");
stack.Push("Page 2");
stack.Push("Page 3");
stack.Push("Page 4");

Console.WriteLine($"Stack: [{string.Join(", ", stack)}]");
Console.WriteLine($"Peek: {stack.Peek()}");   // View top without removing
Console.WriteLine($"Pop: {stack.Pop()}");     // Remove from top
Console.WriteLine($"Pop: {stack.Pop()}");
Console.WriteLine($"Remaining: [{string.Join(", ", stack)}]");
Console.WriteLine($"Count: {stack.Count}");

// =============================================================================
// LINKED LIST
// =============================================================================
Console.WriteLine("\n=== LinkedList<T> ===");

var linked = new LinkedList<string>();
linked.AddLast("B");
linked.AddFirst("A");
linked.AddLast("D");

// Insert after a specific node
var nodeB = linked.Find("B");
if (nodeB != null)
{
    linked.AddAfter(nodeB, "C");
}

Console.Write("LinkedList: ");
foreach (var item in linked)
{
    Console.Write($"{item} -> ");
}
Console.WriteLine("null");

// =============================================================================
// LINQ BASICS
// =============================================================================
Console.WriteLine("\n=== LINQ Basics ===");

int[] data = { 12, 5, 8, 23, 17, 3, 42, 31, 9, 14 };
Console.WriteLine($"Data: [{string.Join(", ", data)}]");

// Filtering
var evens = data.Where(n => n % 2 == 0);
Console.WriteLine($"Even: [{string.Join(", ", evens)}]");

var greaterThan10 = data.Where(n => n > 10);
Console.WriteLine($">10: [{string.Join(", ", greaterThan10)}]");

// Projection (transformation)
var doubled = data.Select(n => n * 2);
Console.WriteLine($"Doubled: [{string.Join(", ", doubled)}]");

// Ordering
var sorted = data.OrderBy(n => n);
Console.WriteLine($"Sorted asc: [{string.Join(", ", sorted)}]");

var sortedDesc = data.OrderByDescending(n => n);
Console.WriteLine($"Sorted desc: [{string.Join(", ", sortedDesc)}]");

// Aggregation
Console.WriteLine($"Sum: {data.Sum()}");
Console.WriteLine($"Average: {data.Average():F2}");
Console.WriteLine($"Min: {data.Min()}, Max: {data.Max()}");
Console.WriteLine($"Count: {data.Count()}");
Console.WriteLine($"Count > 10: {data.Count(n => n > 10)}");

// Element access
Console.WriteLine($"First: {data.First()}");
Console.WriteLine($"Last: {data.Last()}");
Console.WriteLine($"First > 20: {data.First(n => n > 20)}");
Console.WriteLine($"Any > 40: {data.Any(n => n > 40)}");
Console.WriteLine($"All > 0: {data.All(n => n > 0)}");

// Chaining LINQ methods
var result = data
    .Where(n => n > 5)
    .OrderBy(n => n)
    .Select(n => $"{n}*2={n * 2}")
    .Take(5);
Console.WriteLine($"\nChained: [{string.Join(", ", result)}]");

// LINQ with objects
var people = new List<(string Name, int Age, string City)>
{
    ("Alice", 30, "NYC"),
    ("Bob", 25, "LA"),
    ("Charlie", 35, "NYC"),
    ("Diana", 28, "LA"),
    ("Eve", 32, "NYC")
};

// GroupBy
Console.WriteLine("\nGrouped by city:");
var grouped = people.GroupBy(p => p.City);
foreach (var group in grouped)
{
    var names = group.Select(p => p.Name);
    Console.WriteLine($"  {group.Key}: [{string.Join(", ", names)}]");
}

// Query syntax (alternative to method syntax)
Console.WriteLine("\nQuery syntax — NYC residents over 30:");
var query = from p in people
            where p.City == "NYC" && p.Age > 30
            orderby p.Age
            select p;

foreach (var p in query)
{
    Console.WriteLine($"  {p.Name}, age {p.Age}");
}
