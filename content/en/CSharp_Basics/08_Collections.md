# Collections

**Previous**: [Enums and Structs](./07_Enums_and_Structs.md) | **Next**: [Classes and Objects](./09_Classes_and_Objects.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `List<T>` for dynamic, indexed collections
2. Store and retrieve key-value pairs with `Dictionary<TKey, TValue>`
3. Perform set operations with `HashSet<T>`
4. Apply `Queue<T>` and `Stack<T>` for FIFO and LIFO scenarios
5. Iterate collections with `IEnumerable<T>` and `foreach`
6. Use collection initializers and collection expressions
7. Write basic LINQ queries to filter, project, and aggregate data
8. Choose the right collection based on performance characteristics

---

The .NET Base Class Library provides a rich set of generic collection types in the `System.Collections.Generic` namespace. These collections are type-safe, efficient, and cover virtually every data structure need. Understanding when and how to use each collection is a key skill for writing performant C# code.

## 1. List\<T\>

`List<T>` is the most commonly used collection. It is a dynamically resizable array that provides indexed access, insertion, removal, and searching.

### 1.1 Creating and Populating

```csharp
// Empty list
List<int> numbers = new List<int>();

// With initial capacity (avoids resizing)
List<int> preallocated = new List<int>(100);

// Collection initializer
List<string> fruits = new List<string> { "Apple", "Banana", "Cherry" };

// Target-typed new
List<double> scores = new() { 95.5, 87.2, 91.0 };

// From an existing array
int[] array = { 1, 2, 3, 4, 5 };
List<int> fromArray = new List<int>(array);

// Using AddRange
List<int> combined = new();
combined.AddRange(new[] { 10, 20, 30 });
combined.AddRange(new[] { 40, 50 });
```

### 1.2 Accessing and Modifying

```csharp
List<string> colors = new() { "Red", "Green", "Blue", "Yellow" };

// Index access
string first = colors[0];       // "Red"
string last = colors[^1];       // "Yellow" (index-from-end)

// Modify by index
colors[1] = "Lime";

// Count (not Length)
int count = colors.Count; // 4

// Add and Insert
colors.Add("Purple");           // Append
colors.Insert(2, "Orange");     // Insert at index 2

// Remove
colors.Remove("Blue");          // Remove first occurrence, returns bool
colors.RemoveAt(0);             // Remove by index
colors.RemoveAll(c => c.StartsWith("Y")); // Remove all matching predicate

// Check contents
bool hasRed = colors.Contains("Red");
int idx = colors.IndexOf("Orange"); // -1 if not found
```

### 1.3 Searching and Filtering

```csharp
List<int> data = new() { 15, 3, 27, 8, 42, 11, 6, 35 };

// Find first match
int firstOver20 = data.Find(x => x > 20);        // 27
int lastOver20 = data.FindLast(x => x > 20);      // 35
int indexOver20 = data.FindIndex(x => x > 20);     // 2

// Find all matches
List<int> allOver20 = data.FindAll(x => x > 20);   // { 27, 42, 35 }

// Exists and TrueForAll
bool anyNegative = data.Exists(x => x < 0);        // false
bool allPositive = data.TrueForAll(x => x > 0);    // true
```

### 1.4 Sorting

```csharp
List<int> nums = new() { 5, 2, 8, 1, 9 };

// Default sort (ascending)
nums.Sort();
// nums: { 1, 2, 5, 8, 9 }

// Custom comparison
nums.Sort((a, b) => b.CompareTo(a)); // Descending
// nums: { 9, 8, 5, 2, 1 }

// Sort objects by property
List<(string Name, int Age)> people = new()
{
    ("Charlie", 30),
    ("Alice", 25),
    ("Bob", 35)
};

people.Sort((a, b) => a.Age.CompareTo(b.Age));
// Sorted by age: Alice(25), Charlie(30), Bob(35)
```

### 1.5 Iteration and ForEach

```csharp
List<string> items = new() { "Alpha", "Beta", "Gamma" };

// foreach loop
foreach (string item in items)
{
    Console.WriteLine(item);
}

// for loop (when you need the index)
for (int i = 0; i < items.Count; i++)
{
    Console.WriteLine($"{i}: {items[i]}");
}

// ForEach method
items.ForEach(item => Console.WriteLine(item.ToUpper()));

// Convert to array
string[] arr = items.ToArray();
```

## 2. Dictionary\<TKey, TValue\>

A dictionary maps unique keys to values, providing O(1) average-time lookups by key.

### 2.1 Creating and Adding Entries

```csharp
// Empty dictionary
Dictionary<string, int> ages = new Dictionary<string, int>();

// Collection initializer
Dictionary<string, int> scores = new()
{
    { "Alice", 95 },
    { "Bob", 87 },
    { "Charlie", 92 }
};

// Index initializer syntax (C# 6+)
Dictionary<string, string> capitals = new()
{
    ["France"] = "Paris",
    ["Germany"] = "Berlin",
    ["Japan"] = "Tokyo"
};

// Add entries
ages.Add("Alice", 30);       // Throws if key exists
ages["Bob"] = 25;            // Add or overwrite
ages.TryAdd("Alice", 99);    // Returns false if key exists (no exception)
```

### 2.2 Retrieving Values

```csharp
Dictionary<string, int> inventory = new()
{
    ["Apples"] = 50,
    ["Bananas"] = 30,
    ["Oranges"] = 45
};

// Direct access (throws KeyNotFoundException if missing)
int appleCount = inventory["Apples"]; // 50

// Safe access with TryGetValue (preferred)
if (inventory.TryGetValue("Bananas", out int bananaCount))
{
    Console.WriteLine($"Bananas: {bananaCount}"); // 30
}

if (!inventory.TryGetValue("Grapes", out int grapeCount))
{
    Console.WriteLine("Grapes not found");
    // grapeCount is 0 (default)
}

// Check existence
bool hasOranges = inventory.ContainsKey("Oranges");   // true
bool has45 = inventory.ContainsValue(45);             // true (O(n) scan)
```

### 2.3 Removing and Updating

```csharp
Dictionary<string, double> prices = new()
{
    ["Widget"] = 9.99,
    ["Gadget"] = 24.99,
    ["Doohickey"] = 4.99
};

// Remove by key
bool removed = prices.Remove("Doohickey"); // true

// Remove and get the value
if (prices.Remove("Widget", out double widgetPrice))
{
    Console.WriteLine($"Removed Widget at ${widgetPrice}");
}

// Update
prices["Gadget"] = 19.99; // Overwrite existing value

// Conditional update
if (prices.ContainsKey("Gadget"))
{
    prices["Gadget"] *= 0.9; // Apply 10% discount
}

// Clear all entries
prices.Clear();
```

### 2.4 Iterating a Dictionary

```csharp
Dictionary<string, int> wordCount = new()
{
    ["the"] = 42,
    ["and"] = 27,
    ["is"] = 19,
    ["of"] = 31
};

// Iterate key-value pairs
foreach (KeyValuePair<string, int> kvp in wordCount)
{
    Console.WriteLine($"'{kvp.Key}' appears {kvp.Value} times");
}

// Deconstruct KeyValuePair
foreach (var (word, count) in wordCount)
{
    Console.WriteLine($"'{word}': {count}");
}

// Iterate keys only
foreach (string key in wordCount.Keys)
{
    Console.WriteLine(key);
}

// Iterate values only
foreach (int value in wordCount.Values)
{
    Console.WriteLine(value);
}
```

### 2.5 Practical Example: Frequency Counter

```csharp
static Dictionary<string, int> CountWordFrequency(string text)
{
    Dictionary<string, int> freq = new(StringComparer.OrdinalIgnoreCase);
    string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    foreach (string word in words)
    {
        if (freq.TryGetValue(word, out int count))
        {
            freq[word] = count + 1;
        }
        else
        {
            freq[word] = 1;
        }
    }
    return freq;
}

// Alternative using CollectionsMarshal (performance-critical)
// or simply:
// freq[word] = freq.GetValueOrDefault(word) + 1;

var result = CountWordFrequency("the cat sat on the mat the cat");
foreach (var (word, count) in result)
{
    Console.WriteLine($"{word}: {count}");
}
// the: 3, cat: 2, sat: 1, on: 1, mat: 1
```

## 3. HashSet\<T\>

A `HashSet<T>` stores unique elements with O(1) lookups and supports mathematical set operations.

### 3.1 Basic Usage

```csharp
// Create and add elements
HashSet<string> tags = new() { "csharp", "dotnet", "programming" };

// Add returns false if element already exists
bool added1 = tags.Add("tutorial");  // true (new)
bool added2 = tags.Add("csharp");    // false (duplicate)

Console.WriteLine(tags.Count); // 4

// Check membership
bool hasDotnet = tags.Contains("dotnet"); // true

// Remove
tags.Remove("programming"); // true
```

### 3.2 Set Operations

```csharp
HashSet<int> setA = new() { 1, 2, 3, 4, 5 };
HashSet<int> setB = new() { 3, 4, 5, 6, 7 };

// Union: elements in either set
HashSet<int> union = new(setA);
union.UnionWith(setB);
// union: { 1, 2, 3, 4, 5, 6, 7 }

// Intersection: elements in both sets
HashSet<int> intersection = new(setA);
intersection.IntersectWith(setB);
// intersection: { 3, 4, 5 }

// Difference: elements in A but not in B
HashSet<int> difference = new(setA);
difference.ExceptWith(setB);
// difference: { 1, 2 }

// Symmetric difference: elements in either but not both
HashSet<int> symDiff = new(setA);
symDiff.SymmetricExceptWith(setB);
// symDiff: { 1, 2, 6, 7 }

// Subset/superset checks
bool isSubset = setA.IsSubsetOf(new[] { 1, 2, 3, 4, 5, 6 }); // true
bool isSuperset = setA.IsSupersetOf(new[] { 1, 2, 3 });       // true
bool overlaps = setA.Overlaps(setB);                            // true
```

### 3.3 Practical Example: Removing Duplicates

```csharp
List<string> emails = new()
{
    "alice@example.com",
    "bob@example.com",
    "alice@example.com",  // duplicate
    "ALICE@EXAMPLE.COM",  // duplicate (case-insensitive)
    "charlie@example.com"
};

// Case-insensitive dedup
HashSet<string> unique = new(emails, StringComparer.OrdinalIgnoreCase);
Console.WriteLine(unique.Count); // 3

// Convert back to list if needed
List<string> deduped = unique.ToList();
```

## 4. Queue\<T\> and Stack\<T\>

### 4.1 Queue (FIFO)

`Queue<T>` implements a first-in, first-out collection:

```csharp
Queue<string> printQueue = new();

// Enqueue (add to back)
printQueue.Enqueue("Document1.pdf");
printQueue.Enqueue("Photo.jpg");
printQueue.Enqueue("Report.docx");

Console.WriteLine(printQueue.Count); // 3

// Peek at front without removing
string next = printQueue.Peek(); // "Document1.pdf"

// Dequeue (remove from front)
string first = printQueue.Dequeue();  // "Document1.pdf"
string second = printQueue.Dequeue(); // "Photo.jpg"

// Safe dequeue
if (printQueue.TryDequeue(out string? item))
{
    Console.WriteLine($"Printing: {item}"); // "Report.docx"
}

// Check if empty
if (printQueue.TryPeek(out string? _) == false)
{
    Console.WriteLine("Queue is empty");
}
```

### 4.2 Stack (LIFO)

`Stack<T>` implements a last-in, first-out collection:

```csharp
Stack<string> undoStack = new();

// Push (add to top)
undoStack.Push("Type 'Hello'");
undoStack.Push("Bold text");
undoStack.Push("Change font");

Console.WriteLine(undoStack.Count); // 3

// Peek at top without removing
string topAction = undoStack.Peek(); // "Change font"

// Pop (remove from top)
string undone1 = undoStack.Pop(); // "Change font"
string undone2 = undoStack.Pop(); // "Bold text"

// Safe pop
if (undoStack.TryPop(out string? action))
{
    Console.WriteLine($"Undid: {action}"); // "Type 'Hello'"
}
```

### 4.3 Practical Example: Balanced Brackets

```csharp
static bool AreBracketsBalanced(string expression)
{
    Stack<char> stack = new();
    Dictionary<char, char> matchingPairs = new()
    {
        [')'] = '(',
        [']'] = '[',
        ['}'] = '{'
    };

    foreach (char ch in expression)
    {
        if (ch is '(' or '[' or '{')
        {
            stack.Push(ch);
        }
        else if (matchingPairs.ContainsKey(ch))
        {
            if (stack.Count == 0 || stack.Pop() != matchingPairs[ch])
                return false;
        }
    }

    return stack.Count == 0;
}

Console.WriteLine(AreBracketsBalanced("({[]})")); // true
Console.WriteLine(AreBracketsBalanced("({[})"));   // false
Console.WriteLine(AreBracketsBalanced("((())"));   // false
```

## 5. LinkedList\<T\>

`LinkedList<T>` is a doubly-linked list that provides O(1) insertion and removal at any known node, but O(n) indexed access.

### 5.1 Basic Operations

```csharp
LinkedList<string> playlist = new();

// Add elements
playlist.AddLast("Song A");
playlist.AddLast("Song B");
playlist.AddLast("Song C");
playlist.AddFirst("Intro");

// Navigate nodes
LinkedListNode<string>? current = playlist.First;
while (current != null)
{
    Console.WriteLine(current.Value);
    current = current.Next;
}
// Intro, Song A, Song B, Song C

// Find a node
LinkedListNode<string>? songB = playlist.Find("Song B");
if (songB != null)
{
    // Insert before and after a known node
    playlist.AddBefore(songB, "Interlude");
    playlist.AddAfter(songB, "Song B Remix");

    // Remove the node
    playlist.Remove(songB);
}

// Result: Intro, Song A, Interlude, Song B Remix, Song C

// Remove first/last
playlist.RemoveFirst();
playlist.RemoveLast();
```

### 5.2 When to Use LinkedList

Use `LinkedList<T>` when you need frequent insertions/removals at arbitrary positions and already have a reference to the node. For most other scenarios, `List<T>` performs better due to cache locality and lower overhead.

## 6. IEnumerable\<T\> and foreach

### 6.1 The IEnumerable Interface

`IEnumerable<T>` is the fundamental interface for iteration. All collection types implement it, and `foreach` works with any `IEnumerable<T>`.

```csharp
// Any IEnumerable<T> can be iterated with foreach
static void PrintAll<T>(IEnumerable<T> items)
{
    foreach (T item in items)
    {
        Console.WriteLine(item);
    }
}

// Works with any collection
PrintAll(new List<int> { 1, 2, 3 });
PrintAll(new HashSet<string> { "a", "b", "c" });
PrintAll(new int[] { 10, 20, 30 });
```

### 6.2 Yield Return (Iterator Methods)

You can create custom iterables using `yield return`:

```csharp
static IEnumerable<int> Fibonacci(int count)
{
    int a = 0, b = 1;
    for (int i = 0; i < count; i++)
    {
        yield return a;
        (a, b) = (b, a + b);
    }
}

foreach (int fib in Fibonacci(10))
{
    Console.Write($"{fib} "); // 0 1 1 2 3 5 8 13 21 34
}

// Lazy evaluation: values are computed on demand
static IEnumerable<int> EvenNumbers()
{
    int n = 0;
    while (true) // Infinite sequence
    {
        yield return n;
        n += 2;
    }
}

// Take only what you need
foreach (int even in EvenNumbers().Take(5))
{
    Console.Write($"{even} "); // 0 2 4 6 8
}
```

## 7. Collection Initializers and Collection Expressions

### 7.1 Collection Initializers

```csharp
// List initializer
List<int> nums = new() { 1, 2, 3, 4, 5 };

// Dictionary initializer (two syntaxes)
Dictionary<string, int> map = new()
{
    { "one", 1 },   // Add() syntax
    ["two"] = 2,     // Indexer syntax
    ["three"] = 3
};

// HashSet initializer
HashSet<string> tags = new() { "urgent", "important", "review" };

// Nested collection initializer
Dictionary<string, List<string>> groups = new()
{
    ["fruits"] = new() { "apple", "banana" },
    ["veggies"] = new() { "carrot", "pea" }
};
```

### 7.2 Collection Expressions (C# 12)

C# 12 introduces a unified syntax for creating collections using square brackets:

```csharp
// Collection expressions with []
int[] array = [1, 2, 3, 4, 5];
List<int> list = [10, 20, 30];
Span<int> span = [100, 200, 300];
HashSet<string> set = ["a", "b", "c"];

// Spread operator (..) to include elements from other collections
int[] first = [1, 2, 3];
int[] second = [4, 5, 6];
int[] combined = [..first, ..second]; // [1, 2, 3, 4, 5, 6]

// Conditional elements with spread
bool includeExtras = true;
int[] extras = [7, 8, 9];
int[] result = [1, 2, 3, ..(includeExtras ? extras : [])];
// result: [1, 2, 3, 7, 8, 9]

// Empty collection expression
List<string> empty = [];

// Works with method parameters
static void Process(IReadOnlyList<int> data) { }
Process([1, 2, 3]);
```

## 8. Basic LINQ Preview

LINQ (Language Integrated Query) provides a declarative way to query and transform collections. This section covers the essentials; a full treatment will come in a later lesson.

### 8.1 Where (Filtering)

```csharp
using System.Linq;

List<int> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// Filter: keep only even numbers
IEnumerable<int> evens = numbers.Where(n => n % 2 == 0);
// evens: 2, 4, 6, 8, 10

// Chain with ToList to materialize
List<int> evenList = numbers.Where(n => n % 2 == 0).ToList();

// Multiple conditions
var result = numbers.Where(n => n > 3 && n < 8);
// result: 4, 5, 6, 7
```

### 8.2 Select (Projection)

```csharp
List<string> names = ["Alice", "Bob", "Charlie", "Diana"];

// Transform each element
IEnumerable<int> lengths = names.Select(n => n.Length);
// lengths: 5, 3, 7, 5

// Transform to anonymous type
var nameInfo = names.Select(n => new { Name = n, Length = n.Length });
foreach (var info in nameInfo)
{
    Console.WriteLine($"{info.Name}: {info.Length} chars");
}

// Select with index
var indexed = names.Select((name, i) => $"{i + 1}. {name}");
// "1. Alice", "2. Bob", "3. Charlie", "4. Diana"
```

### 8.3 OrderBy and ThenBy

```csharp
List<(string Name, int Age, string City)> people =
[
    ("Charlie", 30, "NYC"),
    ("Alice", 25, "LA"),
    ("Bob", 30, "NYC"),
    ("Diana", 25, "LA")
];

// Sort by age ascending
var byAge = people.OrderBy(p => p.Age);

// Sort by age descending
var byAgeDesc = people.OrderByDescending(p => p.Age);

// Multi-level sort: by age, then by name
var sorted = people
    .OrderBy(p => p.Age)
    .ThenBy(p => p.Name)
    .ToList();
// Alice(25,LA), Diana(25,LA), Bob(30,NYC), Charlie(30,NYC)
```

### 8.4 Aggregation

```csharp
List<int> scores = [85, 92, 78, 95, 88, 70, 91];

int count = scores.Count;                    // 7
int countAbove90 = scores.Count(s => s > 90); // 3
int sum = scores.Sum();                       // 599
double average = scores.Average();            // 85.57...
int min = scores.Min();                       // 70
int max = scores.Max();                       // 95

// First and Last
int first = scores.First();                   // 85
int firstOver90 = scores.First(s => s > 90);  // 92
int lastScore = scores.Last();                // 91

// Any and All
bool anyFailing = scores.Any(s => s < 60);    // false
bool allPassing = scores.All(s => s >= 60);   // true

// Aggregate (custom reduction)
string csv = scores.Aggregate("", (acc, s) => acc == "" ? s.ToString() : $"{acc},{s}");
// "85,92,78,95,88,70,91"
```

### 8.5 Chaining LINQ Methods

```csharp
List<(string Product, string Category, double Price)> products =
[
    ("Laptop", "Electronics", 999.99),
    ("Mouse", "Electronics", 29.99),
    ("Desk", "Furniture", 249.99),
    ("Chair", "Furniture", 399.99),
    ("Keyboard", "Electronics", 79.99),
    ("Lamp", "Furniture", 49.99)
];

// Find the two cheapest electronics
var cheapElectronics = products
    .Where(p => p.Category == "Electronics")
    .OrderBy(p => p.Price)
    .Take(2)
    .Select(p => $"{p.Product}: ${p.Price}")
    .ToList();
// ["Mouse: $29.99", "Keyboard: $79.99"]

// Total price of furniture
double furnitureTotal = products
    .Where(p => p.Category == "Furniture")
    .Sum(p => p.Price);
// 699.97

// Group by category
var grouped = products
    .GroupBy(p => p.Category)
    .Select(g => new { Category = g.Key, Count = g.Count(), AvgPrice = g.Average(p => p.Price) });

foreach (var group in grouped)
{
    Console.WriteLine($"{group.Category}: {group.Count} items, avg ${group.AvgPrice:F2}");
}
// Electronics: 3 items, avg $369.99
// Furniture: 3 items, avg $233.32
```

## 9. Choosing the Right Collection

### 9.1 Performance Characteristics

| Collection | Add | Remove | Lookup | Indexed | Ordered |
|-----------|-----|--------|--------|---------|---------|
| `List<T>` | O(1)* | O(n) | O(n) | O(1) | Yes (insertion) |
| `Dictionary<K,V>` | O(1)* | O(1) | O(1) by key | No | No |
| `HashSet<T>` | O(1)* | O(1) | O(1) | No | No |
| `Queue<T>` | O(1)* | O(1) front | N/A | No | FIFO |
| `Stack<T>` | O(1)* | O(1) top | N/A | No | LIFO |
| `LinkedList<T>` | O(1) at node | O(1) at node | O(n) | No | Yes |
| `SortedList<K,V>` | O(n) | O(n) | O(log n) | O(1) | Sorted by key |
| `SortedDictionary<K,V>` | O(log n) | O(log n) | O(log n) | No | Sorted by key |
| `SortedSet<T>` | O(log n) | O(log n) | O(log n) | No | Sorted |

*Amortized (may trigger resize)

### 9.2 Decision Guide

```csharp
// Need indexed access + dynamic size? -> List<T>
List<string> items = new() { "a", "b", "c" };

// Need key-value mapping? -> Dictionary<TKey, TValue>
Dictionary<int, string> lookup = new() { [1] = "one" };

// Need unique elements + fast lookup? -> HashSet<T>
HashSet<int> seen = new() { 1, 2, 3 };

// Need sorted unique elements? -> SortedSet<T>
SortedSet<int> sorted = new() { 3, 1, 2 }; // Iteration: 1, 2, 3

// Need FIFO processing? -> Queue<T>
Queue<string> tasks = new();

// Need LIFO / undo? -> Stack<T>
Stack<string> undo = new();

// Need sorted key-value with efficient insertion? -> SortedDictionary<TKey, TValue>
SortedDictionary<string, int> sortedMap = new();
```

## 10. ReadOnlyCollection and Immutable Collections

### 10.1 ReadOnlyCollection

Wrapping a collection as read-only prevents consumers from modifying it:

```csharp
using System.Collections.ObjectModel;

List<string> mutableList = new() { "Alice", "Bob", "Charlie" };

// Create a read-only wrapper
ReadOnlyCollection<string> readOnly = mutableList.AsReadOnly();

// Consumers cannot modify
// readOnly.Add("Diana");    // Compile error
// readOnly[0] = "Eve";     // Compile error

// But changes to the underlying list are visible
mutableList.Add("Diana");
Console.WriteLine(readOnly.Count); // 4

// For method signatures, prefer IReadOnlyList<T>
static void Display(IReadOnlyList<string> names)
{
    foreach (string name in names)
    {
        Console.WriteLine(name);
    }
}

Display(readOnly);
Display(mutableList); // List<T> implements IReadOnlyList<T>
```

### 10.2 Immutable Collections

Immutable collections from `System.Collections.Immutable` create new instances for every modification:

```csharp
using System.Collections.Immutable;

// Create an immutable list
ImmutableList<int> immutable = ImmutableList.Create(1, 2, 3);

// "Add" returns a NEW list (original unchanged)
ImmutableList<int> withFour = immutable.Add(4);

Console.WriteLine(immutable.Count);  // 3
Console.WriteLine(withFour.Count);   // 4

// Builder pattern for efficient bulk construction
ImmutableList<int>.Builder builder = ImmutableList.CreateBuilder<int>();
for (int i = 0; i < 1000; i++)
{
    builder.Add(i);
}
ImmutableList<int> largeList = builder.ToImmutable();

// Immutable dictionary
ImmutableDictionary<string, int> dict = ImmutableDictionary<string, int>.Empty
    .Add("Alice", 30)
    .Add("Bob", 25);

ImmutableDictionary<string, int> updated = dict.SetItem("Alice", 31);
Console.WriteLine(dict["Alice"]);    // 30 (unchanged)
Console.WriteLine(updated["Alice"]); // 31

// Immutable sorted set
ImmutableSortedSet<int> sortedSet = ImmutableSortedSet.Create(5, 1, 3, 2, 4);
// Always iterated in order: 1, 2, 3, 4, 5
```

### 10.3 Frozen Collections (.NET 8+)

For collections that are built once and read many times, frozen collections offer the best read performance:

```csharp
using System.Collections.Frozen;

Dictionary<string, int> source = new()
{
    ["key1"] = 1,
    ["key2"] = 2,
    ["key3"] = 3
};

// Create a frozen dictionary (optimized for reads, cannot be modified)
FrozenDictionary<string, int> frozen = source.ToFrozenDictionary();
int value = frozen["key1"]; // Extremely fast lookup

// Frozen set
FrozenSet<string> frozenSet = new[] { "a", "b", "c" }.ToFrozenSet();
bool contains = frozenSet.Contains("b"); // true
```

## 11. Practice Problems

1. **Student Grade Tracker**: Create a `Dictionary<string, List<int>>` that maps student names to their list of test scores. Write methods to add a score for a student, calculate a student's average, find the student with the highest average, and list all students with averages above a threshold.

2. **Set Operations CLI**: Write a program that reads two comma-separated lists of integers from the user and uses `HashSet<T>` to compute and display the union, intersection, difference (A - B), and symmetric difference. Format the output nicely.

3. **Task Queue Simulator**: Implement a simple task processor using `Queue<T>`. Define a struct `Task` with `Name`, `Priority`, and `EstimatedMinutes`. Enqueue several tasks, then process them one by one, printing the remaining estimated time after each task completes.

4. **Undo/Redo System**: Build an undo/redo system using two `Stack<string>` instances (one for undo, one for redo). Support operations: `Execute(string action)`, `Undo()`, and `Redo()`. When a new action is executed, clear the redo stack. Print the action history at each step.

5. **LINQ Data Analysis**: Given a `List<(string Name, string Department, double Salary)>` with at least 10 employees, use LINQ to: (a) find the highest-paid employee in each department, (b) calculate the average salary per department, (c) list employees earning above the company-wide average, and (d) group employees into salary brackets (e.g., <50k, 50k-100k, >100k).
