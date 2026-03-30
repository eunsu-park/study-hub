# Arrays and Strings

**Previous**: [Methods](./05_Methods.md) | **Next**: [Enums and Structs](./07_Enums_and_Structs.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare, initialize, and manipulate single-dimensional, multidimensional, and jagged arrays
2. Use built-in array methods for sorting, searching, and copying
3. Apply range and index operators for slicing arrays
4. Work with strings effectively, understanding their immutable nature
5. Use string interpolation, verbatim strings, and raw string literals
6. Build strings efficiently with StringBuilder
7. Perform character-level operations and understand Unicode basics

---

Arrays and strings are two of the most frequently used data structures in C#. Arrays provide fixed-size, indexed collections for storing elements of the same type, while strings represent sequences of characters with a rich set of manipulation methods. Understanding both deeply is essential for writing efficient, correct C# programs.

## 1. Single-Dimensional Arrays

A single-dimensional array stores elements in a contiguous block of memory, accessed by a zero-based integer index.

### 1.1 Declaration and Initialization

There are several ways to create arrays in C#:

```csharp
// Declare and allocate with default values (0 for int)
int[] numbers = new int[5];

// Declare and initialize with values
int[] primes = new int[] { 2, 3, 5, 7, 11 };

// Shorthand initialization (type inferred from left-hand side)
int[] fibonacci = { 1, 1, 2, 3, 5, 8, 13 };

// Using var with explicit new
var scores = new int[] { 95, 87, 72, 91, 68 };

// Target-typed new (C# 9+)
int[] grades = new[] { 90, 85, 78, 92 };
```

### 1.2 Accessing and Modifying Elements

Elements are accessed using square bracket notation with a zero-based index:

```csharp
int[] data = { 10, 20, 30, 40, 50 };

// Read elements
int first = data[0];    // 10
int third = data[2];    // 30

// Modify elements
data[1] = 25;           // Array is now { 10, 25, 30, 40, 50 }

// Array length
int length = data.Length; // 5

// Iterate with a for loop
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($"data[{i}] = {data[i]}");
}

// Iterate with foreach
foreach (int value in data)
{
    Console.WriteLine(value);
}
```

### 1.3 Index-from-End Operator

C# 8 introduced the `^` operator to index from the end of an array:

```csharp
int[] values = { 10, 20, 30, 40, 50 };

int last = values[^1];       // 50 (last element)
int secondLast = values[^2]; // 40
int first = values[^5];      // 10 (same as values[0] for length-5 array)

// Useful in loops
for (int i = 1; i <= values.Length; i++)
{
    Console.WriteLine(values[^i]);
}
// Output: 50, 40, 30, 20, 10
```

### 1.4 Bounds Checking

Accessing an index outside the valid range throws an `IndexOutOfRangeException`:

```csharp
int[] arr = { 1, 2, 3 };

try
{
    int invalid = arr[5]; // Throws IndexOutOfRangeException
}
catch (IndexOutOfRangeException ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
```

## 2. Multidimensional Arrays

C# supports rectangular (multidimensional) arrays where every row has the same number of columns.

### 2.1 Two-Dimensional Arrays

```csharp
// Declare a 3x4 matrix
int[,] matrix = new int[3, 4];

// Initialize with values
int[,] grid = {
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 10, 11, 12 }
};

// Access elements
int element = grid[1, 2]; // 7 (row 1, column 2)

// Modify elements
grid[0, 0] = 100;

// Get dimensions
int rows = grid.GetLength(0);    // 3
int cols = grid.GetLength(1);    // 4
int totalElements = grid.Length;  // 12

// Iterate over all elements
for (int r = 0; r < rows; r++)
{
    for (int c = 0; c < cols; c++)
    {
        Console.Write($"{grid[r, c],4}");
    }
    Console.WriteLine();
}
```

### 2.2 Three-Dimensional Arrays

```csharp
// A 2x3x4 three-dimensional array
int[,,] cube = new int[2, 3, 4];

cube[0, 1, 2] = 42;

int depth = cube.GetLength(0);  // 2
int rows = cube.GetLength(1);   // 3
int cols = cube.GetLength(2);   // 4
```

### 2.3 Multidimensional Array Limitations

Rectangular arrays allocate all memory contiguously, which provides good cache locality but means every row must have the same column count. They also cannot be easily resized or used with many LINQ methods that expect `IEnumerable<T>`.

## 3. Jagged Arrays

A jagged array is an array of arrays. Each inner array can have a different length.

### 3.1 Declaration and Initialization

```csharp
// Declare a jagged array with 3 rows
int[][] jagged = new int[3][];

// Initialize each row independently
jagged[0] = new int[] { 1, 2, 3 };
jagged[1] = new int[] { 4, 5 };
jagged[2] = new int[] { 6, 7, 8, 9 };

// Shorthand initialization
int[][] triangle = {
    new[] { 1 },
    new[] { 1, 1 },
    new[] { 1, 2, 1 },
    new[] { 1, 3, 3, 1 },
    new[] { 1, 4, 6, 4, 1 }
};
```

### 3.2 Accessing Elements

```csharp
int[][] jagged = {
    new[] { 10, 20, 30 },
    new[] { 40, 50 },
    new[] { 60, 70, 80, 90 }
};

// Access: first bracket selects the row, second selects the column
int value = jagged[2][1]; // 70

// Iterate
for (int i = 0; i < jagged.Length; i++)
{
    Console.Write($"Row {i}: ");
    for (int j = 0; j < jagged[i].Length; j++)
    {
        Console.Write($"{jagged[i][j]} ");
    }
    Console.WriteLine();
}
```

### 3.3 Jagged vs Multidimensional

| Feature | Multidimensional (`int[,]`) | Jagged (`int[][]`) |
|---------|---------------------------|-------------------|
| Row sizes | All equal | Can vary |
| Memory layout | Single contiguous block | Array of array references |
| Performance | Better cache locality | Slightly more indirection |
| LINQ compatibility | Limited | Works with LINQ |
| CLR optimization | Less optimized | Better JIT optimization |

In practice, jagged arrays are often preferred for performance-critical code because the JIT compiler optimizes them more aggressively.

## 4. Array Methods

The `System.Array` class provides many useful static methods for manipulating arrays.

### 4.1 Sort and Reverse

```csharp
int[] numbers = { 5, 3, 8, 1, 9, 2, 7 };

// Sort in ascending order
Array.Sort(numbers);
// numbers: { 1, 2, 3, 5, 7, 8, 9 }

// Reverse the array
Array.Reverse(numbers);
// numbers: { 9, 8, 7, 5, 3, 2, 1 }

// Sort a portion of the array (index 2, count 3)
int[] partial = { 50, 40, 30, 20, 10 };
Array.Sort(partial, 1, 3); // Sorts elements at indices 1, 2, 3
// partial: { 50, 20, 30, 40, 10 }
```

### 4.2 Sorting with Custom Comparison

```csharp
string[] names = { "Charlie", "Alice", "Bob", "Diana" };

// Sort alphabetically (default)
Array.Sort(names);
// names: { "Alice", "Bob", "Charlie", "Diana" }

// Sort by string length using a Comparison<T> delegate
Array.Sort(names, (a, b) => a.Length.CompareTo(b.Length));
// names: { "Bob", "Alice", "Diana", "Charlie" }

// Sort parallel arrays together
int[] ids = { 3, 1, 4, 2 };
string[] labels = { "C", "A", "D", "B" };
Array.Sort(ids, labels);
// ids:    { 1, 2, 3, 4 }
// labels: { "A", "B", "C", "D" }
```

### 4.3 Searching

```csharp
int[] sorted = { 1, 3, 5, 7, 9, 11, 13 };

// BinarySearch (array must be sorted)
int index = Array.BinarySearch(sorted, 7); // 3

// IndexOf and LastIndexOf (work on unsorted arrays)
int[] data = { 10, 20, 30, 20, 40 };
int firstIndex = Array.IndexOf(data, 20);    // 1
int lastIndex = Array.LastIndexOf(data, 20); // 3
int notFound = Array.IndexOf(data, 99);      // -1

// Exists and Find
int[] numbers = { 2, 4, 6, 7, 8, 10 };
bool hasOdd = Array.Exists(numbers, n => n % 2 != 0);  // true
int firstOdd = Array.Find(numbers, n => n % 2 != 0);   // 7
int[] allEven = Array.FindAll(numbers, n => n % 2 == 0); // { 2, 4, 6, 8, 10 }
```

### 4.4 Copy and Resize

```csharp
int[] source = { 1, 2, 3, 4, 5 };

// Copy to a new array
int[] dest = new int[5];
Array.Copy(source, dest, 5);

// Copy a range (source index 1, dest index 2, count 3)
int[] partial = new int[7];
Array.Copy(source, 1, partial, 2, 3);
// partial: { 0, 0, 2, 3, 4, 0, 0 }

// Clone (shallow copy)
int[] cloned = (int[])source.Clone();

// Resize (creates a new array behind the scenes)
int[] growable = { 1, 2, 3 };
Array.Resize(ref growable, 6);
// growable: { 1, 2, 3, 0, 0, 0 }

Array.Resize(ref growable, 2);
// growable: { 1, 2 }
```

### 4.5 Fill and Clear

```csharp
int[] buffer = new int[10];

// Fill entire array with a value
Array.Fill(buffer, -1);
// buffer: { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }

// Fill a range (starting at index 3, count 4)
Array.Fill(buffer, 42, 3, 4);
// buffer: { -1, -1, -1, 42, 42, 42, 42, -1, -1, -1 }

// Clear elements (reset to default)
Array.Clear(buffer, 0, buffer.Length);
// buffer: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
```

## 5. Array Slicing with Ranges

C# 8 introduced the range operator `..` for creating slices of arrays.

### 5.1 Range Syntax

```csharp
int[] arr = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// Range: start..end (start inclusive, end exclusive)
int[] slice1 = arr[1..4];   // { 1, 2, 3 }
int[] slice2 = arr[..3];    // { 0, 1, 2 } (from start)
int[] slice3 = arr[7..];    // { 7, 8, 9 } (to end)
int[] slice4 = arr[..];     // { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } (full copy)

// Combine with index-from-end
int[] slice5 = arr[^3..];   // { 7, 8, 9 } (last 3 elements)
int[] slice6 = arr[1..^1];  // { 1, 2, 3, 4, 5, 6, 7, 8 }
int[] slice7 = arr[^5..^2]; // { 5, 6, 7 }
```

### 5.2 Range Variables

Ranges can be stored in variables of type `Range`:

```csharp
int[] data = { 10, 20, 30, 40, 50, 60, 70 };

Range middle = 2..5;
int[] middleSlice = data[middle]; // { 30, 40, 50 }

Range lastThree = ^3..;
int[] tail = data[lastThree]; // { 50, 60, 70 }
```

### 5.3 Important Notes on Slicing

Slicing always creates a new array (a copy), not a view. If you need a view without allocation, use `Span<T>` or `Memory<T>`:

```csharp
int[] original = { 1, 2, 3, 4, 5 };

// This creates a new array
int[] copy = original[1..4];
copy[0] = 99;
Console.WriteLine(original[1]); // Still 2

// Span provides a view (no allocation)
Span<int> view = original.AsSpan(1, 3);
view[0] = 99;
Console.WriteLine(original[1]); // Now 99
```

## 6. String Fundamentals

In C#, `string` is an alias for `System.String`. Strings are reference types but behave with value-like semantics due to their immutability.

### 6.1 String Immutability

Every operation that appears to modify a string actually creates a new string object:

```csharp
string greeting = "Hello";
string modified = greeting.Replace("H", "J"); // "Jello"

// greeting is still "Hello" (unchanged)
Console.WriteLine(greeting);  // "Hello"
Console.WriteLine(modified);  // "Jello"

// String concatenation creates new objects
string a = "Hello";
string b = a + " World"; // New string allocated
// a still equals "Hello"
```

### 6.2 Common String Methods

```csharp
string text = "  Hello, World!  ";

// Length
int len = text.Length; // 17 (including spaces)

// Trimming
string trimmed = text.Trim();       // "Hello, World!"
string trimStart = text.TrimStart(); // "Hello, World!  "
string trimEnd = text.TrimEnd();     // "  Hello, World!"

// Case conversion
string upper = trimmed.ToUpper();   // "HELLO, WORLD!"
string lower = trimmed.ToLower();   // "hello, world!"

// Searching
bool contains = trimmed.Contains("World");    // true
bool starts = trimmed.StartsWith("Hello");    // true
bool ends = trimmed.EndsWith("!");            // true
int index = trimmed.IndexOf("World");         // 7
int lastIndex = trimmed.LastIndexOf('l');     // 10

// Substring
string sub = trimmed.Substring(7, 5); // "World"

// Replace
string replaced = trimmed.Replace("World", "C#"); // "Hello, C#!"

// Split
string csv = "apple,banana,cherry";
string[] fruits = csv.Split(',');
// fruits: { "apple", "banana", "cherry" }

// Join
string joined = string.Join(" | ", fruits);
// "apple | banana | cherry"
```

### 6.3 String Comparison

```csharp
string a = "hello";
string b = "Hello";

// Case-sensitive comparison
bool equal1 = a == b;                    // false
bool equal2 = a.Equals(b);              // false

// Case-insensitive comparison
bool equal3 = a.Equals(b, StringComparison.OrdinalIgnoreCase); // true
bool equal4 = string.Equals(a, b, StringComparison.OrdinalIgnoreCase); // true

// Compare for ordering
int result = string.Compare(a, b, StringComparison.Ordinal);
// result > 0 because lowercase 'h' > uppercase 'H' in ordinal

// Null-safe comparison
string? maybeNull = null;
bool isNull = string.IsNullOrEmpty(maybeNull);      // true
bool isBlank = string.IsNullOrWhiteSpace("   ");    // true
```

## 7. String Interpolation and Special Strings

### 7.1 String Interpolation

String interpolation (introduced in C# 6) provides a readable way to embed expressions in strings:

```csharp
string name = "Alice";
int age = 30;
double gpa = 3.856;

// Basic interpolation
string intro = $"My name is {name} and I am {age} years old.";

// Expressions inside braces
string info = $"Next year I will be {age + 1}.";

// Format specifiers
string formatted = $"GPA: {gpa:F2}";          // "GPA: 3.86"
string padded = $"Name: {name,-10} Age: {age,5}"; // Left-align name, right-align age

// Alignment and format combined
string currency = $"Price: {19.99m,10:C}"; // "Price:     $19.99"

// Ternary expressions (use parentheses for clarity)
string status = $"Status: {(age >= 18 ? "Adult" : "Minor")}";

// Date formatting
DateTime now = DateTime.Now;
string dateStr = $"Today: {now:yyyy-MM-dd}";
```

### 7.2 Verbatim Strings

Verbatim strings (prefixed with `@`) treat backslashes as literal characters and can span multiple lines:

```csharp
// Regular string requires escape sequences
string path1 = "C:\\Users\\Alice\\Documents\\file.txt";

// Verbatim string treats backslashes literally
string path2 = @"C:\Users\Alice\Documents\file.txt";

// Multiline verbatim string
string poem = @"Roses are red,
Violets are blue,
C# is great,
And so are you.";

// To include a quote in a verbatim string, double it
string quoted = @"She said ""Hello"" to me.";

// Combine verbatim and interpolation
string user = "Alice";
string fullPath = $@"C:\Users\{user}\Documents";
// Or equivalently in C# 8+:
string fullPath2 = @$"C:\Users\{user}\Documents";
```

### 7.3 Raw String Literals (C# 11)

Raw string literals use at least three double-quote characters and eliminate the need for escape sequences entirely:

```csharp
// Basic raw string literal
string json = """
    {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "coding"]
    }
    """;

// The indentation of the closing """ determines the baseline indentation
// (leading whitespace up to that column is stripped)

// Raw interpolated string (use extra $ for each brace level)
string name = "Alice";
int age = 30;

string rawInterpolated = $$"""
    {
        "name": "{{name}}",
        "age": {{age}}
    }
    """;
// Double $$ means interpolation uses {{ }} instead of { }
// This avoids conflicts with JSON braces

// Single-line raw string
string singleLine = """This contains "quotes" without escaping.""";
```

## 8. StringBuilder

When building strings through repeated concatenation, `StringBuilder` avoids the overhead of creating many intermediate string objects.

### 8.1 Basic Usage

```csharp
using System.Text;

// Inefficient: creates many intermediate strings
string result = "";
for (int i = 0; i < 1000; i++)
{
    result += i.ToString() + ", "; // Bad: O(n^2) behavior
}

// Efficient: StringBuilder modifies a buffer in place
var sb = new StringBuilder();
for (int i = 0; i < 1000; i++)
{
    sb.Append(i);
    sb.Append(", ");
}
string efficient = sb.ToString();
```

### 8.2 StringBuilder Methods

```csharp
var sb = new StringBuilder("Hello");

// Append
sb.Append(" World");           // "Hello World"
sb.AppendLine("!");            // "Hello World!\n"
sb.AppendFormat("{0:C}", 9.99); // "Hello World!\n$9.99"

// Insert
sb.Insert(5, ",");             // Inserts comma at position 5

// Replace
sb.Replace("World", "C#");

// Remove
sb.Remove(0, 6);               // Remove 6 chars starting at index 0

// Indexer access
char ch = sb[0];
sb[0] = 'X';

// Length and Capacity
int len = sb.Length;
int cap = sb.Capacity;

// Clear
sb.Clear();

// Chaining (methods return the same StringBuilder)
string output = new StringBuilder()
    .Append("Name: ")
    .Append("Alice")
    .Append(", Age: ")
    .Append(30)
    .ToString();
// "Name: Alice, Age: 30"
```

### 8.3 When to Use StringBuilder

- Use `string` concatenation for a small, fixed number of pieces (e.g., `a + b + c`)
- Use `StringBuilder` when concatenating in a loop or building strings from many dynamic parts
- Use `string.Join` when combining an array or collection with a delimiter
- Use string interpolation for readability with a few embedded values

## 9. Span and Character Operations

### 9.1 Span for String Slicing

`Span<char>` and `ReadOnlySpan<char>` allow you to work with substrings without allocating new string objects:

```csharp
string longText = "The quick brown fox jumps over the lazy dog";

// Slice without allocation
ReadOnlySpan<char> quick = longText.AsSpan(4, 5); // "quick"

// Parse numbers without allocation
ReadOnlySpan<char> numberSpan = "12345".AsSpan(1, 3);
int parsed = int.Parse(numberSpan); // 234

// Split with Span (no array allocation)
ReadOnlySpan<char> csv = "a,b,c,d".AsSpan();
// Use manual scanning or MemoryExtensions.IndexOf
int commaIndex = csv.IndexOf(',');
ReadOnlySpan<char> first = csv[..commaIndex]; // "a"
```

### 9.2 Character Operations

The `char` type represents a single UTF-16 code unit. The `char` struct provides useful classification methods:

```csharp
char letter = 'A';
char digit = '7';
char space = ' ';
char symbol = '#';

// Classification methods
bool isLetter = char.IsLetter(letter);         // true
bool isDigit = char.IsDigit(digit);            // true
bool isWhiteSpace = char.IsWhiteSpace(space);  // true
bool isUpper = char.IsUpper(letter);           // true
bool isLower = char.IsLower('a');              // true
bool isLetterOrDigit = char.IsLetterOrDigit('x'); // true
bool isPunctuation = char.IsPunctuation(',');  // true

// Conversion
char upper = char.ToUpper('a'); // 'A'
char lower = char.ToLower('Z'); // 'z'

// Numeric value
int numericValue = (int)char.GetNumericValue('7'); // 7

// Iterating characters in a string
string word = "Hello";
foreach (char c in word)
{
    Console.Write($"{c}({(int)c}) "); // H(72) e(101) l(108) l(108) o(111)
}
```

### 9.3 Unicode Basics

C# strings are sequences of UTF-16 code units. Most characters use a single `char`, but some (like emoji or rare CJK characters) require a surrogate pair (two `char` values):

```csharp
// Basic Multilingual Plane characters (single char)
string ascii = "Hello";         // 5 chars, 5 code points
string korean = "안녕하세요";     // 5 chars, 5 code points

// Supplementary characters (surrogate pairs)
string emoji = "😀";
Console.WriteLine(emoji.Length);          // 2 (surrogate pair)

// Use StringInfo for accurate character counting
using System.Globalization;
var info = new StringInfo(emoji);
Console.WriteLine(info.LengthInTextElements); // 1

// Enumerate text elements (grapheme clusters)
string mixed = "Hello😀World";
var enumerator = StringInfo.GetTextElementEnumerator(mixed);
while (enumerator.MoveNext())
{
    Console.Write($"[{enumerator.GetTextElement()}]");
}
// [H][e][l][l][o][😀][W][o][r][l][d]
```

## 10. Practice Problems

1. **Array Rotation**: Write a method `void RotateLeft(int[] arr, int positions)` that rotates an array to the left by the given number of positions. For example, rotating `{1, 2, 3, 4, 5}` by 2 positions yields `{3, 4, 5, 1, 2}`. Use array slicing with ranges.

2. **Matrix Transpose**: Write a method `int[,] Transpose(int[,] matrix)` that returns the transpose of a rectangular 2D array. Row `i`, column `j` of the result should equal row `j`, column `i` of the input.

3. **Word Frequency Counter**: Write a method `Dictionary<string, int> CountWords(string text)` that splits a string into words (separated by spaces and punctuation), converts them to lowercase, and returns a dictionary of word frequencies. Use `string.Split` and appropriate `StringSplitOptions`.

4. **Palindrome Checker**: Write a method `bool IsPalindrome(string text)` that checks if a string is a palindrome, ignoring case, spaces, and punctuation. For example, `"A man, a plan, a canal: Panama"` should return `true`. Use `char.IsLetterOrDigit` for filtering.

5. **CSV Parser**: Write a method `string[][] ParseCsv(string csvContent)` that parses a multi-line CSV string into a jagged array of strings. Handle basic cases where fields are separated by commas and rows are separated by newlines. Use `string.Split` and return a jagged array where each inner array represents one row.
