// Lesson 06: Arrays and Strings
// Run: dotnet run

using System;
using System.Text;

// =============================================================================
// SINGLE-DIMENSIONAL ARRAYS
// =============================================================================
Console.WriteLine("=== Single-Dimensional Arrays ===");

// Declaration and initialization
int[] numbers = { 10, 20, 30, 40, 50 };
string[] colors = new string[3];
colors[0] = "Red";
colors[1] = "Green";
colors[2] = "Blue";

// Array properties
Console.WriteLine($"numbers.Length: {numbers.Length}");
Console.WriteLine($"First: {numbers[0]}, Last: {numbers[^1]}");

// Iterate with for
Console.Write("numbers: ");
for (int i = 0; i < numbers.Length; i++)
{
    Console.Write($"{numbers[i]} ");
}
Console.WriteLine();

// Iterate with foreach
Console.Write("colors: ");
foreach (string color in colors)
{
    Console.Write($"{color} ");
}
Console.WriteLine();

// Array methods
int[] data = { 5, 3, 8, 1, 9, 2, 7, 4, 6 };
Console.WriteLine($"\nOriginal: [{string.Join(", ", data)}]");

Array.Sort(data);
Console.WriteLine($"Sorted:   [{string.Join(", ", data)}]");

Array.Reverse(data);
Console.WriteLine($"Reversed: [{string.Join(", ", data)}]");

int idx = Array.IndexOf(data, 5);
Console.WriteLine($"IndexOf(5): {idx}");

// Array.Copy
int[] copy = new int[data.Length];
Array.Copy(data, copy, data.Length);
Console.WriteLine($"Copy: [{string.Join(", ", copy)}]");

// Array.Fill
int[] filled = new int[5];
Array.Fill(filled, 42);
Console.WriteLine($"Filled: [{string.Join(", ", filled)}]");

// =============================================================================
// RANGES AND INDICES (C# 8+)
// =============================================================================
Console.WriteLine("\n=== Ranges and Indices ===");

int[] arr = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// Index from end
Console.WriteLine($"arr[^1] = {arr[^1]}");  // Last element: 9
Console.WriteLine($"arr[^3] = {arr[^3]}");  // Third from end: 7

// Range slicing
int[] slice1 = arr[2..5];    // Elements at index 2, 3, 4
int[] slice2 = arr[..3];     // First 3 elements
int[] slice3 = arr[7..];     // From index 7 to end
int[] slice4 = arr[^3..];    // Last 3 elements

Console.WriteLine($"arr[2..5]:  [{string.Join(", ", slice1)}]");
Console.WriteLine($"arr[..3]:   [{string.Join(", ", slice2)}]");
Console.WriteLine($"arr[7..]:   [{string.Join(", ", slice3)}]");
Console.WriteLine($"arr[^3..]:  [{string.Join(", ", slice4)}]");

// =============================================================================
// MULTIDIMENSIONAL ARRAYS
// =============================================================================
Console.WriteLine("\n=== Multidimensional Arrays ===");

// 2D rectangular array (matrix)
int[,] matrix = {
    { 1, 2, 3 },
    { 4, 5, 6 },
    { 7, 8, 9 }
};

Console.WriteLine($"matrix dimensions: {matrix.GetLength(0)} x {matrix.GetLength(1)}");
Console.WriteLine($"matrix[1,2] = {matrix[1, 2]}");  // 6

// Print matrix
Console.WriteLine("Matrix:");
for (int row = 0; row < matrix.GetLength(0); row++)
{
    Console.Write("  ");
    for (int col = 0; col < matrix.GetLength(1); col++)
    {
        Console.Write($"{matrix[row, col],4}");
    }
    Console.WriteLine();
}

// 3D array
int[,,] cube = new int[2, 2, 2];
cube[0, 0, 0] = 1;
cube[1, 1, 1] = 8;
Console.WriteLine($"\ncube rank: {cube.Rank}, total elements: {cube.Length}");

// =============================================================================
// JAGGED ARRAYS (arrays of arrays)
// =============================================================================
Console.WriteLine("\n=== Jagged Arrays ===");

// Each row can have a different length
int[][] jagged = new int[3][];
jagged[0] = new int[] { 1, 2 };
jagged[1] = new int[] { 3, 4, 5 };
jagged[2] = new int[] { 6, 7, 8, 9 };

Console.WriteLine("Jagged array (triangle):");
for (int row = 0; row < jagged.Length; row++)
{
    Console.Write($"  Row {row} ({jagged[row].Length} elements): ");
    Console.WriteLine($"[{string.Join(", ", jagged[row])}]");
}

// =============================================================================
// STRING FUNDAMENTALS
// =============================================================================
Console.WriteLine("\n=== String Fundamentals ===");

// Strings are immutable reference types
string s1 = "Hello";
string s2 = s1;          // Both reference the same string
s1 += ", World!";        // Creates a NEW string; s2 still points to "Hello"
Console.WriteLine($"s1: {s1}");
Console.WriteLine($"s2: {s2}");

// String properties and indexing
string text = "Hello, C# World!";
Console.WriteLine($"\ntext: \"{text}\"");
Console.WriteLine($"Length: {text.Length}");
Console.WriteLine($"text[0]: '{text[0]}'");
Console.WriteLine($"text[^1]: '{text[^1]}'");

// =============================================================================
// STRING METHODS
// =============================================================================
Console.WriteLine("\n=== String Methods ===");

string sample = "  Hello, C# World!  ";

Console.WriteLine($"Trim:       \"{sample.Trim()}\"");
Console.WriteLine($"TrimStart:  \"{sample.TrimStart()}\"");
Console.WriteLine($"TrimEnd:    \"{sample.TrimEnd()}\"");
Console.WriteLine($"ToUpper:    \"{sample.Trim().ToUpper()}\"");
Console.WriteLine($"ToLower:    \"{sample.Trim().ToLower()}\"");
Console.WriteLine($"Contains:   {sample.Contains("C#")}");
Console.WriteLine($"StartsWith: {sample.Trim().StartsWith("Hello")}");
Console.WriteLine($"EndsWith:   {sample.Trim().EndsWith("!")}");
Console.WriteLine($"IndexOf:    {sample.IndexOf("C#")}");
Console.WriteLine($"Replace:    \"{sample.Trim().Replace("World", "Developer")}\"");
Console.WriteLine($"Substring:  \"{sample.Trim().Substring(7, 2)}\"");

// Split and Join
string csv = "apple,banana,cherry,date";
string[] parts = csv.Split(',');
Console.WriteLine($"\nSplit: [{string.Join(" | ", parts)}]");

string joined = string.Join(" - ", parts);
Console.WriteLine($"Join:  {joined}");

// =============================================================================
// STRING INTERPOLATION AND FORMATTING
// =============================================================================
Console.WriteLine("\n=== String Interpolation & Formatting ===");

string name = "Alice";
int age = 30;
double salary = 75000.5;

// Interpolation
Console.WriteLine($"Name: {name}, Age: {age}");

// Format specifiers in interpolation
Console.WriteLine($"Salary: {salary:C}");          // Currency
Console.WriteLine($"Salary: {salary:N2}");         // Number with 2 decimals
Console.WriteLine($"Hex: {255:X}");                // Hexadecimal
Console.WriteLine($"Percent: {0.856:P1}");         // Percentage
Console.WriteLine($"Date: {DateTime.Now:yyyy-MM-dd}");
Console.WriteLine($"Aligned: |{name,-15}|{age,5}|"); // Left/right alignment

// Verbatim + interpolation
string path = $@"C:\Users\{name}\Documents";
Console.WriteLine($"Path: {path}");

// =============================================================================
// STRINGBUILDER
// =============================================================================
Console.WriteLine("\n=== StringBuilder ===");

// StringBuilder is mutable — efficient for many concatenations
var sb = new StringBuilder();
sb.Append("Hello");
sb.Append(", ");
sb.Append("World!");
Console.WriteLine($"Append: {sb}");

sb.Insert(7, "Beautiful ");
Console.WriteLine($"Insert: {sb}");

sb.Replace("World", "C#");
Console.WriteLine($"Replace: {sb}");

sb.Remove(7, 10); // Remove "Beautiful "
Console.WriteLine($"Remove: {sb}");

// Performance comparison: String vs StringBuilder for many concatenations
var sw = System.Diagnostics.Stopwatch.StartNew();
string strResult = "";
for (int i = 0; i < 10_000; i++)
    strResult += "a";
sw.Stop();
long stringTime = sw.ElapsedMilliseconds;

sw.Restart();
var sbResult = new StringBuilder();
for (int i = 0; i < 10_000; i++)
    sbResult.Append("a");
string _ = sbResult.ToString();
sw.Stop();
long sbTime = sw.ElapsedMilliseconds;

Console.WriteLine($"\n10,000 concatenations:");
Console.WriteLine($"  String:        {stringTime}ms");
Console.WriteLine($"  StringBuilder: {sbTime}ms");

// =============================================================================
// CHAR OPERATIONS
// =============================================================================
Console.WriteLine("\n=== Char Operations ===");

char ch = 'A';
Console.WriteLine($"char: '{ch}'");
Console.WriteLine($"IsLetter:    {char.IsLetter(ch)}");
Console.WriteLine($"IsDigit:     {char.IsDigit(ch)}");
Console.WriteLine($"IsUpper:     {char.IsUpper(ch)}");
Console.WriteLine($"ToLower:     '{char.ToLower(ch)}'");
Console.WriteLine($"IsWhiteSpace(' '): {char.IsWhiteSpace(' ')}");

// Iterating over characters in a string
string word = "Hello123";
int letters = 0, digits = 0;
foreach (char c in word)
{
    if (char.IsLetter(c)) letters++;
    if (char.IsDigit(c)) digits++;
}
Console.WriteLine($"\n\"{word}\" has {letters} letters and {digits} digits.");
