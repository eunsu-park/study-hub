// Lesson 09: Spans and Memory
// Run: dotnet run
// Note: Requires <AllowUnsafeBlocks>true</AllowUnsafeBlocks> in .csproj for stackalloc in some contexts

using System;
using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;

// ============================================================
// 1. Span<T> Basics
// ============================================================

Console.WriteLine("=== Span<T> Basics ===");

// Span<T> is a stack-only ref struct that provides a view over contiguous memory
int[] array = { 10, 20, 30, 40, 50, 60, 70, 80 };

// Create a span over the array (no copy)
Span<int> span = array.AsSpan();
Console.WriteLine($"Full span: [{FormatSpan(span)}]");

// Slicing — zero-allocation sub-view
Span<int> middle = span.Slice(2, 4); // Elements at index 2..5
Console.WriteLine($"Slice(2,4): [{FormatSpan(middle)}]");

// Range syntax also works
Span<int> firstThree = span[..3];
Span<int> lastTwo = span[^2..];
Console.WriteLine($"First three: [{FormatSpan(firstThree)}]");
Console.WriteLine($"Last two: [{FormatSpan(lastTwo)}]");

// Modifying the span modifies the underlying array
middle[0] = 999;
Console.WriteLine($"After middle[0]=999, array[2] = {array[2]}"); // 999

// ============================================================
// 2. ReadOnlySpan<T>
// ============================================================

Console.WriteLine("\n=== ReadOnlySpan<T> ===");

// ReadOnlySpan prevents modification
ReadOnlySpan<int> readOnly = array.AsSpan();
// readOnly[0] = 42; // Compile error: read-only

Console.WriteLine($"ReadOnlySpan[0] = {readOnly[0]}");

// String as ReadOnlySpan<char> — zero-allocation substring
string text = "Hello, World! This is C# Span demo.";
ReadOnlySpan<char> charSpan = text.AsSpan();

// Find and slice without allocating a new string
ReadOnlySpan<char> word = charSpan[7..12]; // "World"
Console.WriteLine($"Sliced word: {word.ToString()}");

// Efficient parsing without string allocation
ReadOnlySpan<char> csv = "Alice,30,Seattle".AsSpan();
int firstComma = csv.IndexOf(',');
int lastComma = csv.LastIndexOf(',');

ReadOnlySpan<char> name = csv[..firstComma];
ReadOnlySpan<char> age = csv[(firstComma + 1)..lastComma];
ReadOnlySpan<char> city = csv[(lastComma + 1)..];

Console.WriteLine($"Parsed CSV: name={name}, age={age}, city={city}");

// ============================================================
// 3. Stackalloc — Stack-Allocated Buffers
// ============================================================

Console.WriteLine("\n=== Stackalloc ===");

// Allocate a small buffer on the stack (no GC pressure)
Span<int> stackBuffer = stackalloc int[8];

for (int i = 0; i < stackBuffer.Length; i++)
    stackBuffer[i] = i * i;

Console.WriteLine($"Stack buffer: [{FormatSpan(stackBuffer)}]");

// Common pattern: stack for small, heap for large
int size = 128;
Span<byte> buffer = size <= 256
    ? stackalloc byte[size]
    : new byte[size];

buffer.Fill(0xAB);
Console.WriteLine($"Buffer[0..4]: {buffer[0]:X2} {buffer[1]:X2} {buffer[2]:X2} {buffer[3]:X2}");

// Stackalloc with initializer
Span<int> primes = stackalloc int[] { 2, 3, 5, 7, 11, 13 };
Console.WriteLine($"Primes: [{FormatSpan(primes)}]");

// ============================================================
// 4. ArrayPool<T> — Rented Buffers
// ============================================================

Console.WriteLine("\n=== ArrayPool<T> ===");

// ArrayPool avoids frequent allocations by reusing arrays
var pool = ArrayPool<byte>.Shared;

// Rent a buffer (may be larger than requested)
byte[] rented = pool.Rent(1024);
Console.WriteLine($"Requested 1024, got {rented.Length} bytes");

try
{
    // Use the buffer
    for (int i = 0; i < 1024; i++)
        rented[i] = (byte)(i % 256);

    Console.WriteLine($"  rented[0]={rented[0]}, rented[1023]={rented[1023]}");
}
finally
{
    // Always return the buffer — optionally clear it
    pool.Return(rented, clearArray: true);
    Console.WriteLine("  Buffer returned to pool.");
}

// Practical: process data in chunks without allocations
Console.WriteLine("\n  Processing in chunks:");
ProcessInChunks(new byte[5000], chunkSize: 1024);

// ============================================================
// 5. Memory<T> — Heap-Compatible Span
// ============================================================

Console.WriteLine("\n=== Memory<T> ===");

// Memory<T> can be stored on the heap (unlike Span<T>)
int[] data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
Memory<int> memory = data.AsMemory();

// Slice Memory<T> — still zero-copy
Memory<int> segment = memory[3..7];
Console.WriteLine($"Segment: [{FormatMemory(segment)}]");

// Convert to Span for stack-only operations
Span<int> spanView = segment.Span;
spanView[0] = 100;
Console.WriteLine($"After modification, data[3] = {data[3]}"); // 100

// Memory<T> can be passed to async methods (Span<T> cannot)
await ProcessMemoryAsync(memory);

// ============================================================
// 6. Performance Comparison
// ============================================================

Console.WriteLine("\n=== Performance Comparison ===");

string longString = new string('x', 10_000);
int iterations = 100_000;

// Substring approach (allocates new strings)
var sw = Stopwatch.StartNew();
for (int i = 0; i < iterations; i++)
{
    string sub = longString.Substring(100, 200);
    _ = sub.Length; // Prevent optimization
}
sw.Stop();
Console.WriteLine($"  Substring: {sw.ElapsedMilliseconds}ms");

// Span approach (zero allocation)
sw.Restart();
for (int i = 0; i < iterations; i++)
{
    ReadOnlySpan<char> sub = longString.AsSpan(100, 200);
    _ = sub.Length;
}
sw.Stop();
Console.WriteLine($"  Span slice: {sw.ElapsedMilliseconds}ms");

// ============================================================
// 7. Span Utilities
// ============================================================

Console.WriteLine("\n=== Span Utilities ===");

Span<int> src = stackalloc int[] { 1, 2, 3, 4, 5 };
Span<int> dst = stackalloc int[5];

// Copy
src.CopyTo(dst);
Console.WriteLine($"Copied: [{FormatSpan(dst)}]");

// Fill
dst.Fill(42);
Console.WriteLine($"Filled: [{FormatSpan(dst)}]");

// Clear
dst.Clear();
Console.WriteLine($"Cleared: [{FormatSpan(dst)}]");

// SequenceEqual
Span<int> a = stackalloc int[] { 1, 2, 3 };
Span<int> b = stackalloc int[] { 1, 2, 3 };
Span<int> c = stackalloc int[] { 1, 2, 4 };
Console.WriteLine($"a.SequenceEqual(b): {a.SequenceEqual(b)}");
Console.WriteLine($"a.SequenceEqual(c): {a.SequenceEqual(c)}");

// ============================================================
// Helper Methods
// ============================================================

static string FormatSpan<T>(Span<T> span)
{
    var parts = new string[span.Length];
    for (int i = 0; i < span.Length; i++)
        parts[i] = span[i]?.ToString() ?? "null";
    return string.Join(", ", parts);
}

static string FormatSpan<T>(ReadOnlySpan<T> span)
{
    var parts = new string[span.Length];
    for (int i = 0; i < span.Length; i++)
        parts[i] = span[i]?.ToString() ?? "null";
    return string.Join(", ", parts);
}

static string FormatMemory<T>(Memory<T> memory)
{
    var span = memory.Span;
    var parts = new string[span.Length];
    for (int i = 0; i < span.Length; i++)
        parts[i] = span[i]?.ToString() ?? "null";
    return string.Join(", ", parts);
}

static void ProcessInChunks(byte[] data, int chunkSize)
{
    var pool = ArrayPool<byte>.Shared;
    byte[] buffer = pool.Rent(chunkSize);
    try
    {
        int offset = 0;
        while (offset < data.Length)
        {
            int remaining = Math.Min(chunkSize, data.Length - offset);
            data.AsSpan(offset, remaining).CopyTo(buffer);
            Console.WriteLine($"    Processed chunk at offset {offset}, size {remaining}");
            offset += remaining;
        }
    }
    finally
    {
        pool.Return(buffer);
    }
}

static async Task ProcessMemoryAsync(Memory<int> memory)
{
    await Task.Delay(10);
    Console.WriteLine($"  Async processed {memory.Length} elements");
}
