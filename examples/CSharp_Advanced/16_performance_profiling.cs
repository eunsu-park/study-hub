// Lesson 16: Performance Profiling
// Run: dotnet run
// Note: For BenchmarkDotNet, create a project and add:
//   dotnet add package BenchmarkDotNet
// Then run in Release mode: dotnet run -c Release

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;

// ============================================================
// 1. Stopwatch — Manual Benchmarking
// ============================================================

Console.WriteLine("=== Stopwatch Benchmarking ===");

const int iterations = 100_000;

// Benchmark string concatenation vs StringBuilder
var sw = Stopwatch.StartNew();
string concat = "";
for (int i = 0; i < 10_000; i++)
    concat += "a";
sw.Stop();
long concatMs = sw.ElapsedMilliseconds;

sw.Restart();
var sb = new StringBuilder();
for (int i = 0; i < 10_000; i++)
    sb.Append("a");
string built = sb.ToString();
sw.Stop();
long builderMs = sw.ElapsedMilliseconds;

Console.WriteLine($"  String concatenation (10K): {concatMs}ms");
Console.WriteLine($"  StringBuilder (10K):        {builderMs}ms");
Console.WriteLine($"  StringBuilder is ~{(concatMs > 0 ? concatMs / Math.Max(builderMs, 1) : 99)}x faster");

// ============================================================
// 2. BenchmarkDotNet Example (Code Pattern)
// ============================================================

Console.WriteLine("\n=== BenchmarkDotNet Pattern ===");
Console.WriteLine("  (Below is the pattern; run with `dotnet run -c Release` for real benchmarks)");

// In a real benchmark project, you would write:
/*
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;

BenchmarkRunner.Run<StringBenchmarks>();

[MemoryDiagnoser]          // Track allocations
[SimpleJob(warmupCount: 3, iterationCount: 10)]
public class StringBenchmarks
{
    private const int N = 1000;

    [Benchmark(Baseline = true)]
    public string Concatenation()
    {
        string result = "";
        for (int i = 0; i < N; i++)
            result += "x";
        return result;
    }

    [Benchmark]
    public string StringBuilder()
    {
        var sb = new StringBuilder();
        for (int i = 0; i < N; i++)
            sb.Append("x");
        return sb.ToString();
    }

    [Benchmark]
    public string StringCreate()
    {
        return string.Create(N, 0, (span, _) =>
        {
            for (int i = 0; i < span.Length; i++)
                span[i] = 'x';
        });
    }
}
*/

// Simulated output format:
Console.WriteLine("""
  | Method         | Mean       | Allocated |
  |--------------- |-----------:|----------:|
  | Concatenation  | 45.123 us  |  512.3 KB |
  | StringBuilder  |  1.234 us  |    4.1 KB |
  | StringCreate   |  0.456 us  |    1.0 KB |
  """);

// ============================================================
// 3. Allocation Analysis
// ============================================================

Console.WriteLine("\n=== Allocation Analysis ===");

// Track GC collections to identify allocation pressure
long gen0Before = GC.CollectionCount(0);
long gen1Before = GC.CollectionCount(1);
long gen2Before = GC.CollectionCount(2);
long memBefore = GC.GetTotalMemory(false);

// Allocating code
var lists = new List<int[]>();
for (int i = 0; i < 10_000; i++)
    lists.Add(new int[100]); // ~400KB per allocation

long memAfter = GC.GetTotalMemory(false);
long gen0After = GC.CollectionCount(0);

Console.WriteLine($"  Memory before: {memBefore / 1024:N0} KB");
Console.WriteLine($"  Memory after:  {memAfter / 1024:N0} KB");
Console.WriteLine($"  Delta:         {(memAfter - memBefore) / 1024:N0} KB");
Console.WriteLine($"  Gen0 collections: {gen0After - gen0Before}");

lists.Clear();
GC.Collect();

// ============================================================
// 4. Avoiding Allocations — Common Patterns
// ============================================================

Console.WriteLine("\n=== Allocation-Free Patterns ===");

// Pattern 1: ArrayPool instead of new array
Console.WriteLine("  --- ArrayPool vs new[] ---");
var pool = ArrayPool<byte>.Shared;

sw.Restart();
for (int i = 0; i < iterations; i++)
{
    var arr = new byte[1024]; // Allocates every time
    _ = arr.Length;
}
sw.Stop();
Console.WriteLine($"  new byte[1024] x{iterations}: {sw.ElapsedMilliseconds}ms");

sw.Restart();
for (int i = 0; i < iterations; i++)
{
    var arr = pool.Rent(1024); // Reuses from pool
    pool.Return(arr);
}
sw.Stop();
Console.WriteLine($"  ArrayPool x{iterations}:     {sw.ElapsedMilliseconds}ms");

// Pattern 2: Span<T> instead of Substring
Console.WriteLine("\n  --- Span vs Substring ---");
string longText = new string('x', 10_000);

sw.Restart();
for (int i = 0; i < iterations; i++)
{
    string sub = longText.Substring(100, 500); // Allocates new string
    _ = sub.Length;
}
sw.Stop();
Console.WriteLine($"  Substring x{iterations}: {sw.ElapsedMilliseconds}ms");

sw.Restart();
for (int i = 0; i < iterations; i++)
{
    ReadOnlySpan<char> sub = longText.AsSpan(100, 500); // Zero allocation
    _ = sub.Length;
}
sw.Stop();
Console.WriteLine($"  Span.Slice x{iterations}: {sw.ElapsedMilliseconds}ms");

// Pattern 3: stackalloc for small buffers
Console.WriteLine("\n  --- stackalloc vs heap ---");
sw.Restart();
for (int i = 0; i < iterations; i++)
{
    var arr = new int[16]; // Heap allocation
    arr[0] = i;
}
sw.Stop();
Console.WriteLine($"  new int[16] x{iterations}: {sw.ElapsedMilliseconds}ms");

sw.Restart();
for (int i = 0; i < iterations; i++)
{
    Span<int> arr = stackalloc int[16]; // Stack allocation
    arr[0] = i;
}
sw.Stop();
Console.WriteLine($"  stackalloc x{iterations}: {sw.ElapsedMilliseconds}ms");

// ============================================================
// 5. LINQ vs Loop Performance
// ============================================================

Console.WriteLine("\n=== LINQ vs Manual Loop ===");

int[] data = Enumerable.Range(0, 1_000_000).ToArray();

sw.Restart();
int linqSum = data.Where(x => x % 2 == 0).Sum();
sw.Stop();
Console.WriteLine($"  LINQ Where+Sum: {sw.ElapsedMilliseconds}ms (result={linqSum})");

sw.Restart();
int loopSum = 0;
for (int i = 0; i < data.Length; i++)
    if (data[i] % 2 == 0) loopSum += data[i];
sw.Stop();
Console.WriteLine($"  Manual loop:    {sw.ElapsedMilliseconds}ms (result={loopSum})");

// ============================================================
// 6. Dictionary Lookup vs List.Find
// ============================================================

Console.WriteLine("\n=== Dictionary vs List Lookup ===");

const int lookupSize = 100_000;
var dict = new Dictionary<int, string>(lookupSize);
var list = new List<(int Key, string Value)>(lookupSize);

for (int i = 0; i < lookupSize; i++)
{
    dict[i] = $"value-{i}";
    list.Add((i, $"value-{i}"));
}

int lookupKey = lookupSize - 1; // Worst case for list

sw.Restart();
for (int i = 0; i < 10_000; i++)
    _ = dict[lookupKey];
sw.Stop();
Console.WriteLine($"  Dictionary lookup (10K): {sw.ElapsedTicks} ticks");

sw.Restart();
for (int i = 0; i < 10_000; i++)
    _ = list.Find(x => x.Key == lookupKey);
sw.Stop();
Console.WriteLine($"  List.Find (10K):         {sw.ElapsedTicks} ticks");

// ============================================================
// 7. Struct vs Class Performance
// ============================================================

Console.WriteLine("\n=== Struct vs Class (Allocation) ===");

const int structIterations = 1_000_000;

sw.Restart();
for (int i = 0; i < structIterations; i++)
{
    var p = new PointStruct(i, i * 2); // Stack, no GC
    _ = p.X + p.Y;
}
sw.Stop();
Console.WriteLine($"  Struct x{structIterations}: {sw.ElapsedMilliseconds}ms");

sw.Restart();
for (int i = 0; i < structIterations; i++)
{
    var p = new PointClass(i, i * 2); // Heap, GC pressure
    _ = p.X + p.Y;
}
sw.Stop();
Console.WriteLine($"  Class  x{structIterations}: {sw.ElapsedMilliseconds}ms");

// ============================================================
// 8. AggressiveInlining Hint
// ============================================================

Console.WriteLine("\n=== Inlining Hint ===");

sw.Restart();
long inlinedSum = 0;
for (int i = 0; i < 10_000_000; i++)
    inlinedSum += FastAdd(i, 1);
sw.Stop();
Console.WriteLine($"  Inlined method (10M calls): {sw.ElapsedMilliseconds}ms, sum={inlinedSum}");

[MethodImpl(MethodImplOptions.AggressiveInlining)]
static int FastAdd(int a, int b) => a + b;

// ============================================================
// Supporting Types
// ============================================================

readonly record struct PointStruct(int X, int Y);

record class PointClass(int X, int Y);
