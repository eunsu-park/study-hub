/*
 * Exercises for Lesson 16: Performance Profiling
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;

// ---------------------------------------------------------------------------
// Exercise 1: Micro-benchmark — string concatenation approaches
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: String Concatenation Benchmark ===");

    const int iterations = 50_000;

    // Approach 1: string += (naive)
    var sw = Stopwatch.StartNew();
    string result1 = "";
    for (int i = 0; i < iterations; i++)
        result1 += "a";
    sw.Stop();
    long naiveMs = sw.ElapsedMilliseconds;

    // Approach 2: StringBuilder
    sw.Restart();
    var sb = new StringBuilder();
    for (int i = 0; i < iterations; i++)
        sb.Append('a');
    string result2 = sb.ToString();
    sw.Stop();
    long sbMs = sw.ElapsedMilliseconds;

    // Approach 3: String.Create
    sw.Restart();
    string result3 = new string('a', iterations);
    sw.Stop();
    long createMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"  Naive +=      : {naiveMs,6}ms (len={result1.Length})");
    Console.WriteLine($"  StringBuilder : {sbMs,6}ms (len={result2.Length})");
    Console.WriteLine($"  String.Create : {createMs,6}ms (len={result3.Length})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Collection performance — List vs Dictionary lookup
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Collection Lookup Performance ===");

    const int size = 100_000;
    const int lookups = 10_000;

    var list = Enumerable.Range(0, size).Select(i => $"item-{i}").ToList();
    var dict = list.ToDictionary(s => s, s => s);
    var hashSet = new HashSet<string>(list);

    var targets = Enumerable.Range(0, lookups)
        .Select(_ => $"item-{Random.Shared.Next(size)}")
        .ToArray();

    // List.Contains (O(n))
    var sw = Stopwatch.StartNew();
    int found1 = targets.Count(t => list.Contains(t));
    sw.Stop();
    long listMs = sw.ElapsedMilliseconds;

    // Dictionary.ContainsKey (O(1))
    sw.Restart();
    int found2 = targets.Count(t => dict.ContainsKey(t));
    sw.Stop();
    long dictMs = sw.ElapsedMilliseconds;

    // HashSet.Contains (O(1))
    sw.Restart();
    int found3 = targets.Count(t => hashSet.Contains(t));
    sw.Stop();
    long hashMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"  List.Contains      : {listMs,6}ms ({found1} found)");
    Console.WriteLine($"  Dictionary.Contains: {dictMs,6}ms ({found2} found)");
    Console.WriteLine($"  HashSet.Contains   : {hashMs,6}ms ({found3} found)");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Avoid boxing — generic vs object comparison
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Boxing Avoidance ===");

    const int iterations = 1_000_000;

    // Boxed version
    var sw = Stopwatch.StartNew();
    long boxedSum = 0;
    for (int i = 0; i < iterations; i++)
    {
        object boxed = i;        // boxing
        boxedSum += (int)boxed;  // unboxing
    }
    sw.Stop();
    long boxedMs = sw.ElapsedMilliseconds;

    // Non-boxed version
    sw.Restart();
    long directSum = 0;
    for (int i = 0; i < iterations; i++)
        directSum += i;
    sw.Stop();
    long directMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"  Boxed   : {boxedMs,4}ms (sum={boxedSum})");
    Console.WriteLine($"  Direct  : {directMs,4}ms (sum={directSum})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: LINQ vs loop — performance tradeoff
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: LINQ vs Manual Loop ===");

    int[] data = Enumerable.Range(0, 1_000_000).ToArray();
    const int runs = 100;

    // LINQ pipeline
    var sw = Stopwatch.StartNew();
    long linqSum = 0;
    for (int r = 0; r < runs; r++)
        linqSum = data.Where(x => x % 2 == 0).Select(x => (long)x * x).Sum();
    sw.Stop();
    long linqMs = sw.ElapsedMilliseconds;

    // Manual loop
    sw.Restart();
    long loopSum = 0;
    for (int r = 0; r < runs; r++)
    {
        loopSum = 0;
        for (int i = 0; i < data.Length; i++)
            if (data[i] % 2 == 0)
                loopSum += (long)data[i] * data[i];
    }
    sw.Stop();
    long loopMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"  LINQ loop : {linqMs,5}ms (sum={linqSum})");
    Console.WriteLine($"  Manual    : {loopMs,5}ms (sum={loopSum})");
    Console.WriteLine($"  Ratio     : {(double)linqMs / Math.Max(loopMs, 1):F1}x");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Memory allocation tracking
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Allocation Tracking ===");

    // Force collection to get baseline
    GC.Collect();
    GC.WaitForPendingFinalizers();
    GC.Collect();

    long before = GC.GetTotalMemory(true);

    // Allocate many small objects
    var items = new List<byte[]>();
    for (int i = 0; i < 10_000; i++)
        items.Add(new byte[100]);

    long after = GC.GetTotalMemory(false);
    long allocated = after - before;

    Console.WriteLine($"  Allocated ~{allocated / 1024}KB for 10,000 x 100-byte arrays");
    Console.WriteLine($"  GC Gen0 collections: {GC.CollectionCount(0)}");
    Console.WriteLine($"  GC Gen1 collections: {GC.CollectionCount(1)}");
    Console.WriteLine($"  GC Gen2 collections: {GC.CollectionCount(2)}");

    // Pool-friendly approach: reuse a single buffer
    GC.Collect();
    long beforePooled = GC.GetTotalMemory(true);
    var buffer = new byte[100];
    for (int i = 0; i < 10_000; i++)
    {
        Array.Clear(buffer);
        buffer[0] = (byte)(i & 0xFF); // simulate work
    }
    long afterPooled = GC.GetTotalMemory(false);

    Console.WriteLine($"  Pooled approach delta: ~{(afterPooled - beforePooled) / 1024}KB");
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
