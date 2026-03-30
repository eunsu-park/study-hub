/*
 * Exercises for Lesson 09: Spans and Memory
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

// ---------------------------------------------------------------------------
// Exercise 1: Span-based string parsing — extract CSV fields without allocation
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Span-Based CSV Parsing ===");

    string csvLine = "Alice,30,Engineering,Senior,95000";
    ReadOnlySpan<char> span = csvLine.AsSpan();

    var fields = new List<string>();
    while (span.Length > 0)
    {
        int commaIndex = span.IndexOf(',');
        if (commaIndex == -1)
        {
            fields.Add(span.ToString());
            break;
        }
        fields.Add(span[..commaIndex].ToString());
        span = span[(commaIndex + 1)..];
    }

    Console.WriteLine($"  Input: {csvLine}");
    for (int i = 0; i < fields.Count; i++)
        Console.WriteLine($"  Field[{i}]: '{fields[i]}'");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Span slicing — parse key=value pairs
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Key=Value Parsing ===");

    string[] inputs = { "host=localhost", "port=8080", "debug=true", "name=Hello World" };

    foreach (var input in inputs)
    {
        ReadOnlySpan<char> span = input.AsSpan();
        int eqIndex = span.IndexOf('=');
        if (eqIndex >= 0)
        {
            var key = span[..eqIndex];
            var value = span[(eqIndex + 1)..];
            Console.WriteLine($"  key='{key}', value='{value}'");
        }
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Span<byte> — zero-allocation integer formatting
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Span<byte> Formatting ===");

    Span<byte> buffer = stackalloc byte[64];
    int[] values = { 42, 255, 1024, 65535 };

    foreach (var val in values)
    {
        int bytesWritten = FormatIntToUtf8(val, buffer);
        string result = Encoding.UTF8.GetString(buffer[..bytesWritten]);
        Console.WriteLine($"  {val} => UTF8 bytes: [{string.Join(", ", buffer[..bytesWritten].ToArray())}] => \"{result}\"");
    }
    Console.WriteLine();
}

int FormatIntToUtf8(int value, Span<byte> destination)
{
    Span<char> chars = stackalloc char[20];
    value.TryFormat(chars, out int charsWritten);
    return Encoding.UTF8.GetBytes(chars[..charsWritten], destination);
}

// ---------------------------------------------------------------------------
// Exercise 4: ArrayPool — rent and return buffers
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: ArrayPool Usage ===");

    var pool = ArrayPool<int>.Shared;
    int[] buffer = pool.Rent(100); // May return >= 100

    Console.WriteLine($"  Requested 100, got buffer of length {buffer.Length}");

    // Fill with squares
    for (int i = 0; i < 100; i++)
        buffer[i] = i * i;

    int sum = 0;
    for (int i = 0; i < 100; i++)
        sum += buffer[i];

    Console.WriteLine($"  Sum of squares 0..99: {sum}");

    pool.Return(buffer, clearArray: true);
    Console.WriteLine($"  Buffer returned to pool (cleared: {buffer[0] == 0})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Performance comparison — string vs Span substring counting
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: String vs Span Performance ===");

    string text = new('a', 10_000) + "needle" + new string('b', 10_000);
    const int iterations = 10_000;

    // String-based
    var sw = Stopwatch.StartNew();
    int count1 = 0;
    for (int i = 0; i < iterations; i++)
        count1 += CountSubstring(text, "needle");
    sw.Stop();
    long stringMs = sw.ElapsedMilliseconds;

    // Span-based
    sw.Restart();
    int count2 = 0;
    for (int i = 0; i < iterations; i++)
        count2 += CountSubstringSpan(text.AsSpan(), "needle".AsSpan());
    sw.Stop();
    long spanMs = sw.ElapsedMilliseconds;

    Console.WriteLine($"  String-based: {count1} found, {stringMs}ms");
    Console.WriteLine($"  Span-based  : {count2} found, {spanMs}ms");
    Console.WriteLine();
}

int CountSubstring(string source, string target)
{
    int count = 0, idx = 0;
    while ((idx = source.IndexOf(target, idx, StringComparison.Ordinal)) >= 0)
    { count++; idx += target.Length; }
    return count;
}

int CountSubstringSpan(ReadOnlySpan<char> source, ReadOnlySpan<char> target)
{
    int count = 0;
    while (source.Length >= target.Length)
    {
        int idx = source.IndexOf(target);
        if (idx < 0) break;
        count++;
        source = source[(idx + target.Length)..];
    }
    return count;
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
