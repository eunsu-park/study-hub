/*
 * Exercises for Lesson 07: Async/Await
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

// ---------------------------------------------------------------------------
// Exercise 1: Basic async — simulate file download
// ---------------------------------------------------------------------------
async Task Exercise1()
{
    Console.WriteLine("=== Exercise 1: Basic Async Download ===");

    var sw = Stopwatch.StartNew();
    string result = await DownloadFileAsync("report.pdf", 500);
    Console.WriteLine($"  {result} (took {sw.ElapsedMilliseconds}ms)");
    Console.WriteLine();
}

async Task<string> DownloadFileAsync(string filename, int delayMs)
{
    Console.WriteLine($"  Starting download: {filename}");
    await Task.Delay(delayMs);
    return $"Downloaded {filename} ({delayMs}ms simulated)";
}

// ---------------------------------------------------------------------------
// Exercise 2: WhenAll — parallel async operations
// ---------------------------------------------------------------------------
async Task Exercise2()
{
    Console.WriteLine("=== Exercise 2: WhenAll — Parallel Downloads ===");

    string[] urls = { "api/users", "api/orders", "api/products", "api/stats" };
    var sw = Stopwatch.StartNew();

    var tasks = urls.Select(url => FetchDataAsync(url));
    string[] results = await Task.WhenAll(tasks);

    Console.WriteLine($"  All completed in {sw.ElapsedMilliseconds}ms:");
    foreach (var r in results)
        Console.WriteLine($"    {r}");
    Console.WriteLine();
}

async Task<string> FetchDataAsync(string url)
{
    int delay = Random.Shared.Next(100, 400);
    await Task.Delay(delay);
    return $"{url} => {delay}ms response";
}

// ---------------------------------------------------------------------------
// Exercise 3: Cancellation — timeout pattern
// ---------------------------------------------------------------------------
async Task Exercise3()
{
    Console.WriteLine("=== Exercise 3: Cancellation Token ===");

    using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));

    try
    {
        await LongRunningOperationAsync(cts.Token);
        Console.WriteLine("  Operation completed.");
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("  Operation was cancelled (timeout hit).");
    }
    Console.WriteLine();
}

async Task LongRunningOperationAsync(CancellationToken ct)
{
    for (int i = 1; i <= 10; i++)
    {
        ct.ThrowIfCancellationRequested();
        Console.WriteLine($"  Step {i}/10...");
        await Task.Delay(100, ct);
    }
}

// ---------------------------------------------------------------------------
// Exercise 4: WhenAny — first responder wins
// ---------------------------------------------------------------------------
async Task Exercise4()
{
    Console.WriteLine("=== Exercise 4: WhenAny — First Responder ===");

    var tasks = new[]
    {
        QueryServerAsync("Server-A", 300),
        QueryServerAsync("Server-B", 150),
        QueryServerAsync("Server-C", 500),
    };

    var winner = await Task.WhenAny(tasks);
    string result = await winner;
    Console.WriteLine($"  First response: {result}");
    Console.WriteLine();
}

async Task<string> QueryServerAsync(string server, int latencyMs)
{
    await Task.Delay(latencyMs);
    return $"{server} responded in {latencyMs}ms";
}

// ---------------------------------------------------------------------------
// Exercise 5: Async stream — IAsyncEnumerable
// ---------------------------------------------------------------------------
async Task Exercise5()
{
    Console.WriteLine("=== Exercise 5: Async Stream ===");

    await foreach (var reading in GenerateSensorReadingsAsync(5))
    {
        Console.WriteLine($"  Sensor: temp={reading.Temperature:F1}C at {reading.Timestamp:HH:mm:ss.fff}");
    }
    Console.WriteLine();
}

async IAsyncEnumerable<SensorReading> GenerateSensorReadingsAsync(int count)
{
    for (int i = 0; i < count; i++)
    {
        await Task.Delay(100);
        yield return new SensorReading(
            20.0 + Random.Shared.NextDouble() * 10.0,
            DateTime.Now);
    }
}

// ---------------------------------------------------------------------------
// Exercise 6: Retry pattern with exponential backoff
// ---------------------------------------------------------------------------
async Task Exercise6()
{
    Console.WriteLine("=== Exercise 6: Retry with Backoff ===");

    int attempt = 0;
    var result = await RetryAsync(async () =>
    {
        attempt++;
        if (attempt < 3) throw new InvalidOperationException($"Attempt {attempt} failed");
        await Task.Delay(10);
        return $"Success on attempt {attempt}";
    }, maxRetries: 5, baseDelayMs: 50);

    Console.WriteLine($"  Result: {result}");
    Console.WriteLine();
}

async Task<T> RetryAsync<T>(Func<Task<T>> operation, int maxRetries, int baseDelayMs)
{
    for (int i = 0; i <= maxRetries; i++)
    {
        try
        {
            return await operation();
        }
        catch (Exception ex) when (i < maxRetries)
        {
            int delay = baseDelayMs * (1 << i);
            Console.WriteLine($"  Retry {i + 1}: {ex.Message} — waiting {delay}ms");
            await Task.Delay(delay);
        }
    }
    throw new InvalidOperationException("Should not reach here");
}

// ---- Run all exercises ----
await Exercise1();
await Exercise2();
await Exercise3();
await Exercise4();
await Exercise5();
await Exercise6();

// ===========================================================================
// Supporting types
// ===========================================================================

record SensorReading(double Temperature, DateTime Timestamp);
