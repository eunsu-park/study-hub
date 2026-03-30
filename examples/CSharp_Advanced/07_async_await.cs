// Lesson 07: Async/Await
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;

// ============================================================
// 1. Basic Async/Await
// ============================================================

Console.WriteLine("=== Basic Async/Await ===");

var sw = Stopwatch.StartNew();

string result = await FetchDataAsync("https://api.example.com/data");
Console.WriteLine($"  Result: {result} (elapsed: {sw.ElapsedMilliseconds}ms)");

// Async method returns Task<T>
async Task<string> FetchDataAsync(string url)
{
    Console.WriteLine($"  Fetching {url}...");
    await Task.Delay(200); // Simulate network latency
    return $"Data from {url}";
}

// ============================================================
// 2. Task.WhenAll — Concurrent Execution
// ============================================================

Console.WriteLine("\n=== Task.WhenAll ===");

sw.Restart();

// Launch multiple tasks concurrently
Task<string> task1 = FetchDataAsync("service-A");
Task<string> task2 = FetchDataAsync("service-B");
Task<string> task3 = FetchDataAsync("service-C");

// Wait for all to complete
string[] results = await Task.WhenAll(task1, task2, task3);

Console.WriteLine($"  All completed in {sw.ElapsedMilliseconds}ms (concurrent, not 600ms)");
foreach (var r in results)
    Console.WriteLine($"    {r}");

// ============================================================
// 3. Task.WhenAny — First Completion
// ============================================================

Console.WriteLine("\n=== Task.WhenAny ===");

async Task<string> SlowService() { await Task.Delay(500); return "Slow"; }
async Task<string> FastService() { await Task.Delay(100); return "Fast"; }

sw.Restart();
var winner = await Task.WhenAny(SlowService(), FastService());
Console.WriteLine($"  Winner: {await winner} (in {sw.ElapsedMilliseconds}ms)");

// ============================================================
// 4. CancellationToken
// ============================================================

Console.WriteLine("\n=== CancellationToken ===");

using var cts = new CancellationTokenSource();

// Cancel after 300ms
cts.CancelAfter(TimeSpan.FromMilliseconds(300));

try
{
    await LongRunningOperationAsync(cts.Token);
}
catch (OperationCanceledException)
{
    Console.WriteLine("  Operation was cancelled!");
}

async Task LongRunningOperationAsync(CancellationToken token)
{
    for (int i = 1; i <= 10; i++)
    {
        // Check for cancellation at each iteration
        token.ThrowIfCancellationRequested();

        Console.WriteLine($"  Step {i}/10...");
        await Task.Delay(100, token);
    }
    Console.WriteLine("  Completed all steps.");
}

// ============================================================
// 5. Async Return Types: Task, Task<T>, ValueTask
// ============================================================

Console.WriteLine("\n=== Async Return Types ===");

// Task — async void-returning method (use Task, not void)
await LogMessageAsync("Hello, async!");

async Task LogMessageAsync(string message)
{
    await Task.Delay(50);
    Console.WriteLine($"  [Log] {message}");
}

// ValueTask — avoids allocation when result is often synchronous
var cached = new CachedService();
Console.WriteLine($"  First call: {await cached.GetValueAsync(1)}");  // Cache miss
Console.WriteLine($"  Second call: {await cached.GetValueAsync(1)}"); // Cache hit (sync)

// ============================================================
// 6. IAsyncEnumerable — Async Streams
// ============================================================

Console.WriteLine("\n=== IAsyncEnumerable ===");

// Consume an async stream
await foreach (var number in GenerateNumbersAsync(5))
{
    Console.WriteLine($"  Received: {number}");
}

// Async stream with cancellation
using var streamCts = new CancellationTokenSource(TimeSpan.FromMilliseconds(350));
Console.WriteLine("\n  With cancellation:");
try
{
    await foreach (var number in GenerateNumbersAsync(10).WithCancellation(streamCts.Token))
    {
        Console.WriteLine($"  Received: {number}");
    }
}
catch (OperationCanceledException)
{
    Console.WriteLine("  Stream cancelled!");
}

async IAsyncEnumerable<int> GenerateNumbersAsync(
    int count,
    [EnumeratorCancellation] CancellationToken token = default)
{
    for (int i = 1; i <= count; i++)
    {
        await Task.Delay(100, token);
        yield return i * 10;
    }
}

// ============================================================
// 7. Exception Handling in Async Code
// ============================================================

Console.WriteLine("\n=== Async Exception Handling ===");

// Single task exception
try
{
    await FailingOperationAsync();
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"  Caught: {ex.Message}");
}

// Multiple task exceptions with WhenAll
var tasks = new[]
{
    Task.Run(async () => { await Task.Delay(50); throw new Exception("Error A"); }),
    Task.Run(async () => { await Task.Delay(100); throw new Exception("Error B"); }),
    Task.Run(async () => { await Task.Delay(30); return "OK"; }),
};

try
{
    await Task.WhenAll(tasks);
}
catch
{
    // Task.WhenAll throws only the first exception;
    // inspect individual tasks for all exceptions
    foreach (var t in tasks)
    {
        if (t.IsFaulted)
            Console.WriteLine($"  Task faulted: {t.Exception?.InnerException?.Message}");
        else if (t.IsCompletedSuccessfully)
            Console.WriteLine($"  Task succeeded");
    }
}

async Task FailingOperationAsync()
{
    await Task.Delay(50);
    throw new InvalidOperationException("Something went wrong");
}

// ============================================================
// 8. Practical: Async Pipeline
// ============================================================

Console.WriteLine("\n=== Async Pipeline ===");

// Simulate a data processing pipeline
var ids = new[] { 1, 2, 3, 4, 5 };
var processedItems = new List<string>();

// Process items concurrently with a degree of parallelism
var semaphore = new SemaphoreSlim(2); // Max 2 concurrent

var processingTasks = ids.Select(async id =>
{
    await semaphore.WaitAsync();
    try
    {
        var data = await FetchItemAsync(id);
        var processed = await TransformAsync(data);
        lock (processedItems)
            processedItems.Add(processed);
    }
    finally
    {
        semaphore.Release();
    }
});

await Task.WhenAll(processingTasks);
processedItems.Sort();

Console.WriteLine("  Processed items:");
foreach (var item in processedItems)
    Console.WriteLine($"    {item}");

async Task<string> FetchItemAsync(int id)
{
    await Task.Delay(100);
    return $"item-{id}";
}

async Task<string> TransformAsync(string data)
{
    await Task.Delay(50);
    return $"[{data.ToUpper()}]";
}

// ============================================================
// Supporting Types
// ============================================================

// ValueTask cache example
class CachedService
{
    private readonly Dictionary<int, string> _cache = new();

    public ValueTask<string> GetValueAsync(int key)
    {
        // Return synchronously from cache (no allocation)
        if (_cache.TryGetValue(key, out var cached))
            return new ValueTask<string>(cached);

        // Cache miss — do async work
        return new ValueTask<string>(LoadAndCacheAsync(key));
    }

    private async Task<string> LoadAndCacheAsync(int key)
    {
        await Task.Delay(100); // Simulate I/O
        var value = $"Value-{key}";
        _cache[key] = value;
        return value;
    }
}
