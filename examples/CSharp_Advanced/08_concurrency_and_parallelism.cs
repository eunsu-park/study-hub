// Lesson 08: Concurrency and Parallelism
// Run: dotnet run

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

// ============================================================
// 1. Thread Basics
// ============================================================

Console.WriteLine("=== Thread Basics ===");

var thread = new Thread(() =>
{
    for (int i = 0; i < 3; i++)
    {
        Console.WriteLine($"  Worker thread: step {i} (Thread {Environment.CurrentManagedThreadId})");
        Thread.Sleep(100);
    }
});

thread.Start();
Console.WriteLine($"  Main thread continues (Thread {Environment.CurrentManagedThreadId})");
thread.Join(); // Wait for thread to finish
Console.WriteLine("  Worker thread completed.");

// ============================================================
// 2. Thread Pool and Task.Run
// ============================================================

Console.WriteLine("\n=== Task.Run (Thread Pool) ===");

var task = Task.Run(() =>
{
    Console.WriteLine($"  Running on thread pool thread {Environment.CurrentManagedThreadId}");
    Thread.Sleep(100);
    return 42;
});

int result = await task;
Console.WriteLine($"  Task result: {result}");

// ============================================================
// 3. Parallel.For and Parallel.ForEach
// ============================================================

Console.WriteLine("\n=== Parallel.For ===");

var sw = Stopwatch.StartNew();

// Parallel.For distributes iterations across threads
var squaredResults = new int[10];
Parallel.For(0, 10, i =>
{
    Thread.Sleep(100); // Simulate work
    squaredResults[i] = i * i;
});

Console.WriteLine($"  Completed in {sw.ElapsedMilliseconds}ms (parallel, not 1000ms)");
Console.WriteLine($"  Results: [{string.Join(", ", squaredResults)}]");

// Parallel.ForEach
Console.WriteLine("\n=== Parallel.ForEach ===");

string[] urls = { "page-1", "page-2", "page-3", "page-4", "page-5" };
var bag = new ConcurrentBag<string>();

sw.Restart();
Parallel.ForEach(urls, new ParallelOptions { MaxDegreeOfParallelism = 3 }, url =>
{
    Thread.Sleep(200); // Simulate network request
    bag.Add($"Fetched {url} on thread {Environment.CurrentManagedThreadId}");
});

Console.WriteLine($"  Completed in {sw.ElapsedMilliseconds}ms");
foreach (var item in bag)
    Console.WriteLine($"    {item}");

// ============================================================
// 4. ConcurrentDictionary
// ============================================================

Console.WriteLine("\n=== ConcurrentDictionary ===");

var wordCount = new ConcurrentDictionary<string, int>();

string[] words = { "hello", "world", "hello", "foo", "world", "hello", "bar", "foo" };

// Thread-safe increment using AddOrUpdate
Parallel.ForEach(words, word =>
{
    wordCount.AddOrUpdate(word, 1, (_, count) => count + 1);
});

Console.WriteLine("  Word counts:");
foreach (var kvp in wordCount.OrderByDescending(x => x.Value))
    Console.WriteLine($"    {kvp.Key}: {kvp.Value}");

// GetOrAdd — atomic get or create
var cache = new ConcurrentDictionary<string, string>();
string value = cache.GetOrAdd("key1", k => $"computed-{k}");
Console.WriteLine($"  GetOrAdd: {value}");

// ============================================================
// 5. Lock and Synchronization
// ============================================================

Console.WriteLine("\n=== Lock ===");

var counter = new ThreadSafeCounter();

var lockTasks = Enumerable.Range(0, 10).Select(_ => Task.Run(() =>
{
    for (int i = 0; i < 1000; i++)
        counter.Increment();
}));

await Task.WhenAll(lockTasks);
Console.WriteLine($"  Counter (10 threads x 1000): {counter.Value}"); // Exactly 10000

// SemaphoreSlim — limit concurrency
Console.WriteLine("\n=== SemaphoreSlim ===");

var semaphore = new SemaphoreSlim(2); // Allow 2 concurrent
var semTasks = Enumerable.Range(1, 5).Select(async i =>
{
    await semaphore.WaitAsync();
    try
    {
        Console.WriteLine($"  Task {i} entered (Thread {Environment.CurrentManagedThreadId})");
        await Task.Delay(200);
        Console.WriteLine($"  Task {i} exiting");
    }
    finally
    {
        semaphore.Release();
    }
});

await Task.WhenAll(semTasks);

// ============================================================
// 6. Channels — Producer/Consumer
// ============================================================

Console.WriteLine("\n=== Channels (Producer/Consumer) ===");

// Bounded channel with capacity 3
var channel = Channel.CreateBounded<string>(new BoundedChannelOptions(3)
{
    FullMode = BoundedChannelFullMode.Wait
});

// Producer task
var producer = Task.Run(async () =>
{
    for (int i = 1; i <= 6; i++)
    {
        await channel.Writer.WriteAsync($"Message-{i}");
        Console.WriteLine($"  [Producer] Sent Message-{i}");
        await Task.Delay(50);
    }
    channel.Writer.Complete(); // Signal no more items
});

// Consumer task
var consumer = Task.Run(async () =>
{
    await foreach (var message in channel.Reader.ReadAllAsync())
    {
        Console.WriteLine($"  [Consumer] Received {message}");
        await Task.Delay(100); // Consumer is slower than producer
    }
    Console.WriteLine("  [Consumer] Channel completed.");
});

await Task.WhenAll(producer, consumer);

// ============================================================
// 7. Multiple Consumers (Fan-out)
// ============================================================

Console.WriteLine("\n=== Multiple Consumers ===");

var fanChannel = Channel.CreateUnbounded<int>();

// Single producer
var fanProducer = Task.Run(async () =>
{
    for (int i = 1; i <= 10; i++)
    {
        await fanChannel.Writer.WriteAsync(i);
        await Task.Delay(30);
    }
    fanChannel.Writer.Complete();
});

// Multiple consumers competing for items
var consumers = Enumerable.Range(1, 3).Select(consumerId => Task.Run(async () =>
{
    await foreach (var item in fanChannel.Reader.ReadAllAsync())
    {
        Console.WriteLine($"  Consumer-{consumerId} processed item {item}");
        await Task.Delay(80);
    }
})).ToArray();

await fanProducer;
await Task.WhenAll(consumers);

// ============================================================
// 8. Interlocked — Lock-Free Atomic Operations
// ============================================================

Console.WriteLine("\n=== Interlocked ===");

int atomicCounter = 0;
var atomicTasks = Enumerable.Range(0, 10).Select(_ => Task.Run(() =>
{
    for (int i = 0; i < 1000; i++)
        Interlocked.Increment(ref atomicCounter);
}));

await Task.WhenAll(atomicTasks);
Console.WriteLine($"  Atomic counter: {atomicCounter}"); // Exactly 10000

// ============================================================
// Supporting Types
// ============================================================

class ThreadSafeCounter
{
    private int _value;
    private readonly object _lock = new();

    public int Value
    {
        get { lock (_lock) return _value; }
    }

    public void Increment()
    {
        lock (_lock)
        {
            _value++;
        }
    }
}
