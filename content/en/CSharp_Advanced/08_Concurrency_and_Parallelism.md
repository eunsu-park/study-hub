# Concurrency and Parallelism

**Previous**: [Async and Await](./07_Async_Await.md) | **Next**: [Spans and Memory](./09_Spans_and_Memory.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between concurrency, parallelism, and asynchrony
2. Use the `Thread` class and `ThreadPool` for low-level threading
3. Apply the Task Parallel Library (TPL) for data-parallel workloads
4. Protect shared state with `lock`, `Monitor`, `SemaphoreSlim`, and `Mutex`
5. Use concurrent collections for thread-safe data access
6. Perform atomic operations with the `Interlocked` class
7. Implement the producer-consumer pattern with `System.Threading.Channels`
8. Apply thread safety best practices to real-world applications

---

Concurrency is about structure: managing multiple tasks that can make progress within overlapping time periods. Parallelism is about execution: running multiple tasks *simultaneously* on multiple CPU cores. C# gives you a rich set of primitives for both. This lesson covers everything from raw threads to high-level channels, teaching you to write correct, performant multi-threaded code.

## 1. Thread Class Basics

The `Thread` class is the lowest-level concurrency primitive in .NET. While you rarely create raw threads in modern code, understanding them is foundational.

### 1.1 Creating and Starting Threads

```csharp
// Basic thread creation
Thread worker = new Thread(() =>
{
    for (int i = 0; i < 5; i++)
    {
        Console.WriteLine($"Worker: {i} (Thread {Thread.CurrentThread.ManagedThreadId})");
        Thread.Sleep(100);
    }
});

worker.Name = "MyWorker";
worker.IsBackground = true; // Won't prevent app from exiting
worker.Start();

Console.WriteLine($"Main thread: {Thread.CurrentThread.ManagedThreadId}");
worker.Join(); // Wait for the worker to finish
Console.WriteLine("Worker completed.");
```

### 1.2 Passing Data to Threads

```csharp
// Using a lambda closure
string message = "Hello from thread";
Thread t = new Thread(() => Console.WriteLine(message));
t.Start();

// Using ParameterizedThreadStart
Thread t2 = new Thread(obj =>
{
    var (name, count) = ((string, int))obj!;
    for (int i = 0; i < count; i++)
        Console.WriteLine($"{name}: iteration {i}");
});
t2.Start(("Worker", 5));
```

### 1.3 Thread Properties

```csharp
Thread current = Thread.CurrentThread;
Console.WriteLine($"Thread ID: {current.ManagedThreadId}");
Console.WriteLine($"Name: {current.Name}");
Console.WriteLine($"IsBackground: {current.IsBackground}");
Console.WriteLine($"IsThreadPoolThread: {current.IsThreadPoolThread}");
Console.WriteLine($"Priority: {current.Priority}");
Console.WriteLine($"State: {current.ThreadState}");
```

## 2. ThreadPool and Work Items

Creating threads is expensive. The `ThreadPool` maintains a pool of reusable worker threads that dramatically reduce overhead for short-lived operations.

### 2.1 Queueing Work

```csharp
// Queue a work item
ThreadPool.QueueUserWorkItem(state =>
{
    Console.WriteLine($"Pool thread {Thread.CurrentThread.ManagedThreadId}: {state}");
}, "work item data");

// Get pool information
ThreadPool.GetMinThreads(out int minWorker, out int minIO);
ThreadPool.GetMaxThreads(out int maxWorker, out int maxIO);
Console.WriteLine($"Workers: {minWorker}-{maxWorker}, IO: {minIO}-{maxIO}");
```

### 2.2 ThreadPool vs Manual Threads

```csharp
// Benchmark: ThreadPool vs new Thread
var sw = System.Diagnostics.Stopwatch.StartNew();
var countdown = new CountdownEvent(1000);

for (int i = 0; i < 1000; i++)
{
    ThreadPool.QueueUserWorkItem(_ =>
    {
        // Minimal work
        countdown.Signal();
    });
}

countdown.Wait();
Console.WriteLine($"ThreadPool: {sw.ElapsedMilliseconds}ms");

sw.Restart();
countdown.Reset(1000);

for (int i = 0; i < 1000; i++)
{
    new Thread(() => countdown.Signal()) { IsBackground = true }.Start();
}

countdown.Wait();
Console.WriteLine($"New threads: {sw.ElapsedMilliseconds}ms");
// ThreadPool is typically 10-100x faster
```

## 3. Task Parallel Library (TPL)

The TPL provides high-level constructs for parallel execution. `Parallel.For` and `Parallel.ForEach` automatically partition work across available cores.

### 3.1 Parallel.For

```csharp
double[] data = new double[10_000_000];
Random random = new();
for (int i = 0; i < data.Length; i++)
    data[i] = random.NextDouble();

// Sequential
var sw = System.Diagnostics.Stopwatch.StartNew();
double[] results = new double[data.Length];
for (int i = 0; i < data.Length; i++)
    results[i] = Math.Sqrt(data[i]);
Console.WriteLine($"Sequential: {sw.ElapsedMilliseconds}ms");

// Parallel
sw.Restart();
Parallel.For(0, data.Length, i =>
{
    results[i] = Math.Sqrt(data[i]);
});
Console.WriteLine($"Parallel:   {sw.ElapsedMilliseconds}ms");
```

### 3.2 Parallel.ForEach

```csharp
List<string> filePaths = Directory.GetFiles("/data", "*.csv").ToList();

Parallel.ForEach(filePaths, new ParallelOptions
{
    MaxDegreeOfParallelism = Environment.ProcessorCount
}, path =>
{
    string content = File.ReadAllText(path);
    int lines = content.Count(c => c == '\n');
    Console.WriteLine($"{Path.GetFileName(path)}: {lines} lines");
});
```

### 3.3 Parallel.ForEachAsync (C# 10+)

```csharp
IEnumerable<string> urls = GetUrls();

await Parallel.ForEachAsync(urls, new ParallelOptions
{
    MaxDegreeOfParallelism = 10
}, async (url, cancellationToken) =>
{
    using var client = new HttpClient();
    string content = await client.GetStringAsync(url, cancellationToken);
    Console.WriteLine($"Downloaded {url}: {content.Length} chars");
});
```

### 3.4 Parallel LINQ (PLINQ)

```csharp
int[] numbers = Enumerable.Range(1, 10_000_000).ToArray();

// Parallel computation
long sum = numbers
    .AsParallel()
    .WithDegreeOfParallelism(4)
    .Where(n => n % 2 == 0)
    .Sum(n => (long)n);

Console.WriteLine($"Sum of even numbers: {sum}");
```

## 4. lock Statement and Monitor

### 4.1 The lock Statement

The `lock` statement provides mutual exclusion — only one thread can hold the lock at a time.

```csharp
public class BankAccount
{
    private readonly object _syncLock = new();
    private decimal _balance;

    public decimal Balance
    {
        get { lock (_syncLock) { return _balance; } }
    }

    public void Deposit(decimal amount)
    {
        lock (_syncLock)
        {
            _balance += amount;
        }
    }

    public bool Withdraw(decimal amount)
    {
        lock (_syncLock)
        {
            if (_balance >= amount)
            {
                _balance -= amount;
                return true;
            }
            return false;
        }
    }

    public void Transfer(BankAccount target, decimal amount)
    {
        // Always lock in a consistent order to prevent deadlocks
        object first = GetHashCode() < target.GetHashCode() ? _syncLock : target._syncLock;
        object second = first == _syncLock ? target._syncLock : _syncLock;

        lock (first)
        {
            lock (second)
            {
                if (_balance >= amount)
                {
                    _balance -= amount;
                    target._balance += amount;
                }
            }
        }
    }
}
```

### 4.2 Monitor (Manual lock Control)

The `lock` statement compiles to `Monitor.Enter`/`Monitor.Exit`. Using `Monitor` directly gives you more control.

```csharp
public class TimedLock
{
    private readonly object _lock = new();

    public bool TryExecute(Action action, TimeSpan timeout)
    {
        bool acquired = false;
        try
        {
            acquired = Monitor.TryEnter(_lock, timeout);
            if (acquired)
            {
                action();
                return true;
            }
            return false;
        }
        finally
        {
            if (acquired)
                Monitor.Exit(_lock);
        }
    }
}
```

### 4.3 Monitor.Wait and Monitor.Pulse

```csharp
public class BoundedBuffer<T>
{
    private readonly Queue<T> _queue = new();
    private readonly object _lock = new();
    private readonly int _maxSize;

    public BoundedBuffer(int maxSize) => _maxSize = maxSize;

    public void Enqueue(T item)
    {
        lock (_lock)
        {
            while (_queue.Count >= _maxSize)
                Monitor.Wait(_lock); // Release lock and wait

            _queue.Enqueue(item);
            Monitor.PulseAll(_lock); // Notify waiting consumers
        }
    }

    public T Dequeue()
    {
        lock (_lock)
        {
            while (_queue.Count == 0)
                Monitor.Wait(_lock); // Release lock and wait

            T item = _queue.Dequeue();
            Monitor.PulseAll(_lock); // Notify waiting producers
            return item;
        }
    }
}
```

## 5. SemaphoreSlim and Mutex

### 5.1 SemaphoreSlim

A semaphore limits the number of threads that can access a resource concurrently. Unlike `lock` (which allows exactly one), a semaphore allows N.

```csharp
public class ConnectionPool
{
    private readonly SemaphoreSlim _semaphore;

    public ConnectionPool(int maxConnections)
    {
        _semaphore = new SemaphoreSlim(maxConnections, maxConnections);
    }

    public async Task<T> ExecuteAsync<T>(Func<Task<T>> operation)
    {
        await _semaphore.WaitAsync();
        try
        {
            return await operation();
        }
        finally
        {
            _semaphore.Release();
        }
    }

    public int AvailableConnections => _semaphore.CurrentCount;
}
```

```csharp
// Usage: limit to 5 concurrent database connections
var pool = new ConnectionPool(maxConnections: 5);

var tasks = Enumerable.Range(0, 20).Select(async i =>
{
    string result = await pool.ExecuteAsync(async () =>
    {
        await Task.Delay(100); // Simulate DB query
        return $"Query {i} result";
    });
    Console.WriteLine(result);
});

await Task.WhenAll(tasks);
```

### 5.2 Mutex

A `Mutex` is a system-wide synchronization primitive. Use it when you need to coordinate between processes (not just threads).

```csharp
public class SingleInstanceApp
{
    private static Mutex? _mutex;

    public static bool TryAcquireLock(string appName)
    {
        _mutex = new Mutex(initiallyOwned: false, name: $"Global\\{appName}");

        try
        {
            return _mutex.WaitOne(TimeSpan.Zero);
        }
        catch (AbandonedMutexException)
        {
            // Previous instance crashed — we now own the mutex
            return true;
        }
    }

    public static void ReleaseLock()
    {
        _mutex?.ReleaseMutex();
        _mutex?.Dispose();
    }
}
```

## 6. Concurrent Collections

The `System.Collections.Concurrent` namespace provides thread-safe collection types that avoid the need for external locking.

### 6.1 ConcurrentDictionary

```csharp
var wordCounts = new ConcurrentDictionary<string, int>();

Parallel.ForEach(documents, doc =>
{
    foreach (string word in doc.Split(' ', StringSplitOptions.RemoveEmptyEntries))
    {
        string lower = word.ToLowerInvariant();
        wordCounts.AddOrUpdate(lower, 1, (key, oldValue) => oldValue + 1);
    }
});

// GetOrAdd with factory
var cache = new ConcurrentDictionary<string, ExpensiveObject>();
ExpensiveObject obj = cache.GetOrAdd("key", key => new ExpensiveObject(key));
```

### 6.2 ConcurrentQueue and ConcurrentStack

```csharp
var queue = new ConcurrentQueue<WorkItem>();

// Producer thread
queue.Enqueue(new WorkItem("task1"));
queue.Enqueue(new WorkItem("task2"));

// Consumer thread
if (queue.TryDequeue(out WorkItem? item))
{
    Process(item);
}

Console.WriteLine($"Queue count: {queue.Count}");
```

### 6.3 ConcurrentBag

`ConcurrentBag<T>` is optimized for scenarios where the same thread produces and consumes items (e.g., parallel loops collecting results).

```csharp
var results = new ConcurrentBag<ProcessingResult>();

Parallel.ForEach(inputData, item =>
{
    ProcessingResult result = Process(item);
    results.Add(result);
});

Console.WriteLine($"Processed {results.Count} items");

// Convert to list for further processing
List<ProcessingResult> sortedResults = results
    .OrderBy(r => r.Timestamp)
    .ToList();
```

### 6.4 BlockingCollection

```csharp
using var collection = new BlockingCollection<int>(boundedCapacity: 10);

// Producer
Task producer = Task.Run(() =>
{
    for (int i = 0; i < 50; i++)
    {
        collection.Add(i);
        Console.WriteLine($"Produced: {i}");
    }
    collection.CompleteAdding();
});

// Consumer
Task consumer = Task.Run(() =>
{
    foreach (int item in collection.GetConsumingEnumerable())
    {
        Console.WriteLine($"Consumed: {item}");
        Thread.Sleep(50); // Simulate processing
    }
});

await Task.WhenAll(producer, consumer);
```

## 7. Interlocked Class for Atomic Operations

The `Interlocked` class provides atomic operations for variables shared between threads, avoiding locks for simple counters and flags.

```csharp
public class AtomicCounter
{
    private long _count;

    public long Count => Interlocked.Read(ref _count);

    public void Increment() => Interlocked.Increment(ref _count);

    public void Decrement() => Interlocked.Decrement(ref _count);

    public void Add(long value) => Interlocked.Add(ref _count, value);

    public long Reset()
    {
        return Interlocked.Exchange(ref _count, 0);
    }

    public bool TryUpdate(long expected, long newValue)
    {
        return Interlocked.CompareExchange(ref _count, newValue, expected) == expected;
    }
}
```

```csharp
// Lock-free maximum tracker
public class AtomicMax
{
    private long _max = long.MinValue;

    public long Max => Interlocked.Read(ref _max);

    public void Update(long value)
    {
        long current = Interlocked.Read(ref _max);
        while (value > current)
        {
            long previous = Interlocked.CompareExchange(ref _max, value, current);
            if (previous == current)
                break; // Successfully updated
            current = previous; // Retry with the new current value
        }
    }
}
```

## 8. ReaderWriterLockSlim

When reads far outnumber writes, `ReaderWriterLockSlim` allows concurrent reads while ensuring exclusive access for writes.

```csharp
public class ThreadSafeCache<TKey, TValue> where TKey : notnull
{
    private readonly Dictionary<TKey, TValue> _cache = new();
    private readonly ReaderWriterLockSlim _lock = new();

    public TValue? Get(TKey key)
    {
        _lock.EnterReadLock();
        try
        {
            return _cache.TryGetValue(key, out TValue? value) ? value : default;
        }
        finally
        {
            _lock.ExitReadLock();
        }
    }

    public void Set(TKey key, TValue value)
    {
        _lock.EnterWriteLock();
        try
        {
            _cache[key] = value;
        }
        finally
        {
            _lock.ExitWriteLock();
        }
    }

    public TValue GetOrAdd(TKey key, Func<TKey, TValue> factory)
    {
        _lock.EnterUpgradeableReadLock();
        try
        {
            if (_cache.TryGetValue(key, out TValue? existing))
                return existing;

            _lock.EnterWriteLock();
            try
            {
                // Double-check after acquiring write lock
                if (_cache.TryGetValue(key, out existing))
                    return existing;

                TValue value = factory(key);
                _cache[key] = value;
                return value;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }
        finally
        {
            _lock.ExitUpgradeableReadLock();
        }
    }

    public int Count
    {
        get
        {
            _lock.EnterReadLock();
            try { return _cache.Count; }
            finally { _lock.ExitReadLock(); }
        }
    }
}
```

## 9. System.Threading.Channels

Channels provide an efficient, bounded, producer-consumer data structure — think of them as modern, async-friendly replacements for `BlockingCollection`.

### 9.1 Bounded and Unbounded Channels

```csharp
using System.Threading.Channels;

// Bounded channel: blocks producers when full
Channel<string> bounded = Channel.CreateBounded<string>(new BoundedChannelOptions(100)
{
    FullMode = BoundedChannelFullMode.Wait, // Default: producers wait
    SingleReader = false,
    SingleWriter = false
});

// Unbounded channel: never blocks producers (memory is the limit)
Channel<string> unbounded = Channel.CreateUnbounded<string>();
```

### 9.2 Writing and Reading

```csharp
Channel<int> channel = Channel.CreateBounded<int>(10);

// Producer
async Task ProduceAsync(ChannelWriter<int> writer)
{
    for (int i = 0; i < 100; i++)
    {
        await writer.WriteAsync(i);
        Console.WriteLine($"Wrote: {i}");
    }
    writer.Complete(); // Signal that no more items will be written
}

// Consumer
async Task ConsumeAsync(ChannelReader<int> reader)
{
    await foreach (int item in reader.ReadAllAsync())
    {
        Console.WriteLine($"Read: {item}");
        await Task.Delay(10); // Simulate processing
    }
}

// Run both
await Task.WhenAll(
    ProduceAsync(channel.Writer),
    ConsumeAsync(channel.Reader));
```

### 9.3 Multiple Consumers

```csharp
Channel<WorkItem> channel = Channel.CreateBounded<WorkItem>(1000);

// Start multiple consumer tasks
int consumerCount = Environment.ProcessorCount;
var consumers = Enumerable.Range(0, consumerCount)
    .Select(id => Task.Run(async () =>
    {
        await foreach (WorkItem item in channel.Reader.ReadAllAsync())
        {
            Console.WriteLine($"Consumer {id} processing: {item.Name}");
            await ProcessAsync(item);
        }
    }))
    .ToArray();

// Producer writes items
foreach (var item in GetWorkItems())
    await channel.Writer.WriteAsync(item);

channel.Writer.Complete();
await Task.WhenAll(consumers);
```

## 10. Producer-Consumer Pattern

The producer-consumer pattern decouples data generation from data processing, allowing each to run at its own pace.

### 10.1 Pipeline with Channels

```csharp
public static class Pipeline
{
    public static ChannelReader<TOut> Transform<TIn, TOut>(
        ChannelReader<TIn> input,
        Func<TIn, TOut> transform,
        int capacity = 100)
    {
        var output = Channel.CreateBounded<TOut>(capacity);

        Task.Run(async () =>
        {
            try
            {
                await foreach (TIn item in input.ReadAllAsync())
                {
                    TOut result = transform(item);
                    await output.Writer.WriteAsync(result);
                }
            }
            finally
            {
                output.Writer.Complete();
            }
        });

        return output.Reader;
    }

    public static ChannelReader<TOut> TransformMany<TIn, TOut>(
        ChannelReader<TIn> input,
        Func<TIn, IEnumerable<TOut>> transform,
        int capacity = 100)
    {
        var output = Channel.CreateBounded<TOut>(capacity);

        Task.Run(async () =>
        {
            try
            {
                await foreach (TIn item in input.ReadAllAsync())
                {
                    foreach (TOut result in transform(item))
                        await output.Writer.WriteAsync(result);
                }
            }
            finally
            {
                output.Writer.Complete();
            }
        });

        return output.Reader;
    }
}
```

```csharp
// Usage: file processing pipeline
// Stage 1: Read file paths
Channel<string> pathChannel = Channel.CreateBounded<string>(50);
_ = Task.Run(async () =>
{
    foreach (string path in Directory.GetFiles("/data", "*.log"))
        await pathChannel.Writer.WriteAsync(path);
    pathChannel.Writer.Complete();
});

// Stage 2: Read file contents
ChannelReader<string> contents = Pipeline.Transform(
    pathChannel.Reader,
    path => File.ReadAllText(path));

// Stage 3: Extract lines
ChannelReader<string> lines = Pipeline.TransformMany(
    contents,
    content => content.Split('\n'));

// Stage 4: Process each line
await foreach (string line in lines.ReadAllAsync())
{
    if (line.Contains("ERROR"))
        Console.WriteLine($"Found error: {line}");
}
```

## 11. Thread Safety Best Practices

### 11.1 Immutability

Immutable objects are inherently thread-safe because their state never changes.

```csharp
// Immutable configuration — no synchronization needed
public record AppConfig(
    string ConnectionString,
    int MaxRetries,
    TimeSpan Timeout);

// Thread-safe state update via immutable replacement
public class ConfigManager
{
    private volatile AppConfig _config;

    public ConfigManager(AppConfig initial) => _config = initial;

    public AppConfig Current => _config;

    public void Update(Func<AppConfig, AppConfig> updater)
    {
        // Atomic replacement using Interlocked
        AppConfig current, updated;
        do
        {
            current = _config;
            updated = updater(current);
        }
        while (Interlocked.CompareExchange(ref _config, updated, current) != current);
    }
}
```

### 11.2 Avoiding Shared Mutable State

```csharp
// BAD: shared mutable list
List<int> results = new();
Parallel.For(0, 1000, i =>
{
    results.Add(Compute(i)); // Race condition!
});

// GOOD: thread-local accumulation, then merge
Parallel.For(0, 1000,
    () => new List<int>(), // Thread-local state
    (i, state, localList) =>
    {
        localList.Add(Compute(i));
        return localList;
    },
    localList =>
    {
        lock (results) { results.AddRange(localList); }
    });
```

### 11.3 The volatile Keyword

```csharp
public class StoppableWorker
{
    private volatile bool _shouldStop;

    public void RequestStop() => _shouldStop = true;

    public void Run()
    {
        while (!_shouldStop)
        {
            // Do work...
            // Without volatile, the JIT might cache _shouldStop in a register
            // and never see the updated value
        }
    }
}
```

## 12. Practical Example: Concurrent Web Scraper

This example combines channels, throttled parallelism, concurrent collections, and cancellation.

```csharp
using System.Collections.Concurrent;
using System.Net.Http;
using System.Threading.Channels;

public class ConcurrentWebScraper
{
    private readonly HttpClient _client;
    private readonly int _maxConcurrency;
    private readonly ConcurrentDictionary<string, ScrapeResult> _results = new();
    private readonly ConcurrentDictionary<string, byte> _visited = new();

    public ConcurrentWebScraper(int maxConcurrency = 10)
    {
        _maxConcurrency = maxConcurrency;
        _client = new HttpClient
        {
            Timeout = TimeSpan.FromSeconds(15)
        };
        _client.DefaultRequestHeaders.UserAgent.ParseAdd("CSharpScraper/1.0");
    }

    public async Task<IReadOnlyDictionary<string, ScrapeResult>> ScrapeAsync(
        IEnumerable<string> seedUrls,
        int maxDepth = 2,
        CancellationToken cancellationToken = default)
    {
        var urlChannel = Channel.CreateBounded<(string Url, int Depth)>(1000);
        var throttle = new SemaphoreSlim(_maxConcurrency);

        // Seed the channel
        foreach (string url in seedUrls)
        {
            _visited.TryAdd(url, 0);
            await urlChannel.Writer.WriteAsync((url, 0), cancellationToken);
        }

        var activeTasks = new ConcurrentBag<Task>();
        int pendingCount = _visited.Count;

        // Worker loop
        var workers = Enumerable.Range(0, _maxConcurrency).Select(_ => Task.Run(async () =>
        {
            await foreach (var (url, depth) in urlChannel.Reader.ReadAllAsync(cancellationToken))
            {
                await throttle.WaitAsync(cancellationToken);
                try
                {
                    ScrapeResult result = await ScrapeOneAsync(url, cancellationToken);
                    _results[url] = result;

                    // Discover new URLs
                    if (depth < maxDepth && result.Links is not null)
                    {
                        foreach (string link in result.Links)
                        {
                            if (_visited.TryAdd(link, 0))
                            {
                                Interlocked.Increment(ref pendingCount);
                                await urlChannel.Writer.WriteAsync(
                                    (link, depth + 1), cancellationToken);
                            }
                        }
                    }
                }
                finally
                {
                    throttle.Release();
                    if (Interlocked.Decrement(ref pendingCount) == 0)
                        urlChannel.Writer.Complete();
                }
            }
        }, cancellationToken)).ToArray();

        await Task.WhenAll(workers);
        return _results;
    }

    private async Task<ScrapeResult> ScrapeOneAsync(
        string url, CancellationToken token)
    {
        try
        {
            string html = await _client.GetStringAsync(url, token);
            var links = ExtractLinks(html, url);

            return new ScrapeResult
            {
                Url = url,
                Success = true,
                ContentLength = html.Length,
                Links = links,
                Title = ExtractTitle(html)
            };
        }
        catch (Exception ex)
        {
            return new ScrapeResult
            {
                Url = url,
                Success = false,
                Error = ex.Message
            };
        }
    }

    private static string? ExtractTitle(string html)
    {
        int start = html.IndexOf("<title>", StringComparison.OrdinalIgnoreCase);
        if (start < 0) return null;
        start += 7;
        int end = html.IndexOf("</title>", start, StringComparison.OrdinalIgnoreCase);
        return end < 0 ? null : html[start..end].Trim();
    }

    private static List<string> ExtractLinks(string html, string baseUrl)
    {
        var links = new List<string>();
        int idx = 0;
        while ((idx = html.IndexOf("href=\"", idx, StringComparison.OrdinalIgnoreCase)) >= 0)
        {
            idx += 6;
            int end = html.IndexOf('"', idx);
            if (end < 0) break;

            string href = html[idx..end];
            if (Uri.TryCreate(new Uri(baseUrl), href, out Uri? absolute)
                && (absolute.Scheme == "http" || absolute.Scheme == "https"))
            {
                links.Add(absolute.GetLeftPart(UriPartial.Path));
            }
        }
        return links.Distinct().Take(20).ToList();
    }
}

public record ScrapeResult
{
    public string Url { get; init; } = "";
    public bool Success { get; init; }
    public int ContentLength { get; init; }
    public string? Title { get; init; }
    public List<string>? Links { get; init; }
    public string? Error { get; init; }
}
```

```csharp
// Usage
var scraper = new ConcurrentWebScraper(maxConcurrency: 5);
using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));

var results = await scraper.ScrapeAsync(
    new[] { "https://example.com" },
    maxDepth: 1,
    cts.Token);

foreach (var (url, result) in results.OrderBy(r => r.Key))
{
    string status = result.Success ? "OK" : "FAIL";
    Console.WriteLine($"[{status}] {url} — {result.Title ?? "No title"} ({result.ContentLength} bytes)");
}
```

## 13. Practice Problems

1. **Thread-Safe Counter**: Implement a `SafeCounter` class with `Increment()`, `Decrement()`, and `Value` using three approaches: (a) `lock`, (b) `Interlocked`, and (c) `ReaderWriterLockSlim`. Write a benchmark that runs 1,000,000 increments from 8 threads for each approach and compares elapsed time.

2. **Parallel Image Processor**: Using `Parallel.For`, process a 2D array of pixel values (simulate a grayscale image as `byte[width, height]`). Apply a 3x3 box blur filter. Ensure boundary pixels are handled correctly. Compare sequential vs parallel performance for a 4096x4096 image.

3. **Producer-Consumer with Channels**: Build a three-stage pipeline using `System.Threading.Channels`: Stage 1 reads file paths from a directory; Stage 2 reads each file and counts word frequencies; Stage 3 merges all word counts into a global `ConcurrentDictionary<string, int>` and writes the top 100 words to a CSV file. Use bounded channels with appropriate capacity.

4. **Dining Philosophers**: Implement the classic dining philosophers problem with 5 philosophers and 5 forks. Use `SemaphoreSlim` for forks. Implement a deadlock-free solution using resource ordering. Print actions (thinking, hungry, eating) with timestamps and verify no deadlock occurs over 30 seconds.

5. **Thread-Safe Observable Collection**: Create a `ConcurrentObservableList<T>` that supports `Add`, `Remove`, `Contains`, and `GetSnapshot()` (returns a read-only copy). Use `ReaderWriterLockSlim` for synchronization. Raise an event `CollectionChanged` after mutations. Write a test where 4 threads add items and 2 threads enumerate snapshots concurrently.
