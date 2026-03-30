# Performance and Profiling

**Previous**: [Interop and Unsafe Code](./15_Interop_and_Unsafe.md) | **Next**: [Capstone: Minimal Web API](./17_Capstone_Web_API.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Set up and use BenchmarkDotNet for micro-benchmarking
2. Write reliable benchmarks with proper methodology
3. Interpret benchmark results including mean, median, and allocations
4. Use `dotnet-counters`, `dotnet-trace`, and `dotnet-dump` for runtime diagnostics
5. Understand GC generations and minimize allocation pressure
6. Apply object pooling to reduce GC overhead
7. Optimize string operations for high-performance scenarios
8. Choose between value types and reference types based on performance needs
9. Identify and optimize hot paths in your applications

---

Performance optimization in .NET is a disciplined process: measure first, identify bottlenecks, optimize targeted areas, and verify improvements with benchmarks. Premature optimization is counterproductive, but understanding how the runtime manages memory, how the garbage collector works, and which patterns cause unnecessary allocations is essential for building responsive, scalable applications. This lesson covers the tools and techniques for measuring, analyzing, and improving .NET application performance.

## 1. BenchmarkDotNet Setup and Usage

### 1.1 Getting Started

BenchmarkDotNet is the standard micro-benchmarking library for .NET:

```bash
# Create a new console project for benchmarks
dotnet new console -o MyBenchmarks
cd MyBenchmarks
dotnet add package BenchmarkDotNet
```

```csharp
// Program.cs
using BenchmarkDotNet.Running;

BenchmarkRunner.Run<StringBenchmarks>();
```

```csharp
// StringBenchmarks.cs
using BenchmarkDotNet.Attributes;

[MemoryDiagnoser]  // Track memory allocations
public class StringBenchmarks
{
    private readonly string[] _words = { "Hello", "World", "From", "BenchmarkDotNet" };

    [Benchmark(Baseline = true)]
    public string ConcatWithPlus()
    {
        string result = "";
        foreach (var word in _words)
            result += word + " ";
        return result;
    }

    [Benchmark]
    public string ConcatWithStringBuilder()
    {
        var sb = new System.Text.StringBuilder();
        foreach (var word in _words)
            sb.Append(word).Append(' ');
        return sb.ToString();
    }

    [Benchmark]
    public string ConcatWithJoin()
    {
        return string.Join(" ", _words);
    }
}
```

### 1.2 Running Benchmarks

```bash
# Always run benchmarks in Release mode!
dotnet run -c Release

# Run specific benchmark class
dotnet run -c Release -- --filter "*StringBenchmarks*"

# Export results
dotnet run -c Release -- --exporters json csv markdown
```

### 1.3 Sample Output

```
|             Method |      Mean |    Error |   StdDev | Ratio |  Gen0 | Allocated | Alloc Ratio |
|------------------- |----------:|---------:|---------:|------:|------:|----------:|------------:|
|    ConcatWithPlus  | 125.4 ns |  2.5 ns  |  2.3 ns  |  1.00 | 0.072 |     304 B |        1.00 |
| ConcatWithStringBu | 78.2 ns   |  1.3 ns  |  1.2 ns  |  0.62 | 0.038 |     160 B |        0.53 |
|    ConcatWithJoin  | 52.1 ns   |  0.8 ns  |  0.7 ns  |  0.42 | 0.024 |     104 B |        0.34 |
```

## 2. Writing Benchmarks

### 2.1 Benchmark Attributes

```csharp
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

[SimpleJob(RuntimeMoniker.Net80)]       // Run on .NET 8
[SimpleJob(RuntimeMoniker.Net90)]       // Also run on .NET 9
[MemoryDiagnoser]                       // Track allocations
[DisassemblyDiagnoser(maxDepth: 2)]     // Show JIT-generated assembly
[RankColumn]                            // Add rank column
public class SortingBenchmarks
{
    [Params(100, 1000, 10_000)]  // Run for each array size
    public int N;

    private int[] _data = null!;

    [GlobalSetup]  // Run once before all benchmarks
    public void Setup()
    {
        var rng = new Random(42);  // Fixed seed for reproducibility
        _data = Enumerable.Range(0, N).Select(_ => rng.Next()).ToArray();
    }

    [IterationSetup]  // Run before each benchmark iteration
    public void IterationSetup()
    {
        // Re-shuffle for each iteration since Sort mutates
        var rng = new Random(42);
        for (int i = _data.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (_data[i], _data[j]) = (_data[j], _data[i]);
        }
    }

    [Benchmark(Baseline = true)]
    public void ArraySort()
    {
        Array.Sort(_data);
    }

    [Benchmark]
    public void SpanSort()
    {
        _data.AsSpan().Sort();
    }

    [Benchmark]
    public int[] LinqOrderBy()
    {
        return _data.OrderBy(x => x).ToArray();
    }
}
```

### 2.2 Parameterized Benchmarks

```csharp
[MemoryDiagnoser]
public class CollectionBenchmarks
{
    [Params(10, 100, 1000)]
    public int Count;

    [ParamsSource(nameof(CollectionTypes))]
    public string CollectionType { get; set; } = "";

    public static IEnumerable<string> CollectionTypes
        => new[] { "List", "Array", "HashSet" };

    [Benchmark]
    public int LookupBenchmark()
    {
        int found = 0;
        var target = Count / 2;

        switch (CollectionType)
        {
            case "List":
                var list = Enumerable.Range(0, Count).ToList();
                if (list.Contains(target)) found++;
                break;
            case "Array":
                var array = Enumerable.Range(0, Count).ToArray();
                if (Array.IndexOf(array, target) >= 0) found++;
                break;
            case "HashSet":
                var set = new HashSet<int>(Enumerable.Range(0, Count));
                if (set.Contains(target)) found++;
                break;
        }

        return found;
    }
}
```

### 2.3 Benchmark Methodology Best Practices

```csharp
[MemoryDiagnoser]
public class MethodologyDemo
{
    // WRONG: Benchmark does nothing observable; JIT may eliminate it
    // [Benchmark]
    // public void BadBenchmark()
    // {
    //     int x = 42 * 100;  // Dead code — JIT removes this
    // }

    // CORRECT: Return the result so the JIT cannot eliminate the work
    [Benchmark]
    public int GoodBenchmark()
    {
        return 42 * 100;  // Must be returned or consumed
    }

    // WRONG: Includes setup cost in benchmark
    // [Benchmark]
    // public int BadSetup()
    // {
    //     var data = Enumerable.Range(0, 1000).ToArray();  // Setup in benchmark!
    //     return data.Sum();
    // }

    private int[] _data = null!;

    [GlobalSetup]
    public void Setup()
    {
        _data = Enumerable.Range(0, 1000).ToArray();  // Setup separated
    }

    // CORRECT: Only measures the operation, not setup
    [Benchmark]
    public int GoodSetup()
    {
        return _data.Sum();
    }
}
```

## 3. Interpreting Benchmark Results

### 3.1 Understanding the Columns

```
|        Method |  N  |       Mean |    Error |   StdDev |     Median | Ratio | RatioSD |   Gen0 |  Gen1 | Allocated |
|-------------- |----:|-----------:|---------:|---------:|-----------:|------:|--------:|-------:|------:|----------:|
|     ArraySort | 100 |   1.234 us | 0.023 us | 0.021 us |   1.230 us |  1.00 |    0.00 |      - |     - |         - |
|      SpanSort | 100 |   1.198 us | 0.018 us | 0.015 us |   1.195 us |  0.97 |    0.02 |      - |     - |         - |
|   LinqOrderBy | 100 |   4.567 us | 0.089 us | 0.083 us |   4.550 us |  3.70 |    0.08 | 0.1221 |     - |     512 B |
```

```csharp
// Key columns explained:
//
// Mean     - Average execution time across all iterations
// Error    - Half of the 99.9% confidence interval (statistical uncertainty)
// StdDev   - Standard deviation (how spread out the measurements are)
// Median   - Middle value (less affected by outliers than Mean)
// Ratio    - Performance relative to the Baseline benchmark (1.00 = same)
// RatioSD  - Standard deviation of the ratio
// Gen0     - Number of Gen 0 garbage collections per 1000 operations
// Gen1     - Number of Gen 1 garbage collections per 1000 operations
// Allocated - Total bytes allocated per single operation
//
// Units: ns = nanoseconds, us = microseconds, ms = milliseconds
//
// Rules of thumb:
// - If Error is > 10% of Mean, results are unreliable
// - Look at Allocated first for memory optimization
// - Gen0 > 0 means the benchmark triggers GC pressure
// - Compare Ratio to baseline, not absolute numbers
```

### 3.2 Analyzing Allocation Patterns

```csharp
[MemoryDiagnoser]
public class AllocationBenchmarks
{
    private readonly int[] _source = Enumerable.Range(0, 1000).ToArray();

    [Benchmark(Baseline = true)]
    public int[] LinqToArray()
    {
        // Allocates: LINQ iterator + final array
        return _source.Where(x => x % 2 == 0).ToArray();
    }

    [Benchmark]
    public int[] ManualFilter()
    {
        // Allocates: one List<int> (internal array resizes) + final array
        var result = new List<int>(_source.Length / 2);  // Pre-sized
        foreach (var x in _source)
        {
            if (x % 2 == 0)
                result.Add(x);
        }
        return result.ToArray();
    }

    [Benchmark]
    public int CountOnly()
    {
        // Zero allocations — just counting
        int count = 0;
        foreach (var x in _source)
        {
            if (x % 2 == 0)
                count++;
        }
        return count;
    }
}

// Typical results:
// LinqToArray:  ~2.5 us, 2.5 KB allocated
// ManualFilter: ~1.8 us, 2.1 KB allocated
// CountOnly:    ~0.3 us, 0 B allocated
```

## 4. dotnet-counters for Real-Time Metrics

### 4.1 Installation and Basic Usage

```bash
# Install the tool
dotnet tool install --global dotnet-counters

# Monitor a running process
dotnet-counters monitor --process-id 12345

# Monitor specific counters
dotnet-counters monitor --process-id 12345 \
  --counters System.Runtime,Microsoft.AspNetCore.Hosting

# Monitor by process name
dotnet-counters monitor --process-id $(pgrep -f MyApp)
```

### 4.2 Key Performance Counters

```bash
# System.Runtime counters:
# cpu-usage              - CPU usage percentage
# working-set            - Working set memory in MB
# gc-heap-size           - GC heap size in MB
# gen-0-gc-count         - Number of Gen 0 collections
# gen-1-gc-count         - Number of Gen 1 collections
# gen-2-gc-count         - Number of Gen 2 collections
# gen-0-size             - Gen 0 heap size
# gen-1-size             - Gen 1 heap size
# gen-2-size             - Gen 2 heap size
# loh-size               - Large Object Heap size
# alloc-rate             - Allocation rate in bytes/sec
# exception-count        - Number of exceptions thrown
# threadpool-thread-count - Thread pool thread count
# threadpool-queue-length - Thread pool work item queue length

# Collect counters to a file
dotnet-counters collect --process-id 12345 \
  --format csv \
  --output perf-counters.csv \
  --counters System.Runtime
```

### 4.3 Programmatic Event Counters

```csharp
using System.Diagnostics.Tracing;

// Create custom event counters for your application
[EventSource(Name = "MyApp.Performance")]
public sealed class AppEventSource : EventSource
{
    public static readonly AppEventSource Instance = new();

    private readonly IncrementingEventCounter _requestCounter;
    private readonly EventCounter _requestDuration;
    private readonly IncrementingPollingCounter _activeConnections;
    private int _connectionCount;

    private AppEventSource()
    {
        _requestCounter = new IncrementingEventCounter("request-count", this)
        {
            DisplayName = "Requests",
            DisplayRateTimeScale = TimeSpan.FromSeconds(1)
        };

        _requestDuration = new EventCounter("request-duration-ms", this)
        {
            DisplayName = "Request Duration (ms)"
        };

        _activeConnections = new IncrementingPollingCounter("active-connections", this,
            () => _connectionCount)
        {
            DisplayName = "Active Connections"
        };
    }

    public void RecordRequest(double durationMs)
    {
        _requestCounter.Increment();
        _requestDuration.WriteMetric(durationMs);
    }

    public void ConnectionOpened() => Interlocked.Increment(ref _connectionCount);
    public void ConnectionClosed() => Interlocked.Decrement(ref _connectionCount);
}

// Usage in your application:
// var sw = Stopwatch.StartNew();
// await HandleRequest();
// sw.Stop();
// AppEventSource.Instance.RecordRequest(sw.Elapsed.TotalMilliseconds);
```

## 5. dotnet-trace for Profiling

### 5.1 Collecting Traces

```bash
# Install the tool
dotnet tool install --global dotnet-trace

# Collect a CPU trace (default profile)
dotnet-trace collect --process-id 12345

# Collect with specific providers
dotnet-trace collect --process-id 12345 \
  --providers Microsoft-DotNETCore-SampleProfiler,Microsoft-Windows-DotNETRuntime

# Collect GC events
dotnet-trace collect --process-id 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x1:5

# Collect for a specific duration
dotnet-trace collect --process-id 12345 --duration 00:00:30

# Convert to SpeedScope format for browser viewing
dotnet-trace convert trace.nettrace --format speedscope
# Open the .speedscope.json file at https://www.speedscope.app/
```

### 5.2 Common Profiling Providers

```bash
# GC detailed events (allocations, collections, finalizers)
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x1:Verbose

# Thread pool events
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x10000:Informational

# Exception events
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x8000:Informational

# HTTP client events
dotnet-trace collect -p 12345 \
  --providers System.Net.Http

# Entity Framework Core events
dotnet-trace collect -p 12345 \
  --providers Microsoft.EntityFrameworkCore
```

## 6. dotnet-dump for Memory Analysis

### 6.1 Capturing and Analyzing Dumps

```bash
# Install the tool
dotnet tool install --global dotnet-dump

# Capture a dump
dotnet-dump collect --process-id 12345

# Analyze the dump
dotnet-dump analyze core_20250115_123456

# Inside the analyzer:
# > dumpheap -stat              # Show heap statistics
# > dumpheap -type System.String # Show all String objects
# > gcroot 0x7f8a1234           # Find what keeps an object alive
# > dumpobj 0x7f8a1234          # Inspect a specific object
# > eeheap -gc                  # GC heap segment information
# > threadpool                  # Thread pool info
# > exit
```

### 6.2 Common Memory Investigation Patterns

```csharp
// Finding memory leaks - common patterns:

// Pattern 1: Event handler leak
public class LeakyService
{
    // LEAK: subscriber holds a reference to publisher
    public void Subscribe(EventPublisher publisher)
    {
        publisher.DataReceived += OnDataReceived;  // Leak!
        // Must unsubscribe: publisher.DataReceived -= OnDataReceived;
    }

    private void OnDataReceived(object? sender, EventArgs e) { }
}

// Pattern 2: Static collection that grows forever
public static class Cache
{
    // LEAK: items are never removed
    private static readonly Dictionary<string, byte[]> _cache = new();

    public static void Add(string key, byte[] data)
    {
        _cache[key] = data;  // Grows forever!
    }

    // FIX: Use a bounded cache or weak references
}

// Pattern 3: Undisposed resources
public class ResourceLeak
{
    public void Process()
    {
        var stream = new FileStream("data.txt", FileMode.Open);  // LEAK!
        // Never disposed — finalizer will eventually clean up, but GC pressure increases

        // FIX:
        using var safeStream = new FileStream("data.txt", FileMode.Open);
    }
}
```

## 7. GC Generations and Allocation Pressure

### 7.1 Understanding GC Generations

```csharp
public class GcGenerations
{
    public static void Demonstrate()
    {
        // .NET GC uses generational collection:
        // Gen 0: Short-lived objects (collected most frequently, ~10ms pause)
        // Gen 1: Medium-lived objects (buffer between Gen 0 and Gen 2)
        // Gen 2: Long-lived objects (collected least frequently, can be expensive)
        // LOH:   Large Object Heap (objects >= 85,000 bytes, collected with Gen 2)

        var obj = new byte[100];
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 0

        GC.Collect(0);  // Promote surviving Gen 0 objects to Gen 1
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 1

        GC.Collect(1);  // Promote surviving Gen 1 objects to Gen 2
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 2

        // GC statistics
        Console.WriteLine($"Gen 0 collections: {GC.CollectionCount(0)}");
        Console.WriteLine($"Gen 1 collections: {GC.CollectionCount(1)}");
        Console.WriteLine($"Gen 2 collections: {GC.CollectionCount(2)}");
        Console.WriteLine($"Total memory: {GC.GetTotalMemory(false):N0} bytes");
        Console.WriteLine($"Total allocated: {GC.GetTotalAllocatedBytes():N0} bytes");

        // GCMemoryInfo for detailed information
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        Console.WriteLine($"Heap size: {info.HeapSizeBytes:N0} bytes");
        Console.WriteLine($"Committed: {info.TotalCommittedBytes:N0} bytes");
    }
}
```

### 7.2 Reducing Allocation Pressure

```csharp
public class AllocationReduction
{
    // BAD: Allocates a new array on every call
    public int[] GetEvenNumbers_Bad(int count)
    {
        return Enumerable.Range(0, count).Where(x => x % 2 == 0).ToArray();
    }

    // BETTER: Reuse a buffer
    private int[] _buffer = new int[1024];

    public Span<int> GetEvenNumbers_Good(int count)
    {
        if (count / 2 > _buffer.Length)
            _buffer = new int[count / 2 + 1];

        int index = 0;
        for (int i = 0; i < count; i++)
        {
            if (i % 2 == 0)
                _buffer[index++] = i;
        }
        return _buffer.AsSpan(0, index);
    }

    // BAD: Boxing value types
    public void BoxingExample_Bad()
    {
        object boxed = 42;             // Boxing: int -> object (heap allocation)
        int unboxed = (int)boxed;      // Unboxing
        Console.WriteLine(boxed);      // ToString() on boxed value
    }

    // GOOD: Avoid boxing
    public void BoxingExample_Good()
    {
        int value = 42;
        Console.WriteLine(value);      // No boxing: int.ToString() called directly
    }

    // BAD: LINQ in hot path (allocates iterators)
    public bool HasExpiredItems_Bad(List<Item> items)
    {
        return items.Any(i => i.ExpiresAt < DateTime.UtcNow);  // Allocates delegate + iterator
    }

    // GOOD: Manual loop (zero allocation)
    public bool HasExpiredItems_Good(List<Item> items)
    {
        var now = DateTime.UtcNow;
        for (int i = 0; i < items.Count; i++)
        {
            if (items[i].ExpiresAt < now)
                return true;
        }
        return false;
    }
}

public class Item
{
    public DateTime ExpiresAt { get; set; }
}
```

## 8. Object Pooling

### 8.1 ObjectPool<T> from Microsoft.Extensions

```csharp
using Microsoft.Extensions.ObjectPool;

// Simple pooling for StringBuilder
public class StringBuilderPoolExample
{
    private static readonly ObjectPool<StringBuilder> Pool =
        new DefaultObjectPoolProvider().CreateStringBuilderPool(
            initialCapacity: 256,
            maximumRetainedCapacity: 4096);

    public string BuildReport(IEnumerable<(string Name, decimal Value)> items)
    {
        var sb = Pool.Get();
        try
        {
            sb.AppendLine("=== Report ===");
            foreach (var (name, value) in items)
            {
                sb.AppendLine($"  {name}: {value:C}");
            }
            sb.AppendLine("==============");
            return sb.ToString();
        }
        finally
        {
            Pool.Return(sb);  // Returns to pool (and clears the StringBuilder)
        }
    }
}
```

### 8.2 Custom Object Pool

```csharp
using System.Collections.Concurrent;

public class ObjectPool<T> where T : class
{
    private readonly ConcurrentBag<T> _pool = new();
    private readonly Func<T> _factory;
    private readonly Action<T>? _reset;
    private readonly int _maxSize;
    private int _count;

    public ObjectPool(Func<T> factory, Action<T>? reset = null, int maxSize = 100)
    {
        _factory = factory;
        _reset = reset;
        _maxSize = maxSize;
    }

    public T Rent()
    {
        if (_pool.TryTake(out T? item))
        {
            Interlocked.Decrement(ref _count);
            return item;
        }
        return _factory();
    }

    public void Return(T item)
    {
        if (Interlocked.Increment(ref _count) <= _maxSize)
        {
            _reset?.Invoke(item);
            _pool.Add(item);
        }
        else
        {
            Interlocked.Decrement(ref _count);
            // Pool is full — let GC collect this instance
        }
    }
}

// Usage:
public class BufferPool
{
    private static readonly ObjectPool<byte[]> Pool = new(
        factory: () => new byte[4096],
        reset: buffer => Array.Clear(buffer),
        maxSize: 50);

    public static byte[] Rent() => Pool.Rent();
    public static void Return(byte[] buffer) => Pool.Return(buffer);
}

// var buffer = BufferPool.Rent();
// try
// {
//     // Use buffer...
// }
// finally
// {
//     BufferPool.Return(buffer);
// }
```

### 8.3 ArrayPool<T>

```csharp
using System.Buffers;

public class ArrayPoolExample
{
    public static double ComputeAverage(IEnumerable<double> source, int estimatedCount)
    {
        // Rent an array from the shared pool (may be larger than requested)
        double[] buffer = ArrayPool<double>.Shared.Rent(estimatedCount);
        try
        {
            int count = 0;
            foreach (double value in source)
            {
                if (count >= buffer.Length)
                {
                    // Need a bigger buffer
                    double[] newBuffer = ArrayPool<double>.Shared.Rent(buffer.Length * 2);
                    buffer.AsSpan(0, count).CopyTo(newBuffer);
                    ArrayPool<double>.Shared.Return(buffer);
                    buffer = newBuffer;
                }
                buffer[count++] = value;
            }

            double sum = 0;
            for (int i = 0; i < count; i++)
                sum += buffer[i];

            return count > 0 ? sum / count : 0;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(buffer, clearArray: true);
        }
    }
}
```

## 9. String Optimization

### 9.1 String Internment

```csharp
public class StringInterning
{
    public static void Demonstrate()
    {
        // String literals are automatically interned
        string a = "Hello";
        string b = "Hello";
        Console.WriteLine(ReferenceEquals(a, b));  // True (same object)

        // Runtime-created strings are NOT interned by default
        string c = new string(new char[] { 'H', 'e', 'l', 'l', 'o' });
        Console.WriteLine(ReferenceEquals(a, c));  // False (different objects)
        Console.WriteLine(a == c);                  // True (same value)

        // Manually intern a string
        string d = string.Intern(c);
        Console.WriteLine(ReferenceEquals(a, d));  // True

        // Check if a string is interned (without interning it)
        string? e = string.IsInterned(c);
        Console.WriteLine(e is not null);  // True (because "Hello" was already interned)
    }
}
```

### 9.2 string.Create for Zero-Allocation Formatting

```csharp
public class StringCreateExamples
{
    // string.Create builds the string in-place without intermediate allocations
    public static string FormatHex(ReadOnlySpan<byte> bytes)
    {
        return string.Create(bytes.Length * 2, bytes.ToArray(), (chars, state) =>
        {
            for (int i = 0; i < state.Length; i++)
            {
                byte b = state[i];
                chars[i * 2] = GetHexChar(b >> 4);
                chars[i * 2 + 1] = GetHexChar(b & 0x0F);
            }
        });

        static char GetHexChar(int value) =>
            (char)(value < 10 ? '0' + value : 'a' + value - 10);
    }

    // Format a timestamp efficiently
    public static string FormatTimestamp(DateTime dt)
    {
        // "2025-01-15T10:30:45"
        return string.Create(19, dt, (chars, dt) =>
        {
            dt.TryFormat(chars, out _, "yyyy-MM-ddTHH:mm:ss");
        });
    }

    // Padding and alignment
    public static string PadCenter(string text, int totalWidth, char padChar = ' ')
    {
        if (text.Length >= totalWidth) return text;
        int leftPad = (totalWidth - text.Length) / 2;
        int rightPad = totalWidth - text.Length - leftPad;

        return string.Create(totalWidth, (text, leftPad, rightPad, padChar), (chars, state) =>
        {
            chars[..state.leftPad].Fill(state.padChar);
            state.text.AsSpan().CopyTo(chars[state.leftPad..]);
            chars[(state.leftPad + state.text.Length)..].Fill(state.padChar);
        });
    }
}
```

### 9.3 StringBuilder Best Practices

```csharp
using System.Text;

public class StringBuilderOptimization
{
    // Pre-size when you know approximate output length
    public static string BuildCsv(List<string[]> rows)
    {
        int estimatedLength = rows.Count * 100;  // Estimate ~100 chars per row
        var sb = new StringBuilder(estimatedLength);

        foreach (var row in rows)
        {
            for (int i = 0; i < row.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(row[i]);
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    // Use Append instead of AppendFormat for performance
    public static string BuildTable(List<(string Name, int Age, string City)> people)
    {
        var sb = new StringBuilder(people.Count * 50);

        // SLOWER:
        // sb.AppendFormat("| {0,-20} | {1,4} | {2,-15} |", p.Name, p.Age, p.City);

        // FASTER: direct Append calls
        foreach (var (name, age, city) in people)
        {
            sb.Append("| ")
              .Append(name.PadRight(20))
              .Append(" | ")
              .Append(age.ToString().PadLeft(4))
              .Append(" | ")
              .Append(city.PadRight(15))
              .AppendLine(" |");
        }

        return sb.ToString();
    }
}
```

## 10. Value Types vs Reference Types Performance

### 10.1 Stack vs Heap Allocation

```csharp
// Reference type: allocated on the HEAP
public class PointClass
{
    public double X { get; set; }
    public double Y { get; set; }
}

// Value type: allocated on the STACK (when local variable)
public struct PointStruct
{
    public double X { get; set; }
    public double Y { get; set; }
}

// readonly struct: compiler-enforced immutability, no defensive copies
public readonly struct PointReadonly
{
    public double X { get; init; }
    public double Y { get; init; }

    public double Length() => Math.Sqrt(X * X + Y * Y);
}

[MemoryDiagnoser]
public class ValueVsReferenceBenchmark
{
    [Benchmark]
    public double CreateManyClasses()
    {
        double sum = 0;
        for (int i = 0; i < 10_000; i++)
        {
            var p = new PointClass { X = i, Y = i * 2 };  // Heap allocation
            sum += p.X + p.Y;
        }
        return sum;
    }

    [Benchmark]
    public double CreateManyStructs()
    {
        double sum = 0;
        for (int i = 0; i < 10_000; i++)
        {
            var p = new PointStruct { X = i, Y = i * 2 };  // Stack — no allocation
            sum += p.X + p.Y;
        }
        return sum;
    }
}

// Typical results:
// CreateManyClasses:  ~50 us,  ~320 KB allocated (10,000 heap objects)
// CreateManyStructs:  ~15 us,  0 B allocated
```

### 10.2 When to Use Structs vs Classes

```csharp
// Use STRUCT when:
// 1. Small (< ~16 bytes ideally, up to ~64 bytes)
// 2. Frequently created and destroyed (high allocation rate)
// 3. Logically represents a single value (like Point, Color, DateTime)
// 4. Immutable (use readonly struct)

public readonly struct Color
{
    public byte R { get; init; }
    public byte G { get; init; }
    public byte B { get; init; }
    public byte A { get; init; }
}

// Use CLASS when:
// 1. Large (many fields)
// 2. Long-lived (cached, stored in collections)
// 3. Needs inheritance/polymorphism
// 4. Needs reference semantics (multiple variables share the same instance)

public class CustomerProfile
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string Email { get; set; } = "";
    public List<Order> Orders { get; set; } = new();
    public Address? ShippingAddress { get; set; }
    // Large object — class is appropriate
}

// Use record struct for value-semantic DTOs
public readonly record struct Temperature(double Value, string Unit)
{
    public double ToCelsius() => Unit switch
    {
        "F" => (Value - 32) * 5 / 9,
        "K" => Value - 273.15,
        _ => Value
    };
}
```

## 11. Hot Path Optimization Techniques

### 11.1 Identifying Hot Paths

```csharp
// A "hot path" is code that executes frequently and is performance-critical.
// Examples:
// - Inner loops of algorithms
// - Request handling in web servers
// - Serialization/deserialization in high-throughput systems
// - Message processing in event-driven systems

// Technique 1: Avoid allocations in hot paths
public class HotPathExample
{
    // Cold path: called once during startup
    public void Initialize()
    {
        // Allocations here are fine
        var config = LoadConfiguration();
        _lookup = BuildLookup(config);
    }

    private Dictionary<string, int> _lookup = new();

    // Hot path: called per request (thousands/sec)
    public int ProcessRequest(ReadOnlySpan<char> key)
    {
        // No allocations! key is a span, not a string
        foreach (var (k, v) in _lookup)
        {
            if (key.SequenceEqual(k))
                return v;
        }
        return -1;
    }

    private Dictionary<string, int> BuildLookup(object config) => new();
    private object LoadConfiguration() => new();
}
```

### 11.2 Branch Prediction and Data Layout

```csharp
public class DataLayoutOptimization
{
    // SLOW: Poor data locality (array of objects = array of pointers to scattered heap)
    public static double SumClasses(PointClass[] points)
    {
        double sum = 0;
        foreach (var p in points)
            sum += p.X + p.Y;  // Each access follows a pointer to the heap
        return sum;
    }

    // FAST: Good data locality (struct array = contiguous memory block)
    public static double SumStructs(PointStruct[] points)
    {
        double sum = 0;
        foreach (var p in points)
            sum += p.X + p.Y;  // Sequential memory access, CPU cache-friendly
        return sum;
    }

    // FASTEST: Structure of Arrays (SoA) for SIMD-friendly access
    public static double SumSoA(double[] xs, double[] ys, int count)
    {
        double sum = 0;
        for (int i = 0; i < count; i++)
            sum += xs[i] + ys[i];
        return sum;
    }
}
```

### 11.3 Avoiding Common Performance Pitfalls

```csharp
public class PerformancePitfalls
{
    // Pitfall 1: Dictionary lookup with string keys (case-insensitive)
    // SLOW: ToLower allocates a new string on every lookup
    public bool LookupSlow(Dictionary<string, int> dict, string key)
    {
        return dict.ContainsKey(key.ToLower());  // Allocates!
    }

    // FAST: Use a case-insensitive comparer when creating the dictionary
    public Dictionary<string, int> CreateFastDict()
    {
        return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    }

    // Pitfall 2: Excessive exception throwing in normal flow
    // SLOW: Exceptions are very expensive
    public int ParseSlow(string input)
    {
        try { return int.Parse(input); }
        catch { return 0; }
    }

    // FAST: Use TryParse
    public int ParseFast(string input)
    {
        return int.TryParse(input, out int result) ? result : 0;
    }

    // Pitfall 3: Concatenating in loops
    // SLOW: O(n^2) string allocations
    public string JoinSlow(string[] items)
    {
        string result = "";
        foreach (var item in items)
            result += item + ",";
        return result;
    }

    // FAST: O(n) with single allocation
    public string JoinFast(string[] items)
    {
        return string.Join(",", items);
    }
}
```

## 12. Practical Example: Benchmarking and Optimizing a Data Processor

### 12.1 The Baseline Implementation

```csharp
public record SensorReading(string SensorId, DateTime Timestamp, double Value);

public class DataProcessor_V1
{
    public Dictionary<string, double> ComputeAverages(List<SensorReading> readings)
    {
        return readings
            .GroupBy(r => r.SensorId)
            .ToDictionary(
                g => g.Key,
                g => g.Average(r => r.Value));
    }

    public List<SensorReading> FilterOutliers(List<SensorReading> readings, double stdDevThreshold)
    {
        var grouped = readings.GroupBy(r => r.SensorId);
        var result = new List<SensorReading>();

        foreach (var group in grouped)
        {
            var values = group.Select(r => r.Value).ToList();
            double mean = values.Average();
            double stdDev = Math.Sqrt(values.Average(v => (v - mean) * (v - mean)));
            double lower = mean - stdDevThreshold * stdDev;
            double upper = mean + stdDevThreshold * stdDev;

            result.AddRange(group.Where(r => r.Value >= lower && r.Value <= upper));
        }

        return result;
    }
}
```

### 12.2 The Optimized Implementation

```csharp
public class DataProcessor_V2
{
    // Avoid LINQ allocations; use dictionary directly
    public Dictionary<string, double> ComputeAverages(List<SensorReading> readings)
    {
        var sums = new Dictionary<string, (double Sum, int Count)>();

        foreach (var reading in readings)
        {
            if (sums.TryGetValue(reading.SensorId, out var current))
            {
                sums[reading.SensorId] = (current.Sum + reading.Value, current.Count + 1);
            }
            else
            {
                sums[reading.SensorId] = (reading.Value, 1);
            }
        }

        var result = new Dictionary<string, double>(sums.Count);
        foreach (var (key, (sum, count)) in sums)
        {
            result[key] = sum / count;
        }
        return result;
    }

    // Single pass for statistics, pre-sized output list
    public List<SensorReading> FilterOutliers(
        List<SensorReading> readings, double stdDevThreshold)
    {
        // Pass 1: Compute statistics per sensor
        var stats = new Dictionary<string, (double Sum, double SumSq, int Count)>();

        foreach (var r in readings)
        {
            if (stats.TryGetValue(r.SensorId, out var s))
                stats[r.SensorId] = (s.Sum + r.Value, s.SumSq + r.Value * r.Value, s.Count + 1);
            else
                stats[r.SensorId] = (r.Value, r.Value * r.Value, 1);
        }

        var bounds = new Dictionary<string, (double Lower, double Upper)>(stats.Count);
        foreach (var (key, (sum, sumSq, count)) in stats)
        {
            double mean = sum / count;
            double variance = sumSq / count - mean * mean;
            double stdDev = Math.Sqrt(Math.Max(0, variance));
            bounds[key] = (mean - stdDevThreshold * stdDev, mean + stdDevThreshold * stdDev);
        }

        // Pass 2: Filter with pre-computed bounds
        var result = new List<SensorReading>(readings.Count);  // Pre-sized
        foreach (var r in readings)
        {
            var (lower, upper) = bounds[r.SensorId];
            if (r.Value >= lower && r.Value <= upper)
                result.Add(r);
        }

        return result;
    }
}
```

### 12.3 The Benchmark

```csharp
[MemoryDiagnoser]
[RankColumn]
public class DataProcessorBenchmark
{
    private List<SensorReading> _readings = null!;

    [Params(1000, 10_000, 100_000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        var sensors = Enumerable.Range(0, 10).Select(i => $"sensor-{i:D3}").ToArray();
        _readings = Enumerable.Range(0, N)
            .Select(i => new SensorReading(
                sensors[i % sensors.Length],
                DateTime.UtcNow.AddSeconds(i),
                rng.NextDouble() * 100))
            .ToList();
    }

    [Benchmark(Baseline = true)]
    public Dictionary<string, double> V1_Averages() => new DataProcessor_V1().ComputeAverages(_readings);

    [Benchmark]
    public Dictionary<string, double> V2_Averages() => new DataProcessor_V2().ComputeAverages(_readings);

    [Benchmark]
    public List<SensorReading> V1_Filter() => new DataProcessor_V1().FilterOutliers(_readings, 2.0);

    [Benchmark]
    public List<SensorReading> V2_Filter() => new DataProcessor_V2().FilterOutliers(_readings, 2.0);
}

// Expected results (approximate):
// V2_Averages: ~40% faster, ~60% less memory
// V2_Filter:   ~50% faster, ~70% less memory (especially at 100K readings)
```

## 13. Practice Problems

1. **Benchmark String Operations**: Write a BenchmarkDotNet benchmark comparing four ways to build a comma-separated list of 1,000 integers: (a) string concatenation with `+`, (b) `StringBuilder`, (c) `string.Join`, and (d) `string.Create`. Show the benchmark class, run it, and explain which is fastest and why.

2. **Object Pool Implementation**: Implement a generic `ObjectPool<T>` that is thread-safe, has a configurable maximum size, and supports a reset action. Use it to pool `MemoryStream` objects. Write a benchmark comparing pooled vs non-pooled `MemoryStream` usage over 10,000 iterations.

3. **GC Pressure Analysis**: Write a program that creates allocation pressure by generating 1 million small objects (e.g., `new byte[64]`) in a loop. Use `GC.CollectionCount` before and after to measure how many Gen 0, Gen 1, and Gen 2 collections occurred. Then rewrite the loop using `ArrayPool<byte>` and compare collection counts.

4. **Value Type Optimization**: You have a particle simulation with 100,000 particles, each with `Position` (X, Y, Z), `Velocity` (X, Y, Z), and `Mass`. Implement it first with a `class Particle` and then with a `struct Particle`. Benchmark the `UpdatePositions` method that advances all particles by one time step. Explain the performance difference.

5. **Hot Path Analysis**: Given the following method that processes HTTP headers, identify three performance problems and rewrite the method to eliminate them. Benchmark both versions.

```csharp
public Dictionary<string, string> ParseHeaders(string rawHeaders)
{
    var result = new Dictionary<string, string>();
    var lines = rawHeaders.Split("\r\n");
    foreach (var line in lines)
    {
        if (line.Contains(":"))
        {
            var parts = line.Split(":");
            result[parts[0].Trim().ToLower()] = parts[1].Trim();
        }
    }
    return result;
}
```
