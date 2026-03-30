# Async and Await

**Previous**: [Events and Delegates](./06_Events_and_Delegates.md) | **Next**: [Concurrency and Parallelism](./08_Concurrency_and_Parallelism.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why asynchronous programming matters for I/O-bound and CPU-bound work
2. Use `Task`, `Task<T>`, and `ValueTask<T>` to represent asynchronous operations
3. Write correct async methods using the `async` and `await` keywords
4. Handle exceptions and cancellation in asynchronous code
5. Compose multiple asynchronous operations with `Task.WhenAll` and `Task.WhenAny`
6. Understand synchronization contexts and when to use `ConfigureAwait(false)`
7. Consume asynchronous streams with `IAsyncEnumerable<T>`
8. Avoid common async pitfalls such as deadlocks and fire-and-forget mistakes

---

Modern applications spend most of their time waiting: waiting for database queries, HTTP responses, file reads, and user input. Traditional synchronous code blocks a thread while waiting, wasting resources that could serve other requests. C#'s `async`/`await` pattern lets you write code that *looks* sequential but releases its thread during waits, dramatically improving throughput and responsiveness. This lesson takes you from the fundamentals of `Task` through advanced topics like `ValueTask`, channels, and async streams.

## 1. Why Async? I/O-Bound vs CPU-Bound Work

Not every workload benefits equally from async. Understanding the distinction between I/O-bound and CPU-bound operations is the first step toward writing efficient asynchronous code.

### 1.1 I/O-Bound Work

I/O-bound work spends time waiting for an external resource: a network socket, a disk controller, or a database engine. The thread doing the waiting is idle; it could be returned to the thread pool to handle other requests.

```csharp
// Synchronous — thread is blocked while waiting for the HTTP response
public string GetPageSync(string url)
{
    using var client = new HttpClient();
    // This call blocks the calling thread until the response arrives
    HttpResponseMessage response = client.Send(new HttpRequestMessage(HttpMethod.Get, url));
    return response.Content.ReadAsStringAsync().Result; // Another block
}
```

```csharp
// Asynchronous — thread is released while waiting
public async Task<string> GetPageAsync(string url)
{
    using var client = new HttpClient();
    HttpResponseMessage response = await client.GetAsync(url);
    return await response.Content.ReadAsStringAsync();
}
```

In the async version, the thread is returned to the pool during both `await` points. On a web server handling thousands of concurrent requests, this difference is enormous.

### 1.2 CPU-Bound Work

CPU-bound work keeps the processor busy with computation (image processing, encryption, simulation). Async does not make CPU work faster; the work still needs a thread. However, you can offload CPU work to a background thread with `Task.Run` to keep the UI responsive.

```csharp
// CPU-bound: offload to a thread pool thread
public async Task<double> ComputePiAsync(int digits)
{
    return await Task.Run(() => ComputePi(digits));
}
```

**Rule of thumb**: Use `async`/`await` *directly* for I/O-bound work. Use `Task.Run` to push CPU-bound work off the calling thread only when you need to keep a UI thread free.

## 2. Task and Task&lt;T&gt;

`Task` represents an asynchronous operation that may or may not have completed. `Task<T>` adds a result value.

### 2.1 Creating and Observing Tasks

```csharp
// A completed task with a value
Task<int> completed = Task.FromResult(42);

// A completed task with no value
Task done = Task.CompletedTask;

// A task that represents a faulted operation
Task faulted = Task.FromException(new InvalidOperationException("oops"));

// A task that represents cancellation
Task cancelled = Task.FromCanceled(new CancellationToken(true));
```

### 2.2 Task Status

```csharp
Task<string> task = SomeAsyncOperation();

Console.WriteLine(task.Status);       // WaitingForActivation, Running, RanToCompletion, Faulted, Canceled
Console.WriteLine(task.IsCompleted);  // true when RanToCompletion, Faulted, or Canceled
Console.WriteLine(task.IsCompletedSuccessfully); // true only when RanToCompletion
```

### 2.3 Continuations (Low-Level)

Before `await`, continuations were set up manually. You rarely need this today, but understanding it helps you reason about what `await` does under the hood.

```csharp
Task<byte[]> downloadTask = client.GetByteArrayAsync(url);

downloadTask.ContinueWith(t =>
{
    if (t.IsFaulted)
    {
        Console.WriteLine($"Error: {t.Exception?.InnerException?.Message}");
        return;
    }
    File.WriteAllBytes("output.dat", t.Result);
}, TaskContinuationOptions.ExecuteSynchronously);
```

## 3. async/await Keywords and How They Work

### 3.1 The Basics

The `async` modifier tells the compiler to transform the method body into a state machine. The `await` keyword marks suspension points where the method yields control until the awaited task completes.

```csharp
public async Task<string> ReadFileAsync(string path)
{
    // Execution reaches here synchronously
    Console.WriteLine("Before await");

    // At this point, if the file read hasn't completed, the method returns
    // a Task<string> to the caller and the thread is released.
    string content = await File.ReadAllTextAsync(path);

    // Execution resumes here (possibly on a different thread)
    Console.WriteLine("After await");
    return content;
}
```

### 3.2 The State Machine Under the Hood

The compiler rewrites every async method into a struct that implements `IAsyncStateMachine`. Each `await` becomes a state transition. When the awaited task completes, the runtime calls `MoveNext()` to advance to the next state.

```csharp
// Conceptual compiler transformation (simplified)
private struct ReadFileAsyncStateMachine : IAsyncStateMachine
{
    public int State;
    public AsyncTaskMethodBuilder<string> Builder;
    public string Path;
    private TaskAwaiter<string> _awaiter;

    public void MoveNext()
    {
        switch (State)
        {
            case 0:
                Console.WriteLine("Before await");
                _awaiter = File.ReadAllTextAsync(Path).GetAwaiter();
                if (!_awaiter.IsCompleted)
                {
                    State = 1;
                    Builder.AwaitUnsafeOnCompleted(ref _awaiter, ref this);
                    return; // Yield
                }
                goto case 1;
            case 1:
                string content = _awaiter.GetResult();
                Console.WriteLine("After await");
                Builder.SetResult(content);
                break;
        }
    }
}
```

Understanding this transformation explains why `async` methods have slightly more overhead than synchronous equivalents, and why `ValueTask` can help in hot paths.

## 4. Async Method Signatures and Return Types

### 4.1 The Three Standard Return Types

```csharp
// Returns a value
public async Task<int> GetCountAsync() { ... }

// Returns no value
public async Task SaveAsync() { ... }

// Returns nothing, cannot be awaited — avoid except for event handlers
public async void OnButtonClick(object sender, EventArgs e) { ... }
```

### 4.2 ValueTask and ValueTask&lt;T&gt;

`ValueTask<T>` is a struct that avoids heap allocation when the result is already available synchronously. Ideal for methods that often complete synchronously (cache hits, buffered reads).

```csharp
private readonly ConcurrentDictionary<string, User> _cache = new();

public ValueTask<User> GetUserAsync(string id)
{
    if (_cache.TryGetValue(id, out User? cached))
    {
        // No Task allocation needed
        return new ValueTask<User>(cached);
    }

    return new ValueTask<User>(LoadUserFromDatabaseAsync(id));
}

private async Task<User> LoadUserFromDatabaseAsync(string id)
{
    // ... database call
    User user = await _db.Users.FindAsync(id);
    _cache[id] = user;
    return user;
}
```

### 4.3 IAsyncEnumerable&lt;T&gt; Return Type

```csharp
public async IAsyncEnumerable<int> GenerateNumbersAsync(int count)
{
    for (int i = 0; i < count; i++)
    {
        await Task.Delay(100);
        yield return i;
    }
}
```

## 5. Exception Handling in Async Code

### 5.1 Exceptions in Awaited Tasks

When you `await` a faulted task, the exception is unwrapped from the `AggregateException` and rethrown at the `await` point. This gives you the natural try/catch experience.

```csharp
public async Task ProcessDataAsync()
{
    try
    {
        string data = await FetchDataAsync("https://api.example.com/data");
        await SaveDataAsync(data);
    }
    catch (HttpRequestException ex)
    {
        Console.WriteLine($"Network error: {ex.Message}");
    }
    catch (IOException ex)
    {
        Console.WriteLine($"Save error: {ex.Message}");
    }
}
```

### 5.2 Multiple Exceptions with Task.WhenAll

When multiple tasks fail, `Task.WhenAll` stores all exceptions in the returned task's `Exception` property, but `await` only rethrows the first one.

```csharp
public async Task ProcessAllAsync()
{
    Task t1 = FailingTask("A");
    Task t2 = FailingTask("B");

    Task allTasks = Task.WhenAll(t1, t2);

    try
    {
        await allTasks;
    }
    catch (Exception ex)
    {
        // ex is only the first exception
        Console.WriteLine($"First: {ex.Message}");

        // Access all exceptions through the task
        if (allTasks.Exception is not null)
        {
            foreach (var inner in allTasks.Exception.InnerExceptions)
            {
                Console.WriteLine($"  - {inner.Message}");
            }
        }
    }
}
```

### 5.3 Exceptions in async void Methods

Exceptions from `async void` methods cannot be caught by the caller. They propagate to the synchronization context and often crash the process.

```csharp
// DANGEROUS — exception will crash the application
public async void FireAndForgetBad()
{
    await Task.Delay(100);
    throw new InvalidOperationException("Unobserved!");
}

// SAFE — return Task so callers can observe exceptions
public async Task FireAndForgetSafe()
{
    await Task.Delay(100);
    throw new InvalidOperationException("Can be caught!");
}
```

## 6. Cancellation with CancellationToken

Cancellation is cooperative in .NET. You pass a `CancellationToken` to async methods, and the method periodically checks whether cancellation has been requested.

### 6.1 Basic Usage

```csharp
public async Task<string> DownloadWithTimeoutAsync(string url, int timeoutMs)
{
    using var cts = new CancellationTokenSource(timeoutMs);

    try
    {
        using var client = new HttpClient();
        return await client.GetStringAsync(url, cts.Token);
    }
    catch (OperationCanceledException)
    {
        Console.WriteLine("Download timed out or was cancelled.");
        return string.Empty;
    }
}
```

### 6.2 Linking Tokens

You can link multiple cancellation sources so that cancelling any one of them cancels the linked token.

```csharp
public async Task ProcessAsync(CancellationToken externalToken)
{
    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        externalToken, timeoutCts.Token);

    await DoWorkAsync(linkedCts.Token);
}
```

### 6.3 Manual Cancellation Checks

For CPU-bound loops, check the token periodically.

```csharp
public async Task<long> CountPrimesAsync(long max, CancellationToken token)
{
    return await Task.Run(() =>
    {
        long count = 0;
        for (long n = 2; n <= max; n++)
        {
            token.ThrowIfCancellationRequested();
            if (IsPrime(n)) count++;
        }
        return count;
    }, token);
}
```

## 7. Task.WhenAll and Task.WhenAny

### 7.1 Task.WhenAll — Parallel Composition

`Task.WhenAll` completes when all tasks complete. It is the primary way to run independent async operations concurrently.

```csharp
public async Task<DashboardData> LoadDashboardAsync()
{
    Task<User> userTask = GetUserAsync();
    Task<List<Order>> ordersTask = GetOrdersAsync();
    Task<List<Notification>> notificationsTask = GetNotificationsAsync();

    // All three requests run concurrently
    await Task.WhenAll(userTask, ordersTask, notificationsTask);

    return new DashboardData
    {
        User = userTask.Result,
        Orders = ordersTask.Result,
        Notifications = notificationsTask.Result
    };
}
```

### 7.2 Task.WhenAny — First Completion

`Task.WhenAny` completes when the first task finishes. Useful for timeouts, redundant requests, or progress reporting.

```csharp
public async Task<string> FetchWithFallbackAsync(string primaryUrl, string fallbackUrl)
{
    using var client = new HttpClient();

    Task<string> primary = client.GetStringAsync(primaryUrl);
    Task<string> fallback = client.GetStringAsync(fallbackUrl);

    Task<string> winner = await Task.WhenAny(primary, fallback);
    return await winner; // Unwrap the result (and propagate any exception)
}
```

### 7.3 Processing Tasks as They Complete

```csharp
public async Task ProcessUrlsAsync(IEnumerable<string> urls)
{
    using var client = new HttpClient();
    var tasks = urls.Select(url => client.GetStringAsync(url)).ToList();

    while (tasks.Count > 0)
    {
        Task<string> finished = await Task.WhenAny(tasks);
        tasks.Remove(finished);

        try
        {
            string content = await finished;
            Console.WriteLine($"Received {content.Length} chars");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Failed: {ex.Message}");
        }
    }
}
```

## 8. ConfigureAwait and Synchronization Contexts

### 8.1 What Is a Synchronization Context?

A synchronization context captures the current execution environment (e.g., UI thread, ASP.NET request context) so that continuations after `await` run in the same context.

```csharp
// On a WPF UI thread:
public async Task UpdateUIAsync()
{
    string data = await FetchDataAsync(); // Yields the UI thread
    // After await, we're back on the UI thread — safe to update controls
    MyLabel.Content = data;
}
```

### 8.2 ConfigureAwait(false)

In library code that does not need to return to the original context, use `ConfigureAwait(false)` to avoid unnecessary context switching and potential deadlocks.

```csharp
public async Task<byte[]> ReadBytesAsync(string path)
{
    // Library code: no UI to return to
    byte[] buffer = await File.ReadAllBytesAsync(path).ConfigureAwait(false);
    return Compress(buffer);
}
```

### 8.3 The Classic Deadlock

```csharp
// ASP.NET Framework (pre-.NET Core) or WPF with SynchronizationContext
public string GetDataDeadlock()
{
    // .Result blocks the UI/request thread
    // The continuation from GetDataAsync tries to resume on that same thread
    // DEADLOCK
    return GetDataAsync().Result;
}

public async Task<string> GetDataAsync()
{
    // Without ConfigureAwait(false), the continuation needs the original context
    return await httpClient.GetStringAsync("https://example.com");
}
```

**Fix**: Use `ConfigureAwait(false)` in the library method, or better yet, use `await` all the way up instead of `.Result`.

## 9. ValueTask for High-Performance Scenarios

### 9.1 When to Use ValueTask

`ValueTask<T>` avoids the heap allocation of a `Task<T>` object when the operation completes synchronously. This matters in hot paths called thousands of times per second.

```csharp
public class BufferedStream
{
    private readonly byte[] _buffer = new byte[4096];
    private int _position;
    private int _length;
    private readonly Stream _inner;

    public ValueTask<int> ReadByteAsync()
    {
        if (_position < _length)
        {
            // Hot path: return from buffer without allocation
            return new ValueTask<int>(_buffer[_position++]);
        }

        // Cold path: actual I/O needed
        return new ValueTask<int>(ReadByteSlowAsync());
    }

    private async Task<int> ReadByteSlowAsync()
    {
        _length = await _inner.ReadAsync(_buffer);
        _position = 0;
        return _length > 0 ? _buffer[_position++] : -1;
    }
}
```

### 9.2 ValueTask Rules

```csharp
// DO: await it once
int result = await GetValueAsync();

// DON'T: await it multiple times
ValueTask<int> vt = GetValueAsync();
int r1 = await vt;
// int r2 = await vt; // UNDEFINED BEHAVIOR

// DON'T: use .Result or .GetAwaiter().GetResult() unless IsCompleted is true
// DON'T: combine with Task.WhenAll (convert to Task first)
Task<int> task = GetValueAsync().AsTask(); // Convert when needed
```

## 10. Common Async Pitfalls

### 10.1 async void

`async void` methods cannot be awaited, their exceptions are unobservable, and they make unit testing difficult. Use `async Task` instead.

```csharp
// BAD: async void
public async void Initialize()
{
    await LoadConfigAsync(); // If this throws, the process may crash
}

// GOOD: async Task
public async Task InitializeAsync()
{
    await LoadConfigAsync();
}
```

### 10.2 Fire-and-Forget Safely

If you truly need fire-and-forget, at least observe exceptions.

```csharp
public static class TaskExtensions
{
    public static async void SafeFireAndForget(
        this Task task,
        Action<Exception>? onError = null)
    {
        try
        {
            await task.ConfigureAwait(false);
        }
        catch (Exception ex)
        {
            onError?.Invoke(ex);
        }
    }
}

// Usage
SendEmailAsync(user.Email).SafeFireAndForget(ex =>
    _logger.LogError(ex, "Failed to send email"));
```

### 10.3 Blocking on Async Code

Never use `.Result`, `.Wait()`, or `.GetAwaiter().GetResult()` in code that has a synchronization context. This is the number one cause of async deadlocks.

```csharp
// BAD — potential deadlock
public void HandleRequest()
{
    var data = GetDataAsync().Result;
}

// GOOD — async all the way
public async Task HandleRequestAsync()
{
    var data = await GetDataAsync();
}
```

### 10.4 Unnecessary async/await

If you are simply passing through a task without doing anything after `await`, you can return the task directly to avoid state machine overhead.

```csharp
// Unnecessary state machine
public async Task<User> GetUserAsync(int id)
{
    return await _repository.FindAsync(id);
}

// More efficient — no state machine created
public Task<User> GetUserAsync(int id)
{
    return _repository.FindAsync(id);
}
```

**Caveat**: Eliding async/await changes exception behavior. Exceptions thrown before the `await` will occur at the call site rather than being wrapped in the returned task. Use the elision only when you understand this trade-off.

## 11. IAsyncEnumerable&lt;T&gt; for Async Streams

`IAsyncEnumerable<T>` (introduced in C# 8) lets you produce and consume sequences where each element may require an asynchronous operation to produce.

### 11.1 Producing an Async Stream

```csharp
public async IAsyncEnumerable<LogEntry> StreamLogsAsync(
    string path,
    [EnumeratorCancellation] CancellationToken cancellationToken = default)
{
    using var reader = new StreamReader(path);

    while (!reader.EndOfStream)
    {
        cancellationToken.ThrowIfCancellationRequested();
        string? line = await reader.ReadLineAsync(cancellationToken);

        if (line is not null)
        {
            yield return ParseLogEntry(line);
        }
    }
}
```

### 11.2 Consuming an Async Stream

```csharp
await foreach (LogEntry entry in StreamLogsAsync("app.log"))
{
    if (entry.Level == LogLevel.Error)
    {
        Console.WriteLine($"[ERROR] {entry.Timestamp}: {entry.Message}");
    }
}
```

### 11.3 LINQ-Style Operations with System.Linq.Async

```csharp
// Install: dotnet add package System.Linq.Async
var errors = StreamLogsAsync("app.log")
    .Where(e => e.Level == LogLevel.Error)
    .Take(10);

await foreach (var error in errors)
{
    Console.WriteLine(error.Message);
}
```

## 12. Practical Example: Parallel HTTP Client

This example demonstrates many of the concepts covered in this lesson: `Task.WhenAll`, `SemaphoreSlim` for throttling, `CancellationToken`, exception handling, and `IAsyncEnumerable`.

```csharp
using System.Net.Http;
using System.Runtime.CompilerServices;

public class ParallelHttpClient : IDisposable
{
    private readonly HttpClient _client;
    private readonly SemaphoreSlim _throttle;
    private readonly int _maxConcurrency;

    public ParallelHttpClient(int maxConcurrency = 10)
    {
        _maxConcurrency = maxConcurrency;
        _client = new HttpClient { Timeout = TimeSpan.FromSeconds(30) };
        _throttle = new SemaphoreSlim(maxConcurrency, maxConcurrency);
    }

    /// <summary>
    /// Downloads all URLs concurrently (up to maxConcurrency), returning results
    /// as they complete.
    /// </summary>
    public async IAsyncEnumerable<DownloadResult> DownloadAllAsync(
        IEnumerable<string> urls,
        [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        var tasks = urls.Select(url => DownloadOneAsync(url, cancellationToken)).ToList();

        while (tasks.Count > 0)
        {
            Task<DownloadResult> completed = await Task.WhenAny(tasks);
            tasks.Remove(completed);
            yield return await completed;
        }
    }

    /// <summary>
    /// Downloads all URLs and returns when all are complete.
    /// </summary>
    public async Task<DownloadResult[]> DownloadAllBatchAsync(
        IEnumerable<string> urls,
        CancellationToken cancellationToken = default)
    {
        Task<DownloadResult>[] tasks = urls
            .Select(url => DownloadOneAsync(url, cancellationToken))
            .ToArray();

        return await Task.WhenAll(tasks);
    }

    private async Task<DownloadResult> DownloadOneAsync(
        string url, CancellationToken cancellationToken)
    {
        await _throttle.WaitAsync(cancellationToken).ConfigureAwait(false);
        var stopwatch = System.Diagnostics.Stopwatch.StartNew();

        try
        {
            string content = await _client.GetStringAsync(url, cancellationToken)
                .ConfigureAwait(false);

            stopwatch.Stop();
            return new DownloadResult
            {
                Url = url,
                Content = content,
                Success = true,
                ElapsedMs = stopwatch.ElapsedMilliseconds
            };
        }
        catch (Exception ex) when (ex is HttpRequestException or TaskCanceledException)
        {
            stopwatch.Stop();
            return new DownloadResult
            {
                Url = url,
                Error = ex.Message,
                Success = false,
                ElapsedMs = stopwatch.ElapsedMilliseconds
            };
        }
        finally
        {
            _throttle.Release();
        }
    }

    public void Dispose()
    {
        _client.Dispose();
        _throttle.Dispose();
    }
}

public record DownloadResult
{
    public string Url { get; init; } = "";
    public string? Content { get; init; }
    public string? Error { get; init; }
    public bool Success { get; init; }
    public long ElapsedMs { get; init; }
}
```

```csharp
// Usage
using var client = new ParallelHttpClient(maxConcurrency: 5);

string[] urls =
{
    "https://api.example.com/users",
    "https://api.example.com/products",
    "https://api.example.com/orders",
    "https://api.example.com/reviews",
    "https://api.example.com/inventory"
};

using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(60));

await foreach (DownloadResult result in client.DownloadAllAsync(urls, cts.Token))
{
    if (result.Success)
        Console.WriteLine($"OK  {result.Url} ({result.ElapsedMs}ms, {result.Content?.Length} chars)");
    else
        Console.WriteLine($"ERR {result.Url}: {result.Error}");
}
```

## 13. Practice Problems

1. **Async File Processor**: Write an async method that reads all `.txt` files from a directory concurrently (limited to 3 at a time using `SemaphoreSlim`), counts the words in each, and returns a `Dictionary<string, int>` mapping file name to word count. Support cancellation via `CancellationToken`.

2. **Retry with Exponential Backoff**: Implement a generic `RetryAsync<T>` method that accepts a `Func<CancellationToken, Task<T>>`, a maximum retry count, and a base delay. On failure, it should wait for `baseDelay * 2^attempt` before retrying. Use `CancellationToken` to allow aborting the retry loop. Return the first successful result or throw an `AggregateException` with all observed exceptions.

3. **Async Rate Limiter**: Create a `RateLimiter` class that limits operations to N per time window (e.g., 10 requests per second). Provide an `async Task WaitForSlotAsync(CancellationToken token)` method that callers invoke before making a request. Use `SemaphoreSlim` and `Task.Delay` internally. Write a test that verifies the rate is respected.

4. **Streaming Pipeline**: Using `IAsyncEnumerable<T>`, create a three-stage pipeline: (a) a producer that reads lines from a large file asynchronously, (b) a transformer that parses each line into a record type, and (c) a consumer that batches records into groups of 100 and writes each batch to a separate output file. Each stage should be a separate async method yielding to the next.

5. **Timeout Decorator**: Write a method `WithTimeout<T>(Func<CancellationToken, Task<T>> operation, TimeSpan timeout)` that wraps any async operation with a timeout. If the timeout expires, the method should cancel the operation (using a linked `CancellationTokenSource`) and throw a `TimeoutException`. Verify that the underlying operation receives the cancellation signal.
