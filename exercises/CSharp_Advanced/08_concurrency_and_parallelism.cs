/*
 * Exercises for Lesson 08: Concurrency and Parallelism
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

// ---------------------------------------------------------------------------
// Exercise 1: Thread-safe counter with Interlocked
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Thread-Safe Counter ===");

    int unsafeCount = 0;
    int safeCount = 0;
    const int iterations = 100_000;

    Parallel.For(0, iterations, _ =>
    {
        unsafeCount++;                       // NOT thread-safe
        Interlocked.Increment(ref safeCount); // Thread-safe
    });

    Console.WriteLine($"  Unsafe counter: {unsafeCount} (expected {iterations})");
    Console.WriteLine($"  Safe counter  : {safeCount} (expected {iterations})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Lock-based thread safety — bank account
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Lock-Based Bank Account ===");

    var account = new BankAccount(1000m);

    var deposits = Task.Run(() =>
    {
        for (int i = 0; i < 100; i++) account.Deposit(10m);
    });

    var withdrawals = Task.Run(() =>
    {
        for (int i = 0; i < 100; i++) account.Withdraw(5m);
    });

    Task.WaitAll(deposits, withdrawals);
    Console.WriteLine($"  Final balance: {account.Balance:C} (expected: $1,500.00)");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Producer-Consumer with BlockingCollection
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Producer-Consumer ===");

    using var queue = new BlockingCollection<WorkItem>(boundedCapacity: 5);
    int processedCount = 0;

    var producer = Task.Run(() =>
    {
        for (int i = 1; i <= 10; i++)
        {
            queue.Add(new WorkItem(i, $"Task-{i}"));
            Console.WriteLine($"  Produced: Task-{i}");
        }
        queue.CompleteAdding();
    });

    var consumer = Task.Run(() =>
    {
        foreach (var item in queue.GetConsumingEnumerable())
        {
            Thread.Sleep(50); // simulate work
            Interlocked.Increment(ref processedCount);
            Console.WriteLine($"  Consumed: {item.Name} (id={item.Id})");
        }
    });

    Task.WaitAll(producer, consumer);
    Console.WriteLine($"  Total processed: {processedCount}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: ConcurrentDictionary — word frequency counter
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: ConcurrentDictionary — Word Frequency ===");

    string[] sentences = {
        "the quick brown fox jumps over the lazy dog",
        "the fox ran across the road and the dog followed",
        "quick brown fox and lazy dog are friends",
    };

    var wordCounts = new ConcurrentDictionary<string, int>();

    Parallel.ForEach(sentences, sentence =>
    {
        foreach (var word in sentence.Split(' '))
            wordCounts.AddOrUpdate(word, 1, (_, count) => count + 1);
    });

    var topWords = wordCounts.OrderByDescending(kv => kv.Value).Take(5);
    foreach (var (word, count) in topWords)
        Console.WriteLine($"  '{word}': {count}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: SemaphoreSlim — rate limiting
// ---------------------------------------------------------------------------
async Task Exercise5()
{
    Console.WriteLine("=== Exercise 5: SemaphoreSlim Rate Limiter ===");

    using var semaphore = new SemaphoreSlim(3, 3); // max 3 concurrent
    var sw = Stopwatch.StartNew();

    var tasks = Enumerable.Range(1, 8).Select(async i =>
    {
        await semaphore.WaitAsync();
        try
        {
            Console.WriteLine($"  [{sw.ElapsedMilliseconds,4}ms] Request {i} started");
            await Task.Delay(200);
            Console.WriteLine($"  [{sw.ElapsedMilliseconds,4}ms] Request {i} done");
        }
        finally
        {
            semaphore.Release();
        }
    });

    await Task.WhenAll(tasks);
    Console.WriteLine($"  All done in {sw.ElapsedMilliseconds}ms");
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
await Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

record WorkItem(int Id, string Name);

class BankAccount
{
    private readonly object _lock = new();
    public decimal Balance { get; private set; }

    public BankAccount(decimal initial) => Balance = initial;

    public void Deposit(decimal amount)
    {
        lock (_lock) { Balance += amount; }
    }

    public bool Withdraw(decimal amount)
    {
        lock (_lock)
        {
            if (Balance >= amount) { Balance -= amount; return true; }
            return false;
        }
    }
}
