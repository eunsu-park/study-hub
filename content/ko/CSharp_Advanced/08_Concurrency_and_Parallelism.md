# 동시성과 병렬성

**이전**: [Async와 Await](./07_Async_Await.md) | **다음**: [Span과 메모리](./09_Spans_and_Memory.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 동시성(concurrency), 병렬성(parallelism), 비동기성(asynchrony)을 구분하기
2. `Thread` 클래스와 `ThreadPool`을 사용한 저수준 스레딩 수행하기
3. 데이터 병렬 워크로드에 작업 병렬 라이브러리(Task Parallel Library, TPL) 적용하기
4. `lock`, `Monitor`, `SemaphoreSlim`, `Mutex`로 공유 상태 보호하기
5. 스레드 안전한 데이터 접근을 위한 동시 컬렉션(concurrent collection) 사용하기
6. `Interlocked` 클래스로 원자적 작업 수행하기
7. `System.Threading.Channels`로 생산자-소비자 패턴 구현하기
8. 실제 애플리케이션에 스레드 안전성 모범 사례 적용하기

---

동시성(concurrency)은 구조에 관한 것입니다: 겹치는 시간 내에 진행할 수 있는 여러 작업을 관리하는 것입니다. 병렬성(parallelism)은 실행에 관한 것입니다: 여러 CPU 코어에서 여러 작업을 *동시에* 실행하는 것입니다. C#은 두 가지 모두를 위한 풍부한 프리미티브 세트를 제공합니다. 이 레슨은 원시 스레드부터 고수준 채널까지, 올바르고 성능이 좋은 멀티스레드 코드를 작성하는 방법을 다룹니다.

## 1. Thread 클래스 기초

`Thread` 클래스는 .NET에서 가장 낮은 수준의 동시성 프리미티브입니다. 현대 코드에서 원시 스레드를 직접 생성하는 경우는 드물지만, 이를 이해하는 것은 기본이 됩니다.

### 1.1 스레드 생성과 시작

```csharp
// 기본 스레드 생성
Thread worker = new Thread(() =>
{
    for (int i = 0; i < 5; i++)
    {
        Console.WriteLine($"워커: {i} (스레드 {Thread.CurrentThread.ManagedThreadId})");
        Thread.Sleep(100);
    }
});

worker.Name = "MyWorker";
worker.IsBackground = true; // 앱 종료를 방해하지 않음
worker.Start();

Console.WriteLine($"메인 스레드: {Thread.CurrentThread.ManagedThreadId}");
worker.Join(); // 워커가 완료될 때까지 대기
Console.WriteLine("워커 완료.");
```

### 1.2 스레드에 데이터 전달

```csharp
// 람다 클로저 사용
string message = "스레드에서 보내는 인사";
Thread t = new Thread(() => Console.WriteLine(message));
t.Start();

// ParameterizedThreadStart 사용
Thread t2 = new Thread(obj =>
{
    var (name, count) = ((string, int))obj!;
    for (int i = 0; i < count; i++)
        Console.WriteLine($"{name}: 반복 {i}");
});
t2.Start(("Worker", 5));
```

### 1.3 스레드 속성

```csharp
Thread current = Thread.CurrentThread;
Console.WriteLine($"스레드 ID: {current.ManagedThreadId}");
Console.WriteLine($"이름: {current.Name}");
Console.WriteLine($"백그라운드 여부: {current.IsBackground}");
Console.WriteLine($"스레드풀 스레드 여부: {current.IsThreadPoolThread}");
Console.WriteLine($"우선순위: {current.Priority}");
Console.WriteLine($"상태: {current.ThreadState}");
```

## 2. ThreadPool과 작업 항목

스레드를 생성하는 것은 비용이 큽니다. `ThreadPool`은 재사용 가능한 워커 스레드 풀을 유지하여 짧은 작업의 오버헤드를 크게 줄입니다.

### 2.1 작업 큐잉

```csharp
// 작업 항목 큐잉
ThreadPool.QueueUserWorkItem(state =>
{
    Console.WriteLine($"풀 스레드 {Thread.CurrentThread.ManagedThreadId}: {state}");
}, "작업 항목 데이터");

// 풀 정보 조회
ThreadPool.GetMinThreads(out int minWorker, out int minIO);
ThreadPool.GetMaxThreads(out int maxWorker, out int maxIO);
Console.WriteLine($"워커: {minWorker}-{maxWorker}, IO: {minIO}-{maxIO}");
```

### 2.2 ThreadPool vs 수동 스레드

```csharp
// 벤치마크: ThreadPool vs new Thread
var sw = System.Diagnostics.Stopwatch.StartNew();
var countdown = new CountdownEvent(1000);

for (int i = 0; i < 1000; i++)
{
    ThreadPool.QueueUserWorkItem(_ =>
    {
        // 최소한의 작업
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
Console.WriteLine($"새 스레드: {sw.ElapsedMilliseconds}ms");
// ThreadPool이 일반적으로 10-100배 빠름
```

## 3. 작업 병렬 라이브러리 (TPL)

TPL은 병렬 실행을 위한 고수준 구조를 제공합니다. `Parallel.For`와 `Parallel.ForEach`는 사용 가능한 코어에 자동으로 작업을 분배합니다.

### 3.1 Parallel.For

```csharp
double[] data = new double[10_000_000];
Random random = new();
for (int i = 0; i < data.Length; i++)
    data[i] = random.NextDouble();

// 순차적
var sw = System.Diagnostics.Stopwatch.StartNew();
double[] results = new double[data.Length];
for (int i = 0; i < data.Length; i++)
    results[i] = Math.Sqrt(data[i]);
Console.WriteLine($"순차적: {sw.ElapsedMilliseconds}ms");

// 병렬
sw.Restart();
Parallel.For(0, data.Length, i =>
{
    results[i] = Math.Sqrt(data[i]);
});
Console.WriteLine($"병렬:   {sw.ElapsedMilliseconds}ms");
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
    Console.WriteLine($"{Path.GetFileName(path)}: {lines}줄");
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
    Console.WriteLine($"다운로드 완료 {url}: {content.Length}자");
});
```

### 3.4 병렬 LINQ (PLINQ)

```csharp
int[] numbers = Enumerable.Range(1, 10_000_000).ToArray();

// 병렬 계산
long sum = numbers
    .AsParallel()
    .WithDegreeOfParallelism(4)
    .Where(n => n % 2 == 0)
    .Sum(n => (long)n);

Console.WriteLine($"짝수의 합: {sum}");
```

## 4. lock 문과 Monitor

### 4.1 lock 문

`lock` 문은 상호 배제를 제공합니다 — 한 번에 하나의 스레드만 잠금을 보유할 수 있습니다.

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
        // 데드락을 방지하기 위해 항상 일관된 순서로 잠금
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

### 4.2 Monitor (수동 잠금 제어)

`lock` 문은 `Monitor.Enter`/`Monitor.Exit`으로 컴파일됩니다. `Monitor`를 직접 사용하면 더 많은 제어가 가능합니다.

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

### 4.3 Monitor.Wait와 Monitor.Pulse

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
                Monitor.Wait(_lock); // 잠금 해제 후 대기

            _queue.Enqueue(item);
            Monitor.PulseAll(_lock); // 대기 중인 소비자에게 알림
        }
    }

    public T Dequeue()
    {
        lock (_lock)
        {
            while (_queue.Count == 0)
                Monitor.Wait(_lock); // 잠금 해제 후 대기

            T item = _queue.Dequeue();
            Monitor.PulseAll(_lock); // 대기 중인 생산자에게 알림
            return item;
        }
    }
}
```

## 5. SemaphoreSlim과 Mutex

### 5.1 SemaphoreSlim

세마포어(semaphore)는 리소스에 동시에 접근할 수 있는 스레드 수를 제한합니다. `lock`(정확히 하나만 허용)과 달리 세마포어는 N개를 허용합니다.

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
// 사용법: 동시 데이터베이스 연결을 5개로 제한
var pool = new ConnectionPool(maxConnections: 5);

var tasks = Enumerable.Range(0, 20).Select(async i =>
{
    string result = await pool.ExecuteAsync(async () =>
    {
        await Task.Delay(100); // DB 쿼리 시뮬레이션
        return $"쿼리 {i} 결과";
    });
    Console.WriteLine(result);
});

await Task.WhenAll(tasks);
```

### 5.2 Mutex

`Mutex`는 시스템 전체 동기화 프리미티브입니다. 프로세스 간(스레드뿐 아니라) 조정이 필요할 때 사용합니다.

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
            // 이전 인스턴스가 크래시됨 — 이제 우리가 뮤텍스를 소유
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

## 6. 동시 컬렉션

`System.Collections.Concurrent` 네임스페이스는 외부 잠금 없이 사용할 수 있는 스레드 안전한 컬렉션 타입을 제공합니다.

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

// 팩토리를 사용한 GetOrAdd
var cache = new ConcurrentDictionary<string, ExpensiveObject>();
ExpensiveObject obj = cache.GetOrAdd("key", key => new ExpensiveObject(key));
```

### 6.2 ConcurrentQueue와 ConcurrentStack

```csharp
var queue = new ConcurrentQueue<WorkItem>();

// 생산자 스레드
queue.Enqueue(new WorkItem("task1"));
queue.Enqueue(new WorkItem("task2"));

// 소비자 스레드
if (queue.TryDequeue(out WorkItem? item))
{
    Process(item);
}

Console.WriteLine($"큐 카운트: {queue.Count}");
```

### 6.3 ConcurrentBag

`ConcurrentBag<T>`는 같은 스레드가 항목을 생산하고 소비하는 시나리오(예: 결과를 수집하는 병렬 루프)에 최적화되어 있습니다.

```csharp
var results = new ConcurrentBag<ProcessingResult>();

Parallel.ForEach(inputData, item =>
{
    ProcessingResult result = Process(item);
    results.Add(result);
});

Console.WriteLine($"{results.Count}개 항목 처리됨");

// 추가 처리를 위해 리스트로 변환
List<ProcessingResult> sortedResults = results
    .OrderBy(r => r.Timestamp)
    .ToList();
```

### 6.4 BlockingCollection

```csharp
using var collection = new BlockingCollection<int>(boundedCapacity: 10);

// 생산자
Task producer = Task.Run(() =>
{
    for (int i = 0; i < 50; i++)
    {
        collection.Add(i);
        Console.WriteLine($"생산: {i}");
    }
    collection.CompleteAdding();
});

// 소비자
Task consumer = Task.Run(() =>
{
    foreach (int item in collection.GetConsumingEnumerable())
    {
        Console.WriteLine($"소비: {item}");
        Thread.Sleep(50); // 처리 시뮬레이션
    }
});

await Task.WhenAll(producer, consumer);
```

## 7. 원자적 작업을 위한 Interlocked 클래스

`Interlocked` 클래스는 스레드 간에 공유되는 변수에 대한 원자적 작업을 제공하여 간단한 카운터와 플래그에 대한 잠금을 피합니다.

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
// 잠금 없는 최댓값 추적기
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
                break; // 성공적으로 업데이트됨
            current = previous; // 새 현재 값으로 재시도
        }
    }
}
```

## 8. ReaderWriterLockSlim

읽기가 쓰기보다 훨씬 많은 경우, `ReaderWriterLockSlim`은 동시 읽기를 허용하면서 쓰기에 대한 배타적 접근을 보장합니다.

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
                // 쓰기 잠금 획득 후 이중 확인
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

채널(Channel)은 효율적인 바운드 생산자-소비자 데이터 구조를 제공합니다 — `BlockingCollection`의 현대적이고 비동기 친화적인 대체물이라고 생각하면 됩니다.

### 9.1 바운드 채널과 언바운드 채널

```csharp
using System.Threading.Channels;

// 바운드 채널: 가득 차면 생산자를 차단
Channel<string> bounded = Channel.CreateBounded<string>(new BoundedChannelOptions(100)
{
    FullMode = BoundedChannelFullMode.Wait, // 기본값: 생산자가 대기
    SingleReader = false,
    SingleWriter = false
});

// 언바운드 채널: 생산자를 절대 차단하지 않음 (메모리가 한계)
Channel<string> unbounded = Channel.CreateUnbounded<string>();
```

### 9.2 쓰기와 읽기

```csharp
Channel<int> channel = Channel.CreateBounded<int>(10);

// 생산자
async Task ProduceAsync(ChannelWriter<int> writer)
{
    for (int i = 0; i < 100; i++)
    {
        await writer.WriteAsync(i);
        Console.WriteLine($"썼음: {i}");
    }
    writer.Complete(); // 더 이상 항목이 쓰여지지 않음을 알림
}

// 소비자
async Task ConsumeAsync(ChannelReader<int> reader)
{
    await foreach (int item in reader.ReadAllAsync())
    {
        Console.WriteLine($"읽음: {item}");
        await Task.Delay(10); // 처리 시뮬레이션
    }
}

// 둘 다 실행
await Task.WhenAll(
    ProduceAsync(channel.Writer),
    ConsumeAsync(channel.Reader));
```

### 9.3 다중 소비자

```csharp
Channel<WorkItem> channel = Channel.CreateBounded<WorkItem>(1000);

// 여러 소비자 태스크 시작
int consumerCount = Environment.ProcessorCount;
var consumers = Enumerable.Range(0, consumerCount)
    .Select(id => Task.Run(async () =>
    {
        await foreach (WorkItem item in channel.Reader.ReadAllAsync())
        {
            Console.WriteLine($"소비자 {id} 처리 중: {item.Name}");
            await ProcessAsync(item);
        }
    }))
    .ToArray();

// 생산자가 항목을 씀
foreach (var item in GetWorkItems())
    await channel.Writer.WriteAsync(item);

channel.Writer.Complete();
await Task.WhenAll(consumers);
```

## 10. 생산자-소비자 패턴

생산자-소비자 패턴은 데이터 생성과 데이터 처리를 분리하여 각각이 자체 속도로 실행될 수 있게 합니다.

### 10.1 채널을 이용한 파이프라인

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
// 사용법: 파일 처리 파이프라인
// 단계 1: 파일 경로 읽기
Channel<string> pathChannel = Channel.CreateBounded<string>(50);
_ = Task.Run(async () =>
{
    foreach (string path in Directory.GetFiles("/data", "*.log"))
        await pathChannel.Writer.WriteAsync(path);
    pathChannel.Writer.Complete();
});

// 단계 2: 파일 내용 읽기
ChannelReader<string> contents = Pipeline.Transform(
    pathChannel.Reader,
    path => File.ReadAllText(path));

// 단계 3: 줄 추출
ChannelReader<string> lines = Pipeline.TransformMany(
    contents,
    content => content.Split('\n'));

// 단계 4: 각 줄 처리
await foreach (string line in lines.ReadAllAsync())
{
    if (line.Contains("ERROR"))
        Console.WriteLine($"에러 발견: {line}");
}
```

## 11. 스레드 안전성 모범 사례

### 11.1 불변성

불변 객체는 상태가 절대 변경되지 않으므로 본질적으로 스레드 안전합니다.

```csharp
// 불변 구성 — 동기화 불필요
public record AppConfig(
    string ConnectionString,
    int MaxRetries,
    TimeSpan Timeout);

// 불변 교체를 통한 스레드 안전한 상태 업데이트
public class ConfigManager
{
    private volatile AppConfig _config;

    public ConfigManager(AppConfig initial) => _config = initial;

    public AppConfig Current => _config;

    public void Update(Func<AppConfig, AppConfig> updater)
    {
        // Interlocked를 사용한 원자적 교체
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

### 11.2 공유 가변 상태 피하기

```csharp
// 나쁜 예: 공유 가변 리스트
List<int> results = new();
Parallel.For(0, 1000, i =>
{
    results.Add(Compute(i)); // 경합 조건!
});

// 좋은 예: 스레드 로컬 축적 후 병합
Parallel.For(0, 1000,
    () => new List<int>(), // 스레드 로컬 상태
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

### 11.3 volatile 키워드

```csharp
public class StoppableWorker
{
    private volatile bool _shouldStop;

    public void RequestStop() => _shouldStop = true;

    public void Run()
    {
        while (!_shouldStop)
        {
            // 작업 수행...
            // volatile 없이는 JIT가 _shouldStop을 레지스터에 캐시하여
            // 업데이트된 값을 절대 볼 수 없을 수 있음
        }
    }
}
```

## 12. 실전 예제: 동시 웹 스크레이퍼

이 예제는 채널, 스로틀된 병렬성, 동시 컬렉션, 취소를 결합합니다.

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

        // 채널에 시드 URL 추가
        foreach (string url in seedUrls)
        {
            _visited.TryAdd(url, 0);
            await urlChannel.Writer.WriteAsync((url, 0), cancellationToken);
        }

        var activeTasks = new ConcurrentBag<Task>();
        int pendingCount = _visited.Count;

        // 워커 루프
        var workers = Enumerable.Range(0, _maxConcurrency).Select(_ => Task.Run(async () =>
        {
            await foreach (var (url, depth) in urlChannel.Reader.ReadAllAsync(cancellationToken))
            {
                await throttle.WaitAsync(cancellationToken);
                try
                {
                    ScrapeResult result = await ScrapeOneAsync(url, cancellationToken);
                    _results[url] = result;

                    // 새 URL 발견
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
// 사용법
var scraper = new ConcurrentWebScraper(maxConcurrency: 5);
using var cts = new CancellationTokenSource(TimeSpan.FromMinutes(2));

var results = await scraper.ScrapeAsync(
    new[] { "https://example.com" },
    maxDepth: 1,
    cts.Token);

foreach (var (url, result) in results.OrderBy(r => r.Key))
{
    string status = result.Success ? "OK" : "실패";
    Console.WriteLine($"[{status}] {url} — {result.Title ?? "제목 없음"} ({result.ContentLength} 바이트)");
}
```

## 13. 연습 문제

1. **스레드 안전 카운터**: 세 가지 접근 방식으로 `SafeCounter` 클래스를 구현하세요: (a) `lock`, (b) `Interlocked`, (c) `ReaderWriterLockSlim`. `Increment()`, `Decrement()`, `Value`를 포함합니다. 각 접근 방식에 대해 8개 스레드에서 1,000,000번의 증가를 실행하는 벤치마크를 작성하고 경과 시간을 비교하세요.

2. **병렬 이미지 처리기**: `Parallel.For`를 사용하여 2D 픽셀 값 배열(그레이스케일 이미지를 `byte[width, height]`로 시뮬레이션)을 처리하세요. 3x3 박스 블러 필터를 적용합니다. 경계 픽셀이 올바르게 처리되도록 하세요. 4096x4096 이미지에 대해 순차적 vs 병렬 성능을 비교하세요.

3. **채널을 이용한 생산자-소비자**: `System.Threading.Channels`를 사용하여 세 단계 파이프라인을 구축하세요: 단계 1은 디렉토리에서 파일 경로를 읽고; 단계 2는 각 파일을 읽어 단어 빈도를 세고; 단계 3은 모든 단어 수를 글로벌 `ConcurrentDictionary<string, int>`에 병합하고 상위 100개 단어를 CSV 파일에 쓰세요. 적절한 용량의 바운드 채널을 사용하세요.

4. **식사하는 철학자들**: 5명의 철학자와 5개의 포크로 고전적인 식사하는 철학자 문제를 구현하세요. 포크에는 `SemaphoreSlim`을 사용합니다. 리소스 순서 지정을 사용하여 데드락 없는 솔루션을 구현하세요. 타임스탬프와 함께 행동(생각 중, 배고픔, 식사 중)을 출력하고 30초 동안 데드락이 발생하지 않는지 확인하세요.

5. **스레드 안전 관찰 가능 컬렉션**: `Add`, `Remove`, `Contains`, `GetSnapshot()`(읽기 전용 복사본 반환)을 지원하는 `ConcurrentObservableList<T>`를 만드세요. 동기화에는 `ReaderWriterLockSlim`을 사용합니다. 변이 후 `CollectionChanged` 이벤트를 발생시키세요. 4개 스레드가 항목을 추가하고 2개 스레드가 동시에 스냅샷을 열거하는 테스트를 작성하세요.
