# 성능과 프로파일링

**이전**: [상호 운용과 안전하지 않은 코드](./15_Interop_and_Unsafe.md) | **다음**: [캡스톤: Minimal Web API](./17_Capstone_Web_API.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 마이크로 벤치마킹을 위해 BenchmarkDotNet을 설정하고 사용할 수 있다
2. 적절한 방법론으로 신뢰할 수 있는 벤치마크를 작성할 수 있다
3. 평균, 중앙값, 할당을 포함한 벤치마크 결과를 해석할 수 있다
4. `dotnet-counters`, `dotnet-trace`, `dotnet-dump`를 런타임 진단에 사용할 수 있다
5. GC 세대를 이해하고 할당 압력을 최소화할 수 있다
6. GC 오버헤드를 줄이기 위해 객체 풀링을 적용할 수 있다
7. 고성능 시나리오에서 문자열 연산을 최적화할 수 있다
8. 성능 요구 사항에 따라 값 타입과 참조 타입 중 선택할 수 있다
9. 애플리케이션에서 핫 경로(Hot Path)를 식별하고 최적화할 수 있다

---

.NET에서의 성능 최적화는 체계적인 프로세스입니다: 먼저 측정하고, 병목 지점을 식별하고, 대상 영역을 최적화하고, 벤치마크로 개선 사항을 검증합니다. 조기 최적화는 비생산적이지만, 런타임이 메모리를 어떻게 관리하는지, 가비지 컬렉터가 어떻게 작동하는지, 어떤 패턴이 불필요한 할당을 유발하는지 이해하는 것은 반응성 있고 확장 가능한 애플리케이션을 구축하는 데 필수적입니다. 이 레슨에서는 .NET 애플리케이션 성능을 측정, 분석, 개선하기 위한 도구와 기법을 다룹니다.

## 1. BenchmarkDotNet 설정 및 사용

### 1.1 시작하기

BenchmarkDotNet은 .NET의 표준 마이크로 벤치마킹 라이브러리입니다:

```bash
# 벤치마크용 새 콘솔 프로젝트 생성
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

[MemoryDiagnoser]  // 메모리 할당 추적
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

### 1.2 벤치마크 실행

```bash
# 항상 Release 모드에서 벤치마크를 실행하세요!
dotnet run -c Release

# 특정 벤치마크 클래스 실행
dotnet run -c Release -- --filter "*StringBenchmarks*"

# 결과 내보내기
dotnet run -c Release -- --exporters json csv markdown
```

### 1.3 샘플 출력

```
|             Method |      Mean |    Error |   StdDev | Ratio |  Gen0 | Allocated | Alloc Ratio |
|------------------- |----------:|---------:|---------:|------:|------:|----------:|------------:|
|    ConcatWithPlus  | 125.4 ns |  2.5 ns  |  2.3 ns  |  1.00 | 0.072 |     304 B |        1.00 |
| ConcatWithStringBu | 78.2 ns   |  1.3 ns  |  1.2 ns  |  0.62 | 0.038 |     160 B |        0.53 |
|    ConcatWithJoin  | 52.1 ns   |  0.8 ns  |  0.7 ns  |  0.42 | 0.024 |     104 B |        0.34 |
```

## 2. 벤치마크 작성

### 2.1 벤치마크 어트리뷰트

```csharp
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Jobs;

[SimpleJob(RuntimeMoniker.Net80)]       // .NET 8에서 실행
[SimpleJob(RuntimeMoniker.Net90)]       // .NET 9에서도 실행
[MemoryDiagnoser]                       // 할당 추적
[DisassemblyDiagnoser(maxDepth: 2)]     // JIT 생성 어셈블리 표시
[RankColumn]                            // 순위 열 추가
public class SortingBenchmarks
{
    [Params(100, 1000, 10_000)]  // 각 배열 크기에 대해 실행
    public int N;

    private int[] _data = null!;

    [GlobalSetup]  // 모든 벤치마크 전에 한 번 실행
    public void Setup()
    {
        var rng = new Random(42);  // 재현성을 위한 고정 시드
        _data = Enumerable.Range(0, N).Select(_ => rng.Next()).ToArray();
    }

    [IterationSetup]  // 각 벤치마크 반복 전에 실행
    public void IterationSetup()
    {
        // Sort가 변형하므로 각 반복마다 다시 섞기
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

### 2.2 매개변수화된 벤치마크

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

### 2.3 벤치마크 방법론 모범 사례

```csharp
[MemoryDiagnoser]
public class MethodologyDemo
{
    // 잘못된 방법: 벤치마크가 관찰 가능한 작업을 하지 않음; JIT가 제거할 수 있음
    // [Benchmark]
    // public void BadBenchmark()
    // {
    //     int x = 42 * 100;  // 죽은 코드 — JIT가 이것을 제거함
    // }

    // 올바른 방법: 결과를 반환하여 JIT가 작업을 제거할 수 없게 함
    [Benchmark]
    public int GoodBenchmark()
    {
        return 42 * 100;  // 반환되거나 소비되어야 함
    }

    // 잘못된 방법: 벤치마크에 설정 비용 포함
    // [Benchmark]
    // public int BadSetup()
    // {
    //     var data = Enumerable.Range(0, 1000).ToArray();  // 벤치마크 내 설정!
    //     return data.Sum();
    // }

    private int[] _data = null!;

    [GlobalSetup]
    public void Setup()
    {
        _data = Enumerable.Range(0, 1000).ToArray();  // 설정 분리
    }

    // 올바른 방법: 설정이 아닌 연산만 측정
    [Benchmark]
    public int GoodSetup()
    {
        return _data.Sum();
    }
}
```

## 3. 벤치마크 결과 해석

### 3.1 열 이해하기

```
|        Method |  N  |       Mean |    Error |   StdDev |     Median | Ratio | RatioSD |   Gen0 |  Gen1 | Allocated |
|-------------- |----:|-----------:|---------:|---------:|-----------:|------:|--------:|-------:|------:|----------:|
|     ArraySort | 100 |   1.234 us | 0.023 us | 0.021 us |   1.230 us |  1.00 |    0.00 |      - |     - |         - |
|      SpanSort | 100 |   1.198 us | 0.018 us | 0.015 us |   1.195 us |  0.97 |    0.02 |      - |     - |         - |
|   LinqOrderBy | 100 |   4.567 us | 0.089 us | 0.083 us |   4.550 us |  3.70 |    0.08 | 0.1221 |     - |     512 B |
```

```csharp
// 주요 열 설명:
//
// Mean     - 모든 반복의 평균 실행 시간
// Error    - 99.9% 신뢰 구간의 절반 (통계적 불확실성)
// StdDev   - 표준 편차 (측정값이 얼마나 퍼져 있는지)
// Median   - 중앙값 (Mean보다 이상치에 덜 영향 받음)
// Ratio    - 기준(Baseline) 벤치마크 대비 성능 (1.00 = 동일)
// RatioSD  - 비율의 표준 편차
// Gen0     - 1000 연산당 Gen 0 가비지 컬렉션 수
// Gen1     - 1000 연산당 Gen 1 가비지 컬렉션 수
// Allocated - 단일 연산당 할당된 총 바이트
//
// 단위: ns = 나노초, us = 마이크로초, ms = 밀리초
//
// 경험 법칙:
// - Error가 Mean의 > 10%이면 결과가 신뢰할 수 없음
// - 메모리 최적화를 위해 먼저 Allocated를 확인
// - Gen0 > 0이면 벤치마크가 GC 압력을 유발함
// - 절대 수치가 아닌 기준 대비 Ratio를 비교
```

### 3.2 할당 패턴 분석

```csharp
[MemoryDiagnoser]
public class AllocationBenchmarks
{
    private readonly int[] _source = Enumerable.Range(0, 1000).ToArray();

    [Benchmark(Baseline = true)]
    public int[] LinqToArray()
    {
        // 할당: LINQ 반복기 + 최종 배열
        return _source.Where(x => x % 2 == 0).ToArray();
    }

    [Benchmark]
    public int[] ManualFilter()
    {
        // 할당: List<int> 하나 (내부 배열 크기 조정) + 최종 배열
        var result = new List<int>(_source.Length / 2);  // 사전 크기 조정
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
        // 할당 제로 — 단순히 카운팅
        int count = 0;
        foreach (var x in _source)
        {
            if (x % 2 == 0)
                count++;
        }
        return count;
    }
}

// 일반적인 결과:
// LinqToArray:  ~2.5 us, 2.5 KB 할당
// ManualFilter: ~1.8 us, 2.1 KB 할당
// CountOnly:    ~0.3 us, 0 B 할당
```

## 4. 실시간 메트릭을 위한 dotnet-counters

### 4.1 설치 및 기본 사용법

```bash
# 도구 설치
dotnet tool install --global dotnet-counters

# 실행 중인 프로세스 모니터링
dotnet-counters monitor --process-id 12345

# 특정 카운터 모니터링
dotnet-counters monitor --process-id 12345 \
  --counters System.Runtime,Microsoft.AspNetCore.Hosting

# 프로세스 이름으로 모니터링
dotnet-counters monitor --process-id $(pgrep -f MyApp)
```

### 4.2 주요 성능 카운터

```bash
# System.Runtime 카운터:
# cpu-usage              - CPU 사용률 백분율
# working-set            - 작업 세트 메모리 (MB)
# gc-heap-size           - GC 힙 크기 (MB)
# gen-0-gc-count         - Gen 0 컬렉션 수
# gen-1-gc-count         - Gen 1 컬렉션 수
# gen-2-gc-count         - Gen 2 컬렉션 수
# gen-0-size             - Gen 0 힙 크기
# gen-1-size             - Gen 1 힙 크기
# gen-2-size             - Gen 2 힙 크기
# loh-size               - 대형 객체 힙 크기
# alloc-rate             - 할당 속도 (바이트/초)
# exception-count        - 발생한 예외 수
# threadpool-thread-count - 스레드 풀 스레드 수
# threadpool-queue-length - 스레드 풀 작업 항목 큐 길이

# 카운터를 파일로 수집
dotnet-counters collect --process-id 12345 \
  --format csv \
  --output perf-counters.csv \
  --counters System.Runtime
```

### 4.3 프로그래밍 방식 이벤트 카운터

```csharp
using System.Diagnostics.Tracing;

// 애플리케이션을 위한 사용자 지정 이벤트 카운터 생성
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

// 애플리케이션에서 사용:
// var sw = Stopwatch.StartNew();
// await HandleRequest();
// sw.Stop();
// AppEventSource.Instance.RecordRequest(sw.Elapsed.TotalMilliseconds);
```

## 5. 프로파일링을 위한 dotnet-trace

### 5.1 트레이스 수집

```bash
# 도구 설치
dotnet tool install --global dotnet-trace

# CPU 트레이스 수집 (기본 프로파일)
dotnet-trace collect --process-id 12345

# 특정 프로바이더로 수집
dotnet-trace collect --process-id 12345 \
  --providers Microsoft-DotNETCore-SampleProfiler,Microsoft-Windows-DotNETRuntime

# GC 이벤트 수집
dotnet-trace collect --process-id 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x1:5

# 특정 기간 동안 수집
dotnet-trace collect --process-id 12345 --duration 00:00:30

# 브라우저 보기를 위한 SpeedScope 형식으로 변환
dotnet-trace convert trace.nettrace --format speedscope
# .speedscope.json 파일을 https://www.speedscope.app/에서 열기
```

### 5.2 일반적인 프로파일링 프로바이더

```bash
# GC 상세 이벤트 (할당, 컬렉션, 파이널라이저)
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x1:Verbose

# 스레드 풀 이벤트
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x10000:Informational

# 예외 이벤트
dotnet-trace collect -p 12345 \
  --providers Microsoft-Windows-DotNETRuntime:0x8000:Informational

# HTTP 클라이언트 이벤트
dotnet-trace collect -p 12345 \
  --providers System.Net.Http

# Entity Framework Core 이벤트
dotnet-trace collect -p 12345 \
  --providers Microsoft.EntityFrameworkCore
```

## 6. 메모리 분석을 위한 dotnet-dump

### 6.1 덤프 캡처 및 분석

```bash
# 도구 설치
dotnet tool install --global dotnet-dump

# 덤프 캡처
dotnet-dump collect --process-id 12345

# 덤프 분석
dotnet-dump analyze core_20250115_123456

# 분석기 내부에서:
# > dumpheap -stat              # 힙 통계 표시
# > dumpheap -type System.String # 모든 String 객체 표시
# > gcroot 0x7f8a1234           # 객체를 살아있게 하는 것 찾기
# > dumpobj 0x7f8a1234          # 특정 객체 검사
# > eeheap -gc                  # GC 힙 세그먼트 정보
# > threadpool                  # 스레드 풀 정보
# > exit
```

### 6.2 일반적인 메모리 조사 패턴

```csharp
// 메모리 누수 찾기 - 일반적인 패턴:

// 패턴 1: 이벤트 핸들러 누수
public class LeakyService
{
    // 누수: 구독자가 게시자에 대한 참조를 보유
    public void Subscribe(EventPublisher publisher)
    {
        publisher.DataReceived += OnDataReceived;  // 누수!
        // 구독 해제 필요: publisher.DataReceived -= OnDataReceived;
    }

    private void OnDataReceived(object? sender, EventArgs e) { }
}

// 패턴 2: 영원히 커지는 정적 컬렉션
public static class Cache
{
    // 누수: 항목이 절대 제거되지 않음
    private static readonly Dictionary<string, byte[]> _cache = new();

    public static void Add(string key, byte[] data)
    {
        _cache[key] = data;  // 영원히 커짐!
    }

    // 수정: 제한된 캐시 또는 약한 참조 사용
}

// 패턴 3: 해제되지 않은 리소스
public class ResourceLeak
{
    public void Process()
    {
        var stream = new FileStream("data.txt", FileMode.Open);  // 누수!
        // 해제되지 않음 — 파이널라이저가 결국 정리하지만 GC 압력 증가

        // 수정:
        using var safeStream = new FileStream("data.txt", FileMode.Open);
    }
}
```

## 7. GC 세대와 할당 압력

### 7.1 GC 세대 이해

```csharp
public class GcGenerations
{
    public static void Demonstrate()
    {
        // .NET GC는 세대별 컬렉션을 사용합니다:
        // Gen 0: 단명 객체 (가장 자주 수집, ~10ms 일시 정지)
        // Gen 1: 중간 수명 객체 (Gen 0과 Gen 2 사이의 버퍼)
        // Gen 2: 장수 객체 (가장 드물게 수집, 비용이 높을 수 있음)
        // LOH:   대형 객체 힙 (>= 85,000바이트 객체, Gen 2와 함께 수집)

        var obj = new byte[100];
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 0

        GC.Collect(0);  // 살아남은 Gen 0 객체를 Gen 1로 승격
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 1

        GC.Collect(1);  // 살아남은 Gen 1 객체를 Gen 2로 승격
        Console.WriteLine($"Generation: {GC.GetGeneration(obj)}");  // 2

        // GC 통계
        Console.WriteLine($"Gen 0 collections: {GC.CollectionCount(0)}");
        Console.WriteLine($"Gen 1 collections: {GC.CollectionCount(1)}");
        Console.WriteLine($"Gen 2 collections: {GC.CollectionCount(2)}");
        Console.WriteLine($"Total memory: {GC.GetTotalMemory(false):N0} bytes");
        Console.WriteLine($"Total allocated: {GC.GetTotalAllocatedBytes():N0} bytes");

        // 상세 정보를 위한 GCMemoryInfo
        GCMemoryInfo info = GC.GetGCMemoryInfo();
        Console.WriteLine($"Heap size: {info.HeapSizeBytes:N0} bytes");
        Console.WriteLine($"Committed: {info.TotalCommittedBytes:N0} bytes");
    }
}
```

### 7.2 할당 압력 줄이기

```csharp
public class AllocationReduction
{
    // 나쁨: 매 호출마다 새 배열 할당
    public int[] GetEvenNumbers_Bad(int count)
    {
        return Enumerable.Range(0, count).Where(x => x % 2 == 0).ToArray();
    }

    // 더 나음: 버퍼 재사용
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

    // 나쁨: 값 타입 박싱
    public void BoxingExample_Bad()
    {
        object boxed = 42;             // 박싱: int -> object (힙 할당)
        int unboxed = (int)boxed;      // 언박싱
        Console.WriteLine(boxed);      // 박싱된 값에서 ToString()
    }

    // 좋음: 박싱 피하기
    public void BoxingExample_Good()
    {
        int value = 42;
        Console.WriteLine(value);      // 박싱 없음: int.ToString()이 직접 호출됨
    }

    // 나쁨: 핫 경로에서 LINQ (반복기 할당)
    public bool HasExpiredItems_Bad(List<Item> items)
    {
        return items.Any(i => i.ExpiresAt < DateTime.UtcNow);  // 대리자 + 반복기 할당
    }

    // 좋음: 수동 루프 (할당 제로)
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

## 8. 객체 풀링

### 8.1 Microsoft.Extensions의 ObjectPool<T>

```csharp
using Microsoft.Extensions.ObjectPool;

// StringBuilder를 위한 간단한 풀링
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
            Pool.Return(sb);  // 풀에 반환 (StringBuilder를 비움)
        }
    }
}
```

### 8.2 사용자 지정 객체 풀

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
            // 풀이 가득 참 — GC가 이 인스턴스를 수집하게 함
        }
    }
}

// 사용:
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
//     // 버퍼 사용...
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
        // 공유 풀에서 배열 대여 (요청보다 클 수 있음)
        double[] buffer = ArrayPool<double>.Shared.Rent(estimatedCount);
        try
        {
            int count = 0;
            foreach (double value in source)
            {
                if (count >= buffer.Length)
                {
                    // 더 큰 버퍼 필요
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

## 9. 문자열 최적화

### 9.1 문자열 인터닝

```csharp
public class StringInterning
{
    public static void Demonstrate()
    {
        // 문자열 리터럴은 자동으로 인터닝됨
        string a = "Hello";
        string b = "Hello";
        Console.WriteLine(ReferenceEquals(a, b));  // True (같은 객체)

        // 런타임에 생성된 문자열은 기본적으로 인터닝되지 않음
        string c = new string(new char[] { 'H', 'e', 'l', 'l', 'o' });
        Console.WriteLine(ReferenceEquals(a, c));  // False (다른 객체)
        Console.WriteLine(a == c);                  // True (같은 값)

        // 수동으로 문자열 인터닝
        string d = string.Intern(c);
        Console.WriteLine(ReferenceEquals(a, d));  // True

        // 문자열이 인터닝되었는지 확인 (인터닝하지 않고)
        string? e = string.IsInterned(c);
        Console.WriteLine(e is not null);  // True ("Hello"가 이미 인터닝되었으므로)
    }
}
```

### 9.2 할당 제로 포맷팅을 위한 string.Create

```csharp
public class StringCreateExamples
{
    // string.Create는 중간 할당 없이 문자열을 제자리에서 구성
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

    // 타임스탬프를 효율적으로 포맷
    public static string FormatTimestamp(DateTime dt)
    {
        // "2025-01-15T10:30:45"
        return string.Create(19, dt, (chars, dt) =>
        {
            dt.TryFormat(chars, out _, "yyyy-MM-ddTHH:mm:ss");
        });
    }

    // 패딩과 정렬
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

### 9.3 StringBuilder 모범 사례

```csharp
using System.Text;

public class StringBuilderOptimization
{
    // 예상 출력 길이를 알 때 사전 크기 조정
    public static string BuildCsv(List<string[]> rows)
    {
        int estimatedLength = rows.Count * 100;  // 행당 ~100자 추정
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

    // 성능을 위해 AppendFormat 대신 Append 사용
    public static string BuildTable(List<(string Name, int Age, string City)> people)
    {
        var sb = new StringBuilder(people.Count * 50);

        // 느림:
        // sb.AppendFormat("| {0,-20} | {1,4} | {2,-15} |", p.Name, p.Age, p.City);

        // 빠름: 직접 Append 호출
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

## 10. 값 타입 vs 참조 타입 성능

### 10.1 스택 vs 힙 할당

```csharp
// 참조 타입: 힙에 할당
public class PointClass
{
    public double X { get; set; }
    public double Y { get; set; }
}

// 값 타입: 스택에 할당 (로컬 변수일 때)
public struct PointStruct
{
    public double X { get; set; }
    public double Y { get; set; }
}

// readonly struct: 컴파일러 강제 불변성, 방어적 복사 없음
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
            var p = new PointClass { X = i, Y = i * 2 };  // 힙 할당
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
            var p = new PointStruct { X = i, Y = i * 2 };  // 스택 — 할당 없음
            sum += p.X + p.Y;
        }
        return sum;
    }
}

// 일반적인 결과:
// CreateManyClasses:  ~50 us,  ~320 KB 할당 (10,000개 힙 객체)
// CreateManyStructs:  ~15 us,  0 B 할당
```

### 10.2 구조체 vs 클래스 사용 시점

```csharp
// 구조체(STRUCT)를 사용하는 경우:
// 1. 작을 때 (이상적으로 < ~16바이트, 최대 ~64바이트)
// 2. 자주 생성되고 파괴될 때 (높은 할당 속도)
// 3. 논리적으로 단일 값을 나타낼 때 (Point, Color, DateTime 같은)
// 4. 불변일 때 (readonly struct 사용)

public readonly struct Color
{
    public byte R { get; init; }
    public byte G { get; init; }
    public byte B { get; init; }
    public byte A { get; init; }
}

// 클래스(CLASS)를 사용하는 경우:
// 1. 클 때 (많은 필드)
// 2. 장수 객체일 때 (캐시됨, 컬렉션에 저장됨)
// 3. 상속/다형성이 필요할 때
// 4. 참조 의미론이 필요할 때 (여러 변수가 같은 인스턴스를 공유)

public class CustomerProfile
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public string Email { get; set; } = "";
    public List<Order> Orders { get; set; } = new();
    public Address? ShippingAddress { get; set; }
    // 대형 객체 — 클래스가 적절
}

// 값 의미론 DTO를 위한 record struct 사용
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

## 11. 핫 경로 최적화 기법

### 11.1 핫 경로 식별

```csharp
// "핫 경로(Hot Path)"는 자주 실행되며 성능이 중요한 코드입니다.
// 예시:
// - 알고리즘의 내부 루프
// - 웹 서버의 요청 처리
// - 높은 처리량 시스템의 직렬화/역직렬화
// - 이벤트 기반 시스템의 메시지 처리

// 기법 1: 핫 경로에서 할당 피하기
public class HotPathExample
{
    // 콜드 경로: 시작 시 한 번 호출
    public void Initialize()
    {
        // 여기서의 할당은 괜찮음
        var config = LoadConfiguration();
        _lookup = BuildLookup(config);
    }

    private Dictionary<string, int> _lookup = new();

    // 핫 경로: 요청마다 호출 (초당 수천 번)
    public int ProcessRequest(ReadOnlySpan<char> key)
    {
        // 할당 없음! key는 문자열이 아닌 span
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

### 11.2 분기 예측과 데이터 레이아웃

```csharp
public class DataLayoutOptimization
{
    // 느림: 나쁜 데이터 지역성 (객체 배열 = 힙에 흩어진 포인터 배열)
    public static double SumClasses(PointClass[] points)
    {
        double sum = 0;
        foreach (var p in points)
            sum += p.X + p.Y;  // 각 접근마다 힙으로의 포인터를 따라감
        return sum;
    }

    // 빠름: 좋은 데이터 지역성 (구조체 배열 = 연속 메모리 블록)
    public static double SumStructs(PointStruct[] points)
    {
        double sum = 0;
        foreach (var p in points)
            sum += p.X + p.Y;  // 순차적 메모리 접근, CPU 캐시 친화적
        return sum;
    }

    // 가장 빠름: 배열의 구조체 (SoA) — SIMD 친화적 접근
    public static double SumSoA(double[] xs, double[] ys, int count)
    {
        double sum = 0;
        for (int i = 0; i < count; i++)
            sum += xs[i] + ys[i];
        return sum;
    }
}
```

### 11.3 일반적인 성능 함정 피하기

```csharp
public class PerformancePitfalls
{
    // 함정 1: 문자열 키를 사용한 딕셔너리 조회 (대소문자 무시)
    // 느림: ToLower가 매 조회마다 새 문자열을 할당
    public bool LookupSlow(Dictionary<string, int> dict, string key)
    {
        return dict.ContainsKey(key.ToLower());  // 할당!
    }

    // 빠름: 딕셔너리 생성 시 대소문자 무시 비교자 사용
    public Dictionary<string, int> CreateFastDict()
    {
        return new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
    }

    // 함정 2: 정상 흐름에서의 과도한 예외 던지기
    // 느림: 예외는 매우 비용이 높음
    public int ParseSlow(string input)
    {
        try { return int.Parse(input); }
        catch { return 0; }
    }

    // 빠름: TryParse 사용
    public int ParseFast(string input)
    {
        return int.TryParse(input, out int result) ? result : 0;
    }

    // 함정 3: 루프에서 문자열 연결
    // 느림: O(n^2) 문자열 할당
    public string JoinSlow(string[] items)
    {
        string result = "";
        foreach (var item in items)
            result += item + ",";
        return result;
    }

    // 빠름: 단일 할당으로 O(n)
    public string JoinFast(string[] items)
    {
        return string.Join(",", items);
    }
}
```

## 12. 실전 예제: 데이터 프로세서 벤치마킹과 최적화

### 12.1 기준 구현

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

### 12.2 최적화된 구현

```csharp
public class DataProcessor_V2
{
    // LINQ 할당 피하기; 딕셔너리 직접 사용
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

    // 통계를 위한 단일 패스, 사전 크기 조정된 출력 리스트
    public List<SensorReading> FilterOutliers(
        List<SensorReading> readings, double stdDevThreshold)
    {
        // 패스 1: 센서별 통계 계산
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

        // 패스 2: 사전 계산된 경계로 필터링
        var result = new List<SensorReading>(readings.Count);  // 사전 크기 조정
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

### 12.3 벤치마크

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

// 예상 결과 (대략):
// V2_Averages: ~40% 더 빠름, ~60% 적은 메모리
// V2_Filter:   ~50% 더 빠름, ~70% 적은 메모리 (특히 100K 리딩에서)
```

## 13. 연습 문제

1. **문자열 연산 벤치마크**: 1,000개의 정수로 쉼표로 구분된 리스트를 만드는 네 가지 방법을 비교하는 BenchmarkDotNet 벤치마크를 작성하세요: (a) `+`를 사용한 문자열 연결, (b) `StringBuilder`, (c) `string.Join`, (d) `string.Create`. 벤치마크 클래스를 보여주고, 실행하고, 어떤 것이 가장 빠르며 그 이유를 설명하세요.

2. **객체 풀 구현**: 스레드 안전하고 최대 크기를 구성할 수 있으며 초기화 동작을 지원하는 제네릭 `ObjectPool<T>`를 구현하세요. `MemoryStream` 객체를 풀링하는 데 사용하세요. 10,000회 반복에 걸쳐 풀링된 것과 풀링되지 않은 `MemoryStream` 사용을 비교하는 벤치마크를 작성하세요.

3. **GC 압력 분석**: 루프에서 100만 개의 작은 객체(예: `new byte[64]`)를 생성하여 할당 압력을 만드는 프로그램을 작성하세요. 전후의 `GC.CollectionCount`를 사용하여 Gen 0, Gen 1, Gen 2 컬렉션이 몇 번 발생했는지 측정하세요. 그런 다음 `ArrayPool<byte>`를 사용하여 루프를 다시 작성하고 컬렉션 횟수를 비교하세요.

4. **값 타입 최적화**: `Position` (X, Y, Z), `Velocity` (X, Y, Z), `Mass`가 있는 100,000개 입자의 입자 시뮬레이션이 있습니다. 먼저 `class Particle`로, 그 다음 `struct Particle`로 구현하세요. 모든 입자를 한 타임스텝 전진시키는 `UpdatePositions` 메서드를 벤치마킹하세요. 성능 차이를 설명하세요.

5. **핫 경로 분석**: HTTP 헤더를 처리하는 다음 메서드에서 세 가지 성능 문제를 식별하고 이를 제거하도록 메서드를 다시 작성하세요. 두 버전을 벤치마킹하세요.

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
