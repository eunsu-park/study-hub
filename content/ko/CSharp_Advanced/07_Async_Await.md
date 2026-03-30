# Async와 Await

**이전**: [이벤트와 델리게이트](./06_Events_and_Delegates.md) | **다음**: [동시성과 병렬성](./08_Concurrency_and_Parallelism.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 비동기 프로그래밍(asynchronous programming)이 I/O 바운드 및 CPU 바운드 작업에 왜 중요한지 설명하기
2. `Task`, `Task<T>`, `ValueTask<T>`를 사용하여 비동기 작업 표현하기
3. `async`와 `await` 키워드를 사용하여 올바른 비동기 메서드 작성하기
4. 비동기 코드에서 예외와 취소 처리하기
5. `Task.WhenAll`과 `Task.WhenAny`로 여러 비동기 작업 합성하기
6. 동기화 컨텍스트(synchronization context)를 이해하고 `ConfigureAwait(false)` 사용 시점 파악하기
7. `IAsyncEnumerable<T>`로 비동기 스트림 소비하기
8. 데드락과 fire-and-forget 실수 같은 일반적인 비동기 함정 피하기

---

현대 애플리케이션은 대부분의 시간을 대기하며 보냅니다: 데이터베이스 쿼리, HTTP 응답, 파일 읽기, 사용자 입력을 기다립니다. 전통적인 동기 코드는 대기하는 동안 스레드를 차단하여 다른 요청을 처리할 수 있는 리소스를 낭비합니다. C#의 `async`/`await` 패턴을 사용하면 *순차적으로 보이는* 코드를 작성하면서도 대기 중에는 스레드를 해제하여 처리량과 응답성을 극적으로 향상시킬 수 있습니다. 이 레슨에서는 `Task`의 기본부터 `ValueTask`, 채널, 비동기 스트림 같은 고급 주제까지 다룹니다.

## 1. 왜 Async인가? I/O 바운드 vs CPU 바운드 작업

모든 워크로드가 비동기로부터 동일한 이점을 얻는 것은 아닙니다. I/O 바운드와 CPU 바운드 작업의 차이를 이해하는 것이 효율적인 비동기 코드를 작성하는 첫 번째 단계입니다.

### 1.1 I/O 바운드 작업

I/O 바운드 작업은 외부 리소스(네트워크 소켓, 디스크 컨트롤러, 데이터베이스 엔진)를 기다리는 데 시간을 소비합니다. 대기 중인 스레드는 유휴 상태이므로 다른 요청을 처리하기 위해 스레드 풀에 반환될 수 있습니다.

```csharp
// 동기식 — HTTP 응답을 기다리는 동안 스레드가 차단됨
public string GetPageSync(string url)
{
    using var client = new HttpClient();
    // 이 호출은 응답이 도착할 때까지 호출 스레드를 차단함
    HttpResponseMessage response = client.Send(new HttpRequestMessage(HttpMethod.Get, url));
    return response.Content.ReadAsStringAsync().Result; // 또 다른 차단
}
```

```csharp
// 비동기식 — 대기하는 동안 스레드가 해제됨
public async Task<string> GetPageAsync(string url)
{
    using var client = new HttpClient();
    HttpResponseMessage response = await client.GetAsync(url);
    return await response.Content.ReadAsStringAsync();
}
```

비동기 버전에서는 두 `await` 지점 모두에서 스레드가 풀에 반환됩니다. 수천 개의 동시 요청을 처리하는 웹 서버에서 이 차이는 매우 큽니다.

### 1.2 CPU 바운드 작업

CPU 바운드 작업은 계산(이미지 처리, 암호화, 시뮬레이션)으로 프로세서를 바쁘게 유지합니다. 비동기로 CPU 작업이 더 빨라지지는 않으며, 작업에는 여전히 스레드가 필요합니다. 하지만 `Task.Run`으로 CPU 작업을 백그라운드 스레드로 오프로드하여 UI의 응답성을 유지할 수 있습니다.

```csharp
// CPU 바운드: 스레드 풀 스레드로 오프로드
public async Task<double> ComputePiAsync(int digits)
{
    return await Task.Run(() => ComputePi(digits));
}
```

**경험 법칙**: I/O 바운드 작업에는 `async`/`await`를 *직접* 사용하세요. CPU 바운드 작업은 UI 스레드를 자유롭게 유지해야 할 때만 `Task.Run`을 사용하여 호출 스레드에서 분리하세요.

## 2. Task와 Task&lt;T&gt;

`Task`는 완료되었거나 아직 완료되지 않은 비동기 작업을 나타냅니다. `Task<T>`는 결과 값을 추가합니다.

### 2.1 Task 생성과 관찰

```csharp
// 값이 있는 완료된 태스크
Task<int> completed = Task.FromResult(42);

// 값이 없는 완료된 태스크
Task done = Task.CompletedTask;

// 실패한 작업을 나타내는 태스크
Task faulted = Task.FromException(new InvalidOperationException("oops"));

// 취소를 나타내는 태스크
Task cancelled = Task.FromCanceled(new CancellationToken(true));
```

### 2.2 Task 상태

```csharp
Task<string> task = SomeAsyncOperation();

Console.WriteLine(task.Status);       // WaitingForActivation, Running, RanToCompletion, Faulted, Canceled
Console.WriteLine(task.IsCompleted);  // RanToCompletion, Faulted, Canceled일 때 true
Console.WriteLine(task.IsCompletedSuccessfully); // RanToCompletion일 때만 true
```

### 2.3 연속(Continuation) (저수준)

`await` 이전에는 연속을 수동으로 설정했습니다. 오늘날 이것이 필요한 경우는 드물지만, 이를 이해하면 `await`가 내부적으로 무엇을 하는지 추론하는 데 도움이 됩니다.

```csharp
Task<byte[]> downloadTask = client.GetByteArrayAsync(url);

downloadTask.ContinueWith(t =>
{
    if (t.IsFaulted)
    {
        Console.WriteLine($"에러: {t.Exception?.InnerException?.Message}");
        return;
    }
    File.WriteAllBytes("output.dat", t.Result);
}, TaskContinuationOptions.ExecuteSynchronously);
```

## 3. async/await 키워드와 동작 원리

### 3.1 기본 사항

`async` 수정자는 컴파일러에게 메서드 본문을 상태 머신(state machine)으로 변환하라고 알려줍니다. `await` 키워드는 대기 중인 태스크가 완료될 때까지 메서드가 제어를 양보하는 일시 중단 지점을 표시합니다.

```csharp
public async Task<string> ReadFileAsync(string path)
{
    // 실행은 여기까지 동기적으로 진행됨
    Console.WriteLine("await 이전");

    // 이 시점에서 파일 읽기가 완료되지 않았다면, 메서드는
    // 호출자에게 Task<string>을 반환하고 스레드가 해제됨
    string content = await File.ReadAllTextAsync(path);

    // 실행이 여기서 재개됨 (다른 스레드일 수 있음)
    Console.WriteLine("await 이후");
    return content;
}
```

### 3.2 내부의 상태 머신

컴파일러는 모든 비동기 메서드를 `IAsyncStateMachine`을 구현하는 구조체로 다시 작성합니다. 각 `await`는 상태 전환이 됩니다. 대기 중인 태스크가 완료되면 런타임이 `MoveNext()`를 호출하여 다음 상태로 진행합니다.

```csharp
// 개념적 컴파일러 변환 (간략화)
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
                Console.WriteLine("await 이전");
                _awaiter = File.ReadAllTextAsync(Path).GetAwaiter();
                if (!_awaiter.IsCompleted)
                {
                    State = 1;
                    Builder.AwaitUnsafeOnCompleted(ref _awaiter, ref this);
                    return; // 양보
                }
                goto case 1;
            case 1:
                string content = _awaiter.GetResult();
                Console.WriteLine("await 이후");
                Builder.SetResult(content);
                break;
        }
    }
}
```

이 변환을 이해하면 `async` 메서드가 동기 메서드보다 약간 더 많은 오버헤드를 가지는 이유와 `ValueTask`가 핫 경로에서 도움이 되는 이유를 알 수 있습니다.

## 4. 비동기 메서드 시그니처와 반환 타입

### 4.1 세 가지 표준 반환 타입

```csharp
// 값을 반환
public async Task<int> GetCountAsync() { ... }

// 값을 반환하지 않음
public async Task SaveAsync() { ... }

// 아무것도 반환하지 않으며 await 불가 — 이벤트 핸들러 외에는 사용 자제
public async void OnButtonClick(object sender, EventArgs e) { ... }
```

### 4.2 ValueTask와 ValueTask&lt;T&gt;

`ValueTask<T>`는 결과가 이미 동기적으로 사용 가능할 때 힙 할당을 피하는 구조체입니다. 동기적으로 완료되는 경우가 많은 메서드(캐시 히트, 버퍼 읽기)에 이상적입니다.

```csharp
private readonly ConcurrentDictionary<string, User> _cache = new();

public ValueTask<User> GetUserAsync(string id)
{
    if (_cache.TryGetValue(id, out User? cached))
    {
        // Task 할당 불필요
        return new ValueTask<User>(cached);
    }

    return new ValueTask<User>(LoadUserFromDatabaseAsync(id));
}

private async Task<User> LoadUserFromDatabaseAsync(string id)
{
    // ... 데이터베이스 호출
    User user = await _db.Users.FindAsync(id);
    _cache[id] = user;
    return user;
}
```

### 4.3 IAsyncEnumerable&lt;T&gt; 반환 타입

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

## 5. 비동기 코드에서의 예외 처리

### 5.1 대기된 태스크의 예외

실패한 태스크를 `await`하면 예외가 `AggregateException`에서 언래핑되어 `await` 지점에서 다시 던져집니다. 이로써 자연스러운 try/catch 경험을 얻을 수 있습니다.

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
        Console.WriteLine($"네트워크 오류: {ex.Message}");
    }
    catch (IOException ex)
    {
        Console.WriteLine($"저장 오류: {ex.Message}");
    }
}
```

### 5.2 Task.WhenAll에서의 다중 예외

여러 태스크가 실패하면 `Task.WhenAll`은 반환된 태스크의 `Exception` 속성에 모든 예외를 저장하지만, `await`는 첫 번째 예외만 다시 던집니다.

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
        // ex는 첫 번째 예외만 포함
        Console.WriteLine($"첫 번째: {ex.Message}");

        // 태스크를 통해 모든 예외에 접근
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

### 5.3 async void 메서드의 예외

`async void` 메서드의 예외는 호출자가 catch할 수 없습니다. 예외는 동기화 컨텍스트로 전파되어 종종 프로세스를 크래시시킵니다.

```csharp
// 위험 — 예외가 애플리케이션을 크래시시킴
public async void FireAndForgetBad()
{
    await Task.Delay(100);
    throw new InvalidOperationException("관찰되지 않음!");
}

// 안전 — 호출자가 예외를 관찰할 수 있도록 Task를 반환
public async Task FireAndForgetSafe()
{
    await Task.Delay(100);
    throw new InvalidOperationException("catch할 수 있음!");
}
```

## 6. CancellationToken을 이용한 취소

.NET에서 취소는 협력적입니다. `CancellationToken`을 비동기 메서드에 전달하면 메서드가 주기적으로 취소가 요청되었는지 확인합니다.

### 6.1 기본 사용법

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
        Console.WriteLine("다운로드가 시간 초과되었거나 취소되었습니다.");
        return string.Empty;
    }
}
```

### 6.2 토큰 연결

여러 취소 소스를 연결하여 그 중 하나를 취소하면 연결된 토큰도 취소되도록 할 수 있습니다.

```csharp
public async Task ProcessAsync(CancellationToken externalToken)
{
    using var timeoutCts = new CancellationTokenSource(TimeSpan.FromSeconds(30));
    using var linkedCts = CancellationTokenSource.CreateLinkedTokenSource(
        externalToken, timeoutCts.Token);

    await DoWorkAsync(linkedCts.Token);
}
```

### 6.3 수동 취소 확인

CPU 바운드 루프의 경우 주기적으로 토큰을 확인합니다.

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

## 7. Task.WhenAll과 Task.WhenAny

### 7.1 Task.WhenAll — 병렬 합성

`Task.WhenAll`은 모든 태스크가 완료될 때 완료됩니다. 독립적인 비동기 작업을 동시에 실행하는 주요 방법입니다.

```csharp
public async Task<DashboardData> LoadDashboardAsync()
{
    Task<User> userTask = GetUserAsync();
    Task<List<Order>> ordersTask = GetOrdersAsync();
    Task<List<Notification>> notificationsTask = GetNotificationsAsync();

    // 세 요청이 동시에 실행됨
    await Task.WhenAll(userTask, ordersTask, notificationsTask);

    return new DashboardData
    {
        User = userTask.Result,
        Orders = ordersTask.Result,
        Notifications = notificationsTask.Result
    };
}
```

### 7.2 Task.WhenAny — 첫 번째 완료

`Task.WhenAny`는 첫 번째 태스크가 완료될 때 완료됩니다. 타임아웃, 중복 요청, 진행 보고에 유용합니다.

```csharp
public async Task<string> FetchWithFallbackAsync(string primaryUrl, string fallbackUrl)
{
    using var client = new HttpClient();

    Task<string> primary = client.GetStringAsync(primaryUrl);
    Task<string> fallback = client.GetStringAsync(fallbackUrl);

    Task<string> winner = await Task.WhenAny(primary, fallback);
    return await winner; // 결과를 언래핑 (예외가 있으면 전파)
}
```

### 7.3 완료되는 순서대로 태스크 처리

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
            Console.WriteLine($"{content.Length}자 수신");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"실패: {ex.Message}");
        }
    }
}
```

## 8. ConfigureAwait와 동기화 컨텍스트

### 8.1 동기화 컨텍스트란?

동기화 컨텍스트(synchronization context)는 현재 실행 환경(예: UI 스레드, ASP.NET 요청 컨텍스트)을 캡처하여 `await` 이후의 연속이 동일한 컨텍스트에서 실행되도록 합니다.

```csharp
// WPF UI 스레드에서:
public async Task UpdateUIAsync()
{
    string data = await FetchDataAsync(); // UI 스레드를 양보
    // await 이후, UI 스레드에 돌아옴 — 컨트롤 업데이트가 안전함
    MyLabel.Content = data;
}
```

### 8.2 ConfigureAwait(false)

원래 컨텍스트로 돌아갈 필요가 없는 라이브러리 코드에서는 `ConfigureAwait(false)`를 사용하여 불필요한 컨텍스트 전환과 잠재적 데드락을 방지합니다.

```csharp
public async Task<byte[]> ReadBytesAsync(string path)
{
    // 라이브러리 코드: 돌아갈 UI가 없음
    byte[] buffer = await File.ReadAllBytesAsync(path).ConfigureAwait(false);
    return Compress(buffer);
}
```

### 8.3 클래식 데드락

```csharp
// ASP.NET Framework (.NET Core 이전) 또는 SynchronizationContext가 있는 WPF
public string GetDataDeadlock()
{
    // .Result가 UI/요청 스레드를 차단함
    // GetDataAsync의 연속이 같은 스레드에서 재개하려고 함
    // 데드락
    return GetDataAsync().Result;
}

public async Task<string> GetDataAsync()
{
    // ConfigureAwait(false) 없이는 연속이 원래 컨텍스트를 필요로 함
    return await httpClient.GetStringAsync("https://example.com");
}
```

**해결 방법**: 라이브러리 메서드에서 `ConfigureAwait(false)`를 사용하거나, 더 좋은 방법은 `.Result` 대신 끝까지 `await`를 사용하는 것입니다.

## 9. 고성능 시나리오를 위한 ValueTask

### 9.1 ValueTask를 사용해야 할 때

`ValueTask<T>`는 작업이 동기적으로 완료될 때 `Task<T>` 객체의 힙 할당을 피합니다. 이것은 초당 수천 번 호출되는 핫 경로에서 중요합니다.

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
            // 핫 경로: 할당 없이 버퍼에서 반환
            return new ValueTask<int>(_buffer[_position++]);
        }

        // 콜드 경로: 실제 I/O가 필요
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

### 9.2 ValueTask 규칙

```csharp
// 해야 할 것: 한 번만 await
int result = await GetValueAsync();

// 하면 안 되는 것: 여러 번 await
ValueTask<int> vt = GetValueAsync();
int r1 = await vt;
// int r2 = await vt; // 정의되지 않은 동작

// 하면 안 되는 것: IsCompleted가 true가 아닌 한 .Result나 .GetAwaiter().GetResult() 사용
// 하면 안 되는 것: Task.WhenAll과 결합 (먼저 Task로 변환)
Task<int> task = GetValueAsync().AsTask(); // 필요할 때 변환
```

## 10. 일반적인 비동기 함정

### 10.1 async void

`async void` 메서드는 await할 수 없고, 예외를 관찰할 수 없으며, 단위 테스트를 어렵게 만듭니다. 대신 `async Task`를 사용하세요.

```csharp
// 나쁜 예: async void
public async void Initialize()
{
    await LoadConfigAsync(); // 이것이 예외를 던지면 프로세스가 크래시될 수 있음
}

// 좋은 예: async Task
public async Task InitializeAsync()
{
    await LoadConfigAsync();
}
```

### 10.2 안전한 Fire-and-Forget

fire-and-forget가 정말 필요한 경우, 최소한 예외를 관찰하세요.

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

// 사용법
SendEmailAsync(user.Email).SafeFireAndForget(ex =>
    _logger.LogError(ex, "이메일 전송 실패"));
```

### 10.3 비동기 코드에서의 차단

동기화 컨텍스트가 있는 코드에서 `.Result`, `.Wait()`, `.GetAwaiter().GetResult()`를 절대 사용하지 마세요. 이것이 비동기 데드락의 가장 흔한 원인입니다.

```csharp
// 나쁜 예 — 잠재적 데드락
public void HandleRequest()
{
    var data = GetDataAsync().Result;
}

// 좋은 예 — 끝까지 async
public async Task HandleRequestAsync()
{
    var data = await GetDataAsync();
}
```

### 10.4 불필요한 async/await

`await` 이후에 아무것도 하지 않고 단순히 태스크를 전달하는 경우, 상태 머신 오버헤드를 피하기 위해 태스크를 직접 반환할 수 있습니다.

```csharp
// 불필요한 상태 머신
public async Task<User> GetUserAsync(int id)
{
    return await _repository.FindAsync(id);
}

// 더 효율적 — 상태 머신이 생성되지 않음
public Task<User> GetUserAsync(int id)
{
    return _repository.FindAsync(id);
}
```

**주의 사항**: async/await를 생략하면 예외 동작이 변경됩니다. `await` 전에 던져진 예외는 반환된 태스크에 래핑되지 않고 호출 지점에서 발생합니다. 이 트레이드오프를 이해한 경우에만 생략하세요.

## 11. 비동기 스트림을 위한 IAsyncEnumerable&lt;T&gt;

`IAsyncEnumerable<T>` (C# 8에서 도입)는 각 요소를 생산하는 데 비동기 작업이 필요할 수 있는 시퀀스를 생산하고 소비할 수 있게 해줍니다.

### 11.1 비동기 스트림 생산

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

### 11.2 비동기 스트림 소비

```csharp
await foreach (LogEntry entry in StreamLogsAsync("app.log"))
{
    if (entry.Level == LogLevel.Error)
    {
        Console.WriteLine($"[ERROR] {entry.Timestamp}: {entry.Message}");
    }
}
```

### 11.3 System.Linq.Async를 이용한 LINQ 스타일 작업

```csharp
// 설치: dotnet add package System.Linq.Async
var errors = StreamLogsAsync("app.log")
    .Where(e => e.Level == LogLevel.Error)
    .Take(10);

await foreach (var error in errors)
{
    Console.WriteLine(error.Message);
}
```

## 12. 실전 예제: 병렬 HTTP 클라이언트

이 예제는 이 레슨에서 다룬 많은 개념을 보여줍니다: `Task.WhenAll`, 스로틀링을 위한 `SemaphoreSlim`, `CancellationToken`, 예외 처리, 그리고 `IAsyncEnumerable`.

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
    /// 모든 URL을 동시에 다운로드(최대 maxConcurrency)하며,
    /// 완료되는 대로 결과를 반환합니다.
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
    /// 모든 URL을 다운로드하고 모두 완료되면 반환합니다.
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
// 사용법
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
        Console.WriteLine($"OK  {result.Url} ({result.ElapsedMs}ms, {result.Content?.Length}자)");
    else
        Console.WriteLine($"ERR {result.Url}: {result.Error}");
}
```

## 13. 연습 문제

1. **비동기 파일 처리기**: 디렉토리에서 모든 `.txt` 파일을 동시에 읽되(`SemaphoreSlim`을 사용하여 한 번에 3개로 제한), 각 파일의 단어 수를 세고 파일 이름과 단어 수를 매핑하는 `Dictionary<string, int>`을 반환하는 비동기 메서드를 작성하세요. `CancellationToken`을 통한 취소를 지원하세요.

2. **지수 백오프를 이용한 재시도**: `Func<CancellationToken, Task<T>>`, 최대 재시도 횟수, 기본 지연 시간을 받는 제네릭 `RetryAsync<T>` 메서드를 구현하세요. 실패 시 `baseDelay * 2^attempt` 만큼 대기한 후 재시도해야 합니다. `CancellationToken`을 사용하여 재시도 루프를 중단할 수 있어야 합니다. 첫 번째 성공 결과를 반환하거나 관찰된 모든 예외를 포함하는 `AggregateException`을 던지세요.

3. **비동기 속도 제한기**: 시간 창당 N개 작업으로 제한하는(예: 초당 10개 요청) `RateLimiter` 클래스를 만드세요. 호출자가 요청 전에 호출하는 `async Task WaitForSlotAsync(CancellationToken token)` 메서드를 제공하세요. 내부적으로 `SemaphoreSlim`과 `Task.Delay`를 사용하세요. 속도가 준수되는지 검증하는 테스트를 작성하세요.

4. **스트리밍 파이프라인**: `IAsyncEnumerable<T>`를 사용하여 세 단계 파이프라인을 만드세요: (a) 대용량 파일에서 비동기적으로 줄을 읽는 프로듀서, (b) 각 줄을 레코드 타입으로 파싱하는 변환기, (c) 레코드를 100개씩 묶어 각 배치를 별도의 출력 파일에 쓰는 소비자. 각 단계는 다음 단계에 양보하는 별도의 비동기 메서드여야 합니다.

5. **타임아웃 데코레이터**: 모든 비동기 작업에 타임아웃을 래핑하는 `WithTimeout<T>(Func<CancellationToken, Task<T>> operation, TimeSpan timeout)` 메서드를 작성하세요. 타임아웃이 만료되면 연결된 `CancellationTokenSource`를 사용하여 작업을 취소하고 `TimeoutException`을 던져야 합니다. 기본 작업이 취소 신호를 수신하는지 검증하세요.
