# Nullable 참조 타입

**이전**: [패턴 매칭](./04_Pattern_Matching.md) | **다음**: [레코드와 불변성](./06_Records_and_Immutability.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. null 참조 문제와 nullable 참조 타입이 도입된 이유 설명하기
2. 파일 수준과 프로젝트 수준에서 nullable 컨텍스트를 활성화하고 구성하기
3. nullable(`?`)과 non-nullable 선언으로 타입에 주석 달기
4. null 용서 연산자(`!`)를 적절하고 절제하여 사용하기
5. null 병합(`??`), null 조건부(`?.`), null 병합 할당(`??=`) 연산자 적용하기
6. 컴파일러가 nullable에 대한 흐름 분석을 수행하는 방식 이해하기
7. 복잡한 null 계약을 표현하기 위한 nullable 특성 사용하기
8. 기존 코드베이스를 nullable 참조 타입으로 마이그레이션하기

---

`NullReferenceException`은 C#(및 다른 많은 언어)에서 가장 흔한 런타임 오류 중 하나입니다. 1965년에 null 참조를 도입한 Tony Hoare는 이를 유명하게도 자신의 "10억 달러짜리 실수"라고 불렀습니다. C# 8에서 도입된 nullable 참조 타입(NRT)은 코드가 null을 역참조할 수 있을 때 경고하는 정적 분석을 컴파일러에 추가하여, 런타임 오류의 한 범주를 컴파일 타임 경고로 전환합니다.

## 1. Null 참조 문제

### 1.1 Null이 위험한 이유

NRT 이전에는 C#의 모든 참조 타입이 암시적으로 nullable이었습니다. 컴파일러는 절대 null이 아니어야 하는 변수와 의도적으로 null일 수 있는 변수를 구분하는 데 도움을 주지 않았습니다.

```csharp
// NRT 이전 — 모든 것이 null이 될 수 있음, 경고 없음
public class UserService
{
    public User GetUser(int id)
    {
        // 사용자를 찾지 못하면 null을 반환할 수 있음 — 호출자는 전혀 모름
        return _database.FindById(id); // null일 수 있음!
    }
}

// 호출자는 반환 타입을 신뢰하지만 NullReferenceException 발생
var user = service.GetUser(999);
Console.WriteLine(user.Name); // 폭발 — NullReferenceException
```

### 1.2 전통적인 Null 가드

컴파일러 지원 없이 개발자들은 수동 null 검사에 의존했으며, 이는 장황하고 오류가 발생하기 쉽습니다.

```csharp
// 방어적 프로그래밍 — 모든 곳에 null 검사
public void ProcessUser(User? user)
{
    if (user == null)
        throw new ArgumentNullException(nameof(user));

    if (user.Address == null)
        throw new InvalidOperationException("사용자에게 주소가 없습니다");

    if (user.Address.City == null)
        throw new InvalidOperationException("주소에 도시가 없습니다");

    Console.WriteLine(user.Address.City.ToUpper());
}
```

## 2. Nullable 컨텍스트 활성화

### 2.1 프로젝트 수준 (권장)

가장 일반적인 접근 방식은 `.csproj` 파일에서 전체 프로젝트에 NRT를 활성화하는 것입니다.

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>  <!-- 전체 프로젝트에 활성화 -->
  </PropertyGroup>
</Project>
```

`<Nullable>` 요소는 다음 값을 지원합니다:

| 값 | 주석 | 경고 |
|----|------|------|
| `enable` | 예 | 예 |
| `warnings` | 아니오 | 예 |
| `annotations` | 예 | 아니오 |
| `disable` | 아니오 | 아니오 |

### 2.2 파일 수준 지시문

`#nullable` 지시문을 사용하여 파일별로 프로젝트 설정을 재정의할 수 있습니다.

```csharp
#nullable enable   // 이 지점부터 주석과 경고 모두 활성화

public class MyClass
{
    public string Name { get; set; } = ""; // Non-nullable — 초기화 필수
    public string? NickName { get; set; }  // Nullable — null 가능
}

#nullable disable  // 이 지점부터 비활성화 (C# 8 이전 동작으로 복귀)

public class LegacyClass
{
    public string Name { get; set; } // 경고 없음, 이전 동작
}
```

### 2.3 범위 지시문

```csharp
#nullable enable

public void Method()
{
    string nonNull = "hello";

    #nullable disable warnings  // 경고 억제하되 주석은 유지
    string? maybeNull = null;
    Console.WriteLine(maybeNull.Length); // 경고 없음 (억제됨)
    #nullable restore warnings  // 외부 컨텍스트 설정으로 복원

    Console.WriteLine(maybeNull.Length); // 경고 복원됨
}
```

## 3. Nullable 주석

### 3.1 Non-Nullable vs Nullable 참조 타입

NRT가 활성화되면 참조 타입은 기본적으로 non-nullable입니다. `?`를 추가하면 nullable이 됩니다.

```csharp
#nullable enable

string name = "Alice";     // Non-nullable — null이 아님을 보장
string? nickname = null;   // Nullable — null 가능

// 컴파일러 경고:
name = null;               // 경고 CS8600: null 리터럴을 non-nullable 타입으로 변환
Console.WriteLine(nickname.Length); // 경고 CS8602: 가능한 null 참조의 역참조
```

### 3.2 메서드 시그니처

```csharp
public class UserRepository
{
    // 반환 타입이 말함: 항상 user를 반환 (절대 null이 아님)
    public User GetUserOrThrow(int id)
    {
        return _users.TryGetValue(id, out var user)
            ? user
            : throw new KeyNotFoundException($"사용자 {id}을(를) 찾을 수 없음");
    }

    // 반환 타입이 말함: null을 반환할 수 있음
    public User? FindUser(int id)
    {
        _users.TryGetValue(id, out var user);
        return user; // null일 수 있음 — 호출자가 경고받음
    }

    // 매개변수가 말함: null은 허용되지 않음
    public void UpdateUser(User user)
    {
        ArgumentNullException.ThrowIfNull(user);
        _users[user.Id] = user;
    }

    // 매개변수가 말함: null이 허용됨
    public void SetNickname(int userId, string? nickname)
    {
        var user = GetUserOrThrow(userId);
        user.Nickname = nickname; // OK — Nickname은 string?
    }
}
```

### 3.3 컬렉션과 제네릭

```csharp
// non-nullable 문자열 목록 — 어떤 요소도 null이 될 수 없음
List<string> names = new() { "Alice", "Bob" };
// names.Add(null); // 경고!

// nullable 문자열 목록 — 요소가 null이 될 수 있음
List<string?> optionalNames = new() { "Alice", null, "Charlie" };

// nullable 값을 가진 딕셔너리
Dictionary<string, string?> settings = new()
{
    ["theme"] = "dark",
    ["locale"] = null  // OK — 값이 string?
};

// non-null 제네릭 제약
public T EnsureNotNull<T>(T? value) where T : notnull
{
    return value ?? throw new ArgumentNullException(nameof(value));
}
```

## 4. Null 용서 연산자 (!)

null 용서(또는 null 억제) 연산자 `!`는 컴파일러에게 "이것이 null이 아님을 알고 있으니 신뢰하라"고 알립니다. 절제하여 사용하세요 — 안전성 없이 경고만 억제합니다.

### 4.1 적절한 사용 시점

```csharp
// 컴파일러가 볼 수 없는 수동 null 검사 후
public void Process(Dictionary<string, string> dict, string key)
{
    if (dict.ContainsKey(key))
    {
        // 컴파일러는 ContainsKey가 인덱서가 null을 반환하지 않음을 보장하는지 모름
        string value = dict[key]!; // OK — 방금 검사함
    }
}

// 프레임워크 초기화 패턴 (예: 의존성 주입)
public class MyController
{
    // 어떤 메서드 호출 전에 DI 프레임워크가 설정
    [Inject] public ILogger Logger { get; set; } = null!;
}

// 테스트 설정
[Fact]
public void Test_Something()
{
    var service = new MyService();
    // Setup()이 모든 속성을 초기화함을 앎
    service.Setup();
    var result = service.Data!.FirstItem; // 거짓 양성 억제
}
```

### 4.2 피해야 할 때

```csharp
// 나쁨: 정당한 경고를 억제
string? name = GetName();
Console.WriteLine(name!.Length); // 위험! name이 실제로 null일 수 있음

// 좋음: null 케이스를 처리
string? name = GetName();
if (name is not null)
{
    Console.WriteLine(name.Length); // 안전 — 컴파일러가 non-null임을 앎
}

// 좋음: 기본값 제공
Console.WriteLine((name ?? "unknown").Length);
```

## 5. Null 연산자

C#은 편리한 null 처리를 위한 여러 연산자를 제공합니다.

### 5.1 Null 조건부 연산자 (?.)

왼쪽이 null이면 예외를 던지는 대신 null로 단락합니다.

```csharp
string? name = null;

// ?. 없이
int length1 = name != null ? name.Length : 0;

// ?. 사용
int? length2 = name?.Length; // null (NullReferenceException이 아님)
int length3 = name?.Length ?? 0; // 0

// 체이닝
string? city = user?.Address?.City?.ToUpper();

// 인덱서와 함께
int? first = list?[0];

// 메서드 호출과 함께
string? upper = name?.ToUpper();
```

### 5.2 Null 병합 연산자 (??)

왼쪽 피연산자가 non-null이면 반환하고, 그렇지 않으면 오른쪽 피연산자를 반환합니다.

```csharp
string? input = null;

// 기본값 제공
string result = input ?? "default";
Console.WriteLine(result); // default

// 여러 대체 값 체이닝
string? primary = null;
string? secondary = null;
string? tertiary = "fallback";
string value = primary ?? secondary ?? tertiary ?? "최후의 수단";
Console.WriteLine(value); // fallback

// 메서드 호출과 함께
string config = GetConfigValue("key") ?? LoadDefault("key") ?? "하드코딩된 값";
```

### 5.3 Null 병합 할당 (??=)

현재 null인 경우에만 왼쪽 피연산자에 할당합니다.

```csharp
List<string>? names = null;

// 이전 방식
if (names == null)
    names = new List<string>();

// ??= 사용
names ??= new List<string>();

// 지연 초기화에 유용
private Dictionary<string, object>? _cache;
public Dictionary<string, object> Cache => _cache ??= new Dictionary<string, object>();

// 또 다른 일반적인 패턴 — 기본 매개변수 값
public void Configure(Action<Options>? configure = null)
{
    var options = new Options();
    configure ??= static _ => { }; // null이면 no-op
    configure(options);
}
```

### 5.4 Null 연산자 조합

```csharp
public class Config
{
    private Dictionary<string, string>? _overrides;
    private Dictionary<string, string>? _defaults;

    public string GetValue(string key)
    {
        return _overrides?.GetValueOrDefault(key)
            ?? _defaults?.GetValueOrDefault(key)
            ?? throw new KeyNotFoundException($"설정 키 '{key}'를 찾을 수 없음");
    }
}

// 이벤트 호출에서의 null 조건부
public event EventHandler<string>? MessageReceived;
protected void OnMessage(string msg) => MessageReceived?.Invoke(this, msg);
```

## 6. 컴파일러 흐름 분석

C# 컴파일러는 제어 흐름을 통해 nullable을 추적합니다. null 검사 후 타입을 자동으로 좁힙니다.

### 6.1 기본 흐름 분석

```csharp
public void Process(string? input)
{
    // 여기서 'input'은 string? — null일 수 있음
    // Console.WriteLine(input.Length); // 경고!

    if (input is null)
        return;

    // null 검사 후, 컴파일러는 'input'이 string (non-null)임을 앎
    Console.WriteLine(input.Length); // 경고 없음 — 흐름 분석이 타입을 좁힘
}
```

### 6.2 패턴 기반 좁히기

```csharp
public void HandleValue(object? value)
{
    // 타입 패턴이 좁히고 바인딩
    if (value is string text)
    {
        Console.WriteLine(text.Length); // text는 string이지 string?가 아님
    }

    // switch 표현식
    string result = value switch
    {
        string s => s.ToUpper(),    // s는 non-null string
        int n => n.ToString(),      // n은 non-null int (값 타입)
        null => "(null)",
        _ => value.ToString() ?? "" // value는 여기서 non-null (null은 위에서 처리)
    };
}
```

### 6.3 단언 메서드

일부 메서드는 값이 non-null임을 단언합니다. 컴파일러는 특성을 통해 이를 인식할 수 있습니다.

```csharp
// ArgumentNullException.ThrowIfNull (내장, 컴파일러가 인식)
public void SetName(string? name)
{
    ArgumentNullException.ThrowIfNull(name);
    // ThrowIfNull 후, 컴파일러는 'name'이 non-null임을 앎
    _name = name; // 경고 없음
}

// Debug.Assert — 기본적으로 흐름 분석에 영향 안 줌
public void Process(string? data)
{
    Debug.Assert(data != null);
    // 경고: 컴파일러는 null 분석에서 Debug.Assert를 신뢰하지 않음
    // [DoesNotReturnIf(false)] 특성을 사용하여 가르칠 수 있음
}
```

### 6.4 흐름 분석 한계

```csharp
public void Limitations(string? a, string? b)
{
    // 컴파일러는 단순 조건을 통해 추적함
    if (a != null && b != null)
    {
        Console.WriteLine(a.Length + b.Length); // OK
    }

    // 컴파일러는 메서드 호출을 통해 추적하지 못함
    bool isValid = a != null;
    if (isValid)
    {
        // Console.WriteLine(a.Length); // 경고! 컴파일러가 추적을 잃음
        Console.WriteLine(a!.Length);   // 억제를 위해 !가 필요
    }

    // 컴파일러는 ??와 ??=를 통해 추적
    a ??= "default";
    Console.WriteLine(a.Length); // OK — ??= 후 a는 non-null 보장
}
```

## 7. Nullable 특성

`System.Diagnostics.CodeAnalysis` 네임스페이스는 `?`만으로는 표현할 수 없는 null 계약에 대한 추가 정보를 컴파일러에 제공하는 특성을 제공합니다.

### 7.1 사전 조건 특성

```csharp
using System.Diagnostics.CodeAnalysis;

public class Validator
{
    // [NotNull] — 메서드가 정상 반환하면 매개변수가 non-null
    public static void EnsureNotNull([NotNull] string? value)
    {
        if (value is null)
            throw new ArgumentNullException(nameof(value));
        // 여기에 도달하면 value는 non-null
    }

    // [DoesNotReturnIf] — 매개변수가 주어진 bool과 같으면 메서드가 반환하지 않음
    public static void Assert([DoesNotReturnIf(false)] bool condition, string? message = null)
    {
        if (!condition)
            throw new InvalidOperationException(message ?? "단언 실패");
    }
}

// 사용법
public void Process(string? input)
{
    Validator.EnsureNotNull(input);
    Console.WriteLine(input.Length); // 경고 없음 — 컴파일러가 input이 non-null임을 앎

    string? name = GetName();
    Validator.Assert(name != null, "이름은 null이 아니어야 합니다");
    Console.WriteLine(name.Length); // 경고 없음
}
```

### 7.2 사후 조건 특성

```csharp
public class Parser
{
    // [NotNullWhen(true)] — 메서드가 true를 반환하면 매개변수가 non-null
    public static bool TryParse(string? input, [NotNullWhen(true)] out string? result)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            result = null;
            return false;
        }
        result = input.Trim();
        return true;
    }

    // [MaybeNullWhen(false)] — 메서드가 false를 반환하면 출력이 null일 수 있음
    public static bool TryGetValue<T>(
        Dictionary<string, T> dict,
        string key,
        [MaybeNullWhen(false)] out T value)
    {
        return dict.TryGetValue(key, out value);
    }
}

// 사용법
if (Parser.TryParse(input, out var parsed))
{
    Console.WriteLine(parsed.Length); // 경고 없음 — NotNullWhen(true)가 non-null을 보장
}
```

### 7.3 멤버 특성

```csharp
public class Cache<TKey, TValue> where TKey : notnull
{
    private readonly Dictionary<TKey, TValue> _dict = new();

    // [MaybeNull] — T가 non-nullable이어도 반환 값이 null일 수 있음
    [return: MaybeNull]
    public TValue GetOrDefault(TKey key)
    {
        _dict.TryGetValue(key, out var value);
        return value; // default(TValue)일 수 있으며, 참조 타입의 경우 null
    }

    // [DisallowNull] — 타입이 허용해도 매개변수가 null이 아니어야 함
    public void Set(TKey key, [DisallowNull] TValue? value)
    {
        // value가 TValue?로 타이핑될 수 있지만 호출자가 null을 전달하면 경고
        _dict[key] = value;
    }

    // [MemberNotNull] — 메서드 반환 후 멤버가 non-null임을 보장
    private string? _connectionString;

    [MemberNotNull(nameof(_connectionString))]
    public void Initialize(string connectionString)
    {
        _connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
    }

    // [MemberNotNullWhen] — 메서드가 특정 bool을 반환하면 멤버가 non-null
    [MemberNotNullWhen(true, nameof(_connectionString))]
    public bool IsInitialized => _connectionString is not null;
}
```

### 7.4 Nullable 특성 요약

| 특성 | 적용 대상 | 의미 |
|------|----------|------|
| `[NotNull]` | 매개변수, out | 메서드 반환 후 non-null |
| `[MaybeNull]` | 반환값, out, 속성 | T가 non-nullable이어도 null일 수 있음 |
| `[AllowNull]` | 매개변수, 속성 | T가 non-nullable이어도 null을 허용 |
| `[DisallowNull]` | 매개변수, 속성 | T가 nullable이어도 null을 거부 |
| `[NotNullWhen(bool)]` | 매개변수, out | 메서드가 주어진 bool을 반환하면 non-null |
| `[MaybeNullWhen(bool)]` | out | 메서드가 주어진 bool을 반환하면 null일 수 있음 |
| `[NotNullIfNotNull(param)]` | 반환값 | 지정된 매개변수가 non-null이면 non-null |
| `[DoesNotReturn]` | 메서드 | 메서드가 절대 반환하지 않음 (항상 예외) |
| `[DoesNotReturnIf(bool)]` | 매개변수 | 매개변수가 bool과 같으면 메서드가 반환하지 않음 |
| `[MemberNotNull(member)]` | 메서드 | 메서드 반환 후 지정된 멤버가 non-null |
| `[MemberNotNullWhen(bool, member)]` | 메서드 | 메서드가 bool을 반환하면 지정된 멤버가 non-null |

## 8. 마이그레이션 전략

### 8.1 점진적 도입

```xml
<!-- 1단계: 경고만으로 시작 (코드 변경 불필요) -->
<Nullable>warnings</Nullable>

<!-- 2단계: 준비되면 완전 활성화 -->
<Nullable>enable</Nullable>
```

### 8.2 파일별 마이그레이션

```csharp
// 새 파일: 완전히 nullable
#nullable enable

// 레거시 파일: 임시로 제외
#nullable disable

// 마이그레이션 중: 활성화하고 파일별로 경고 수정
#nullable enable
// 이 파일의 모든 경고를 수정한 후 다음으로 이동
```

### 8.3 일반적인 마이그레이션 패턴

```csharp
// 패턴 1: 늦게 초기화되는 속성
// 이전:
public string Name { get; set; } // 경고: non-nullable이 초기화되지 않음

// 옵션 A: 기본값으로 초기화
public string Name { get; set; } = "";

// 옵션 B: 진짜 null일 수 있으면 nullable로
public string? Name { get; set; }

// 옵션 C: 'required' 사용 (C# 11) — 초기화 시 설정 필수
public required string Name { get; set; }

// 옵션 D: DI/프레임워크가 초기화하는 속성에 null!
public string Name { get; set; } = null!;

// 패턴 2: Dictionary TryGetValue
string? value;
if (dict.TryGetValue(key, out value))
{
    Console.WriteLine(value.Length); // 안전 — TryGetValue에 [NotNullWhen(true)] 있음
}

// 패턴 3: LINQ FirstOrDefault
var item = list.FirstOrDefault(x => x.Id == targetId);
if (item is not null)
{
    Process(item);
}
// 존재함을 확신하면 First() 사용
var item = list.First(x => x.Id == targetId);
```

## 9. Null 안전성 모범 사례

### 9.1 설계 원칙

```csharp
// 1. Non-nullable 타입 선호 — null을 예외로, 규범이 아닌 것으로
public class User
{
    public required string Name { get; init; }      // 항상 이름이 있음
    public required string Email { get; init; }     // 항상 이메일이 있음
    public string? Bio { get; set; }                // 자기소개는 선택
    public string? AvatarUrl { get; set; }          // 아바타는 선택
}

// 2. null 대신 Null 객체 패턴 사용
public interface ILogger { void Log(string message); }
public class NullLogger : ILogger { public void Log(string message) { } }

ILogger logger = GetLogger() ?? new NullLogger(); // 절대 null이 아님

// 3. null 대신 빈 컬렉션 반환
public IReadOnlyList<Order> GetOrders(int customerId)
{
    // 좋음: 빈 목록 반환
    return _db.Orders.Where(o => o.CustomerId == customerId).ToList();
    // 나쁨: 주문이 없으면 null 반환
}

// 4. 실패할 수 있는 연산에 결과 타입 사용
public record Result<T>(T? Value, string? Error)
{
    public bool IsSuccess => Error is null;
    public static Result<T> Ok(T value) => new(value, null);
    public static Result<T> Fail(string error) => new(default, error);
}
```

### 9.2 가드 절

```csharp
public class OrderService
{
    private readonly IRepository _repo;
    private readonly ILogger _logger;

    // 생성자 가드 절
    public OrderService(IRepository repo, ILogger logger)
    {
        _repo = repo ?? throw new ArgumentNullException(nameof(repo));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    // .NET 6+ — 더 깔끔한 구문
    public OrderService(IRepository repo, ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(repo);
        ArgumentNullException.ThrowIfNull(logger);
        _repo = repo;
        _logger = logger;
    }
}
```

## 10. 실전 예제: 클래스를 Null 안전하게 리팩토링

### 10.1 이전 — Null 안전하지 않은 코드

```csharp
// nullable 컨텍스트 없음 — 잠재적 버그들
public class CustomerProfile
{
    public string FirstName;
    public string LastName;
    public string Email;
    public string Phone;     // 선택
    public Address Address;  // 선택

    public string GetDisplayName()
    {
        return FirstName + " " + LastName; // 둘 중 하나가 null이면 NPE
    }

    public string GetCity()
    {
        return Address.City; // Address가 null이면 NPE
    }
}

public class Address
{
    public string Street;
    public string City;
    public string State;
    public string ZipCode;
}
```

### 10.2 이후 — Null 안전한 코드

```csharp
#nullable enable

public class CustomerProfile
{
    public required string FirstName { get; init; }
    public required string LastName { get; init; }
    public required string Email { get; init; }
    public string? Phone { get; set; }        // 명시적으로 선택
    public Address? Address { get; set; }      // 명시적으로 선택

    public string DisplayName => $"{FirstName} {LastName}"; // 안전 — 둘 다 필수

    public string GetCity() =>
        Address?.City ?? "알 수 없음"; // 안전 — null Address와 null City를 처리

    public bool HasCompleteProfile =>
        Phone is not null &&
        Address is { Street: not null, City: not null, State: not null, ZipCode: not null };
}

public class Address
{
    public required string Street { get; init; }
    public required string City { get; init; }
    public required string State { get; init; }
    public required string ZipCode { get; init; }
}

// 사용법 — 컴파일러가 초기화를 강제
var customer = new CustomerProfile
{
    FirstName = "Alice",
    LastName = "Smith",
    Email = "alice@example.com",
    // Phone과 Address는 선택 — 경고 없음
};

Console.WriteLine(customer.DisplayName);          // Alice Smith
Console.WriteLine(customer.GetCity());            // 알 수 없음
Console.WriteLine(customer.HasCompleteProfile);   // False

customer.Address = new Address
{
    Street = "123 Main St",
    City = "Seattle",
    State = "WA",
    ZipCode = "98101"
};

Console.WriteLine(customer.GetCity());            // Seattle
Console.WriteLine(customer.HasCompleteProfile);   // False (Phone이 여전히 null)
```

## 11. 연습 문제

1. **Nullable 마이그레이션**: 다음 NRT 이전 클래스를 완전히 null 안전하게 마이그레이션하세요. `?` 주석, `required` 키워드, 가드 절, null 조건부 연산자를 적절히 추가하세요. 어떤 필드가 nullable이어야 하고 어떤 것이 아니어야 하는지 식별하세요.
   ```csharp
   public class Order {
       public int Id;
       public Customer Customer;
       public List<OrderItem> Items;
       public string CouponCode;
       public string ShippingNotes;
       public decimal GetTotal() => Items.Sum(i => i.Price * i.Quantity);
   }
   ```

2. **사용자 정의 TryParse**: `Start`와 `End` 속성을 가진 `DateRange` 구조체를 구현하세요. `"2024-01-01..2024-12-31"` 같은 문자열을 파싱하는 `TryParse(string? input, out DateRange? result)` 메서드를 작성하세요. `[NotNullWhen(true)]`를 올바르게 사용하세요. null 입력, 빈 문자열, 유효하지 않은 형식, Start > End인 범위를 처리하세요.

3. **Null 안전한 빌더 패턴**: 플루언트 API를 가진 `ConnectionStringBuilder`를 구현하세요. 일부 속성은 필수(Server, Database)이고, 다른 것은 선택(Port, Username, Password)입니다. `Build()` 메서드는 `string`(non-nullable)을 반환하고 필수 속성이 없으면 예외를 던져야 합니다. 적절한 곳에 `[MemberNotNull]`을 사용하세요.

4. **흐름 분석 탐색기**: 컴파일러의 흐름 분석이 추적할 수 있는 것과 없는 것을 보여주는 일련의 메서드를 작성하세요. 최소 5가지 예제를 포함하세요: (a) `if` null 검사, (b) `is not null` 패턴, (c) `??=` 할당, (d) 추적을 잃는 메서드 호출, (e) `[NotNullWhen]`으로 추적 복원.

5. **제네릭 Null 안전 컬렉션**: `List<T?>`를 래핑하되 절대 null을 반환하지 않는 메서드를 제공하는 `NullSafeList<T>`를 구현하세요: `GetOrDefault(int index, T defaultValue)`, `FirstOrDefault(T defaultValue)`, `FindOrDefault(Predicate<T> predicate, T defaultValue)`. 모두 `T?`가 아닌 `T`를 반환합니다. 모든 매개변수와 반환값에 올바른 nullable 주석을 다세요.
