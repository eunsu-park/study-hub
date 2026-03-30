# 레코드와 불변성

**이전**: [Nullable 참조 타입](./05_Nullable_Reference_Types.md) | **다음**: [Async와 Await](./07_Async_Await.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 소프트웨어 설계에서 불변성(immutability)의 이점 설명하기
2. 위치 구문으로 record class와 record struct 선언하기
3. with 표현식을 사용한 비파괴적 변경 수행하기
4. 레코드의 값 기반 동등성(equality) 의미 이해하기
5. 레코드 계층 구조에 상속 적용하기
6. 레코드를 구성 요소 변수로 분해하기
7. init 전용 속성과 읽기 전용 컬렉션을 사용한 불변 설계 활용하기
8. 불변 이벤트 레코드를 사용한 이벤트 소싱 시스템 구축하기

---

불변성(immutability) — 생성 후 상태를 변경할 수 없는 객체를 만드는 관행 — 은 신뢰할 수 있는 소프트웨어의 초석입니다. 불변 객체는 본질적으로 스레드 안전하고, 추론하기 쉬우며, 의도하지 않은 변경과 관련된 전체 범주의 버그에서 자유롭습니다. C# 9에서 도입되고 C# 10에서 확장된 C# 레코드(record)는 간결한 구문, 값 기반 동등성, `with` 표현식을 통한 비파괴적 변경과 함께 불변 데이터 타입에 대한 일급 언어 지원을 제공합니다.

## 1. 불변성 개념과 이점

### 1.1 불변성이 중요한 이유

```csharp
// 문제: 가변 객체는 미묘한 버그를 유발
public class MutablePoint
{
    public double X { get; set; }
    public double Y { get; set; }
}

var point = new MutablePoint { X = 3, Y = 4 };
var dict = new Dictionary<MutablePoint, string> { [point] = "origin" };

// 딕셔너리 키로 사용한 후 변경 — 해시가 변경되어 키가 "유실"됨
point.X = 99;
Console.WriteLine(dict.ContainsKey(point)); // False! 항목이 고아가 됨.

// 해결책: 불변 객체는 이 문제가 없음
public record ImmutablePoint(double X, double Y);

var p = new ImmutablePoint(3, 4);
var dict2 = new Dictionary<ImmutablePoint, string> { [p] = "origin" };
// p.X = 99; // 컴파일 오류 — X는 init 전용
Console.WriteLine(dict2.ContainsKey(p)); // True — 항상
```

### 1.2 이점 요약

| 이점 | 설명 |
|------|------|
| 스레드 안전성 | 동기화 불필요 — 상태가 변하지 않음 |
| 예측 가능성 | 원격 작용 없음; 받은 객체가 그대로의 객체 |
| 안전한 공유 | 방어적 복사 없이 참조로 전달 |
| 해시 안정성 | 딕셔너리 키와 집합 요소로 안전 |
| 실행 취소/이력 | 이전 상태가 자연스럽게 보존 |
| 테스팅 | 가변 상태의 설정/해제 불필요 |

## 2. Record Class 선언

### 2.1 위치 레코드 구문

레코드를 선언하는 가장 간결한 방법입니다. 컴파일러가 생성자, 속성, 분해자, 동등성 멤버, `ToString`을 생성합니다.

```csharp
// 위치 레코드 — 컴파일러가 모든 것을 생성
public record Person(string FirstName, string LastName, int Age);

// 컴파일러가 생성하는 것 (개념적):
// - 생성자: Person(string FirstName, string LastName, int Age)
// - Init 전용 속성: string FirstName { get; init; }
// - Deconstruct 메서드: void Deconstruct(out string, out string, out int)
// - 값 기반 Equals, GetHashCode, ==, !=
// - ToString: Person { FirstName = ..., LastName = ..., Age = ... }
// - with 표현식 지원 (복제 + 수정)

var alice = new Person("Alice", "Smith", 30);
Console.WriteLine(alice);
// Person { FirstName = Alice, LastName = Smith, Age = 30 }
```

### 2.2 명목적 (비위치) 레코드 구문

명시적 속성 정의로 레코드를 선언할 수도 있으며, 이는 더 많은 제어를 제공합니다.

```csharp
public record Product
{
    public required string Name { get; init; }
    public required decimal Price { get; init; }
    public string? Description { get; init; }
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

var widget = new Product
{
    Name = "Widget",
    Price = 9.99m,
    Description = "훌륭한 위젯"
};

Console.WriteLine(widget);
// Product { Name = Widget, Price = 9.99, Description = 훌륭한 위젯, CreatedAt = ... }
```

### 2.3 혼합: 위치 매개변수와 추가 속성

```csharp
public record Employee(string Name, string Department)
{
    // 위치 매개변수 외의 추가 속성
    public int YearsOfService { get; init; }
    public List<string> Skills { get; init; } = new();

    // 계산 속성
    public bool IsSenior => YearsOfService >= 5;
}

var emp = new Employee("Alice", "Engineering")
{
    YearsOfService = 7,
    Skills = { "C#", "SQL", "Docker" }
};

Console.WriteLine(emp);
Console.WriteLine($"시니어: {emp.IsSenior}"); // True
```

### 2.4 유효성 검사가 있는 레코드

```csharp
public record Email
{
    public string Address { get; }

    public Email(string address)
    {
        if (string.IsNullOrWhiteSpace(address))
            throw new ArgumentException("이메일은 비어있을 수 없습니다", nameof(address));
        if (!address.Contains('@'))
            throw new ArgumentException("유효하지 않은 이메일 형식", nameof(address));

        Address = address.Trim().ToLowerInvariant();
    }

    // ToString을 재정의하여 주소만 표시
    public override string ToString() => Address;
}

var email = new Email("  Alice@Example.COM  ");
Console.WriteLine(email); // alice@example.com
```

## 3. With 표현식

with 표현식은 지정된 속성이 변경된 레코드의 복사본을 생성합니다. 이것이 "비파괴적 변경"입니다 — 원본은 변경되지 않습니다.

### 3.1 기본 사용법

```csharp
public record Point(double X, double Y);

var p1 = new Point(3, 4);
var p2 = p1 with { X = 10 };     // new Point(10, 4)
var p3 = p1 with { Y = -1 };     // new Point(3, -1)
var p4 = p1 with { X = 0, Y = 0 }; // new Point(0, 0)

Console.WriteLine(p1); // Point { X = 3, Y = 4 } — 변경 없음
Console.WriteLine(p2); // Point { X = 10, Y = 4 }
Console.WriteLine(p3); // Point { X = 3, Y = -1 }
```

### 3.2 실전에서의 With 표현식

```csharp
public record Configuration(
    string Host,
    int Port,
    bool UseSsl,
    TimeSpan Timeout,
    int MaxRetries);

var defaultConfig = new Configuration(
    Host: "localhost",
    Port: 8080,
    UseSsl: false,
    Timeout: TimeSpan.FromSeconds(30),
    MaxRetries: 3);

// 프로덕션 오버라이드 — 다른 것만 변경
var prodConfig = defaultConfig with
{
    Host = "api.example.com",
    Port = 443,
    UseSsl = true,
    Timeout = TimeSpan.FromSeconds(10)
};

// 스테이징 — 프로덕션에서 시작하되 다른 호스트
var stagingConfig = prodConfig with { Host = "staging.example.com" };

Console.WriteLine(defaultConfig);
Console.WriteLine(prodConfig);
Console.WriteLine(stagingConfig);
```

### 3.3 With 표현식은 얕은 복사를 생성

```csharp
public record Team(string Name, List<string> Members);

var team1 = new Team("Alpha", new List<string> { "Alice", "Bob" });
var team2 = team1 with { Name = "Beta" };

// 주의: Members 리스트가 공유됨 (얕은 복사)
team2.Members.Add("Charlie");
Console.WriteLine(string.Join(", ", team1.Members)); // Alice, Bob, Charlie (!)

// 이를 피하려면 불변 컬렉션 사용
public record SafeTeam(string Name, ImmutableList<string> Members);

var safe1 = new SafeTeam("Alpha", ImmutableList.Create("Alice", "Bob"));
var safe2 = safe1 with { Name = "Beta", Members = safe1.Members.Add("Charlie") };
Console.WriteLine(string.Join(", ", safe1.Members)); // Alice, Bob (변경 없음)
Console.WriteLine(string.Join(", ", safe2.Members)); // Alice, Bob, Charlie
```

## 4. 값 기반 동등성

레코드는 기본적으로 값 기반 동등성을 사용합니다 — 참조 아이덴티티와 관계없이 모든 속성이 같으면 두 레코드는 같습니다.

### 4.1 동등성 비교

```csharp
public record Coordinate(double Latitude, double Longitude);

var a = new Coordinate(47.6062, -122.3321); // 시애틀
var b = new Coordinate(47.6062, -122.3321); // 같은 좌표

Console.WriteLine(a == b);           // True (값 동등성)
Console.WriteLine(a.Equals(b));      // True
Console.WriteLine(ReferenceEquals(a, b)); // False (다른 객체)

// 클래스 동작과 비교
public class CoordinateClass
{
    public double Latitude { get; init; }
    public double Longitude { get; init; }
}

var c = new CoordinateClass { Latitude = 47.6062, Longitude = -122.3321 };
var d = new CoordinateClass { Latitude = 47.6062, Longitude = -122.3321 };

Console.WriteLine(c == d);           // False (클래스의 참조 동등성)
Console.WriteLine(c.Equals(d));      // False
```

### 4.2 딕셔너리 키와 집합에서의 레코드

```csharp
public record CacheKey(string Endpoint, string Method, string? QueryString);

var cache = new Dictionary<CacheKey, string>();

cache[new CacheKey("/api/users", "GET", null)] = "캐시된 사용자 목록";
cache[new CacheKey("/api/users", "GET", "?page=2")] = "캐시된 2페이지";

// 구조적으로 동일한 키로 조회
var key = new CacheKey("/api/users", "GET", null);
Console.WriteLine(cache[key]); // 캐시된 사용자 목록

// HashSet에서도 동작
var visited = new HashSet<CacheKey>();
visited.Add(new CacheKey("/api/users", "GET", null));
Console.WriteLine(visited.Contains(new CacheKey("/api/users", "GET", null))); // True
```

### 4.3 사용자 정의 동등성

특정 속성에 대한 동등성을 재정의할 수 있습니다 (예: 대소문자 무시 문자열 비교):

```csharp
public record PersonName(string First, string Last)
{
    public virtual bool Equals(PersonName? other)
    {
        if (other is null) return false;
        return string.Equals(First, other.First, StringComparison.OrdinalIgnoreCase)
            && string.Equals(Last, other.Last, StringComparison.OrdinalIgnoreCase);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(
            First.ToUpperInvariant().GetHashCode(),
            Last.ToUpperInvariant().GetHashCode());
    }
}

var a = new PersonName("Alice", "Smith");
var b = new PersonName("ALICE", "SMITH");
Console.WriteLine(a == b); // True (대소문자 무시)
```

## 5. Record Struct vs Record Class

C# 10에서 `record struct` — 레코드 의미를 가진 값 타입 — 이 도입되었습니다.

### 5.1 선언

```csharp
// record class (참조 타입) — 'record'의 기본
public record class PersonRecord(string Name, int Age);
// 동일한 약식:
public record PersonRecord2(string Name, int Age);

// record struct (값 타입) — C# 10
public record struct PointRecord(double X, double Y);

// readonly record struct — 진정한 불변 값 타입
public readonly record struct ImmutablePoint(double X, double Y);
```

### 5.2 주요 차이점

```csharp
// record class: init 전용 속성 (기본적으로 불변)
public record ClassRecord(string Name);
var cr = new ClassRecord("Alice");
// cr.Name = "Bob"; // 오류 — init 전용

// record struct: 기본적으로 가변 속성
public record struct MutableStructRecord(string Name);
var msr = new MutableStructRecord("Alice");
msr.Name = "Bob"; // OK — 가변!

// readonly record struct: init 전용 속성 (불변)
public readonly record struct ReadonlyStructRecord(string Name);
var rsr = new ReadonlyStructRecord("Alice");
// rsr.Name = "Bob"; // 오류 — readonly
```

### 5.3 어떤 것을 사용할지

| 기능 | `record class` | `record struct` | `readonly record struct` |
|------|----------------|-----------------|--------------------------|
| 할당 위치 | 힙 | 스택 (보통) | 스택 (보통) |
| 기본 가변성 | 불변 (init) | 가변 | 불변 (init) |
| 상속 지원 | 예 | 아니오 | 아니오 |
| `with` 표현식 | 예 | 예 | 예 |
| Null | null 가능 | null 불가 | null 불가 |
| 적합한 용도 | 도메인 엔티티, DTO | 작은 데이터 (2-3 필드) | 수학 타입, 좌표 |

```csharp
// 좋은 record class 후보: 도메인 객체, DTO, 명령
public record CreateUserCommand(string Name, string Email, string Password);
public record UserDto(int Id, string Name, string Email, DateTime CreatedAt);

// 좋은 readonly record struct 후보: 작은 값 객체
public readonly record struct Money(decimal Amount, string Currency);
public readonly record struct DateRange(DateOnly Start, DateOnly End);
public readonly record struct Color(byte R, byte G, byte B, byte A = 255);
```

## 6. 레코드 상속

record class는 상속을 지원합니다. record struct는 지원하지 않습니다.

### 6.1 기본 상속

```csharp
public abstract record Shape(string Color);
public record Circle(string Color, double Radius) : Shape(Color);
public record Rectangle(string Color, double Width, double Height) : Shape(Color);
public record Triangle(string Color, double Base, double Height) : Shape(Color);

Shape shape = new Circle("Red", 5.0);
Console.WriteLine(shape);
// Circle { Color = Red, Radius = 5 }
```

### 6.2 상속과 동등성

레코드는 상속 계층 구조에서 동등성을 올바르게 처리합니다 — 같은 기본 속성을 공유하더라도 `Circle`은 `Rectangle`과 절대 같지 않습니다.

```csharp
Shape s1 = new Circle("Red", 5.0);
Shape s2 = new Circle("Red", 5.0);
Shape s3 = new Rectangle("Red", 5.0, 5.0);

Console.WriteLine(s1 == s2); // True (같은 타입과 값)
Console.WriteLine(s1 == s3); // False (다른 타입)

// EqualityContract 속성이 타입 안전한 비교를 보장
```

### 6.3 With 표현식과 상속

```csharp
Circle c1 = new Circle("Red", 5.0);
Circle c2 = c1 with { Radius = 10.0 };       // Circle { Color = Red, Radius = 10 }
Circle c3 = c1 with { Color = "Blue" };       // Circle { Color = Blue, Radius = 5 }

// with 표현식은 실제 타입을 보존
Shape s = c1;
Shape s2 = s with { Color = "Green" }; // 여전히 Circle!
Console.WriteLine(s2.GetType().Name);  // Circle
Console.WriteLine(s2);                 // Circle { Color = Green, Radius = 5 }
```

### 6.4 봉인 레코드

```csharp
// 추가 상속 방지
public sealed record FinalProduct(string Name, decimal Price);
// public record SpecialProduct(...) : FinalProduct(...); // 오류 — sealed
```

## 7. 분해

위치 레코드는 자동으로 `Deconstruct` 메서드를 생성하여 튜플 유사 분해를 가능하게 합니다.

### 7.1 기본 분해

```csharp
public record Person(string FirstName, string LastName, int Age);

var person = new Person("Alice", "Smith", 30);

// 변수로 분해
var (first, last, age) = person;
Console.WriteLine($"{first} {last}, 나이 {age}"); // Alice Smith, 나이 30

// 디스카드를 사용한 부분 분해
var (name, _, _) = person;
Console.WriteLine(name); // Alice
```

### 7.2 패턴 매칭에서의 분해

```csharp
public record Order(string Product, int Quantity, decimal UnitPrice);

var orders = new[]
{
    new Order("Widget", 5, 9.99m),
    new Order("Gadget", 100, 2.50m),
    new Order("Doohickey", 1, 499.99m),
};

foreach (var order in orders)
{
    var message = order switch
    {
        (_, >= 50, _) => $"대량 주문: {order.Product}",
        (_, _, >= 100) => $"프리미엄 상품: {order.Product}",
        var (product, qty, price) when qty * price > 100
            => $"고가치: {product} (${qty * price:F2})",
        _ => $"일반: {order.Product}"
    };
    Console.WriteLine(message);
}
// 대량 주문: Gadget
// 프리미엄 상품: Doohickey
// 일반: Widget
```

### 7.3 비레코드에 대한 사용자 정의 Deconstruct

```csharp
public class Range
{
    public int Start { get; }
    public int End { get; }
    public int Length => End - Start;

    public Range(int start, int end) => (Start, End) = (start, end);

    public void Deconstruct(out int start, out int end)
    {
        start = Start;
        end = End;
    }

    public void Deconstruct(out int start, out int end, out int length)
    {
        start = Start;
        end = End;
        length = Length;
    }
}

var range = new Range(5, 15);
var (s, e) = range;
Console.WriteLine($"{s}부터 {e}까지"); // 5부터 15까지

var (start, end, len) = range;
Console.WriteLine($"{start}부터 {end}까지, 길이 {len}"); // 5부터 15까지, 길이 10
```

## 8. Init 전용 속성과 읽기 전용 컬렉션

### 8.1 Init 전용 속성

```csharp
public class AppSettings
{
    public required string ConnectionString { get; init; }
    public required string ApiKey { get; init; }
    public int MaxRetries { get; init; } = 3;
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);
}

var settings = new AppSettings
{
    ConnectionString = "Server=localhost;Database=mydb",
    ApiKey = "secret-key",
    MaxRetries = 5
};

// settings.ConnectionString = "other"; // 오류 — init 전용
Console.WriteLine(settings.Timeout); // 00:00:30 (기본값)
```

### 8.2 불변 컬렉션

```csharp
using System.Collections.Immutable;

// ImmutableList<T>
var list1 = ImmutableList.Create(1, 2, 3);
var list2 = list1.Add(4);          // 새 리스트 [1,2,3,4] 반환
var list3 = list1.Remove(2);       // 새 리스트 [1,3] 반환
Console.WriteLine(list1.Count);     // 3 (변경 없음)
Console.WriteLine(list2.Count);     // 4

// ImmutableDictionary<K,V>
var dict1 = ImmutableDictionary<string, int>.Empty
    .Add("a", 1)
    .Add("b", 2);
var dict2 = dict1.SetItem("a", 10); // "a"=10인 새 딕셔너리 반환
Console.WriteLine(dict1["a"]);       // 1 (변경 없음)
Console.WriteLine(dict2["a"]);       // 10

// ImmutableArray<T> — ImmutableList보다 더 나은 캐시 지역성
var arr1 = ImmutableArray.Create(10, 20, 30);
var arr2 = arr1.Add(40);
Console.WriteLine(arr1.Length); // 3
Console.WriteLine(arr2.Length); // 4

// 효율적인 대량 구성을 위한 빌더 패턴
var builder = ImmutableList.CreateBuilder<string>();
builder.Add("Alice");
builder.Add("Bob");
builder.Add("Charlie");
ImmutableList<string> immutable = builder.ToImmutable();
```

### 8.3 FrozenDictionary와 FrozenSet (.NET 8+)

```csharp
using System.Collections.Frozen;

// 읽기 집중적이고 한 번만 쓰는 시나리오에 최적화
var data = new Dictionary<string, int>
{
    ["red"] = 0xFF0000,
    ["green"] = 0x00FF00,
    ["blue"] = 0x0000FF,
};

FrozenDictionary<string, int> frozen = data.ToFrozenDictionary();
Console.WriteLine(frozen["red"]); // 16711680

// FrozenSet
FrozenSet<string> validCommands = new[] { "start", "stop", "restart" }.ToFrozenSet();
Console.WriteLine(validCommands.Contains("start")); // True
```

## 9. 실전 예제: 불변 이벤트를 사용한 이벤트 소싱

이벤트 소싱(event sourcing)은 상태 변경을 불변 이벤트의 시퀀스로 저장합니다. 레코드는 이 패턴에 완벽한 데이터 구조입니다.

### 9.1 이벤트 정의

```csharp
public abstract record DomainEvent(DateTime OccurredAt);

public record AccountCreated(
    string AccountId,
    string OwnerName,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record MoneyDeposited(
    string AccountId,
    decimal Amount,
    string Description,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record MoneyWithdrawn(
    string AccountId,
    decimal Amount,
    string Description,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record AccountClosed(
    string AccountId,
    string Reason,
    DateTime OccurredAt) : DomainEvent(OccurredAt);
```

### 9.2 불변 상태

```csharp
public record AccountState(
    string AccountId,
    string OwnerName,
    decimal Balance,
    bool IsActive,
    ImmutableList<DomainEvent> History)
{
    public static AccountState Initial => new(
        "", "", 0m, false, ImmutableList<DomainEvent>.Empty);
}
```

### 9.3 이벤트 적용 (Fold)

```csharp
public static class AccountProjection
{
    public static AccountState Apply(AccountState state, DomainEvent @event) =>
        @event switch
        {
            AccountCreated e => state with
            {
                AccountId = e.AccountId,
                OwnerName = e.OwnerName,
                Balance = 0m,
                IsActive = true,
                History = state.History.Add(e)
            },

            MoneyDeposited e => state with
            {
                Balance = state.Balance + e.Amount,
                History = state.History.Add(e)
            },

            MoneyWithdrawn e when e.Amount <= state.Balance => state with
            {
                Balance = state.Balance - e.Amount,
                History = state.History.Add(e)
            },

            MoneyWithdrawn e => throw new InvalidOperationException(
                $"잔액 부족: 잔고={state.Balance}, 출금={e.Amount}"),

            AccountClosed e => state with
            {
                IsActive = false,
                History = state.History.Add(e)
            },

            _ => throw new ArgumentException($"알 수 없는 이벤트: {@event.GetType().Name}")
        };

    // 이벤트 스트림에서 상태 재구성
    public static AccountState Rebuild(IEnumerable<DomainEvent> events) =>
        events.Aggregate(AccountState.Initial, Apply);
}
```

### 9.4 이벤트 소싱 계좌 사용

```csharp
var now = DateTime.UtcNow;

var events = new DomainEvent[]
{
    new AccountCreated("ACC-001", "Alice Smith", now),
    new MoneyDeposited("ACC-001", 1000m, "초기 입금", now.AddMinutes(1)),
    new MoneyDeposited("ACC-001", 500m, "급여", now.AddDays(1)),
    new MoneyWithdrawn("ACC-001", 200m, "식료품", now.AddDays(2)),
    new MoneyDeposited("ACC-001", 300m, "프리랜서 작업", now.AddDays(3)),
};

var currentState = AccountProjection.Rebuild(events);

Console.WriteLine($"계좌: {currentState.AccountId}");
Console.WriteLine($"소유자: {currentState.OwnerName}");
Console.WriteLine($"잔고: {currentState.Balance:C}");
Console.WriteLine($"활성: {currentState.IsActive}");
Console.WriteLine($"이벤트: {currentState.History.Count}");

// 출력:
// 계좌: ACC-001
// 소유자: Alice Smith
// 잔고: $1,600.00
// 활성: True
// 이벤트: 5

// 시간 여행 — 처음 3개 이벤트만 재생하여 과거 상태 확인
var pastState = AccountProjection.Rebuild(events.Take(3));
Console.WriteLine($"3개 이벤트 후 잔고: {pastState.Balance:C}");
// 3개 이벤트 후 잔고: $1,500.00

// 감사 추적
Console.WriteLine("\n거래 이력:");
foreach (var evt in currentState.History)
{
    string desc = evt switch
    {
        AccountCreated e => $"{e.OwnerName}을 위한 계좌 생성",
        MoneyDeposited e => $"+{e.Amount:C} ({e.Description})",
        MoneyWithdrawn e => $"-{e.Amount:C} ({e.Description})",
        AccountClosed e => $"계좌 폐쇄: {e.Reason}",
        _ => "알 수 없는 이벤트"
    };
    Console.WriteLine($"  [{evt.OccurredAt:g}] {desc}");
}
```

## 10. 연습 문제

1. **불변 스택**: `ImmutableStack<T>`를 레코드로 구현하세요: `record ImmutableStack<T>(T Head, ImmutableStack<T>? Tail)`. `Push(T item)`, `Pop(out T item)`, `Peek()`, `IsEmpty` 메서드를 추가하세요. 각 연산은 새 스택을 반환해야 합니다. 정적 `Empty` 속성을 포함하세요. 스택이 진정으로 불변임을 보여주는 테스트를 작성하세요.

2. **버전 관리 문서**: `Title`, `Content`, `Version`(int) 속성을 가진 `Document` 레코드를 만드세요. 증가된 버전으로 새 Document를 반환하는 `EditDocument(Document doc, string newContent)` 함수를 구현하세요. 모든 버전을 `ImmutableList<Document>`에 저장하고 과거 버전을 조회하는 `GetVersion(int version)`을 구현하세요.

3. **레코드 동등성 사용자 정의**: `Name`과 `Value` 문자열 속성을 가진 `CaseInsensitiveRecord` 레코드를 만드세요. `Name` 비교는 대소문자 무시, `Value` 비교는 대소문자 구분이 되도록 동등성을 재정의하세요. 다른 대소문자의 인스턴스를 만들어 `HashSet`에서 동등성과 해시 코드 동작을 검증하세요.

4. **불변 설정을 위한 빌더**: 8개 이상의 속성(host, port, ssl, timeout, maxConnections, logLevel, ImmutableList로서의 corsOrigins, ImmutableDictionary로서의 headers)을 가진 `ServerConfig` 레코드를 설계하세요. 플루언트 메서드를 가진 `ServerConfigBuilder` 클래스를 만드세요. 빌더는 `Build()`에서 모든 필수 필드를 검증하고 불변 레코드를 반환해야 합니다.

5. **이벤트 소싱을 사용한 장바구니**: 레코드를 사용하여 이벤트 소싱으로 장바구니를 모델링하세요. 이벤트 정의: `CartCreated`, `ItemAdded(string ProductId, int Quantity, decimal Price)`, `ItemRemoved(string ProductId)`, `QuantityChanged(string ProductId, int NewQuantity)`, `CartCheckedOut`. 장바구니 상태는 `ImmutableDictionary<string, CartItem>`을 포함하는 레코드여야 합니다. `Apply`와 `Rebuild`를 구현한 후 장바구니를 생성하고, 3개 항목 추가, 1개 제거, 다른 하나의 수량 변경, 체크아웃하는 시나리오를 작성하세요.
