# 열거형과 구조체

**이전**: [배열과 문자열](./06_Arrays_and_Strings.md) | **다음**: [컬렉션](./08_Collections.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 사용자 정의 기본 타입으로 열거형(Enum)을 선언하고 사용하기
2. 비트 연산을 위한 `[Flags]` 특성 적용하기
3. 런타임에서 열거형 값을 파싱하고 반복하기
4. 구조체(Struct)를 정의하고 값 의미론 이해하기
5. 불변 값 타입을 위한 `readonly struct` 사용하기
6. 구조체와 클래스를 언제 사용해야 하는지 구분하기
7. 경량 데이터 운반체로서 레코드 구조체(Record Struct)와 튜플(Tuple) 활용하기

---

열거형(Enum)과 구조체(Struct)는 C#의 두 가지 기본 값 타입입니다. 열거형은 관련 상수 집합에 의미 있는 이름을 부여하여 코드를 더 읽기 쉽고 오류가 적게 만듭니다. 구조체는 작은 데이터 그룹을 위한 경량의 스택 할당 컨테이너를 제공합니다. 두 가지를 합치면 C#의 값 타입 프로그래밍의 근간을 이룹니다.

## 1. 열거형 선언과 사용

열거형(Enumeration)은 명명된 정수 상수 집합을 정의합니다.

### 1.1 기본 열거형 선언

```csharp
// 기본 열거형 선언
enum Season
{
    Spring,   // 0
    Summer,   // 1
    Autumn,   // 2
    Winter    // 3
}

// 사용
Season current = Season.Summer;
Console.WriteLine(current);        // "Summer"
Console.WriteLine((int)current);   // 1

// 비교
if (current == Season.Summer)
{
    Console.WriteLine("휴가 시간입니다!");
}

// 열거형에 대한 switch
string description = current switch
{
    Season.Spring => "꽃이 핀다",
    Season.Summer => "해가 높다",
    Season.Autumn => "낙엽이 진다",
    Season.Winter => "눈이 내린다",
    _ => "알 수 없는 계절"
};
```

### 1.2 사용자 정의 값

기본적으로 열거형 멤버는 0부터 시작하여 1씩 증가합니다. 명시적 값을 할당할 수 있습니다:

```csharp
enum HttpStatusCode
{
    OK = 200,
    Created = 201,
    NoContent = 204,
    BadRequest = 400,
    Unauthorized = 401,
    Forbidden = 403,
    NotFound = 404,
    InternalServerError = 500
}

HttpStatusCode status = HttpStatusCode.NotFound;
int code = (int)status; // 404

// 열거형 멤버는 다른 멤버를 참조할 수 있음
enum Priority
{
    Low = 1,
    Medium = 5,
    High = 10,
    Critical = High + 10  // 20
}
```

### 1.3 메서드에서의 열거형

```csharp
enum Direction { North, South, East, West }

static (int dx, int dy) GetDelta(Direction dir)
{
    return dir switch
    {
        Direction.North => (0, 1),
        Direction.South => (0, -1),
        Direction.East  => (1, 0),
        Direction.West  => (-1, 0),
        _ => throw new ArgumentOutOfRangeException(nameof(dir))
    };
}

// 사용
var (dx, dy) = GetDelta(Direction.North);
Console.WriteLine($"dx={dx}, dy={dy}"); // dx=0, dy=1
```

## 2. 열거형 기본 타입

기본적으로 열거형은 `int`를 기본 타입으로 사용합니다. 다른 정수 타입을 지정할 수 있습니다.

### 2.1 기본 타입 지정

```csharp
// 메모리 절약을 위해 byte 사용 (0-255 범위)
enum Color : byte
{
    Red = 1,
    Green = 2,
    Blue = 3
}

// 큰 값을 위해 long 사용
enum FileSize : long
{
    Kilobyte = 1024L,
    Megabyte = 1024L * 1024,
    Gigabyte = 1024L * 1024 * 1024,
    Terabyte = 1024L * 1024 * 1024 * 1024
}

// 지원되는 기본 타입: byte, sbyte, short, ushort, int, uint, long, ulong
Console.WriteLine(sizeof(Color));    // 1 바이트
Console.WriteLine((long)FileSize.Terabyte); // 1099511627776
```

### 2.2 열거형과 정수 간 캐스팅

```csharp
enum Planet { Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, Neptune }

// 열거형에서 int로
int earthIndex = (int)Planet.Earth; // 2

// int에서 열거형으로
Planet planet = (Planet)4; // Jupiter

// 주의: 유효하지 않은 값도 런타임에서 허용됨
Planet invalid = (Planet)99; // 컴파일 오류 없음, 하지만 논리적으로 유효하지 않음
Console.WriteLine(invalid);  // "99"

// Enum.IsDefined로 유효성 검사
bool isValid = Enum.IsDefined(typeof(Planet), 4);  // true
bool isInvalid = Enum.IsDefined(typeof(Planet), 99); // false
```

## 3. 비트 연산을 위한 플래그 열거형

`[Flags]` 특성은 비트 연산자를 사용하여 열거형 값을 조합할 수 있게 해주며, 옵션 집합을 표현하는 데 유용합니다.

### 3.1 플래그 열거형 선언

```csharp
[Flags]
enum FilePermissions
{
    None    = 0,       // 0b0000
    Read    = 1,       // 0b0001
    Write   = 2,       // 0b0010
    Execute = 4,       // 0b0100
    All     = Read | Write | Execute  // 0b0111 = 7
}

// 비트 OR로 플래그 조합
FilePermissions userPerms = FilePermissions.Read | FilePermissions.Write;
Console.WriteLine(userPerms); // "Read, Write"

// HasFlag로 플래그 설정 여부 확인
bool canRead = userPerms.HasFlag(FilePermissions.Read);   // true
bool canExec = userPerms.HasFlag(FilePermissions.Execute); // false

// 대안: 비트 AND로 확인 (더 빠름)
bool canWrite = (userPerms & FilePermissions.Write) != 0; // true
```

### 3.2 플래그 조작

```csharp
[Flags]
enum DaysOfWeek
{
    None      = 0,
    Monday    = 1,
    Tuesday   = 2,
    Wednesday = 4,
    Thursday  = 8,
    Friday    = 16,
    Saturday  = 32,
    Sunday    = 64,
    Weekdays  = Monday | Tuesday | Wednesday | Thursday | Friday,
    Weekend   = Saturday | Sunday,
    All       = Weekdays | Weekend
}

DaysOfWeek schedule = DaysOfWeek.Monday | DaysOfWeek.Wednesday | DaysOfWeek.Friday;

// 플래그 추가
schedule |= DaysOfWeek.Tuesday;

// 플래그 제거
schedule &= ~DaysOfWeek.Monday;

// 플래그 토글
schedule ^= DaysOfWeek.Friday;

// 멤버십 확인
bool worksWednesday = schedule.HasFlag(DaysOfWeek.Wednesday); // true

// 일정에 평일이 포함되어 있는지 확인
bool hasWeekday = (schedule & DaysOfWeek.Weekdays) != 0; // true

Console.WriteLine(schedule); // "Tuesday, Wednesday"
```

### 3.3 플래그 모범 사례

```csharp
// 항상 2의 거듭제곱을 할당 (또는 명확성을 위해 비트 시프트 사용)
[Flags]
enum TextStyle
{
    None          = 0,
    Bold          = 1 << 0,  // 1
    Italic        = 1 << 1,  // 2
    Underline     = 1 << 2,  // 4
    Strikethrough = 1 << 3,  // 8
    Highlight     = 1 << 4,  // 16
    BoldItalic    = Bold | Italic  // 편의 조합
}

TextStyle style = TextStyle.Bold | TextStyle.Underline;
Console.WriteLine(style); // "Bold, Underline"
```

## 4. 열거형 유틸리티 메서드

`System.Enum` 클래스는 런타임에서 열거형을 다루기 위한 정적 메서드를 제공합니다.

### 4.1 Parse와 TryParse

```csharp
enum Color { Red, Green, Blue }

// 문자열에서 파싱 (기본적으로 대소문자 구분)
Color parsed = (Color)Enum.Parse(typeof(Color), "Green");
Console.WriteLine(parsed); // Green

// 제네릭 파싱 (C# 7+)
Color parsed2 = Enum.Parse<Color>("Blue");

// 대소문자 무시 파싱
Color parsed3 = Enum.Parse<Color>("red", ignoreCase: true);

// TryParse (안전, 예외 없음)
if (Enum.TryParse<Color>("Green", out Color result))
{
    Console.WriteLine($"파싱됨: {result}"); // 파싱됨: Green
}

if (!Enum.TryParse<Color>("Yellow", out Color _))
{
    Console.WriteLine("Yellow는 유효한 Color가 아닙니다");
}
```

### 4.2 GetValues와 GetNames

```csharp
enum Fruit { Apple, Banana, Cherry, Date, Elderberry }

// 모든 값 가져오기
Fruit[] allFruits = Enum.GetValues<Fruit>();
foreach (Fruit f in allFruits)
{
    Console.WriteLine($"{f} = {(int)f}");
}
// Apple = 0, Banana = 1, Cherry = 2, Date = 3, Elderberry = 4

// 모든 이름을 문자열로 가져오기
string[] names = Enum.GetNames<Fruit>();
Console.WriteLine(string.Join(", ", names));
// "Apple, Banana, Cherry, Date, Elderberry"

// 검색 딕셔너리 구축
Dictionary<string, Fruit> lookup = new();
foreach (Fruit f in Enum.GetValues<Fruit>())
{
    lookup[f.ToString()] = f;
}
```

### 4.3 열거형을 표시 문자열로 변환

```csharp
enum OrderStatus
{
    Pending,
    InProgress,
    Shipped,
    Delivered,
    Cancelled
}

// 간단한 접근: ToString과 수동 서식
static string ToDisplayName(OrderStatus status)
{
    return status switch
    {
        OrderStatus.InProgress => "진행 중",
        _ => status.ToString()
    };
}

// 드롭다운 메뉴를 위한 모든 값 반복
static List<(int Value, string Label)> GetDropdownOptions<T>() where T : struct, Enum
{
    var options = new List<(int, string)>();
    foreach (T value in Enum.GetValues<T>())
    {
        options.Add((Convert.ToInt32(value), value.ToString()));
    }
    return options;
}
```

## 5. 구조체 선언

구조체(Struct)는 필드, 메서드, 속성, 생성자를 포함할 수 있는 값 타입입니다. 구조체는 (지역 변수로 사용될 때) 스택에 할당되며 값으로 복사됩니다.

### 5.1 기본 구조체

```csharp
struct Point
{
    public double X;
    public double Y;

    // 생성자
    public Point(double x, double y)
    {
        X = x;
        Y = y;
    }

    // 메서드
    public double DistanceTo(Point other)
    {
        double dx = X - other.X;
        double dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }

    // ToString 재정의
    public override string ToString() => $"({X}, {Y})";
}

// 사용
Point p1 = new Point(3, 4);
Point p2 = new Point(0, 0);
double dist = p1.DistanceTo(p2); // 5.0
Console.WriteLine(p1);           // "(3, 4)"

// 기본 생성자 (모든 필드가 기본값으로 설정)
Point origin = new Point(); // X=0, Y=0
Point same = default;       // 역시 X=0, Y=0
```

### 5.2 속성이 있는 구조체

```csharp
struct Rectangle
{
    public double Width { get; set; }
    public double Height { get; set; }

    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }

    // 계산된 속성
    public double Area => Width * Height;
    public double Perimeter => 2 * (Width + Height);
    public bool IsSquare => Width == Height;

    public override string ToString()
        => $"Rectangle({Width} x {Height}, Area={Area})";
}

Rectangle rect = new Rectangle(5, 3);
Console.WriteLine(rect.Area);      // 15
Console.WriteLine(rect.Perimeter); // 16
Console.WriteLine(rect.IsSquare);  // false
```

## 6. 값 의미론 vs 참조 의미론

구조체와 클래스의 가장 중요한 차이점은 대입하거나 메서드에 전달할 때의 동작 방식입니다.

### 6.1 복사 동작

```csharp
// 구조체 (값 타입): 대입이 데이터를 복사
struct PointStruct
{
    public int X, Y;
}

PointStruct a = new PointStruct { X = 1, Y = 2 };
PointStruct b = a;  // b는 a의 복사본
b.X = 99;

Console.WriteLine(a.X); // 1 (변경되지 않음)
Console.WriteLine(b.X); // 99

// 클래스 (참조 타입): 대입이 참조를 복사
class PointClass
{
    public int X, Y;
}

PointClass c = new PointClass { X = 1, Y = 2 };
PointClass d = c;  // d는 c와 같은 객체를 가리킴
d.X = 99;

Console.WriteLine(c.X); // 99 (변경됨!)
Console.WriteLine(d.X); // 99
```

### 6.2 메서드 매개변수 동작

```csharp
struct ValuePoint { public int X, Y; }

static void ModifyStruct(ValuePoint p)
{
    p.X = 999; // 로컬 복사본을 수정; 호출자의 값은 변경되지 않음
}

static void ModifyStructByRef(ref ValuePoint p)
{
    p.X = 999; // 호출자의 값을 직접 수정
}

ValuePoint pt = new ValuePoint { X = 1, Y = 2 };

ModifyStruct(pt);
Console.WriteLine(pt.X); // 1 (변경되지 않음)

ModifyStructByRef(ref pt);
Console.WriteLine(pt.X); // 999 (변경됨)
```

### 6.3 동등성

```csharp
struct Coordinate
{
    public int X, Y;
}

Coordinate a = new Coordinate { X = 5, Y = 10 };
Coordinate b = new Coordinate { X = 5, Y = 10 };

// 구조체는 기본적으로 값 동등성 사용 (리플렉션을 통해, 느림)
Console.WriteLine(a.Equals(b)); // true

// 성능을 위해 Equals와 GetHashCode 재정의
struct BetterCoordinate : IEquatable<BetterCoordinate>
{
    public int X, Y;

    public bool Equals(BetterCoordinate other)
        => X == other.X && Y == other.Y;

    public override bool Equals(object? obj)
        => obj is BetterCoordinate other && Equals(other);

    public override int GetHashCode()
        => HashCode.Combine(X, Y);

    public static bool operator ==(BetterCoordinate left, BetterCoordinate right)
        => left.Equals(right);

    public static bool operator !=(BetterCoordinate left, BetterCoordinate right)
        => !left.Equals(right);
}
```

## 7. Readonly 구조체

`readonly struct`는 생성 후 어떤 인스턴스 멤버도 구조체의 상태를 수정할 수 없음을 보장합니다.

### 7.1 선언

```csharp
readonly struct Vector3
{
    public double X { get; }
    public double Y { get; }
    public double Z { get; }

    public Vector3(double x, double y, double z)
    {
        X = x;
        Y = y;
        Z = z;
    }

    public double Magnitude
        => Math.Sqrt(X * X + Y * Y + Z * Z);

    public Vector3 Normalized()
    {
        double mag = Magnitude;
        return new Vector3(X / mag, Y / mag, Z / mag);
    }

    // 연산자 오버로딩
    public static Vector3 operator +(Vector3 a, Vector3 b)
        => new Vector3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

    public static Vector3 operator *(Vector3 v, double scalar)
        => new Vector3(v.X * scalar, v.Y * scalar, v.Z * scalar);

    public static double Dot(Vector3 a, Vector3 b)
        => a.X * b.X + a.Y * b.Y + a.Z * b.Z;

    public override string ToString() => $"<{X}, {Y}, {Z}>";
}

Vector3 v1 = new Vector3(1, 2, 3);
Vector3 v2 = new Vector3(4, 5, 6);
Vector3 sum = v1 + v2;                    // <5, 7, 9>
double dot = Vector3.Dot(v1, v2);         // 32
Vector3 scaled = v1 * 2.0;               // <2, 4, 6>
Console.WriteLine(v1.Normalized());        // <0.267..., 0.534..., 0.801...>
```

### 7.2 비 Readonly 구조체의 Readonly 멤버

전체 구조체가 `readonly`가 아니더라도 개별 멤버를 `readonly`로 표시할 수 있습니다:

```csharp
struct MutablePoint
{
    public double X;
    public double Y;

    // 이 메서드는 구조체를 수정하지 않겠다고 약속
    public readonly double DistanceFromOrigin()
        => Math.Sqrt(X * X + Y * Y);

    // 이 메서드는 상태를 수정하므로 readonly가 될 수 없음
    public void Reset()
    {
        X = 0;
        Y = 0;
    }
}
```

## 8. 구조체 vs 클래스: 언제 어떤 것을 사용할까

### 8.1 결정 가이드라인

| **구조체**를 사용할 때... | **클래스**를 사용할 때... |
|--------------------------|------------------------|
| 데이터가 작을 때 (약 16바이트 미만) | 데이터가 크거나 복잡할 때 |
| 단일 값을 나타낼 때 (좌표 같은) | 정체성을 가진 엔티티를 나타낼 때 |
| 인스턴스가 수명이 짧을 때 | 인스턴스가 오래 살거나 공유될 때 |
| 상속이 필요 없을 때 | 상속이나 다형성이 필요할 때 |
| 불변성이 바람직할 때 | 공유 참조와 함께 변경 가능한 상태가 필요할 때 |
| 빈번하게 할당될 때 (GC 부담 회피) | null 의미론이 필요할 때 |

### 8.2 .NET 프레임워크의 예시

```csharp
// .NET에서 구조체인 것들 (작고, 값과 유사)
DateTime birthday = new DateTime(1990, 5, 15);
TimeSpan duration = TimeSpan.FromHours(2.5);
Guid id = Guid.NewGuid();
decimal price = 29.99m;
KeyValuePair<string, int> pair = new("Alice", 30);

// .NET에서 클래스인 것들 (복잡하고, 정체성 기반)
string name = "Hello";         // 불변 참조 타입
List<int> list = new();        // 변경 가능, 공유됨
Exception ex = new Exception(); // 상속 계층 구조
Stream stream = File.Open("f.txt", FileMode.Open); // 리소스 관리
```

### 8.3 박싱 경고

구조체가 `object` 또는 인터페이스 변수에 할당되면 박싱(Boxing, 힙으로 복사)됩니다:

```csharp
struct SmallData { public int Value; }

SmallData data = new SmallData { Value = 42 };

// 박싱: 구조체를 힙으로 복사하여 object로 래핑
object boxed = data;

// 언박싱: 힙에서 다시 복사
SmallData unboxed = (SmallData)boxed;

// 언박싱된 것을 수정해도 박싱된 것에 영향 없음
unboxed.Value = 99;
Console.WriteLine(((SmallData)boxed).Value); // 여전히 42

// 핫 경로에서 박싱 피하기 (object 대신 제네릭 사용)
```

## 9. 레코드 구조체

C# 10에서 내장 동등성(Equality), `ToString`, 해체(Deconstruction)를 갖춘 간결한 값 타입 데이터 운반체인 `record struct`가 도입되었습니다.

### 9.1 위치 레코드 구조체

```csharp
// 간결한 구문: 컴파일러가 속성, Equals, GetHashCode, ToString 생성
record struct Point(double X, double Y);

Point p1 = new Point(3, 4);
Point p2 = new Point(3, 4);

Console.WriteLine(p1);          // "Point { X = 3, Y = 4 }"
Console.WriteLine(p1 == p2);    // true (값 동등성)

// 해체
var (x, y) = p1;
Console.WriteLine($"x={x}, y={y}"); // x=3, y=4

// 기본적으로 변경 가능 (record class와 다름)
p1.X = 10;
Console.WriteLine(p1); // "Point { X = 10, Y = 4 }"
```

### 9.2 Readonly 레코드 구조체

```csharp
// 불변 레코드 구조체
readonly record struct Color(byte R, byte G, byte B)
{
    // 추가 계산된 속성
    public string Hex => $"#{R:X2}{G:X2}{B:X2}";
}

Color red = new Color(255, 0, 0);
Console.WriteLine(red);      // "Color { R = 255, G = 0, B = 0 }"
Console.WriteLine(red.Hex);  // "#FF0000"

// with 표현식으로 수정된 복사본 생성
Color darkRed = red with { R = 139 };
Console.WriteLine(darkRed.Hex); // "#8B0000"
```

### 9.3 레코드 구조체 vs 구조체 vs 레코드 클래스

```csharp
// 일반 구조체: 수동으로 Equals/GetHashCode/ToString 작성
struct ManualPoint
{
    public double X, Y;
    // Equals, GetHashCode, ToString을 수동으로 재정의해야 함
}

// 레코드 구조체: 자동 생성된 Equals/GetHashCode/ToString, 값 타입
record struct AutoPoint(double X, double Y);

// 레코드 클래스: 자동 생성, 참조 타입, 기본적으로 불변
record class RefPoint(double X, double Y);

// 주요 차이점:
// - record struct는 값 타입 (스택 할당, 대입 시 복사)
// - record class는 참조 타입 (힙 할당, 대입 시 공유)
// - record struct는 기본적으로 변경 가능; record class는 기본적으로 불변
```

## 10. 튜플

튜플(Tuple)은 완전한 타입을 정의하지 않고 여러 값을 그룹화하는 경량 방법을 제공합니다.

### 10.1 ValueTuple 기초

```csharp
// 튜플 생성
(string Name, int Age) person = ("Alice", 30);
Console.WriteLine(person.Name); // "Alice"
Console.WriteLine(person.Age);  // 30

// var와 함께 튜플
var point = (X: 3.0, Y: 4.0);
Console.WriteLine($"({point.X}, {point.Y})");

// 이름 없는 튜플 (Item1, Item2 등으로 접근)
var unnamed = (42, "hello", true);
Console.WriteLine(unnamed.Item1); // 42
Console.WriteLine(unnamed.Item2); // "hello"
```

### 10.2 메서드에서의 튜플

```csharp
// 메서드에서 여러 값 반환
static (double Min, double Max, double Average) GetStats(int[] numbers)
{
    double min = numbers.Min();
    double max = numbers.Max();
    double avg = numbers.Average();
    return (min, max, avg);
}

int[] data = { 5, 3, 8, 1, 9 };
var stats = GetStats(data);
Console.WriteLine($"최소={stats.Min}, 최대={stats.Max}, 평균={stats.Average}");
// 최소=1, 최대=9, 평균=5.2

// 개별 변수로 해체
var (min, max, avg) = GetStats(data);
Console.WriteLine($"범위: {min}에서 {max}까지");
```

### 10.3 튜플 동등성과 해체

```csharp
// 튜플은 값 동등성을 지원
var a = (1, "hello");
var b = (1, "hello");
Console.WriteLine(a == b); // true

// 버림(discard)을 사용한 해체
var (name, _, age) = ("Alice", "Middle", 30);
// 두 번째 요소를 무시

// 튜플 대입 (교환)
int x = 1, y = 2;
(x, y) = (y, x);
Console.WriteLine($"x={x}, y={y}"); // x=2, y=1

// 중첩 튜플
var nested = ((1, 2), (3, 4));
var ((a1, a2), (b1, b2)) = nested;
```

### 10.4 튜플 vs 명명된 타입: 언제 사용할까

```csharp
// 좋은 사용: 몇 개의 값을 반환하는 내부 헬퍼 메서드
static (bool Success, string Message) Validate(string input)
{
    if (string.IsNullOrWhiteSpace(input))
        return (false, "입력이 비어 있을 수 없습니다");
    if (input.Length > 100)
        return (false, "입력이 너무 깁니다");
    return (true, "유효합니다");
}

// 공개 API나 복잡한 데이터에는 명명된 타입 고려
// (더 나은 가독성과 문서화)
record struct ValidationResult(bool Success, string Message);

static ValidationResult ValidateBetter(string input)
{
    if (string.IsNullOrWhiteSpace(input))
        return new ValidationResult(false, "입력이 비어 있을 수 없습니다");
    return new ValidationResult(true, "유효합니다");
}
```

## 11. 연습 문제

1. **신호등 열거형**: `Red`, `Yellow`, `Green` 값을 가진 열거형 `TrafficLight`를 정의하세요. 순환에서 다음 상태를 반환하는 메서드 `TrafficLight NextLight(TrafficLight current)`를 작성하세요 (Red -> Green -> Yellow -> Red). 또한 각 신호 색상에 대한 권장 대기 시간(초)을 반환하는 메서드도 작성하세요.

2. **권한 시스템**: `None`, `Read`, `Write`, `Execute`, `Delete`, `Admin`(모든 것의 조합) 값을 가진 `[Flags]` 열거형 `Permission`을 만드세요. `void GrantPermission(ref Permission current, Permission toGrant)`, `void RevokePermission(ref Permission current, Permission toRevoke)`, `bool HasPermission(Permission current, Permission required)` 메서드를 작성하세요.

3. **불변 Money 구조체**: `decimal Amount`와 `string Currency` 속성을 가진 `readonly struct Money`를 만드세요. `+`, `-`, `*`(숫자로 스케일링) 연산자 오버로딩을 구현하세요. 덧셈이나 뺄셈에서 통화가 일치하지 않으면 `InvalidOperationException`을 던지세요. `ToString`을 재정의하여 `"$123.45 USD"`로 표시하세요.

4. **색상 변환기**: 16진수 문자열(`"#FF0000"`은 빨강)과의 상호 변환 메서드를 가진 `readonly record struct RgbColor(byte R, byte G, byte B)`를 만드세요. 또한 두 색상을 구성 요소별로 평균하는 `RgbColor Mix(RgbColor other)` 메서드도 추가하세요.

5. **열거형 파서 메뉴**: 열거형의 모든 가능한 값을 출력하고, 사용자에게 이름을 입력하도록 요청하고, 파싱된 열거형 값을 반환하는 제네릭 메서드 `T PromptForEnum<T>(string prompt) where T : struct, Enum`을 작성하세요. `Enum.GetNames<T>()`와 `Enum.TryParse<T>()`를 사용하세요. 잘못된 입력에 대해 다시 요청하도록 처리하세요.
