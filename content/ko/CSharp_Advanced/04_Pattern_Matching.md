# 패턴 매칭

**이전**: [LINQ](./03_LINQ.md) | **다음**: [Nullable 참조 타입](./05_Nullable_Reference_Types.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 타입 패턴을 사용하여 하나의 식에서 안전하게 캐스팅하고 타입 검사하기
2. 속성 패턴을 적용하여 객체 구조에 대해 매칭하기
3. 위치 패턴으로 객체 분해하기
4. 관계형 패턴과 논리 패턴을 작성하여 범위와 조건 표현하기
5. 리스트 패턴을 사용하여 시퀀스를 형태로 매칭하기
6. switch 표현식에서 패턴을 결합하여 간결하고 완전한 분기 작성하기
7. 중첩 패턴 매칭을 사용하여 실용적인 식 평가기 구축하기

---

C#의 패턴 매칭(pattern matching)은 값을 형태 — 타입, 구조, 속성, 위치 — 에 대해 검사하고 단일 선언적 식에서 데이터를 추출할 수 있게 합니다. C# 7부터 C# 12까지 점진적으로 도입된 패턴 매칭은 많은 장황한 `if-else` 체인과 `switch` 문을 명확하고, 간결하며, 컴파일러가 검증하는 코드로 대체할 수 있는 종합적인 기능으로 발전했습니다.

## 1. 타입 패턴

타입 패턴은 값이 주어진 타입인지 검사하고 선택적으로 변수에 바인딩합니다.

### 1.1 타입 패턴이 있는 `is` 표현식

```csharp
object value = "Hello, World!";

// 변수 바인딩이 있는 타입 패턴
if (value is string text)
{
    Console.WriteLine($"길이 {text.Length}의 문자열"); // 길이 13의 문자열
}

// 부정 타입 패턴
if (value is not int)
{
    Console.WriteLine("정수가 아닙니다");
}

// 메서드에서의 타입 패턴
static string Describe(object obj)
{
    if (obj is int n)
        return $"정수: {n}";
    if (obj is double d)
        return $"실수: {d:F2}";
    if (obj is string s)
        return $"문자열: \"{s}\"";
    if (obj is null)
        return "Null";
    return $"알 수 없음: {obj.GetType().Name}";
}

Console.WriteLine(Describe(42));        // 정수: 42
Console.WriteLine(Describe(3.14));      // 실수: 3.14
Console.WriteLine(Describe("test"));    // 문자열: "test"
Console.WriteLine(Describe(null!));     // Null
```

### 1.2 Switch에서의 타입 패턴

```csharp
static double CalculateArea(object shape) => shape switch
{
    Circle c => Math.PI * c.Radius * c.Radius,
    Rectangle r => r.Width * r.Height,
    Triangle t => 0.5 * t.Base * t.Height,
    _ => throw new ArgumentException($"알 수 없는 도형: {shape.GetType().Name}")
};

record Circle(double Radius);
record Rectangle(double Width, double Height);
record Triangle(double Base, double Height);

Console.WriteLine(CalculateArea(new Circle(5)));          // 78.54
Console.WriteLine(CalculateArea(new Rectangle(4, 6)));    // 24
Console.WriteLine(CalculateArea(new Triangle(3, 8)));     // 12
```

## 2. 상수 패턴

상수 패턴은 `null`을 포함한 컴파일 타임 상수에 대해 값을 매칭합니다.

```csharp
static string ClassifyHttpStatus(int statusCode) => statusCode switch
{
    200 => "OK",
    201 => "Created",
    204 => "No Content",
    301 => "Moved Permanently",
    400 => "Bad Request",
    401 => "Unauthorized",
    403 => "Forbidden",
    404 => "Not Found",
    500 => "Internal Server Error",
    503 => "Service Unavailable",
    _ => $"알 수 없음 ({statusCode})"
};

Console.WriteLine(ClassifyHttpStatus(200)); // OK
Console.WriteLine(ClassifyHttpStatus(404)); // Not Found
Console.WriteLine(ClassifyHttpStatus(418)); // 알 수 없음 (418)

// Null 상수 패턴
static string SafeToUpper(string? input) => input switch
{
    null => "(null)",
    "" => "(비어있음)",
    _ => input.ToUpper()
};
```

## 3. 속성 패턴

속성 패턴(property pattern)은 객체의 속성이나 필드 값에 대해 매칭합니다. 중괄호 `{ }`를 사용하여 속성 조건을 지정합니다.

### 3.1 기본 속성 패턴

```csharp
record Address(string City, string State, string ZipCode);
record Person(string Name, int Age, Address Address);

static string DescribePerson(Person p) => p switch
{
    { Age: < 13 } => "어린이",
    { Age: < 18 } => "청소년",
    { Age: >= 65 } => "시니어",
    { Name: "Alice" } => "Alice입니다!",
    _ => "성인"
};

var alice = new Person("Alice", 30, new Address("Seattle", "WA", "98101"));
Console.WriteLine(DescribePerson(alice)); // Alice입니다!

var kid = new Person("Timmy", 8, new Address("Portland", "OR", "97201"));
Console.WriteLine(DescribePerson(kid)); // 어린이
```

### 3.2 중첩 속성 패턴

```csharp
static decimal CalculateShipping(Person customer) => customer switch
{
    { Address: { State: "WA" } } => 0.00m,       // 워싱턴주 무료 배송
    { Address: { State: "OR" or "CA" } } => 5.99m, // 서부 해안
    { Address: { State: "NY" or "NJ" } } => 7.99m, // 동부 해안
    { Age: >= 65 } => 2.99m,                        // 시니어 할인
    _ => 9.99m                                       // 표준
};

Console.WriteLine(CalculateShipping(alice)); // 0.00 (WA)

// 축약 중첩 접근 (C# 10+) — 점 표기법
static string GetRegion(Person p) => p switch
{
    { Address.State: "WA" or "OR" or "CA" } => "서부 해안",
    { Address.State: "NY" or "NJ" or "CT" } => "동부 해안",
    { Address.State: "TX" or "FL" } => "남부",
    _ => "기타"
};
```

### 3.3 변수 바인딩이 있는 속성 패턴

```csharp
static string FormatAddress(Person p) => p switch
{
    { Name: var name, Address: { City: var city, State: "WA" } }
        => $"{name}은(는) {city}, 워싱턴에 거주합니다",
    { Name: var name, Address: { City: var city, State: var state } }
        => $"{name}은(는) {city}, {state}에 거주합니다",
};

Console.WriteLine(FormatAddress(alice)); // Alice은(는) Seattle, 워싱턴에 거주합니다
```

## 4. 위치 패턴

위치 패턴(positional pattern)은 `Deconstruct` 메서드가 있는 타입(또는 위치 레코드/튜플)의 구성 요소에 대해 분해를 사용하여 매칭합니다.

### 4.1 튜플 패턴

```csharp
static string ClassifyPoint(int x, int y) => (x, y) switch
{
    (0, 0) => "원점",
    (> 0, > 0) => "제1사분면",
    (< 0, > 0) => "제2사분면",
    (< 0, < 0) => "제3사분면",
    (> 0, < 0) => "제4사분면",
    (0, _) => "Y축",
    (_, 0) => "X축"
};

Console.WriteLine(ClassifyPoint(3, 4));   // 제1사분면
Console.WriteLine(ClassifyPoint(-1, 5));  // 제2사분면
Console.WriteLine(ClassifyPoint(0, 0));   // 원점
Console.WriteLine(ClassifyPoint(0, -7));  // Y축
```

### 4.2 위치 레코드

```csharp
record Point3D(double X, double Y, double Z);

static string Classify(Point3D p) => p switch
{
    (0, 0, 0) => "원점",
    (_, 0, 0) => "X축 위",
    (0, _, 0) => "Y축 위",
    (0, 0, _) => "Z축 위",
    (var x, var y, 0) => $"XY 평면의 ({x}, {y})",
    (var x, var y, var z) => $"3D 점 ({x}, {y}, {z})"
};

Console.WriteLine(Classify(new Point3D(0, 0, 0)));   // 원점
Console.WriteLine(Classify(new Point3D(5, 0, 0)));   // X축 위
Console.WriteLine(Classify(new Point3D(3, 4, 0)));   // XY 평면의 (3, 4)
Console.WriteLine(Classify(new Point3D(1, 2, 3)));   // 3D 점 (1, 2, 3)
```

### 4.3 사용자 정의 Deconstruct 메서드

```csharp
public class Temperature
{
    public double Celsius { get; }
    public Temperature(double celsius) => Celsius = celsius;

    public void Deconstruct(out double celsius, out double fahrenheit)
    {
        celsius = Celsius;
        fahrenheit = Celsius * 9 / 5 + 32;
    }
}

static string DescribeTemp(Temperature t) => t switch
{
    ( < -40, _) => "극한 추위",
    ( < 0, _) => "영하",
    ( < 20, _) => "서늘함",
    ( < 30, _) => "쾌적함",
    ( < 40, _) => "더움",
    _ => "극한 더위"
};

Console.WriteLine(DescribeTemp(new Temperature(-50)));  // 극한 추위
Console.WriteLine(DescribeTemp(new Temperature(22)));   // 쾌적함
Console.WriteLine(DescribeTemp(new Temperature(45)));   // 극한 더위
```

## 5. 관계형 패턴

관계형 패턴(relational pattern)은 `<`, `>`, `<=`, `>=`를 사용하여 값을 상수와 비교합니다.

```csharp
static string GradeFromScore(int score) => score switch
{
    >= 97 => "A+",
    >= 93 => "A",
    >= 90 => "A-",
    >= 87 => "B+",
    >= 83 => "B",
    >= 80 => "B-",
    >= 77 => "C+",
    >= 73 => "C",
    >= 70 => "C-",
    >= 67 => "D+",
    >= 63 => "D",
    >= 60 => "D-",
    _ => "F"
};

Console.WriteLine(GradeFromScore(95)); // A
Console.WriteLine(GradeFromScore(82)); // B-
Console.WriteLine(GradeFromScore(55)); // F
```

## 6. 논리 패턴

논리 패턴(logical pattern)은 `and`, `or`, `not`을 사용하여 다른 패턴을 결합합니다.

### 6.1 And / Or 패턴

```csharp
static string ClassifyTemperature(double temp) => temp switch
{
    >= -10 and < 0 => "추움",
    >= 0 and < 15 => "서늘함",
    >= 15 and < 25 => "쾌적함",
    >= 25 and < 35 => "따뜻함",
    >= 35 and < 45 => "더움",
    _ => "극한"
};

Console.WriteLine(ClassifyTemperature(22));  // 쾌적함
Console.WriteLine(ClassifyTemperature(-5));  // 추움
Console.WriteLine(ClassifyTemperature(50));  // 극한

// Or 패턴 — 여러 값 매칭
static bool IsWeekend(DayOfWeek day) => day is DayOfWeek.Saturday or DayOfWeek.Sunday;

static bool IsVowel(char c) => char.ToLower(c) is 'a' or 'e' or 'i' or 'o' or 'u';
```

### 6.2 Not 패턴

```csharp
// !=보다 깔끔함
if (value is not null)
{
    Console.WriteLine(value);
}

// 타입 패턴과 결합된 not
if (shape is not Circle)
{
    Console.WriteLine("원이 아닙니다");
}

// switch에서의 not
static string Validate(int age) => age switch
{
    not (>= 0 and <= 150) => "유효하지 않은 나이",
    < 18 => "미성년자",
    >= 65 => "시니어",
    _ => "성인"
};
```

### 6.3 복잡한 논리 조합

```csharp
static string ClassifyCharacter(char c) => c switch
{
    >= 'a' and <= 'z' => "소문자",
    >= 'A' and <= 'Z' => "대문자",
    >= '0' and <= '9' => "숫자",
    ' ' or '\t' or '\n' or '\r' => "공백",
    '.' or ',' or ';' or ':' or '!' or '?' => "구두점",
    _ => "기타"
};

Console.WriteLine(ClassifyCharacter('g')); // 소문자
Console.WriteLine(ClassifyCharacter('5')); // 숫자
Console.WriteLine(ClassifyCharacter('!')); // 구두점
```

## 7. 리스트 패턴 (C# 11)

리스트 패턴(list pattern)은 시퀀스(배열, 리스트, 스팬)를 형태 — 특정 요소, 길이, 슬라이스 — 로 매칭합니다.

### 7.1 기본 리스트 패턴

```csharp
int[] numbers = { 1, 2, 3, 4, 5 };

// 정확한 매칭
bool isOneTwoThree = numbers is [1, 2, 3]; // false (5개 요소, 3개가 아님)

// 1로 시작하고 5로 끝나는 5개 요소 배열 매칭
bool matchShape = numbers is [1, _, _, _, 5]; // true

// "아무 요소" 디스카드 패턴
bool startsWithOne = numbers is [1, ..]; // true

// 슬라이스 패턴(..)은 0개 이상의 요소를 매칭
bool endsWithFive = numbers is [.., 5]; // true

// 빈 배열
int[] empty = Array.Empty<int>();
bool isEmpty = empty is []; // true
```

### 7.2 리스트 패턴에서의 변수 바인딩

```csharp
static string DescribeArray(int[] arr) => arr switch
{
    [] => "비어있음",
    [var single] => $"단일: {single}",
    [var first, var second] => $"쌍: {first}, {second}",
    [var first, .., var last] => $"{first}부터 {last}까지의 배열 ({arr.Length}개 요소)",
};

Console.WriteLine(DescribeArray(Array.Empty<int>()));     // 비어있음
Console.WriteLine(DescribeArray(new[] { 42 }));           // 단일: 42
Console.WriteLine(DescribeArray(new[] { 1, 2 }));         // 쌍: 1, 2
Console.WriteLine(DescribeArray(new[] { 1, 2, 3, 4, 5 })); // 1부터 5까지의 배열 (5개 요소)
```

### 7.3 리스트에서의 중첩 패턴

```csharp
static string AnalyzeSequence(int[] seq) => seq switch
{
    [> 0, > 0, > 0] => "세 개의 양수",
    [< 0, .., < 0] => "음수로 시작하고 음수로 끝남",
    [0, ..] => "0으로 시작",
    [_, _, _, ..] when seq.All(x => x % 2 == 0) => "3개 이상의 짝수",
    _ => "기타"
};

Console.WriteLine(AnalyzeSequence(new[] { 1, 2, 3 }));      // 세 개의 양수
Console.WriteLine(AnalyzeSequence(new[] { -1, 5, -3 }));     // 음수로 시작하고 음수로 끝남
Console.WriteLine(AnalyzeSequence(new[] { 0, 7, 8 }));       // 0으로 시작
```

### 7.4 변수 캡처가 있는 슬라이스 패턴

```csharp
// 슬라이스를 변수에 캡처 (C# 11)
static string ProcessCommand(string[] args) => args switch
{
    ["help"] => "도움말 표시 중...",
    ["version"] => "v1.0.0",
    ["add", var item] => $"추가 중: {item}",
    ["remove", var item] => $"제거 중: {item}",
    ["search", .. var terms] => $"검색 중: {string.Join(" ", terms)}",
    [var cmd, ..] => $"알 수 없는 명령: {cmd}",
    [] => "명령이 지정되지 않음"
};

Console.WriteLine(ProcessCommand(new[] { "help" }));
// 도움말 표시 중...
Console.WriteLine(ProcessCommand(new[] { "add", "milk" }));
// 추가 중: milk
Console.WriteLine(ProcessCommand(new[] { "search", "C#", "patterns" }));
// 검색 중: C# patterns
```

## 8. Var 패턴과 디스카드 패턴

### 8.1 Var 패턴

`var` 패턴은 항상 매칭되며 값을 새 변수에 바인딩합니다. 중간 값을 캡처하려는 복잡한 패턴 내부에서 유용합니다.

```csharp
static string AnalyzeOrder(decimal amount, string country) => (amount, country) switch
{
    ( > 1000, "US") => "대형 미국 주문 — 무료 배송",
    ( > 500, "US") => "중형 미국 주문 — 배송비 할인",
    (var a, "US") when a > 0 => $"소형 미국 주문 (${a})",
    (var a, var c) when a > 0 => $"{c}으로의 국제 주문 (${a})",
    _ => "유효하지 않은 주문"
};
```

### 8.2 디스카드 패턴

`_` 디스카드 패턴은 모든 것에 매칭되며 값을 버립니다. "모든 것 포착"이나 자리 표시자 역할을 합니다.

```csharp
// switch 표현식에서 — 기본 분기
var result = input switch
{
    1 => "one",
    2 => "two",
    _ => "기타"  // 디스카드 — 모든 것에 매칭
};

// 위치 패턴에서 — 구성 요소 무시
static bool IsOnAxis(Point3D p) => p is (_, 0, 0) or (0, _, 0) or (0, 0, _);
```

## 9. Switch 표현식 — 종합 정리

switch 표현식은 패턴을 사용하는 가장 강력한 방법입니다. 문이 아닌 식이므로 값을 반환합니다.

### 9.1 완전성

컴파일러는 switch 표현식이 완전한지 — 모든 가능한 입력이 처리되는지 검사합니다. 커버리지가 불완전하면 경고가 생성됩니다.

```csharp
// 열거형 — 컴파일러가 모든 가능한 값을 앎
enum Season { Spring, Summer, Autumn, Winter }

static string Describe(Season s) => s switch
{
    Season.Spring => "꽃이 피는 계절",
    Season.Summer => "햇살이 비치는 계절",
    Season.Autumn => "낙엽이 지는 계절",
    Season.Winter => "눈이 내리는 계절",
    // _ 필요 없음 — 모든 열거형 값이 커버됨
};
```

### 9.2 `when`을 사용한 가드

임의의 불리언 조건으로 패턴을 세분화하기 위해 `when` 절을 추가합니다.

```csharp
record Order(string Product, int Quantity, decimal UnitPrice)
{
    public decimal Total => Quantity * UnitPrice;
}

static string ClassifyOrder(Order order) => order switch
{
    { Quantity: <= 0 } => "유효하지 않음: 수량이 0 이하",
    { UnitPrice: <= 0 } => "유효하지 않음: 가격이 0 이하",
    { Total: > 10_000 } when order.Product.StartsWith("PREMIUM")
        => "VIP 프리미엄 주문 — 전담 담당자 배정",
    { Total: > 10_000 } => "대형 주문 — 영업부 알림",
    { Total: > 1_000 } => "중형 주문",
    _ => "일반 주문"
};
```

### 9.3 중첩 패턴 조합

```csharp
record Customer(string Name, string Tier, Address Address);

static decimal CalculateDiscount(Customer c, Order o) => (c, o) switch
{
    ({ Tier: "Gold" }, { Total: > 5000 }) => 0.20m,
    ({ Tier: "Gold" }, _) => 0.15m,
    ({ Tier: "Silver" }, { Total: > 5000 }) => 0.12m,
    ({ Tier: "Silver" }, _) => 0.08m,
    ({ Address.State: "WA" }, { Total: > 1000 }) => 0.05m, // 지역 로열티
    _ => 0.0m
};
```

## 10. 실전 예제: 식 평가기

재귀 데이터 구조에 패턴 매칭을 사용하여 간단한 산술 식 평가기를 만들어 보겠습니다.

### 10.1 식 트리

```csharp
public abstract record Expr;
public record Num(double Value) : Expr;
public record Add(Expr Left, Expr Right) : Expr;
public record Sub(Expr Left, Expr Right) : Expr;
public record Mul(Expr Left, Expr Right) : Expr;
public record Div(Expr Left, Expr Right) : Expr;
public record Neg(Expr Operand) : Expr;
public record Var(string Name) : Expr;
```

### 10.2 평가기

```csharp
public static class ExprEvaluator
{
    public static double Evaluate(Expr expr, Dictionary<string, double>? vars = null) =>
        expr switch
        {
            Num(var v) => v,
            Add(var l, var r) => Evaluate(l, vars) + Evaluate(r, vars),
            Sub(var l, var r) => Evaluate(l, vars) - Evaluate(r, vars),
            Mul(var l, var r) => Evaluate(l, vars) * Evaluate(r, vars),
            Div(var l, Num(0)) => throw new DivideByZeroException(),
            Div(var l, var r) => Evaluate(l, vars) / Evaluate(r, vars),
            Neg(var operand) => -Evaluate(operand, vars),
            Var(var name) when vars?.ContainsKey(name) == true => vars[name],
            Var(var name) => throw new ArgumentException($"정의되지 않은 변수: {name}"),
            _ => throw new ArgumentException($"알 수 없는 식: {expr}")
        };

    public static string PrettyPrint(Expr expr) => expr switch
    {
        Num(var v) => v.ToString("G"),
        Var(var name) => name,
        Neg(var op) => $"-({PrettyPrint(op)})",
        Add(var l, var r) => $"({PrettyPrint(l)} + {PrettyPrint(r)})",
        Sub(var l, var r) => $"({PrettyPrint(l)} - {PrettyPrint(r)})",
        Mul(var l, var r) => $"({PrettyPrint(l)} * {PrettyPrint(r)})",
        Div(var l, var r) => $"({PrettyPrint(l)} / {PrettyPrint(r)})",
        _ => "?"
    };

    // 패턴 매칭을 사용한 단순화기
    public static Expr Simplify(Expr expr) => expr switch
    {
        // x + 0 = x, 0 + x = x
        Add(var x, Num(0)) => Simplify(x),
        Add(Num(0), var x) => Simplify(x),

        // x * 1 = x, 1 * x = x
        Mul(var x, Num(1)) => Simplify(x),
        Mul(Num(1), var x) => Simplify(x),

        // x * 0 = 0, 0 * x = 0
        Mul(_, Num(0)) => new Num(0),
        Mul(Num(0), _) => new Num(0),

        // x - 0 = x
        Sub(var x, Num(0)) => Simplify(x),

        // --x = x
        Neg(Neg(var x)) => Simplify(x),

        // 상수 접기
        Add(Num(var a), Num(var b)) => new Num(a + b),
        Sub(Num(var a), Num(var b)) => new Num(a - b),
        Mul(Num(var a), Num(var b)) => new Num(a * b),
        Div(Num(var a), Num(var b)) when b != 0 => new Num(a / b),

        // 하위 식으로 재귀
        Add(var l, var r) => new Add(Simplify(l), Simplify(r)),
        Sub(var l, var r) => new Sub(Simplify(l), Simplify(r)),
        Mul(var l, var r) => new Mul(Simplify(l), Simplify(r)),
        Div(var l, var r) => new Div(Simplify(l), Simplify(r)),
        Neg(var op) => new Neg(Simplify(op)),

        _ => expr
    };
}
```

### 10.3 평가기 사용

```csharp
// 구성: (3 + x) * (2 - 1)
Expr expr = new Mul(
    new Add(new Num(3), new Var("x")),
    new Sub(new Num(2), new Num(1))
);

Console.WriteLine(ExprEvaluator.PrettyPrint(expr));
// ((3 + x) * (2 - 1))

var vars = new Dictionary<string, double> { ["x"] = 7 };
Console.WriteLine(ExprEvaluator.Evaluate(expr, vars)); // 10

// 단순화: (2 - 1)이 Num(1)이 되고, (3 + x) * 1이 (3 + x)가 됨
var simplified = ExprEvaluator.Simplify(expr);
Console.WriteLine(ExprEvaluator.PrettyPrint(simplified)); // (3 + x)

// 다른 단순화: 0 + (5 * 1) -> 5
Expr expr2 = new Add(new Num(0), new Mul(new Num(5), new Num(1)));
var s2 = ExprEvaluator.Simplify(expr2);
Console.WriteLine(ExprEvaluator.PrettyPrint(s2)); // 5
```

## 11. 연습 문제

1. **패턴을 사용한 FizzBuzz**: 튜플 `(n % 3, n % 5)`에 대한 단일 switch 표현식과 상수 패턴을 사용하여 FizzBuzz(1~100)를 다시 작성하세요. "Fizz", "Buzz", "FizzBuzz" 또는 숫자를 출력하세요.

2. **JSON 유사 값 분류기**: JSON 값 타입의 구별 합집합을 정의하세요: `JsonNull`, `JsonBool(bool Value)`, `JsonNumber(double Value)`, `JsonString(string Value)`, `JsonArray(List<JsonValue> Items)`, `JsonObject(Dictionary<string, JsonValue> Properties)`. 패턴 매칭을 사용하여 배열과 객체에 대한 중첩 깊이를 포함하는 사람이 읽을 수 있는 설명을 반환하는 `Describe(JsonValue v)` 메서드를 작성하세요.

3. **리스트 패턴을 사용한 명령 파서**: `string[]` 명령줄을 리스트 패턴으로 파싱하세요. 지원: `["git", "commit", "-m", var message]`, `["git", "push", var remote, var branch]`, `["git", "log", "--oneline", "-n", var count]`, 그리고 `["git", .. var rest]`를 포괄 처리로. 파싱된 각 명령을 설명하는 레코드를 반환하세요.

4. **온도 변환기**: `(double Value, string Unit)` 튜플을 받아 위치 패턴 + 상수 패턴을 사용하여 섭씨, 화씨, 켈빈 간 변환하는 메서드를 작성하세요. 유효하지 않은 단위에 대해 명확한 오류 메시지를 처리하세요.

5. **이진 트리 깊이**: `Leaf(int Value)`와 `Branch(Tree Left, Tree Right)`를 가진 재귀 `record Tree`를 정의하세요. 패턴 매칭을 사용하여 `MaxDepth(Tree t)`와 `Sum(Tree t)`을 작성하세요. 그런 다음 트리의 거울 이미지를 반환하는 `Mirror(Tree t)`를 작성하세요.
