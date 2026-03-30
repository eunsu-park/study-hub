# 메서드

**이전**: [제어 흐름](./04_Control_Flow.md) | **다음**: [배열과 문자열](./06_Arrays_and_Strings.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 다양한 매개변수 타입으로 메서드를 선언하고 호출한다
2. `ref`, `out`, `in`, `params` 매개변수 한정자를 사용한다
3. 선택적 매개변수를 정의하고 명명된 인수(Named Argument)를 사용한다
4. 다른 매개변수 시그니처로 메서드를 오버로딩(Overloading)한다
5. 간결한 식 본문 메서드(Expression-Bodied Method)를 작성한다
6. 로컬 함수(Local Function)를 생성하고 사용한다
7. 재귀 알고리즘을 구현한다
8. 정적 메서드와 인스턴스 메서드를 구별한다
9. 튜플 반환 타입으로 여러 값을 반환한다

---

**메서드(Method)**는 특정 작업을 수행하는 명명된 코드 블록입니다. 메서드는 C#에서 코드를 구성하고 재사용하는 주요 방법입니다. 메서드는 입력(매개변수)을 받아 연산을 수행하고, 선택적으로 결과를 반환합니다. 잘 설계된 메서드는 프로그램을 읽기, 테스트, 유지보수하기 쉽게 만듭니다.

## 1. 메서드 선언과 호출

### 1.1 기본 메서드 구조

```csharp
// 메서드 선언
// <접근 한정자> <반환 타입> <이름>(<매개변수>)
// {
//     <본문>
//     return <값>; // void가 아닌 경우
// }

void SayHello()
{
    Console.WriteLine("Hello!");
}

int Add(int a, int b)
{
    return a + b;
}

string FormatName(string first, string last)
{
    return $"{last}, {first}";
}

// 메서드 호출
SayHello();                         // Hello!
int sum = Add(3, 5);                // 8
string name = FormatName("John", "Doe");  // "Doe, John"
Console.WriteLine($"합계: {sum}, 이름: {name}");
```

### 1.2 반환 타입

```csharp
// void — 반환 값 없음
void PrintLine(string text)
{
    Console.WriteLine($">>> {text}");
}

// 특정 반환 타입
double CalculateArea(double radius)
{
    return Math.PI * radius * radius;
}

// 불리언 반환
bool IsEven(int number)
{
    return number % 2 == 0;
}

// 가드 절을 위한 조기 반환
double SafeDivide(double a, double b)
{
    if (b == 0)
    {
        Console.WriteLine("경고: 0으로 나누기");
        return 0;  // 조기 반환
    }
    return a / b;
}

Console.WriteLine($"면적: {CalculateArea(5):F2}");    // 면적: 78.54
Console.WriteLine($"IsEven(7): {IsEven(7)}");          // IsEven(7): False
Console.WriteLine($"10 / 3 = {SafeDivide(10, 3):F2}"); // 10 / 3 = 3.33
```

### 1.3 다른 컨텍스트에서 표현식으로 사용되는 메서드

```csharp
// 메서드는 반환 타입의 표현식이 기대되는 모든 곳에서 사용 가능
int max = Math.Max(Add(1, 2), Add(3, 4));
Console.WriteLine(max);  // 7

// 체이닝
string result = FormatName("Jane", "Smith").ToUpper();
Console.WriteLine(result);  // SMITH, JANE

// 문자열 보간에서
Console.WriteLine($"원의 면적: {CalculateArea(3):F4}");
```

## 2. 매개변수 전달: 값, ref, out, in

매개변수가 전달되는 방식을 이해하는 것은 매우 중요합니다. 기본적으로 C#은 매개변수를 **값으로 전달(Pass by Value)**합니다. 즉, 메서드는 복사본을 받습니다.

### 2.1 값에 의한 전달 (기본)

```csharp
void TryToChange(int x)
{
    x = 999;  // 로컬 복사본만 변경
    Console.WriteLine($"메서드 내부: x = {x}");
}

int number = 42;
TryToChange(number);
Console.WriteLine($"메서드 이후: number = {number}");
// 출력:
// 메서드 내부: x = 999
// 메서드 이후: number = 42  (변경되지 않음!)
```

### 2.2 참조에 의한 전달: ref

`ref` 키워드는 변수의 참조를 전달하여 메서드가 원래 값을 수정할 수 있게 합니다:

```csharp
void Swap(ref int a, ref int b)
{
    int temp = a;
    a = b;
    b = temp;
}

int x = 10, y = 20;
Console.WriteLine($"이전: x={x}, y={y}");  // 이전: x=10, y=20
Swap(ref x, ref y);
Console.WriteLine($"이후: x={x}, y={y}");  // 이후: x=20, y=10

// ref 매개변수는 전달 전에 초기화되어야 함
// int uninitialized;
// Swap(ref uninitialized, ref y);  // 컴파일 에러: 할당되지 않은 변수 사용

// 실용 예제: 증가 후 반환
void IncrementBy(ref int value, int amount)
{
    value += amount;
}

int counter = 0;
IncrementBy(ref counter, 5);
IncrementBy(ref counter, 3);
Console.WriteLine($"카운터: {counter}");  // 8
```

### 2.3 출력 매개변수: out

`out` 키워드는 `ref`와 유사하지만 출력 전용입니다. 변수는 전달 전에 초기화할 필요가 없으며, 메서드는 반환 전에 반드시 값을 **할당해야** 합니다:

```csharp
// 전통적인 out 사용: 여러 값 반환
bool TryDivide(int dividend, int divisor, out int quotient, out int remainder)
{
    if (divisor == 0)
    {
        quotient = 0;
        remainder = 0;
        return false;
    }
    quotient = dividend / divisor;
    remainder = dividend % divisor;
    return true;
}

if (TryDivide(17, 5, out int q, out int r))
{
    Console.WriteLine($"17 / 5 = {q} 나머지 {r}");  // 17 / 5 = 3 나머지 2
}

// 인라인 out 변수 선언 (C# 7+)
if (int.TryParse("123", out int parsed))
{
    Console.WriteLine($"파싱됨: {parsed}");  // 파싱됨: 123
}

// 필요 없는 out 값 폐기
if (int.TryParse("456", out _))
{
    Console.WriteLine("유효한 숫자입니다");
}

// 복잡한 데이터와 함께 out 매개변수
void GetMinMax(int[] array, out int min, out int max)
{
    min = int.MaxValue;
    max = int.MinValue;
    foreach (int val in array)
    {
        if (val < min) min = val;
        if (val > max) max = val;
    }
}

int[] data = { 5, 2, 8, 1, 9, 3 };
GetMinMax(data, out int minimum, out int maximum);
Console.WriteLine($"최솟값: {minimum}, 최댓값: {maximum}");  // 최솟값: 1, 최댓값: 9
```

### 2.4 읽기 전용 참조: in

`in` 키워드는 참조로 전달하지만 메서드가 값을 수정하는 것을 방지합니다. 큰 구조체에 대한 최적화입니다:

```csharp
// in은 수정을 방지 — 큰 구조체에 유용
double CalculateDistance(in (double X, double Y) p1, in (double X, double Y) p2)
{
    // p1.X = 0;  // 컴파일 에러: 'in' 매개변수를 수정할 수 없음
    double dx = p2.X - p1.X;
    double dy = p2.Y - p1.Y;
    return Math.Sqrt(dx * dx + dy * dy);
}

var point1 = (X: 0.0, Y: 0.0);
var point2 = (X: 3.0, Y: 4.0);
double dist = CalculateDistance(in point1, in point2);
Console.WriteLine($"거리: {dist}");  // 거리: 5

// 호출 시 'in' 키워드는 선택적
double dist2 = CalculateDistance(point1, point2);  // 역시 작동
```

### 2.5 비교 표

```csharp
// 매개변수 한정자 요약:
// ──────────────────────────────────────────────────────
// 한정자    | 방향     | 전달 전 초기화? | 수정 가능?
// ──────────────────────────────────────────────────────
// (없음)    | 입력     | 예              | 복사본만
// ref       | 입출력   | 예              | 예
// out       | 출력     | 아니오          | 반드시 할당
// in        | 입력     | 예              | 아니오
// ──────────────────────────────────────────────────────
```

## 3. params: 가변 개수 인수

`params` 키워드를 사용하면 메서드가 지정된 타입의 인수를 원하는 수만큼 받을 수 있습니다:

```csharp
// params는 마지막 매개변수여야 하며 배열 타입이어야 함
int Sum(params int[] numbers)
{
    int total = 0;
    foreach (int n in numbers)
    {
        total += n;
    }
    return total;
}

// 원하는 수만큼 인수로 호출
Console.WriteLine(Sum());              // 0
Console.WriteLine(Sum(1));             // 1
Console.WriteLine(Sum(1, 2, 3));       // 6
Console.WriteLine(Sum(1, 2, 3, 4, 5)); // 15

// 배열을 직접 전달할 수도 있음
int[] values = { 10, 20, 30 };
Console.WriteLine(Sum(values));        // 60

// 다른 매개변수와 함께 params
string JoinWithSeparator(string separator, params string[] items)
{
    return string.Join(separator, items);
}

Console.WriteLine(JoinWithSeparator(", ", "apple", "banana", "cherry"));
// 출력: apple, banana, cherry

// 실용 예제: 가변 인수 로깅
void Log(string level, string message, params object[] args)
{
    string formatted = string.Format(message, args);
    Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] [{level}] {formatted}");
}

Log("INFO", "사용자 {0}이(가) {1}에서 로그인", "Alice", "192.168.1.1");
Log("ERROR", "{0}개 항목 처리 실패", 42);
```

## 4. 선택적 매개변수와 명명된 인수

### 4.1 선택적 매개변수

기본값이 있는 매개변수는 선택적입니다 — 호출자가 생략할 수 있습니다:

```csharp
void PrintMessage(string message, int repeat = 1, bool uppercase = false)
{
    string output = uppercase ? message.ToUpper() : message;
    for (int i = 0; i < repeat; i++)
    {
        Console.WriteLine(output);
    }
}

PrintMessage("Hello");                         // Hello (한 번, 소문자)
PrintMessage("Hello", 3);                      // Hello Hello Hello
PrintMessage("Hello", 2, true);               // HELLO HELLO
PrintMessage("Hello", uppercase: true);        // HELLO (명명된 인수로 'repeat' 건너뛰기)

// 선택적 매개변수는 필수 매개변수 뒤에 와야 함
string CreateGreeting(string name, string prefix = "Mr.", string suffix = "")
{
    return $"{prefix} {name}{(suffix.Length > 0 ? $", {suffix}" : "")}";
}

Console.WriteLine(CreateGreeting("Smith"));                    // Mr. Smith
Console.WriteLine(CreateGreeting("Smith", "Dr."));             // Dr. Smith
Console.WriteLine(CreateGreeting("Smith", "Dr.", "PhD"));      // Dr. Smith, PhD
```

### 4.2 명명된 인수

명명된 인수를 사용하면 이름으로 매개변수를 지정할 수 있으며, 순서에 관계없이 사용할 수 있습니다:

```csharp
void CreateUser(string name, int age, string email, string role = "user")
{
    Console.WriteLine($"이름: {name}, 나이: {age}, 이메일: {email}, 역할: {role}");
}

// 위치 기반 (순서대로여야 함)
CreateUser("Alice", 30, "alice@example.com");

// 이름 기반 (어떤 순서든 가능)
CreateUser(email: "bob@example.com", name: "Bob", age: 25);

// 혼합 (위치 기반 먼저, 그 다음 이름 기반)
CreateUser("Charlie", age: 35, email: "charlie@example.com", role: "admin");

// 많은 선택적 매개변수가 있을 때 명명된 인수가 특히 유용
void Configure(
    string host = "localhost",
    int port = 8080,
    bool ssl = false,
    int timeout = 30,
    int maxRetries = 3)
{
    Console.WriteLine($"Host: {host}:{port}, SSL: {ssl}, Timeout: {timeout}s, Retries: {maxRetries}");
}

Configure(port: 443, ssl: true);  // 기본값과 다른 것만 지정
Configure(maxRetries: 5, timeout: 60);
```

## 5. 메서드 오버로딩

메서드 오버로딩(Overloading)은 같은 이름이지만 다른 매개변수 목록을 가진 여러 메서드를 허용합니다:

```csharp
// 같은 이름, 다른 매개변수 타입
int Multiply(int a, int b)
{
    Console.WriteLine("int * int");
    return a * b;
}

double Multiply(double a, double b)
{
    Console.WriteLine("double * double");
    return a * b;
}

string Multiply(string text, int count)
{
    Console.WriteLine("string * int");
    return string.Concat(Enumerable.Repeat(text, count));
}

Console.WriteLine(Multiply(3, 4));          // int * int → 12
Console.WriteLine(Multiply(3.5, 2.0));      // double * double → 7
Console.WriteLine(Multiply("Ha", 3));       // string * int → HaHaHa

// 같은 이름, 다른 매개변수 수
double Average(double a, double b)
{
    return (a + b) / 2;
}

double Average(double a, double b, double c)
{
    return (a + b + c) / 3;
}

double Average(params double[] values)
{
    return values.Length > 0 ? values.Average() : 0;
}

Console.WriteLine(Average(10.0, 20.0));          // 2개 매개변수 버전: 15
Console.WriteLine(Average(10.0, 20.0, 30.0));    // 3개 매개변수 버전: 20
Console.WriteLine(Average(1.0, 2.0, 3.0, 4.0)); // params 버전: 2.5
```

### 5.1 오버로딩 규칙

```csharp
// 오버로딩은 반환 타입이 아닌 매개변수 목록에 기반
// 다음은 유효한 오버로드:
void Process(int x) { }
void Process(string x) { }
void Process(int x, int y) { }

// 이것은 유효한 오버로드가 아님 (같은 매개변수 목록, 다른 반환 타입):
// int Process(int x) { return x; }  // 컴파일 에러: 이미 정의됨

// ref, out, in은 서로 다른 시그니처로 간주됨
void Transform(ref int x) { x *= 2; }
void Transform(out int x) { x = 42; }
// 하지만 ref와 out은 서로 오버로드할 수 없음:
// void Transform(ref int x) { }  // Transform(out int x)가 있으면 컴파일 에러
```

## 6. 식 본문 메서드

단일 표현식으로 구성된 간단한 메서드에는 `=>` 화살표 구문을 사용합니다:

```csharp
// 전통적인 메서드 본문
int Square(int x)
{
    return x * x;
}

// 식 본문 동등물 (더 짧음)
int SquareExpr(int x) => x * x;

// 추가 예제
double CircleArea(double r) => Math.PI * r * r;
bool IsPositive(int n) => n > 0;
string Greet(string name) => $"Hello, {name}!";
int Max(int a, int b) => a > b ? a : b;
void PrintStars(int count) => Console.WriteLine(new string('*', count));

// 식 본문 메서드 사용
Console.WriteLine(SquareExpr(7));          // 49
Console.WriteLine(CircleArea(5));          // 78.539...
Console.WriteLine(IsPositive(-3));         // False
Console.WriteLine(Greet("World"));         // Hello, World!
Console.WriteLine(Max(10, 20));            // 20
PrintStars(10);                             // **********
```

## 7. 로컬 함수

로컬 함수(Local Function)는 다른 메서드 내부에 정의된 메서드입니다. 한 곳에서만 의미 있는 헬퍼 로직에 유용합니다:

```csharp
// 유효성 검사를 위한 로컬 함수
void ProcessOrder(string productId, int quantity)
{
    // 로컬 함수 — ProcessOrder 외부에서는 보이지 않음
    bool IsValid()
    {
        if (string.IsNullOrEmpty(productId)) return false;
        if (quantity <= 0) return false;
        if (quantity > 1000) return false;
        return true;
    }

    if (!IsValid())
    {
        Console.WriteLine("잘못된 주문입니다.");
        return;
    }

    Console.WriteLine($"처리 중: {quantity}x {productId}");
}

ProcessOrder("SKU-123", 5);    // 처리 중: 5x SKU-123
ProcessOrder("", 5);            // 잘못된 주문입니다.
ProcessOrder("SKU-123", -1);   // 잘못된 주문입니다.

// 로컬 함수는 둘러싸는 스코프의 변수를 캡처할 수 있음
int[] FilterAndSum(int[] numbers, int threshold)
{
    var filtered = new List<int>();
    int sum = 0;

    void Accumulate(int value)
    {
        // 둘러싸는 스코프의 'threshold'에 접근
        if (value > threshold)
        {
            filtered.Add(value);
            sum += value;
        }
    }

    foreach (int n in numbers)
    {
        Accumulate(n);
    }

    Console.WriteLine($"{threshold}보다 큰 값의 합: {sum}");
    return filtered.ToArray();
}

int[] result = FilterAndSum(new[] { 1, 5, 3, 8, 2, 7, 4 }, 4);
Console.WriteLine($"필터링됨: [{string.Join(", ", result)}]");
// 출력: 4보다 큰 값의 합: 20
// 출력: 필터링됨: [5, 8, 7]

// 정적 로컬 함수 (C# 8+) — 둘러싸는 변수를 캡처할 수 없음
int Calculate(int x, int y)
{
    return AddAndDouble(x, y);

    // 정적 로컬 함수 — 모든 데이터를 매개변수로 받아야 함
    static int AddAndDouble(int a, int b) => (a + b) * 2;
}

Console.WriteLine(Calculate(3, 4));  // 14
```

## 8. 재귀

재귀 메서드(Recursive Method)는 자기 자신을 호출합니다. 모든 재귀 메서드에는 무한 재귀를 방지하기 위한 **기저 조건(Base Case, 종료 조건)**이 필요합니다.

### 8.1 고전적인 재귀 예제

```csharp
// 팩토리얼: n! = n * (n-1) * ... * 1
long Factorial(int n)
{
    if (n <= 1) return 1;       // 기저 조건
    return n * Factorial(n - 1); // 재귀 조건
}

Console.WriteLine($"5! = {Factorial(5)}");    // 120
Console.WriteLine($"10! = {Factorial(10)}");  // 3628800

// 피보나치: F(n) = F(n-1) + F(n-2)
int Fibonacci(int n)
{
    if (n <= 0) return 0;       // 기저 조건
    if (n == 1) return 1;       // 기저 조건
    return Fibonacci(n - 1) + Fibonacci(n - 2);  // 재귀 조건
}

for (int i = 0; i <= 10; i++)
{
    Console.Write($"{Fibonacci(i)} ");
}
Console.WriteLine();
// 출력: 0 1 1 2 3 5 8 13 21 34 55
```

### 8.2 거듭제곱과 최대공약수

```csharp
// 거듭제곱: base^exponent
double Power(double baseVal, int exponent)
{
    if (exponent == 0) return 1;
    if (exponent < 0) return 1.0 / Power(baseVal, -exponent);
    return baseVal * Power(baseVal, exponent - 1);
}

Console.WriteLine($"2^10 = {Power(2, 10)}");    // 1024
Console.WriteLine($"3^-2 = {Power(3, -2):F4}"); // 0.1111

// 유클리드 알고리즘을 사용한 최대공약수
int Gcd(int a, int b)
{
    if (b == 0) return a;
    return Gcd(b, a % b);
}

Console.WriteLine($"GCD(48, 18) = {Gcd(48, 18)}");  // 6
Console.WriteLine($"GCD(100, 75) = {Gcd(100, 75)}"); // 25
```

### 8.3 이진 탐색 (재귀)

```csharp
int BinarySearch(int[] sorted, int target, int low, int high)
{
    if (low > high) return -1;  // 기저 조건: 찾지 못함

    int mid = low + (high - low) / 2;

    if (sorted[mid] == target) return mid;
    if (sorted[mid] < target) return BinarySearch(sorted, target, mid + 1, high);
    return BinarySearch(sorted, target, low, mid - 1);
}

int[] array = { 2, 5, 8, 12, 16, 23, 38, 56, 72, 91 };
int index = BinarySearch(array, 23, 0, array.Length - 1);
Console.WriteLine($"인덱스 {index}에서 23을 찾음");  // 인덱스 5에서 23을 찾음
```

### 8.4 재귀 vs 반복

```csharp
// 재귀적 합 (단순하지만 큰 n에서 스택 오버플로 가능)
long SumRecursive(int n)
{
    if (n <= 0) return 0;
    return n + SumRecursive(n - 1);
}

// 반복적 합 (스택 오버플로 위험 없음)
long SumIterative(int n)
{
    long total = 0;
    for (int i = 1; i <= n; i++)
        total += i;
    return total;
}

// 수학 공식 (최고 성능)
long SumFormula(int n) => (long)n * (n + 1) / 2;

Console.WriteLine(SumRecursive(100));  // 5050
Console.WriteLine(SumIterative(100));  // 5050
Console.WriteLine(SumFormula(100));    // 5050
```

## 9. 정적 메서드 vs 인스턴스 메서드 (미리보기)

이것은 미리보기입니다 — 클래스는 레슨 9에서 자세히 다룹니다.

```csharp
class Calculator
{
    // 인스턴스 필드
    private int _memory = 0;

    // 인스턴스 메서드 — 객체가 필요 (인스턴스 데이터에 접근)
    public void Store(int value)
    {
        _memory = value;
    }

    public int Recall()
    {
        return _memory;
    }

    // 정적 메서드 — 객체가 필요 없음 (인스턴스 데이터 없음)
    public static int Add(int a, int b)
    {
        return a + b;
    }

    public static double SquareRoot(double x)
    {
        return Math.Sqrt(x);
    }
}

// 정적 메서드: 클래스 자체에서 호출
int sum = Calculator.Add(3, 5);
double sqrt = Calculator.SquareRoot(16);
Console.WriteLine($"Add: {sum}, Sqrt: {sqrt}");

// 인스턴스 메서드: 객체에서 호출
var calc = new Calculator();
calc.Store(42);
Console.WriteLine($"리콜: {calc.Recall()}");  // 42
```

## 10. 튜플 반환 타입

튜플(Tuple)을 사용하면 사용자 정의 클래스를 정의하지 않고도 메서드에서 여러 값을 반환할 수 있습니다:

```csharp
// 이름 없는 튜플
(int, int) Divide(int dividend, int divisor)
{
    return (dividend / divisor, dividend % divisor);
}

var result = Divide(17, 5);
Console.WriteLine($"몫: {result.Item1}, 나머지: {result.Item2}");

// 명명된 튜플 (가독성이 훨씬 좋음)
(int Quotient, int Remainder) DivideNamed(int dividend, int divisor)
{
    return (dividend / divisor, dividend % divisor);
}

var namedResult = DivideNamed(17, 5);
Console.WriteLine($"몫: {namedResult.Quotient}, 나머지: {namedResult.Remainder}");

// 분해(Deconstruction) — 튜플 요소를 개별 변수로 추출
var (q, r) = DivideNamed(17, 5);
Console.WriteLine($"q={q}, r={r}");

// 더 복잡한 예제: 통계
(double Mean, double Min, double Max, int Count) GetStats(params double[] values)
{
    if (values.Length == 0)
        return (0, 0, 0, 0);

    double sum = 0, min = double.MaxValue, max = double.MinValue;
    foreach (double v in values)
    {
        sum += v;
        if (v < min) min = v;
        if (v > max) max = v;
    }
    return (sum / values.Length, min, max, values.Length);
}

var stats = GetStats(4.5, 2.1, 8.3, 1.7, 6.9);
Console.WriteLine($"평균: {stats.Mean:F2}, 최솟값: {stats.Min}, 최댓값: {stats.Max}, 개수: {stats.Count}");
// 출력: 평균: 4.70, 최솟값: 1.7, 최댓값: 8.3, 개수: 5

// 불필요한 튜플 요소 폐기
var (mean, _, _, count) = GetStats(1, 2, 3, 4, 5);
Console.WriteLine($"{count}개 값의 평균: {mean}");

// 튜플 vs out 매개변수
// 튜플: 더 깔끔한 구문, 명명된 필드, 쉬운 분해
// out: 더 전통적, TryParse 패턴과 함께 사용
```

## 11. 연습 문제

1. **온도 변환기**: `Convert`라는 오버로딩된 메서드 세 개를 작성하세요: 섭씨를 화씨로 변환하는 것(`double` 받음), 화씨를 섭씨로 변환하는 것(`double`과 `bool` 플래그 `toFahrenheit = true` 받음), 온도 배열을 변환하는 것. 공식: F = C * 9/5 + 32, C = (F - 32) * 5/9를 사용하세요.

2. **문자열 유틸리티**: 가능한 경우 식 본문 구문을 사용하여 다음 메서드를 작성하세요: (a) `Reverse(string s)` — 뒤집힌 문자열 반환, (b) `IsPalindrome(string s)` — 문자열이 앞뒤로 같게 읽히는지 확인 (대소문자 무시), (c) `CountVowels(string s)` — 모음의 수 반환, (d) `Truncate(string s, int maxLength, string ellipsis = "...")` — 선택적 줄임표로 잘라내기.

3. **재귀 하노이의 탑**: 하노이의 탑 퍼즐을 재귀적으로 구현하세요. 메서드는 디스크의 수와 세 개의 기둥 이름(출발, 보조, 도착)을 받아야 합니다. 각 이동을 출력하세요. 디스크 3개는 7번, 4개는 15번의 이동이 필요함을 확인하세요.

4. **튜플을 사용한 통계**: 명명된 튜플을 반환하는 `Analyze(params int[] numbers)` 메서드를 작성하세요: `(int Sum, double Average, int Min, int Max, int Range, double Variance)`. 분산 계산에 로컬 함수를 사용하세요. 최소 세 가지 다른 데이터셋으로 테스트하세요.

5. **Ref/Out 스왑과 파싱**: (a) 두 `string` 변수를 교환하는 `ref`를 사용한 제네릭과 유사한 메서드를 작성하세요. (b) `"3.5, 4.2"`와 같은 문자열을 두 좌표로 파싱하는 `TryParsePoint(string input, out double x, out double y)` 메서드를 작성하세요. 성공 시 `true`를 반환합니다. 잘못된 형식을 우아하게 처리하세요.

---

**이전**: [제어 흐름](./04_Control_Flow.md) | **다음**: [배열과 문자열](./06_Arrays_and_Strings.md)
