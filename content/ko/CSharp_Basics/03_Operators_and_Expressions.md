# 연산자와 표현식

**이전**: [변수와 타입](./02_Variables_and_Types.md) | **다음**: [제어 흐름](./04_Control_Flow.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 수학 계산에 산술 연산자를 사용한다
2. 관계 연산자로 값을 비교한다
3. 논리 연산자로 조건을 결합한다
4. 비트 연산자로 개별 비트를 조작한다
5. 복합 대입 연산자로 대입을 간소화한다
6. 널 병합(Null-Coalescing) 및 널 조건부(Null-Conditional) 연산자로 null 값을 안전하게 처리한다
7. 삼항 연산자(Ternary Operator)로 간결한 조건 표현식을 작성한다
8. 복잡한 표현식에서 올바른 연산자 우선순위를 적용한다
9. checked/unchecked 컨텍스트로 산술 오버플로를 감지하고 방지한다

---

연산자는 하나, 둘, 또는 세 개의 피연산자에 특정 연산을 수행하도록 컴파일러에 지시하는 기호입니다. **표현식(Expression)**은 피연산자와 연산자의 조합으로 단일 값으로 평가됩니다. C#은 거의 모든 프로그램에서 사용하게 될 풍부한 연산자 세트를 가지고 있습니다.

## 1. 산술 연산자

산술 연산자는 숫자 피연산자에 대해 수학적 계산을 수행합니다.

### 1.1 기본 산술

```csharp
int a = 17;
int b = 5;

Console.WriteLine($"a + b = {a + b}");   // 22  (덧셈)
Console.WriteLine($"a - b = {a - b}");   // 12  (뺄셈)
Console.WriteLine($"a * b = {a * b}");   // 85  (곱셈)
Console.WriteLine($"a / b = {a / b}");   // 3   (정수 나눗셈 — 절삭)
Console.WriteLine($"a % b = {a % b}");   // 2   (나머지)
```

### 1.2 정수 나눗셈 vs 부동소수점 나눗셈

중요한 구별: 두 피연산자가 모두 정수인 경우 `/`는 정수 나눗셈(소수 부분 절삭)을 수행합니다. 하나 이상의 피연산자가 부동소수점 타입이면 실수 나눗셈이 수행됩니다:

```csharp
// 정수 나눗셈 — 소수 부분이 버려짐
Console.WriteLine(7 / 2);       // 3 (3.5가 아님)
Console.WriteLine(-7 / 2);      // -3 (0을 향해 절삭)

// 부동소수점 나눗셈 — 소수 부분 유지
Console.WriteLine(7.0 / 2);     // 3.5
Console.WriteLine(7 / 2.0);     // 3.5
Console.WriteLine((double)7 / 2); // 3.5 (한 피연산자를 캐스트)

// 0으로 나누기
// Console.WriteLine(7 / 0);    // 컴파일 에러 (상수 0으로 정수 나눗셈)
Console.WriteLine(7.0 / 0);     // Infinity
Console.WriteLine(-7.0 / 0);    // -Infinity
Console.WriteLine(0.0 / 0);     // NaN
```

### 1.3 나머지 연산자

나머지(모듈러스) 연산자 `%`는 많은 일반적인 작업에 유용합니다:

```csharp
// 짝수인지 홀수인지 확인
for (int i = 1; i <= 10; i++)
{
    string parity = (i % 2 == 0) ? "짝수" : "홀수";
    Console.WriteLine($"{i}은(는) {parity}");
}

// 순환 (원형 인덱싱)
string[] days = { "월", "화", "수", "목", "금", "토", "일" };
for (int i = 0; i < 14; i++)
{
    Console.WriteLine($"Day {i}: {days[i % 7]}");
}

// 자릿수 추출
int number = 12345;
while (number > 0)
{
    int digit = number % 10;
    Console.Write($"{digit} ");  // 5 4 3 2 1
    number /= 10;
}
Console.WriteLine();
```

### 1.4 증가와 감소

```csharp
int x = 5;

// 전위: 값을 사용하기 전에 증가/감소
Console.WriteLine(++x);  // 6 (x는 이제 6)
Console.WriteLine(--x);  // 5 (x는 이제 5)

// 후위: 값을 사용한 후에 증가/감소
Console.WriteLine(x++);  // 5 (5를 출력한 후 x가 6이 됨)
Console.WriteLine(x--);  // 6 (6을 출력한 후 x가 5가 됨)
Console.WriteLine(x);    // 5

// 반복문에서의 일반적인 사용
for (int i = 0; i < 5; i++)  // i++는 후위지만 결과를 사용하지 않음
{
    Console.Write($"{i} ");  // 0 1 2 3 4
}
Console.WriteLine();
```

### 1.5 단항 양수와 음수

```csharp
int positive = +5;   // 5 (단항 양수 — 거의 사용하지 않음)
int negative = -5;   // -5 (단항 음수 — 부정)
int negated = -negative;  // 5 (이중 부정)
Console.WriteLine($"{positive}, {negative}, {negated}");
```

## 2. 비교(관계) 연산자

비교 연산자는 `bool` 값(`true` 또는 `false`)으로 평가됩니다:

```csharp
int x = 10;
int y = 20;

Console.WriteLine($"x == y: {x == y}");   // False (같음)
Console.WriteLine($"x != y: {x != y}");   // True  (같지 않음)
Console.WriteLine($"x < y:  {x < y}");    // True  (미만)
Console.WriteLine($"x > y:  {x > y}");    // False (초과)
Console.WriteLine($"x <= y: {x <= y}");   // True  (이하)
Console.WriteLine($"x >= y: {x >= y}");   // False (이상)
```

### 2.1 다른 타입 비교

```csharp
// 숫자 비교는 타입 간에 작동 (암시적 변환)
Console.WriteLine(42 == 42.0);       // True (int와 double 비교)
Console.WriteLine(42 == 42L);        // True (int와 long 비교)

// 문자열 비교
string a = "hello";
string b = "hello";
string c = "Hello";
Console.WriteLine(a == b);           // True (문자열의 값 동등성)
Console.WriteLine(a == c);           // False (대소문자 구분)

// 객체 참조 비교
object obj1 = new object();
object obj2 = new object();
object obj3 = obj1;
Console.WriteLine(obj1 == obj2);     // False (다른 객체)
Console.WriteLine(obj1 == obj3);     // True (같은 참조)

// null 비교
string? name = null;
Console.WriteLine(name == null);     // True
Console.WriteLine(name is null);     // True (패턴 매칭 — 권장)
```

### 2.2 연쇄 비교

Python과 달리 C#은 `1 < x < 10`과 같은 연쇄 비교를 지원하지 않습니다. 논리 연산자를 사용해야 합니다:

```csharp
int value = 5;

// 이것은 예상대로 작동하지 않음:
// bool inRange = 1 < value < 10;  // 컴파일 에러

// 대신 논리 AND를 사용:
bool inRange = value > 1 && value < 10;
Console.WriteLine($"{value}이(가) (1, 10) 범위 내: {inRange}");  // True
```

## 3. 논리 연산자

논리 연산자는 불리언 표현식을 결합하거나 수정합니다:

```csharp
bool a = true;
bool b = false;

// 논리 AND — 둘 다 true일 때만 true
Console.WriteLine($"a && b: {a && b}");   // False
Console.WriteLine($"a && a: {a && a}");   // True

// 논리 OR — 하나라도 true이면 true
Console.WriteLine($"a || b: {a || b}");   // True
Console.WriteLine($"b || b: {b || b}");   // False

// 논리 NOT — 값을 반전
Console.WriteLine($"!a: {!a}");           // False
Console.WriteLine($"!b: {!b}");           // True

// 논리 XOR — 정확히 하나만 true일 때 true
Console.WriteLine($"a ^ b: {a ^ b}");    // True
Console.WriteLine($"a ^ a: {a ^ a}");    // False
```

### 3.1 단축 평가

`&&`와 `||`는 **단축 평가(Short-Circuit Evaluation)**를 사용합니다: 왼쪽 피연산자가 이미 결과를 결정하면 오른쪽 피연산자는 평가되지 않습니다.

```csharp
// 단축 AND: 왼쪽이 false이면 오른쪽이 평가되지 않음
int[] numbers = null!;
// 단축 없이는 NullReferenceException이 발생할 것:
if (numbers != null && numbers.Length > 0)
{
    Console.WriteLine($"첫 번째: {numbers[0]}");
}

// 단축 OR: 왼쪽이 true이면 오른쪽이 평가되지 않음
string? input = null;
string result = input ?? "default";  // 정확히 ||은 아니지만 유사한 개념

// 단축 평가 시연
int counter = 0;
bool Increment() { counter++; return true; }

bool test1 = false && Increment();   // Increment()가 호출되지 않음
Console.WriteLine($"counter = {counter}");  // 0

bool test2 = true || Increment();    // Increment()가 호출되지 않음
Console.WriteLine($"counter = {counter}");  // 0

bool test3 = true && Increment();    // Increment()가 호출됨
Console.WriteLine($"counter = {counter}");  // 1
```

### 3.2 비단축 연산자

`&`와 `|` 연산자(중복 없이)는 항상 양쪽을 모두 평가합니다:

```csharp
int x = 0;
bool alwaysEval = false & (++x > 0);  // 왼쪽이 false임에도 x가 증가됨
Console.WriteLine(x);  // 1

// 양쪽 피연산자의 부수 효과가 필요할 때 & 및 |를 사용
// (실제로는 드물게 사용 — &&와 ||을 권장)
```

## 4. 비트 연산자

비트 연산자는 정수 타입의 개별 비트에 대해 작동합니다:

```csharp
int a = 0b_1100;  // 10진수 12
int b = 0b_1010;  // 10진수 10

// AND — 두 비트가 모두 1일 때만 1
Console.WriteLine($"a & b  = {a & b}");    // 8  (0b_1000)

// OR — 어느 비트든 1이면 1
Console.WriteLine($"a | b  = {a | b}");    // 14 (0b_1110)

// XOR — 비트가 다르면 1
Console.WriteLine($"a ^ b  = {a ^ b}");    // 6  (0b_0110)

// NOT — 모든 비트를 반전
Console.WriteLine($"~a     = {~a}");       // -13 (32비트 모두 반전)

// 왼쪽 시프트 — 2^n을 곱함
Console.WriteLine($"a << 1 = {a << 1}");   // 24 (0b_11000)
Console.WriteLine($"a << 2 = {a << 2}");   // 48 (0b_110000)

// 오른쪽 시프트 — 2^n으로 나눔
Console.WriteLine($"a >> 1 = {a >> 1}");   // 6  (0b_0110)
Console.WriteLine($"a >> 2 = {a >> 2}");   // 3  (0b_0011)

// 부호 없는 오른쪽 시프트 (C# 11+) — 항상 0으로 채움
Console.WriteLine($"a >>> 1 = {a >>> 1}"); // 6
```

### 4.1 일반적인 비트 조작 패턴

```csharp
// 짝수인지 확인 (마지막 비트가 0)
int num = 42;
bool isEven = (num & 1) == 0;
Console.WriteLine($"{num}은(는) 짝수: {isEven}");  // True

// 특정 비트 설정
int flags = 0;
flags |= (1 << 3);  // 비트 3 설정
Console.WriteLine($"비트 3 설정 후: {flags}");  // 8 (0b_1000)

// 특정 비트 해제
flags &= ~(1 << 3);  // 비트 3 해제
Console.WriteLine($"비트 3 해제 후: {flags}");  // 0

// 특정 비트 토글
flags = 0b_1010;
flags ^= (1 << 1);  // 비트 1 토글
Console.WriteLine($"비트 1 토글 후: {Convert.ToString(flags, 2)}");  // 1000

// 특정 비트가 설정되어 있는지 확인
int value = 0b_1010;
bool bit1Set = (value & (1 << 1)) != 0;
bool bit2Set = (value & (1 << 2)) != 0;
Console.WriteLine($"비트 1 설정됨: {bit1Set}");  // True
Console.WriteLine($"비트 2 설정됨: {bit2Set}");  // False

// 설정된 비트 수 세기 (팝카운트)
int PopCount(int n)
{
    int count = 0;
    while (n != 0)
    {
        count += n & 1;
        n >>= 1;
    }
    return count;
}
Console.WriteLine($"PopCount(0b_1011) = {PopCount(0b_1011)}");  // 3
```

### 4.2 실용 예제: RGB 색상 조작

```csharp
// RGB 값을 하나의 int로 패킹 (0xAARRGGBB 형식)
int PackColor(byte r, byte g, byte b, byte a = 255)
{
    return (a << 24) | (r << 16) | (g << 8) | b;
}

// 개별 채널 언패킹
byte GetRed(int color)   => (byte)((color >> 16) & 0xFF);
byte GetGreen(int color) => (byte)((color >> 8) & 0xFF);
byte GetBlue(int color)  => (byte)(color & 0xFF);
byte GetAlpha(int color) => (byte)((color >> 24) & 0xFF);

int purple = PackColor(128, 0, 128);
Console.WriteLine($"Color: 0x{purple:X8}");
Console.WriteLine($"R={GetRed(purple)}, G={GetGreen(purple)}, B={GetBlue(purple)}");
// 출력: Color: 0xFF800080
// 출력: R=128, G=0, B=128
```

## 5. 대입 연산자

### 5.1 단순 대입

```csharp
int x = 10;       // x에 10을 대입
string name = "Alice";
```

### 5.2 복합 대입

복합 대입 연산자는 연산과 대입을 결합합니다:

```csharp
int x = 10;

x += 5;    // x = x + 5;   → 15
x -= 3;    // x = x - 3;   → 12
x *= 2;    // x = x * 2;   → 24
x /= 4;    // x = x / 4;   → 6
x %= 4;    // x = x % 4;   → 2

// 비트 복합 대입
int flags = 0;
flags |= 0b_0100;   // 비트 설정:  flags = 0b_0100
flags &= 0b_1100;   // 비트 해제:  flags = 0b_0100
flags ^= 0b_0110;   // 비트 토글:  flags = 0b_0010
flags <<= 2;         // 왼쪽 시프트: flags = 0b_1000
flags >>= 1;         // 오른쪽 시프트: flags = 0b_0100

Console.WriteLine($"flags = {Convert.ToString(flags, 2)}");
```

### 5.3 널 병합 대입 (??=)

```csharp
string? name = null;

// 현재 null인 경우에만 대입
name ??= "Default Name";
Console.WriteLine(name);  // "Default Name"

// name이 더 이상 null이 아니므로 대입하지 않음
name ??= "Other Name";
Console.WriteLine(name);  // "Default Name" (변경 없음)

// 지연 초기화에 유용
List<int>? cache = null;
// ... 이후 코드에서 ...
cache ??= new List<int>();  // cache가 null일 때만 리스트 생성
cache.Add(42);
```

## 6. 널 병합 및 널 조건부 연산자

이 연산자들은 C#에서 안전한 null 처리에 필수적입니다.

### 6.1 널 병합 연산자 (??)

왼쪽 피연산자가 null이 아니면 반환하고, 그렇지 않으면 오른쪽 피연산자를 반환합니다:

```csharp
string? input = null;
string result = input ?? "default";
Console.WriteLine(result);  // "default"

input = "hello";
result = input ?? "default";
Console.WriteLine(result);  // "hello"

// 여러 대안 연쇄
string? primary = null;
string? secondary = null;
string? tertiary = "found!";
string value = primary ?? secondary ?? tertiary ?? "last resort";
Console.WriteLine(value);  // "found!"

// 값 타입과 함께
int? maybeNumber = null;
int number = maybeNumber ?? -1;
Console.WriteLine(number);  // -1
```

### 6.2 널 조건부 연산자 (?.)

객체가 null이 아닌 경우에만 멤버에 접근하고, 그렇지 않으면 null을 반환합니다:

```csharp
string? name = null;

// 널 조건부 없이는 — NullReferenceException이 발생할 것
// int length = name.Length;

// 널 조건부 사용 — 예외 대신 null을 반환
int? length = name?.Length;
Console.WriteLine(length);  // (null)

// 여러 널 조건부 접근 연쇄
string?[] names = { "Alice", null, "Charlie" };
Console.WriteLine(names[1]?.ToUpper()?.Substring(0, 3));  // (null — 예외 없음)
Console.WriteLine(names[0]?.ToUpper()?.Substring(0, 3));  // ALI

// 널 병합과 결합
string display = name?.ToUpper() ?? "N/A";
Console.WriteLine(display);  // "N/A"

// 인덱서에서 널 조건부
int[]? numbers = null;
int? first = numbers?[0];  // null (예외 없음)
Console.WriteLine(first);

// 메서드 호출에서 널 조건부
string? text = "Hello, World";
bool? contains = text?.Contains("World");
Console.WriteLine(contains);  // True
```

## 7. 삼항(조건부) 연산자

삼항 연산자 `?:`는 간단한 `if`/`else`의 간결한 대안입니다:

```csharp
// 구문: 조건 ? 참일_때_값 : 거짓일_때_값

int age = 20;
string status = age >= 18 ? "성인" : "미성년자";
Console.WriteLine(status);  // "성인"

// 동등한 if/else (더 장황함):
// string status;
// if (age >= 18) status = "성인";
// else status = "미성년자";

// 중첩 삼항 (아껴서 사용 — 가독성을 떨어뜨릴 수 있음)
int score = 85;
string grade = score >= 90 ? "A"
             : score >= 80 ? "B"
             : score >= 70 ? "C"
             : score >= 60 ? "D"
             : "F";
Console.WriteLine($"점수 {score}: 등급 {grade}");  // 점수 85: 등급 B

// 문자열 보간에서
int count = 5;
Console.WriteLine($"You have {count} item{(count != 1 ? "s" : "")}.");
// 출력: You have 5 items.

// 메서드 호출과 함께
int x = -5;
int absolute = x >= 0 ? x : -x;
Console.WriteLine($"|{x}| = {absolute}");  // |-5| = 5
```

## 8. 연산자 우선순위

연산자는 특정 순서로 평가됩니다. 높은 우선순위는 연산자가 더 강하게 결합됨을 의미합니다:

```
우선순위     연산자                               결합 방향
───────────  ─────────────────────────────────  ─────────────
1 (최고)     x.y  x?.y  x?[i]  f(x)  a[i]     왼쪽에서 오른쪽
             x++  x--  new  typeof  sizeof
2            +x  -x  !x  ~x  ++x  --x          오른쪽에서 왼쪽
             (T)x  await
3            x * y   x / y   x % y              왼쪽에서 오른쪽
4            x + y   x - y                      왼쪽에서 오른쪽
5            x << y  x >> y  x >>> y            왼쪽에서 오른쪽
6            x < y   x > y   x <= y  x >= y    왼쪽에서 오른쪽
             is  as
7            x == y  x != y                     왼쪽에서 오른쪽
8            x & y                              왼쪽에서 오른쪽
9            x ^ y                              왼쪽에서 오른쪽
10           x | y                              왼쪽에서 오른쪽
11           x && y                             왼쪽에서 오른쪽
12           x || y                             왼쪽에서 오른쪽
13           x ?? y                             왼쪽에서 오른쪽
14           c ? t : f                          오른쪽에서 왼쪽
15 (최저)    =  +=  -=  *=  /=  %=              오른쪽에서 왼쪽
             &=  |=  ^=  <<=  >>=  ??=
```

### 8.1 우선순위 예제

```csharp
// 덧셈 전에 곱셈
int result1 = 2 + 3 * 4;
Console.WriteLine(result1);  // 14 (20이 아님)

// 논리 연산 전에 비교
bool result2 = 5 > 3 && 2 < 4;
Console.WriteLine(result2);  // True ((5 > 3) && (2 < 4)로 평가됨)

// 괄호가 우선순위를 재정의
int result3 = (2 + 3) * 4;
Console.WriteLine(result3);  // 20

// 흔한 실수
int a = 5, b = 3;
// bool wrong = a & b == 0;      // a & (b == 0)으로 평가됨 — 의도한 것이 아닐 수 있음!
bool correct = (a & b) == 0;     // 올바름: &를 먼저 적용한 후 비교

// 대입은 오른쪽 결합
int x, y, z;
x = y = z = 10;  // z=10, 그 다음 y=10, 그 다음 x=10
Console.WriteLine($"x={x}, y={y}, z={z}");  // x=10, y=10, z=10
```

### 8.2 모범 사례: 괄호 사용

의심스러울 때는 괄호를 추가하세요. 의도를 명확하게 하고 미묘한 버그를 방지합니다:

```csharp
// 괄호 없이는 모호함
int result = a + b * c - d / e;

// 괄호로 명확하게
int result_clear = a + (b * c) - (d / e);

// 우선순위를 알더라도 괄호가 독자에게 도움
bool isValid = (age >= 18) && (age <= 65) && (hasLicense == true);
```

## 9. 검사된(Checked) 및 미검사(Unchecked) 산술

기본적으로 C#의 정수 산술은 오버플로 시 예외를 던지지 않고 순환합니다. `checked`와 `unchecked` 키워드가 이 동작을 제어합니다.

### 9.1 기본 동작 (미검사)

```csharp
int max = int.MaxValue;  // 2,147,483,647
int overflow = max + 1;
Console.WriteLine(overflow);  // -2,147,483,648 (조용히 순환!)

byte b = 255;
b++;
Console.WriteLine(b);  // 0 (순환)
```

### 9.2 검사된(Checked) 컨텍스트

```csharp
// checked 블록 — 오버플로 시 OverflowException을 던짐
try
{
    checked
    {
        int max = int.MaxValue;
        int overflow = max + 1;  // OverflowException을 던짐!
    }
}
catch (OverflowException ex)
{
    Console.WriteLine($"오버플로 감지: {ex.Message}");
}

// checked 표현식 — 단일 연산에 대해
try
{
    int result = checked(int.MaxValue + 1);
}
catch (OverflowException)
{
    Console.WriteLine("단일 표현식에서 오버플로!");
}
```

### 9.3 미검사(Unchecked) 컨텍스트

```csharp
// 명시적으로 미검사 (기본과 동일하지만 의도를 명확히 함)
unchecked
{
    int max = int.MaxValue;
    int overflow = max + 1;
    Console.WriteLine(overflow);  // -2,147,483,648 (예외 없음)
}

// 오버플로가 의도적일 때 유용 (예: 해시 코드 계산)
unchecked
{
    int hash = 17;
    hash = hash * 31 + "hello".GetHashCode();
    hash = hash * 31 + 42.GetHashCode();
    Console.WriteLine($"Hash: {hash}");
}
```

### 9.4 프로젝트 전체 검사된 산술

`.csproj`에서 전체 프로젝트에 대해 검사된 산술을 활성화할 수 있습니다:

```xml
<PropertyGroup>
  <CheckForOverflowUnderflow>true</CheckForOverflowUnderflow>
</PropertyGroup>
```

## 10. 기타 연산자

### 10.1 typeof, sizeof, nameof

```csharp
// typeof — System.Type 객체를 가져옴
Type intType = typeof(int);
Console.WriteLine(intType.FullName);  // System.Int32

// sizeof — 바이트 단위 크기 (비관리 타입에서만 또는 알려진 타입)
Console.WriteLine(sizeof(int));       // 4
Console.WriteLine(sizeof(double));    // 8
Console.WriteLine(sizeof(char));      // 2
Console.WriteLine(sizeof(bool));      // 1
Console.WriteLine(sizeof(long));      // 8

// nameof — 변수, 타입, 멤버의 이름을 문자열로 가져옴
string variableName = "test";
Console.WriteLine(nameof(variableName));      // "variableName"
Console.WriteLine(nameof(Console.WriteLine)); // "WriteLine"
Console.WriteLine(nameof(String));            // "String"
```

### 10.2 is 연산자 (타입 테스트)

```csharp
object value = 42;

if (value is int)
{
    Console.WriteLine("정수입니다!");
}

// 패턴 변수와 함께 is (C# 7+)
if (value is int number)
{
    Console.WriteLine($"정수 값은 {number}");
}

// 상수 패턴과 함께 is
if (value is not null)
{
    Console.WriteLine($"값은 null이 아님: {value}");
}
```

### 10.3 as 연산자 (안전한 캐스트)

```csharp
object obj = "Hello, World!";

// as — 캐스트 실패 시 예외 대신 null을 반환
string? str = obj as string;
if (str != null)
{
    Console.WriteLine($"문자열: {str}");
}

// as는 참조 타입과 널러블 값 타입에서만 작동
// int num = obj as int;  // 컴파일 에러: int는 널러블이 아님
int? num = obj as int?;    // OK: null을 반환 (obj는 string이지 int가 아님)
Console.WriteLine(num);    // (null)
```

## 11. 연습 문제

1. **표현식 평가기**: 코드를 실행하지 않고 각 표현식의 출력을 예측하세요. 그런 다음 실행하여 확인하세요:
   ```csharp
   Console.WriteLine(5 + 3 * 2);
   Console.WriteLine((5 + 3) * 2);
   Console.WriteLine(10 / 3 + 10 % 3);
   Console.WriteLine(true || false && false);
   Console.WriteLine((true || false) && false);
   ```

2. **비트 플래그 시스템**: 비트 연산자를 사용하여 권한 시스템을 만드세요. `Read = 1`, `Write = 2`, `Execute = 4`, `Admin = 8` 플래그를 정의하세요. 다음 메서드를 작성하세요: (a) 권한 부여, (b) 권한 철회, (c) 권한 설정 여부 확인, (d) 모든 활성 권한 표시. 여러 권한의 결합과 확인을 시연하세요.

3. **널 안전 체인**: 다음 클래스 구조가 주어졌을 때, 깊이 중첩된 프로퍼티에 안전하게 접근하기 위한 널 조건부 연산자 체인을 작성하세요:
   ```csharp
   class Company { public Department? MainDepartment; }
   class Department { public Employee? Lead; }
   class Employee { public Address? HomeAddress; }
   class Address { public string? City; }
   ```
   도시를 출력하거나, 체인의 어느 부분이라도 null이면 "Unknown"을 출력하세요.

4. **오버플로 탐정**: `byte`, `short`, `int`, `long`에 대한 오버플로를 시연하는 프로그램을 작성하세요. 각 타입에 대해 다음을 보여주세요: (a) 최댓값, (b) unchecked 컨텍스트에서 1을 더한 결과, (c) `checked` 컨텍스트에서 `OverflowException`이 발생하는 것.

5. **등급 계산기**: 사용자로부터 숫자 점수(0-100)를 읽고 삼항 연산자만 사용하여 (if 문 없이) 문자 등급으로 변환하는 프로그램을 작성하세요. `TryParse`와 널 병합 연산자를 사용하여 잘못된 입력(숫자가 아닌 것, 범위 밖)을 처리하세요.

---

**이전**: [변수와 타입](./02_Variables_and_Types.md) | **다음**: [제어 흐름](./04_Control_Flow.md)
