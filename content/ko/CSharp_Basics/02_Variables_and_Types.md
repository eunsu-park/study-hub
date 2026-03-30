# 변수와 타입

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [연산자와 표현식](./03_Operators_and_Expressions.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 값 타입(Value Type)과 참조 타입(Reference Type)을 구별한다
2. 명시적 타입과 `var`를 사용하여 변수를 선언하고 초기화한다
3. 모든 일반적인 숫자 타입과 그 범위를 이해한다
4. 불변 값을 위해 `const`와 `readonly`를 사용한다
5. `?` 접미사를 사용하여 널러블 값 타입(Nullable Value Type)을 다룬다
6. 암시적 및 명시적 타입 변환을 안전하게 수행한다
7. 문자열 리터럴, 축어 문자열(Verbatim String), 원시 문자열(Raw String), 보간(Interpolation)을 사용한다

---

모든 프로그램은 데이터를 다루며, C#에서 모든 데이터는 타입을 갖습니다. C# 타입 시스템은 주류 프로그래밍 언어 중 가장 풍부한 시스템 중 하나로, 값 타입(스택에 직접 저장)과 참조 타입(힙에 저장되고 스택에 참조가 있음)의 명확한 구분을 제공합니다. 이 구분을 이해하는 것이 올바르고 효율적인 C# 코드를 작성하는 데 핵심입니다.

## 1. C# 타입 시스템 개요

C#은 **정적 타입(Statically Typed)** 언어입니다: 모든 변수, 매개변수, 표현식은 컴파일 시점에 알려진 타입을 갖습니다. 타입 시스템은 두 가지 주요 카테고리로 구성됩니다:

```
                    C# 타입
                       │
            ┌──────────┴──────────┐
        값 타입              참조 타입
            │                      │
    ┌───────┼───────┐       ┌──────┼──────┐
  기본형  구조체  열거형    클래스  인터페이스  델리게이트
    │                       │
  int, double,           string, object,
  bool, char, ...        배열, ...
```

### 1.1 값 타입 vs 참조 타입

핵심 차이는 메모리에 저장되는 방식입니다:

```csharp
// 값 타입: 데이터가 변수에 직접 저장됨
int x = 42;
int y = x;    // y는 값의 복사본을 받음
y = 100;
Console.WriteLine(x);  // 42 — x는 변경되지 않음
Console.WriteLine(y);  // 100

// 참조 타입: 변수는 데이터에 대한 참조(포인터)를 저장함
int[] a = { 1, 2, 3 };
int[] b = a;          // b는 참조의 복사본을 받음 (같은 데이터를 가리킴)
b[0] = 99;
Console.WriteLine(a[0]);  // 99 — a와 b가 같은 배열을 가리키므로 a[0]도 변경됨!
```

메모리 레이아웃:

```
값 타입 (스택):             참조 타입 (스택 + 힙):

스택:                       스택:            힙:
┌────────┐                  ┌────────┐      ┌─────────────┐
│ x: 42  │                  │ a: ──────────▶│ { 1, 2, 3 } │
├────────┤                  ├────────┤      └─────────────┘
│ y: 100 │                  │ b: ──────────▶  (같은 객체)
└────────┘                  └────────┘
```

## 2. 정수 타입

C#은 크기와 음수 지원 여부에 따라 여러 정수 타입을 제공합니다:

```csharp
// 부호 있는 정수 (음수 가능)
sbyte  smallestSigned = -128;           // 8비트:  -128 ~ 127
short  shortValue     = -32_768;        // 16비트: -32,768 ~ 32,767
int    intValue       = -2_147_483_648; // 32비트: ~-21억 ~ ~21억
long   longValue      = -9_000_000_000; // 64비트: ~-922경 ~ ~922경

// 부호 없는 정수 (비음수만)
byte   byteValue      = 255;            // 8비트:  0 ~ 255
ushort ushortValue     = 65_535;         // 16비트: 0 ~ 65,535
uint   uintValue       = 4_294_967_295; // 32비트: 0 ~ ~43억
ulong  ulongValue      = 18_000_000_000_000_000_000; // 64비트: 0 ~ ~1844경

// nint과 nuint — 플랫폼 의존적 (32비트 또는 64비트)
nint  nativeInt  = 42;   // IntPtr과 같은 크기
nuint nativeUint = 42;   // UIntPtr과 같은 크기
```

가독성을 높이기 위해 밑줄(`_`)을 자릿수 구분자로 사용하고 있습니다. 값에는 영향을 주지 않습니다.

### 2.1 정수 리터럴

```csharp
// 10진수 (기수 10)
int decimal_ = 42;

// 16진수 (기수 16) — 접두사 0x
int hex = 0x2A;           // 42

// 2진수 (기수 2) — 접두사 0b
int binary = 0b0010_1010; // 42

// long 리터럴 — 접미사 L
long bigNumber = 3_000_000_000L;

// unsigned 리터럴 — 접미사 U
uint positive = 42U;

// unsigned long — 접미사 UL
ulong veryBig = 42UL;

Console.WriteLine($"decimal={decimal_}, hex={hex}, binary={binary}");
// 출력: decimal=42, hex=42, binary=42
```

### 2.2 타입 범위 확인

```csharp
Console.WriteLine($"int:    {int.MinValue:N0} ~ {int.MaxValue:N0}");
Console.WriteLine($"long:   {long.MinValue:N0} ~ {long.MaxValue:N0}");
Console.WriteLine($"byte:   {byte.MinValue} ~ {byte.MaxValue}");
Console.WriteLine($"short:  {short.MinValue:N0} ~ {short.MaxValue:N0}");

// 출력:
// int:    -2,147,483,648 ~ 2,147,483,647
// long:   -9,223,372,036,854,775,808 ~ 9,223,372,036,854,775,807
// byte:   0 ~ 255
// short:  -32,768 ~ 32,767
```

## 3. 부동소수점 및 decimal 타입

소수 부분이 있는 숫자를 위해 C#은 정밀도와 성능 특성이 다른 세 가지 타입을 제공합니다:

```csharp
// float: 32비트, ~6-7 유효 숫자, 접미사 F
float  pi_f = 3.14159265f;
Console.WriteLine($"float:   {pi_f}");  // 3.1415927 (제한된 정밀도)

// double: 64비트, ~15-17 유효 숫자 (리터럴의 기본값)
double pi_d = 3.141592653589793;
Console.WriteLine($"double:  {pi_d}");  // 3.141592653589793

// decimal: 128비트, 28-29 유효 숫자, 접미사 M
// 금융 계산에 가장 적합 — 이진 부동소수점 반올림 없음
decimal price = 19.99m;
decimal tax   = 0.08m;
decimal total = price + (price * tax);
Console.WriteLine($"decimal: {total}");  // 21.5892 (정확함)
```

### 3.1 부동소수점 정밀도 함정

```csharp
// 고전적인 부동소수점 놀라움
double a = 0.1 + 0.2;
Console.WriteLine(a == 0.3);         // False!
Console.WriteLine($"{a:R}");         // 0.30000000000000004

// decimal은 10진 소수에 대해 이 문제가 없음
decimal b = 0.1m + 0.2m;
Console.WriteLine(b == 0.3m);        // True
Console.WriteLine(b);                // 0.3
```

### 3.2 특수 부동소수점 값

```csharp
double posInf = double.PositiveInfinity;
double negInf = double.NegativeInfinity;
double nan    = double.NaN;

Console.WriteLine(1.0 / 0.0);                    // Infinity
Console.WriteLine(-1.0 / 0.0);                   // -Infinity
Console.WriteLine(0.0 / 0.0);                    // NaN
Console.WriteLine(double.IsNaN(nan));             // True
Console.WriteLine(double.IsInfinity(posInf));     // True

// NaN은 자기 자신을 포함하여 어떤 것과도 같지 않음
Console.WriteLine(nan == nan);                    // False
Console.WriteLine(double.NaN == double.NaN);      // False
```

## 4. 불리언과 문자 타입

### 4.1 bool

`bool` 타입은 `true` 또는 `false`만 보유합니다. C나 C++와 달리 정수는 암시적으로 `bool`로 변환될 수 없습니다:

```csharp
bool isReady = true;
bool isComplete = false;

// 표현식에서의 불리언
bool isAdult = 21 >= 18;        // true
bool isEqual = (3 + 4) == 7;    // true

// 이것은 불가능합니다 (C/C++와 달리):
// if (1) { }  // 컴파일 에러: 'int'를 'bool'로 암시적 변환 불가

// 명시적이어야 합니다:
if (isReady)
{
    Console.WriteLine("준비됨!");
}
```

### 4.2 char

`char` 타입은 단일 유니코드 문자(UTF-16 코드 유닛, 16비트)를 나타냅니다:

```csharp
char letter = 'A';
char digit  = '7';
char symbol = '€';
char korean = '한';
char emoji  = '☺';  // 기본 이모지 (BMP)

// 문자 이스케이프 시퀀스
char newline  = '\n';   // 줄바꿈
char tab      = '\t';   // 탭
char backslash = '\\';  // 백슬래시
char quote    = '\'';   // 작은따옴표
char nullChar = '\0';   // 널 문자

// 유니코드 이스케이프
char omega = '\u03A9';  // Ω
Console.WriteLine(omega); // Ω

// char는 실제로 16비트 부호 없는 정수
Console.WriteLine((int)letter);    // 65
Console.WriteLine((char)65);       // A
Console.WriteLine((int)'0');       // 48

// 문자 메서드
Console.WriteLine(char.IsLetter('A'));     // True
Console.WriteLine(char.IsDigit('7'));      // True
Console.WriteLine(char.IsUpper('A'));      // True
Console.WriteLine(char.ToLower('A'));      // a
Console.WriteLine(char.IsWhiteSpace(' ')); // True
```

## 5. 참조 타입: string, object, dynamic

### 5.1 string

`string`은 참조 타입이지만, 문자열은 **불변(Immutable)**이기 때문에 값 타입처럼 동작합니다 — 한 번 생성되면 문자열을 변경할 수 없습니다:

```csharp
string greeting = "Hello";
string name = "World";
string message = greeting + ", " + name + "!";
Console.WriteLine(message);  // Hello, World!

// 문자열은 불변 — 문자열을 "변경"하면 새로운 것이 생성됨
string original = "Hello";
string modified = original.Replace("H", "J");
Console.WriteLine(original);  // Hello (변경되지 않음)
Console.WriteLine(modified);  // Jello (새 문자열)

// 문자열 비교
string a = "hello";
string b = "Hello";
Console.WriteLine(a == b);                                        // False (대소문자 구분)
Console.WriteLine(a.Equals(b, StringComparison.OrdinalIgnoreCase)); // True

// 일반적인 문자열 메서드
string text = "  Hello, World!  ";
Console.WriteLine(text.Trim());              // "Hello, World!"
Console.WriteLine(text.ToUpper());           // "  HELLO, WORLD!  "
Console.WriteLine(text.Contains("World"));   // True
Console.WriteLine(text.IndexOf("World"));    // 9
Console.WriteLine(text.Substring(8, 5));     // "orld!"
Console.WriteLine(text.Length);              // 17
```

### 5.2 object

`object`는 C#에서 모든 타입의 기본 타입입니다. 모든 타입(값 타입과 참조 타입 모두)은 궁극적으로 `object`를 상속합니다:

```csharp
object obj1 = 42;           // int가 object로 박싱됨
object obj2 = "Hello";      // string이 object에 할당됨
object obj3 = 3.14;         // double이 object로 박싱됨
object obj4 = true;         // bool이 object로 박싱됨

Console.WriteLine(obj1.GetType());  // System.Int32
Console.WriteLine(obj2.GetType());  // System.String

// 언박싱 — 값 추출 (정확한 타입으로 캐스트해야 함)
int number = (int)obj1;
Console.WriteLine(number);  // 42

// ToString() — 모든 object가 이 메서드를 가짐
Console.WriteLine(obj1.ToString());  // "42"
Console.WriteLine(obj3.ToString());  // "3.14"
```

### 5.3 dynamic

`dynamic` 타입은 컴파일 시점 타입 검사를 우회합니다. 타입은 런타임에 결정됩니다:

```csharp
dynamic value = 42;
Console.WriteLine(value.GetType());  // System.Int32

value = "Now I'm a string";
Console.WriteLine(value.GetType());  // System.String

value = new[] { 1, 2, 3 };
Console.WriteLine(value.Length);     // 3

// 이것은 컴파일되지만 런타임에 RuntimeBinderException을 던짐:
// dynamic d = "hello";
// int x = d * 2;  // 런타임 에러: string과 int에 *를 적용할 수 없음
```

`dynamic`은 아껴서 사용하세요 — 컴파일 시점 안전성이 희생됩니다. 주로 상호운용성(COM, 동적 언어, 리플렉션)에 사용됩니다.

## 6. var와 타입 추론

`var` 키워드는 대입의 오른쪽에서 컴파일러가 타입을 추론하게 합니다:

```csharp
var count = 42;              // int (추론됨)
var pi = 3.14159;            // double (추론됨)
var name = "Alice";          // string (추론됨)
var items = new List<int>(); // List<int> (추론됨)
var flag = true;             // bool (추론됨)

// var는 여전히 정적 타입 — 타입을 변경할 수 없음
// count = "hello";  // 컴파일 에러: string을 int로 암시적 변환 불가

// var는 초기화가 필요
// var x;  // 컴파일 에러: 암시적으로 타입이 지정된 변수는 초기화해야 함
```

### 6.1 var를 사용할 때

```csharp
// 오른쪽에서 타입이 명확할 때 var를 사용
var names = new List<string>();                     // 명백히 List<string>
var lookup = new Dictionary<string, List<int>>();    // 긴 타입 반복 회피
var stream = File.OpenRead("data.txt");              // 명백히 FileStream

// 타입이 명확하지 않을 때 var를 피함
var result = Calculate(x, y);  // result의 타입이 무엇인가? 불명확.
double result = Calculate(x, y);  // 더 좋음 — 타입이 명시적

// 익명 타입에는 var가 필수
var anon = new { Name = "Alice", Age = 30 };
Console.WriteLine($"{anon.Name} is {anon.Age}");
```

## 7. const와 readonly

### 7.1 const — 컴파일 시점 상수

`const`는 컴파일 시점에 알려져야 하고 절대 변경할 수 없는 값을 선언합니다:

```csharp
const double Pi = 3.14159265358979;
const int MaxRetries = 3;
const string AppName = "MyApp";

// const 값은 컴파일된 코드에 직접 포함됨
// 수정할 수 없음:
// Pi = 3.0;  // 컴파일 에러

// const는 기본형, string, null에만 사용 가능
// const DateTime now = DateTime.Now;  // 컴파일 에러: 상수 표현식이 아님

// 클래스의 상수
class Config
{
    public const int MaxConnections = 100;
    public const string Version = "1.0.0";
    public const double Gravity = 9.81;
}

// 인스턴스 없이 접근 (암시적으로 static)
Console.WriteLine(Config.MaxConnections);  // 100
```

### 7.2 readonly — 런타임 상수

한 번만 설정해야 하지만 컴파일 시점에 결정할 수 없는 값에는 `readonly`를 사용합니다 (클래스 레슨에서 더 자세히 다루지만 여기서 미리보기를 보겠습니다):

```csharp
class AppSettings
{
    public readonly DateTime StartTime;
    public readonly string MachineName;

    public AppSettings()
    {
        StartTime = DateTime.Now;          // 런타임에 설정
        MachineName = Environment.MachineName;
    }
}
```

### 7.3 const vs readonly

```csharp
// const: 컴파일 시점, 암시적으로 static, 제한된 타입
// readonly: 런타임, 인스턴스별 (또는 static), 모든 타입

class Example
{
    public const int CompileTimeValue = 42;           // 컴파일 시점에 알아야 함
    public readonly int RuntimeValue;                  // 생성자에서 설정 가능
    public static readonly int SharedValue = GetValue(); // 시작 시 한 번 계산

    public Example(int value)
    {
        RuntimeValue = value;
    }

    private static int GetValue() => Environment.ProcessorCount;
}
```

## 8. 널러블 값 타입

값 타입은 일반적으로 `null`이 될 수 없습니다. `?`를 추가하면 널러블(Nullable)이 됩니다:

```csharp
// 일반 int는 null이 될 수 없음
int count = 0;
// count = null;  // 컴파일 에러

// 널러블 int는 null이 될 수 있음
int? maybeCount = null;
int? definiteCount = 42;

Console.WriteLine(maybeCount.HasValue);     // False
Console.WriteLine(definiteCount.HasValue);  // True
Console.WriteLine(definiteCount.Value);     // 42

// null일 때 .Value에 접근하면 InvalidOperationException 발생
// Console.WriteLine(maybeCount.Value);  // 런타임 에러!

// 널 병합 연산자로 안전하게 접근
int safeCount = maybeCount ?? 0;  // null이면 0 사용
Console.WriteLine(safeCount);     // 0

// GetValueOrDefault
Console.WriteLine(maybeCount.GetValueOrDefault());     // 0
Console.WriteLine(maybeCount.GetValueOrDefault(-1));   // -1
Console.WriteLine(definiteCount.GetValueOrDefault());  // 42

// 실제 활용: 데이터베이스 결과, 선택적 구성
double? temperature = ReadSensor();   // 센서 실패 시 null 반환 가능
bool? userConsent = null;             // 삼중 상태: true, false, 또는 알 수 없음
```

### 8.1 널러블 참조 타입

C# 8 이상에서는 널러블 참조 타입을 도입했습니다 (`.csproj`에서 `<Nullable>enable</Nullable>`로 활성화):

```csharp
// 널러블 참조 타입이 활성화된 경우:
string name = "Alice";   // 널 불가 — null을 할당하면 컴파일러가 경고
string? nickname = null;  // 명시적 널러블 — null 허용

// 컴파일러 경고가 NullReferenceException 방지에 도움
// string bad = null;  // 경고: null 리터럴 또는 가능한 null 값 변환

// 널 조건부 연산자
int? length = nickname?.Length;  // nickname이 null이면 null, 아니면 Length

// 널 용서 연산자 (null이 아님을 알 때 경고 억제)
string definitelyNotNull = nickname!;  // 컴파일러에게 "이건 null이 아님을 믿어"라고 알림
```

## 9. 타입 변환

### 9.1 암시적 변환 (확대 변환)

데이터 손실 위험이 없을 때 암시적 변환이 자동으로 발생합니다:

```csharp
// 작은 타입에서 큰 타입으로 — 항상 안전
byte b = 42;
int i = b;       // byte → int (암시적)
long l = i;      // int → long (암시적)
float f = l;     // long → float (암시적, 하지만 정밀도 손실 가능!)
double d = f;    // float → double (암시적)

// int에서 decimal로의 변환은 암시적 (정밀도 손실 없음)
decimal dec = i;  // int → decimal (암시적)

Console.WriteLine($"byte={b}, int={i}, long={l}, float={f}, double={d}, decimal={dec}");
```

### 9.2 명시적 변환 (축소 변환 / 캐스팅)

데이터 손실이 가능할 때 명시적 캐스트를 사용해야 합니다:

```csharp
double pi = 3.14159;
int truncated = (int)pi;           // 3 (소수 부분 손실)
Console.WriteLine(truncated);

long big = 3_000_000_000;
int overflow = (int)big;           // -1294967296 (오버플로! 순환)
Console.WriteLine(overflow);

float precise = 1.23456789f;
int rounded = (int)precise;        // 1 (반올림이 아닌 절삭)
Console.WriteLine(rounded);

// 숫자 타입 간 캐스팅
double temperature = 98.6;
int tempInt = (int)temperature;    // 98
float tempFloat = (float)temperature;  // 98.6
```

### 9.3 Convert 클래스

`Convert` 클래스는 반올림과 null을 처리하는 메서드를 제공합니다:

```csharp
// Convert는 절삭 대신 반올림
double value = 3.7;
int converted = Convert.ToInt32(value);  // 4 (은행가 반올림)
int casted = (int)value;                 // 3 (절삭)
Console.WriteLine($"Convert: {converted}, Cast: {casted}");

// 문자열을 숫자로 변환
string numberStr = "42";
int number = Convert.ToInt32(numberStr);
double dbl = Convert.ToDouble("3.14");
bool flag = Convert.ToBoolean("true");

Console.WriteLine($"int={number}, double={dbl}, bool={flag}");

// Convert는 null을 우아하게 처리
string? nullStr = null;
int fromNull = Convert.ToInt32(nullStr);  // 0 (예외가 아님)
```

### 9.4 문자열 파싱

```csharp
// Parse — 잘못된 입력에 FormatException을 던짐
int parsed = int.Parse("42");
double dParsed = double.Parse("3.14");

// TryParse — bool을 반환, 예외를 던지지 않음 (권장)
if (int.TryParse("42", out int result))
{
    Console.WriteLine($"파싱 성공: {result}");
}

if (int.TryParse("not a number", out int failed))
{
    Console.WriteLine($"성공: {failed}");
}
else
{
    Console.WriteLine("파싱 실패");  // 이것이 실행됨
}

// 다양한 타입으로 TryParse
bool validDouble = double.TryParse("3.14", out double dResult);
bool validBool = bool.TryParse("true", out bool bResult);
bool validDate = DateTime.TryParse("2024-01-15", out DateTime dtResult);

Console.WriteLine($"double={dResult}, bool={bResult}, date={dtResult:yyyy-MM-dd}");
```

## 10. 기본값과 초기화

C#에서 모든 타입에는 기본값이 있습니다:

```csharp
// 일반적인 타입의 기본값
Console.WriteLine(default(int));       // 0
Console.WriteLine(default(double));    // 0
Console.WriteLine(default(bool));      // False
Console.WriteLine(default(char));      // '\0' (널 문자)
Console.WriteLine(default(string));    // (null — 빈 줄)
Console.WriteLine(default(int?));      // (null — 빈 줄)

// var와 함께 default 키워드 사용
int x = default;           // 0
string s = default!;       // null (널 용서 연산자 포함)
bool b = default;          // false

// 클래스의 필드는 자동으로 기본값을 받음
class Defaults
{
    public int Number;         // 0
    public double Fraction;    // 0.0
    public bool Flag;          // false
    public string? Text;       // null
    public int? Optional;      // null
}

var d = new Defaults();
Console.WriteLine($"Number={d.Number}, Flag={d.Flag}, Text={d.Text ?? "null"}");
// 출력: Number=0, Flag=False, Text=null
```

## 11. 심화 문자열 리터럴

C#은 문자열 리터럴을 작성하는 여러 방법을 제공합니다:

```csharp
// 1. 일반 문자열 리터럴
string regular = "Hello\nWorld";  // \n은 줄바꿈

// 2. 축어 문자열 리터럴 — @ 접두사
// 이스케이프 시퀀스가 처리되지 않음 (따옴표용 ""는 예외)
string path = @"C:\Users\Alice\Documents";    // 백슬래시를 이스케이프할 필요 없음
string multiLine = @"Line 1
Line 2
Line 3";
string withQuote = @"She said ""hello""";

// 3. 문자열 보간 — $ 접두사
string name = "Alice";
int age = 30;
string intro = $"My name is {name} and I am {age} years old.";
Console.WriteLine(intro);

// 4. 축어 + 보간 결합 — $@ 또는 @$
string filePath = $@"C:\Users\{name}\Documents";
Console.WriteLine(filePath);  // C:\Users\Alice\Documents

// 5. 원시 문자열 리터럴 (C# 11+) — 삼중 (또는 그 이상) 따옴표
string rawJson = """
    {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "coding"]
    }
    """;
Console.WriteLine(rawJson);

// 6. 원시 보간 문자열
int count = 5;
string rawInterpolated = $"""
    The count is {count}.
    To use braces in output: {{like this}}.
    """;
Console.WriteLine(rawInterpolated);

// 7. 형식 지정자가 포함된 보간
double price = 49.99;
DateTime now = DateTime.Now;
Console.WriteLine($"Price: {price:C}");           // Price: $49.99 (통화)
Console.WriteLine($"Price: {price:F4}");           // Price: 49.9900 (소수점 4자리)
Console.WriteLine($"Date: {now:yyyy-MM-dd}");      // Date: 2024-01-15
Console.WriteLine($"Hex: {255:X}");                // Hex: FF
Console.WriteLine($"Padded: {42,10}");             // Padded:         42 (오른쪽 정렬)
Console.WriteLine($"Padded: {42,-10}|");           // Padded: 42        | (왼쪽 정렬)
```

## 12. 연습 문제

1. **타입 크기 표**: 각 숫자 타입의 이름, 바이트 단위 크기(값 타입은 `sizeof` 사용), 최솟값, 최댓값을 보여주는 표를 출력하는 프로그램을 작성하세요. `sbyte`, `byte`, `short`, `ushort`, `int`, `uint`, `long`, `ulong`, `float`, `double`, `decimal`을 포함하세요.

2. **부동소수점 비교**: `double`의 부동소수점 부정확성을 보여주는 프로그램을 작성하세요 (예: `0.1 + 0.2 != 0.3`). 그런 다음 `decimal`로 같은 계산을 수행하고 기대한 결과가 나오는지 확인하세요. 부동소수점 놀라움의 최소 세 가지 다른 예를 포함하세요.

3. **타입 변환 탐색기**: 사용자로부터 (`Console.ReadLine()`으로) 문자열 입력을 받은 후, `TryParse`를 사용하여 `int`, `double`, `bool`, `DateTime`으로 파싱을 시도하는 프로그램을 작성하세요. 어떤 변환이 성공했는지와 결과 값을 출력하세요.

4. **문자열 조작**: 전체 이름을 입력으로 받아 (예: "John Michael Smith") 다음을 출력하는 프로그램을 작성하세요: (a) 문자 수, (b) 대문자로 변환된 이름, (c) 뒤집힌 이름, (d) 각 단어를 별도로 대문자로 변환, (e) 이니셜 (예: "JMS"). `string` 메서드만 사용하세요 — `StringBuilder`는 아직 사용하지 않습니다.

5. **널러블 계산기**: 피연산자에 `int?`를 사용하는 계산기를 작성하세요. 사용자가 빈 문자열을 입력하면 `null`로 처리하세요. null 전파를 처리하세요: 어느 한쪽 피연산자가 `null`이면 결과도 `null`이어야 합니다. null 결과에는 "결과: N/A"를, 그 외에는 실제 값을 출력하세요.

---

**이전**: [시작하기](./01_Getting_Started.md) | **다음**: [연산자와 표현식](./03_Operators_and_Expressions.md)
