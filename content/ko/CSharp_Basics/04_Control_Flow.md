# 제어 흐름

**이전**: [연산자와 표현식](./03_Operators_and_Expressions.md) | **다음**: [메서드](./05_Methods.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `if`, `else if`, `else` 문을 사용하여 조건부 실행을 한다
2. 상수 패턴으로 `switch` 문을 작성한다
3. 간결한 값 매핑을 위해 `switch` 표현식을 사용한다
4. `for`, `foreach`, `while`, `do-while` 반복문으로 반복한다
5. `break`, `continue`, `return`으로 반복문 실행을 제어한다
6. 중첩 반복문을 다루고 레이블 점프를 위해 `goto`를 사용한다
7. `switch` 구조에서 기본적인 패턴 매칭(Pattern Matching)을 적용한다

---

제어 흐름 문은 문장이 실행되는 순서를 결정합니다. 제어 흐름 없이는 프로그램이 분기나 반복 없이 위에서 아래로 모든 줄을 실행할 것입니다. C#은 모든 프로그램에서 사용하게 될 풍부한 조건문과 반복문 구조를 제공합니다.

## 1. if 문

`if` 문은 조건이 `true`로 평가될 때만 코드 블록을 실행합니다:

### 1.1 단순 if

```csharp
int temperature = 35;

if (temperature > 30)
{
    Console.WriteLine("밖이 덥습니다!");
}
```

### 1.2 if-else

```csharp
int age = 16;

if (age >= 18)
{
    Console.WriteLine("당신은 성인입니다.");
}
else
{
    Console.WriteLine("당신은 미성년자입니다.");
}
```

### 1.3 if-else if-else 체인

```csharp
int score = 85;

if (score >= 90)
{
    Console.WriteLine("등급: A");
}
else if (score >= 80)
{
    Console.WriteLine("등급: B");
}
else if (score >= 70)
{
    Console.WriteLine("등급: C");
}
else if (score >= 60)
{
    Console.WriteLine("등급: D");
}
else
{
    Console.WriteLine("등급: F");
}
```

### 1.4 중첩 if 문

```csharp
int age = 25;
bool hasLicense = true;

if (age >= 18)
{
    if (hasLicense)
    {
        Console.WriteLine("운전할 수 있습니다.");
    }
    else
    {
        Console.WriteLine("먼저 면허가 필요합니다.");
    }
}
else
{
    Console.WriteLine("운전하기에 너무 어립니다.");
}

// 논리 AND를 사용한 평탄화 버전 (간단할 때 권장)
if (age >= 18 && hasLicense)
{
    Console.WriteLine("운전할 수 있습니다.");
}
```

### 1.5 한 줄 if (중괄호 없이)

```csharp
int x = 10;

// 합법적이지만 권장하지 않음 — 버그를 유발하기 쉬움
if (x > 5)
    Console.WriteLine("x는 5보다 큽니다");

// 항상 중괄호를 사용 — 더 명확하고 안전함
if (x > 5)
{
    Console.WriteLine("x는 5보다 큽니다");
}
```

### 1.6 조건부 대입 패턴

```csharp
// 패턴: 조건에 기반한 대입
int temperature = 28;
string description;

if (temperature > 30)
    description = "더움";
else if (temperature > 20)
    description = "따뜻함";
else if (temperature > 10)
    description = "시원함";
else
    description = "추움";

Console.WriteLine($"날씨는 {description}입니다.");

// 간단한 경우에는 삼항으로 더 짧게
string simple = temperature > 25 ? "따뜻함" : "따뜻하지 않음";
```

## 2. switch 문

`switch` 문은 단일 값을 여러 상수와 비교할 때 긴 `if-else if` 체인의 대안입니다:

### 2.1 상수를 사용한 기본 switch

```csharp
int dayNumber = 3;

switch (dayNumber)
{
    case 1:
        Console.WriteLine("월요일");
        break;
    case 2:
        Console.WriteLine("화요일");
        break;
    case 3:
        Console.WriteLine("수요일");
        break;
    case 4:
        Console.WriteLine("목요일");
        break;
    case 5:
        Console.WriteLine("금요일");
        break;
    case 6:
        Console.WriteLine("토요일");
        break;
    case 7:
        Console.WriteLine("일요일");
        break;
    default:
        Console.WriteLine("잘못된 요일 번호");
        break;
}
```

### 2.2 다중 케이스 레이블 (폴스루)

C#에서는 케이스 본문이 비어 있지 않은 한 한 케이스에서 다른 케이스로 폴스루(Fall-Through)할 수 없습니다. 여러 레이블이 같은 본문을 공유할 수 있습니다:

```csharp
int month = 4;

switch (month)
{
    case 1: case 3: case 5: case 7: case 8: case 10: case 12:
        Console.WriteLine("31일");
        break;
    case 4: case 6: case 9: case 11:
        Console.WriteLine("30일");
        break;
    case 2:
        Console.WriteLine("28일 또는 29일");
        break;
    default:
        Console.WriteLine("잘못된 월");
        break;
}
```

### 2.3 문자열을 사용한 switch

```csharp
string command = "start";

switch (command.ToLower())
{
    case "start":
        Console.WriteLine("엔진 시작 중...");
        break;
    case "stop":
        Console.WriteLine("엔진 정지 중...");
        break;
    case "pause":
        Console.WriteLine("일시 정지 중...");
        break;
    case "resume":
        Console.WriteLine("재개 중...");
        break;
    default:
        Console.WriteLine($"알 수 없는 명령: {command}");
        break;
}
```

### 2.4 when 가드가 있는 switch

```csharp
int number = 42;

switch (number)
{
    case int n when n < 0:
        Console.WriteLine("음수");
        break;
    case 0:
        Console.WriteLine("영");
        break;
    case int n when n > 0 && n <= 10:
        Console.WriteLine("작은 양수");
        break;
    case int n when n > 10 && n <= 100:
        Console.WriteLine("중간 양수");
        break;
    default:
        Console.WriteLine("큰 양수");
        break;
}
```

## 3. switch 표현식

C# 8에서는 값을 결과에 매핑하는 더 간결한 구문인 **switch 표현식(Switch Expression)**을 도입했습니다:

### 3.1 기본 switch 표현식

```csharp
int dayNumber = 5;

string dayName = dayNumber switch
{
    1 => "월요일",
    2 => "화요일",
    3 => "수요일",
    4 => "목요일",
    5 => "금요일",
    6 => "토요일",
    7 => "일요일",
    _ => "잘못됨"  // _는 폐기 패턴 (default와 같음)
};

Console.WriteLine(dayName);  // 금요일
```

### 3.2 조건이 있는 switch 표현식

```csharp
int score = 85;

string grade = score switch
{
    >= 90 => "A",
    >= 80 => "B",
    >= 70 => "C",
    >= 60 => "D",
    _ => "F"
};

Console.WriteLine($"점수 {score} = 등급 {grade}");  // 점수 85 = 등급 B
```

### 3.3 다중 패턴을 사용한 switch 표현식

```csharp
int month = 7;

int daysInMonth = month switch
{
    1 or 3 or 5 or 7 or 8 or 10 or 12 => 31,
    4 or 6 or 9 or 11 => 30,
    2 => 28,
    _ => throw new ArgumentException($"잘못된 월: {month}")
};

Console.WriteLine($"{month}월은 {daysInMonth}일입니다.");
```

### 3.4 튜플을 사용한 switch 표현식

```csharp
string season = (month: 7, hemisphere: "north") switch
{
    (>= 3 and <= 5, "north") => "봄",
    (>= 6 and <= 8, "north") => "여름",
    (>= 9 and <= 11, "north") => "가을",
    (12 or 1 or 2, "north") => "겨울",
    (>= 3 and <= 5, "south") => "가을",
    (>= 6 and <= 8, "south") => "겨울",
    (>= 9 and <= 11, "south") => "봄",
    (12 or 1 or 2, "south") => "여름",
    _ => "알 수 없음"
};

Console.WriteLine(season);  // 여름
```

## 4. for 반복문

`for` 반복문은 코드 블록을 특정 횟수만큼 반복합니다:

### 4.1 기본 for 반복문

```csharp
// 1부터 5까지 숫자 출력
for (int i = 1; i <= 5; i++)
{
    Console.WriteLine($"반복 {i}");
}

// 구조: for (초기화; 조건; 반복자)
// 1. 초기화는 반복문 전에 한 번 실행
// 2. 조건은 각 반복 전에 확인
// 3. 반복자는 각 반복 후에 실행
```

### 4.2 역방향 카운트

```csharp
for (int i = 10; i >= 1; i--)
{
    Console.Write($"{i} ");
}
Console.WriteLine("발사!");
// 출력: 10 9 8 7 6 5 4 3 2 1 발사!
```

### 4.3 스텝 크기

```csharp
// 2씩 증가
for (int i = 0; i <= 20; i += 2)
{
    Console.Write($"{i} ");
}
Console.WriteLine();
// 출력: 0 2 4 6 8 10 12 14 16 18 20

// 3씩 증가
for (int i = 0; i < 30; i += 3)
{
    Console.Write($"{i} ");
}
Console.WriteLine();
```

### 4.4 다중 변수

```csharp
// for 반복문에서 두 개의 변수
for (int i = 0, j = 10; i < j; i++, j--)
{
    Console.WriteLine($"i={i}, j={j}");
}
// 출력:
// i=0, j=10
// i=1, j=9
// i=2, j=8
// i=3, j=7
// i=4, j=6
```

### 4.5 무한 반복문

```csharp
// 무한 반복문 (break로 종료)
// for (;;)
// {
//     Console.Write("종료하려면 'quit'을 입력하세요: ");
//     string? input = Console.ReadLine();
//     if (input == "quit") break;
//     Console.WriteLine($"입력한 값: {input}");
// }
```

### 4.6 일반적인 패턴

```csharp
// 1부터 100까지의 합
int sum = 0;
for (int i = 1; i <= 100; i++)
{
    sum += i;
}
Console.WriteLine($"1-100 합계: {sum}");  // 5050

// 팩토리얼
int n = 10;
long factorial = 1;
for (int i = 2; i <= n; i++)
{
    factorial *= i;
}
Console.WriteLine($"{n}! = {factorial}");  // 3628800

// 2의 거듭제곱
for (int i = 0; i < 16; i++)
{
    Console.WriteLine($"2^{i} = {1 << i}");
}
```

## 5. foreach 반복문

`foreach` 반복문은 인덱스 관리 없이 컬렉션의 요소를 반복합니다:

```csharp
// 배열
int[] numbers = { 10, 20, 30, 40, 50 };
foreach (int num in numbers)
{
    Console.Write($"{num} ");
}
Console.WriteLine();

// 문자열 (문자별 반복)
string word = "Hello";
foreach (char ch in word)
{
    Console.Write($"'{ch}' ");
}
Console.WriteLine();
// 출력: 'H' 'e' 'l' 'l' 'o'

// 리스트
List<string> fruits = new() { "Apple", "Banana", "Cherry" };
foreach (string fruit in fruits)
{
    Console.WriteLine($"  - {fruit}");
}

// 딕셔너리
Dictionary<string, int> ages = new()
{
    ["Alice"] = 30,
    ["Bob"] = 25,
    ["Charlie"] = 35
};
foreach (KeyValuePair<string, int> kvp in ages)
{
    Console.WriteLine($"{kvp.Key}은(는) {kvp.Value}세입니다.");
}

// foreach에서 튜플 분해 (C# 7+)
foreach (var (name, age) in ages)
{
    Console.WriteLine($"{name}: {age}");
}

// 범위 (C#과 LINQ 사용)
foreach (int i in Enumerable.Range(1, 5))
{
    Console.Write($"{i} ");  // 1 2 3 4 5
}
Console.WriteLine();
```

### 5.1 foreach vs for: 언제 어떤 것을 사용할까

```csharp
int[] data = { 10, 20, 30, 40, 50 };

// 각 요소만 필요할 때 foreach 사용
foreach (int item in data)
{
    Console.WriteLine(item);
}

// 인덱스가 필요할 때 for 사용
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($"data[{i}] = {data[i]}");
}

// 요소를 수정해야 할 때 for 사용
for (int i = 0; i < data.Length; i++)
{
    data[i] *= 2;  // foreach로는 할 수 없음
}
```

## 6. while 반복문

`while` 반복문은 조건이 `true`인 동안 반복합니다. 조건은 각 반복 **전에** 확인됩니다:

```csharp
// 카운트 업
int count = 1;
while (count <= 5)
{
    Console.WriteLine($"카운트: {count}");
    count++;
}

// 유효한 입력을 받을 때까지 읽기
Console.Write("양수를 입력하세요: ");
int number = 0;
while (number <= 0)
{
    string? input = Console.ReadLine();
    if (int.TryParse(input, out number) && number > 0)
    {
        Console.WriteLine($"입력한 값: {number}");
    }
    else
    {
        number = 0;
        Console.Write("잘못됨. 다시 시도: ");
    }
}

// 콜라츠 추측
int n = 27;
int steps = 0;
Console.Write($"{n}");
while (n != 1)
{
    n = (n % 2 == 0) ? n / 2 : 3 * n + 1;
    Console.Write($" → {n}");
    steps++;
}
Console.WriteLine($"\n{steps}단계 만에 1에 도달.");
```

## 7. do-while 반복문

`do-while` 반복문은 `while`과 유사하지만 조건은 각 반복 **후에** 확인됩니다. 이는 최소 한 번의 실행을 보장합니다:

```csharp
// 항상 최소 한 번 실행
int count = 10;
do
{
    Console.WriteLine($"카운트: {count}");
    count++;
} while (count <= 5);
// 출력: 카운트: 10 (10 > 5임에도 한 번 실행)

// 메뉴 시스템 — do-while의 자연스러운 사용 사례
int choice;
do
{
    Console.WriteLine("\n=== 메뉴 ===");
    Console.WriteLine("1. 인사하기");
    Console.WriteLine("2. 날짜 표시");
    Console.WriteLine("3. 난수 표시");
    Console.WriteLine("0. 종료");
    Console.Write("선택: ");

    if (!int.TryParse(Console.ReadLine(), out choice))
    {
        choice = -1;
    }

    switch (choice)
    {
        case 1:
            Console.WriteLine("안녕하세요!");
            break;
        case 2:
            Console.WriteLine($"오늘: {DateTime.Now:yyyy-MM-dd}");
            break;
        case 3:
            Console.WriteLine($"난수: {Random.Shared.Next(1, 101)}");
            break;
        case 0:
            Console.WriteLine("안녕히 가세요!");
            break;
        default:
            Console.WriteLine("잘못된 선택입니다.");
            break;
    }
} while (choice != 0);
```

### 7.1 while vs do-while

```csharp
// while: 먼저 확인, 실행하지 않을 수 있음
int x = 10;
while (x < 5)
{
    Console.WriteLine("이것은 출력되지 않음");
    x++;
}

// do-while: 먼저 실행, 후에 확인
x = 10;
do
{
    Console.WriteLine("이것은 한 번 출력됨");  // 출력됨!
    x++;
} while (x < 5);
```

## 8. break, continue, return

### 8.1 break — 반복문 탈출

```csharp
// 50보다 큰 첫 번째 7의 배수 찾기
for (int i = 51; ; i++)
{
    if (i % 7 == 0)
    {
        Console.WriteLine($"찾음: {i}");  // 56
        break;
    }
}

// while 반복문에서 break
int sum = 0;
int num = 1;
while (true)
{
    sum += num;
    if (sum > 100)
    {
        Console.WriteLine($"num={num}에서 합이 100을 초과, sum={sum}");
        break;
    }
    num++;
}
```

### 8.2 continue — 다음 반복으로 건너뛰기

```csharp
// 홀수만 출력
for (int i = 1; i <= 20; i++)
{
    if (i % 2 == 0) continue;  // 짝수 건너뛰기
    Console.Write($"{i} ");
}
Console.WriteLine();
// 출력: 1 3 5 7 9 11 13 15 17 19

// 입력 처리 시 빈 줄 건너뛰기
string[] lines = { "Hello", "", "World", "  ", "C#" };
foreach (string line in lines)
{
    if (string.IsNullOrWhiteSpace(line)) continue;
    Console.WriteLine($"처리 중: '{line}'");
}
```

### 8.3 return — 메서드 종료

```csharp
// return은 반복문이 아닌 전체 메서드를 종료
int FindFirst(int[] arr, int target)
{
    for (int i = 0; i < arr.Length; i++)
    {
        if (arr[i] == target)
            return i;  // 즉시 메서드 종료
    }
    return -1;  // 찾지 못함
}

int[] data = { 5, 3, 8, 1, 9 };
int index = FindFirst(data, 8);
Console.WriteLine($"인덱스 {index}에서 찾음");  // 2
```

## 9. 중첩 반복문과 goto

### 9.1 중첩 반복문

```csharp
// 구구단
for (int i = 1; i <= 9; i++)
{
    for (int j = 1; j <= 9; j++)
    {
        Console.Write($"{i * j,4}");
    }
    Console.WriteLine();
}

// 삼각형 패턴
for (int i = 1; i <= 5; i++)
{
    for (int j = 0; j < i; j++)
    {
        Console.Write("* ");
    }
    Console.WriteLine();
}
// 출력:
// *
// * *
// * * *
// * * * *
// * * * * *
```

### 9.2 goto를 사용한 중첩 반복문 탈출

C#에서 `break`는 가장 안쪽 반복문만 탈출합니다. 여러 반복문을 탈출하려면 레이블과 함께 `goto`를 사용할 수 있습니다:

```csharp
// i * j == 42인 첫 번째 쌍 (i, j) 찾기
for (int i = 1; i <= 10; i++)
{
    for (int j = 1; j <= 10; j++)
    {
        if (i * j == 42)
        {
            Console.WriteLine($"찾음: {i} * {j} = 42");
            goto FoundIt;  // 두 반복문 모두 탈출
        }
    }
}
Console.WriteLine("찾지 못함.");
FoundIt:
Console.WriteLine("검색 완료.");

// 대안: 플래그 변수 사용 (goto 없이)
bool found = false;
for (int i = 1; i <= 10 && !found; i++)
{
    for (int j = 1; j <= 10 && !found; j++)
    {
        if (i * j == 42)
        {
            Console.WriteLine($"찾음: {i} * {j} = 42");
            found = true;
        }
    }
}

// 대안: 메서드로 추출하고 return 사용
(int i, int j) FindProduct(int target)
{
    for (int i = 1; i <= 10; i++)
        for (int j = 1; j <= 10; j++)
            if (i * j == target)
                return (i, j);
    return (-1, -1);
}

var result = FindProduct(42);
Console.WriteLine($"찾음: {result.i} * {result.j} = 42");
```

## 10. switch에서의 패턴 매칭 기초

C#은 `switch`를 단순한 상수 비교를 넘어 확장하는 강력한 패턴 매칭을 제공합니다:

### 10.1 타입 패턴

```csharp
object value = 42;

switch (value)
{
    case int i:
        Console.WriteLine($"정수: {i}");
        break;
    case string s:
        Console.WriteLine($"문자열: {s}");
        break;
    case double d:
        Console.WriteLine($"실수: {d}");
        break;
    case null:
        Console.WriteLine("Null 값");
        break;
    default:
        Console.WriteLine($"기타 타입: {value.GetType().Name}");
        break;
}
```

### 10.2 관계 패턴 (C# 9+)

```csharp
int temperature = 25;

string description = temperature switch
{
    < 0 => "영하",
    >= 0 and < 10 => "추움",
    >= 10 and < 20 => "시원함",
    >= 20 and < 30 => "따뜻함",
    >= 30 and < 40 => "더움",
    >= 40 => "극심한 더위",
};

Console.WriteLine($"{temperature}°C: {description}");
```

### 10.3 논리 패턴 (and, or, not)

```csharp
char ch = 'A';

string category = ch switch
{
    >= 'a' and <= 'z' => "소문자",
    >= 'A' and <= 'Z' => "대문자",
    >= '0' and <= '9' => "숫자",
    ' ' or '\t' or '\n' => "공백",
    not (' ' or '\t' or '\n') and (>= ' ' and <= '~') => "기호",
    _ => "기타"
};

Console.WriteLine($"'{ch}'은(는) {category}");  // 'A'은(는) 대문자
```

### 10.4 프로퍼티 패턴

```csharp
// 프로퍼티 패턴 미리보기 (클래스와 함께 더 많이 사용)
var point = new { X = 3, Y = 4 };

string quadrant = point switch
{
    { X: 0, Y: 0 } => "원점",
    { X: > 0, Y: > 0 } => "제1사분면",
    { X: < 0, Y: > 0 } => "제2사분면",
    { X: < 0, Y: < 0 } => "제3사분면",
    { X: > 0, Y: < 0 } => "제4사분면",
    { X: 0 } or { Y: 0 } => "축 위",
    _ => "알 수 없음"
};

Console.WriteLine($"({point.X}, {point.Y})은(는) {quadrant}");
```

## 11. 연습 문제

1. **FizzBuzz**: 1부터 100까지의 숫자를 출력하는 프로그램을 작성하세요. 3의 배수에는 숫자 대신 "Fizz"를 출력합니다. 5의 배수에는 "Buzz"를 출력합니다. 3과 5 모두의 배수에는 "FizzBuzz"를 출력합니다. `for` 반복문과 `if`/`else if` 문을 사용하세요.

2. **숫자 맞추기 게임**: `Random.Shared.Next(1, 101)`을 사용하여 1과 100 사이의 난수를 생성하세요. `do-while` 반복문으로 사용자에게 반복적으로 추측을 요청하세요. 각 추측 후 "너무 높음", "너무 낮음", 또는 "정답!"을 출력하세요. 시도 횟수를 세고 표시하세요.

3. **소수 찾기**: 2와 200 사이의 모든 소수를 찾아 출력하는 프로그램을 작성하세요. 중첩 반복문을 사용하세요: 외부 반복문은 후보를 반복하고 내부 반복문은 약수를 확인합니다. 내부 반복문을 최적화하기 위해 `break`를 사용하세요.

4. **다이아몬드 패턴**: 중첩 `for` 반복문을 사용하여 별표로 다이아몬드 패턴을 출력하세요. 프로그램은 사용자로부터 홀수 `n`을 입력받아 가장 넓은 지점에서 `n`행인 다이아몬드를 출력해야 합니다. 예를 들어 `n = 5`인 경우:
   ```
     *
    ***
   *****
    ***
     *
   ```

5. **switch 표현식을 사용한 미니 계산기**: 사용자로부터 두 숫자와 연산자(`+`, `-`, `*`, `/`, `%`)를 읽는 계산기를 작성하세요. `switch` 표현식을 사용하여 연산을 수행하세요. 0으로 나누기, 잘못된 연산자, 숫자가 아닌 입력을 우아하게 처리하세요.

---

**이전**: [연산자와 표현식](./03_Operators_and_Expressions.md) | **다음**: [메서드](./05_Methods.md)
