# 배열과 문자열

**이전**: [메서드](./05_Methods.md) | **다음**: [열거형과 구조체](./07_Enums_and_Structs.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 1차원, 다차원, 가변 배열(Jagged Array)을 선언, 초기화, 조작하기
2. 정렬, 검색, 복사를 위한 내장 배열 메서드 사용하기
3. 배열 슬라이싱을 위한 범위(Range)와 인덱스(Index) 연산자 적용하기
4. 문자열의 불변(Immutable) 특성을 이해하고 효과적으로 작업하기
5. 문자열 보간, 축어 문자열(Verbatim String), 원시 문자열 리터럴(Raw String Literal) 사용하기
6. StringBuilder로 효율적으로 문자열 구성하기
7. 문자 수준 연산 수행 및 유니코드 기초 이해하기

---

배열(Array)과 문자열(String)은 C#에서 가장 자주 사용되는 두 가지 데이터 구조입니다. 배열은 동일한 타입의 요소를 저장하기 위한 고정 크기의 인덱스 기반 컬렉션을 제공하고, 문자열은 풍부한 조작 메서드를 갖춘 문자 시퀀스를 나타냅니다. 두 가지를 깊이 이해하는 것은 효율적이고 올바른 C# 프로그램을 작성하는 데 필수적입니다.

## 1. 1차원 배열

1차원 배열은 연속된 메모리 블록에 요소를 저장하며, 0부터 시작하는 정수 인덱스로 접근합니다.

### 1.1 선언과 초기화

C#에서 배열을 생성하는 방법은 여러 가지가 있습니다:

```csharp
// 기본값(int의 경우 0)으로 선언 및 할당
int[] numbers = new int[5];

// 값과 함께 선언 및 초기화
int[] primes = new int[] { 2, 3, 5, 7, 11 };

// 축약 초기화 (왼쪽에서 타입 추론)
int[] fibonacci = { 1, 1, 2, 3, 5, 8, 13 };

// var와 명시적 new 사용
var scores = new int[] { 95, 87, 72, 91, 68 };

// 대상 타입 new (C# 9+)
int[] grades = new[] { 90, 85, 78, 92 };
```

### 1.2 요소 접근 및 수정

요소는 0부터 시작하는 인덱스와 대괄호 표기법으로 접근합니다:

```csharp
int[] data = { 10, 20, 30, 40, 50 };

// 요소 읽기
int first = data[0];    // 10
int third = data[2];    // 30

// 요소 수정
data[1] = 25;           // 배열이 이제 { 10, 25, 30, 40, 50 }

// 배열 길이
int length = data.Length; // 5

// for 루프로 반복
for (int i = 0; i < data.Length; i++)
{
    Console.WriteLine($"data[{i}] = {data[i]}");
}

// foreach로 반복
foreach (int value in data)
{
    Console.WriteLine(value);
}
```

### 1.3 끝에서부터 인덱스 연산자

C# 8에서 배열 끝에서부터 인덱싱하기 위한 `^` 연산자가 도입되었습니다:

```csharp
int[] values = { 10, 20, 30, 40, 50 };

int last = values[^1];       // 50 (마지막 요소)
int secondLast = values[^2]; // 40
int first = values[^5];      // 10 (길이 5인 배열에서 values[0]과 동일)

// 루프에서 유용
for (int i = 1; i <= values.Length; i++)
{
    Console.WriteLine(values[^i]);
}
// 출력: 50, 40, 30, 20, 10
```

### 1.4 범위 검사

유효 범위 밖의 인덱스에 접근하면 `IndexOutOfRangeException`이 발생합니다:

```csharp
int[] arr = { 1, 2, 3 };

try
{
    int invalid = arr[5]; // IndexOutOfRangeException 발생
}
catch (IndexOutOfRangeException ex)
{
    Console.WriteLine($"오류: {ex.Message}");
}
```

## 2. 다차원 배열

C#은 모든 행이 동일한 열 수를 가지는 직사각형(다차원) 배열을 지원합니다.

### 2.1 2차원 배열

```csharp
// 3x4 행렬 선언
int[,] matrix = new int[3, 4];

// 값으로 초기화
int[,] grid = {
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 10, 11, 12 }
};

// 요소 접근
int element = grid[1, 2]; // 7 (1행, 2열)

// 요소 수정
grid[0, 0] = 100;

// 차원 크기 가져오기
int rows = grid.GetLength(0);    // 3
int cols = grid.GetLength(1);    // 4
int totalElements = grid.Length;  // 12

// 모든 요소 반복
for (int r = 0; r < rows; r++)
{
    for (int c = 0; c < cols; c++)
    {
        Console.Write($"{grid[r, c],4}");
    }
    Console.WriteLine();
}
```

### 2.2 3차원 배열

```csharp
// 2x3x4 3차원 배열
int[,,] cube = new int[2, 3, 4];

cube[0, 1, 2] = 42;

int depth = cube.GetLength(0);  // 2
int rows = cube.GetLength(1);   // 3
int cols = cube.GetLength(2);   // 4
```

### 2.3 다차원 배열의 제한사항

직사각형 배열은 모든 메모리를 연속적으로 할당하여 좋은 캐시 지역성(Cache Locality)을 제공하지만, 모든 행이 동일한 열 수를 가져야 합니다. 또한 쉽게 크기를 변경할 수 없고 `IEnumerable<T>`를 기대하는 많은 LINQ 메서드와 함께 사용하기 어렵습니다.

## 3. 가변 배열 (Jagged Array)

가변 배열은 배열의 배열입니다. 각 내부 배열은 서로 다른 길이를 가질 수 있습니다.

### 3.1 선언과 초기화

```csharp
// 3개 행을 가진 가변 배열 선언
int[][] jagged = new int[3][];

// 각 행을 독립적으로 초기화
jagged[0] = new int[] { 1, 2, 3 };
jagged[1] = new int[] { 4, 5 };
jagged[2] = new int[] { 6, 7, 8, 9 };

// 축약 초기화
int[][] triangle = {
    new[] { 1 },
    new[] { 1, 1 },
    new[] { 1, 2, 1 },
    new[] { 1, 3, 3, 1 },
    new[] { 1, 4, 6, 4, 1 }
};
```

### 3.2 요소 접근

```csharp
int[][] jagged = {
    new[] { 10, 20, 30 },
    new[] { 40, 50 },
    new[] { 60, 70, 80, 90 }
};

// 접근: 첫 번째 괄호가 행을, 두 번째 괄호가 열을 선택
int value = jagged[2][1]; // 70

// 반복
for (int i = 0; i < jagged.Length; i++)
{
    Console.Write($"행 {i}: ");
    for (int j = 0; j < jagged[i].Length; j++)
    {
        Console.Write($"{jagged[i][j]} ");
    }
    Console.WriteLine();
}
```

### 3.3 가변 배열 vs 다차원 배열

| 특성 | 다차원 (`int[,]`) | 가변 (`int[][]`) |
|------|-------------------|-----------------|
| 행 크기 | 모두 동일 | 다를 수 있음 |
| 메모리 레이아웃 | 단일 연속 블록 | 배열 참조의 배열 |
| 성능 | 더 나은 캐시 지역성 | 약간 더 많은 간접 참조 |
| LINQ 호환성 | 제한적 | LINQ와 함께 사용 가능 |
| CLR 최적화 | 덜 최적화됨 | 더 나은 JIT 최적화 |

실제로 가변 배열은 JIT 컴파일러가 더 적극적으로 최적화하기 때문에 성능에 민감한 코드에서 선호되는 경우가 많습니다.

## 4. 배열 메서드

`System.Array` 클래스는 배열 조작을 위한 많은 유용한 정적 메서드를 제공합니다.

### 4.1 정렬과 역순

```csharp
int[] numbers = { 5, 3, 8, 1, 9, 2, 7 };

// 오름차순 정렬
Array.Sort(numbers);
// numbers: { 1, 2, 3, 5, 7, 8, 9 }

// 배열 역순
Array.Reverse(numbers);
// numbers: { 9, 8, 7, 5, 3, 2, 1 }

// 배열 일부 정렬 (인덱스 2, 개수 3)
int[] partial = { 50, 40, 30, 20, 10 };
Array.Sort(partial, 1, 3); // 인덱스 1, 2, 3의 요소를 정렬
// partial: { 50, 20, 30, 40, 10 }
```

### 4.2 사용자 정의 비교로 정렬

```csharp
string[] names = { "Charlie", "Alice", "Bob", "Diana" };

// 알파벳순 정렬 (기본)
Array.Sort(names);
// names: { "Alice", "Bob", "Charlie", "Diana" }

// Comparison<T> 대리자를 사용하여 문자열 길이로 정렬
Array.Sort(names, (a, b) => a.Length.CompareTo(b.Length));
// names: { "Bob", "Alice", "Diana", "Charlie" }

// 병렬 배열을 함께 정렬
int[] ids = { 3, 1, 4, 2 };
string[] labels = { "C", "A", "D", "B" };
Array.Sort(ids, labels);
// ids:    { 1, 2, 3, 4 }
// labels: { "A", "B", "C", "D" }
```

### 4.3 검색

```csharp
int[] sorted = { 1, 3, 5, 7, 9, 11, 13 };

// BinarySearch (배열이 정렬되어 있어야 함)
int index = Array.BinarySearch(sorted, 7); // 3

// IndexOf와 LastIndexOf (정렬되지 않은 배열에서도 작동)
int[] data = { 10, 20, 30, 20, 40 };
int firstIndex = Array.IndexOf(data, 20);    // 1
int lastIndex = Array.LastIndexOf(data, 20); // 3
int notFound = Array.IndexOf(data, 99);      // -1

// Exists와 Find
int[] numbers = { 2, 4, 6, 7, 8, 10 };
bool hasOdd = Array.Exists(numbers, n => n % 2 != 0);  // true
int firstOdd = Array.Find(numbers, n => n % 2 != 0);   // 7
int[] allEven = Array.FindAll(numbers, n => n % 2 == 0); // { 2, 4, 6, 8, 10 }
```

### 4.4 복사와 크기 변경

```csharp
int[] source = { 1, 2, 3, 4, 5 };

// 새 배열로 복사
int[] dest = new int[5];
Array.Copy(source, dest, 5);

// 범위 복사 (소스 인덱스 1, 대상 인덱스 2, 개수 3)
int[] partial = new int[7];
Array.Copy(source, 1, partial, 2, 3);
// partial: { 0, 0, 2, 3, 4, 0, 0 }

// Clone (얕은 복사)
int[] cloned = (int[])source.Clone();

// Resize (내부적으로 새 배열 생성)
int[] growable = { 1, 2, 3 };
Array.Resize(ref growable, 6);
// growable: { 1, 2, 3, 0, 0, 0 }

Array.Resize(ref growable, 2);
// growable: { 1, 2 }
```

### 4.5 채우기와 지우기

```csharp
int[] buffer = new int[10];

// 전체 배열을 값으로 채우기
Array.Fill(buffer, -1);
// buffer: { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 }

// 범위 채우기 (인덱스 3부터, 개수 4)
Array.Fill(buffer, 42, 3, 4);
// buffer: { -1, -1, -1, 42, 42, 42, 42, -1, -1, -1 }

// 요소 지우기 (기본값으로 리셋)
Array.Clear(buffer, 0, buffer.Length);
// buffer: { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
```

## 5. 범위를 이용한 배열 슬라이싱

C# 8에서 배열 슬라이스를 생성하기 위한 범위 연산자 `..`가 도입되었습니다.

### 5.1 범위 구문

```csharp
int[] arr = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

// 범위: start..end (start 포함, end 미포함)
int[] slice1 = arr[1..4];   // { 1, 2, 3 }
int[] slice2 = arr[..3];    // { 0, 1, 2 } (처음부터)
int[] slice3 = arr[7..];    // { 7, 8, 9 } (끝까지)
int[] slice4 = arr[..];     // { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 } (전체 복사)

// 끝에서부터 인덱스와 결합
int[] slice5 = arr[^3..];   // { 7, 8, 9 } (마지막 3개 요소)
int[] slice6 = arr[1..^1];  // { 1, 2, 3, 4, 5, 6, 7, 8 }
int[] slice7 = arr[^5..^2]; // { 5, 6, 7 }
```

### 5.2 범위 변수

범위는 `Range` 타입의 변수에 저장할 수 있습니다:

```csharp
int[] data = { 10, 20, 30, 40, 50, 60, 70 };

Range middle = 2..5;
int[] middleSlice = data[middle]; // { 30, 40, 50 }

Range lastThree = ^3..;
int[] tail = data[lastThree]; // { 50, 60, 70 }
```

### 5.3 슬라이싱에 대한 중요 사항

슬라이싱은 항상 뷰가 아닌 새 배열(복사본)을 생성합니다. 할당 없이 뷰가 필요하면 `Span<T>` 또는 `Memory<T>`를 사용하세요:

```csharp
int[] original = { 1, 2, 3, 4, 5 };

// 이것은 새 배열을 생성
int[] copy = original[1..4];
copy[0] = 99;
Console.WriteLine(original[1]); // 여전히 2

// Span은 뷰를 제공 (할당 없음)
Span<int> view = original.AsSpan(1, 3);
view[0] = 99;
Console.WriteLine(original[1]); // 이제 99
```

## 6. 문자열 기초

C#에서 `string`은 `System.String`의 별칭입니다. 문자열은 참조 타입이지만 불변성(Immutability)으로 인해 값과 유사한 의미론을 가집니다.

### 6.1 문자열 불변성

문자열을 수정하는 것처럼 보이는 모든 연산은 실제로 새로운 문자열 객체를 생성합니다:

```csharp
string greeting = "Hello";
string modified = greeting.Replace("H", "J"); // "Jello"

// greeting은 여전히 "Hello" (변경되지 않음)
Console.WriteLine(greeting);  // "Hello"
Console.WriteLine(modified);  // "Jello"

// 문자열 연결은 새 객체를 생성
string a = "Hello";
string b = a + " World"; // 새 문자열 할당
// a는 여전히 "Hello"
```

### 6.2 일반적인 문자열 메서드

```csharp
string text = "  Hello, World!  ";

// 길이
int len = text.Length; // 17 (공백 포함)

// 트리밍(공백 제거)
string trimmed = text.Trim();       // "Hello, World!"
string trimStart = text.TrimStart(); // "Hello, World!  "
string trimEnd = text.TrimEnd();     // "  Hello, World!"

// 대소문자 변환
string upper = trimmed.ToUpper();   // "HELLO, WORLD!"
string lower = trimmed.ToLower();   // "hello, world!"

// 검색
bool contains = trimmed.Contains("World");    // true
bool starts = trimmed.StartsWith("Hello");    // true
bool ends = trimmed.EndsWith("!");            // true
int index = trimmed.IndexOf("World");         // 7
int lastIndex = trimmed.LastIndexOf('l');     // 10

// 부분 문자열
string sub = trimmed.Substring(7, 5); // "World"

// 치환
string replaced = trimmed.Replace("World", "C#"); // "Hello, C#!"

// 분할
string csv = "apple,banana,cherry";
string[] fruits = csv.Split(',');
// fruits: { "apple", "banana", "cherry" }

// 결합
string joined = string.Join(" | ", fruits);
// "apple | banana | cherry"
```

### 6.3 문자열 비교

```csharp
string a = "hello";
string b = "Hello";

// 대소문자 구분 비교
bool equal1 = a == b;                    // false
bool equal2 = a.Equals(b);              // false

// 대소문자 무시 비교
bool equal3 = a.Equals(b, StringComparison.OrdinalIgnoreCase); // true
bool equal4 = string.Equals(a, b, StringComparison.OrdinalIgnoreCase); // true

// 순서 비교
int result = string.Compare(a, b, StringComparison.Ordinal);
// result > 0 (서수 비교에서 소문자 'h' > 대문자 'H')

// null 안전 비교
string? maybeNull = null;
bool isNull = string.IsNullOrEmpty(maybeNull);      // true
bool isBlank = string.IsNullOrWhiteSpace("   ");    // true
```

## 7. 문자열 보간과 특수 문자열

### 7.1 문자열 보간

문자열 보간(C# 6에서 도입)은 문자열에 표현식을 읽기 쉽게 포함하는 방법을 제공합니다:

```csharp
string name = "Alice";
int age = 30;
double gpa = 3.856;

// 기본 보간
string intro = $"제 이름은 {name}이고 {age}살입니다.";

// 중괄호 안의 표현식
string info = $"내년에 저는 {age + 1}살이 됩니다.";

// 서식 지정자
string formatted = $"GPA: {gpa:F2}";          // "GPA: 3.86"
string padded = $"이름: {name,-10} 나이: {age,5}"; // 이름 왼쪽 정렬, 나이 오른쪽 정렬

// 정렬과 서식 결합
string currency = $"가격: {19.99m,10:C}"; // "가격:     $19.99"

// 삼항 표현식 (가독성을 위해 괄호 사용)
string status = $"상태: {(age >= 18 ? "성인" : "미성년자")}";

// 날짜 서식
DateTime now = DateTime.Now;
string dateStr = $"오늘: {now:yyyy-MM-dd}";
```

### 7.2 축어 문자열 (Verbatim String)

축어 문자열(`@` 접두사)은 백슬래시를 리터럴 문자로 처리하며 여러 줄에 걸칠 수 있습니다:

```csharp
// 일반 문자열은 이스케이프 시퀀스 필요
string path1 = "C:\\Users\\Alice\\Documents\\file.txt";

// 축어 문자열은 백슬래시를 그대로 처리
string path2 = @"C:\Users\Alice\Documents\file.txt";

// 여러 줄 축어 문자열
string poem = @"장미는 빨갛고,
제비꽃은 파랗고,
C#은 훌륭하고,
당신도 그렇습니다.";

// 축어 문자열에 따옴표를 포함하려면 두 번 쓰기
string quoted = @"그녀가 ""안녕""이라고 말했다.";

// 축어와 보간 결합
string user = "Alice";
string fullPath = $@"C:\Users\{user}\Documents";
// 또는 C# 8+ 에서 동등하게:
string fullPath2 = @$"C:\Users\{user}\Documents";
```

### 7.3 원시 문자열 리터럴 (C# 11)

원시 문자열 리터럴은 최소 세 개의 큰따옴표 문자를 사용하며 이스케이프 시퀀스의 필요성을 완전히 없앱니다:

```csharp
// 기본 원시 문자열 리터럴
string json = """
    {
        "name": "Alice",
        "age": 30,
        "hobbies": ["reading", "coding"]
    }
    """;

// 닫는 """의 들여쓰기가 기준 들여쓰기를 결정
// (해당 열까지의 선행 공백이 제거됨)

// 원시 보간 문자열 (각 중괄호 수준마다 추가 $ 사용)
string name = "Alice";
int age = 30;

string rawInterpolated = $$"""
    {
        "name": "{{name}}",
        "age": {{age}}
    }
    """;
// 이중 $$는 보간이 { } 대신 {{ }}를 사용함을 의미
// JSON 중괄호와의 충돌을 방지

// 한 줄 원시 문자열
string singleLine = """이것은 이스케이프 없이 "따옴표"를 포함합니다.""";
```

## 8. StringBuilder

반복적인 연결을 통해 문자열을 구성할 때, `StringBuilder`는 많은 중간 문자열 객체 생성의 오버헤드를 피합니다.

### 8.1 기본 사용법

```csharp
using System.Text;

// 비효율적: 많은 중간 문자열 생성
string result = "";
for (int i = 0; i < 1000; i++)
{
    result += i.ToString() + ", "; // 나쁨: O(n^2) 동작
}

// 효율적: StringBuilder가 버퍼를 제자리에서 수정
var sb = new StringBuilder();
for (int i = 0; i < 1000; i++)
{
    sb.Append(i);
    sb.Append(", ");
}
string efficient = sb.ToString();
```

### 8.2 StringBuilder 메서드

```csharp
var sb = new StringBuilder("Hello");

// 추가
sb.Append(" World");           // "Hello World"
sb.AppendLine("!");            // "Hello World!\n"
sb.AppendFormat("{0:C}", 9.99); // "Hello World!\n$9.99"

// 삽입
sb.Insert(5, ",");             // 위치 5에 쉼표 삽입

// 치환
sb.Replace("World", "C#");

// 제거
sb.Remove(0, 6);               // 인덱스 0부터 6개 문자 제거

// 인덱서 접근
char ch = sb[0];
sb[0] = 'X';

// 길이와 용량
int len = sb.Length;
int cap = sb.Capacity;

// 지우기
sb.Clear();

// 체이닝 (메서드가 동일한 StringBuilder를 반환)
string output = new StringBuilder()
    .Append("이름: ")
    .Append("Alice")
    .Append(", 나이: ")
    .Append(30)
    .ToString();
// "이름: Alice, 나이: 30"
```

### 8.3 StringBuilder 사용 시점

- 소수의 고정된 조각을 연결할 때는 `string` 연결 사용 (예: `a + b + c`)
- 루프에서 연결하거나 많은 동적 부분으로 문자열을 구성할 때는 `StringBuilder` 사용
- 구분자로 배열이나 컬렉션을 결합할 때는 `string.Join` 사용
- 몇 개의 포함된 값으로 가독성이 필요할 때는 문자열 보간 사용

## 9. Span과 문자 연산

### 9.1 문자열 슬라이싱을 위한 Span

`Span<char>`과 `ReadOnlySpan<char>`을 사용하면 새 문자열 객체를 할당하지 않고 부분 문자열 작업을 할 수 있습니다:

```csharp
string longText = "The quick brown fox jumps over the lazy dog";

// 할당 없이 슬라이스
ReadOnlySpan<char> quick = longText.AsSpan(4, 5); // "quick"

// 할당 없이 숫자 파싱
ReadOnlySpan<char> numberSpan = "12345".AsSpan(1, 3);
int parsed = int.Parse(numberSpan); // 234

// Span으로 분할 (배열 할당 없음)
ReadOnlySpan<char> csv = "a,b,c,d".AsSpan();
// 수동 스캔 또는 MemoryExtensions.IndexOf 사용
int commaIndex = csv.IndexOf(',');
ReadOnlySpan<char> first = csv[..commaIndex]; // "a"
```

### 9.2 문자 연산

`char` 타입은 단일 UTF-16 코드 단위를 나타냅니다. `char` 구조체는 유용한 분류 메서드를 제공합니다:

```csharp
char letter = 'A';
char digit = '7';
char space = ' ';
char symbol = '#';

// 분류 메서드
bool isLetter = char.IsLetter(letter);         // true
bool isDigit = char.IsDigit(digit);            // true
bool isWhiteSpace = char.IsWhiteSpace(space);  // true
bool isUpper = char.IsUpper(letter);           // true
bool isLower = char.IsLower('a');              // true
bool isLetterOrDigit = char.IsLetterOrDigit('x'); // true
bool isPunctuation = char.IsPunctuation(',');  // true

// 변환
char upper = char.ToUpper('a'); // 'A'
char lower = char.ToLower('Z'); // 'z'

// 숫자 값
int numericValue = (int)char.GetNumericValue('7'); // 7

// 문자열의 문자 반복
string word = "Hello";
foreach (char c in word)
{
    Console.Write($"{c}({(int)c}) "); // H(72) e(101) l(108) l(108) o(111)
}
```

### 9.3 유니코드 기초

C# 문자열은 UTF-16 코드 단위의 시퀀스입니다. 대부분의 문자는 하나의 `char`을 사용하지만, 일부(이모지나 희귀한 CJK 문자 등)는 서로게이트 쌍(Surrogate Pair, 두 개의 `char` 값)이 필요합니다:

```csharp
// 기본 다국어 평면 문자 (단일 char)
string ascii = "Hello";         // 5 chars, 5 코드 포인트
string korean = "안녕하세요";     // 5 chars, 5 코드 포인트

// 보충 문자 (서로게이트 쌍)
string emoji = "😀";
Console.WriteLine(emoji.Length);          // 2 (서로게이트 쌍)

// 정확한 문자 수 계산을 위해 StringInfo 사용
using System.Globalization;
var info = new StringInfo(emoji);
Console.WriteLine(info.LengthInTextElements); // 1

// 텍스트 요소(자소 클러스터) 열거
string mixed = "Hello😀World";
var enumerator = StringInfo.GetTextElementEnumerator(mixed);
while (enumerator.MoveNext())
{
    Console.Write($"[{enumerator.GetTextElement()}]");
}
// [H][e][l][l][o][😀][W][o][r][l][d]
```

## 10. 연습 문제

1. **배열 회전**: 주어진 위치 수만큼 배열을 왼쪽으로 회전시키는 메서드 `void RotateLeft(int[] arr, int positions)`를 작성하세요. 예를 들어, `{1, 2, 3, 4, 5}`를 2만큼 회전하면 `{3, 4, 5, 1, 2}`가 됩니다. 범위를 이용한 배열 슬라이싱을 사용하세요.

2. **행렬 전치**: 직사각형 2D 배열의 전치를 반환하는 메서드 `int[,] Transpose(int[,] matrix)`를 작성하세요. 결과의 행 `i`, 열 `j`는 입력의 행 `j`, 열 `i`와 같아야 합니다.

3. **단어 빈도 카운터**: 문자열을 단어로 분할하고(공백과 구두점으로 구분), 소문자로 변환한 후, 단어 빈도의 딕셔너리를 반환하는 메서드 `Dictionary<string, int> CountWords(string text)`를 작성하세요. `string.Split`과 적절한 `StringSplitOptions`를 사용하세요.

4. **회문 검사기**: 대소문자, 공백, 구두점을 무시하고 문자열이 회문(Palindrome)인지 확인하는 메서드 `bool IsPalindrome(string text)`를 작성하세요. 예를 들어, `"A man, a plan, a canal: Panama"`는 `true`를 반환해야 합니다. 필터링에 `char.IsLetterOrDigit`를 사용하세요.

5. **CSV 파서**: 여러 줄의 CSV 문자열을 가변 배열로 파싱하는 메서드 `string[][] ParseCsv(string csvContent)`를 작성하세요. 필드가 쉼표로 구분되고 행이 줄바꿈으로 구분되는 기본 사례를 처리하세요. `string.Split`을 사용하고 각 내부 배열이 한 행을 나타내는 가변 배열을 반환하세요.
