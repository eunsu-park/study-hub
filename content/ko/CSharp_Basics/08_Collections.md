# 컬렉션

**이전**: [열거형과 구조체](./07_Enums_and_Structs.md) | **다음**: [클래스와 객체](./09_Classes_and_Objects.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 동적 인덱스 기반 컬렉션으로 `List<T>` 사용하기
2. `Dictionary<TKey, TValue>`로 키-값 쌍 저장 및 검색하기
3. `HashSet<T>`로 집합 연산 수행하기
4. FIFO 및 LIFO 시나리오에 `Queue<T>`와 `Stack<T>` 적용하기
5. `IEnumerable<T>`와 `foreach`로 컬렉션 반복하기
6. 컬렉션 초기화자와 컬렉션 표현식 사용하기
7. 데이터를 필터링, 투영, 집계하기 위한 기본 LINQ 쿼리 작성하기
8. 성능 특성에 따라 적절한 컬렉션 선택하기

---

.NET 기본 클래스 라이브러리는 `System.Collections.Generic` 네임스페이스에서 풍부한 제네릭 컬렉션 타입을 제공합니다. 이러한 컬렉션은 타입 안전하고, 효율적이며, 거의 모든 데이터 구조 요구를 충족합니다. 각 컬렉션을 언제 어떻게 사용하는지 이해하는 것은 성능 좋은 C# 코드를 작성하기 위한 핵심 기술입니다.

## 1. List\<T\>

`List<T>`는 가장 일반적으로 사용되는 컬렉션입니다. 인덱스 접근, 삽입, 제거, 검색을 제공하는 동적으로 크기가 조정되는 배열입니다.

### 1.1 생성과 채우기

```csharp
// 빈 리스트
List<int> numbers = new List<int>();

// 초기 용량 지정 (크기 조정 방지)
List<int> preallocated = new List<int>(100);

// 컬렉션 초기화자
List<string> fruits = new List<string> { "Apple", "Banana", "Cherry" };

// 대상 타입 new
List<double> scores = new() { 95.5, 87.2, 91.0 };

// 기존 배열로부터
int[] array = { 1, 2, 3, 4, 5 };
List<int> fromArray = new List<int>(array);

// AddRange 사용
List<int> combined = new();
combined.AddRange(new[] { 10, 20, 30 });
combined.AddRange(new[] { 40, 50 });
```

### 1.2 접근과 수정

```csharp
List<string> colors = new() { "Red", "Green", "Blue", "Yellow" };

// 인덱스 접근
string first = colors[0];       // "Red"
string last = colors[^1];       // "Yellow" (끝에서부터 인덱스)

// 인덱스로 수정
colors[1] = "Lime";

// Count (Length가 아님)
int count = colors.Count; // 4

// Add와 Insert
colors.Add("Purple");           // 끝에 추가
colors.Insert(2, "Orange");     // 인덱스 2에 삽입

// Remove
colors.Remove("Blue");          // 첫 번째 발견 항목 제거, bool 반환
colors.RemoveAt(0);             // 인덱스로 제거
colors.RemoveAll(c => c.StartsWith("Y")); // 조건에 맞는 모든 항목 제거

// 내용 확인
bool hasRed = colors.Contains("Red");
int idx = colors.IndexOf("Orange"); // 없으면 -1
```

### 1.3 검색과 필터링

```csharp
List<int> data = new() { 15, 3, 27, 8, 42, 11, 6, 35 };

// 첫 번째 일치 항목 찾기
int firstOver20 = data.Find(x => x > 20);        // 27
int lastOver20 = data.FindLast(x => x > 20);      // 35
int indexOver20 = data.FindIndex(x => x > 20);     // 2

// 모든 일치 항목 찾기
List<int> allOver20 = data.FindAll(x => x > 20);   // { 27, 42, 35 }

// Exists와 TrueForAll
bool anyNegative = data.Exists(x => x < 0);        // false
bool allPositive = data.TrueForAll(x => x > 0);    // true
```

### 1.4 정렬

```csharp
List<int> nums = new() { 5, 2, 8, 1, 9 };

// 기본 정렬 (오름차순)
nums.Sort();
// nums: { 1, 2, 5, 8, 9 }

// 사용자 정의 비교
nums.Sort((a, b) => b.CompareTo(a)); // 내림차순
// nums: { 9, 8, 5, 2, 1 }

// 속성으로 객체 정렬
List<(string Name, int Age)> people = new()
{
    ("Charlie", 30),
    ("Alice", 25),
    ("Bob", 35)
};

people.Sort((a, b) => a.Age.CompareTo(b.Age));
// 나이순 정렬: Alice(25), Charlie(30), Bob(35)
```

### 1.5 반복과 ForEach

```csharp
List<string> items = new() { "Alpha", "Beta", "Gamma" };

// foreach 루프
foreach (string item in items)
{
    Console.WriteLine(item);
}

// for 루프 (인덱스가 필요할 때)
for (int i = 0; i < items.Count; i++)
{
    Console.WriteLine($"{i}: {items[i]}");
}

// ForEach 메서드
items.ForEach(item => Console.WriteLine(item.ToUpper()));

// 배열로 변환
string[] arr = items.ToArray();
```

## 2. Dictionary\<TKey, TValue\>

딕셔너리(Dictionary)는 고유한 키를 값에 매핑하며, 키에 의한 평균 O(1) 시간 검색을 제공합니다.

### 2.1 생성과 항목 추가

```csharp
// 빈 딕셔너리
Dictionary<string, int> ages = new Dictionary<string, int>();

// 컬렉션 초기화자
Dictionary<string, int> scores = new()
{
    { "Alice", 95 },
    { "Bob", 87 },
    { "Charlie", 92 }
};

// 인덱스 초기화자 구문 (C# 6+)
Dictionary<string, string> capitals = new()
{
    ["France"] = "Paris",
    ["Germany"] = "Berlin",
    ["Japan"] = "Tokyo"
};

// 항목 추가
ages.Add("Alice", 30);       // 키가 존재하면 예외 발생
ages["Bob"] = 25;            // 추가 또는 덮어쓰기
ages.TryAdd("Alice", 99);    // 키가 존재하면 false 반환 (예외 없음)
```

### 2.2 값 검색

```csharp
Dictionary<string, int> inventory = new()
{
    ["Apples"] = 50,
    ["Bananas"] = 30,
    ["Oranges"] = 45
};

// 직접 접근 (없으면 KeyNotFoundException 발생)
int appleCount = inventory["Apples"]; // 50

// TryGetValue로 안전한 접근 (권장)
if (inventory.TryGetValue("Bananas", out int bananaCount))
{
    Console.WriteLine($"바나나: {bananaCount}"); // 30
}

if (!inventory.TryGetValue("Grapes", out int grapeCount))
{
    Console.WriteLine("포도를 찾을 수 없습니다");
    // grapeCount는 0 (기본값)
}

// 존재 확인
bool hasOranges = inventory.ContainsKey("Oranges");   // true
bool has45 = inventory.ContainsValue(45);             // true (O(n) 스캔)
```

### 2.3 제거와 업데이트

```csharp
Dictionary<string, double> prices = new()
{
    ["Widget"] = 9.99,
    ["Gadget"] = 24.99,
    ["Doohickey"] = 4.99
};

// 키로 제거
bool removed = prices.Remove("Doohickey"); // true

// 제거하면서 값 가져오기
if (prices.Remove("Widget", out double widgetPrice))
{
    Console.WriteLine($"Widget 제거 (가격: ${widgetPrice})");
}

// 업데이트
prices["Gadget"] = 19.99; // 기존 값 덮어쓰기

// 조건부 업데이트
if (prices.ContainsKey("Gadget"))
{
    prices["Gadget"] *= 0.9; // 10% 할인 적용
}

// 모든 항목 제거
prices.Clear();
```

### 2.4 딕셔너리 반복

```csharp
Dictionary<string, int> wordCount = new()
{
    ["the"] = 42,
    ["and"] = 27,
    ["is"] = 19,
    ["of"] = 31
};

// 키-값 쌍 반복
foreach (KeyValuePair<string, int> kvp in wordCount)
{
    Console.WriteLine($"'{kvp.Key}'가 {kvp.Value}번 등장");
}

// KeyValuePair 해체
foreach (var (word, count) in wordCount)
{
    Console.WriteLine($"'{word}': {count}");
}

// 키만 반복
foreach (string key in wordCount.Keys)
{
    Console.WriteLine(key);
}

// 값만 반복
foreach (int value in wordCount.Values)
{
    Console.WriteLine(value);
}
```

### 2.5 실용 예제: 빈도 카운터

```csharp
static Dictionary<string, int> CountWordFrequency(string text)
{
    Dictionary<string, int> freq = new(StringComparer.OrdinalIgnoreCase);
    string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    foreach (string word in words)
    {
        if (freq.TryGetValue(word, out int count))
        {
            freq[word] = count + 1;
        }
        else
        {
            freq[word] = 1;
        }
    }
    return freq;
}

// CollectionsMarshal 사용 대안 (성능에 민감한 경우)
// 또는 간단히:
// freq[word] = freq.GetValueOrDefault(word) + 1;

var result = CountWordFrequency("the cat sat on the mat the cat");
foreach (var (word, count) in result)
{
    Console.WriteLine($"{word}: {count}");
}
// the: 3, cat: 2, sat: 1, on: 1, mat: 1
```

## 3. HashSet\<T\>

`HashSet<T>`는 고유한 요소를 O(1) 검색으로 저장하며 수학적 집합 연산을 지원합니다.

### 3.1 기본 사용법

```csharp
// 생성과 요소 추가
HashSet<string> tags = new() { "csharp", "dotnet", "programming" };

// Add는 요소가 이미 존재하면 false 반환
bool added1 = tags.Add("tutorial");  // true (신규)
bool added2 = tags.Add("csharp");    // false (중복)

Console.WriteLine(tags.Count); // 4

// 멤버십 확인
bool hasDotnet = tags.Contains("dotnet"); // true

// 제거
tags.Remove("programming"); // true
```

### 3.2 집합 연산

```csharp
HashSet<int> setA = new() { 1, 2, 3, 4, 5 };
HashSet<int> setB = new() { 3, 4, 5, 6, 7 };

// 합집합: 어느 쪽이든 포함된 요소
HashSet<int> union = new(setA);
union.UnionWith(setB);
// union: { 1, 2, 3, 4, 5, 6, 7 }

// 교집합: 양쪽 모두에 포함된 요소
HashSet<int> intersection = new(setA);
intersection.IntersectWith(setB);
// intersection: { 3, 4, 5 }

// 차집합: A에는 있지만 B에는 없는 요소
HashSet<int> difference = new(setA);
difference.ExceptWith(setB);
// difference: { 1, 2 }

// 대칭 차집합: 한쪽에만 있고 양쪽 모두에는 없는 요소
HashSet<int> symDiff = new(setA);
symDiff.SymmetricExceptWith(setB);
// symDiff: { 1, 2, 6, 7 }

// 부분집합/상위집합 확인
bool isSubset = setA.IsSubsetOf(new[] { 1, 2, 3, 4, 5, 6 }); // true
bool isSuperset = setA.IsSupersetOf(new[] { 1, 2, 3 });       // true
bool overlaps = setA.Overlaps(setB);                            // true
```

### 3.3 실용 예제: 중복 제거

```csharp
List<string> emails = new()
{
    "alice@example.com",
    "bob@example.com",
    "alice@example.com",  // 중복
    "ALICE@EXAMPLE.COM",  // 중복 (대소문자 무시)
    "charlie@example.com"
};

// 대소문자 무시 중복 제거
HashSet<string> unique = new(emails, StringComparer.OrdinalIgnoreCase);
Console.WriteLine(unique.Count); // 3

// 필요하면 리스트로 다시 변환
List<string> deduped = unique.ToList();
```

## 4. Queue\<T\>와 Stack\<T\>

### 4.1 큐 (FIFO)

`Queue<T>`는 선입선출(First-In, First-Out) 컬렉션을 구현합니다:

```csharp
Queue<string> printQueue = new();

// Enqueue (뒤에 추가)
printQueue.Enqueue("Document1.pdf");
printQueue.Enqueue("Photo.jpg");
printQueue.Enqueue("Report.docx");

Console.WriteLine(printQueue.Count); // 3

// 제거 없이 앞 요소 확인
string next = printQueue.Peek(); // "Document1.pdf"

// Dequeue (앞에서 제거)
string first = printQueue.Dequeue();  // "Document1.pdf"
string second = printQueue.Dequeue(); // "Photo.jpg"

// 안전한 Dequeue
if (printQueue.TryDequeue(out string? item))
{
    Console.WriteLine($"인쇄 중: {item}"); // "Report.docx"
}

// 비어 있는지 확인
if (printQueue.TryPeek(out string? _) == false)
{
    Console.WriteLine("큐가 비어 있습니다");
}
```

### 4.2 스택 (LIFO)

`Stack<T>`는 후입선출(Last-In, First-Out) 컬렉션을 구현합니다:

```csharp
Stack<string> undoStack = new();

// Push (맨 위에 추가)
undoStack.Push("'Hello' 입력");
undoStack.Push("텍스트 굵게");
undoStack.Push("글꼴 변경");

Console.WriteLine(undoStack.Count); // 3

// 제거 없이 맨 위 요소 확인
string topAction = undoStack.Peek(); // "글꼴 변경"

// Pop (맨 위에서 제거)
string undone1 = undoStack.Pop(); // "글꼴 변경"
string undone2 = undoStack.Pop(); // "텍스트 굵게"

// 안전한 Pop
if (undoStack.TryPop(out string? action))
{
    Console.WriteLine($"실행 취소: {action}"); // "'Hello' 입력"
}
```

### 4.3 실용 예제: 괄호 균형 검사

```csharp
static bool AreBracketsBalanced(string expression)
{
    Stack<char> stack = new();
    Dictionary<char, char> matchingPairs = new()
    {
        [')'] = '(',
        [']'] = '[',
        ['}'] = '{'
    };

    foreach (char ch in expression)
    {
        if (ch is '(' or '[' or '{')
        {
            stack.Push(ch);
        }
        else if (matchingPairs.ContainsKey(ch))
        {
            if (stack.Count == 0 || stack.Pop() != matchingPairs[ch])
                return false;
        }
    }

    return stack.Count == 0;
}

Console.WriteLine(AreBracketsBalanced("({[]})")); // true
Console.WriteLine(AreBracketsBalanced("({[})"));   // false
Console.WriteLine(AreBracketsBalanced("((())"));   // false
```

## 5. LinkedList\<T\>

`LinkedList<T>`는 알려진 노드에서 O(1) 삽입과 제거를 제공하지만, 인덱스 접근은 O(n)인 이중 연결 리스트입니다.

### 5.1 기본 연산

```csharp
LinkedList<string> playlist = new();

// 요소 추가
playlist.AddLast("노래 A");
playlist.AddLast("노래 B");
playlist.AddLast("노래 C");
playlist.AddFirst("인트로");

// 노드 탐색
LinkedListNode<string>? current = playlist.First;
while (current != null)
{
    Console.WriteLine(current.Value);
    current = current.Next;
}
// 인트로, 노래 A, 노래 B, 노래 C

// 노드 찾기
LinkedListNode<string>? songB = playlist.Find("노래 B");
if (songB != null)
{
    // 알려진 노드 앞과 뒤에 삽입
    playlist.AddBefore(songB, "인터루드");
    playlist.AddAfter(songB, "노래 B 리믹스");

    // 노드 제거
    playlist.Remove(songB);
}

// 결과: 인트로, 노래 A, 인터루드, 노래 B 리믹스, 노래 C

// 첫 번째/마지막 제거
playlist.RemoveFirst();
playlist.RemoveLast();
```

### 5.2 LinkedList 사용 시점

임의 위치에서 빈번한 삽입/제거가 필요하고 이미 노드에 대한 참조를 가지고 있을 때 `LinkedList<T>`를 사용하세요. 대부분의 다른 시나리오에서는 캐시 지역성과 낮은 오버헤드 덕분에 `List<T>`가 더 좋은 성능을 보입니다.

## 6. IEnumerable\<T\>과 foreach

### 6.1 IEnumerable 인터페이스

`IEnumerable<T>`는 반복을 위한 기본 인터페이스입니다. 모든 컬렉션 타입이 이를 구현하며, `foreach`는 모든 `IEnumerable<T>`와 함께 작동합니다.

```csharp
// 모든 IEnumerable<T>는 foreach로 반복 가능
static void PrintAll<T>(IEnumerable<T> items)
{
    foreach (T item in items)
    {
        Console.WriteLine(item);
    }
}

// 모든 컬렉션에서 작동
PrintAll(new List<int> { 1, 2, 3 });
PrintAll(new HashSet<string> { "a", "b", "c" });
PrintAll(new int[] { 10, 20, 30 });
```

### 6.2 Yield Return (반복자 메서드)

`yield return`을 사용하여 사용자 정의 반복 가능 객체를 만들 수 있습니다:

```csharp
static IEnumerable<int> Fibonacci(int count)
{
    int a = 0, b = 1;
    for (int i = 0; i < count; i++)
    {
        yield return a;
        (a, b) = (b, a + b);
    }
}

foreach (int fib in Fibonacci(10))
{
    Console.Write($"{fib} "); // 0 1 1 2 3 5 8 13 21 34
}

// 지연 평가: 값이 필요할 때 계산됨
static IEnumerable<int> EvenNumbers()
{
    int n = 0;
    while (true) // 무한 시퀀스
    {
        yield return n;
        n += 2;
    }
}

// 필요한 만큼만 가져오기
foreach (int even in EvenNumbers().Take(5))
{
    Console.Write($"{even} "); // 0 2 4 6 8
}
```

## 7. 컬렉션 초기화자와 컬렉션 표현식

### 7.1 컬렉션 초기화자

```csharp
// 리스트 초기화자
List<int> nums = new() { 1, 2, 3, 4, 5 };

// 딕셔너리 초기화자 (두 가지 구문)
Dictionary<string, int> map = new()
{
    { "one", 1 },   // Add() 구문
    ["two"] = 2,     // 인덱서 구문
    ["three"] = 3
};

// HashSet 초기화자
HashSet<string> tags = new() { "urgent", "important", "review" };

// 중첩 컬렉션 초기화자
Dictionary<string, List<string>> groups = new()
{
    ["fruits"] = new() { "apple", "banana" },
    ["veggies"] = new() { "carrot", "pea" }
};
```

### 7.2 컬렉션 표현식 (C# 12)

C# 12에서 대괄호를 사용한 통합 컬렉션 생성 구문이 도입되었습니다:

```csharp
// []을 사용한 컬렉션 표현식
int[] array = [1, 2, 3, 4, 5];
List<int> list = [10, 20, 30];
Span<int> span = [100, 200, 300];
HashSet<string> set = ["a", "b", "c"];

// 스프레드 연산자(..)로 다른 컬렉션의 요소 포함
int[] first = [1, 2, 3];
int[] second = [4, 5, 6];
int[] combined = [..first, ..second]; // [1, 2, 3, 4, 5, 6]

// 스프레드와 조건부 요소
bool includeExtras = true;
int[] extras = [7, 8, 9];
int[] result = [1, 2, 3, ..(includeExtras ? extras : [])];
// result: [1, 2, 3, 7, 8, 9]

// 빈 컬렉션 표현식
List<string> empty = [];

// 메서드 매개변수와 함께 사용
static void Process(IReadOnlyList<int> data) { }
Process([1, 2, 3]);
```

## 8. 기본 LINQ 미리보기

LINQ(Language Integrated Query)는 컬렉션을 쿼리하고 변환하는 선언적 방법을 제공합니다. 이 섹션에서는 필수 사항을 다루며, 자세한 내용은 이후 레슨에서 다룹니다.

### 8.1 Where (필터링)

```csharp
using System.Linq;

List<int> numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// 필터: 짝수만 유지
IEnumerable<int> evens = numbers.Where(n => n % 2 == 0);
// evens: 2, 4, 6, 8, 10

// ToList로 체인하여 구체화
List<int> evenList = numbers.Where(n => n % 2 == 0).ToList();

// 여러 조건
var result = numbers.Where(n => n > 3 && n < 8);
// result: 4, 5, 6, 7
```

### 8.2 Select (투영)

```csharp
List<string> names = ["Alice", "Bob", "Charlie", "Diana"];

// 각 요소 변환
IEnumerable<int> lengths = names.Select(n => n.Length);
// lengths: 5, 3, 7, 5

// 익명 타입으로 변환
var nameInfo = names.Select(n => new { Name = n, Length = n.Length });
foreach (var info in nameInfo)
{
    Console.WriteLine($"{info.Name}: {info.Length}자");
}

// 인덱스와 함께 Select
var indexed = names.Select((name, i) => $"{i + 1}. {name}");
// "1. Alice", "2. Bob", "3. Charlie", "4. Diana"
```

### 8.3 OrderBy와 ThenBy

```csharp
List<(string Name, int Age, string City)> people =
[
    ("Charlie", 30, "NYC"),
    ("Alice", 25, "LA"),
    ("Bob", 30, "NYC"),
    ("Diana", 25, "LA")
];

// 나이 오름차순 정렬
var byAge = people.OrderBy(p => p.Age);

// 나이 내림차순 정렬
var byAgeDesc = people.OrderByDescending(p => p.Age);

// 다중 수준 정렬: 나이순, 그 다음 이름순
var sorted = people
    .OrderBy(p => p.Age)
    .ThenBy(p => p.Name)
    .ToList();
// Alice(25,LA), Diana(25,LA), Bob(30,NYC), Charlie(30,NYC)
```

### 8.4 집계

```csharp
List<int> scores = [85, 92, 78, 95, 88, 70, 91];

int count = scores.Count;                    // 7
int countAbove90 = scores.Count(s => s > 90); // 3
int sum = scores.Sum();                       // 599
double average = scores.Average();            // 85.57...
int min = scores.Min();                       // 70
int max = scores.Max();                       // 95

// First와 Last
int first = scores.First();                   // 85
int firstOver90 = scores.First(s => s > 90);  // 92
int lastScore = scores.Last();                // 91

// Any와 All
bool anyFailing = scores.Any(s => s < 60);    // false
bool allPassing = scores.All(s => s >= 60);   // true

// Aggregate (사용자 정의 축약)
string csv = scores.Aggregate("", (acc, s) => acc == "" ? s.ToString() : $"{acc},{s}");
// "85,92,78,95,88,70,91"
```

### 8.5 LINQ 메서드 체이닝

```csharp
List<(string Product, string Category, double Price)> products =
[
    ("Laptop", "Electronics", 999.99),
    ("Mouse", "Electronics", 29.99),
    ("Desk", "Furniture", 249.99),
    ("Chair", "Furniture", 399.99),
    ("Keyboard", "Electronics", 79.99),
    ("Lamp", "Furniture", 49.99)
];

// 가장 저렴한 전자제품 2개 찾기
var cheapElectronics = products
    .Where(p => p.Category == "Electronics")
    .OrderBy(p => p.Price)
    .Take(2)
    .Select(p => $"{p.Product}: ${p.Price}")
    .ToList();
// ["Mouse: $29.99", "Keyboard: $79.99"]

// 가구 총 가격
double furnitureTotal = products
    .Where(p => p.Category == "Furniture")
    .Sum(p => p.Price);
// 699.97

// 카테고리별 그룹화
var grouped = products
    .GroupBy(p => p.Category)
    .Select(g => new { Category = g.Key, Count = g.Count(), AvgPrice = g.Average(p => p.Price) });

foreach (var group in grouped)
{
    Console.WriteLine($"{group.Category}: {group.Count}개 항목, 평균 ${group.AvgPrice:F2}");
}
// Electronics: 3개 항목, 평균 $369.99
// Furniture: 3개 항목, 평균 $233.32
```

## 9. 적절한 컬렉션 선택하기

### 9.1 성능 특성

| 컬렉션 | 추가 | 제거 | 검색 | 인덱스 접근 | 순서 유지 |
|--------|------|------|------|-----------|---------|
| `List<T>` | O(1)* | O(n) | O(n) | O(1) | 예 (삽입 순) |
| `Dictionary<K,V>` | O(1)* | O(1) | O(1) 키 기준 | 아니오 | 아니오 |
| `HashSet<T>` | O(1)* | O(1) | O(1) | 아니오 | 아니오 |
| `Queue<T>` | O(1)* | O(1) 앞쪽 | N/A | 아니오 | FIFO |
| `Stack<T>` | O(1)* | O(1) 맨 위 | N/A | 아니오 | LIFO |
| `LinkedList<T>` | O(1) 노드 기준 | O(1) 노드 기준 | O(n) | 아니오 | 예 |
| `SortedList<K,V>` | O(n) | O(n) | O(log n) | O(1) | 키 기준 정렬 |
| `SortedDictionary<K,V>` | O(log n) | O(log n) | O(log n) | 아니오 | 키 기준 정렬 |
| `SortedSet<T>` | O(log n) | O(log n) | O(log n) | 아니오 | 정렬됨 |

*분할 상환(Amortized, 크기 조정이 발생할 수 있음)

### 9.2 결정 가이드

```csharp
// 인덱스 접근 + 동적 크기가 필요? -> List<T>
List<string> items = new() { "a", "b", "c" };

// 키-값 매핑이 필요? -> Dictionary<TKey, TValue>
Dictionary<int, string> lookup = new() { [1] = "one" };

// 고유 요소 + 빠른 검색이 필요? -> HashSet<T>
HashSet<int> seen = new() { 1, 2, 3 };

// 정렬된 고유 요소가 필요? -> SortedSet<T>
SortedSet<int> sorted = new() { 3, 1, 2 }; // 반복: 1, 2, 3

// FIFO 처리가 필요? -> Queue<T>
Queue<string> tasks = new();

// LIFO / 실행 취소가 필요? -> Stack<T>
Stack<string> undo = new();

// 효율적인 삽입과 함께 정렬된 키-값이 필요? -> SortedDictionary<TKey, TValue>
SortedDictionary<string, int> sortedMap = new();
```

## 10. ReadOnlyCollection과 불변 컬렉션

### 10.1 ReadOnlyCollection

컬렉션을 읽기 전용으로 래핑하면 소비자가 수정하는 것을 방지합니다:

```csharp
using System.Collections.ObjectModel;

List<string> mutableList = new() { "Alice", "Bob", "Charlie" };

// 읽기 전용 래퍼 생성
ReadOnlyCollection<string> readOnly = mutableList.AsReadOnly();

// 소비자가 수정할 수 없음
// readOnly.Add("Diana");    // 컴파일 오류
// readOnly[0] = "Eve";     // 컴파일 오류

// 하지만 기본 리스트의 변경은 반영됨
mutableList.Add("Diana");
Console.WriteLine(readOnly.Count); // 4

// 메서드 시그니처에는 IReadOnlyList<T> 선호
static void Display(IReadOnlyList<string> names)
{
    foreach (string name in names)
    {
        Console.WriteLine(name);
    }
}

Display(readOnly);
Display(mutableList); // List<T>는 IReadOnlyList<T>를 구현
```

### 10.2 불변 컬렉션

`System.Collections.Immutable`의 불변 컬렉션은 수정할 때마다 새 인스턴스를 생성합니다:

```csharp
using System.Collections.Immutable;

// 불변 리스트 생성
ImmutableList<int> immutable = ImmutableList.Create(1, 2, 3);

// "Add"는 새 리스트를 반환 (원본 변경 없음)
ImmutableList<int> withFour = immutable.Add(4);

Console.WriteLine(immutable.Count);  // 3
Console.WriteLine(withFour.Count);   // 4

// 효율적인 대량 생성을 위한 빌더 패턴
ImmutableList<int>.Builder builder = ImmutableList.CreateBuilder<int>();
for (int i = 0; i < 1000; i++)
{
    builder.Add(i);
}
ImmutableList<int> largeList = builder.ToImmutable();

// 불변 딕셔너리
ImmutableDictionary<string, int> dict = ImmutableDictionary<string, int>.Empty
    .Add("Alice", 30)
    .Add("Bob", 25);

ImmutableDictionary<string, int> updated = dict.SetItem("Alice", 31);
Console.WriteLine(dict["Alice"]);    // 30 (변경 없음)
Console.WriteLine(updated["Alice"]); // 31

// 불변 정렬 집합
ImmutableSortedSet<int> sortedSet = ImmutableSortedSet.Create(5, 1, 3, 2, 4);
// 항상 순서대로 반복: 1, 2, 3, 4, 5
```

### 10.3 고정 컬렉션 (.NET 8+)

한 번 구축하고 여러 번 읽는 컬렉션에는 고정 컬렉션(Frozen Collection)이 최고의 읽기 성능을 제공합니다:

```csharp
using System.Collections.Frozen;

Dictionary<string, int> source = new()
{
    ["key1"] = 1,
    ["key2"] = 2,
    ["key3"] = 3
};

// 고정 딕셔너리 생성 (읽기에 최적화, 수정 불가)
FrozenDictionary<string, int> frozen = source.ToFrozenDictionary();
int value = frozen["key1"]; // 매우 빠른 검색

// 고정 집합
FrozenSet<string> frozenSet = new[] { "a", "b", "c" }.ToFrozenSet();
bool contains = frozenSet.Contains("b"); // true
```

## 11. 연습 문제

1. **학생 성적 추적기**: 학생 이름을 시험 점수 리스트에 매핑하는 `Dictionary<string, List<int>>`를 만드세요. 학생에게 점수를 추가하고, 학생의 평균을 계산하고, 최고 평균을 가진 학생을 찾고, 임계값 이상의 평균을 가진 모든 학생을 나열하는 메서드를 작성하세요.

2. **집합 연산 CLI**: 사용자로부터 쉼표로 구분된 두 정수 리스트를 읽고 `HashSet<T>`를 사용하여 합집합, 교집합, 차집합(A - B), 대칭 차집합을 계산하여 표시하는 프로그램을 작성하세요. 출력을 깔끔하게 서식화하세요.

3. **작업 큐 시뮬레이터**: `Queue<T>`를 사용하여 간단한 작업 처리기를 구현하세요. `Name`, `Priority`, `EstimatedMinutes` 필드를 가진 구조체 `Task`를 정의하세요. 여러 작업을 큐에 넣은 다음 하나씩 처리하며, 각 작업 완료 후 남은 예상 시간을 출력하세요.

4. **실행 취소/다시 실행 시스템**: 두 개의 `Stack<string>` 인스턴스(하나는 실행 취소용, 하나는 다시 실행용)를 사용하여 실행 취소/다시 실행 시스템을 구축하세요. `Execute(string action)`, `Undo()`, `Redo()` 연산을 지원하세요. 새 작업이 실행되면 다시 실행 스택을 비우세요. 각 단계에서 작업 이력을 출력하세요.

5. **LINQ 데이터 분석**: 최소 10명의 직원이 포함된 `List<(string Name, string Department, double Salary)>`가 주어졌을 때, LINQ를 사용하여 다음을 수행하세요: (a) 각 부서에서 가장 높은 급여를 받는 직원 찾기, (b) 부서별 평균 급여 계산, (c) 회사 전체 평균 이상의 급여를 받는 직원 나열, (d) 급여 구간(예: 5만 미만, 5만~10만, 10만 초과)으로 직원 그룹화.
