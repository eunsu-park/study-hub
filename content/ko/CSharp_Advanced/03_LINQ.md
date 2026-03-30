# LINQ

**이전**: [람다 표현식과 클로저](./02_Lambda_and_Closures.md) | **다음**: [패턴 매칭](./04_Pattern_Matching.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 쿼리 구문과 메서드 구문을 모두 사용하여 LINQ 쿼리 작성하기
2. 지연 실행(deferred execution)을 설명하고 쿼리를 구체화해야 할 시점 파악하기
3. 필터링, 프로젝션, 정렬, 그룹화, 조인 연산자 적용하기
4. 집계, 집합, 요소 연산자를 효과적으로 사용하기
5. `SelectMany`로 중첩 컬렉션 평탄화하기
6. 사용자 정의 LINQ 확장 메서드 작성하기
7. LINQ 성능에 대해 추론하고 일반적인 함정 피하기

---

LINQ(Language Integrated Query)는 C#에서 가장 강력한 기능 중 하나입니다. 인메모리 컬렉션, 데이터베이스, XML, JSON 등 모든 소스의 데이터를 쿼리하고 변환하기 위한 통합된 선언적 구문을 제공합니다. 데이터를 필터링, 정렬, 그룹화하기 위한 명령형 루프를 작성하는 대신, LINQ를 사용하면 읽기 쉽고 조합 가능한 파이프라인으로 *원하는 것*을 기술할 수 있습니다.

## 1. LINQ 개요와 철학

### 1.1 LINQ란?

LINQ는 쿼리 기능을 C# 언어에 직접 통합합니다. 핵심적으로 LINQ는 `IEnumerable<T>`(인메모리 데이터용) 또는 `IQueryable<T>`(외부 데이터 소스용)를 구현하는 모든 타입에서 동작합니다.

```csharp
// 명령형 접근 — 결과를 어떻게(HOW) 얻을지
var results = new List<string>();
foreach (var name in names)
{
    if (name.Length > 3)
    {
        results.Add(name.ToUpper());
    }
}
results.Sort();

// LINQ 접근 — 어떤(WHAT) 결과를 원하는지
var results = names
    .Where(name => name.Length > 3)
    .Select(name => name.ToUpper())
    .OrderBy(name => name)
    .ToList();
```

### 1.2 LINQ 프로바이더

LINQ는 인메모리 컬렉션에 국한되지 않습니다. 다양한 프로바이더가 LINQ 표현식을 다른 대상으로 변환합니다:

```csharp
// LINQ to Objects — 인메모리 IEnumerable<T>
var inMemory = numbers.Where(n => n > 10).ToList();

// LINQ to Entities (EF Core) — SQL로 변환
var fromDb = dbContext.Products
    .Where(p => p.Price > 10)
    .OrderBy(p => p.Name)
    .ToListAsync();

// LINQ to XML
var elements = xdoc.Descendants("item")
    .Where(e => (int)e.Attribute("qty")! > 5)
    .Select(e => e.Value);
```

## 2. 쿼리 구문

쿼리 구문(query syntax)은 C# 언어에 내장된 SQL 유사 키워드를 사용합니다. 컴파일러가 이를 메서드 호출로 변환합니다.

### 2.1 기본 쿼리 구문

```csharp
int[] numbers = { 5, 12, 8, 3, 17, 1, 25, 9, 14 };

// 기본 쿼리: 필터 + 정렬 + 프로젝션
var query = from n in numbers
            where n > 5
            orderby n descending
            select n * 2;

foreach (var item in query)
    Console.Write($"{item} "); // 50 34 28 24 18 16
Console.WriteLine();
```

### 2.2 다중 소스 쿼리

```csharp
string[] colors = { "red", "green", "blue" };
string[] sizes = { "S", "M", "L" };

// 교차 조인 (카테시안 곱)
var combinations = from color in colors
                   from size in sizes
                   select $"{color}-{size}";

Console.WriteLine(string.Join(", ", combinations));
// red-S, red-M, red-L, green-S, green-M, green-L, blue-S, blue-M, blue-L
```

### 2.3 중간 값을 위한 Let 절

```csharp
string[] words = { "Hello", "Beautiful", "World", "LINQ", "Rocks" };

var query = from w in words
            let lower = w.ToLower()
            let len = w.Length
            where len > 4
            orderby len descending
            select new { Word = lower, Length = len };

foreach (var item in query)
    Console.WriteLine($"{item.Word} ({item.Length})");
// beautiful (9)
// hello (5)
// world (5)
// rocks (5)
```

### 2.4 그룹화

```csharp
var students = new[]
{
    new { Name = "Alice", Grade = "A" },
    new { Name = "Bob", Grade = "B" },
    new { Name = "Charlie", Grade = "A" },
    new { Name = "Diana", Grade = "C" },
    new { Name = "Eve", Grade = "B" },
    new { Name = "Frank", Grade = "A" },
};

var grouped = from s in students
              group s by s.Grade into gradeGroup
              orderby gradeGroup.Key
              select new
              {
                  Grade = gradeGroup.Key,
                  Count = gradeGroup.Count(),
                  Students = string.Join(", ", gradeGroup.Select(g => g.Name))
              };

foreach (var g in grouped)
    Console.WriteLine($"등급 {g.Grade}: {g.Count}명 ({g.Students})");
// 등급 A: 3명 (Alice, Charlie, Frank)
// 등급 B: 2명 (Bob, Eve)
// 등급 C: 1명 (Diana)
```

### 2.5 조인

```csharp
var departments = new[]
{
    new { Id = 1, Name = "Engineering" },
    new { Id = 2, Name = "Marketing" },
    new { Id = 3, Name = "Finance" },
};

var employees = new[]
{
    new { Name = "Alice", DeptId = 1 },
    new { Name = "Bob", DeptId = 2 },
    new { Name = "Charlie", DeptId = 1 },
    new { Name = "Diana", DeptId = 3 },
    new { Name = "Eve", DeptId = 2 },
};

var query = from e in employees
            join d in departments on e.DeptId equals d.Id
            select new { e.Name, Department = d.Name };

foreach (var item in query)
    Console.WriteLine($"{item.Name} -> {item.Department}");
// Alice -> Engineering
// Bob -> Marketing
// Charlie -> Engineering
// Diana -> Finance
// Eve -> Marketing
```

## 3. 메서드 구문

메서드 구문(method syntax, 플루언트 구문이라고도 함)은 LINQ 확장 메서드를 직접 호출합니다. 쿼리 구문보다 더 유연하며 모든 LINQ 연산자를 지원합니다.

### 3.1 기본 메서드 구문

```csharp
int[] numbers = { 5, 12, 8, 3, 17, 1, 25, 9, 14 };

var result = numbers
    .Where(n => n > 5)
    .OrderByDescending(n => n)
    .Select(n => n * 2);

Console.WriteLine(string.Join(", ", result)); // 50, 34, 28, 24, 18, 16
```

### 3.2 메서드 구문 동등 표현

```csharp
// GroupBy
var grouped = employees
    .GroupBy(e => e.DeptId)
    .Select(g => new { DeptId = g.Key, Count = g.Count() });

// Join
var joined = employees
    .Join(
        departments,
        e => e.DeptId,       // 외부 키
        d => d.Id,           // 내부 키
        (e, d) => new { e.Name, Department = d.Name }); // 결과

// GroupJoin (왼쪽 외부 조인과 동등)
var deptWithEmployees = departments
    .GroupJoin(
        employees,
        d => d.Id,
        e => e.DeptId,
        (dept, emps) => new
        {
            Department = dept.Name,
            Employees = emps.Select(e => e.Name).ToList()
        });

foreach (var d in deptWithEmployees)
    Console.WriteLine($"{d.Department}: {string.Join(", ", d.Employees)}");
```

## 4. 쿼리 구문 vs 메서드 구문 비교

대부분의 쿼리는 어느 스타일로든 작성할 수 있습니다. 컴파일러는 쿼리 구문을 메서드 호출로 변환하므로 동일한 IL을 생성합니다.

```csharp
var data = new[]
{
    new { Name = "Widget", Category = "A", Price = 25.0 },
    new { Name = "Gadget", Category = "B", Price = 45.0 },
    new { Name = "Doohickey", Category = "A", Price = 15.0 },
    new { Name = "Thingamajig", Category = "B", Price = 80.0 },
    new { Name = "Whatchamacallit", Category = "A", Price = 35.0 },
};

// 쿼리 구문
var q1 = from item in data
         where item.Price > 20
         group item by item.Category into catGroup
         select new { Category = catGroup.Key, Avg = catGroup.Average(x => x.Price) };

// 메서드 구문 — 동일한 결과
var q2 = data
    .Where(item => item.Price > 20)
    .GroupBy(item => item.Category)
    .Select(g => new { Category = g.Key, Avg = g.Average(x => x.Price) });

// 가이드라인:
// - 쿼리 구문은 조인과 그룹 연산에서 종종 더 깔끔함
// - 메서드 구문은 쿼리 키워드가 없는 연산자(Take, Skip, Distinct 등)에 필수
// - 가독성이 향상될 때 둘 다 혼합 사용
```

## 5. 지연 실행과 구체화

LINQ에서 가장 중요한 개념 중 하나는 지연 실행(deferred execution)입니다. LINQ 쿼리는 정의될 때 실행되지 않고, 순회할 때 실행됩니다.

### 5.1 지연 실행 시연

```csharp
var numbers = new List<int> { 1, 2, 3, 4, 5 };

// 쿼리 정의 — 아직 실행 안 됨
var query = numbers.Where(n => n > 2);

// 쿼리 정의 후 소스 수정
numbers.Add(6);
numbers.Add(7);

// 이제 순회하며 실행 — 6과 7이 포함됨
Console.WriteLine(string.Join(", ", query)); // 3, 4, 5, 6, 7
```

### 5.2 부수 효과로 지연 실행 확인

```csharp
var names = new[] { "Alice", "Bob", "Charlie" };

var query = names.Where(n =>
{
    Console.WriteLine($"  평가 중: {n}");
    return n.Length > 3;
});

Console.WriteLine("쿼리 정의됨. 아직 평가 안 됨.");
Console.WriteLine("이제 순회:");

foreach (var name in query) // 여기서 평가 발생
    Console.WriteLine($"  결과: {name}");

// 출력:
// 쿼리 정의됨. 아직 평가 안 됨.
// 이제 순회:
//   평가 중: Alice
//   결과: Alice
//   평가 중: Bob
//   평가 중: Charlie
//   결과: Charlie
```

### 5.3 구체화 연산자

구체화 연산자(materialization operator)는 즉시 실행을 강제하고 결과를 메모리에 저장합니다.

```csharp
var numbers = Enumerable.Range(1, 10);

// 즉시 실행하고 결과를 저장
List<int> list = numbers.Where(n => n > 5).ToList();
int[] array = numbers.Where(n => n > 5).ToArray();
Dictionary<int, int> dict = numbers.ToDictionary(n => n, n => n * n);
HashSet<int> set = numbers.ToHashSet();

// Count, Sum 등도 즉시 실행을 트리거
int count = numbers.Count(n => n > 5); // 5
int sum = numbers.Sum(); // 55
```

### 5.4 다중 열거 경고

지연 쿼리는 각 순회마다 다시 실행되므로, 쿼리를 여러 번 열거하면 파이프라인이 여러 번 실행됩니다. 이는 버그나 성능 문제를 유발할 수 있습니다.

```csharp
IEnumerable<int> ExpensiveQuery()
{
    Console.WriteLine("  [비용이 큰 쿼리 실행]");
    return Enumerable.Range(1, 5).Select(x => x * x);
}

var query = ExpensiveQuery();

// 두 번 열거하면 쿼리가 두 번 실행됨
Console.WriteLine("첫 번째: " + string.Join(", ", query));
Console.WriteLine("두 번째: " + string.Join(", ", query));
// 출력:
//   [비용이 큰 쿼리 실행]
// 첫 번째: 1, 4, 9, 16, 25
//   [비용이 큰 쿼리 실행]
// 두 번째: 1, 4, 9, 16, 25

// 수정: 두 번 이상 열거해야 하면 구체화
var materialized = ExpensiveQuery().ToList();
Console.WriteLine("첫 번째: " + string.Join(", ", materialized));  // 재실행 없음
Console.WriteLine("두 번째: " + string.Join(", ", materialized)); // 재실행 없음
```

## 6. 일반 연산자

### 6.1 필터링

```csharp
var people = new[]
{
    new { Name = "Alice", Age = 30 },
    new { Name = "Bob", Age = 25 },
    new { Name = "Charlie", Age = 35 },
    new { Name = "Diana", Age = 28 },
    new { Name = "Eve", Age = 30 },
};

// Where — 조건으로 필터링
var adults = people.Where(p => p.Age >= 30);

// OfType — 타입으로 필터링
object[] mixed = { 1, "hello", 2, "world", 3.14, 4 };
var integers = mixed.OfType<int>(); // 1, 2, 4

// Distinct
int[] dupes = { 1, 2, 2, 3, 3, 3, 4 };
var unique = dupes.Distinct(); // 1, 2, 3, 4

// DistinctBy (C# 10 / .NET 6+)
var uniqueByAge = people.DistinctBy(p => p.Age);
```

### 6.2 프로젝션

```csharp
var numbers = Enumerable.Range(1, 5);

// Select — 각 요소 변환
var squares = numbers.Select(n => n * n); // 1, 4, 9, 16, 25

// 인덱스가 있는 Select
var indexed = numbers.Select((n, i) => $"[{i}]={n}");
// [0]=1, [1]=2, [2]=3, [3]=4, [4]=5

// 익명 타입 프로젝션
var projected = people.Select(p => new { p.Name, AgeGroup = p.Age >= 30 ? "시니어" : "주니어" });
```

### 6.3 정렬

```csharp
// OrderBy / OrderByDescending
var byAge = people.OrderBy(p => p.Age);
var byAgeDesc = people.OrderByDescending(p => p.Age);

// ThenBy — 보조 정렬
var sorted = people
    .OrderBy(p => p.Age)
    .ThenBy(p => p.Name);
// (Bob,25), (Diana,28), (Alice,30), (Eve,30), (Charlie,35)

// Reverse
var reversed = people.Reverse();
```

### 6.4 그룹화

```csharp
var products = new[]
{
    new { Name = "Laptop", Category = "Electronics", Price = 999m },
    new { Name = "Phone", Category = "Electronics", Price = 699m },
    new { Name = "Desk", Category = "Furniture", Price = 350m },
    new { Name = "Chair", Category = "Furniture", Price = 250m },
    new { Name = "Tablet", Category = "Electronics", Price = 449m },
};

var groups = products.GroupBy(p => p.Category);

foreach (var group in groups)
{
    Console.WriteLine($"{group.Key}:");
    foreach (var product in group)
        Console.WriteLine($"  {product.Name}: ${product.Price}");
    Console.WriteLine($"  평균: ${group.Average(p => p.Price):F2}");
}
// Electronics:
//   Laptop: $999
//   Phone: $699
//   Tablet: $449
//   평균: $715.67
// Furniture:
//   Desk: $350
//   Chair: $250
//   평균: $300.00
```

## 7. 집계 연산자

```csharp
var numbers = new[] { 10, 20, 30, 40, 50 };

Console.WriteLine(numbers.Count());            // 5
Console.WriteLine(numbers.Sum());              // 150
Console.WriteLine(numbers.Average());          // 30
Console.WriteLine(numbers.Min());              // 10
Console.WriteLine(numbers.Max());              // 50
Console.WriteLine(numbers.MinBy(n => Math.Abs(n - 25))); // 20 (25에 가장 가까움)
Console.WriteLine(numbers.MaxBy(n => Math.Abs(n - 25))); // 50 (25에서 가장 멈)

// Aggregate — 사용자 정의 누적 (fold/reduce)
// Aggregate를 통한 합계:
int sum = numbers.Aggregate((acc, n) => acc + n); // 150

// 시드가 있는 Aggregate — 문장 구성
string sentence = numbers.Aggregate(
    "숫자:",                                // 시드
    (acc, n) => $"{acc} {n}",             // 누적기
    result => result + ".");               // 결과 선택기
Console.WriteLine(sentence); // 숫자: 10 20 30 40 50.

// 누적 곱
long product = new[] { 2, 3, 4, 5 }.Aggregate(1L, (acc, n) => acc * n);
Console.WriteLine(product); // 120
```

## 8. 집합 연산자

```csharp
var a = new[] { 1, 2, 3, 4, 5 };
var b = new[] { 3, 4, 5, 6, 7 };

// Distinct — 고유 요소
Console.WriteLine(string.Join(", ", a.Concat(b).Distinct()));
// 1, 2, 3, 4, 5, 6, 7

// Union — 합집합 (두 컬렉션의 고유 요소)
Console.WriteLine(string.Join(", ", a.Union(b)));
// 1, 2, 3, 4, 5, 6, 7

// Intersect — 교집합 (양쪽 모두에 있는 요소)
Console.WriteLine(string.Join(", ", a.Intersect(b)));
// 3, 4, 5

// Except — 차집합 (a에는 있지만 b에는 없는 요소)
Console.WriteLine(string.Join(", ", a.Except(b)));
// 1, 2

// ExceptBy / IntersectBy / UnionBy (.NET 6+)
var inventory = new[] { ("Widget", 5), ("Gadget", 10), ("Doohickey", 3) };
var discontinued = new[] { "Gadget", "Thingamajig" };

var active = inventory.ExceptBy(discontinued, item => item.Item1);
foreach (var (name, qty) in active)
    Console.WriteLine($"{name}: {qty}");
// Widget: 5
// Doohickey: 3
```

## 9. 요소 연산자

```csharp
var names = new[] { "Alice", "Bob", "Charlie", "Diana" };

// First / FirstOrDefault
Console.WriteLine(names.First());                       // Alice
Console.WriteLine(names.First(n => n.StartsWith("C"))); // Charlie
Console.WriteLine(names.FirstOrDefault(n => n.StartsWith("Z")) ?? "없음"); // 없음

// Last / LastOrDefault
Console.WriteLine(names.Last()); // Diana

// Single — 정확히 하나의 요소 (0개 또는 1개 초과 시 예외)
var single = new[] { 42 };
Console.WriteLine(single.Single()); // 42
// names.Single(); // 예외 — 요소가 둘 이상

// SingleOrDefault — 0개 또는 1개 요소
Console.WriteLine(names.SingleOrDefault(n => n == "Bob")); // Bob

// ElementAt / ElementAtOrDefault
Console.WriteLine(names.ElementAt(2));              // Charlie
Console.WriteLine(names.ElementAtOrDefault(99));    // null (string의 기본값)

// C# 12 — Index/Range 지원
Console.WriteLine(names.ElementAt(^1)); // Diana (마지막 요소)
```

## 10. SelectMany — 중첩 컬렉션 평탄화

`SelectMany`는 가장 강력한 LINQ 연산자 중 하나입니다. 각 요소를 컬렉션으로 프로젝션한 후 모든 컬렉션을 단일 시퀀스로 평탄화합니다.

### 10.1 기본 평탄화

```csharp
var sentences = new[]
{
    "The quick brown fox",
    "jumps over the lazy dog",
    "and runs away fast"
};

// 각 문장을 단어로 분리하고 평탄화
var words = sentences.SelectMany(s => s.Split(' '));
Console.WriteLine(string.Join(", ", words));
// The, quick, brown, fox, jumps, over, the, lazy, dog, and, runs, away, fast
```

### 10.2 중첩 객체 평탄화

```csharp
var departments = new[]
{
    new { Name = "Engineering", Members = new[] { "Alice", "Bob" } },
    new { Name = "Marketing", Members = new[] { "Charlie", "Diana", "Eve" } },
    new { Name = "Finance", Members = new[] { "Frank" } },
};

// 모든 멤버의 평탄한 목록
var allMembers = departments.SelectMany(d => d.Members);
Console.WriteLine(string.Join(", ", allMembers));
// Alice, Bob, Charlie, Diana, Eve, Frank

// 결과 선택기 사용 — 부서 정보 포함
var memberDetails = departments.SelectMany(
    d => d.Members,
    (dept, member) => new { dept.Name, Member = member });

foreach (var m in memberDetails)
    Console.WriteLine($"{m.Member}은(는) {m.Name}에서 근무");
```

### 10.3 인덱스가 있는 SelectMany

```csharp
var matrix = new[]
{
    new[] { 1, 2, 3 },
    new[] { 4, 5, 6 },
    new[] { 7, 8, 9 },
};

var cells = matrix.SelectMany(
    (row, rowIdx) => row.Select((val, colIdx) => new { Row = rowIdx, Col = colIdx, Value = val }));

foreach (var cell in cells)
    Console.Write($"[{cell.Row},{cell.Col}]={cell.Value} ");
// [0,0]=1 [0,1]=2 [0,2]=3 [1,0]=4 [1,1]=5 [1,2]=6 [2,0]=7 [2,1]=8 [2,2]=9
```

## 11. 사용자 정의 LINQ 확장 메서드 작성

LINQ의 힘은 조합 가능성에서 옵니다. LINQ 파이프라인에 원활하게 참여하는 자체 확장 메서드를 작성할 수 있습니다.

### 11.1 사용자 정의 필터: WhereNot

```csharp
public static class LinqExtensions
{
    /// <summary>Where의 역 — 조건에 일치하지 않는 요소를 유지합니다.</summary>
    public static IEnumerable<T> WhereNot<T>(
        this IEnumerable<T> source, Func<T, bool> predicate)
    {
        foreach (var item in source)
        {
            if (!predicate(item))
                yield return item;
        }
    }

    /// <summary>요소를 지정된 크기의 청크로 묶습니다.</summary>
    public static IEnumerable<IReadOnlyList<T>> Batch<T>(
        this IEnumerable<T> source, int batchSize)
    {
        if (batchSize <= 0) throw new ArgumentOutOfRangeException(nameof(batchSize));

        var batch = new List<T>(batchSize);
        foreach (var item in source)
        {
            batch.Add(item);
            if (batch.Count == batchSize)
            {
                yield return batch.AsReadOnly();
                batch = new List<T>(batchSize);
            }
        }
        if (batch.Count > 0)
            yield return batch.AsReadOnly();
    }

    /// <summary>두 시퀀스를 요소 단위로 교차 배치합니다.</summary>
    public static IEnumerable<T> Interleave<T>(
        this IEnumerable<T> first, IEnumerable<T> second)
    {
        using var e1 = first.GetEnumerator();
        using var e2 = second.GetEnumerator();

        while (true)
        {
            bool has1 = e1.MoveNext();
            bool has2 = e2.MoveNext();

            if (has1) yield return e1.Current;
            if (has2) yield return e2.Current;
            if (!has1 && !has2) yield break;
        }
    }

    /// <summary>누적 합계와 함께 요소를 반환합니다.</summary>
    public static IEnumerable<(T Item, decimal RunningTotal)> WithRunningTotal<T>(
        this IEnumerable<T> source, Func<T, decimal> selector)
    {
        decimal total = 0;
        foreach (var item in source)
        {
            total += selector(item);
            yield return (item, total);
        }
    }
}
```

### 11.2 사용자 정의 확장 사용

```csharp
var numbers = Enumerable.Range(1, 20);

// WhereNot
var notDivisibleBy3 = numbers.WhereNot(n => n % 3 == 0);
Console.WriteLine(string.Join(", ", notDivisibleBy3));
// 1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20

// Batch (참고: .NET 6+에는 동일한 기능의 Chunk가 있음)
foreach (var batch in numbers.Batch(6))
    Console.WriteLine(string.Join(", ", batch));
// 1, 2, 3, 4, 5, 6
// 7, 8, 9, 10, 11, 12
// 13, 14, 15, 16, 17, 18
// 19, 20

// Interleave
var odds = new[] { 1, 3, 5, 7 };
var evens = new[] { 2, 4, 6, 8, 10 };
Console.WriteLine(string.Join(", ", odds.Interleave(evens)));
// 1, 2, 3, 4, 5, 6, 7, 8, 10

// 누적 합계
var orders = new[]
{
    new { Item = "Widget", Price = 9.99m },
    new { Item = "Gadget", Price = 24.99m },
    new { Item = "Doohickey", Price = 4.49m },
};

foreach (var (order, total) in orders.WithRunningTotal(o => o.Price))
    Console.WriteLine($"{order.Item}: ${order.Price} (누적: ${total})");
// Widget: $9.99 (누적: $9.99)
// Gadget: $24.99 (누적: $34.98)
// Doohickey: $4.49 (누적: $39.47)
```

## 12. 성능 고려사항

### 12.1 LINQ 오버헤드

LINQ는 수작업 루프에 비해 약간의 오버헤드가 있습니다: 델리게이트 호출, 열거자 할당, 잠재적 클로저 할당. 핫 경로(hot path)에서는 가독성 이점이 비용을 정당화하는지 고려하세요.

```csharp
// LINQ — 명확하지만 열거자와 델리게이트를 할당
int sum1 = numbers.Where(n => n % 2 == 0).Sum();

// 수동 루프 — 할당 없음, 타이트 루프에서 더 빠름
int sum2 = 0;
foreach (var n in numbers)
{
    if (n % 2 == 0)
        sum2 += n;
}

// Span 기반 — 배열에 대해 할당 제로
int sum3 = 0;
ReadOnlySpan<int> span = numbers;
foreach (var n in span)
{
    if (n % 2 == 0)
        sum3 += n;
}
```

### 12.2 일반적인 성능 함정

```csharp
var items = Enumerable.Range(1, 1_000_000);

// 나쁨: Count()가 전체 시퀀스를 순회한 후, Any()가 다시 순회
if (items.Where(x => x > 500_000).Count() > 0) { } // O(n)

// 좋음: Any()는 첫 번째 일치에서 단락
if (items.Any(x => x > 500_000)) { } // 최선의 경우 O(1)

// 나쁨: OrderBy 후 First — 모든 것을 정렬한 후 하나만 취함
var min1 = items.OrderBy(x => x).First(); // O(n log n)

// 좋음: Min은 단일 패스
var min2 = items.Min(); // O(n)

// 나쁨: 지연 쿼리의 다중 열거
var query = items.Where(x => x % 7 == 0);
Console.WriteLine(query.Count());   // 열거
Console.WriteLine(query.Sum());     // 다시 열거

// 좋음: 한 번 구체화
var list = items.Where(x => x % 7 == 0).ToList();
Console.WriteLine(list.Count);
Console.WriteLine(list.Sum());
```

## 13. 연습 문제

1. **학생 성적표**: 이름과 (과목, 점수) 튜플 목록을 가진 학생 목록이 주어졌을 때, LINQ를 사용하여: (a) 평균 점수 기준 상위 3명의 학생 찾기, (b) 모든 학생의 평균이 가장 높은 과목 찾기, (c) 평균 점수 기준 학점 문자(A: 90+, B: 80+, C: 70+, D: 60+, F: 60 미만)로 학생 그룹화하기.

2. **단어 빈도 카운터**: 텍스트 단락이 주어졌을 때, LINQ를 사용하여 단어로 분리하고, 소문자로 정규화하고, 구두점을 제거하고, 빈도 내림차순으로 정렬된 `Dictionary<string, int>` 단어 빈도표를 생성하세요. 가장 많이 나타나는 상위 10개 단어를 출력하세요.

3. **평탄화와 변환**: `List<List<(string Key, int Value)>>`가 주어졌을 때, `SelectMany`를 사용하여 평탄화한 후 Key로 그룹화하고, 각 Key의 Value에 대한 합계와 평균을 계산하세요. 결과를 합계 내림차순으로 정렬된 익명 객체 목록으로 반환하세요.

4. **사용자 정의 LINQ 연산자 — Pairwise**: 연속 쌍을 `(T First, T Second)`로 반환하는 `Pairwise<T>(this IEnumerable<T> source)`를 구현하세요. `[1,2,3,4,5]`의 경우 `(1,2), (2,3), (3,4), (4,5)`를 반환해야 합니다. 이를 사용하여 정렬된 배열에서 연속 요소 간 최대 차이를 찾으세요.

5. **지연 실행 퍼즐**: 목록을 필터링하는 LINQ 쿼리를 작성하세요. 쿼리를 정의한 후(순회하기 전에), 목록에 새 항목을 추가하고 기존 항목을 제거하세요. 쿼리가 반환하는 것을 예측하고 확인하세요. 그런 다음 즉시 `ToList()`를 사용하도록 코드를 수정하고 동작이 어떻게 변하는지 보여주세요.
