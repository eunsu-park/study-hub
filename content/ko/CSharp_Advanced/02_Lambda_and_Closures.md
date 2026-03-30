# 람다 표현식과 클로저

**이전**: [델리게이트와 이벤트](./01_Delegates_and_Events.md) | **다음**: [LINQ](./03_LINQ.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 식 구문과 문 구문을 모두 사용하여 람다 표현식 작성하기
2. `Func<T>`, `Action<T>`, `Predicate<T>` 델리게이트와 함께 람다 사용하기
3. 클로저(closure)가 변수를 캡처하는 방식을 이해하고 일반적인 함정 식별하기
4. 의도하지 않은 캡처를 방지하기 위해 정적 람다(static lambda) 적용하기
5. 자연 함수 타입과 메서드 그룹 변환 활용하기
6. 로컬 함수와 람다를 비교하여 올바른 도구 선택하기
7. 람다를 사용한 함수 합성으로 처리 파이프라인 구축하기

---

람다 표현식(lambda expression)은 델리게이트 변수에 할당하거나, 인자로 전달하거나, 델리게이트나 식 트리(expression tree)가 필요한 곳 어디서든 사용할 수 있는 간결한 인라인 함수입니다. 클로저(closure) — 둘러싸는 스코프에서 변수를 캡처하는 능력 — 와 결합하면, 람다는 LINQ, 이벤트 처리, C#의 함수형 스타일 프로그래밍의 근간이 됩니다.

## 1. 람다 표현식 구문

### 1.1 식 람다

식 람다(expression lambda)는 본문이 단일 식입니다. 식의 결과가 반환 값이 됩니다.

```csharp
// 단일 매개변수 — 괄호 생략 가능
Func<int, int> square = x => x * x;
Console.WriteLine(square(5)); // 25

// 복수 매개변수 — 괄호 필수
Func<int, int, int> add = (a, b) => a + b;
Console.WriteLine(add(3, 7)); // 10

// 매개변수 없음
Func<DateTime> now = () => DateTime.UtcNow;
Console.WriteLine(now());

// 명시적 매개변수 타입 (컴파일러가 추론할 수 없을 때 유용)
Func<int, string> toHex = (int n) => $"0x{n:X}";
Console.WriteLine(toHex(255)); // 0xFF
```

### 1.2 문 람다

문 람다(statement lambda)는 중괄호로 둘러싸인 블록 본문을 갖습니다. 여러 문을 포함할 수 있으며, 반환 타입이 있는 경우 명시적 `return`이 필요합니다.

```csharp
Func<int, int, int> safeDivide = (a, b) =>
{
    if (b == 0)
    {
        Console.WriteLine("경고: 0으로 나누기, 0을 반환합니다");
        return 0;
    }
    return a / b;
};

Console.WriteLine(safeDivide(10, 3));  // 3
Console.WriteLine(safeDivide(10, 0));  // 경고... 그 다음 0
```

### 1.3 람다 매개변수의 디스카드

특정 매개변수가 필요하지 않을 때, 디스카드(`_`)를 사용하여 의도를 표현합니다:

```csharp
// sender나 args가 필요하지 않은 이벤트 핸들러
button.Click += (_, _) => Console.WriteLine("클릭됨!");

// 인덱스가 무관한 프로젝션
Func<string, int, string> ignoreIndex = (name, _) => name.ToUpper();
```

### 1.4 람다 반환 타입 지정 (C# 10+)

반환 타입이 모호한 경우 명시적으로 지정할 수 있습니다:

```csharp
// 명시적 반환 타입 없이 — 일부 컨텍스트에서 컴파일러가 Func<bool>과 Func<object> 중 결정 불가
// var choose = () => true ? 1 : null; // 오류

// 명시적 반환 타입
var choose = int? () => true ? 1 : null;
Console.WriteLine(choose()); // 1

// 메서드 오버로드 해석에 유용
var parse = int (string s) => int.Parse(s);
```

## 2. 내장 델리게이트 타입과 람다

### 2.1 Action\<T\> — 부수 효과 람다

```csharp
Action<string> greet = name => Console.WriteLine($"안녕하세요, {name}!");
greet("Alice"); // 안녕하세요, Alice!

Action<List<int>> shuffleAndPrint = list =>
{
    var rng = new Random();
    int n = list.Count;
    while (n > 1)
    {
        int k = rng.Next(n--);
        (list[n], list[k]) = (list[k], list[n]);
    }
    Console.WriteLine(string.Join(", ", list));
};

shuffleAndPrint(new List<int> { 1, 2, 3, 4, 5 });
```

### 2.2 Func\<T, TResult\> — 변환 람다

```csharp
Func<string, int> wordCount = text =>
    text.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;

Console.WriteLine(wordCount("The quick brown fox")); // 4

// Func 체이닝
Func<string, string> trim = s => s.Trim();
Func<string, string> lower = s => s.ToLowerInvariant();
Func<string, string> slugify = s => lower(trim(s)).Replace(' ', '-');

Console.WriteLine(slugify("  Hello World  ")); // hello-world
```

### 2.3 Predicate\<T\> — 필터 람다

```csharp
Predicate<string> isEmail = s =>
    s.Contains('@') && s.Contains('.') && s.IndexOf('@') < s.LastIndexOf('.');

var addresses = new List<string>
{
    "alice@example.com", "not-an-email", "bob@test.org", "bad@", "@.com"
};

List<string> validEmails = addresses.FindAll(isEmail);
Console.WriteLine(string.Join(", ", validEmails));
// alice@example.com, bob@test.org
```

### 2.4 Comparison\<T\> — 정렬 람다

```csharp
var people = new List<(string Name, int Age)>
{
    ("Alice", 30), ("Bob", 25), ("Charlie", 35), ("Diana", 28)
};

// 나이 오름차순 정렬
people.Sort((a, b) => a.Age.CompareTo(b.Age));
Console.WriteLine(string.Join(", ", people.Select(p => $"{p.Name}({p.Age})")));
// Bob(25), Diana(28), Alice(30), Charlie(35)

// 이름 내림차순 정렬
people.Sort((a, b) => string.Compare(b.Name, a.Name, StringComparison.Ordinal));
Console.WriteLine(string.Join(", ", people.Select(p => p.Name)));
// Diana, Charlie, Bob, Alice
```

## 3. 클로저와 변수 캡처

클로저(closure)는 둘러싸는 스코프의 변수를 참조하는 람다입니다. 람다는 그 변수를 "캡처"하여 둘러싸는 메서드가 반환된 후에도 변수를 유지합니다.

### 3.1 기본 클로저

```csharp
public static Func<int, int> CreateMultiplier(int factor)
{
    // 'factor'는 람다에 의해 캡처됨
    return x => x * factor;
}

var triple = CreateMultiplier(3);
var tenTimes = CreateMultiplier(10);

Console.WriteLine(triple(5));    // 15
Console.WriteLine(tenTimes(5));  // 50
```

### 3.2 클로저의 내부 동작

컴파일러는 캡처된 변수를 보유하는 숨겨진 클래스를 생성합니다. 람다는 해당 클래스의 메서드가 됩니다.

```csharp
// 작성한 코드:
int counter = 0;
Action increment = () => counter++;

// 컴파일러가 대략 생성하는 코드:
// class <>c__DisplayClass
// {
//     public int counter;
//     public void <Main>b__0() => counter++;
// }
// var display = new <>c__DisplayClass { counter = 0 };
// Action increment = display.<Main>b__0;
```

### 3.3 가변 캡처

캡처된 변수는 복사가 아닌 참조로 공유됩니다. 람다 내부에서 변수를 변경하면 외부 스코프에 영향을 미치고, 그 반대도 마찬가지입니다.

```csharp
int count = 0;

Action increment = () => count++;
Action print = () => Console.WriteLine($"Count: {count}");

increment();
increment();
increment();
print(); // Count: 3

count = 100;
print(); // Count: 100
```

### 3.4 루프 변수에 대한 클로저 — 고전적 함정

```csharp
// 버그: 모든 action이 동일한 변수 'i'를 캡처
var actions = new List<Action>();
for (int i = 0; i < 5; i++)
{
    actions.Add(() => Console.Write(i + " "));
}
foreach (var action in actions)
    action();
// 출력: 5 5 5 5 5  (0 1 2 3 4가 아님!)

// 수정: 루프 내부에 로컬 복사본 생성
var fixedActions = new List<Action>();
for (int i = 0; i < 5; i++)
{
    int captured = i; // 로컬 복사본 — 각 반복에서 자체 변수를 가짐
    fixedActions.Add(() => Console.Write(captured + " "));
}
foreach (var action in fixedActions)
    action();
// 출력: 0 1 2 3 4

// 참고: C# 5 이후 foreach에는 이 문제가 없음
var foreachActions = new List<Action>();
foreach (var name in new[] { "Alice", "Bob", "Charlie" })
{
    foreachActions.Add(() => Console.Write(name + " "));
}
foreach (var action in foreachActions)
    action();
// 출력: Alice Bob Charlie (올바름)
```

### 3.5 클로저 수명과 메모리

캡처된 변수는 그것을 참조하는 델리게이트가 살아 있는 한 존재합니다. 이는 예상치 못한 메모리 보유를 유발할 수 있습니다:

```csharp
public static Func<int> CreateCounter()
{
    var largeData = new byte[10_000_000]; // 10 MB
    int count = 0;

    // 'largeData'는 'count'만 사용하더라도 캡처됨
    // 전체 디스플레이 클래스(둘 다 포함)가 살아있게 됨
    return () =>
    {
        // 여기서 largeData를 참조하면, 이 델리게이트가
        // 존재하는 한 GC되지 않음
        return ++count;
    };
}

// 더 나은 방법: 불필요한 변수 캡처 피하기
public static Func<int> CreateCounterFixed()
{
    int count = 0;
    return () => ++count;
    // largeData는 스코프에 없어 캡처되지 않으며, GC될 수 있음
}
```

## 4. 정적 람다

C# 9에서는 둘러싸는 스코프에서 어떤 변수도 캡처하지 않음을 보장하는 `static` 람다를 도입했습니다. 람다 본문이 인스턴스나 로컬 변수를 참조하면 컴파일러가 오류를 발생시킵니다.

### 4.1 구문과 목적

```csharp
// 일반 람다 — 아무것도 캡처하지 않지만, 컴파일러가 이를 강제하지 않음
Func<int, int> regular = x => x * 2;

// 정적 람다 — 컴파일러가 아무것도 캡처하지 않음을 강제
Func<int, int> @static = static x => x * 2;

// 다음은 컴파일 오류:
int factor = 3;
// Func<int, int> bad = static x => x * factor; // 오류: 'factor'를 캡처할 수 없음
```

### 4.2 성능 이점

정적 람다는 클로저 객체 할당을 피할 수 있으며, 컴파일러가 상태가 없으므로 델리게이트 인스턴스를 캐시할 수 있습니다.

```csharp
// 비정적: 눈에 보이는 캡처가 없어도 클로저를 할당할 수 있음
var numbers = Enumerable.Range(1, 100);

// 정적: 클로저 할당이 없음을 보장
var evens = numbers.Where(static x => x % 2 == 0);
var doubled = evens.Select(static x => x * 2);

Console.WriteLine(doubled.Sum()); // 5100
```

### 4.3 정적 람다와 정적 로컬 변수

정적 람다를 const 값이나 정적 멤버와 함께 사용할 수 있습니다:

```csharp
const int Threshold = 100;

Func<int, bool> isAboveThreshold = static x => x > Threshold; // OK — const는 인라인됨
Console.WriteLine(isAboveThreshold(150)); // True
```

## 5. 자연 함수 타입과 메서드 그룹 변환

### 5.1 자연 함수 타입 (C# 10+)

C# 10부터 컴파일러는 `var`에 할당된 람다와 메서드 그룹의 델리게이트 타입을 추론할 수 있습니다.

```csharp
// C# 10 이전 — 델리게이트 타입을 지정해야 함
Func<int, int, int> add1 = (a, b) => a + b;

// C# 10+ — 컴파일러가 Func<int, int, int>를 추론
var add2 = (int a, int b) => a + b;

// var 추론을 위해 매개변수 타입이 명시적이어야 함
// var bad = (a, b) => a + b; // 오류 — 타입 추론 불가

// 자연 타입을 가진 메서드 그룹
var writeLine = Console.WriteLine; // Action<string> 추론 (또는 오버로드 중 하나)
```

### 5.2 메서드 그룹 변환

메서드 그룹(괄호 없는 메서드 이름)은 호환되는 델리게이트 타입으로 변환할 수 있습니다.

```csharp
// 인자로서의 메서드 그룹
var numbers = new List<int> { 3, 1, 4, 1, 5 };
numbers.Sort(int.CompareTo); // 오류 — 예상대로 동작하지 않음

// 올바른 메서드 그룹 사용
numbers.ForEach(Console.WriteLine); // Action<int> -> Console.WriteLine(int)

// 델리게이트 변수에 메서드 그룹
Func<string, bool> isNullOrEmpty = string.IsNullOrEmpty;
Console.WriteLine(isNullOrEmpty("")); // True

// 인스턴스 메서드 그룹
var sb = new System.Text.StringBuilder();
Action<string> append = sb.Append; // 오류 — Append는 void가 아닌 StringBuilder를 반환

// 호환되는 오버로드 사용
Func<string, System.Text.StringBuilder> appendFunc = sb.Append;
appendFunc("Hello");
appendFunc(" World");
Console.WriteLine(sb); // Hello World
```

## 6. 식 본문 멤버

식 본문(expression-bodied) 구문은 `=>` 화살표를 사용하여 간결한 단일 식 멤버 정의에 사용합니다. 그 자체로 람다는 아니지만 같은 화살표 구문을 사용합니다.

### 6.1 메서드, 속성, 인덱서

```csharp
public class Circle
{
    public double Radius { get; }

    // 식 본문 생성자
    public Circle(double radius) => Radius = radius;

    // 식 본문 읽기 전용 속성
    public double Diameter => Radius * 2;

    // 식 본문 메서드
    public double Area() => Math.PI * Radius * Radius;

    // get과 set이 있는 식 본문 속성
    private string _name = "";
    public string Name
    {
        get => _name;
        set => _name = value ?? throw new ArgumentNullException(nameof(value));
    }

    // 식 본문 인덱서
    private readonly double[] _points = new double[10];
    public double this[int i] => _points[i];

    // 식 본문 종료자
    ~Circle() => Console.WriteLine("Circle이 종료됨");

    // 식 본문 ToString
    public override string ToString() => $"Circle(r={Radius})";
}
```

### 6.2 식 본문을 사용할 때

식 본문은 간단한 단일 식 멤버에 가장 적합합니다. 로직에 여러 문이 필요해지면 가독성을 위해 블록 본문으로 전환하세요.

```csharp
// 좋음 — 간단하고 명확
public bool IsValid => Name.Length > 0 && Age >= 0;

// 논쟁의 여지 — 복잡해지고 있음
public string Display => $"{Name} ({Age}) - {(IsValid ? "OK" : "Invalid")}";

// 너무 복잡 — 블록 본문 사용 권장
public string DetailedReport()
{
    var sb = new System.Text.StringBuilder();
    sb.AppendLine($"Name: {Name}");
    sb.AppendLine($"Age: {Age}");
    sb.AppendLine($"Valid: {IsValid}");
    return sb.ToString();
}
```

## 7. 로컬 함수 vs 람다

C# 7에서는 다른 메서드 내부에 정의되는 명명된 메서드인 로컬 함수(local function)를 도입했습니다. 람다와 상당히 겹치지만 중요한 차이점이 있습니다.

### 7.1 구문 비교

```csharp
public void DemonstrateComparison()
{
    // 델리게이트 변수에 할당된 람다
    Func<int, int> squareLambda = x => x * x;

    // 로컬 함수 — 델리게이트 할당 없음
    int SquareLocal(int x) => x * x;

    Console.WriteLine(squareLambda(5)); // 25
    Console.WriteLine(SquareLocal(5));  // 25
}
```

### 7.2 주요 차이점

```csharp
public static void Differences()
{
    // 1. 로컬 함수는 추가 할당 없이 재귀 가능
    int Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);
    Console.WriteLine(Factorial(10)); // 3628800

    // 람다 재귀는 먼저 변수에 할당해야 함
    Func<int, int> factLambda = null!;
    factLambda = n => n <= 1 ? 1 : n * factLambda(n - 1);
    Console.WriteLine(factLambda(10)); // 3628800

    // 2. 로컬 함수는 제네릭 지원
    T Identity<T>(T value) => value;
    Console.WriteLine(Identity(42));
    Console.WriteLine(Identity("hello"));
    // 람다는 제네릭이 될 수 없음

    // 3. 로컬 함수는 'ref', 'in', 'out' 매개변수 사용 가능
    void Swap(ref int a, ref int b) => (a, b) = (b, a);
    int x = 1, y = 2;
    Swap(ref x, ref y);
    Console.WriteLine($"{x}, {y}"); // 2, 1

    // 4. 로컬 함수는 반복자(iterator)가 될 수 있음
    IEnumerable<int> Range(int start, int count)
    {
        for (int i = 0; i < count; i++)
            yield return start + i;
    }
    Console.WriteLine(string.Join(", ", Range(5, 3))); // 5, 6, 7

    // 5. 로컬 함수는 Task 없이 async가 될 수 있음
    async IAsyncEnumerable<int> AsyncRange(int count)
    {
        for (int i = 0; i < count; i++)
        {
            await Task.Delay(10);
            yield return i;
        }
    }
}
```

### 7.3 정적 로컬 함수

정적 로컬 함수(C# 8)는 둘러싸는 메서드에서 어떤 변수도 캡처할 수 없어 우발적 클로저를 방지합니다:

```csharp
public static double CalculateDistance(double x1, double y1, double x2, double y2)
{
    double dx = x2 - x1;
    double dy = y2 - y1;
    return Hypotenuse(dx, dy);

    // 정적 로컬 함수 — 모든 데이터가 매개변수로 전달됨
    static double Hypotenuse(double a, double b)
        => Math.Sqrt(a * a + b * b);

    // 다음은 실패:
    // static double Bad() => dx + dy; // 오류: dx를 캡처할 수 없음
}
```

### 7.4 선택 가이드

| 기능 | 람다 | 로컬 함수 |
|------|------|----------|
| 델리게이트 인자로 전달 | 선호됨 | 델리게이트로 래핑 필요 |
| 재귀 | 어색함 | 자연스러움 |
| 제네릭 | 불가능 | 지원 |
| ref/out 매개변수 | 불가능 | 지원 |
| 반복자 (yield) | 불가능 | 지원 |
| 성능 (할당 없음) | 클로저 할당 가능 | 할당 없을 수 있음 |
| 비동기 열거형 | 불가능 | 지원 |
| 정적 가능 여부 | 예 (C# 9) | 예 (C# 8) |

## 8. 실전 예제: 람다로 파이프라인 구축

함수 합성을 사용하여 텍스트 처리 파이프라인을 만들어 보겠습니다.

### 8.1 파이프라인 빌더

```csharp
public class Pipeline<T>
{
    private readonly List<Func<T, T>> _steps = new();

    public Pipeline<T> AddStep(Func<T, T> step)
    {
        _steps.Add(step);
        return this; // 플루언트 API
    }

    public Pipeline<T> AddConditionalStep(Func<T, bool> predicate, Func<T, T> step)
    {
        _steps.Add(value => predicate(value) ? step(value) : value);
        return this;
    }

    public T Execute(T input)
    {
        T result = input;
        foreach (var step in _steps)
        {
            result = step(result);
        }
        return result;
    }

    // 모든 단계를 단일 함수로 합성
    public Func<T, T> Compile()
    {
        var steps = _steps.ToList(); // 스냅샷
        return input =>
        {
            T result = input;
            foreach (var step in steps)
                result = step(result);
            return result;
        };
    }
}
```

### 8.2 텍스트 처리 파이프라인

```csharp
var textPipeline = new Pipeline<string>()
    .AddStep(s => s.Trim())
    .AddStep(s => s.ToLowerInvariant())
    .AddStep(s => System.Text.RegularExpressions.Regex.Replace(s, @"\s+", " "))
    .AddStep(s => s.Replace(' ', '-'))
    .AddConditionalStep(
        s => s.Length > 50,
        s => s[..50] + "...");

string slug = textPipeline.Execute("   Hello   Beautiful    World   ");
Console.WriteLine(slug); // hello-beautiful-world

// 반복 사용을 위해 컴파일
Func<string, string> slugify = textPipeline.Compile();
Console.WriteLine(slugify("  Another   Test  ")); // another-test
```

### 8.3 로깅이 포함된 숫자 파이프라인

```csharp
var mathPipeline = new Pipeline<double>()
    .AddStep(x => { Console.Write($"  입력: {x}"); return x; })
    .AddStep(x => { double r = Math.Abs(x); Console.Write($" -> 절대값: {r}"); return r; })
    .AddStep(x => { double r = Math.Round(x, 2); Console.Write($" -> 반올림: {r}"); return r; })
    .AddStep(x => { double r = Math.Sqrt(x); Console.Write($" -> 제곱근: {r:F4}"); return r; })
    .AddStep(x => { Console.WriteLine(); return x; });

double result = mathPipeline.Execute(-17.456);
Console.WriteLine($"최종 결과: {result:F4}");
// 입력: -17.456 -> 절대값: 17.456 -> 반올림: 17.46 -> 제곱근: 4.1785
// 최종 결과: 4.1785
```

### 8.4 함수 합성 유틸리티

```csharp
public static class FuncExtensions
{
    /// <summary>두 함수 합성: (f, g) => x => g(f(x))</summary>
    public static Func<T, TResult2> Then<T, TResult1, TResult2>(
        this Func<T, TResult1> first,
        Func<TResult1, TResult2> second)
    {
        return x => second(first(x));
    }

    /// <summary>함수를 N번 적용</summary>
    public static Func<T, T> Repeat<T>(this Func<T, T> f, int times)
    {
        return x =>
        {
            T result = x;
            for (int i = 0; i < times; i++)
                result = f(result);
            return result;
        };
    }

    /// <summary>순수 함수 메모이제이션</summary>
    public static Func<T, TResult> Memoize<T, TResult>(this Func<T, TResult> f)
        where T : notnull
    {
        var cache = new Dictionary<T, TResult>();
        return x =>
        {
            if (!cache.TryGetValue(x, out var result))
            {
                result = f(x);
                cache[x] = result;
            }
            return result;
        };
    }
}

// 사용법
Func<string, string> trim = s => s.Trim();
Func<string, string> lower = s => s.ToLower();
Func<string, int> length = s => s.Length;

// 합성: trim -> lower -> length
Func<string, int> pipeline = trim.Then(lower).Then(length);
Console.WriteLine(pipeline("  Hello World  ")); // 11

// 함수 반복
Func<int, int> doubleIt = x => x * 2;
Func<int, int> eightTimes = doubleIt.Repeat(3); // 2^3 = 8배
Console.WriteLine(eightTimes(5)); // 40

// 비용이 큰 함수 메모이제이션
Func<int, long> fibonacci = null!;
fibonacci = n => n <= 1 ? n : fibonacci(n - 1) + fibonacci(n - 2);
var memoFib = fibonacci.Memoize();
// 참고: 외부 호출만 메모이제이션됨; 깊은 메모이제이션을 위해서는
// 재귀 호출도 메모이제이션된 버전을 통해야 함.
```

## 9. 연습 문제

1. **클로저 카운터 팩토리**: 세 개의 델리게이트를 반환하는 함수 `CreateCounter(int start, int step)`를 작성하세요: `Func<int> next` (현재 값을 반환하고 전진), `Func<int> peek` (전진 없이 현재 값 반환), `Action reset` (시작으로 리셋). 세 개 모두 같은 캡처된 상태를 공유해야 합니다. `next()`를 반복 호출하면 등차수열이 생성됨을 보여주세요.

2. **루프 캡처 디버깅**: 다음 코드가 왜 10을 열 번 출력하는지 설명하세요. 그런 다음 (a) 로컬 변수 복사, (b) `foreach` 루프, (c) 인덱스를 캡처 대신 매개변수로 전달하는 정적 람다 접근 방식을 사용하여 수정하세요.
   ```csharp
   var funcs = new List<Func<int>>();
   for (int i = 0; i < 10; i++)
       funcs.Add(() => i);
   funcs.ForEach(f => Console.Write(f() + " "));
   ```

3. **제네릭 파이프라인 빌더**: 섹션 8의 `Pipeline<T>` 클래스를 확장하여 값을 임시로 중간 타입으로 변환하고 작업을 적용한 후 다시 변환하는 `AddStep<TIntermediate>(Func<T, TIntermediate>, Func<TIntermediate, T>)` 메서드를 지원하세요. 이를 사용하여 문자 배열로 변환, 뒤집기, 다시 변환하는 문자열 파이프라인을 만드세요.

4. **만료 기능이 있는 메모이제이션**: `MemoizeWithExpiry<T, TResult>(this Func<T, TResult> f, TimeSpan ttl)` 확장 메서드를 구현하세요. 캐시된 결과는 `ttl`이 경과하면 만료되어야 합니다. 타이밍에 `Stopwatch`나 `DateTime`을 사용하세요. `Thread.Sleep`으로 비용이 큰 계산을 시뮬레이션하는 함수로 테스트하세요.

5. **람다 vs 로컬 함수 벤치마크**: 람다와 로컬 함수를 각각 백만 번 타이트 루프에서 호출하는 코드를 작성하세요. `System.Diagnostics.Stopwatch`를 사용하여 경과 시간을 측정하세요. 시나리오를 다양화하세요: (a) 캡처 없음, (b) int 변수 하나 캡처, (c) 큰 객체 캡처. 각 접근 방식이 언제 선호되는지에 대한 결과를 보고하세요.
