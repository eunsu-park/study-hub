# 제네릭 (Generics)

**이전**: [인터페이스](./12_Interfaces.md) | **다음**: [예외 처리](./14_Exception_Handling.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 제네릭이 타입 안전성, 코드 재사용, 성능을 어떻게 향상시키는지 설명할 수 있다
2. 하나 이상의 타입 매개변수를 가진 제네릭 클래스를 만들 수 있다
3. 어떤 타입에서도 작동하는 제네릭 메서드를 작성할 수 있다
4. 제네릭 인터페이스를 정의하고 구현할 수 있다
5. 타입 제약 조건(type constraint)을 적용하여 제네릭 매개변수를 제한할 수 있다
6. 제네릭 타입에서 `default` 키워드를 사용할 수 있다
7. 내장 제네릭 컬렉션을 능숙하게 사용할 수 있다
8. 공변성(covariance)과 반공변성(contravariance)의 기본을 이해할 수 있다

---

제네릭(Generics)은 컴파일 시점에 타입 안전성을 유지하면서 어떤 데이터 타입에서도 작동하는 코드를 작성할 수 있게 합니다. 제네릭 이전에는 개발자들이 `object` 타입과 캐스팅에 의존했는데, 이는 런타임 오류와 박싱(boxing)으로 인한 성능 오버헤드를 초래했습니다. 제네릭은 이러한 문제를 우아하게 해결하며, `List<T>`와 `Dictionary<TKey, TValue>` 같은 컬렉션부터 LINQ, 비동기 패턴, 의존성 주입까지 .NET 프레임워크 전체에서 사용됩니다. 이 레슨에서는 자신만의 제네릭 타입을 만들고 사용하는 방법을 다룹니다.

## 1. 왜 제네릭인가?

### 1.1 제네릭 없이의 문제

제네릭 이전에는 "범용" 컬렉션이 `object`를 사용하여 캐스팅이 필요하고 타입 안전성을 잃었습니다.

```csharp
using System.Collections;

// 비제네릭 ArrayList: object를 저장
ArrayList list = new ArrayList();
list.Add(42);
list.Add("hello");  // 컴파일 오류 없음! 혼합 타입 허용.
list.Add(3.14);

// 가져올 때 캐스팅해야 함
int first = (int)list[0];       // 작동
// int second = (int)list[1];   // 런타임 오류! "hello"는 int가 아님

// 값 타입은 박싱됨 (힙 할당 + 오버헤드)
list.Add(100);  // 100은 int(값 타입)에서 object(참조 타입)로 박싱됨
```

### 1.2 제네릭으로의 해결

```csharp
using System.Collections.Generic;

// 제네릭 List<T>: 타입 안전
List<int> numbers = new List<int>();
numbers.Add(42);
// numbers.Add("hello");  // 컴파일 오류! int만 허용.
numbers.Add(100);

int first = numbers[0];  // 캐스팅 불필요
// 값 타입에 대한 박싱 없음 — 더 나은 성능
```

### 1.3 세 가지 핵심 이점

```csharp
// 1. 타입 안전성: 오류가 런타임이 아닌 컴파일 시점에 잡힘
List<string> names = new List<string>();
names.Add("Alice");
// names.Add(42);  // 컴파일 오류

// 2. 코드 재사용: 하나의 구현으로 모든 타입에 작동
List<int> ints = new List<int>();
List<double> doubles = new List<double>();
List<string> strings = new List<string>();
// 모두 같은 List<T> 코드를 사용

// 3. 성능: 값 타입에 대한 박싱/언박싱 없음
List<int> efficient = new List<int>();  // int가 직접 저장됨, 박싱 없음
efficient.Add(1);
efficient.Add(2);
int sum = efficient[0] + efficient[1];  // 언박싱 불필요
```

## 2. 제네릭 클래스

제네릭 클래스는 클래스가 인스턴스화될 때 지정되는 하나 이상의 타입 매개변수를 사용합니다.

### 2.1 간단한 제네릭 클래스

```csharp
public class Box<T>
{
    private T _content;

    public Box(T content)
    {
        _content = content;
    }

    public T Content
    {
        get => _content;
        set => _content = value;
    }

    public bool IsEmpty => _content == null;

    public override string ToString()
    {
        return $"Box<{typeof(T).Name}>: {_content}";
    }
}
```

```csharp
Box<int> intBox = new Box<int>(42);
Console.WriteLine(intBox.Content);    // 42
Console.WriteLine(intBox);            // "Box<Int32>: 42"

Box<string> stringBox = new Box<string>("Hello");
Console.WriteLine(stringBox.Content); // "Hello"

Box<DateTime> dateBox = new Box<DateTime>(DateTime.Now);
Console.WriteLine(dateBox);           // "Box<DateTime>: 3/29/2026 ..."
```

### 2.2 제네릭 스택 구현

```csharp
public class SimpleStack<T>
{
    private T[] _items;
    private int _count;
    private const int DefaultCapacity = 4;

    public SimpleStack()
    {
        _items = new T[DefaultCapacity];
        _count = 0;
    }

    public int Count => _count;
    public bool IsEmpty => _count == 0;

    public void Push(T item)
    {
        if (_count == _items.Length)
        {
            // 용량 두 배로 증가
            T[] newItems = new T[_items.Length * 2];
            Array.Copy(_items, newItems, _count);
            _items = newItems;
        }
        _items[_count++] = item;
    }

    public T Pop()
    {
        if (IsEmpty)
            throw new InvalidOperationException("Stack is empty.");

        T item = _items[--_count];
        _items[_count] = default(T);  // 참조 정리
        return item;
    }

    public T Peek()
    {
        if (IsEmpty)
            throw new InvalidOperationException("Stack is empty.");
        return _items[_count - 1];
    }
}
```

```csharp
SimpleStack<int> intStack = new SimpleStack<int>();
intStack.Push(10);
intStack.Push(20);
intStack.Push(30);

Console.WriteLine(intStack.Peek());  // 30
Console.WriteLine(intStack.Pop());   // 30
Console.WriteLine(intStack.Pop());   // 20
Console.WriteLine(intStack.Count);   // 1

SimpleStack<string> stringStack = new SimpleStack<string>();
stringStack.Push("first");
stringStack.Push("second");
Console.WriteLine(stringStack.Pop()); // "second"
```

### 2.3 다중 타입 매개변수

```csharp
public class Pair<T1, T2>
{
    public T1 First { get; set; }
    public T2 Second { get; set; }

    public Pair(T1 first, T2 second)
    {
        First = first;
        Second = second;
    }

    public override string ToString()
    {
        return $"({First}, {Second})";
    }
}

public class Triple<T1, T2, T3> : Pair<T1, T2>
{
    public T3 Third { get; set; }

    public Triple(T1 first, T2 second, T3 third) : base(first, second)
    {
        Third = third;
    }

    public override string ToString()
    {
        return $"({First}, {Second}, {Third})";
    }
}
```

```csharp
Pair<string, int> nameAge = new Pair<string, int>("Alice", 30);
Console.WriteLine(nameAge);  // "(Alice, 30)"

Pair<int, bool> result = new Pair<int, bool>(200, true);
Console.WriteLine(result);   // "(200, True)"

Triple<string, int, double> student = new Triple<string, int, double>("Bob", 25, 3.8);
Console.WriteLine(student);  // "(Bob, 25, 3.8)"
```

## 3. 제네릭 메서드

메서드는 포함하는 클래스와 독립적으로 제네릭일 수 있습니다. 제네릭 메서드는 자체 타입 매개변수를 선언합니다.

### 3.1 기본 제네릭 메서드

```csharp
public class Utility
{
    // 제네릭 메서드: 어떤 타입에서도 작동
    public static void Swap<T>(ref T a, ref T b)
    {
        T temp = a;
        a = b;
        b = temp;
    }

    // 반환 타입이 있는 제네릭 메서드
    public static T Max<T>(T a, T b) where T : IComparable<T>
    {
        return a.CompareTo(b) >= 0 ? a : b;
    }

    // 배열을 생성하는 제네릭 메서드
    public static T[] CreateArray<T>(int size, T defaultValue)
    {
        T[] array = new T[size];
        for (int i = 0; i < size; i++)
        {
            array[i] = defaultValue;
        }
        return array;
    }
}
```

```csharp
int x = 5, y = 10;
Utility.Swap(ref x, ref y);  // 타입이 int로 추론됨
Console.WriteLine($"x={x}, y={y}");  // x=10, y=5

string a = "hello", b = "world";
Utility.Swap(ref a, ref b);
Console.WriteLine($"a={a}, b={b}");  // a=world, b=hello

int bigger = Utility.Max(42, 17);
Console.WriteLine(bigger);  // 42

string[] names = Utility.CreateArray(5, "N/A");
Console.WriteLine(string.Join(", ", names));  // "N/A, N/A, N/A, N/A, N/A"
```

### 3.2 타입 추론

컴파일러는 대부분 타입 매개변수를 추론할 수 있으므로 명시적으로 지정할 필요가 없습니다.

```csharp
// 명시적 타입 인수
Utility.Swap<int>(ref x, ref y);

// 추론된 타입 인수 (모호하지 않을 때 선호)
Utility.Swap(ref x, ref y);

// 추론이 모호할 때는 명시적이 필요
// Utility.Max("hello", 42);  // 오류: T를 추론할 수 없음
```

### 3.3 제네릭 확장 메서드

```csharp
public static class EnumerableExtensions
{
    // 속성의 최솟값을 가진 요소 찾기
    public static T MinBy<T, TKey>(this IEnumerable<T> source, Func<T, TKey> selector)
        where TKey : IComparable<TKey>
    {
        T minItem = default;
        bool first = true;

        foreach (T item in source)
        {
            if (first || selector(item).CompareTo(selector(minItem)) < 0)
            {
                minItem = item;
                first = false;
            }
        }

        if (first) throw new InvalidOperationException("Sequence is empty.");
        return minItem;
    }

    // 시퀀스를 N개 그룹으로 나누기
    public static IEnumerable<List<T>> Chunk<T>(this IEnumerable<T> source, int size)
    {
        List<T> chunk = new List<T>(size);
        foreach (T item in source)
        {
            chunk.Add(item);
            if (chunk.Count == size)
            {
                yield return chunk;
                chunk = new List<T>(size);
            }
        }
        if (chunk.Count > 0)
            yield return chunk;
    }
}
```

```csharp
var people = new[]
{
    new { Name = "Alice", Age = 30 },
    new { Name = "Bob", Age = 25 },
    new { Name = "Charlie", Age = 35 }
};

var youngest = people.MinBy(p => p.Age);
Console.WriteLine($"Youngest: {youngest.Name}");  // Bob

int[] numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
foreach (var chunk in numbers.Chunk(3))
{
    Console.WriteLine(string.Join(", ", chunk));
}
// 1, 2, 3
// 4, 5, 6
// 7, 8, 9
// 10
```

## 4. 제네릭 인터페이스

인터페이스도 제네릭일 수 있어 타입 안전한 계약을 가능하게 합니다.

### 4.1 제네릭 인터페이스 정의

```csharp
public interface IRepository<T>
{
    T GetById(int id);
    IEnumerable<T> GetAll();
    void Add(T entity);
    void Update(T entity);
    bool Delete(int id);
    int Count { get; }
}
```

### 4.2 제네릭 인터페이스 구현

```csharp
public class InMemoryRepository<T> : IRepository<T>
{
    private readonly Dictionary<int, T> _store = new Dictionary<int, T>();
    private int _nextId = 1;
    private readonly Func<T, int> _idSelector;
    private readonly Action<T, int> _idSetter;

    public InMemoryRepository(Func<T, int> idSelector, Action<T, int> idSetter)
    {
        _idSelector = idSelector;
        _idSetter = idSetter;
    }

    public int Count => _store.Count;

    public T GetById(int id)
    {
        return _store.TryGetValue(id, out T item) ? item : default;
    }

    public IEnumerable<T> GetAll() => _store.Values;

    public void Add(T entity)
    {
        int id = _nextId++;
        _idSetter(entity, id);
        _store[id] = entity;
    }

    public void Update(T entity)
    {
        int id = _idSelector(entity);
        if (_store.ContainsKey(id))
            _store[id] = entity;
    }

    public bool Delete(int id)
    {
        return _store.Remove(id);
    }
}
```

```csharp
public class Product
{
    public int Id { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }

    public override string ToString() => $"[{Id}] {Name}: ${Price:F2}";
}

var repo = new InMemoryRepository<Product>(
    p => p.Id,
    (p, id) => p.Id = id
);

repo.Add(new Product { Name = "Laptop", Price = 999.99m });
repo.Add(new Product { Name = "Mouse", Price = 29.99m });
repo.Add(new Product { Name = "Keyboard", Price = 79.99m });

foreach (Product p in repo.GetAll())
{
    Console.WriteLine(p);
}
// [1] Laptop: $999.99
// [2] Mouse: $29.99
// [3] Keyboard: $79.99

Console.WriteLine($"Total items: {repo.Count}");  // 3
```

### 4.3 다중 제네릭 인터페이스

```csharp
public interface IMapper<TSource, TDest>
{
    TDest Map(TSource source);
    TSource ReverseMap(TDest dest);
}

public class ProductDto
{
    public string ProductName { get; set; }
    public string PriceText { get; set; }
}

public class ProductMapper : IMapper<Product, ProductDto>
{
    public ProductDto Map(Product source)
    {
        return new ProductDto
        {
            ProductName = source.Name,
            PriceText = $"${source.Price:F2}"
        };
    }

    public Product ReverseMap(ProductDto dest)
    {
        decimal price = decimal.Parse(dest.PriceText.TrimStart('$'));
        return new Product { Name = dest.ProductName, Price = price };
    }
}
```

## 5. 타입 제약 조건

타입 제약 조건(type constraint)은 타입 인수로 사용할 수 있는 타입을 제한하여, 타입 매개변수에서 특정 메서드를 호출하거나 속성에 접근할 수 있게 합니다.

### 5.1 사용 가능한 제약 조건

```csharp
// where T : struct          — T는 값 타입이어야 함 (int, double, bool, struct, enum)
// where T : class           — T는 참조 타입이어야 함 (class, interface, delegate, array)
// where T : class?          — T는 nullable 참조 타입이어야 함 (C# 8+)
// where T : new()           — T는 public 매개변수 없는 생성자가 있어야 함
// where T : BaseClass       — T는 BaseClass이거나 파생 클래스여야 함
// where T : IInterface      — T는 IInterface를 구현해야 함
// where T : notnull         — T는 non-nullable 타입이어야 함 (C# 8+)
// where T : unmanaged       — T는 비관리 타입이어야 함 (참조 없음)
```

### 5.2 `class` 제약 조건

```csharp
public class Cache<T> where T : class
{
    private Dictionary<string, T> _items = new Dictionary<string, T>();

    public void Set(string key, T value)
    {
        _items[key] = value;
    }

    public T Get(string key)
    {
        // T가 참조 타입이므로 null을 반환할 수 있음
        return _items.TryGetValue(key, out T value) ? value : null;
    }
}

Cache<string> cache = new Cache<string>();
cache.Set("name", "Alice");
string name = cache.Get("name");      // "Alice"
string missing = cache.Get("email");  // null

// Cache<int> intCache;  // 오류: int는 값 타입이지 class가 아님
```

### 5.3 `struct` 제약 조건

```csharp
public struct Optional<T> where T : struct
{
    private readonly T? _value;

    public Optional(T value)
    {
        _value = value;
        HasValue = true;
    }

    public bool HasValue { get; }

    public T Value => HasValue
        ? _value.Value
        : throw new InvalidOperationException("No value present.");

    public T GetValueOrDefault(T defaultValue)
    {
        return HasValue ? _value.Value : defaultValue;
    }

    public override string ToString()
    {
        return HasValue ? _value.ToString() : "<empty>";
    }
}

Optional<int> opt1 = new Optional<int>(42);
Console.WriteLine(opt1.Value);              // 42

Optional<int> opt2 = default;
Console.WriteLine(opt2.HasValue);           // False
Console.WriteLine(opt2.GetValueOrDefault(-1)); // -1
```

### 5.4 `new()` 제약 조건

`new()` 제약 조건은 제네릭 클래스 내에서 `T`의 인스턴스를 생성할 수 있도록 보장합니다.

```csharp
public class Factory<T> where T : new()
{
    public T Create()
    {
        return new T();  // new() 제약 조건 때문에만 가능
    }

    public List<T> CreateMany(int count)
    {
        List<T> items = new List<T>(count);
        for (int i = 0; i < count; i++)
        {
            items.Add(new T());
        }
        return items;
    }
}

public class Widget
{
    public int Id { get; set; }
    public string Name { get; set; } = "Default Widget";
}

Factory<Widget> factory = new Factory<Widget>();
Widget w = factory.Create();
Console.WriteLine(w.Name);  // "Default Widget"

List<Widget> widgets = factory.CreateMany(3);
Console.WriteLine(widgets.Count);  // 3
```

### 5.5 인터페이스와 기본 클래스 제약 조건

```csharp
// 인터페이스 제약 조건: T는 IComparable<T>를 구현해야 함
public static T FindMax<T>(IEnumerable<T> items) where T : IComparable<T>
{
    T max = default;
    bool first = true;

    foreach (T item in items)
    {
        if (first || item.CompareTo(max) > 0)
        {
            max = item;
            first = false;
        }
    }

    if (first) throw new InvalidOperationException("Sequence is empty.");
    return max;
}

// 기본 클래스 제약 조건
public class AnimalShelter<T> where T : Animal
{
    private readonly List<T> _animals = new List<T>();

    public void Admit(T animal)
    {
        _animals.Add(animal);
        Console.WriteLine($"Admitted {animal.Name}");  // Animal.Name에 접근 가능
    }

    public T FindByName(string name)
    {
        return _animals.FirstOrDefault(a => a.Name == name);
    }
}
```

### 5.6 다중 제약 조건

단일 타입 매개변수에 여러 제약 조건을 적용할 수 있고, 여러 타입 매개변수에 제약 조건을 적용할 수도 있습니다.

```csharp
public class Repository<TEntity, TKey>
    where TEntity : class, IIdentifiable<TKey>, new()
    where TKey : IEquatable<TKey>
{
    private readonly List<TEntity> _items = new List<TEntity>();

    public TEntity FindById(TKey id)
    {
        return _items.FirstOrDefault(item => item.Id.Equals(id));
    }

    public TEntity CreateNew()
    {
        return new TEntity();
    }

    public void Add(TEntity entity)
    {
        _items.Add(entity);
    }
}

public interface IIdentifiable<TKey>
{
    TKey Id { get; set; }
}

public class User : IIdentifiable<int>
{
    public int Id { get; set; }
    public string Name { get; set; }
}
```

## 6. 제네릭에서의 `default` 키워드

`default` 키워드는 타입의 기본값을 반환합니다: 숫자 타입은 `0`, `bool`은 `false`, 참조 타입은 `null`, 값 타입 구조체는 0으로 초기화된 값입니다.

### 6.1 `default(T)` 사용

```csharp
public class SafeQueue<T>
{
    private readonly Queue<T> _queue = new Queue<T>();

    public void Enqueue(T item)
    {
        _queue.Enqueue(item);
    }

    // 큐가 비어 있으면 예외 대신 default(T)를 반환
    public T DequeueOrDefault()
    {
        if (_queue.Count == 0)
            return default(T);  // 참조 타입은 null, 값 타입은 0
        return _queue.Dequeue();
    }

    // Try 패턴
    public bool TryDequeue(out T result)
    {
        if (_queue.Count > 0)
        {
            result = _queue.Dequeue();
            return true;
        }
        result = default;
        return false;
    }
}
```

```csharp
SafeQueue<int> intQueue = new SafeQueue<int>();
int val = intQueue.DequeueOrDefault();
Console.WriteLine(val);  // 0

SafeQueue<string> strQueue = new SafeQueue<string>();
string str = strQueue.DequeueOrDefault();
Console.WriteLine(str == null);  // True

// 간소화된 default 리터럴 (C# 7.1+)
int x = default;      // 0
string s = default;    // null
bool b = default;      // false
double d = default;    // 0.0
```

### 6.2 Default와 비교

```csharp
public class ResultWrapper<T>
{
    public T Value { get; }
    public bool HasValue { get; }
    public string Error { get; }

    private ResultWrapper(T value, bool hasValue, string error)
    {
        Value = value;
        HasValue = hasValue;
        Error = error;
    }

    public static ResultWrapper<T> Success(T value)
    {
        return new ResultWrapper<T>(value, true, null);
    }

    public static ResultWrapper<T> Failure(string error)
    {
        return new ResultWrapper<T>(default, false, error);
    }

    public override string ToString()
    {
        return HasValue ? $"Success: {Value}" : $"Failure: {Error}";
    }
}
```

```csharp
var result1 = ResultWrapper<int>.Success(42);
Console.WriteLine(result1);  // "Success: 42"

var result2 = ResultWrapper<string>.Failure("Not found");
Console.WriteLine(result2);  // "Failure: Not found"
```

## 7. 제네릭 컬렉션 정리

`System.Collections.Generic` 네임스페이스는 가장 자주 사용되는 제네릭 컬렉션을 제공합니다.

### 7.1 List&lt;T&gt;

```csharp
List<string> fruits = new List<string> { "Apple", "Banana", "Cherry" };
fruits.Add("Date");
fruits.Insert(1, "Avocado");
fruits.Remove("Banana");
fruits.Sort();

Console.WriteLine(string.Join(", ", fruits));
// Apple, Avocado, Cherry, Date

// 유용한 메서드
bool hasApple = fruits.Contains("Apple");         // true
int index = fruits.IndexOf("Cherry");             // 2
List<string> aFruits = fruits.FindAll(f => f.StartsWith("A")); // Apple, Avocado
string first = fruits.Find(f => f.Length > 5);    // Avocado
```

### 7.2 Dictionary&lt;TKey, TValue&gt;

```csharp
Dictionary<string, int> scores = new Dictionary<string, int>
{
    ["Alice"] = 95,
    ["Bob"] = 87,
    ["Charlie"] = 92
};

scores["Diana"] = 88;

// TryGetValue로 안전한 접근
if (scores.TryGetValue("Alice", out int aliceScore))
{
    Console.WriteLine($"Alice: {aliceScore}");  // 95
}

// 순회
foreach (KeyValuePair<string, int> kvp in scores)
{
    Console.WriteLine($"{kvp.Key}: {kvp.Value}");
}

// 딕셔너리에서의 LINQ
var topStudents = scores
    .Where(kvp => kvp.Value >= 90)
    .OrderByDescending(kvp => kvp.Value)
    .Select(kvp => kvp.Key);

Console.WriteLine(string.Join(", ", topStudents));  // Alice, Charlie
```

### 7.3 HashSet&lt;T&gt;, Queue&lt;T&gt;, Stack&lt;T&gt;

```csharp
// HashSet: 고유 요소, O(1) 조회
HashSet<int> set = new HashSet<int> { 1, 2, 3, 4, 5 };
set.Add(3);  // 이미 존재하므로 효과 없음
Console.WriteLine(set.Count);  // 5
Console.WriteLine(set.Contains(3));  // True

HashSet<int> other = new HashSet<int> { 3, 4, 5, 6, 7 };
set.IntersectWith(other);
Console.WriteLine(string.Join(", ", set));  // 3, 4, 5

// Queue: FIFO (선입선출)
Queue<string> queue = new Queue<string>();
queue.Enqueue("First");
queue.Enqueue("Second");
queue.Enqueue("Third");
Console.WriteLine(queue.Dequeue());  // "First"
Console.WriteLine(queue.Peek());     // "Second"

// Stack: LIFO (후입선출)
Stack<string> stack = new Stack<string>();
stack.Push("Bottom");
stack.Push("Middle");
stack.Push("Top");
Console.WriteLine(stack.Pop());   // "Top"
Console.WriteLine(stack.Peek());  // "Middle"
```

### 7.4 SortedDictionary와 SortedSet

```csharp
// SortedDictionary: 키가 항상 정렬됨
SortedDictionary<string, int> sorted = new SortedDictionary<string, int>
{
    ["Charlie"] = 3,
    ["Alice"] = 1,
    ["Bob"] = 2
};

foreach (var kvp in sorted)
{
    Console.WriteLine($"{kvp.Key}: {kvp.Value}");
}
// Alice: 1
// Bob: 2
// Charlie: 3

// SortedSet: 고유 요소, 항상 정렬됨
SortedSet<int> sortedSet = new SortedSet<int> { 5, 3, 8, 1, 9 };
Console.WriteLine(string.Join(", ", sortedSet));  // 1, 3, 5, 8, 9
Console.WriteLine(sortedSet.Min);  // 1
Console.WriteLine(sortedSet.Max);  // 9
```

## 8. 공변성과 반공변성 — 소개

공변성(Covariance)과 반공변성(Contravariance)은 제네릭 타입 관계가 상속과 어떻게 작동하는지를 설명합니다. 이 개념은 제네릭 인터페이스와 대리자(delegate)에 적용됩니다.

### 8.1 문제

```csharp
// Dog은 Animal의 하위 타입
public class Animal { public string Name { get; set; } }
public class Dog : Animal { }

// 하지만 List<Dog>은 List<Animal>의 하위 타입이 아님
List<Dog> dogs = new List<Dog>();
// List<Animal> animals = dogs;  // 컴파일 오류!

// 왜? 이것이 작동하면 다음과 같이 할 수 있기 때문:
// animals.Add(new Cat());  // List<Dog>에 Cat? 타입 안전성 깨짐!
```

### 8.2 공변성 (`out` 키워드)

공변성(covariance)은 제네릭 타입을 더 파생된 타입으로 사용할 수 있게 합니다. `out` 키워드로 선언하며 타입 매개변수가 출력 위치에서만 사용됨을 의미합니다.

```csharp
// IEnumerable<T>는 IEnumerable<out T>로 선언됨
// IEnumerable<Dog>을 IEnumerable<Animal>에 할당할 수 있음을 의미
IEnumerable<Dog> dogs = new List<Dog>
{
    new Dog { Name = "Rex" },
    new Dog { Name = "Buddy" }
};

IEnumerable<Animal> animals = dogs;  // 공변성: 작동!

foreach (Animal a in animals)
{
    Console.WriteLine(a.Name);  // Rex, Buddy
}
```

```csharp
// 사용자 정의 공변 인터페이스
public interface IProducer<out T>
{
    T Produce();
    // void Consume(T item);  // 오류: T는 입력 위치에서 사용할 수 없음
}

public class DogProducer : IProducer<Dog>
{
    public Dog Produce() => new Dog { Name = "NewDog" };
}

// 공변성: IProducer<Dog>을 IProducer<Animal>로 사용 가능
IProducer<Animal> animalProducer = new DogProducer();
Animal animal = animalProducer.Produce();  // Dog을 반환
```

### 8.3 반공변성 (`in` 키워드)

반공변성(contravariance)은 제네릭 타입을 덜 파생된 타입으로 사용할 수 있게 합니다. `in` 키워드로 선언하며 타입 매개변수가 입력 위치에서만 사용됨을 의미합니다.

```csharp
// IComparer<T>는 IComparer<in T>로 선언됨
public class AnimalNameComparer : IComparer<Animal>
{
    public int Compare(Animal x, Animal y)
    {
        return string.Compare(x.Name, y.Name, StringComparison.Ordinal);
    }
}

// 반공변성: IComparer<Animal>을 IComparer<Dog>으로 사용 가능
IComparer<Dog> dogComparer = new AnimalNameComparer();

List<Dog> dogs = new List<Dog>
{
    new Dog { Name = "Rex" },
    new Dog { Name = "Buddy" },
    new Dog { Name = "Max" }
};

dogs.Sort(dogComparer);  // Dog에 AnimalNameComparer 사용
foreach (Dog d in dogs)
{
    Console.WriteLine(d.Name);  // Buddy, Max, Rex
}
```

```csharp
// 사용자 정의 반공변 인터페이스
public interface IConsumer<in T>
{
    void Consume(T item);
    // T Produce();  // 오류: T는 출력 위치에서 사용할 수 없음
}

public class AnimalPrinter : IConsumer<Animal>
{
    public void Consume(Animal item)
    {
        Console.WriteLine($"Animal: {item.Name}");
    }
}

// 반공변성: IConsumer<Animal>을 IConsumer<Dog>으로 사용 가능
IConsumer<Dog> dogPrinter = new AnimalPrinter();
dogPrinter.Consume(new Dog { Name = "Rex" });  // "Animal: Rex"
```

### 8.4 빠른 참조

| 키워드 | 방향 | 예시 | 의미 |
|---|---|---|---|
| `out` (공변성) | 출력만 | `IEnumerable<out T>` | `IEnumerable<Dog>`을 `IEnumerable<Animal>`에 할당 가능 |
| `in` (반공변성) | 입력만 | `IComparer<in T>` | `IComparer<Animal>`을 `IComparer<Dog>`에 할당 가능 |
| 없음 | 불변 | `List<T>` | `List<Dog>`과 `List<Animal>` 사이의 암시적 변환 없음 |

## 9. 실전 예제: 제네릭 리포지토리 패턴

제네릭 클래스, 인터페이스, 제약 조건, 컬렉션을 모두 연결하는 완전한 제네릭 리포지토리를 만들어 봅시다.

### 9.1 엔티티 기본과 인터페이스

```csharp
public interface IEntity
{
    int Id { get; set; }
    DateTime CreatedAt { get; set; }
}

public abstract class EntityBase : IEntity
{
    public int Id { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
}
```

### 9.2 제네릭 리포지토리

```csharp
public interface IRepository<T> where T : IEntity
{
    T GetById(int id);
    IReadOnlyList<T> GetAll();
    IReadOnlyList<T> Find(Func<T, bool> predicate);
    void Add(T entity);
    void Update(T entity);
    bool Remove(int id);
    int Count { get; }
}

public class GenericRepository<T> : IRepository<T> where T : class, IEntity, new()
{
    private readonly Dictionary<int, T> _store = new Dictionary<int, T>();
    private int _nextId = 1;

    public int Count => _store.Count;

    public T GetById(int id)
    {
        return _store.TryGetValue(id, out T entity) ? entity : null;
    }

    public IReadOnlyList<T> GetAll()
    {
        return _store.Values.OrderBy(e => e.Id).ToList().AsReadOnly();
    }

    public IReadOnlyList<T> Find(Func<T, bool> predicate)
    {
        return _store.Values.Where(predicate).ToList().AsReadOnly();
    }

    public void Add(T entity)
    {
        entity.Id = _nextId++;
        entity.CreatedAt = DateTime.UtcNow;
        _store[entity.Id] = entity;
    }

    public void Update(T entity)
    {
        if (!_store.ContainsKey(entity.Id))
            throw new KeyNotFoundException($"Entity with Id {entity.Id} not found.");
        _store[entity.Id] = entity;
    }

    public bool Remove(int id)
    {
        return _store.Remove(id);
    }
}
```

### 9.3 도메인 엔티티

```csharp
public class Customer : EntityBase
{
    public string Name { get; set; }
    public string Email { get; set; }

    public override string ToString() => $"[{Id}] {Name} ({Email})";
}

public class Order : EntityBase
{
    public int CustomerId { get; set; }
    public List<string> Items { get; set; } = new List<string>();
    public decimal Total { get; set; }

    public override string ToString() => $"[{Id}] Order for Customer {CustomerId}: ${Total:F2}";
}
```

### 9.4 리포지토리 사용

```csharp
class Program
{
    static void Main()
    {
        // 고객 리포지토리
        IRepository<Customer> customerRepo = new GenericRepository<Customer>();
        customerRepo.Add(new Customer { Name = "Alice", Email = "alice@example.com" });
        customerRepo.Add(new Customer { Name = "Bob", Email = "bob@example.com" });
        customerRepo.Add(new Customer { Name = "Charlie", Email = "charlie@example.com" });

        Console.WriteLine("=== All Customers ===");
        foreach (Customer c in customerRepo.GetAll())
        {
            Console.WriteLine(c);
        }

        // 주문 리포지토리
        IRepository<Order> orderRepo = new GenericRepository<Order>();
        orderRepo.Add(new Order
        {
            CustomerId = 1,
            Items = new List<string> { "Laptop", "Mouse" },
            Total = 1029.98m
        });
        orderRepo.Add(new Order
        {
            CustomerId = 2,
            Items = new List<string> { "Keyboard" },
            Total = 79.99m
        });
        orderRepo.Add(new Order
        {
            CustomerId = 1,
            Items = new List<string> { "Monitor" },
            Total = 349.99m
        });

        Console.WriteLine("\n=== All Orders ===");
        foreach (Order o in orderRepo.GetAll())
        {
            Console.WriteLine(o);
        }

        // 고객 1의 주문 찾기
        Console.WriteLine("\n=== Alice's Orders ===");
        var aliceOrders = orderRepo.Find(o => o.CustomerId == 1);
        foreach (Order o in aliceOrders)
        {
            Console.WriteLine(o);
        }

        Console.WriteLine($"\nTotal customers: {customerRepo.Count}");
        Console.WriteLine($"Total orders: {orderRepo.Count}");
    }
}
```

출력:
```
=== All Customers ===
[1] Alice (alice@example.com)
[2] Bob (bob@example.com)
[3] Charlie (charlie@example.com)

=== All Orders ===
[1] Order for Customer 1: $1029.98
[2] Order for Customer 2: $79.99
[3] Order for Customer 1: $349.99

=== Alice's Orders ===
[1] Order for Customer 1: $1029.98
[3] Order for Customer 1: $349.99

Total customers: 3
Total orders: 3
```

## 10. 연습 문제

1. **제네릭 Pair와 Triple**: `First`와 `Second` 속성, `Swap()` 메서드(`Pair<T2, T1>`을 반환), `IEquatable<Pair<T1, T2>>`를 구현하는 `Pair<T1, T2>` 클래스를 만드세요. 그런 다음 Pair를 확장하는 `Triple<T1, T2, T3>`를 만드세요. `Zip<T1, T2>(T1[] firsts, T2[] seconds)`라는 정적 제네릭 메서드를 작성하여 Pair 배열을 반환하세요. 다양한 타입 조합으로 테스트하세요.

2. **제네릭 정렬 리스트**: `T : IComparable<T>` 제약 조건이 있는 `SortedList<T>` 클래스를 구현하세요. 항상 정렬된 순서를 유지해야 합니다. `Add(T item)`, `Remove(T item)`, 이진 검색을 사용하는 `Contains(T item)`, `IndexOf(T item)`, `T this[int index]` 인덱서를 제공하세요. `IEnumerable<T>`를 구현하세요. `int`와 사용자 정의 `Temperature` 구조체로 테스트하세요.

3. **만료 기능이 있는 제네릭 캐시**: `TKey : IEquatable<TKey>` 제약 조건이 있는 `TimedCache<TKey, TValue>` 클래스를 만드세요. 항목은 구성 가능한 `TimeSpan` 후에 만료되어야 합니다. `Set(TKey key, TValue value)`, `TryGet(TKey key, out TValue value)`, `Remove(TKey key)`, `CleanExpired()`를 제공하세요. 항목을 추가하고, 기다린 후, 만료된 항목이 반환되지 않는 것을 확인하여 테스트하세요.

4. **공변성 탐구**: `GetById(int id)` 메서드와 `GetAll()` 메서드가 있는 `IReadOnlyRepository<out T>` 인터페이스를 만드세요. `Animal`과 `Dog`에 대한 구체적 리포지토리를 만드세요. `IReadOnlyRepository<Dog>`을 `IReadOnlyRepository<Animal>`에 할당할 수 있음을 시연하세요. 그런 다음 `IWriteRepository<in T>`를 만들어 반공변성을 시연하세요. `IRepository<T>`(읽기와 쓰기 모두 포함)가 공변적이거나 반공변적일 수 없는 이유를 설명하세요.

5. **제네릭 파이프라인**: 변환 단계를 연결하는 `Pipeline<TInput, TOutput>` 클래스를 만드세요. `AddStep<TIntermediate>(Func<current, TIntermediate> step)`로 단계 추가를 지원하고 `Execute(TInput input)`로 실행해야 합니다. 예: `string -> int -> double -> string`. 내부적으로 `Func<object, object>` 리스트를 사용하거나(적절한 캐스팅과 함께) 재귀적 제네릭 접근법을 사용하여 구현하세요. 최소 3개의 연결된 변환으로 테스트하세요.
