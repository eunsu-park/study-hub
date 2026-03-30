# Generics

**Previous**: [Interfaces](./12_Interfaces.md) | **Next**: [Exception Handling](./14_Exception_Handling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain why generics improve type safety, code reuse, and performance
2. Create generic classes with one or more type parameters
3. Write generic methods that work with any type
4. Define and implement generic interfaces
5. Apply type constraints to restrict generic parameters
6. Use the `default` keyword with generic types
7. Work fluently with built-in generic collections
8. Understand the basics of covariance and contravariance

---

Generics allow you to write code that works with any data type while preserving type safety at compile time. Before generics, developers relied on the `object` type and casting, which led to runtime errors and performance overhead from boxing. Generics solve these problems elegantly, and they are used throughout the .NET framework — from collections like `List<T>` and `Dictionary<TKey, TValue>` to LINQ, async patterns, and dependency injection. This lesson covers how to create and use your own generic types.

## 1. Why Generics?

### 1.1 The Problem Without Generics

Before generics, a "general purpose" collection used `object`, requiring casts and losing type safety.

```csharp
using System.Collections;

// Non-generic ArrayList: stores objects
ArrayList list = new ArrayList();
list.Add(42);
list.Add("hello");  // No compile error! Mixed types allowed.
list.Add(3.14);

// Must cast when retrieving
int first = (int)list[0];       // Works
// int second = (int)list[1];   // Runtime error! "hello" is not int

// Value types get boxed (heap allocation + overhead)
list.Add(100);  // 100 is boxed from int (value type) to object (reference type)
```

### 1.2 The Solution With Generics

```csharp
using System.Collections.Generic;

// Generic List<T>: type-safe
List<int> numbers = new List<int>();
numbers.Add(42);
// numbers.Add("hello");  // Compile error! Only int allowed.
numbers.Add(100);

int first = numbers[0];  // No cast needed
// No boxing for value types — better performance
```

### 1.3 Three Key Benefits

```csharp
// 1. TYPE SAFETY: Errors caught at compile time, not runtime
List<string> names = new List<string>();
names.Add("Alice");
// names.Add(42);  // Compile error

// 2. CODE REUSE: One implementation works for all types
List<int> ints = new List<int>();
List<double> doubles = new List<double>();
List<string> strings = new List<string>();
// All use the same List<T> code

// 3. PERFORMANCE: No boxing/unboxing for value types
List<int> efficient = new List<int>();  // int stored directly, no boxing
efficient.Add(1);
efficient.Add(2);
int sum = efficient[0] + efficient[1];  // No unboxing needed
```

## 2. Generic Classes

A generic class uses one or more type parameters that are specified when the class is instantiated.

### 2.1 A Simple Generic Class

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

### 2.2 Generic Stack Implementation

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
            // Double the capacity
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
        _items[_count] = default(T);  // Clear the reference
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

### 2.3 Multiple Type Parameters

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

## 3. Generic Methods

Methods can be generic independently of their containing class. A generic method declares its own type parameter(s).

### 3.1 Basic Generic Methods

```csharp
public class Utility
{
    // Generic method: works with any type
    public static void Swap<T>(ref T a, ref T b)
    {
        T temp = a;
        a = b;
        b = temp;
    }

    // Generic method with return type
    public static T Max<T>(T a, T b) where T : IComparable<T>
    {
        return a.CompareTo(b) >= 0 ? a : b;
    }

    // Generic method that creates an array
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
Utility.Swap(ref x, ref y);  // Type inferred as int
Console.WriteLine($"x={x}, y={y}");  // x=10, y=5

string a = "hello", b = "world";
Utility.Swap(ref a, ref b);
Console.WriteLine($"a={a}, b={b}");  // a=world, b=hello

int bigger = Utility.Max(42, 17);
Console.WriteLine(bigger);  // 42

string[] names = Utility.CreateArray(5, "N/A");
Console.WriteLine(string.Join(", ", names));  // "N/A, N/A, N/A, N/A, N/A"
```

### 3.2 Type Inference

The compiler can often infer the type parameter, so you do not need to specify it explicitly.

```csharp
// Explicit type argument
Utility.Swap<int>(ref x, ref y);

// Inferred type argument (preferred when unambiguous)
Utility.Swap(ref x, ref y);

// Explicit is needed when inference is ambiguous
// Utility.Max("hello", 42);  // Error: cannot infer T
```

### 3.3 Generic Extension Methods

```csharp
public static class EnumerableExtensions
{
    // Find the element with the minimum value of a property
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

    // Chunk a sequence into groups of N
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

## 4. Generic Interfaces

Interfaces can also be generic, enabling type-safe contracts.

### 4.1 Defining Generic Interfaces

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

### 4.2 Implementing Generic Interfaces

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

### 4.3 Multiple Generic Interfaces

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

## 5. Type Constraints

Type constraints restrict which types can be used as type arguments, enabling you to call specific methods or access properties on the type parameter.

### 5.1 Available Constraints

```csharp
// where T : struct          — T must be a value type (int, double, bool, struct, enum)
// where T : class           — T must be a reference type (class, interface, delegate, array)
// where T : class?          — T must be a nullable reference type (C# 8+)
// where T : new()           — T must have a public parameterless constructor
// where T : BaseClass       — T must be or derive from BaseClass
// where T : IInterface      — T must implement IInterface
// where T : notnull         — T must be a non-nullable type (C# 8+)
// where T : unmanaged       — T must be an unmanaged type (no references)
```

### 5.2 The `class` Constraint

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
        // Can return null because T is a reference type
        return _items.TryGetValue(key, out T value) ? value : null;
    }
}

Cache<string> cache = new Cache<string>();
cache.Set("name", "Alice");
string name = cache.Get("name");      // "Alice"
string missing = cache.Get("email");  // null

// Cache<int> intCache;  // Error: int is a value type, not class
```

### 5.3 The `struct` Constraint

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

### 5.4 The `new()` Constraint

The `new()` constraint ensures you can create instances of `T` inside the generic class.

```csharp
public class Factory<T> where T : new()
{
    public T Create()
    {
        return new T();  // Only possible because of new() constraint
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

### 5.5 Interface and Base Class Constraints

```csharp
// Interface constraint: T must implement IComparable<T>
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

// Base class constraint
public class AnimalShelter<T> where T : Animal
{
    private readonly List<T> _animals = new List<T>();

    public void Admit(T animal)
    {
        _animals.Add(animal);
        Console.WriteLine($"Admitted {animal.Name}");  // Can access Animal.Name
    }

    public T FindByName(string name)
    {
        return _animals.FirstOrDefault(a => a.Name == name);
    }
}
```

### 5.6 Multiple Constraints

You can apply multiple constraints to a single type parameter, and constraints to multiple type parameters.

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

## 6. The `default` Keyword with Generics

The `default` keyword returns the default value for a type: `0` for numeric types, `false` for `bool`, `null` for reference types, and a zeroed struct for value types.

### 6.1 Using `default(T)`

```csharp
public class SafeQueue<T>
{
    private readonly Queue<T> _queue = new Queue<T>();

    public void Enqueue(T item)
    {
        _queue.Enqueue(item);
    }

    // Returns default(T) if queue is empty instead of throwing
    public T DequeueOrDefault()
    {
        if (_queue.Count == 0)
            return default(T);  // null for reference types, 0 for value types
        return _queue.Dequeue();
    }

    // Try-pattern
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

// Simplified default literal (C# 7.1+)
int x = default;      // 0
string s = default;    // null
bool b = default;      // false
double d = default;    // 0.0
```

### 6.2 Comparing with Default

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

## 7. Generic Collections Recap

The `System.Collections.Generic` namespace provides the most commonly used generic collections.

### 7.1 List&lt;T&gt;

```csharp
List<string> fruits = new List<string> { "Apple", "Banana", "Cherry" };
fruits.Add("Date");
fruits.Insert(1, "Avocado");
fruits.Remove("Banana");
fruits.Sort();

Console.WriteLine(string.Join(", ", fruits));
// Apple, Avocado, Cherry, Date

// Useful methods
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

// Safe access with TryGetValue
if (scores.TryGetValue("Alice", out int aliceScore))
{
    Console.WriteLine($"Alice: {aliceScore}");  // 95
}

// Iterate
foreach (KeyValuePair<string, int> kvp in scores)
{
    Console.WriteLine($"{kvp.Key}: {kvp.Value}");
}

// LINQ on dictionary
var topStudents = scores
    .Where(kvp => kvp.Value >= 90)
    .OrderByDescending(kvp => kvp.Value)
    .Select(kvp => kvp.Key);

Console.WriteLine(string.Join(", ", topStudents));  // Alice, Charlie
```

### 7.3 HashSet&lt;T&gt;, Queue&lt;T&gt;, Stack&lt;T&gt;

```csharp
// HashSet: unique elements, O(1) lookup
HashSet<int> set = new HashSet<int> { 1, 2, 3, 4, 5 };
set.Add(3);  // No effect, already exists
Console.WriteLine(set.Count);  // 5
Console.WriteLine(set.Contains(3));  // True

HashSet<int> other = new HashSet<int> { 3, 4, 5, 6, 7 };
set.IntersectWith(other);
Console.WriteLine(string.Join(", ", set));  // 3, 4, 5

// Queue: FIFO
Queue<string> queue = new Queue<string>();
queue.Enqueue("First");
queue.Enqueue("Second");
queue.Enqueue("Third");
Console.WriteLine(queue.Dequeue());  // "First"
Console.WriteLine(queue.Peek());     // "Second"

// Stack: LIFO
Stack<string> stack = new Stack<string>();
stack.Push("Bottom");
stack.Push("Middle");
stack.Push("Top");
Console.WriteLine(stack.Pop());   // "Top"
Console.WriteLine(stack.Peek());  // "Middle"
```

### 7.4 SortedDictionary and SortedSet

```csharp
// SortedDictionary: keys are always sorted
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

// SortedSet: unique elements, always sorted
SortedSet<int> sortedSet = new SortedSet<int> { 5, 3, 8, 1, 9 };
Console.WriteLine(string.Join(", ", sortedSet));  // 1, 3, 5, 8, 9
Console.WriteLine(sortedSet.Min);  // 1
Console.WriteLine(sortedSet.Max);  // 9
```

## 8. Covariance and Contravariance — Introduction

Covariance and contravariance describe how generic type relationships work with inheritance. These concepts apply to generic interfaces and delegates.

### 8.1 The Problem

```csharp
// Dog is a subtype of Animal
public class Animal { public string Name { get; set; } }
public class Dog : Animal { }

// But List<Dog> is NOT a subtype of List<Animal>
List<Dog> dogs = new List<Dog>();
// List<Animal> animals = dogs;  // Compile error!

// Why? Because if this worked, you could do:
// animals.Add(new Cat());  // A Cat in a List<Dog>? Type safety broken!
```

### 8.2 Covariance (`out` Keyword)

Covariance allows a generic type to be used as a more derived type. It is declared with the `out` keyword and means the type parameter is used only in output positions.

```csharp
// IEnumerable<T> is declared as IEnumerable<out T>
// This means IEnumerable<Dog> CAN be assigned to IEnumerable<Animal>
IEnumerable<Dog> dogs = new List<Dog>
{
    new Dog { Name = "Rex" },
    new Dog { Name = "Buddy" }
};

IEnumerable<Animal> animals = dogs;  // Covariance: this works!

foreach (Animal a in animals)
{
    Console.WriteLine(a.Name);  // Rex, Buddy
}
```

```csharp
// Custom covariant interface
public interface IProducer<out T>
{
    T Produce();
    // void Consume(T item);  // Error: T cannot be used in input position
}

public class DogProducer : IProducer<Dog>
{
    public Dog Produce() => new Dog { Name = "NewDog" };
}

// Covariance: IProducer<Dog> can be used as IProducer<Animal>
IProducer<Animal> animalProducer = new DogProducer();
Animal animal = animalProducer.Produce();  // Returns a Dog
```

### 8.3 Contravariance (`in` Keyword)

Contravariance allows a generic type to be used as a less derived type. It is declared with the `in` keyword and means the type parameter is used only in input positions.

```csharp
// IComparer<T> is declared as IComparer<in T>
public class AnimalNameComparer : IComparer<Animal>
{
    public int Compare(Animal x, Animal y)
    {
        return string.Compare(x.Name, y.Name, StringComparison.Ordinal);
    }
}

// Contravariance: IComparer<Animal> can be used as IComparer<Dog>
IComparer<Dog> dogComparer = new AnimalNameComparer();

List<Dog> dogs = new List<Dog>
{
    new Dog { Name = "Rex" },
    new Dog { Name = "Buddy" },
    new Dog { Name = "Max" }
};

dogs.Sort(dogComparer);  // Uses AnimalNameComparer for Dogs
foreach (Dog d in dogs)
{
    Console.WriteLine(d.Name);  // Buddy, Max, Rex
}
```

```csharp
// Custom contravariant interface
public interface IConsumer<in T>
{
    void Consume(T item);
    // T Produce();  // Error: T cannot be used in output position
}

public class AnimalPrinter : IConsumer<Animal>
{
    public void Consume(Animal item)
    {
        Console.WriteLine($"Animal: {item.Name}");
    }
}

// Contravariance: IConsumer<Animal> can be used as IConsumer<Dog>
IConsumer<Dog> dogPrinter = new AnimalPrinter();
dogPrinter.Consume(new Dog { Name = "Rex" });  // "Animal: Rex"
```

### 8.4 Quick Reference

| Keyword | Direction | Example | Meaning |
|---|---|---|---|
| `out` (covariance) | Output only | `IEnumerable<out T>` | `IEnumerable<Dog>` assignable to `IEnumerable<Animal>` |
| `in` (contravariance) | Input only | `IComparer<in T>` | `IComparer<Animal>` assignable to `IComparer<Dog>` |
| Neither | Invariant | `List<T>` | No implicit conversion between `List<Dog>` and `List<Animal>` |

## 9. Practical Example: Generic Repository Pattern

Let us build a complete generic repository that ties together generic classes, interfaces, constraints, and collections.

### 9.1 The Entity Base and Interface

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

### 9.2 The Generic Repository

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

### 9.3 Domain Entities

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

### 9.4 Using the Repository

```csharp
class Program
{
    static void Main()
    {
        // Customer repository
        IRepository<Customer> customerRepo = new GenericRepository<Customer>();
        customerRepo.Add(new Customer { Name = "Alice", Email = "alice@example.com" });
        customerRepo.Add(new Customer { Name = "Bob", Email = "bob@example.com" });
        customerRepo.Add(new Customer { Name = "Charlie", Email = "charlie@example.com" });

        Console.WriteLine("=== All Customers ===");
        foreach (Customer c in customerRepo.GetAll())
        {
            Console.WriteLine(c);
        }

        // Order repository
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

        // Find orders for customer 1
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

Output:
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

## 10. Practice Problems

1. **Generic Pair and Triple**: Create a `Pair<T1, T2>` class with `First` and `Second` properties, a `Swap()` method that returns a `Pair<T2, T1>`, and implements `IEquatable<Pair<T1, T2>>`. Then create a `Triple<T1, T2, T3>` that extends Pair. Write a static generic method `Zip<T1, T2>(T1[] firsts, T2[] seconds)` that returns an array of Pairs. Test with different type combinations.

2. **Generic Sorted List**: Implement a `SortedList<T>` class where `T : IComparable<T>`. It should maintain elements in sorted order at all times. Provide methods: `Add(T item)`, `Remove(T item)`, `Contains(T item)` using binary search, `IndexOf(T item)`, and `T this[int index]` indexer. Implement `IEnumerable<T>`. Test with both `int` and a custom `Temperature` struct.

3. **Generic Cache with Expiry**: Create a `TimedCache<TKey, TValue>` class where `TKey : IEquatable<TKey>`. Items should expire after a configurable `TimeSpan`. Provide: `Set(TKey key, TValue value)`, `TryGet(TKey key, out TValue value)`, `Remove(TKey key)`, and `CleanExpired()`. Test by adding items, waiting, and verifying expired items are not returned.

4. **Covariance Exploration**: Create an interface `IReadOnlyRepository<out T>` with a `GetById(int id)` method and a `GetAll()` method. Create concrete repositories for `Animal` and `Dog`. Demonstrate that `IReadOnlyRepository<Dog>` can be assigned to `IReadOnlyRepository<Animal>`. Then try to create an `IWriteRepository<in T>` and demonstrate contravariance. Explain why `IRepository<T>` (with both read and write) cannot be covariant or contravariant.

5. **Generic Pipeline**: Create a `Pipeline<TInput, TOutput>` class that chains transformation steps. It should support adding steps via `AddStep<TIntermediate>(Func<current, TIntermediate> step)` and executing with `Execute(TInput input)`. For example: `string -> int -> double -> string`. Implement it using a list of `Func<object, object>` internally (with proper casting) or using a recursive generic approach. Test with at least 3 chained transformations.
