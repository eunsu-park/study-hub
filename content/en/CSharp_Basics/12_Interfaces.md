# Interfaces

**Previous**: [Inheritance](./11_Inheritance.md) | **Next**: [Generics](./13_Generics.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare and implement interfaces with methods, properties, events, and indexers
2. Implement multiple interfaces in a single class
3. Use explicit interface implementation to resolve naming conflicts
4. Leverage default interface methods introduced in C# 8
5. Design interface inheritance hierarchies
6. Implement standard .NET interfaces like `IComparable<T>` and `IEquatable<T>`
7. Create iterable types with `IEnumerable<T>` and `IEnumerator<T>`
8. Choose between interfaces and abstract classes for your designs

---

Interfaces define a contract that classes or structs must fulfill. Unlike inheritance, which models an "is-a" relationship, interfaces model a "can-do" capability. A class can implement multiple interfaces, giving C# a form of multiple inheritance for behavior without the complexities of multiple class inheritance. Interfaces are the backbone of many design patterns, dependency injection frameworks, and the .NET standard library itself.

## 1. Interface Declaration

An interface declares a set of members that implementing types must provide. Interfaces can contain method signatures, properties, events, and indexers.

### 1.1 Basic Interface Syntax

```csharp
public interface IShape
{
    // Method signature (no body)
    double Area();
    double Perimeter();

    // Property signature
    string Name { get; }

    // No fields allowed in interfaces
    // int x;  // Error
}
```

By convention, C# interface names start with a capital `I` (e.g., `IShape`, `IDisposable`, `IComparable`).

### 1.2 Properties in Interfaces

```csharp
public interface IIdentifiable
{
    // Read-only property
    string Id { get; }

    // Read-write property
    string DisplayName { get; set; }
}

public interface ITimestamped
{
    DateTime CreatedAt { get; }
    DateTime? UpdatedAt { get; set; }
}
```

### 1.3 Events in Interfaces

```csharp
public interface INotifier
{
    event EventHandler<string> OnNotification;
    void SendNotification(string message);
}
```

### 1.4 Indexers in Interfaces

```csharp
public interface IReadOnlyCollection
{
    int Count { get; }
    string this[int index] { get; }
}
```

## 2. Implementing Interfaces

A class implements an interface by listing it after the colon and providing implementations for all members.

### 2.1 Basic Implementation

```csharp
public interface IGreeter
{
    string Greet(string name);
    string Farewell(string name);
}

public class EnglishGreeter : IGreeter
{
    public string Greet(string name)
    {
        return $"Hello, {name}!";
    }

    public string Farewell(string name)
    {
        return $"Goodbye, {name}!";
    }
}

public class SpanishGreeter : IGreeter
{
    public string Greet(string name)
    {
        return $"Hola, {name}!";
    }

    public string Farewell(string name)
    {
        return $"Adios, {name}!";
    }
}
```

```csharp
IGreeter greeter = new EnglishGreeter();
Console.WriteLine(greeter.Greet("Alice"));     // "Hello, Alice!"

greeter = new SpanishGreeter();
Console.WriteLine(greeter.Greet("Alice"));     // "Hola, Alice!"
```

### 2.2 Implementing All Members Is Required

If a class does not implement all interface members, the compiler will report an error — unless the class is abstract.

```csharp
public interface IVehicle
{
    void Start();
    void Stop();
    int Speed { get; }
}

// This abstract class can leave some members unimplemented
public abstract class VehicleBase : IVehicle
{
    public abstract void Start();
    public abstract void Stop();
    public abstract int Speed { get; }
}

// Concrete class must implement everything
public class Car : VehicleBase
{
    private int _speed;
    public override int Speed => _speed;

    public override void Start()
    {
        _speed = 10;
        Console.WriteLine("Car started.");
    }

    public override void Stop()
    {
        _speed = 0;
        Console.WriteLine("Car stopped.");
    }
}
```

### 2.3 Structs Implementing Interfaces

Structs can also implement interfaces. This is common in .NET (e.g., `int` implements `IComparable<int>`).

```csharp
public interface IPrintable
{
    string ToPrintString();
}

public struct Temperature : IPrintable
{
    public double Celsius { get; set; }

    public Temperature(double celsius)
    {
        Celsius = celsius;
    }

    public string ToPrintString()
    {
        return $"{Celsius:F1}C / {Celsius * 9 / 5 + 32:F1}F";
    }
}

Temperature t = new Temperature(100);
Console.WriteLine(t.ToPrintString());  // "100.0C / 212.0F"
```

## 3. Multiple Interface Implementation

A class can implement multiple interfaces, which is how C# achieves a form of multiple inheritance.

### 3.1 Implementing Multiple Interfaces

```csharp
public interface ISerializable
{
    string Serialize();
}

public interface IDeserializable
{
    void Deserialize(string data);
}

public interface ILoggable
{
    void Log(string message);
}

public class UserProfile : ISerializable, IDeserializable, ILoggable
{
    public string Username { get; set; }
    public string Email { get; set; }

    public string Serialize()
    {
        return $"{Username}|{Email}";
    }

    public void Deserialize(string data)
    {
        string[] parts = data.Split('|');
        Username = parts[0];
        Email = parts[1];
    }

    public void Log(string message)
    {
        Console.WriteLine($"[UserProfile:{Username}] {message}");
    }
}
```

```csharp
UserProfile user = new UserProfile { Username = "alice", Email = "alice@example.com" };

// Use through different interface references
ISerializable serializable = user;
string data = serializable.Serialize();
Console.WriteLine(data);  // "alice|alice@example.com"

ILoggable loggable = user;
loggable.Log("Profile accessed.");  // "[UserProfile:alice] Profile accessed."
```

### 3.2 Combining Interface and Base Class

A class can inherit from one base class and implement multiple interfaces.

```csharp
public class Animal
{
    public string Name { get; set; }
}

public interface ISwimmable
{
    void Swim();
}

public interface IFlyable
{
    void Fly();
}

public class Duck : Animal, ISwimmable, IFlyable
{
    public void Swim()
    {
        Console.WriteLine($"{Name} is swimming.");
    }

    public void Fly()
    {
        Console.WriteLine($"{Name} is flying.");
    }
}
```

Note that the base class must come first in the list, before any interfaces.

## 4. Explicit Interface Implementation

When a class implements two interfaces that have methods with the same signature, or when you want to hide an interface method from the class's public API, you use explicit interface implementation.

### 4.1 Resolving Naming Conflicts

```csharp
public interface IFileReader
{
    string Read();
}

public interface INetworkReader
{
    string Read();
}

public class DataReader : IFileReader, INetworkReader
{
    // Explicit implementation for IFileReader.Read
    string IFileReader.Read()
    {
        return "Reading from file...";
    }

    // Explicit implementation for INetworkReader.Read
    string INetworkReader.Read()
    {
        return "Reading from network...";
    }
}
```

```csharp
DataReader reader = new DataReader();
// reader.Read();  // Error: ambiguous, not accessible directly

// Must cast to the specific interface
IFileReader fileReader = reader;
Console.WriteLine(fileReader.Read());      // "Reading from file..."

INetworkReader networkReader = reader;
Console.WriteLine(networkReader.Read());   // "Reading from network..."
```

### 4.2 Hiding Interface Members

Explicit implementation can also hide members that should only be accessed through the interface reference.

```csharp
public interface IResettable
{
    void Reset();
}

public class GameState : IResettable
{
    public int Score { get; set; }
    public int Level { get; set; }

    // Explicit: Reset() is only visible through IResettable reference
    void IResettable.Reset()
    {
        Score = 0;
        Level = 1;
        Console.WriteLine("Game state has been reset.");
    }

    // Public method for normal use
    public void NewGame()
    {
        ((IResettable)this).Reset();
        Console.WriteLine("Starting new game...");
    }
}
```

```csharp
GameState game = new GameState { Score = 100, Level = 5 };
// game.Reset();  // Error: not accessible directly

IResettable resettable = game;
resettable.Reset();  // Works through the interface reference

game.NewGame();      // Also works, calls Reset() internally
```

### 4.3 Implicit vs Explicit: Quick Comparison

```csharp
public interface IAnimal
{
    void Speak();
}

// Implicit: Speak() is public on the class
public class Dog : IAnimal
{
    public void Speak() => Console.WriteLine("Woof!");
}

// Explicit: Speak() is only accessible through IAnimal
public class Cat : IAnimal
{
    void IAnimal.Speak() => Console.WriteLine("Meow!");
}

Dog dog = new Dog();
dog.Speak();              // Works
((IAnimal)dog).Speak();   // Also works

Cat cat = new Cat();
// cat.Speak();            // Error
((IAnimal)cat).Speak();   // Works
```

## 5. Default Interface Methods (C# 8+)

Starting with C# 8, interfaces can include method bodies — called default implementations. This allows adding new methods to an existing interface without breaking classes that already implement it.

### 5.1 Basic Default Methods

```csharp
public interface ILogger
{
    void Log(string message);

    // Default implementation — classes do not need to override this
    void LogError(string message)
    {
        Log($"[ERROR] {message}");
    }

    void LogWarning(string message)
    {
        Log($"[WARNING] {message}");
    }

    void LogInfo(string message)
    {
        Log($"[INFO] {message}");
    }
}

public class ConsoleLogger : ILogger
{
    public void Log(string message)
    {
        Console.WriteLine($"[{DateTime.Now:HH:mm:ss}] {message}");
    }
    // LogError, LogWarning, LogInfo are inherited with default behavior
}
```

```csharp
ILogger logger = new ConsoleLogger();
logger.Log("Direct message");
logger.LogError("Something failed");
logger.LogWarning("Low disk space");
logger.LogInfo("Process started");

// Note: default methods are only accessible through the interface reference
ConsoleLogger concrete = new ConsoleLogger();
concrete.Log("works");         // Works (explicitly implemented)
// concrete.LogError("test");  // Error unless ConsoleLogger explicitly implements it
ILogger asInterface = concrete;
asInterface.LogError("test");  // Works through interface reference
```

### 5.2 Overriding Default Methods

A class can still override a default implementation if it needs custom behavior.

```csharp
public class FileLogger : ILogger
{
    private readonly string _filePath;

    public FileLogger(string path)
    {
        _filePath = path;
    }

    public void Log(string message)
    {
        File.AppendAllText(_filePath, message + Environment.NewLine);
    }

    // Override the default LogError to add extra context
    public void LogError(string message)
    {
        string enhanced = $"[ERROR @ {DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}";
        Log(enhanced);
    }
}
```

### 5.3 Static Members in Interfaces (C# 11+)

C# 11 introduced static abstract members in interfaces, enabling generic math and other patterns.

```csharp
public interface IAddable<T> where T : IAddable<T>
{
    static abstract T operator +(T left, T right);
    static abstract T Zero { get; }
}

public struct Money : IAddable<Money>
{
    public decimal Amount { get; }

    public Money(decimal amount) => Amount = amount;

    public static Money Zero => new Money(0);

    public static Money operator +(Money left, Money right)
        => new Money(left.Amount + right.Amount);

    public override string ToString() => $"${Amount:F2}";
}
```

## 6. Interface Inheritance

Interfaces can inherit from other interfaces, building up richer contracts.

### 6.1 Basic Interface Inheritance

```csharp
public interface IReadable
{
    string Read();
}

public interface IWritable
{
    void Write(string data);
}

// Combines both interfaces
public interface IReadWritable : IReadable, IWritable
{
    void Flush();
}
```

```csharp
public class MemoryBuffer : IReadWritable
{
    private readonly List<string> _buffer = new List<string>();

    public string Read()
    {
        return _buffer.Count > 0 ? _buffer[0] : "";
    }

    public void Write(string data)
    {
        _buffer.Add(data);
    }

    public void Flush()
    {
        _buffer.Clear();
        Console.WriteLine("Buffer flushed.");
    }
}
```

### 6.2 Building Complex Hierarchies

```csharp
public interface IEntity
{
    int Id { get; }
}

public interface IAuditable : IEntity
{
    DateTime CreatedAt { get; }
    DateTime? ModifiedAt { get; }
    string CreatedBy { get; }
}

public interface ISoftDeletable : IEntity
{
    bool IsDeleted { get; set; }
    DateTime? DeletedAt { get; set; }
}

public interface IFullyTracked : IAuditable, ISoftDeletable
{
    // Inherits: Id, CreatedAt, ModifiedAt, CreatedBy, IsDeleted, DeletedAt
    string Version { get; }
}
```

```csharp
public class Document : IFullyTracked
{
    public int Id { get; set; }
    public DateTime CreatedAt { get; set; }
    public DateTime? ModifiedAt { get; set; }
    public string CreatedBy { get; set; }
    public bool IsDeleted { get; set; }
    public DateTime? DeletedAt { get; set; }
    public string Version { get; set; }
    public string Title { get; set; }
    public string Content { get; set; }
}
```

## 7. `IComparable<T>` and `IEquatable<T>`

These standard .NET interfaces allow your types to participate in sorting and equality comparisons.

### 7.1 Implementing `IComparable<T>`

```csharp
public class Student : IComparable<Student>
{
    public string Name { get; set; }
    public double Gpa { get; set; }

    public int CompareTo(Student other)
    {
        if (other == null) return 1;

        // Sort by GPA descending (higher GPA first)
        int result = other.Gpa.CompareTo(Gpa);
        if (result == 0)
        {
            // If same GPA, sort by name ascending
            result = string.Compare(Name, other.Name, StringComparison.Ordinal);
        }
        return result;
    }

    public override string ToString() => $"{Name} (GPA: {Gpa:F2})";
}
```

```csharp
List<Student> students = new List<Student>
{
    new Student { Name = "Alice", Gpa = 3.8 },
    new Student { Name = "Bob", Gpa = 3.9 },
    new Student { Name = "Charlie", Gpa = 3.8 },
    new Student { Name = "Diana", Gpa = 4.0 }
};

students.Sort();  // Uses IComparable<Student>.CompareTo

foreach (Student s in students)
{
    Console.WriteLine(s);
}
// Output:
// Diana (GPA: 4.00)
// Bob (GPA: 3.90)
// Alice (GPA: 3.80)
// Charlie (GPA: 3.80)
```

### 7.2 Implementing `IEquatable<T>`

```csharp
public class Product : IEquatable<Product>
{
    public string Sku { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }

    public bool Equals(Product other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Sku == other.Sku;
    }

    public override bool Equals(object obj)
    {
        return Equals(obj as Product);
    }

    public override int GetHashCode()
    {
        return Sku?.GetHashCode() ?? 0;
    }

    public static bool operator ==(Product left, Product right)
    {
        if (left is null) return right is null;
        return left.Equals(right);
    }

    public static bool operator !=(Product left, Product right)
    {
        return !(left == right);
    }
}
```

```csharp
Product a = new Product { Sku = "ABC123", Name = "Widget", Price = 9.99m };
Product b = new Product { Sku = "ABC123", Name = "Widget v2", Price = 12.99m };
Product c = new Product { Sku = "XYZ789", Name = "Gadget", Price = 19.99m };

Console.WriteLine(a.Equals(b));    // True (same SKU)
Console.WriteLine(a == b);        // True
Console.WriteLine(a.Equals(c));    // False

// Works correctly in collections
HashSet<Product> products = new HashSet<Product> { a, b, c };
Console.WriteLine(products.Count);  // 2 (a and b are considered equal)
```

## 8. `IEnumerable<T>` — Implementing Iteration

Implementing `IEnumerable<T>` allows your custom types to be used with `foreach` loops and LINQ.

### 8.1 Basic IEnumerable Implementation

```csharp
using System.Collections;
using System.Collections.Generic;

public class NumberRange : IEnumerable<int>
{
    private readonly int _start;
    private readonly int _end;

    public NumberRange(int start, int end)
    {
        _start = start;
        _end = end;
    }

    public IEnumerator<int> GetEnumerator()
    {
        for (int i = _start; i <= _end; i++)
        {
            yield return i;
        }
    }

    // Required by IEnumerable (non-generic version)
    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
```

```csharp
NumberRange range = new NumberRange(1, 10);

foreach (int n in range)
{
    Console.Write($"{n} ");
}
// Output: 1 2 3 4 5 6 7 8 9 10

// Works with LINQ
int sum = range.Where(n => n % 2 == 0).Sum();
Console.WriteLine($"\nSum of even numbers: {sum}");  // 30
```

### 8.2 Custom Collection with IEnumerable

```csharp
public class Playlist : IEnumerable<string>
{
    private readonly List<string> _songs = new List<string>();

    public int Count => _songs.Count;

    public void Add(string song)
    {
        _songs.Add(song);
        Console.WriteLine($"Added: {song}");
    }

    public bool Remove(string song)
    {
        return _songs.Remove(song);
    }

    public string this[int index] => _songs[index];

    public IEnumerator<string> GetEnumerator()
    {
        return _songs.GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
```

```csharp
Playlist myPlaylist = new Playlist();
myPlaylist.Add("Song A");
myPlaylist.Add("Song B");
myPlaylist.Add("Song C");

foreach (string song in myPlaylist)
{
    Console.WriteLine($"Playing: {song}");
}

// LINQ works too
var sorted = myPlaylist.OrderBy(s => s).ToList();
```

### 8.3 Manual IEnumerator Implementation

For educational purposes, here is a manual implementation without `yield return`.

```csharp
public class FibonacciSequence : IEnumerable<long>
{
    private readonly int _count;

    public FibonacciSequence(int count)
    {
        _count = count;
    }

    public IEnumerator<long> GetEnumerator()
    {
        return new FibonacciEnumerator(_count);
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    private class FibonacciEnumerator : IEnumerator<long>
    {
        private readonly int _count;
        private int _index = -1;
        private long _previous = 0;
        private long _current = 1;

        public FibonacciEnumerator(int count) { _count = count; }

        public long Current { get; private set; }
        object IEnumerator.Current => Current;

        public bool MoveNext()
        {
            _index++;
            if (_index >= _count) return false;

            if (_index == 0) { Current = 0; return true; }
            if (_index == 1) { Current = 1; _previous = 0; _current = 1; return true; }

            long next = _previous + _current;
            _previous = _current;
            _current = next;
            Current = next;
            return true;
        }

        public void Reset()
        {
            _index = -1;
            _previous = 0;
            _current = 1;
        }

        public void Dispose() { }
    }
}
```

```csharp
FibonacciSequence fib = new FibonacciSequence(10);
foreach (long n in fib)
{
    Console.Write($"{n} ");
}
// Output: 0 1 1 2 3 5 8 13 21 34
```

## 9. Interface vs Abstract Class

Choosing between an interface and an abstract class is one of the most common design decisions in C#.

### 9.1 Key Differences

| Feature | Interface | Abstract Class |
|---|---|---|
| Multiple inheritance | Yes (a class can implement many) | No (single class inheritance) |
| Fields | Not allowed | Allowed |
| Constructors | Not allowed | Allowed |
| Access modifiers on members | Public by default (explicit can vary) | Any access modifier |
| Default implementation | C# 8+ only | Always supported |
| State (instance data) | No (no fields) | Yes |
| Value type support | Structs can implement interfaces | Structs cannot inherit classes |

### 9.2 When to Use Which

```csharp
// USE INTERFACE when:
// - You need multiple inheritance of behavior
// - You want to define a capability that unrelated classes share
// - You are defining a contract without shared state

public interface IExportable
{
    byte[] ExportToPdf();
    byte[] ExportToCsv();
}

// Both Report and Invoice can export, but they are unrelated
public class Report : IExportable { /* ... */ }
public class Invoice : IExportable { /* ... */ }


// USE ABSTRACT CLASS when:
// - You want to share code (fields, methods) among closely related classes
// - You need constructors or state management in the base
// - You want to define a template method pattern

public abstract class DatabaseRepository
{
    protected readonly string _connectionString;  // Shared state

    protected DatabaseRepository(string connectionString)
    {
        _connectionString = connectionString;     // Shared constructor logic
    }

    // Template method
    public List<T> GetAll<T>()
    {
        Connect();
        var results = ExecuteQuery<T>(GetSelectAllQuery());
        Disconnect();
        return results;
    }

    protected abstract string GetSelectAllQuery();
    protected abstract List<T> ExecuteQuery<T>(string query);
    protected abstract void Connect();
    protected abstract void Disconnect();
}
```

### 9.3 Combining Both

The most powerful approach often combines an interface (for the contract) with an abstract class (for shared implementation).

```csharp
// Interface defines the contract
public interface IRepository<T>
{
    T GetById(int id);
    IEnumerable<T> GetAll();
    void Add(T entity);
    void Update(T entity);
    void Delete(int id);
}

// Abstract class provides shared implementation
public abstract class RepositoryBase<T> : IRepository<T>
{
    protected readonly List<T> _items = new List<T>();

    public abstract T GetById(int id);

    public IEnumerable<T> GetAll() => _items.AsReadOnly();

    public virtual void Add(T entity)
    {
        _items.Add(entity);
    }

    public abstract void Update(T entity);
    public abstract void Delete(int id);
}
```

## 10. Practical Example: Plugin System / Strategy Pattern

Let us build a practical example that showcases interfaces as the foundation for a plugin system using the Strategy pattern.

### 10.1 Defining the Plugin Interface

```csharp
public interface ITextFormatter
{
    string Name { get; }
    string Description { get; }
    string Format(string input);
}
```

### 10.2 Implementing Multiple Formatters (Plugins)

```csharp
public class UpperCaseFormatter : ITextFormatter
{
    public string Name => "Uppercase";
    public string Description => "Converts all text to uppercase.";

    public string Format(string input)
    {
        return input.ToUpper();
    }
}

public class MarkdownBoldFormatter : ITextFormatter
{
    public string Name => "Markdown Bold";
    public string Description => "Wraps text in Markdown bold syntax.";

    public string Format(string input)
    {
        return $"**{input}**";
    }
}

public class CaesarCipherFormatter : ITextFormatter
{
    public string Name => "Caesar Cipher";
    public string Description => "Shifts each letter by 3 positions.";
    private readonly int _shift;

    public CaesarCipherFormatter(int shift = 3)
    {
        _shift = shift;
    }

    public string Format(string input)
    {
        char[] result = new char[input.Length];
        for (int i = 0; i < input.Length; i++)
        {
            char c = input[i];
            if (char.IsLetter(c))
            {
                char baseChar = char.IsUpper(c) ? 'A' : 'a';
                result[i] = (char)(((c - baseChar + _shift) % 26) + baseChar);
            }
            else
            {
                result[i] = c;
            }
        }
        return new string(result);
    }
}

public class ReverseFormatter : ITextFormatter
{
    public string Name => "Reverse";
    public string Description => "Reverses the text.";

    public string Format(string input)
    {
        char[] chars = input.ToCharArray();
        Array.Reverse(chars);
        return new string(chars);
    }
}
```

### 10.3 The Pipeline (Composite Formatter)

```csharp
public class FormatterPipeline : ITextFormatter
{
    private readonly List<ITextFormatter> _formatters = new List<ITextFormatter>();

    public string Name => "Pipeline";
    public string Description => $"Applies {_formatters.Count} formatters in sequence.";

    public FormatterPipeline Add(ITextFormatter formatter)
    {
        _formatters.Add(formatter);
        return this;  // Fluent API
    }

    public string Format(string input)
    {
        string result = input;
        foreach (ITextFormatter formatter in _formatters)
        {
            result = formatter.Format(result);
        }
        return result;
    }
}
```

### 10.4 The Plugin Manager

```csharp
public class PluginManager
{
    private readonly Dictionary<string, ITextFormatter> _plugins
        = new Dictionary<string, ITextFormatter>(StringComparer.OrdinalIgnoreCase);

    public void Register(ITextFormatter formatter)
    {
        _plugins[formatter.Name] = formatter;
        Console.WriteLine($"Registered plugin: {formatter.Name}");
    }

    public ITextFormatter GetFormatter(string name)
    {
        if (_plugins.TryGetValue(name, out ITextFormatter formatter))
        {
            return formatter;
        }
        throw new KeyNotFoundException($"Plugin '{name}' not found.");
    }

    public IEnumerable<ITextFormatter> GetAllFormatters()
    {
        return _plugins.Values;
    }

    public void ListPlugins()
    {
        Console.WriteLine("Available plugins:");
        foreach (var plugin in _plugins.Values)
        {
            Console.WriteLine($"  - {plugin.Name}: {plugin.Description}");
        }
    }
}
```

### 10.5 Putting It All Together

```csharp
class Program
{
    static void Main()
    {
        // Set up the plugin system
        PluginManager manager = new PluginManager();
        manager.Register(new UpperCaseFormatter());
        manager.Register(new MarkdownBoldFormatter());
        manager.Register(new CaesarCipherFormatter());
        manager.Register(new ReverseFormatter());

        manager.ListPlugins();
        Console.WriteLine();

        // Use individual formatters
        string text = "Hello, World!";
        Console.WriteLine($"Original: {text}");

        ITextFormatter upper = manager.GetFormatter("Uppercase");
        Console.WriteLine($"Uppercase: {upper.Format(text)}");

        ITextFormatter cipher = manager.GetFormatter("Caesar Cipher");
        Console.WriteLine($"Caesar: {cipher.Format(text)}");

        // Build a pipeline
        FormatterPipeline pipeline = new FormatterPipeline()
            .Add(new UpperCaseFormatter())
            .Add(new ReverseFormatter());

        Console.WriteLine($"Pipeline (upper + reverse): {pipeline.Format(text)}");

        // Strategy pattern: swap formatters at runtime
        Console.WriteLine("\n--- Strategy Pattern Demo ---");
        ITextFormatter strategy = new UpperCaseFormatter();
        Console.WriteLine($"Strategy 1: {strategy.Format(text)}");

        strategy = new CaesarCipherFormatter(5);
        Console.WriteLine($"Strategy 2: {strategy.Format(text)}");

        strategy = new ReverseFormatter();
        Console.WriteLine($"Strategy 3: {strategy.Format(text)}");
    }
}
```

Output:
```
Registered plugin: Uppercase
Registered plugin: Markdown Bold
Registered plugin: Caesar Cipher
Registered plugin: Reverse
Available plugins:
  - Uppercase: Converts all text to uppercase.
  - Markdown Bold: Wraps text in Markdown bold syntax.
  - Caesar Cipher: Shifts each letter by 3 positions.
  - Reverse: Reverses the text.

Original: Hello, World!
Uppercase: HELLO, WORLD!
Caesar: Khoor, Zruog!
Pipeline (upper + reverse): !DLROW ,OLLEH

--- Strategy Pattern Demo ---
Strategy 1: HELLO, WORLD!
Strategy 2: Mjqqt, Btwqi!
Strategy 3: !dlroW ,olleH
```

## 11. Practice Problems

1. **IDrawable System**: Define an `IDrawable` interface with `Draw()`, `Resize(double factor)`, and a `Bounds` property returning a `(double Width, double Height)` tuple. Implement it for `Circle`, `Rectangle`, and `TextBox` classes. Create a `Canvas` class that holds a `List<IDrawable>` and has methods `DrawAll()`, `ResizeAll(double factor)`, and `GetLargest()` (by area). Demonstrate adding different shapes to the canvas and performing operations.

2. **Multiple Interface Contact**: Create three interfaces: `IEmailable` (with `EmailAddress` property and `SendEmail(string subject, string body)` method), `IPhoneable` (with `PhoneNumber` property and `Call()` method), and `ITextable` (with `TextNumber` property and `SendText(string message)` method). Create a `BusinessContact` class that implements all three, and a `PersonalContact` that implements only `IPhoneable` and `ITextable`. Write a method that accepts `IPhoneable` and calls any phoneable contact.

3. **Explicit Interface Challenge**: Create two interfaces `IUSDateFormat` and `IEUDateFormat`, each with a `FormatDate(DateTime date)` method. Implement both in a `DateFormatter` class using explicit interface implementation so that the US format returns `MM/dd/yyyy` and the EU format returns `dd.MM.yyyy`. Also add a public `Format(DateTime date, string style)` method that delegates to the appropriate interface implementation based on the style parameter.

4. **IComparable and IEquatable**: Create a `Movie` class with `Title`, `Year`, `Rating` (1-10), and `Director` properties. Implement `IComparable<Movie>` (sort by rating descending, then by year ascending) and `IEquatable<Movie>` (equality by title and year). Create a list of 10 movies, sort them, remove duplicates using a `HashSet<Movie>`, and find the top 3 rated movies using LINQ.

5. **Custom IEnumerable**: Create a `Matrix` class that stores a 2D array of integers and implements `IEnumerable<int>` to iterate through all elements in row-major order. Add methods for `RowSum(int row)`, `ColumnSum(int col)`, and `Transpose()`. Demonstrate using `foreach` to iterate over all elements and using LINQ's `Sum()`, `Max()`, `Min()`, and `Average()` on the matrix.
