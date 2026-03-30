# 인터페이스 (Interfaces)

**이전**: [상속](./11_Inheritance.md) | **다음**: [제네릭](./13_Generics.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 메서드, 속성, 이벤트, 인덱서가 포함된 인터페이스를 선언하고 구현할 수 있다
2. 단일 클래스에서 여러 인터페이스를 구현할 수 있다
3. 명시적 인터페이스 구현(explicit interface implementation)을 사용하여 이름 충돌을 해결할 수 있다
4. C# 8에서 도입된 기본 인터페이스 메서드(default interface method)를 활용할 수 있다
5. 인터페이스 상속 계층 구조를 설계할 수 있다
6. `IComparable<T>` 및 `IEquatable<T>` 같은 표준 .NET 인터페이스를 구현할 수 있다
7. `IEnumerable<T>`와 `IEnumerator<T>`를 사용하여 반복 가능한 타입을 만들 수 있다
8. 설계에 인터페이스와 추상 클래스 중 적합한 것을 선택할 수 있다

---

인터페이스(Interface)는 클래스나 구조체가 충족해야 하는 계약을 정의합니다. "is-a" 관계를 모델링하는 상속과 달리, 인터페이스는 "can-do" 능력을 모델링합니다. 클래스는 여러 인터페이스를 구현할 수 있어, 다중 클래스 상속의 복잡성 없이 C#에 동작의 다중 상속 형태를 제공합니다. 인터페이스는 많은 디자인 패턴, 의존성 주입(dependency injection) 프레임워크, 그리고 .NET 표준 라이브러리 자체의 근간입니다.

## 1. 인터페이스 선언

인터페이스는 구현 타입이 제공해야 하는 멤버 집합을 선언합니다. 인터페이스에는 메서드 시그니처, 속성, 이벤트, 인덱서가 포함될 수 있습니다.

### 1.1 기본 인터페이스 구문

```csharp
public interface IShape
{
    // 메서드 시그니처 (본문 없음)
    double Area();
    double Perimeter();

    // 속성 시그니처
    string Name { get; }

    // 인터페이스에서 필드는 허용되지 않음
    // int x;  // 오류
}
```

관례적으로, C# 인터페이스 이름은 대문자 `I`로 시작합니다(예: `IShape`, `IDisposable`, `IComparable`).

### 1.2 인터페이스의 속성

```csharp
public interface IIdentifiable
{
    // 읽기 전용 속성
    string Id { get; }

    // 읽기-쓰기 속성
    string DisplayName { get; set; }
}

public interface ITimestamped
{
    DateTime CreatedAt { get; }
    DateTime? UpdatedAt { get; set; }
}
```

### 1.3 인터페이스의 이벤트

```csharp
public interface INotifier
{
    event EventHandler<string> OnNotification;
    void SendNotification(string message);
}
```

### 1.4 인터페이스의 인덱서

```csharp
public interface IReadOnlyCollection
{
    int Count { get; }
    string this[int index] { get; }
}
```

## 2. 인터페이스 구현

클래스는 콜론 뒤에 인터페이스를 나열하고 모든 멤버에 대한 구현을 제공하여 인터페이스를 구현합니다.

### 2.1 기본 구현

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

### 2.2 모든 멤버 구현은 필수

클래스가 모든 인터페이스 멤버를 구현하지 않으면 컴파일러가 오류를 보고합니다 — 클래스가 abstract인 경우는 예외입니다.

```csharp
public interface IVehicle
{
    void Start();
    void Stop();
    int Speed { get; }
}

// 이 추상 클래스는 일부 멤버를 구현하지 않은 채로 둘 수 있음
public abstract class VehicleBase : IVehicle
{
    public abstract void Start();
    public abstract void Stop();
    public abstract int Speed { get; }
}

// 구체적 클래스는 모든 것을 구현해야 함
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

### 2.3 인터페이스를 구현하는 구조체

구조체(struct)도 인터페이스를 구현할 수 있습니다. 이는 .NET에서 일반적입니다(예: `int`는 `IComparable<int>`를 구현합니다).

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

## 3. 다중 인터페이스 구현

클래스는 여러 인터페이스를 구현할 수 있으며, 이것이 C#이 다중 상속의 한 형태를 달성하는 방법입니다.

### 3.1 여러 인터페이스 구현

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

// 다른 인터페이스 참조를 통해 사용
ISerializable serializable = user;
string data = serializable.Serialize();
Console.WriteLine(data);  // "alice|alice@example.com"

ILoggable loggable = user;
loggable.Log("Profile accessed.");  // "[UserProfile:alice] Profile accessed."
```

### 3.2 기본 클래스와 인터페이스 결합

클래스는 하나의 기본 클래스를 상속하면서 여러 인터페이스를 구현할 수 있습니다.

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

기본 클래스는 목록에서 인터페이스보다 먼저 와야 합니다.

## 4. 명시적 인터페이스 구현

클래스가 같은 시그니처의 메서드를 가진 두 인터페이스를 구현하거나, 인터페이스 메서드를 클래스의 공개 API에서 숨기고 싶을 때 명시적 인터페이스 구현(explicit interface implementation)을 사용합니다.

### 4.1 이름 충돌 해결

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
    // IFileReader.Read에 대한 명시적 구현
    string IFileReader.Read()
    {
        return "Reading from file...";
    }

    // INetworkReader.Read에 대한 명시적 구현
    string INetworkReader.Read()
    {
        return "Reading from network...";
    }
}
```

```csharp
DataReader reader = new DataReader();
// reader.Read();  // 오류: 모호함, 직접 접근 불가

// 특정 인터페이스로 캐스팅해야 함
IFileReader fileReader = reader;
Console.WriteLine(fileReader.Read());      // "Reading from file..."

INetworkReader networkReader = reader;
Console.WriteLine(networkReader.Read());   // "Reading from network..."
```

### 4.2 인터페이스 멤버 숨기기

명시적 구현은 인터페이스 참조를 통해서만 접근해야 하는 멤버를 숨길 수도 있습니다.

```csharp
public interface IResettable
{
    void Reset();
}

public class GameState : IResettable
{
    public int Score { get; set; }
    public int Level { get; set; }

    // 명시적: Reset()은 IResettable 참조를 통해서만 보임
    void IResettable.Reset()
    {
        Score = 0;
        Level = 1;
        Console.WriteLine("Game state has been reset.");
    }

    // 일반 사용을 위한 public 메서드
    public void NewGame()
    {
        ((IResettable)this).Reset();
        Console.WriteLine("Starting new game...");
    }
}
```

```csharp
GameState game = new GameState { Score = 100, Level = 5 };
// game.Reset();  // 오류: 직접 접근 불가

IResettable resettable = game;
resettable.Reset();  // 인터페이스 참조를 통해 작동

game.NewGame();      // 내부적으로 Reset()을 호출하여 작동
```

### 4.3 암시적 vs 명시적: 빠른 비교

```csharp
public interface IAnimal
{
    void Speak();
}

// 암시적: Speak()은 클래스에서 public
public class Dog : IAnimal
{
    public void Speak() => Console.WriteLine("Woof!");
}

// 명시적: Speak()은 IAnimal을 통해서만 접근 가능
public class Cat : IAnimal
{
    void IAnimal.Speak() => Console.WriteLine("Meow!");
}

Dog dog = new Dog();
dog.Speak();              // 작동
((IAnimal)dog).Speak();   // 역시 작동

Cat cat = new Cat();
// cat.Speak();            // 오류
((IAnimal)cat).Speak();   // 작동
```

## 5. 기본 인터페이스 메서드 (C# 8+)

C# 8부터 인터페이스에 메서드 본문을 포함할 수 있습니다 — 이를 기본 구현(default implementation)이라고 합니다. 이를 통해 이미 인터페이스를 구현한 클래스를 깨뜨리지 않고 새 메서드를 추가할 수 있습니다.

### 5.1 기본 메서드 기초

```csharp
public interface ILogger
{
    void Log(string message);

    // 기본 구현 — 클래스에서 재정의할 필요 없음
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
    // LogError, LogWarning, LogInfo는 기본 동작으로 상속됨
}
```

```csharp
ILogger logger = new ConsoleLogger();
logger.Log("Direct message");
logger.LogError("Something failed");
logger.LogWarning("Low disk space");
logger.LogInfo("Process started");

// 참고: 기본 메서드는 인터페이스 참조를 통해서만 접근 가능
ConsoleLogger concrete = new ConsoleLogger();
concrete.Log("works");         // 작동 (명시적으로 구현됨)
// concrete.LogError("test");  // ConsoleLogger가 명시적으로 구현하지 않으면 오류
ILogger asInterface = concrete;
asInterface.LogError("test");  // 인터페이스 참조를 통해 작동
```

### 5.2 기본 메서드 재정의

클래스는 커스텀 동작이 필요한 경우 기본 구현을 재정의할 수 있습니다.

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

    // 기본 LogError를 재정의하여 추가 컨텍스트 추가
    public void LogError(string message)
    {
        string enhanced = $"[ERROR @ {DateTime.Now:yyyy-MM-dd HH:mm:ss}] {message}";
        Log(enhanced);
    }
}
```

### 5.3 인터페이스의 정적 멤버 (C# 11+)

C# 11에서는 인터페이스에 정적 추상 멤버(static abstract member)가 도입되어 제네릭 수학 등의 패턴을 가능하게 합니다.

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

## 6. 인터페이스 상속

인터페이스는 다른 인터페이스를 상속하여 더 풍부한 계약을 구성할 수 있습니다.

### 6.1 기본 인터페이스 상속

```csharp
public interface IReadable
{
    string Read();
}

public interface IWritable
{
    void Write(string data);
}

// 두 인터페이스를 결합
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

### 6.2 복잡한 계층 구조 구성

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
    // 상속받은 것: Id, CreatedAt, ModifiedAt, CreatedBy, IsDeleted, DeletedAt
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

## 7. `IComparable<T>`와 `IEquatable<T>`

이 표준 .NET 인터페이스를 사용하면 사용자 정의 타입이 정렬과 동등성 비교에 참여할 수 있습니다.

### 7.1 `IComparable<T>` 구현

```csharp
public class Student : IComparable<Student>
{
    public string Name { get; set; }
    public double Gpa { get; set; }

    public int CompareTo(Student other)
    {
        if (other == null) return 1;

        // GPA 내림차순으로 정렬 (높은 GPA가 먼저)
        int result = other.Gpa.CompareTo(Gpa);
        if (result == 0)
        {
            // GPA가 같으면 이름 오름차순으로 정렬
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

students.Sort();  // IComparable<Student>.CompareTo 사용

foreach (Student s in students)
{
    Console.WriteLine(s);
}
// 출력:
// Diana (GPA: 4.00)
// Bob (GPA: 3.90)
// Alice (GPA: 3.80)
// Charlie (GPA: 3.80)
```

### 7.2 `IEquatable<T>` 구현

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

Console.WriteLine(a.Equals(b));    // True (같은 SKU)
Console.WriteLine(a == b);        // True
Console.WriteLine(a.Equals(c));    // False

// 컬렉션에서 올바르게 작동
HashSet<Product> products = new HashSet<Product> { a, b, c };
Console.WriteLine(products.Count);  // 2 (a와 b는 동등하다고 간주)
```

## 8. `IEnumerable<T>` — 반복 구현

`IEnumerable<T>`를 구현하면 사용자 정의 타입을 `foreach` 루프와 LINQ에서 사용할 수 있습니다.

### 8.1 기본 IEnumerable 구현

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

    // IEnumerable(비제네릭 버전)에서 필요
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
// 출력: 1 2 3 4 5 6 7 8 9 10

// LINQ와 함께 작동
int sum = range.Where(n => n % 2 == 0).Sum();
Console.WriteLine($"\nSum of even numbers: {sum}");  // 30
```

### 8.2 IEnumerable을 사용한 사용자 정의 컬렉션

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

// LINQ도 작동
var sorted = myPlaylist.OrderBy(s => s).ToList();
```

### 8.3 수동 IEnumerator 구현

학습 목적으로, `yield return` 없이 수동으로 구현하는 방법입니다.

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
// 출력: 0 1 1 2 3 5 8 13 21 34
```

## 9. 인터페이스 vs 추상 클래스

인터페이스와 추상 클래스 중 선택하는 것은 C#에서 가장 흔한 설계 결정 중 하나입니다.

### 9.1 주요 차이점

| 특징 | 인터페이스 | 추상 클래스 |
|---|---|---|
| 다중 상속 | 예 (클래스가 여러 개 구현 가능) | 아니오 (단일 클래스 상속) |
| 필드 | 허용되지 않음 | 허용됨 |
| 생성자 | 허용되지 않음 | 허용됨 |
| 멤버의 접근 한정자 | 기본적으로 public (명시적은 다를 수 있음) | 모든 접근 한정자 가능 |
| 기본 구현 | C# 8+ 이상에서만 | 항상 지원 |
| 상태 (인스턴스 데이터) | 없음 (필드 없음) | 있음 |
| 값 타입 지원 | 구조체가 인터페이스 구현 가능 | 구조체는 클래스 상속 불가 |

### 9.2 어떤 것을 사용할지

```csharp
// 인터페이스를 사용하는 경우:
// - 동작의 다중 상속이 필요할 때
// - 관련 없는 클래스가 공유하는 능력을 정의할 때
// - 공유 상태 없이 계약을 정의할 때

public interface IExportable
{
    byte[] ExportToPdf();
    byte[] ExportToCsv();
}

// Report과 Invoice 모두 내보낼 수 있지만 서로 관련 없음
public class Report : IExportable { /* ... */ }
public class Invoice : IExportable { /* ... */ }


// 추상 클래스를 사용하는 경우:
// - 밀접하게 관련된 클래스 간에 코드(필드, 메서드)를 공유하고 싶을 때
// - 기본에 생성자나 상태 관리가 필요할 때
// - 템플릿 메서드 패턴을 정의하고 싶을 때

public abstract class DatabaseRepository
{
    protected readonly string _connectionString;  // 공유 상태

    protected DatabaseRepository(string connectionString)
    {
        _connectionString = connectionString;     // 공유 생성자 로직
    }

    // 템플릿 메서드
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

### 9.3 둘의 결합

가장 강력한 접근법은 종종 인터페이스(계약용)와 추상 클래스(공유 구현용)를 결합하는 것입니다.

```csharp
// 인터페이스가 계약을 정의
public interface IRepository<T>
{
    T GetById(int id);
    IEnumerable<T> GetAll();
    void Add(T entity);
    void Update(T entity);
    void Delete(int id);
}

// 추상 클래스가 공유 구현을 제공
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

## 10. 실전 예제: 플러그인 시스템 / 전략 패턴

전략 패턴(Strategy Pattern)을 사용한 플러그인 시스템의 기반으로서 인터페이스를 보여주는 실전 예제를 만들어 봅시다.

### 10.1 플러그인 인터페이스 정의

```csharp
public interface ITextFormatter
{
    string Name { get; }
    string Description { get; }
    string Format(string input);
}
```

### 10.2 여러 포매터(플러그인) 구현

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

### 10.3 파이프라인 (복합 포매터)

```csharp
public class FormatterPipeline : ITextFormatter
{
    private readonly List<ITextFormatter> _formatters = new List<ITextFormatter>();

    public string Name => "Pipeline";
    public string Description => $"Applies {_formatters.Count} formatters in sequence.";

    public FormatterPipeline Add(ITextFormatter formatter)
    {
        _formatters.Add(formatter);
        return this;  // 플루언트 API
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

### 10.4 플러그인 관리자

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

### 10.5 모든 것 조합하기

```csharp
class Program
{
    static void Main()
    {
        // 플러그인 시스템 설정
        PluginManager manager = new PluginManager();
        manager.Register(new UpperCaseFormatter());
        manager.Register(new MarkdownBoldFormatter());
        manager.Register(new CaesarCipherFormatter());
        manager.Register(new ReverseFormatter());

        manager.ListPlugins();
        Console.WriteLine();

        // 개별 포매터 사용
        string text = "Hello, World!";
        Console.WriteLine($"Original: {text}");

        ITextFormatter upper = manager.GetFormatter("Uppercase");
        Console.WriteLine($"Uppercase: {upper.Format(text)}");

        ITextFormatter cipher = manager.GetFormatter("Caesar Cipher");
        Console.WriteLine($"Caesar: {cipher.Format(text)}");

        // 파이프라인 구성
        FormatterPipeline pipeline = new FormatterPipeline()
            .Add(new UpperCaseFormatter())
            .Add(new ReverseFormatter());

        Console.WriteLine($"Pipeline (upper + reverse): {pipeline.Format(text)}");

        // 전략 패턴: 런타임에 포매터 교체
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

출력:
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

## 11. 연습 문제

1. **IDrawable 시스템**: `Draw()`, `Resize(double factor)`, 그리고 `(double Width, double Height)` 튜플을 반환하는 `Bounds` 속성이 있는 `IDrawable` 인터페이스를 정의하세요. `Circle`, `Rectangle`, `TextBox` 클래스에 구현하세요. `List<IDrawable>`을 보유하고 `DrawAll()`, `ResizeAll(double factor)`, `GetLargest()` (면적 기준) 메서드가 있는 `Canvas` 클래스를 만드세요. 캔버스에 다양한 도형을 추가하고 연산을 수행하는 것을 시연하세요.

2. **다중 인터페이스 연락처**: 세 개의 인터페이스를 만드세요: `IEmailable`(`EmailAddress` 속성과 `SendEmail(string subject, string body)` 메서드), `IPhoneable`(`PhoneNumber` 속성과 `Call()` 메서드), `ITextable`(`TextNumber` 속성과 `SendText(string message)` 메서드). 세 가지 모두 구현하는 `BusinessContact` 클래스와, `IPhoneable`과 `ITextable`만 구현하는 `PersonalContact`를 만드세요. `IPhoneable`을 받아 전화 가능한 모든 연락처에 전화를 거는 메서드를 작성하세요.

3. **명시적 인터페이스 도전**: 각각 `FormatDate(DateTime date)` 메서드가 있는 `IUSDateFormat`과 `IEUDateFormat` 두 인터페이스를 만드세요. 명시적 인터페이스 구현을 사용하여 `DateFormatter` 클래스에 둘 다 구현하세요. US 형식은 `MM/dd/yyyy`를, EU 형식은 `dd.MM.yyyy`를 반환하게 하세요. 스타일 매개변수에 따라 적절한 인터페이스 구현에 위임하는 public `Format(DateTime date, string style)` 메서드도 추가하세요.

4. **IComparable과 IEquatable**: `Title`, `Year`, `Rating`(1-10), `Director` 속성이 있는 `Movie` 클래스를 만드세요. `IComparable<Movie>`(평점 내림차순, 그 다음 연도 오름차순)와 `IEquatable<Movie>`(제목과 연도로 동등성)를 구현하세요. 10개의 영화 목록을 만들고, 정렬하고, `HashSet<Movie>`로 중복을 제거하고, LINQ를 사용하여 상위 3개 영화를 찾으세요.

5. **사용자 정의 IEnumerable**: 2D 정수 배열을 저장하고 행 우선 순서(row-major order)로 모든 요소를 순회하는 `IEnumerable<int>`를 구현하는 `Matrix` 클래스를 만드세요. `RowSum(int row)`, `ColumnSum(int col)`, `Transpose()` 메서드를 추가하세요. `foreach`로 모든 요소를 순회하고 LINQ의 `Sum()`, `Max()`, `Min()`, `Average()`를 매트릭스에 사용하는 것을 시연하세요.
