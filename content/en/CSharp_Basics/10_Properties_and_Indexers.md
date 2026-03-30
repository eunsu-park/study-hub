# Properties and Indexers

**Previous**: [Classes and Objects](./09_Classes_and_Objects.md) | **Next**: [Inheritance](./11_Inheritance.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Write full property syntax with explicit backing fields
2. Use auto-implemented properties for concise declarations
3. Create read-only and init-only properties
4. Define computed properties using expression bodies
5. Add validation logic in property setters
6. Implement indexers for custom collection-like access
7. Use static properties and the `required` keyword

---

Properties are a core feature of C# that provide controlled access to an object's data. They look like fields to the caller but behave like methods internally, allowing validation, computation, and encapsulation. Indexers extend this concept to provide array-like access using brackets. Together, properties and indexers form the standard way to expose data in well-designed C# classes.

## 1. Full Property Syntax

A full (manually implemented) property uses an explicit backing field with `get` and `set` accessors.

### 1.1 Basic Get/Set Property

```csharp
class Person
{
    // Backing field
    private string _name;

    // Property with explicit get and set
    public string Name
    {
        get
        {
            return _name;
        }
        set
        {
            _name = value; // 'value' is the implicit parameter in set
        }
    }

    // Another property with backing field
    private int _age;

    public int Age
    {
        get { return _age; }
        set { _age = value; }
    }

    public Person(string name, int age)
    {
        _name = name;
        _age = age;
    }
}

Person p = new Person("Alice", 30);
Console.WriteLine(p.Name); // Calls get accessor -> "Alice"
p.Age = 31;                // Calls set accessor
Console.WriteLine(p.Age);  // 31
```

### 1.2 Read-Only and Write-Only Properties

```csharp
class Temperature
{
    private double _celsius;

    // Read-only property (only get accessor)
    public double Celsius
    {
        get { return _celsius; }
    }

    // Write-only property (only set accessor) — rare, but valid
    public double SetCelsius
    {
        set { _celsius = value; }
    }

    public Temperature(double celsius)
    {
        _celsius = celsius;
    }
}

Temperature t = new Temperature(100);
Console.WriteLine(t.Celsius); // 100
// t.Celsius = 200;          // Compile error: read-only property

t.SetCelsius = 200;          // OK: write-only property
Console.WriteLine(t.Celsius); // 200
```

### 1.3 Access Modifier on Accessors

You can restrict one accessor to be more private than the property itself:

```csharp
class Account
{
    private decimal _balance;

    // Public get, private set
    public decimal Balance
    {
        get { return _balance; }
        private set { _balance = value; } // Only accessible within this class
    }

    public Account(decimal initialBalance)
    {
        Balance = initialBalance; // Uses private set
    }

    public void Deposit(decimal amount)
    {
        Balance += amount; // Uses private set internally
    }
}

Account acc = new Account(1000);
Console.WriteLine(acc.Balance); // 1000 (public get)
acc.Deposit(500);
Console.WriteLine(acc.Balance); // 1500
// acc.Balance = 9999;          // Compile error: set is private
```

## 2. Auto-Implemented Properties

When no special logic is needed in the get or set accessor, auto-properties provide a concise syntax. The compiler generates a hidden backing field automatically.

### 2.1 Basic Auto-Properties

```csharp
class Car
{
    // Auto-implemented properties
    public string Make { get; set; }
    public string Model { get; set; }
    public int Year { get; set; }
    public double Mileage { get; set; }

    public Car(string make, string model, int year)
    {
        Make = make;
        Model = model;
        Year = year;
        Mileage = 0;
    }

    public override string ToString()
        => $"{Year} {Make} {Model} ({Mileage:N0} miles)";
}

Car car = new Car("Toyota", "Camry", 2023);
car.Mileage = 15000;
Console.WriteLine(car); // "2023 Toyota Camry (15,000 miles)"
```

### 2.2 Auto-Properties with Default Values

```csharp
class Settings
{
    public string Theme { get; set; } = "Light";
    public int FontSize { get; set; } = 14;
    public bool ShowLineNumbers { get; set; } = true;
    public List<string> RecentFiles { get; set; } = new();

    public override string ToString()
        => $"Theme={Theme}, FontSize={FontSize}, Lines={ShowLineNumbers}";
}

Settings s = new Settings();
Console.WriteLine(s.Theme);    // "Light" (default)
Console.WriteLine(s.FontSize); // 14 (default)

s.Theme = "Dark";
Console.WriteLine(s); // "Theme=Dark, FontSize=14, Lines=True"
```

### 2.3 Private Set Auto-Properties

```csharp
class Order
{
    public string OrderId { get; private set; }
    public DateTime CreatedAt { get; private set; }
    public string Status { get; private set; }

    public Order(string orderId)
    {
        OrderId = orderId;
        CreatedAt = DateTime.Now;
        Status = "Pending";
    }

    public void Ship()
    {
        Status = "Shipped"; // Allowed: within the class
    }

    public void Deliver()
    {
        Status = "Delivered";
    }
}

Order order = new Order("ORD-001");
Console.WriteLine(order.Status); // "Pending"
order.Ship();
Console.WriteLine(order.Status); // "Shipped"
// order.Status = "Cancelled";   // Compile error: private set
```

## 3. Read-Only Properties

Properties with only a `get` accessor are read-only and can only be set during construction.

### 3.1 Get-Only Auto-Properties

```csharp
class ImmutablePoint
{
    // Get-only: can only be set in constructor or initializer
    public double X { get; }
    public double Y { get; }

    public ImmutablePoint(double x, double y)
    {
        X = x; // Allowed in constructor
        Y = y;
    }

    public double DistanceTo(ImmutablePoint other)
    {
        double dx = X - other.X;
        double dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}

ImmutablePoint p = new ImmutablePoint(3, 4);
Console.WriteLine(p.X); // 3
// p.X = 10;            // Compile error: get-only property
```

### 3.2 Readonly with Backing Field

```csharp
class Circle
{
    private readonly double _radius;

    public double Radius
    {
        get { return _radius; }
    }

    // Read-only computed property (no backing field needed)
    public double Area
    {
        get { return Math.PI * _radius * _radius; }
    }

    public double Circumference
    {
        get { return 2 * Math.PI * _radius; }
    }

    public Circle(double radius)
    {
        if (radius < 0)
            throw new ArgumentException("Radius cannot be negative.");
        _radius = radius;
    }
}

Circle c = new Circle(5);
Console.WriteLine($"Radius: {c.Radius}");           // 5
Console.WriteLine($"Area: {c.Area:F2}");             // 78.54
Console.WriteLine($"Circumference: {c.Circumference:F2}"); // 31.42
```

## 4. Init-Only Properties

C# 9 introduced `init` accessors, which allow setting a property only during object initialization (constructor or object initializer), but not afterward.

### 4.1 Basic Init-Only Properties

```csharp
class UserProfile
{
    public string Username { get; init; }
    public string Email { get; init; }
    public DateTime JoinDate { get; init; }
    public string Bio { get; set; } = ""; // Mutable after creation

    public UserProfile() { }
}

// Can set during initialization
UserProfile profile = new UserProfile
{
    Username = "alice",
    Email = "alice@example.com",
    JoinDate = DateTime.Now
};

// Can still modify mutable properties
profile.Bio = "Software developer";

// Cannot modify init-only properties after initialization
// profile.Username = "bob";    // Compile error
// profile.Email = "new@email"; // Compile error
```

### 4.2 Init-Only with Constructor

```csharp
class Product
{
    public string Name { get; init; }
    public decimal Price { get; init; }
    public string Category { get; init; }

    // Constructor can set init-only properties
    public Product(string name, decimal price, string category)
    {
        Name = name;
        Price = price;
        Category = category;
    }
}

Product p = new Product("Laptop", 999.99m, "Electronics");
// p.Price = 899.99m; // Compile error: init-only
```

### 4.3 Init-Only with Records

Init-only properties work naturally with records for immutable data:

```csharp
record class Customer
{
    public string Name { get; init; }
    public string Email { get; init; }
    public string Tier { get; init; } = "Standard";
}

Customer c = new Customer { Name = "Alice", Email = "alice@ex.com" };

// Use 'with' expression to create a modified copy
Customer upgraded = c with { Tier = "Premium" };

Console.WriteLine(c.Tier);        // "Standard"
Console.WriteLine(upgraded.Tier); // "Premium"
```

## 5. Computed Properties

Computed (calculated) properties derive their value from other data rather than storing it directly.

### 5.1 Expression-Bodied Properties

```csharp
class Rectangle
{
    public double Width { get; set; }
    public double Height { get; set; }

    // Expression-bodied read-only properties (computed on each access)
    public double Area => Width * Height;
    public double Perimeter => 2 * (Width + Height);
    public double Diagonal => Math.Sqrt(Width * Width + Height * Height);
    public bool IsSquare => Width == Height;
    public string Summary => $"{Width}x{Height} (Area={Area:F1})";

    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }
}

Rectangle r = new Rectangle(4, 3);
Console.WriteLine(r.Area);      // 12
Console.WriteLine(r.Diagonal);  // 5
Console.WriteLine(r.IsSquare);  // false
Console.WriteLine(r.Summary);   // "4x3 (Area=12.0)"

r.Width = 3;
Console.WriteLine(r.IsSquare);  // true (recomputed)
```

### 5.2 Expression-Bodied Methods vs Properties

```csharp
class DateRange
{
    public DateTime Start { get; init; }
    public DateTime End { get; init; }

    // Property: use for cheap, state-derived values
    public int TotalDays => (End - Start).Days;
    public bool IsActive => DateTime.Now >= Start && DateTime.Now <= End;

    // Method: use when it performs work, has side effects, or takes parameters
    public bool Contains(DateTime date) => date >= Start && date <= End;
    public DateRange ExtendBy(int days) => new DateRange { Start = Start, End = End.AddDays(days) };
}
```

### 5.3 Full Computed Property with Logic

```csharp
class TemperatureConverter
{
    private double _celsius;

    public double Celsius
    {
        get => _celsius;
        set => _celsius = value;
    }

    // Computed property with full get/set logic
    public double Fahrenheit
    {
        get => _celsius * 9.0 / 5.0 + 32.0;
        set => _celsius = (value - 32.0) * 5.0 / 9.0;
    }

    public double Kelvin
    {
        get => _celsius + 273.15;
        set => _celsius = value - 273.15;
    }
}

TemperatureConverter t = new TemperatureConverter();

t.Celsius = 100;
Console.WriteLine(t.Fahrenheit); // 212
Console.WriteLine(t.Kelvin);    // 373.15

t.Fahrenheit = 32;
Console.WriteLine(t.Celsius);   // 0
Console.WriteLine(t.Kelvin);    // 273.15
```

## 6. Property Validation

One of the main advantages of properties over fields is the ability to validate values in the setter.

### 6.1 Validation in Set Accessor

```csharp
class Student
{
    private string _name;
    private int _age;
    private double _gpa;

    public string Name
    {
        get => _name;
        set
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("Name cannot be empty.");
            if (value.Length > 100)
                throw new ArgumentException("Name cannot exceed 100 characters.");
            _name = value.Trim();
        }
    }

    public int Age
    {
        get => _age;
        set
        {
            if (value < 0 || value > 150)
                throw new ArgumentOutOfRangeException(nameof(value), "Age must be 0-150.");
            _age = value;
        }
    }

    public double GPA
    {
        get => _gpa;
        set
        {
            if (value < 0.0 || value > 4.0)
                throw new ArgumentOutOfRangeException(nameof(value), "GPA must be 0.0-4.0.");
            _gpa = value;
        }
    }

    public Student(string name, int age, double gpa)
    {
        Name = name;   // Validation runs in the setter
        Age = age;
        GPA = gpa;
    }
}

Student s = new Student("Alice", 20, 3.8);
Console.WriteLine(s.Name); // "Alice"

try
{
    s.Age = -5; // Throws ArgumentOutOfRangeException
}
catch (ArgumentOutOfRangeException ex)
{
    Console.WriteLine(ex.Message);
}
```

### 6.2 Notification on Change

```csharp
class ObservableValue
{
    private int _value;

    public int Value
    {
        get => _value;
        set
        {
            if (_value != value)
            {
                int oldValue = _value;
                _value = value;
                Console.WriteLine($"Value changed: {oldValue} -> {value}");
            }
        }
    }
}

ObservableValue ov = new ObservableValue();
ov.Value = 10; // "Value changed: 0 -> 10"
ov.Value = 10; // No output (same value)
ov.Value = 20; // "Value changed: 10 -> 20"
```

### 6.3 Clamping Values

```csharp
class AudioPlayer
{
    private int _volume;

    public int Volume
    {
        get => _volume;
        set => _volume = Math.Clamp(value, 0, 100); // Silently clamp to range
    }

    private double _playbackSpeed;

    public double PlaybackSpeed
    {
        get => _playbackSpeed;
        set => _playbackSpeed = Math.Clamp(value, 0.25, 4.0);
    }

    public AudioPlayer()
    {
        Volume = 50;
        PlaybackSpeed = 1.0;
    }
}

AudioPlayer player = new AudioPlayer();
player.Volume = 200;     // Clamped to 100
Console.WriteLine(player.Volume); // 100

player.Volume = -10;     // Clamped to 0
Console.WriteLine(player.Volume); // 0
```

## 7. Indexers

An indexer allows objects to be accessed using bracket notation, like an array.

### 7.1 Basic Indexer

```csharp
class Sentence
{
    private string[] _words;

    public Sentence(string text)
    {
        _words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    // Indexer
    public string this[int index]
    {
        get
        {
            if (index < 0 || index >= _words.Length)
                throw new IndexOutOfRangeException();
            return _words[index];
        }
        set
        {
            if (index < 0 || index >= _words.Length)
                throw new IndexOutOfRangeException();
            _words[index] = value;
        }
    }

    public int WordCount => _words.Length;

    public override string ToString() => string.Join(" ", _words);
}

Sentence s = new Sentence("The quick brown fox");
Console.WriteLine(s[0]);     // "The"
Console.WriteLine(s[2]);     // "brown"

s[3] = "dog";
Console.WriteLine(s);        // "The quick brown dog"
Console.WriteLine(s.WordCount); // 4
```

### 7.2 String-Keyed Indexer

Indexers are not limited to integer keys:

```csharp
class JsonObject
{
    private Dictionary<string, object> _data = new();

    // String indexer
    public object this[string key]
    {
        get
        {
            if (!_data.ContainsKey(key))
                throw new KeyNotFoundException($"Key '{key}' not found.");
            return _data[key];
        }
        set
        {
            _data[key] = value;
        }
    }

    public bool HasKey(string key) => _data.ContainsKey(key);
    public int Count => _data.Count;
}

JsonObject obj = new JsonObject();
obj["name"] = "Alice";
obj["age"] = 30;
obj["active"] = true;

Console.WriteLine(obj["name"]); // "Alice"
Console.WriteLine(obj["age"]);  // 30
Console.WriteLine(obj.Count);   // 3
```

### 7.3 Read-Only Indexer

```csharp
class FibonacciSequence
{
    private Dictionary<int, long> _cache = new() { [0] = 0, [1] = 1 };

    // Read-only indexer (expression-bodied)
    public long this[int n] => GetFibonacci(n);

    private long GetFibonacci(int n)
    {
        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(n));
        if (_cache.TryGetValue(n, out long cached))
            return cached;
        long result = GetFibonacci(n - 1) + GetFibonacci(n - 2);
        _cache[n] = result;
        return result;
    }
}

FibonacciSequence fib = new FibonacciSequence();
Console.WriteLine(fib[0]);   // 0
Console.WriteLine(fib[1]);   // 1
Console.WriteLine(fib[10]);  // 55
Console.WriteLine(fib[20]);  // 6765
```

## 8. Multi-Parameter Indexers

Indexers can accept multiple parameters, useful for grid or matrix-like access.

### 8.1 Two-Dimensional Indexer

```csharp
class Grid<T>
{
    private T[,] _data;

    public int Rows { get; }
    public int Columns { get; }

    public Grid(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        _data = new T[rows, columns];
    }

    // Multi-parameter indexer
    public T this[int row, int col]
    {
        get
        {
            ValidateBounds(row, col);
            return _data[row, col];
        }
        set
        {
            ValidateBounds(row, col);
            _data[row, col] = value;
        }
    }

    private void ValidateBounds(int row, int col)
    {
        if (row < 0 || row >= Rows || col < 0 || col >= Columns)
            throw new IndexOutOfRangeException(
                $"({row},{col}) is out of bounds for {Rows}x{Columns} grid.");
    }

    public void Fill(T value)
    {
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Columns; c++)
                _data[r, c] = value;
    }
}

Grid<int> grid = new Grid<int>(3, 4);
grid[0, 0] = 1;
grid[1, 2] = 42;
grid[2, 3] = 99;

Console.WriteLine(grid[1, 2]); // 42
```

### 8.2 Mixed-Type Indexer Parameters

```csharp
class SpreadSheet
{
    private Dictionary<(int Row, string Col), string> _cells = new();

    // Indexer with int row and string column (like "A1", "B3")
    public string this[int row, string col]
    {
        get => _cells.TryGetValue((row, col), out string? val) ? val : "";
        set => _cells[(row, col)] = value;
    }

    public int CellCount => _cells.Count;
}

SpreadSheet sheet = new SpreadSheet();
sheet[1, "A"] = "Name";
sheet[1, "B"] = "Age";
sheet[2, "A"] = "Alice";
sheet[2, "B"] = "30";

Console.WriteLine(sheet[1, "A"]); // "Name"
Console.WriteLine(sheet[2, "B"]); // "30"
Console.WriteLine(sheet[3, "C"]); // "" (empty default)
```

## 9. Static Properties

Static properties belong to the class rather than to any instance.

### 9.1 Singleton Pattern with Static Property

```csharp
class AppLogger
{
    // Private static instance
    private static AppLogger? _instance;
    private static readonly object _lock = new();

    private List<string> _logs = new();

    // Private constructor prevents external instantiation
    private AppLogger() { }

    // Static property providing the single instance
    public static AppLogger Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_lock)
                {
                    _instance ??= new AppLogger();
                }
            }
            return _instance;
        }
    }

    public int LogCount => _logs.Count;

    public void Log(string message)
    {
        _logs.Add($"[{DateTime.Now:HH:mm:ss}] {message}");
    }

    public IReadOnlyList<string> GetLogs() => _logs.AsReadOnly();
}

AppLogger.Instance.Log("Application started");
AppLogger.Instance.Log("User logged in");
Console.WriteLine(AppLogger.Instance.LogCount); // 2
```

### 9.2 Configuration with Static Properties

```csharp
class AppConfig
{
    public static string AppName { get; set; } = "MyApp";
    public static string Version { get; } = "2.0.0";
    public static bool IsDebug { get; set; } = false;
    public static int MaxRetries { get; set; } = 3;

    // Computed static property
    public static string FullVersion => $"{AppName} v{Version}" + (IsDebug ? " (Debug)" : "");
}

AppConfig.IsDebug = true;
Console.WriteLine(AppConfig.FullVersion); // "MyApp v2.0.0 (Debug)"
```

## 10. Required Members

C# 11 introduced the `required` modifier to enforce that callers set specific properties during initialization.

### 10.1 Required Properties

```csharp
class Employee
{
    public required string Name { get; set; }
    public required string Department { get; set; }
    public required string EmployeeId { get; init; }

    // Optional properties with defaults
    public string Title { get; set; } = "Staff";
    public DateTime HireDate { get; init; } = DateTime.Now;
}

// Must set all required properties
Employee emp = new Employee
{
    Name = "Alice Johnson",
    Department = "Engineering",
    EmployeeId = "EMP-001"
};

// Optional properties can be set or left as default
Employee emp2 = new Employee
{
    Name = "Bob Smith",
    Department = "Marketing",
    EmployeeId = "EMP-002",
    Title = "Manager"
};

// This would NOT compile (missing required members):
// Employee bad = new Employee { Name = "Charlie" };
```

### 10.2 Required with Constructor

```csharp
class ApiClient
{
    public required string BaseUrl { get; init; }
    public required string ApiKey { get; init; }
    public int TimeoutSeconds { get; init; } = 30;
    public bool EnableLogging { get; init; } = false;

    // SetsRequiredMembers attribute indicates the constructor sets all required members
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public ApiClient(string baseUrl, string apiKey)
    {
        BaseUrl = baseUrl;
        ApiKey = apiKey;
    }

    // Parameterless constructor still requires members to be set via initializer
    public ApiClient() { }

    public override string ToString()
        => $"ApiClient({BaseUrl}, timeout={TimeoutSeconds}s, log={EnableLogging})";
}

// Using constructor (no need to set required members again)
ApiClient client1 = new ApiClient("https://api.example.com", "key-123");

// Using object initializer (must set required members)
ApiClient client2 = new ApiClient
{
    BaseUrl = "https://api.example.com",
    ApiKey = "key-456",
    TimeoutSeconds = 60,
    EnableLogging = true
};
```

### 10.3 Required in Hierarchies

```csharp
class BaseEntity
{
    public required int Id { get; init; }
    public required DateTime CreatedAt { get; init; }
}

class Customer : BaseEntity
{
    public required string Name { get; set; }
    public required string Email { get; set; }
    public string? Phone { get; set; }
}

// Must set required members from both base and derived class
Customer c = new Customer
{
    Id = 1,
    CreatedAt = DateTime.Now,
    Name = "Alice",
    Email = "alice@example.com",
    Phone = "555-0123" // Optional
};
```

## 11. Practice Problems

1. **Temperature Class**: Create a `Temperature` class with a `Celsius` property. Add computed properties `Fahrenheit` and `Kelvin` that both have `get` and `set` accessors, converting to/from Celsius internally. Add validation that prevents setting temperature below absolute zero (-273.15 C).

2. **SafeArray Indexer**: Create a generic class `SafeArray<T>` that wraps an array and provides an indexer. If an out-of-bounds index is accessed for reading, return `default(T)` instead of throwing. If an out-of-bounds index is used for writing, automatically resize the internal array. Add a `Length` property.

3. **Config Builder with Validation**: Design a `ServerConfig` class using required and init-only properties for `Host` (must not be empty) and `Port` (must be 1-65535). Add optional properties `UseSsl` (default false), `MaxConnections` (default 100, must be > 0), and `Timeout` (default 30, must be > 0). Validate constraints using property setters that throw on invalid input.

4. **Matrix Class**: Create a `Matrix` class with a two-parameter indexer `this[int row, int col]`. Support construction from a 2D array. Add computed properties `Rows`, `Columns`, and `IsSquare`. Add a `Transpose()` method that returns a new Matrix.

5. **Property Change Tracker**: Create a class `TrackedObject` with properties `Name`, `Value`, and `Description` where every set logs the change to an internal `List<string>` recording the property name, old value, and new value. Add a read-only property `ChangeHistory` that returns the list. Use this to demonstrate how property setters enable audit trails.
