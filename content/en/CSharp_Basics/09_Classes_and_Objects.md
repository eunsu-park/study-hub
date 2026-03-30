# Classes and Objects

**Previous**: [Collections](./08_Collections.md) | **Next**: [Properties and Indexers](./10_Properties_and_Indexers.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Declare classes with fields and appropriate access modifiers
2. Write constructors including default, parameterized, and chained constructors
3. Use object initializers for concise object creation
4. Define and use static members and static classes
5. Understand the purpose and usage of the `this` keyword
6. Implement `IDisposable` for deterministic resource cleanup
7. Work with partial and nested classes
8. Distinguish reference equality from value equality

---

Classes are the primary building blocks of object-oriented programming in C#. A class defines a blueprint for objects that encapsulate data (fields and properties) and behavior (methods). Understanding how to design and use classes effectively is central to writing well-structured C# applications.

## 1. Class Declaration

A class groups related data and behavior into a single unit.

### 1.1 Basic Class Structure

```csharp
class Dog
{
    // Fields (data)
    public string Name;
    public string Breed;
    public int Age;

    // Method (behavior)
    public string Bark()
    {
        return $"{Name} says: Woof!";
    }

    public string GetInfo()
    {
        return $"{Name} is a {Age}-year-old {Breed}.";
    }
}

// Creating an object (instance of the class)
Dog myDog = new Dog();
myDog.Name = "Rex";
myDog.Breed = "German Shepherd";
myDog.Age = 5;

Console.WriteLine(myDog.Bark());    // "Rex says: Woof!"
Console.WriteLine(myDog.GetInfo()); // "Rex is a 5-year-old German Shepherd."
```

### 1.2 Classes Are Reference Types

When you assign one class variable to another, both variables refer to the same object in memory:

```csharp
class Counter
{
    public int Value;
}

Counter a = new Counter { Value = 10 };
Counter b = a; // b references the same object

b.Value = 99;
Console.WriteLine(a.Value); // 99 (same object)

// Null is a valid value for reference types
Counter? c = null;
// c.Value would throw NullReferenceException
```

## 2. Fields and Access Modifiers

Access modifiers control the visibility of class members.

### 2.1 Access Modifier Summary

| Modifier | Accessibility |
|----------|--------------|
| `public` | Accessible from anywhere |
| `private` | Accessible only within the declaring class |
| `protected` | Accessible within the class and its subclasses |
| `internal` | Accessible within the same assembly (project) |
| `protected internal` | Accessible within the same assembly OR from subclasses |
| `private protected` | Accessible within the class or subclasses in the same assembly |

### 2.2 Field Declarations

```csharp
class BankAccount
{
    // Private fields (implementation detail)
    private string _accountNumber;
    private decimal _balance;
    private static int _nextId = 1000;

    // Public field (generally discouraged; prefer properties)
    public string OwnerName;

    // Readonly field (set only in constructor or declaration)
    private readonly DateTime _createdAt;

    // Constant field (compile-time constant, implicitly static)
    public const decimal MinimumBalance = 100.00m;

    // Static readonly (runtime constant)
    public static readonly string BankName = "National Bank";

    public BankAccount(string owner, string accountNumber)
    {
        OwnerName = owner;
        _accountNumber = accountNumber;
        _balance = 0;
        _createdAt = DateTime.Now;
        _nextId++;
    }

    public decimal GetBalance() => _balance;

    public void Deposit(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("Amount must be positive.");
        _balance += amount;
    }

    public void Withdraw(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("Amount must be positive.");
        if (_balance - amount < MinimumBalance)
            throw new InvalidOperationException("Insufficient funds.");
        _balance -= amount;
    }
}
```

### 2.3 Naming Conventions

```csharp
class StyleExample
{
    // Private fields: _camelCase (with underscore prefix)
    private int _count;
    private string _name;

    // Public fields (rare): PascalCase
    public int Id;

    // Constants: PascalCase
    public const int MaxRetries = 3;

    // Methods: PascalCase
    public void DoSomething() { }

    // Parameters and local variables: camelCase
    public void Process(int itemCount)
    {
        int localVar = itemCount * 2;
    }
}
```

## 3. Constructors

Constructors initialize objects when they are created with the `new` keyword.

### 3.1 Default Constructor

If you do not define any constructor, C# provides a default parameterless constructor that sets all fields to their default values:

```csharp
class SimpleClass
{
    public int Number;     // Default: 0
    public string? Text;   // Default: null
    public bool Flag;      // Default: false
}

SimpleClass obj = new SimpleClass();
Console.WriteLine(obj.Number); // 0
Console.WriteLine(obj.Text);   // null (prints empty)
Console.WriteLine(obj.Flag);   // false
```

### 3.2 Parameterized Constructors

```csharp
class Person
{
    public string Name;
    public int Age;
    public string Email;

    // Parameterized constructor
    public Person(string name, int age, string email)
    {
        Name = name;
        Age = age;
        Email = email;
    }

    public override string ToString()
        => $"Person({Name}, {Age}, {Email})";
}

// Must use the defined constructor
Person alice = new Person("Alice", 30, "alice@example.com");

// This would NOT compile (no parameterless constructor defined):
// Person bob = new Person();
```

### 3.3 Constructor Overloading

```csharp
class Product
{
    public string Name;
    public double Price;
    public string Category;

    // Full constructor
    public Product(string name, double price, string category)
    {
        Name = name;
        Price = price;
        Category = category;
    }

    // Partial constructor with default category
    public Product(string name, double price)
    {
        Name = name;
        Price = price;
        Category = "General";
    }

    // Minimal constructor
    public Product(string name)
    {
        Name = name;
        Price = 0.0;
        Category = "Unknown";
    }
}

Product p1 = new Product("Laptop", 999.99, "Electronics");
Product p2 = new Product("Widget", 4.99);
Product p3 = new Product("Mystery");
```

### 3.4 Constructor Chaining with :this()

Constructor chaining avoids duplicating initialization logic:

```csharp
class Employee
{
    public string Name;
    public string Department;
    public double Salary;
    public DateTime HireDate;

    // Primary constructor (all parameters)
    public Employee(string name, string department, double salary, DateTime hireDate)
    {
        Name = name;
        Department = department;
        Salary = salary;
        HireDate = hireDate;
    }

    // Chain to primary: default hire date to today
    public Employee(string name, string department, double salary)
        : this(name, department, salary, DateTime.Now)
    {
    }

    // Chain further: default salary
    public Employee(string name, string department)
        : this(name, department, 50000.0)
    {
    }

    // Chain further: default department
    public Employee(string name)
        : this(name, "Unassigned")
    {
    }

    public override string ToString()
        => $"{Name} ({Department}) - ${Salary:N0}, hired {HireDate:d}";
}

Employee e1 = new Employee("Alice", "Engineering", 120000, new DateTime(2020, 3, 15));
Employee e2 = new Employee("Bob", "Marketing", 85000);
Employee e3 = new Employee("Charlie", "Sales");
Employee e4 = new Employee("Diana");
```

### 3.5 Primary Constructors (C# 12)

C# 12 allows declaring constructor parameters directly on the class:

```csharp
class Point(double x, double y)
{
    // x and y are available as parameters throughout the class
    public double X => x;
    public double Y => y;

    public double DistanceFromOrigin()
        => Math.Sqrt(x * x + y * y);

    public override string ToString() => $"({x}, {y})";
}

Point p = new Point(3, 4);
Console.WriteLine(p.DistanceFromOrigin()); // 5
```

## 4. Object Initialization

### 4.1 Object Initializers

Object initializers let you set public fields and properties at creation time without needing a matching constructor:

```csharp
class Config
{
    public string Host = "localhost";
    public int Port = 8080;
    public bool UseSsl = false;
    public string? ApiKey;
    public int TimeoutSeconds = 30;
}

// Set only the fields you want to override
Config config = new Config
{
    Host = "api.example.com",
    Port = 443,
    UseSsl = true,
    ApiKey = "secret-key-123"
    // TimeoutSeconds keeps its default value of 30
};
```

### 4.2 Combining Constructors and Initializers

```csharp
class HttpRequest
{
    public string Url;
    public string Method;
    public Dictionary<string, string> Headers;
    public string? Body;
    public int TimeoutMs;

    public HttpRequest(string url, string method = "GET")
    {
        Url = url;
        Method = method;
        Headers = new Dictionary<string, string>();
        TimeoutMs = 5000;
    }
}

// Constructor + object initializer
HttpRequest request = new HttpRequest("https://api.example.com/data")
{
    Method = "POST",
    Body = "{\"key\": \"value\"}",
    TimeoutMs = 10000,
    Headers = { ["Content-Type"] = "application/json", ["Authorization"] = "Bearer token123" }
};
```

### 4.3 Required Members (C# 11)

The `required` keyword forces callers to set specific members:

```csharp
class User
{
    public required string Username;
    public required string Email;
    public string DisplayName = "";
    public DateTime CreatedAt = DateTime.Now;
}

// Must set required members in the initializer
User user = new User
{
    Username = "alice",
    Email = "alice@example.com"
    // DisplayName and CreatedAt are optional
};

// This would NOT compile:
// User bad = new User(); // Missing required members
```

## 5. Static Members and Static Classes

### 5.1 Static Fields and Methods

Static members belong to the class itself, not to any instance:

```csharp
class MathHelper
{
    // Static field
    public static readonly double Pi = 3.14159265358979;
    private static int _callCount = 0;

    // Static method
    public static double CircleArea(double radius)
    {
        _callCount++;
        return Pi * radius * radius;
    }

    public static double CircleCircumference(double radius)
    {
        _callCount++;
        return 2 * Pi * radius;
    }

    public static int GetCallCount() => _callCount;
}

// Called on the class, not an instance
double area = MathHelper.CircleArea(5.0); // 78.54...
double circ = MathHelper.CircleCircumference(5.0);
Console.WriteLine(MathHelper.GetCallCount()); // 2
```

### 5.2 Static Constructors

A static constructor runs once, before the class is first used:

```csharp
class AppConfig
{
    public static string Environment;
    public static string Version;

    // Static constructor: no access modifier, no parameters
    static AppConfig()
    {
        // Initialize from environment or config file
        Environment = System.Environment.GetEnvironmentVariable("APP_ENV") ?? "development";
        Version = "1.0.0";
        Console.WriteLine("AppConfig initialized.");
    }
}

// Static constructor runs automatically on first access
Console.WriteLine(AppConfig.Environment); // "development"
```

### 5.3 Static Classes

A static class cannot be instantiated and can only contain static members:

```csharp
static class StringExtensions
{
    public static string Truncate(string input, int maxLength)
    {
        if (string.IsNullOrEmpty(input) || input.Length <= maxLength)
            return input;
        return input[..maxLength] + "...";
    }

    public static string Repeat(string input, int count)
    {
        return string.Concat(Enumerable.Repeat(input, count));
    }

    public static bool IsNumeric(string input)
    {
        return double.TryParse(input, out _);
    }
}

string truncated = StringExtensions.Truncate("Hello, World!", 5); // "Hello..."
string repeated = StringExtensions.Repeat("Ha", 3);               // "HaHaHa"
bool isNum = StringExtensions.IsNumeric("42.5");                   // true
```

## 6. The this Keyword

The `this` keyword refers to the current instance of the class.

### 6.1 Disambiguating Fields from Parameters

```csharp
class Rectangle
{
    private double _width;
    private double _height;

    // When parameter names match field names, use 'this' to disambiguate
    // (though using _ prefix convention mostly avoids this need)
    public Rectangle(double width, double height)
    {
        this._width = width;   // Explicit, but _ prefix already clarifies
        this._height = height;
    }

    public double Area() => _width * _height;
}

// More common scenario: when parameter names shadow fields
class Circle
{
    public double Radius;

    public Circle(double Radius)
    {
        this.Radius = Radius; // 'this.Radius' is the field, 'Radius' is the parameter
    }
}
```

### 6.2 Returning this for Fluent APIs

```csharp
class QueryBuilder
{
    private string _table = "";
    private string _where = "";
    private string _orderBy = "";
    private int _limit = 0;

    public QueryBuilder From(string table)
    {
        _table = table;
        return this; // Enable chaining
    }

    public QueryBuilder Where(string condition)
    {
        _where = condition;
        return this;
    }

    public QueryBuilder OrderBy(string column)
    {
        _orderBy = column;
        return this;
    }

    public QueryBuilder Limit(int count)
    {
        _limit = count;
        return this;
    }

    public string Build()
    {
        string query = $"SELECT * FROM {_table}";
        if (!string.IsNullOrEmpty(_where)) query += $" WHERE {_where}";
        if (!string.IsNullOrEmpty(_orderBy)) query += $" ORDER BY {_orderBy}";
        if (_limit > 0) query += $" LIMIT {_limit}";
        return query;
    }
}

// Fluent method chaining
string sql = new QueryBuilder()
    .From("users")
    .Where("age > 18")
    .OrderBy("name")
    .Limit(10)
    .Build();
// "SELECT * FROM users WHERE age > 18 ORDER BY name LIMIT 10"
```

### 6.3 Passing this to Other Methods

```csharp
class Node
{
    public string Name;
    public Node? Parent;

    public Node(string name)
    {
        Name = name;
    }

    public Node AddChild(string childName)
    {
        Node child = new Node(childName);
        child.Parent = this; // Pass current node as the parent
        return child;
    }
}

Node root = new Node("Root");
Node child = root.AddChild("Child1");
Console.WriteLine(child.Parent?.Name); // "Root"
```

## 7. Finalizers and IDisposable

### 7.1 Finalizers

A finalizer (destructor) runs when the garbage collector reclaims the object. It is used for cleaning up unmanaged resources, but it is rarely needed directly:

```csharp
class ResourceHolder
{
    private IntPtr _nativeHandle;

    public ResourceHolder()
    {
        _nativeHandle = AllocateNativeResource();
        Console.WriteLine("Resource allocated.");
    }

    // Finalizer (called by GC, non-deterministic timing)
    ~ResourceHolder()
    {
        FreeNativeResource(_nativeHandle);
        Console.WriteLine("Resource freed by finalizer.");
    }

    private static IntPtr AllocateNativeResource() => IntPtr.Zero; // Placeholder
    private static void FreeNativeResource(IntPtr handle) { }      // Placeholder
}
```

### 7.2 IDisposable Pattern

For deterministic cleanup, implement `IDisposable` and use the `using` statement:

```csharp
class DatabaseConnection : IDisposable
{
    private bool _disposed = false;
    private string _connectionString;

    public DatabaseConnection(string connectionString)
    {
        _connectionString = connectionString;
        Console.WriteLine($"Connected to: {connectionString}");
    }

    public void ExecuteQuery(string sql)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Console.WriteLine($"Executing: {sql}");
    }

    // IDisposable implementation
    public void Dispose()
    {
        if (!_disposed)
        {
            // Release managed resources
            Console.WriteLine("Connection closed.");
            _disposed = true;
        }
        GC.SuppressFinalize(this); // No need for finalizer if Dispose was called
    }

    ~DatabaseConnection()
    {
        Dispose();
    }
}

// Using statement ensures Dispose is called even if an exception occurs
using (var conn = new DatabaseConnection("Server=localhost;Database=test"))
{
    conn.ExecuteQuery("SELECT * FROM users");
} // Dispose() called here automatically

// Using declaration (C# 8+): disposes at end of enclosing scope
void ProcessData()
{
    using var conn = new DatabaseConnection("Server=localhost;Database=test");
    conn.ExecuteQuery("SELECT * FROM orders");
    // conn.Dispose() called when method exits
}
```

## 8. Partial Classes

The `partial` keyword allows splitting a class definition across multiple files. This is commonly used by code generators (e.g., WinForms designers, source generators).

### 8.1 Splitting a Class

```csharp
// File: User.cs
partial class User
{
    public string Name;
    public string Email;

    public User(string name, string email)
    {
        Name = name;
        Email = email;
    }
}

// File: User.Validation.cs
partial class User
{
    public bool IsValid()
    {
        return !string.IsNullOrWhiteSpace(Name)
            && !string.IsNullOrWhiteSpace(Email)
            && Email.Contains('@');
    }

    public List<string> GetValidationErrors()
    {
        List<string> errors = new();
        if (string.IsNullOrWhiteSpace(Name))
            errors.Add("Name is required.");
        if (string.IsNullOrWhiteSpace(Email))
            errors.Add("Email is required.");
        else if (!Email.Contains('@'))
            errors.Add("Email must contain @.");
        return errors;
    }
}

// File: User.Display.cs
partial class User
{
    public override string ToString()
        => $"{Name} <{Email}>";

    public string ToJson()
        => $"{{\"name\": \"{Name}\", \"email\": \"{Email}\"}}";
}

// Usage (all parts combined into one class)
User user = new User("Alice", "alice@example.com");
Console.WriteLine(user.IsValid());   // true
Console.WriteLine(user.ToString());  // "Alice <alice@example.com>"
Console.WriteLine(user.ToJson());
```

### 8.2 Partial Methods

Partial methods allow one part of a class to declare a method signature and another part to provide the implementation:

```csharp
// Generated code
partial class Order
{
    public decimal Total;

    public void Process()
    {
        // Call partial method (no-op if not implemented)
        OnProcessing();
        Console.WriteLine($"Processing order: ${Total}");
        OnProcessed();
    }

    // Partial method declarations
    partial void OnProcessing();
    partial void OnProcessed();
}

// Custom code
partial class Order
{
    partial void OnProcessing()
    {
        Console.WriteLine("About to process order...");
    }

    partial void OnProcessed()
    {
        Console.WriteLine("Order processed successfully.");
    }
}
```

## 9. Nested Classes

A class can be defined inside another class. Nested classes have access to the outer class's private members.

### 9.1 Basic Nested Class

```csharp
class LinkedList
{
    // Nested class: implementation detail not exposed publicly
    private class Node
    {
        public int Value;
        public Node? Next;

        public Node(int value)
        {
            Value = value;
            Next = null;
        }
    }

    private Node? _head;
    private int _count;

    public void AddFirst(int value)
    {
        Node newNode = new Node(value);
        newNode.Next = _head;
        _head = newNode;
        _count++;
    }

    public void PrintAll()
    {
        Node? current = _head;
        while (current != null)
        {
            Console.Write($"{current.Value} -> ");
            current = current.Next;
        }
        Console.WriteLine("null");
    }

    public int Count => _count;
}

LinkedList list = new LinkedList();
list.AddFirst(3);
list.AddFirst(2);
list.AddFirst(1);
list.PrintAll(); // 1 -> 2 -> 3 -> null
```

### 9.2 Public Nested Class (Builder Pattern)

```csharp
class Pizza
{
    public string Size { get; }
    public string Crust { get; }
    public List<string> Toppings { get; }

    // Private constructor: only Builder can create a Pizza
    private Pizza(string size, string crust, List<string> toppings)
    {
        Size = size;
        Crust = crust;
        Toppings = toppings;
    }

    public override string ToString()
        => $"{Size} pizza on {Crust} crust with {string.Join(", ", Toppings)}";

    // Public nested builder class
    public class Builder
    {
        private string _size = "Medium";
        private string _crust = "Regular";
        private List<string> _toppings = new();

        public Builder SetSize(string size) { _size = size; return this; }
        public Builder SetCrust(string crust) { _crust = crust; return this; }
        public Builder AddTopping(string topping) { _toppings.Add(topping); return this; }

        // Build accesses Pizza's private constructor
        public Pizza Build() => new Pizza(_size, _crust, new List<string>(_toppings));
    }
}

Pizza pizza = new Pizza.Builder()
    .SetSize("Large")
    .SetCrust("Thin")
    .AddTopping("Mozzarella")
    .AddTopping("Pepperoni")
    .AddTopping("Mushrooms")
    .Build();

Console.WriteLine(pizza);
// "Large pizza on Thin crust with Mozzarella, Pepperoni, Mushrooms"
```

## 10. Reference Equality vs Value Equality

### 10.1 Default Reference Equality

By default, the `==` operator and `Equals` method for classes check whether two variables point to the same object (reference equality):

```csharp
class Point
{
    public int X, Y;
    public Point(int x, int y) { X = x; Y = y; }
}

Point a = new Point(3, 4);
Point b = new Point(3, 4);
Point c = a;

Console.WriteLine(a == b);          // false (different objects)
Console.WriteLine(a == c);          // true (same object)
Console.WriteLine(a.Equals(b));     // false (default: reference equality)
Console.WriteLine(ReferenceEquals(a, b)); // false
Console.WriteLine(ReferenceEquals(a, c)); // true
```

### 10.2 Implementing Value Equality

Override `Equals`, `GetHashCode`, and optionally `==`/`!=` to provide value-based equality:

```csharp
class Coordinate : IEquatable<Coordinate>
{
    public double Latitude { get; }
    public double Longitude { get; }

    public Coordinate(double latitude, double longitude)
    {
        Latitude = latitude;
        Longitude = longitude;
    }

    // IEquatable<Coordinate> implementation
    public bool Equals(Coordinate? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Latitude == other.Latitude && Longitude == other.Longitude;
    }

    // Override Object.Equals
    public override bool Equals(object? obj)
        => Equals(obj as Coordinate);

    // MUST override GetHashCode when overriding Equals
    public override int GetHashCode()
        => HashCode.Combine(Latitude, Longitude);

    // Operator overloads
    public static bool operator ==(Coordinate? left, Coordinate? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(Coordinate? left, Coordinate? right)
        => !(left == right);

    public override string ToString()
        => $"({Latitude}, {Longitude})";
}

Coordinate nyc1 = new Coordinate(40.7128, -74.0060);
Coordinate nyc2 = new Coordinate(40.7128, -74.0060);
Coordinate la = new Coordinate(34.0522, -118.2437);

Console.WriteLine(nyc1 == nyc2);         // true (value equality)
Console.WriteLine(nyc1 == la);           // false
Console.WriteLine(nyc1.GetHashCode() == nyc2.GetHashCode()); // true

// Works correctly in collections
HashSet<Coordinate> visited = new() { nyc1 };
Console.WriteLine(visited.Contains(nyc2)); // true (same value)
```

### 10.3 Record Classes (Shortcut for Value Equality)

Record classes provide built-in value equality without the boilerplate:

```csharp
record class Coordinate(double Latitude, double Longitude);

Coordinate a = new Coordinate(40.7128, -74.0060);
Coordinate b = new Coordinate(40.7128, -74.0060);

Console.WriteLine(a == b);      // true (value equality, built-in)
Console.WriteLine(a.GetHashCode() == b.GetHashCode()); // true

// with expression for non-destructive mutation
Coordinate moved = a with { Latitude = 41.0 };
Console.WriteLine(moved); // Coordinate { Latitude = 41, Longitude = -74.006 }
```

## 11. Practice Problems

1. **Library Book Class**: Create a `Book` class with fields for `Title`, `Author`, `ISBN`, and `IsCheckedOut`. Add constructors (parameterized with chaining), a `CheckOut()` method, a `Return()` method, and a `ToString()` override. The ISBN should be readonly after construction.

2. **Counter with Static Tracking**: Create a `Counter` class where each instance has its own `Count` value and an `Increment()` method. Use a static field to track how many total increments have been performed across all Counter instances. Add a static method `GetGlobalCount()`.

3. **Fluent Email Builder**: Design an `EmailMessage` class with a nested `Builder` class. The builder should support chaining methods: `From(string)`, `To(string)`, `Subject(string)`, `Body(string)`, `AddAttachment(string)`, and `Build()`. Validate that From, To, and Subject are set before building.

4. **Disposable TempFile**: Create a `TempFile` class that implements `IDisposable`. The constructor creates a temporary file (using `Path.GetTempFileName()`), and `Dispose` deletes it. Add `Write(string content)` and `ReadAll()` methods. Demonstrate using it with a `using` statement.

5. **Value Equality for Playing Cards**: Create a `Card` class with `Suit` (enum: Hearts, Diamonds, Clubs, Spades) and `Rank` (enum: Ace through King). Implement `IEquatable<Card>`, override `Equals`, `GetHashCode`, and the `==`/`!=` operators. Verify that two cards with the same suit and rank are considered equal, and that they work correctly in a `HashSet<Card>`.
