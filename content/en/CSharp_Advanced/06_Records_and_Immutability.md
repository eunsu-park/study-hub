# Records and Immutability

**Previous**: [Nullable Reference Types](./05_Nullable_Reference_Types.md) | **Next**: [Async and Await](./07_Async_Await.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the benefits of immutability in software design
2. Declare record classes and record structs with positional syntax
3. Use with-expressions for non-destructive mutation
4. Understand value-based equality semantics in records
5. Apply inheritance with record hierarchies
6. Deconstruct records into component variables
7. Use init-only properties and readonly collections for immutable designs
8. Build event-sourced systems using immutable event records

---

Immutability — the practice of creating objects whose state cannot change after construction — is a cornerstone of reliable software. Immutable objects are inherently thread-safe, easy to reason about, and free from an entire category of bugs related to unintended mutation. C# records, introduced in C# 9 and expanded in C# 10, provide first-class language support for immutable data types with concise syntax, value-based equality, and non-destructive mutation via `with`-expressions.

## 1. Immutability Concepts and Benefits

### 1.1 Why Immutability Matters

```csharp
// PROBLEM: Mutable objects cause subtle bugs
public class MutablePoint
{
    public double X { get; set; }
    public double Y { get; set; }
}

var point = new MutablePoint { X = 3, Y = 4 };
var dict = new Dictionary<MutablePoint, string> { [point] = "origin" };

// Mutation after using as dictionary key — hash changes, key is now "lost"
point.X = 99;
Console.WriteLine(dict.ContainsKey(point)); // False! The entry is orphaned.

// SOLUTION: Immutable objects don't have this problem
public record ImmutablePoint(double X, double Y);

var p = new ImmutablePoint(3, 4);
var dict2 = new Dictionary<ImmutablePoint, string> { [p] = "origin" };
// p.X = 99; // Compile error — X is init-only
Console.WriteLine(dict2.ContainsKey(p)); // True — always
```

### 1.2 Benefits Summary

| Benefit | Explanation |
|---------|-------------|
| Thread safety | No synchronization needed — state never changes |
| Predictability | No action-at-a-distance; the object you receive is the object you have |
| Safe sharing | Pass by reference without defensive copying |
| Hash stability | Safe as dictionary keys and set elements |
| Undo/history | Previous states naturally preserved |
| Testing | No setup/teardown of mutable state |

## 2. Record Class Declaration

### 2.1 Positional Record Syntax

The most concise way to declare a record. The compiler generates a constructor, properties, deconstructor, equality members, and `ToString`.

```csharp
// Positional record — compiler generates everything
public record Person(string FirstName, string LastName, int Age);

// What the compiler generates (conceptually):
// - Constructor: Person(string FirstName, string LastName, int Age)
// - Init-only properties: string FirstName { get; init; }
// - Deconstruct method: void Deconstruct(out string, out string, out int)
// - Value-based Equals, GetHashCode, ==, !=
// - ToString: Person { FirstName = ..., LastName = ..., Age = ... }
// - with-expression support (clone + modify)

var alice = new Person("Alice", "Smith", 30);
Console.WriteLine(alice);
// Person { FirstName = Alice, LastName = Smith, Age = 30 }
```

### 2.2 Nominal (Non-Positional) Record Syntax

You can also declare records with explicit property definitions, which gives you more control.

```csharp
public record Product
{
    public required string Name { get; init; }
    public required decimal Price { get; init; }
    public string? Description { get; init; }
    public DateTime CreatedAt { get; init; } = DateTime.UtcNow;
}

var widget = new Product
{
    Name = "Widget",
    Price = 9.99m,
    Description = "A fine widget"
};

Console.WriteLine(widget);
// Product { Name = Widget, Price = 9.99, Description = A fine widget, CreatedAt = ... }
```

### 2.3 Mixed: Positional Parameters with Extra Properties

```csharp
public record Employee(string Name, string Department)
{
    // Additional properties beyond the positional ones
    public int YearsOfService { get; init; }
    public List<string> Skills { get; init; } = new();

    // Computed property
    public bool IsSenior => YearsOfService >= 5;
}

var emp = new Employee("Alice", "Engineering")
{
    YearsOfService = 7,
    Skills = { "C#", "SQL", "Docker" }
};

Console.WriteLine(emp);
Console.WriteLine($"Senior: {emp.IsSenior}"); // True
```

### 2.4 Record with Validation

```csharp
public record Email
{
    public string Address { get; }

    public Email(string address)
    {
        if (string.IsNullOrWhiteSpace(address))
            throw new ArgumentException("Email cannot be empty", nameof(address));
        if (!address.Contains('@'))
            throw new ArgumentException("Invalid email format", nameof(address));

        Address = address.Trim().ToLowerInvariant();
    }

    // Override ToString to show just the address
    public override string ToString() => Address;
}

var email = new Email("  Alice@Example.COM  ");
Console.WriteLine(email); // alice@example.com
```

## 3. With-Expressions

With-expressions create a copy of a record with specified properties changed. This is "non-destructive mutation" — the original is unchanged.

### 3.1 Basic Usage

```csharp
public record Point(double X, double Y);

var p1 = new Point(3, 4);
var p2 = p1 with { X = 10 };     // new Point(10, 4)
var p3 = p1 with { Y = -1 };     // new Point(3, -1)
var p4 = p1 with { X = 0, Y = 0 }; // new Point(0, 0)

Console.WriteLine(p1); // Point { X = 3, Y = 4 } — unchanged
Console.WriteLine(p2); // Point { X = 10, Y = 4 }
Console.WriteLine(p3); // Point { X = 3, Y = -1 }
```

### 3.2 With-Expressions in Practice

```csharp
public record Configuration(
    string Host,
    int Port,
    bool UseSsl,
    TimeSpan Timeout,
    int MaxRetries);

var defaultConfig = new Configuration(
    Host: "localhost",
    Port: 8080,
    UseSsl: false,
    Timeout: TimeSpan.FromSeconds(30),
    MaxRetries: 3);

// Production override — only change what's different
var prodConfig = defaultConfig with
{
    Host = "api.example.com",
    Port = 443,
    UseSsl = true,
    Timeout = TimeSpan.FromSeconds(10)
};

// Staging — starts from production but different host
var stagingConfig = prodConfig with { Host = "staging.example.com" };

Console.WriteLine(defaultConfig);
Console.WriteLine(prodConfig);
Console.WriteLine(stagingConfig);
```

### 3.3 With-Expression Creates a Shallow Copy

```csharp
public record Team(string Name, List<string> Members);

var team1 = new Team("Alpha", new List<string> { "Alice", "Bob" });
var team2 = team1 with { Name = "Beta" };

// CAUTION: Members list is shared (shallow copy)
team2.Members.Add("Charlie");
Console.WriteLine(string.Join(", ", team1.Members)); // Alice, Bob, Charlie (!)

// To avoid this, use immutable collections
public record SafeTeam(string Name, ImmutableList<string> Members);

var safe1 = new SafeTeam("Alpha", ImmutableList.Create("Alice", "Bob"));
var safe2 = safe1 with { Name = "Beta", Members = safe1.Members.Add("Charlie") };
Console.WriteLine(string.Join(", ", safe1.Members)); // Alice, Bob (unchanged)
Console.WriteLine(string.Join(", ", safe2.Members)); // Alice, Bob, Charlie
```

## 4. Value-Based Equality

Records use value-based equality by default — two records are equal if all their properties are equal, regardless of reference identity.

### 4.1 Equality Comparison

```csharp
public record Coordinate(double Latitude, double Longitude);

var a = new Coordinate(47.6062, -122.3321); // Seattle
var b = new Coordinate(47.6062, -122.3321); // Same coordinates

Console.WriteLine(a == b);           // True (value equality)
Console.WriteLine(a.Equals(b));      // True
Console.WriteLine(ReferenceEquals(a, b)); // False (different objects)

// Compare with class behavior
public class CoordinateClass
{
    public double Latitude { get; init; }
    public double Longitude { get; init; }
}

var c = new CoordinateClass { Latitude = 47.6062, Longitude = -122.3321 };
var d = new CoordinateClass { Latitude = 47.6062, Longitude = -122.3321 };

Console.WriteLine(c == d);           // False (reference equality for classes)
Console.WriteLine(c.Equals(d));      // False
```

### 4.2 Records as Dictionary Keys and in Sets

```csharp
public record CacheKey(string Endpoint, string Method, string? QueryString);

var cache = new Dictionary<CacheKey, string>();

cache[new CacheKey("/api/users", "GET", null)] = "cached user list";
cache[new CacheKey("/api/users", "GET", "?page=2")] = "cached page 2";

// Lookup with a structurally identical key
var key = new CacheKey("/api/users", "GET", null);
Console.WriteLine(cache[key]); // cached user list

// Works with HashSet too
var visited = new HashSet<CacheKey>();
visited.Add(new CacheKey("/api/users", "GET", null));
Console.WriteLine(visited.Contains(new CacheKey("/api/users", "GET", null))); // True
```

### 4.3 Custom Equality

You can override equality for specific properties (e.g., case-insensitive string comparison):

```csharp
public record PersonName(string First, string Last)
{
    public virtual bool Equals(PersonName? other)
    {
        if (other is null) return false;
        return string.Equals(First, other.First, StringComparison.OrdinalIgnoreCase)
            && string.Equals(Last, other.Last, StringComparison.OrdinalIgnoreCase);
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(
            First.ToUpperInvariant().GetHashCode(),
            Last.ToUpperInvariant().GetHashCode());
    }
}

var a = new PersonName("Alice", "Smith");
var b = new PersonName("ALICE", "SMITH");
Console.WriteLine(a == b); // True (case-insensitive)
```

## 5. Record Struct vs Record Class

C# 10 introduced `record struct` — a value type with record semantics.

### 5.1 Declaration

```csharp
// record class (reference type) — default for 'record'
public record class PersonRecord(string Name, int Age);
// Equivalent short form:
public record PersonRecord2(string Name, int Age);

// record struct (value type) — C# 10
public record struct PointRecord(double X, double Y);

// readonly record struct — truly immutable value type
public readonly record struct ImmutablePoint(double X, double Y);
```

### 5.2 Key Differences

```csharp
// record class: init-only properties (immutable by default)
public record ClassRecord(string Name);
var cr = new ClassRecord("Alice");
// cr.Name = "Bob"; // Error — init-only

// record struct: mutable properties by default
public record struct MutableStructRecord(string Name);
var msr = new MutableStructRecord("Alice");
msr.Name = "Bob"; // OK — mutable!

// readonly record struct: init-only properties (immutable)
public readonly record struct ReadonlyStructRecord(string Name);
var rsr = new ReadonlyStructRecord("Alice");
// rsr.Name = "Bob"; // Error — readonly
```

### 5.3 When to Use Which

| Feature | `record class` | `record struct` | `readonly record struct` |
|---------|----------------|-----------------|--------------------------|
| Allocated on | Heap | Stack (usually) | Stack (usually) |
| Default mutability | Immutable (init) | Mutable | Immutable (init) |
| Supports inheritance | Yes | No | No |
| `with` expressions | Yes | Yes | Yes |
| Null | Can be null | Cannot be null | Cannot be null |
| Best for | Domain entities, DTOs | Small data (2-3 fields) | Math types, coordinates |

```csharp
// Good record class candidates: domain objects, DTOs, commands
public record CreateUserCommand(string Name, string Email, string Password);
public record UserDto(int Id, string Name, string Email, DateTime CreatedAt);

// Good readonly record struct candidates: small value objects
public readonly record struct Money(decimal Amount, string Currency);
public readonly record struct DateRange(DateOnly Start, DateOnly End);
public readonly record struct Color(byte R, byte G, byte B, byte A = 255);
```

## 6. Record Inheritance

Record classes support inheritance. Record structs do not.

### 6.1 Basic Inheritance

```csharp
public abstract record Shape(string Color);
public record Circle(string Color, double Radius) : Shape(Color);
public record Rectangle(string Color, double Width, double Height) : Shape(Color);
public record Triangle(string Color, double Base, double Height) : Shape(Color);

Shape shape = new Circle("Red", 5.0);
Console.WriteLine(shape);
// Circle { Color = Red, Radius = 5 }
```

### 6.2 Equality with Inheritance

Records handle equality correctly across inheritance hierarchies — a `Circle` is never equal to a `Rectangle`, even if they share the same base properties.

```csharp
Shape s1 = new Circle("Red", 5.0);
Shape s2 = new Circle("Red", 5.0);
Shape s3 = new Rectangle("Red", 5.0, 5.0);

Console.WriteLine(s1 == s2); // True (same type and values)
Console.WriteLine(s1 == s3); // False (different types)

// The EqualityContract property ensures type-safe comparison
```

### 6.3 With-Expressions and Inheritance

```csharp
Circle c1 = new Circle("Red", 5.0);
Circle c2 = c1 with { Radius = 10.0 };       // Circle { Color = Red, Radius = 10 }
Circle c3 = c1 with { Color = "Blue" };       // Circle { Color = Blue, Radius = 5 }

// with-expressions preserve the actual type
Shape s = c1;
Shape s2 = s with { Color = "Green" }; // Still a Circle!
Console.WriteLine(s2.GetType().Name);  // Circle
Console.WriteLine(s2);                 // Circle { Color = Green, Radius = 5 }
```

### 6.4 Sealed Records

```csharp
// Prevent further inheritance
public sealed record FinalProduct(string Name, decimal Price);
// public record SpecialProduct(...) : FinalProduct(...); // Error — sealed
```

## 7. Deconstruction

Positional records automatically generate a `Deconstruct` method, enabling tuple-like decomposition.

### 7.1 Basic Deconstruction

```csharp
public record Person(string FirstName, string LastName, int Age);

var person = new Person("Alice", "Smith", 30);

// Deconstruct into variables
var (first, last, age) = person;
Console.WriteLine($"{first} {last}, age {age}"); // Alice Smith, age 30

// Partial deconstruction with discards
var (name, _, _) = person;
Console.WriteLine(name); // Alice
```

### 7.2 Deconstruction in Pattern Matching

```csharp
public record Order(string Product, int Quantity, decimal UnitPrice);

var orders = new[]
{
    new Order("Widget", 5, 9.99m),
    new Order("Gadget", 100, 2.50m),
    new Order("Doohickey", 1, 499.99m),
};

foreach (var order in orders)
{
    var message = order switch
    {
        (_, >= 50, _) => $"Bulk order: {order.Product}",
        (_, _, >= 100) => $"Premium item: {order.Product}",
        var (product, qty, price) when qty * price > 100
            => $"High-value: {product} (${qty * price:F2})",
        _ => $"Standard: {order.Product}"
    };
    Console.WriteLine(message);
}
// Bulk order: Gadget
// Premium item: Doohickey
// Standard: Widget
```

### 7.3 Custom Deconstruct on Non-Records

```csharp
public class Range
{
    public int Start { get; }
    public int End { get; }
    public int Length => End - Start;

    public Range(int start, int end) => (Start, End) = (start, end);

    public void Deconstruct(out int start, out int end)
    {
        start = Start;
        end = End;
    }

    public void Deconstruct(out int start, out int end, out int length)
    {
        start = Start;
        end = End;
        length = Length;
    }
}

var range = new Range(5, 15);
var (s, e) = range;
Console.WriteLine($"{s} to {e}"); // 5 to 15

var (start, end, len) = range;
Console.WriteLine($"{start} to {end}, length {len}"); // 5 to 15, length 10
```

## 8. Init-Only Properties and Readonly Collections

### 8.1 Init-Only Properties

```csharp
public class AppSettings
{
    public required string ConnectionString { get; init; }
    public required string ApiKey { get; init; }
    public int MaxRetries { get; init; } = 3;
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(30);
}

var settings = new AppSettings
{
    ConnectionString = "Server=localhost;Database=mydb",
    ApiKey = "secret-key",
    MaxRetries = 5
};

// settings.ConnectionString = "other"; // Error — init-only
Console.WriteLine(settings.Timeout); // 00:00:30 (default)
```

### 8.2 Immutable Collections

```csharp
using System.Collections.Immutable;

// ImmutableList<T>
var list1 = ImmutableList.Create(1, 2, 3);
var list2 = list1.Add(4);          // Returns new list [1,2,3,4]
var list3 = list1.Remove(2);       // Returns new list [1,3]
Console.WriteLine(list1.Count);     // 3 (unchanged)
Console.WriteLine(list2.Count);     // 4

// ImmutableDictionary<K,V>
var dict1 = ImmutableDictionary<string, int>.Empty
    .Add("a", 1)
    .Add("b", 2);
var dict2 = dict1.SetItem("a", 10); // Returns new dict with "a"=10
Console.WriteLine(dict1["a"]);       // 1 (unchanged)
Console.WriteLine(dict2["a"]);       // 10

// ImmutableArray<T> — better cache locality than ImmutableList
var arr1 = ImmutableArray.Create(10, 20, 30);
var arr2 = arr1.Add(40);
Console.WriteLine(arr1.Length); // 3
Console.WriteLine(arr2.Length); // 4

// Builder pattern for efficient bulk construction
var builder = ImmutableList.CreateBuilder<string>();
builder.Add("Alice");
builder.Add("Bob");
builder.Add("Charlie");
ImmutableList<string> immutable = builder.ToImmutable();
```

### 8.3 FrozenDictionary and FrozenSet (.NET 8+)

```csharp
using System.Collections.Frozen;

// Optimized for read-heavy, write-once scenarios
var data = new Dictionary<string, int>
{
    ["red"] = 0xFF0000,
    ["green"] = 0x00FF00,
    ["blue"] = 0x0000FF,
};

FrozenDictionary<string, int> frozen = data.ToFrozenDictionary();
Console.WriteLine(frozen["red"]); // 16711680

// FrozenSet
FrozenSet<string> validCommands = new[] { "start", "stop", "restart" }.ToFrozenSet();
Console.WriteLine(validCommands.Contains("start")); // True
```

## 9. Practical Example: Event Sourcing with Immutable Events

Event sourcing stores state changes as a sequence of immutable events. Records are the perfect data structure for this pattern.

### 9.1 Event Definitions

```csharp
public abstract record DomainEvent(DateTime OccurredAt);

public record AccountCreated(
    string AccountId,
    string OwnerName,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record MoneyDeposited(
    string AccountId,
    decimal Amount,
    string Description,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record MoneyWithdrawn(
    string AccountId,
    decimal Amount,
    string Description,
    DateTime OccurredAt) : DomainEvent(OccurredAt);

public record AccountClosed(
    string AccountId,
    string Reason,
    DateTime OccurredAt) : DomainEvent(OccurredAt);
```

### 9.2 Immutable State

```csharp
public record AccountState(
    string AccountId,
    string OwnerName,
    decimal Balance,
    bool IsActive,
    ImmutableList<DomainEvent> History)
{
    public static AccountState Initial => new(
        "", "", 0m, false, ImmutableList<DomainEvent>.Empty);
}
```

### 9.3 Event Application (Fold)

```csharp
public static class AccountProjection
{
    public static AccountState Apply(AccountState state, DomainEvent @event) =>
        @event switch
        {
            AccountCreated e => state with
            {
                AccountId = e.AccountId,
                OwnerName = e.OwnerName,
                Balance = 0m,
                IsActive = true,
                History = state.History.Add(e)
            },

            MoneyDeposited e => state with
            {
                Balance = state.Balance + e.Amount,
                History = state.History.Add(e)
            },

            MoneyWithdrawn e when e.Amount <= state.Balance => state with
            {
                Balance = state.Balance - e.Amount,
                History = state.History.Add(e)
            },

            MoneyWithdrawn e => throw new InvalidOperationException(
                $"Insufficient funds: balance={state.Balance}, withdrawal={e.Amount}"),

            AccountClosed e => state with
            {
                IsActive = false,
                History = state.History.Add(e)
            },

            _ => throw new ArgumentException($"Unknown event: {@event.GetType().Name}")
        };

    // Rebuild state from event stream
    public static AccountState Rebuild(IEnumerable<DomainEvent> events) =>
        events.Aggregate(AccountState.Initial, Apply);
}
```

### 9.4 Using the Event-Sourced Account

```csharp
var now = DateTime.UtcNow;

var events = new DomainEvent[]
{
    new AccountCreated("ACC-001", "Alice Smith", now),
    new MoneyDeposited("ACC-001", 1000m, "Initial deposit", now.AddMinutes(1)),
    new MoneyDeposited("ACC-001", 500m, "Salary", now.AddDays(1)),
    new MoneyWithdrawn("ACC-001", 200m, "Groceries", now.AddDays(2)),
    new MoneyDeposited("ACC-001", 300m, "Freelance work", now.AddDays(3)),
};

var currentState = AccountProjection.Rebuild(events);

Console.WriteLine($"Account: {currentState.AccountId}");
Console.WriteLine($"Owner: {currentState.OwnerName}");
Console.WriteLine($"Balance: {currentState.Balance:C}");
Console.WriteLine($"Active: {currentState.IsActive}");
Console.WriteLine($"Events: {currentState.History.Count}");

// Output:
// Account: ACC-001
// Owner: Alice Smith
// Balance: $1,600.00
// Active: True
// Events: 5

// Time travel — replay only first 3 events to see past state
var pastState = AccountProjection.Rebuild(events.Take(3));
Console.WriteLine($"Balance after 3 events: {pastState.Balance:C}");
// Balance after 3 events: $1,500.00

// Audit trail
Console.WriteLine("\nTransaction History:");
foreach (var evt in currentState.History)
{
    string desc = evt switch
    {
        AccountCreated e => $"Account created for {e.OwnerName}",
        MoneyDeposited e => $"+{e.Amount:C} ({e.Description})",
        MoneyWithdrawn e => $"-{e.Amount:C} ({e.Description})",
        AccountClosed e => $"Account closed: {e.Reason}",
        _ => "Unknown event"
    };
    Console.WriteLine($"  [{evt.OccurredAt:g}] {desc}");
}
```

## 10. Practice Problems

1. **Immutable Stack**: Implement an `ImmutableStack<T>` as a record: `record ImmutableStack<T>(T Head, ImmutableStack<T>? Tail)`. Add methods `Push(T item)`, `Pop(out T item)`, `Peek()`, and `IsEmpty`. Each operation should return a new stack. Include a static `Empty` property. Write tests that demonstrate the stack is truly immutable.

2. **Version-Controlled Document**: Create a `Document` record with `Title`, `Content`, and `Version` (int). Implement an `EditDocument(Document doc, string newContent)` function that returns a new Document with incremented version. Store all versions in an `ImmutableList<Document>` and implement `GetVersion(int version)` to retrieve any past version.

3. **Record Equality Customization**: Create a `CaseInsensitiveRecord` record with `Name` and `Value` string properties. Override equality so that `Name` comparison is case-insensitive while `Value` comparison is case-sensitive. Verify by creating instances with different casings and testing equality and hash code behavior with a `HashSet`.

4. **Builder for Immutable Config**: Design a `ServerConfig` record with 8+ properties (host, port, ssl, timeout, maxConnections, logLevel, corsOrigins as ImmutableList, headers as ImmutableDictionary). Create a `ServerConfigBuilder` class with fluent methods. The builder should validate all required fields on `Build()` and return the immutable record.

5. **Shopping Cart with Event Sourcing**: Model a shopping cart using event sourcing with records. Define events: `CartCreated`, `ItemAdded(string ProductId, int Quantity, decimal Price)`, `ItemRemoved(string ProductId)`, `QuantityChanged(string ProductId, int NewQuantity)`, `CartCheckedOut`. The cart state should be a record containing an `ImmutableDictionary<string, CartItem>`. Implement `Apply` and `Rebuild`, then write a scenario that creates a cart, adds 3 items, removes 1, changes quantity of another, and checks out.
