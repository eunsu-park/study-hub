# Nullable Reference Types

**Previous**: [Pattern Matching](./04_Pattern_Matching.md) | **Next**: [Records and Immutability](./06_Records_and_Immutability.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the null reference problem and why nullable reference types were introduced
2. Enable and configure nullable contexts at file and project level
3. Annotate types with nullable (`?`) and non-nullable declarations
4. Use the null-forgiving operator (`!`) appropriately and sparingly
5. Apply null-coalescing (`??`), null-conditional (`?.`), and null-coalescing assignment (`??=`) operators
6. Understand how the compiler performs flow analysis for nullability
7. Use nullable attributes to express complex null contracts
8. Migrate existing codebases to nullable reference types

---

`NullReferenceException` is one of the most common runtime errors in C# (and many other languages). Tony Hoare, who introduced null references in 1965, famously called it his "billion-dollar mistake." Nullable reference types (NRT), introduced in C# 8, add static analysis to the compiler that warns you when code might dereference null, turning a class of runtime errors into compile-time warnings.

## 1. The Null Reference Problem

### 1.1 Why Null Is Dangerous

Before NRT, every reference type in C# was implicitly nullable. The compiler provided no help distinguishing between variables that should never be null and those that intentionally might be.

```csharp
// Before NRT — everything can be null, no warnings
public class UserService
{
    public User GetUser(int id)
    {
        // Might return null if user not found — caller has no idea
        return _database.FindById(id); // might be null!
    }
}

// Caller trusts the return type but gets a NullReferenceException
var user = service.GetUser(999);
Console.WriteLine(user.Name); // BOOM — NullReferenceException
```

### 1.2 Traditional Null Guards

Without compiler support, developers relied on manual null checks, which are verbose and error-prone.

```csharp
// Defensive programming — null checks everywhere
public void ProcessUser(User? user)
{
    if (user == null)
        throw new ArgumentNullException(nameof(user));

    if (user.Address == null)
        throw new InvalidOperationException("User has no address");

    if (user.Address.City == null)
        throw new InvalidOperationException("Address has no city");

    Console.WriteLine(user.Address.City.ToUpper());
}
```

## 2. Enabling Nullable Context

### 2.1 Project-Level (Recommended)

The most common approach is to enable NRT for the entire project in the `.csproj` file.

```xml
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>  <!-- Enable for entire project -->
  </PropertyGroup>
</Project>
```

The `<Nullable>` element supports these values:

| Value | Annotations | Warnings |
|-------|-------------|----------|
| `enable` | Yes | Yes |
| `warnings` | No | Yes |
| `annotations` | Yes | No |
| `disable` | No | No |

### 2.2 File-Level Directives

You can override the project setting per-file using `#nullable` directives.

```csharp
#nullable enable   // Enable both annotations and warnings from this point

public class MyClass
{
    public string Name { get; set; } = ""; // Non-nullable — must be initialized
    public string? NickName { get; set; }  // Nullable — can be null
}

#nullable disable  // Disable from this point (revert to pre-C# 8 behavior)

public class LegacyClass
{
    public string Name { get; set; } // No warnings, old behavior
}
```

### 2.3 Scoped Directives

```csharp
#nullable enable

public void Method()
{
    string nonNull = "hello";

    #nullable disable warnings  // Suppress warnings but keep annotations
    string? maybeNull = null;
    Console.WriteLine(maybeNull.Length); // No warning (suppressed)
    #nullable restore warnings  // Restore to outer context setting

    Console.WriteLine(maybeNull.Length); // Warning restored
}
```

## 3. Nullable Annotations

### 3.1 Non-Nullable vs Nullable Reference Types

With NRT enabled, reference types are non-nullable by default. Append `?` to make them nullable.

```csharp
#nullable enable

string name = "Alice";     // Non-nullable — guaranteed not null
string? nickname = null;   // Nullable — can be null

// Compiler warnings:
name = null;               // Warning CS8600: Converting null literal to non-nullable type
Console.WriteLine(nickname.Length); // Warning CS8602: Dereference of a possibly null reference
```

### 3.2 Method Signatures

```csharp
public class UserRepository
{
    // Return type says: this ALWAYS returns a user (never null)
    public User GetUserOrThrow(int id)
    {
        return _users.TryGetValue(id, out var user)
            ? user
            : throw new KeyNotFoundException($"User {id} not found");
    }

    // Return type says: this MIGHT return null
    public User? FindUser(int id)
    {
        _users.TryGetValue(id, out var user);
        return user; // might be null — caller is warned
    }

    // Parameter says: null is NOT acceptable
    public void UpdateUser(User user)
    {
        ArgumentNullException.ThrowIfNull(user);
        _users[user.Id] = user;
    }

    // Parameter says: null IS acceptable
    public void SetNickname(int userId, string? nickname)
    {
        var user = GetUserOrThrow(userId);
        user.Nickname = nickname; // OK — Nickname is string?
    }
}
```

### 3.3 Collections and Generics

```csharp
// List of non-nullable strings — no element can be null
List<string> names = new() { "Alice", "Bob" };
// names.Add(null); // Warning!

// List of nullable strings — elements can be null
List<string?> optionalNames = new() { "Alice", null, "Charlie" };

// Dictionary with nullable values
Dictionary<string, string?> settings = new()
{
    ["theme"] = "dark",
    ["locale"] = null  // OK — value is string?
};

// Generic constraint for non-null
public T EnsureNotNull<T>(T? value) where T : notnull
{
    return value ?? throw new ArgumentNullException(nameof(value));
}
```

## 4. The Null-Forgiving Operator (!)

The null-forgiving (or null-suppression) operator `!` tells the compiler: "I know this is not null, trust me." Use it sparingly — it suppresses warnings without providing safety.

### 4.1 When It's Appropriate

```csharp
// After a manual null check the compiler can't see
public void Process(Dictionary<string, string> dict, string key)
{
    if (dict.ContainsKey(key))
    {
        // Compiler doesn't know ContainsKey guarantees the indexer won't return null
        string value = dict[key]!; // OK — we just checked
    }
}

// Framework initialization patterns (e.g., dependency injection)
public class MyController
{
    // Set by DI framework before any method is called
    [Inject] public ILogger Logger { get; set; } = null!;
}

// Test setup
[Fact]
public void Test_Something()
{
    var service = new MyService();
    // We know Setup() initializes all properties
    service.Setup();
    var result = service.Data!.FirstItem; // suppress false positive
}
```

### 4.2 When to Avoid It

```csharp
// BAD: suppressing a legitimate warning
string? name = GetName();
Console.WriteLine(name!.Length); // Danger! name might actually be null

// GOOD: handle the null case
string? name = GetName();
if (name is not null)
{
    Console.WriteLine(name.Length); // Safe — compiler knows it's not null
}

// GOOD: provide a default
Console.WriteLine((name ?? "unknown").Length);
```

## 5. Null Operators

C# provides several operators for ergonomic null handling.

### 5.1 Null-Conditional Operator (?.)

Short-circuits to null if the left side is null, instead of throwing.

```csharp
string? name = null;

// Without ?.
int length1 = name != null ? name.Length : 0;

// With ?.
int? length2 = name?.Length; // null (not NullReferenceException)
int length3 = name?.Length ?? 0; // 0

// Chaining
string? city = user?.Address?.City?.ToUpper();

// With indexers
int? first = list?[0];

// With method calls
string? upper = name?.ToUpper();
```

### 5.2 Null-Coalescing Operator (??)

Returns the left operand if non-null; otherwise, returns the right operand.

```csharp
string? input = null;

// Provide a default value
string result = input ?? "default";
Console.WriteLine(result); // default

// Chain multiple fallbacks
string? primary = null;
string? secondary = null;
string? tertiary = "fallback";
string value = primary ?? secondary ?? tertiary ?? "last resort";
Console.WriteLine(value); // fallback

// With method calls
string config = GetConfigValue("key") ?? LoadDefault("key") ?? "hardcoded";
```

### 5.3 Null-Coalescing Assignment (??=)

Assigns to the left operand only if it is currently null.

```csharp
List<string>? names = null;

// Old way
if (names == null)
    names = new List<string>();

// With ??=
names ??= new List<string>();

// Useful for lazy initialization
private Dictionary<string, object>? _cache;
public Dictionary<string, object> Cache => _cache ??= new Dictionary<string, object>();

// Another common pattern — default parameter values
public void Configure(Action<Options>? configure = null)
{
    var options = new Options();
    configure ??= static _ => { }; // no-op if null
    configure(options);
}
```

### 5.4 Combining Null Operators

```csharp
public class Config
{
    private Dictionary<string, string>? _overrides;
    private Dictionary<string, string>? _defaults;

    public string GetValue(string key)
    {
        return _overrides?.GetValueOrDefault(key)
            ?? _defaults?.GetValueOrDefault(key)
            ?? throw new KeyNotFoundException($"Config key '{key}' not found");
    }
}

// Null-conditional with event invocation
public event EventHandler<string>? MessageReceived;
protected void OnMessage(string msg) => MessageReceived?.Invoke(this, msg);
```

## 6. Compiler Flow Analysis

The C# compiler tracks nullability through control flow. After a null check, it narrows the type automatically.

### 6.1 Basic Flow Analysis

```csharp
public void Process(string? input)
{
    // Here, 'input' is string? — might be null
    // Console.WriteLine(input.Length); // Warning!

    if (input is null)
        return;

    // After the null check, compiler knows 'input' is string (non-null)
    Console.WriteLine(input.Length); // No warning — flow analysis narrowed the type
}
```

### 6.2 Pattern-Based Narrowing

```csharp
public void HandleValue(object? value)
{
    // Type pattern narrows and binds
    if (value is string text)
    {
        Console.WriteLine(text.Length); // text is string, not string?
    }

    // Switch expression
    string result = value switch
    {
        string s => s.ToUpper(),    // s is non-null string
        int n => n.ToString(),      // n is non-null int (value type)
        null => "(null)",
        _ => value.ToString() ?? "" // value is non-null here (null handled above)
    };
}
```

### 6.3 Assertion Methods

Some methods assert that a value is non-null. The compiler can recognize these through attributes.

```csharp
// ArgumentNullException.ThrowIfNull (built-in, recognized by compiler)
public void SetName(string? name)
{
    ArgumentNullException.ThrowIfNull(name);
    // After ThrowIfNull, compiler knows 'name' is non-null
    _name = name; // No warning
}

// Debug.Assert — does NOT affect flow analysis by default
public void Process(string? data)
{
    Debug.Assert(data != null);
    // Warning: compiler doesn't trust Debug.Assert for null analysis
    // Use [DoesNotReturnIf(false)] attribute to teach it
}
```

### 6.4 Flow Analysis Limitations

```csharp
public void Limitations(string? a, string? b)
{
    // Compiler DOES track through simple conditions
    if (a != null && b != null)
    {
        Console.WriteLine(a.Length + b.Length); // OK
    }

    // Compiler DOES NOT track through method calls
    bool isValid = a != null;
    if (isValid)
    {
        // Console.WriteLine(a.Length); // Warning! Compiler lost track
        Console.WriteLine(a!.Length);   // Need ! to suppress
    }

    // Compiler tracks through ?? and ??=
    a ??= "default";
    Console.WriteLine(a.Length); // OK — a is guaranteed non-null after ??=
}
```

## 7. Nullable Attributes

The `System.Diagnostics.CodeAnalysis` namespace provides attributes that give the compiler additional information about null contracts that cannot be expressed through `?` alone.

### 7.1 Precondition Attributes

```csharp
using System.Diagnostics.CodeAnalysis;

public class Validator
{
    // [NotNull] — the parameter will be non-null when the method returns normally
    public static void EnsureNotNull([NotNull] string? value)
    {
        if (value is null)
            throw new ArgumentNullException(nameof(value));
        // If we reach here, value is non-null
    }

    // [DoesNotReturnIf] — method does not return if parameter equals the given bool
    public static void Assert([DoesNotReturnIf(false)] bool condition, string? message = null)
    {
        if (!condition)
            throw new InvalidOperationException(message ?? "Assertion failed");
    }
}

// Usage
public void Process(string? input)
{
    Validator.EnsureNotNull(input);
    Console.WriteLine(input.Length); // No warning — compiler knows input is non-null

    string? name = GetName();
    Validator.Assert(name != null, "Name must not be null");
    Console.WriteLine(name.Length); // No warning
}
```

### 7.2 Postcondition Attributes

```csharp
public class Parser
{
    // [NotNullWhen(true)] — parameter is non-null when method returns true
    public static bool TryParse(string? input, [NotNullWhen(true)] out string? result)
    {
        if (string.IsNullOrWhiteSpace(input))
        {
            result = null;
            return false;
        }
        result = input.Trim();
        return true;
    }

    // [MaybeNullWhen(false)] — output may be null when method returns false
    public static bool TryGetValue<T>(
        Dictionary<string, T> dict,
        string key,
        [MaybeNullWhen(false)] out T value)
    {
        return dict.TryGetValue(key, out value);
    }
}

// Usage
if (Parser.TryParse(input, out var parsed))
{
    Console.WriteLine(parsed.Length); // No warning — NotNullWhen(true) guarantees non-null
}
```

### 7.3 Member Attributes

```csharp
public class Cache<TKey, TValue> where TKey : notnull
{
    private readonly Dictionary<TKey, TValue> _dict = new();

    // [MaybeNull] — return value might be null even though T might be non-nullable
    [return: MaybeNull]
    public TValue GetOrDefault(TKey key)
    {
        _dict.TryGetValue(key, out var value);
        return value; // might be default(TValue), which is null for reference types
    }

    // [DisallowNull] — parameter must not be null even though type allows it
    public void Set(TKey key, [DisallowNull] TValue? value)
    {
        // value can be typed as TValue? but callers get a warning if they pass null
        _dict[key] = value;
    }

    // [MemberNotNull] — guarantees a member is non-null after the method returns
    private string? _connectionString;

    [MemberNotNull(nameof(_connectionString))]
    public void Initialize(string connectionString)
    {
        _connectionString = connectionString ?? throw new ArgumentNullException(nameof(connectionString));
    }

    // [MemberNotNullWhen] — member is non-null when method returns a specific bool
    [MemberNotNullWhen(true, nameof(_connectionString))]
    public bool IsInitialized => _connectionString is not null;
}
```

### 7.4 Summary of Nullable Attributes

| Attribute | Applied To | Meaning |
|-----------|-----------|---------|
| `[NotNull]` | Parameter, out | Non-null after method returns |
| `[MaybeNull]` | Return, out, property | May be null even if T is non-nullable |
| `[AllowNull]` | Parameter, property | Accepts null even if T is non-nullable |
| `[DisallowNull]` | Parameter, property | Rejects null even if T is nullable |
| `[NotNullWhen(bool)]` | Parameter, out | Non-null when method returns the given bool |
| `[MaybeNullWhen(bool)]` | out | May be null when method returns the given bool |
| `[NotNullIfNotNull(param)]` | Return | Non-null if the named parameter is non-null |
| `[DoesNotReturn]` | Method | Method never returns (always throws) |
| `[DoesNotReturnIf(bool)]` | Parameter | Method doesn't return if parameter equals the bool |
| `[MemberNotNull(member)]` | Method | Named member is non-null after method returns |
| `[MemberNotNullWhen(bool, member)]` | Method | Named member is non-null when method returns the bool |

## 8. Migration Strategies

### 8.1 Incremental Adoption

```xml
<!-- Step 1: Start with warnings only (no code changes needed) -->
<Nullable>warnings</Nullable>

<!-- Step 2: Enable fully when ready -->
<Nullable>enable</Nullable>
```

### 8.2 File-by-File Migration

```csharp
// New files: fully nullable
#nullable enable

// Legacy files: opt out temporarily
#nullable disable

// Migrating: enable and fix warnings one file at a time
#nullable enable
// Fix all warnings in this file, then move to the next
```

### 8.3 Common Migration Patterns

```csharp
// Pattern 1: Late-initialized properties
// Before:
public string Name { get; set; } // Warning: non-nullable not initialized

// Option A: Initialize with default
public string Name { get; set; } = "";

// Option B: Make nullable if it can truly be null
public string? Name { get; set; }

// Option C: Use 'required' (C# 11) — must be set during initialization
public required string Name { get; set; }

// Option D: null! for DI/framework-initialized properties
public string Name { get; set; } = null!;

// Pattern 2: Dictionary TryGetValue
string? value;
if (dict.TryGetValue(key, out value))
{
    Console.WriteLine(value.Length); // Safe — TryGetValue has [NotNullWhen(true)]
}

// Pattern 3: LINQ FirstOrDefault
var item = list.FirstOrDefault(x => x.Id == targetId);
if (item is not null)
{
    Process(item);
}
// Or use First() if you know it exists
var item = list.First(x => x.Id == targetId);
```

## 9. Best Practices for Null Safety

### 9.1 Design Principles

```csharp
// 1. Prefer non-nullable types — make null the exception, not the norm
public class User
{
    public required string Name { get; init; }      // Always has a name
    public required string Email { get; init; }     // Always has an email
    public string? Bio { get; set; }                // Bio is optional
    public string? AvatarUrl { get; set; }          // Avatar is optional
}

// 2. Use the Null Object pattern instead of null
public interface ILogger { void Log(string message); }
public class NullLogger : ILogger { public void Log(string message) { } }

ILogger logger = GetLogger() ?? new NullLogger(); // Never null

// 3. Return empty collections instead of null
public IReadOnlyList<Order> GetOrders(int customerId)
{
    // GOOD: return empty list
    return _db.Orders.Where(o => o.CustomerId == customerId).ToList();
    // BAD: return null when no orders
}

// 4. Use result types for operations that can fail
public record Result<T>(T? Value, string? Error)
{
    public bool IsSuccess => Error is null;
    public static Result<T> Ok(T value) => new(value, null);
    public static Result<T> Fail(string error) => new(default, error);
}
```

### 9.2 Guard Clauses

```csharp
public class OrderService
{
    private readonly IRepository _repo;
    private readonly ILogger _logger;

    // Constructor guard clauses
    public OrderService(IRepository repo, ILogger logger)
    {
        _repo = repo ?? throw new ArgumentNullException(nameof(repo));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
    }

    // .NET 6+ — cleaner syntax
    public OrderService(IRepository repo, ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(repo);
        ArgumentNullException.ThrowIfNull(logger);
        _repo = repo;
        _logger = logger;
    }
}
```

## 10. Practical Example: Refactoring a Class to Be Null-Safe

### 10.1 Before — Null-Unsafe Code

```csharp
// No nullable context — bugs waiting to happen
public class CustomerProfile
{
    public string FirstName;
    public string LastName;
    public string Email;
    public string Phone;     // optional
    public Address Address;  // optional

    public string GetDisplayName()
    {
        return FirstName + " " + LastName; // NPE if either is null
    }

    public string GetCity()
    {
        return Address.City; // NPE if Address is null
    }
}

public class Address
{
    public string Street;
    public string City;
    public string State;
    public string ZipCode;
}
```

### 10.2 After — Null-Safe Code

```csharp
#nullable enable

public class CustomerProfile
{
    public required string FirstName { get; init; }
    public required string LastName { get; init; }
    public required string Email { get; init; }
    public string? Phone { get; set; }        // explicitly optional
    public Address? Address { get; set; }      // explicitly optional

    public string DisplayName => $"{FirstName} {LastName}"; // safe — both required

    public string GetCity() =>
        Address?.City ?? "Unknown"; // safe — handles null Address and null City

    public bool HasCompleteProfile =>
        Phone is not null &&
        Address is { Street: not null, City: not null, State: not null, ZipCode: not null };
}

public class Address
{
    public required string Street { get; init; }
    public required string City { get; init; }
    public required string State { get; init; }
    public required string ZipCode { get; init; }
}

// Usage — compiler enforces initialization
var customer = new CustomerProfile
{
    FirstName = "Alice",
    LastName = "Smith",
    Email = "alice@example.com",
    // Phone and Address are optional — no warning
};

Console.WriteLine(customer.DisplayName);          // Alice Smith
Console.WriteLine(customer.GetCity());            // Unknown
Console.WriteLine(customer.HasCompleteProfile);   // False

customer.Address = new Address
{
    Street = "123 Main St",
    City = "Seattle",
    State = "WA",
    ZipCode = "98101"
};

Console.WriteLine(customer.GetCity());            // Seattle
Console.WriteLine(customer.HasCompleteProfile);   // False (Phone still null)
```

## 11. Practice Problems

1. **Nullable Migration**: Take the following pre-NRT class and migrate it to be fully null-safe. Add `?` annotations, `required` keywords, guard clauses, and null-conditional operators as appropriate. Identify which fields should be nullable and which should not.
   ```csharp
   public class Order {
       public int Id;
       public Customer Customer;
       public List<OrderItem> Items;
       public string CouponCode;
       public string ShippingNotes;
       public decimal GetTotal() => Items.Sum(i => i.Price * i.Quantity);
   }
   ```

2. **Custom TryParse**: Implement a `DateRange` struct with `Start` and `End` properties. Write a `TryParse(string? input, out DateRange? result)` method that parses strings like `"2024-01-01..2024-12-31"`. Use `[NotNullWhen(true)]` correctly. Handle null input, empty strings, invalid formats, and ranges where Start > End.

3. **Null-Safe Builder Pattern**: Implement a `ConnectionStringBuilder` with a fluent API. Some properties are required (Server, Database), others are optional (Port, Username, Password). The `Build()` method should return `string` (non-nullable) and throw if required properties are missing. Use `[MemberNotNull]` where appropriate.

4. **Flow Analysis Explorer**: Write a series of methods that demonstrate what the compiler's flow analysis can and cannot track. Include at least 5 examples: (a) `if` null check, (b) `is not null` pattern, (c) `??=` assignment, (d) method call that loses tracking, (e) `[NotNullWhen]` restoration of tracking.

5. **Generic Null-Safe Collection**: Implement `NullSafeList<T>` that wraps `List<T?>` but provides methods that never return null: `GetOrDefault(int index, T defaultValue)`, `FirstOrDefault(T defaultValue)`, `FindOrDefault(Predicate<T> predicate, T defaultValue)`. All return `T`, not `T?`. Annotate all parameters and returns with correct nullability.
