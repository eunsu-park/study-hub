/*
 * Exercises for Lesson 05: Nullable Reference Types
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Null-safe string operations
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Null-Safe String Operations ===");

    string? name = "Alice";
    string? nullName = null;

    Console.WriteLine($"  Length of '{name}': {SafeLength(name)}");
    Console.WriteLine($"  Length of null: {SafeLength(nullName)}");
    Console.WriteLine($"  Upper of '{name}': {SafeUpper(name)}");
    Console.WriteLine($"  Upper of null: {SafeUpper(nullName)}");
    Console.WriteLine();
}

int SafeLength(string? input) => input?.Length ?? 0;
string SafeUpper(string? input) => input?.ToUpper() ?? "(null)";

// ---------------------------------------------------------------------------
// Exercise 2: Null-coalescing patterns for default values
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Null-Coalescing Defaults ===");

    var config = new Dictionary<string, string?>
    {
        ["host"] = "localhost",
        ["port"] = null,
        ["timeout"] = "30",
    };

    string host = config.GetValueOrDefault("host") ?? "127.0.0.1";
    string port = config.GetValueOrDefault("port") ?? "8080";
    string timeout = config.GetValueOrDefault("timeout") ?? "60";
    string retries = config.GetValueOrDefault("retries") ?? "3";

    Console.WriteLine($"  host={host}, port={port}, timeout={timeout}, retries={retries}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Refactor a class to use nullable annotations
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Nullable-Annotated Class ===");

    var customer1 = new Customer("Alice", "alice@example.com");
    var customer2 = new Customer("Bob", null);

    Console.WriteLine($"  {customer1.Name}: {customer1.GetContactEmail()}");
    Console.WriteLine($"  {customer2.Name}: {customer2.GetContactEmail()}");
    Console.WriteLine($"  Domain: {customer1.GetEmailDomain()}");
    Console.WriteLine($"  Domain: {customer2.GetEmailDomain()}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Null-safe chain — nested object navigation
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Null-Safe Chain ===");

    var order1 = new Order(
        new OrderCustomer("Alice", new Address("123 Main St", "Springfield")));
    var order2 = new Order(new OrderCustomer("Bob", null));
    var order3 = new Order(null);

    Console.WriteLine($"  Order1 city: {GetCity(order1)}");
    Console.WriteLine($"  Order2 city: {GetCity(order2)}");
    Console.WriteLine($"  Order3 city: {GetCity(order3)}");
    Console.WriteLine();
}

string GetCity(Order order) =>
    order.Customer?.HomeAddress?.City ?? "Unknown";

// ---------------------------------------------------------------------------
// Exercise 5: TryGet pattern — safe dictionary lookups
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: TryGet Pattern ===");

    var registry = new SafeRegistry<string>();
    registry.Set("name", "Alice");
    registry.Set("role", "Admin");

    Console.WriteLine($"  name  => {registry.GetOrDefault("name", "N/A")}");
    Console.WriteLine($"  role  => {registry.GetOrDefault("role", "N/A")}");
    Console.WriteLine($"  email => {registry.GetOrDefault("email", "N/A")}");

    if (registry.TryGet("name", out var value))
        Console.WriteLine($"  TryGet name: {value}");
    if (!registry.TryGet("missing", out _))
        Console.WriteLine($"  TryGet missing: not found");
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

class Customer
{
    public string Name { get; }
    public string? Email { get; }

    public Customer(string name, string? email)
    {
        Name = name;
        Email = email;
    }

    public string GetContactEmail() => Email ?? $"{Name.ToLower()}@default.com";

    public string? GetEmailDomain() => Email?.Split('@').LastOrDefault();
}

record Address(string Street, string City);
record OrderCustomer(string Name, Address? HomeAddress);
record Order(OrderCustomer? Customer);

class SafeRegistry<T> where T : class
{
    private readonly Dictionary<string, T> _store = new();

    public void Set(string key, T value) => _store[key] = value;

    public T GetOrDefault(string key, T defaultValue) =>
        _store.TryGetValue(key, out var val) ? val : defaultValue;

    public bool TryGet(string key, out T? value) =>
        _store.TryGetValue(key, out value);
}
