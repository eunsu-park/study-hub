// Lesson 05: Nullable Reference Types
// Run: dotnet run
// Note: Ensure <Nullable>enable</Nullable> in your .csproj

#nullable enable

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;

// ============================================================
// 1. Nullable Annotations Basics
// ============================================================

Console.WriteLine("=== Nullable Annotations ===");

// Non-nullable: the compiler warns if null is assigned
string name = "Alice";
Console.WriteLine($"Non-nullable name: {name}");

// Nullable: explicitly allows null with the ? suffix
string? nickname = null;
Console.WriteLine($"Nullable nickname: {nickname ?? "(not set)"}");

// Assigning null to non-nullable triggers warning CS8600
// string bad = null; // Warning: Converting null literal

// ============================================================
// 2. Null-Conditional and Null-Coalescing Operators
// ============================================================

Console.WriteLine("\n=== Null Operators ===");

string? possiblyNull = GetNullableString(useNull: true);

// Null-conditional operator ?.
int? length = possiblyNull?.Length;
Console.WriteLine($"possiblyNull?.Length = {length?.ToString() ?? "null"}");

// Null-coalescing operator ??
string safe = possiblyNull ?? "default";
Console.WriteLine($"possiblyNull ?? \"default\" = {safe}");

// Null-coalescing assignment ??=
string? lazyValue = null;
lazyValue ??= "initialized";
Console.WriteLine($"After ??= : {lazyValue}");

// Chained null-conditional
var person = new PersonRecord("Alice", new AddressRecord("Seattle", null));
string? zip = person.Address?.ZipCode?.ToUpper();
Console.WriteLine($"Zip code: {zip ?? "(none)"}");

// ============================================================
// 3. Flow Analysis — Compiler Tracks Null State
// ============================================================

Console.WriteLine("\n=== Flow Analysis ===");

string? input = GetNullableString(useNull: false);

// Before null check, the compiler considers 'input' as possibly null
// input.ToUpper(); // Warning: Dereference of a possibly null reference

// After a null check, the compiler knows it is safe
if (input is not null)
{
    // No warning here — flow analysis proves input is non-null
    Console.WriteLine($"  input.ToUpper() = {input.ToUpper()}");
}

// Pattern matching also narrows nullability
if (input is string validInput)
{
    Console.WriteLine($"  Pattern matched: {validInput.Length} chars");
}

// Throwing on null also narrows
string guaranteed = input ?? throw new InvalidOperationException("Unexpected null");
Console.WriteLine($"  Guaranteed: {guaranteed}");

// ============================================================
// 4. Null-Forgiving Operator (!)
// ============================================================

Console.WriteLine("\n=== Null-Forgiving Operator ===");

// The ! operator suppresses nullable warnings — use sparingly
string? maybeNull = "actually not null";
string forced = maybeNull!; // "Trust me, compiler"
Console.WriteLine($"  Forced: {forced}");

// Useful when you know better than the compiler
// Example: dictionary lookup after ContainsKey
var dict = new Dictionary<string, string> { ["key"] = "value" };
if (dict.ContainsKey("key"))
{
    string val = dict["key"]!; // Safe because we checked
    Console.WriteLine($"  Dict value: {val}");
}

// ============================================================
// 5. Nullable in Collections and Generics
// ============================================================

Console.WriteLine("\n=== Nullable in Collections ===");

// List of non-nullable strings
List<string> names = new() { "Alice", "Bob", "Charlie" };

// List of nullable strings
List<string?> optionalNames = new() { "Alice", null, "Charlie" };

foreach (var n in optionalNames)
{
    // 'n' is string? here — must handle null
    Console.WriteLine($"  Name: {n?.ToUpper() ?? "(missing)"}");
}

// Generic constraint with notnull
var registry = new Registry<string>();
registry.Add("key1", "value1");
registry.Add("key2", "value2");
Console.WriteLine($"  Registry lookup: {registry.Get("key1")}");

// ============================================================
// 6. MemberNotNull and NotNullWhen Attributes
// ============================================================

Console.WriteLine("\n=== NotNullWhen / MemberNotNull ===");

var parser = new ConfigParser();

// TryParse pattern with NotNullWhen
if (parser.TryGetValue("timeout", out string? value))
{
    // Compiler knows 'value' is non-null here thanks to [NotNullWhen(true)]
    Console.WriteLine($"  Parsed value: {value.ToUpper()}");
}
else
{
    Console.WriteLine("  Key not found");
}

// ============================================================
// 7. Practical: Safe Navigation Pattern
// ============================================================

Console.WriteLine("\n=== Safe Navigation Pattern ===");

UserProfile? profile = LoadProfile(userId: 42);

// Chain of nullable navigation
string displayName = profile?.DisplayName
    ?? profile?.Email?.Split('@')[0]
    ?? "Anonymous";

Console.WriteLine($"  Display name: {displayName}");

// Guard clause pattern
void ProcessProfile(UserProfile? p)
{
    if (p is null)
    {
        Console.WriteLine("  Profile is null — skipping");
        return;
    }

    // After the null guard, 'p' is non-null for the rest of the method
    Console.WriteLine($"  Processing: {p.DisplayName ?? p.Email ?? "unknown"}");
}

ProcessProfile(profile);
ProcessProfile(null);

// ============================================================
// Helper Methods
// ============================================================

static string? GetNullableString(bool useNull)
    => useNull ? null : "hello world";

static UserProfile? LoadProfile(int userId)
    => userId == 42
        ? new UserProfile { Email = "alice@example.com", DisplayName = null }
        : null;

// ============================================================
// Supporting Types
// ============================================================

record PersonRecord(string Name, AddressRecord? Address);
record AddressRecord(string City, string? ZipCode);

class UserProfile
{
    public string? DisplayName { get; set; }
    public string? Email { get; set; }
}

// Generic class with notnull constraint
class Registry<TKey> where TKey : notnull
{
    private readonly Dictionary<TKey, string> _store = new();

    public void Add(TKey key, string value) => _store[key] = value;

    public string? Get(TKey key)
        => _store.TryGetValue(key, out var val) ? val : null;
}

// Class demonstrating NotNullWhen attribute
class ConfigParser
{
    private readonly Dictionary<string, string> _config = new()
    {
        ["timeout"] = "30",
        ["retries"] = "3",
    };

    // [NotNullWhen(true)] tells the compiler that 'value' is non-null when returning true
    public bool TryGetValue(string key, [NotNullWhen(true)] out string? value)
        => _config.TryGetValue(key, out value);
}
