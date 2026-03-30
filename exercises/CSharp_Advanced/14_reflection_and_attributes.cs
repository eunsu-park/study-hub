/*
 * Exercises for Lesson 14: Reflection and Attributes
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

// ---------------------------------------------------------------------------
// Exercise 1: Type inspection — enumerate members of a type
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Type Inspection ===");

    Type type = typeof(SampleEntity);

    Console.WriteLine($"  Type: {type.FullName}");
    Console.WriteLine($"  Is class: {type.IsClass}");
    Console.WriteLine($"  Base type: {type.BaseType?.Name}");

    Console.WriteLine("  Properties:");
    foreach (var prop in type.GetProperties())
        Console.WriteLine($"    {prop.PropertyType.Name} {prop.Name} (get={prop.CanRead}, set={prop.CanWrite})");

    Console.WriteLine("  Methods (declared):");
    foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
        Console.WriteLine($"    {method.ReturnType.Name} {method.Name}({string.Join(", ", method.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"))})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Custom attribute — attribute-based validation
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Attribute-Based Validation ===");

    var validUser = new UserForm { Name = "Alice", Email = "alice@test.com", Age = 25 };
    var invalidUser = new UserForm { Name = "", Email = "bad", Age = -5 };

    var results1 = Validate(validUser);
    Console.WriteLine($"  Valid user errors: {(results1.Count == 0 ? "none" : string.Join("; ", results1))}");

    var results2 = Validate(invalidUser);
    Console.WriteLine($"  Invalid user errors:");
    foreach (var err in results2)
        Console.WriteLine($"    - {err}");
    Console.WriteLine();
}

List<string> Validate(object obj)
{
    var errors = new List<string>();
    var type = obj.GetType();

    foreach (var prop in type.GetProperties())
    {
        var value = prop.GetValue(obj);

        foreach (var attr in prop.GetCustomAttributes<ValidationAttribute>())
        {
            if (!attr.IsValid(value))
                errors.Add($"{prop.Name}: {attr.ErrorMessage}");
        }
    }
    return errors;
}

// ---------------------------------------------------------------------------
// Exercise 3: Dynamic method invocation
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Dynamic Method Invocation ===");

    var calculator = new MathOperations();
    Type calcType = calculator.GetType();

    string[] methodNames = { "Add", "Multiply", "Power" };
    (int a, int b)[] args = { (3, 4), (5, 6), (2, 10) };

    for (int i = 0; i < methodNames.Length; i++)
    {
        var method = calcType.GetMethod(methodNames[i]);
        if (method != null)
        {
            var result = method.Invoke(calculator, new object[] { args[i].a, args[i].b });
            Console.WriteLine($"  {methodNames[i]}({args[i].a}, {args[i].b}) = {result}");
        }
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Plugin discovery — find and instantiate types by interface
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Plugin Discovery ===");

    var pluginType = typeof(IPlugin);
    var assembly = Assembly.GetExecutingAssembly();

    var plugins = assembly.GetTypes()
        .Where(t => pluginType.IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract)
        .Select(t => (IPlugin)Activator.CreateInstance(t)!)
        .OrderBy(p => p.Name)
        .ToList();

    Console.WriteLine($"  Discovered {plugins.Count} plugins:");
    foreach (var plugin in plugins)
    {
        Console.WriteLine($"    {plugin.Name}: {plugin.Execute("test-input")}");
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Attribute-driven serializer — custom [Serialize] attribute
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Attribute-Driven Serializer ===");

    var config = new ServerConfig
    {
        Host = "localhost",
        Port = 8080,
        SecretKey = "my-secret",
        MaxConnections = 100,
        DebugMode = true
    };

    var serialized = AttributeSerializer.Serialize(config);
    Console.WriteLine("  Serialized (only [Serialize] fields):");
    foreach (var (key, value) in serialized)
        Console.WriteLine($"    {key} = {value}");
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

class SampleEntity
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public DateTime CreatedAt { get; set; }

    public string GetDisplayName() => $"[{Id}] {Name}";
    public bool IsRecent(int days) => (DateTime.Now - CreatedAt).TotalDays < days;
}

// ---- Validation attributes ----

[AttributeUsage(AttributeTargets.Property, AllowMultiple = true)]
abstract class ValidationAttribute : Attribute
{
    public string ErrorMessage { get; set; } = "Validation failed";
    public abstract bool IsValid(object? value);
}

class RequiredAttribute : ValidationAttribute
{
    public RequiredAttribute() { ErrorMessage = "is required"; }
    public override bool IsValid(object? value) =>
        value != null && (value is not string s || !string.IsNullOrWhiteSpace(s));
}

class RangeAttribute : ValidationAttribute
{
    public int Min { get; }
    public int Max { get; }
    public RangeAttribute(int min, int max) { Min = min; Max = max; ErrorMessage = $"must be between {min} and {max}"; }
    public override bool IsValid(object? value) =>
        value is int n && n >= Min && n <= Max;
}

class EmailAttribute : ValidationAttribute
{
    public EmailAttribute() { ErrorMessage = "must be a valid email"; }
    public override bool IsValid(object? value) =>
        value is string s && s.Contains('@') && s.Contains('.');
}

class UserForm
{
    [Required] public string Name { get; set; } = "";
    [Required][Email] public string Email { get; set; } = "";
    [Range(0, 150)] public int Age { get; set; }
}

// ---- Math operations ----

class MathOperations
{
    public int Add(int a, int b) => a + b;
    public int Multiply(int a, int b) => a * b;
    public double Power(int baseVal, int exp) => Math.Pow(baseVal, exp);
}

// ---- Plugin interface ----

interface IPlugin
{
    string Name { get; }
    string Execute(string input);
}

class UpperCasePlugin : IPlugin
{
    public string Name => "UpperCase";
    public string Execute(string input) => input.ToUpper();
}

class ReversePlugin : IPlugin
{
    public string Name => "Reverse";
    public string Execute(string input) => new string(input.Reverse().ToArray());
}

class Base64Plugin : IPlugin
{
    public string Name => "Base64Encode";
    public string Execute(string input) => Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(input));
}

// ---- Attribute-driven serializer ----

[AttributeUsage(AttributeTargets.Property)]
class SerializeAttribute : Attribute
{
    public string? Alias { get; set; }
}

class ServerConfig
{
    [Serialize] public string Host { get; set; } = "";
    [Serialize(Alias = "listen_port")] public int Port { get; set; }
    public string SecretKey { get; set; } = ""; // NOT serialized
    [Serialize] public int MaxConnections { get; set; }
    [Serialize(Alias = "debug")] public bool DebugMode { get; set; }
}

static class AttributeSerializer
{
    public static List<(string Key, string Value)> Serialize(object obj)
    {
        var result = new List<(string, string)>();
        foreach (var prop in obj.GetType().GetProperties())
        {
            var attr = prop.GetCustomAttribute<SerializeAttribute>();
            if (attr == null) continue;
            string key = attr.Alias ?? prop.Name;
            string value = prop.GetValue(obj)?.ToString() ?? "null";
            result.Add((key, value));
        }
        return result;
    }
}
