// Lesson 14: Reflection and Attributes
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;

// ============================================================
// 1. Custom Attributes
// ============================================================

Console.WriteLine("=== Custom Attributes ===");

// Read attribute from a class
var classAttr = typeof(UserController).GetCustomAttribute<RouteAttribute>();
Console.WriteLine($"UserController route: {classAttr?.Path}");

// Read attributes from methods
foreach (var method in typeof(UserController).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
{
    var httpAttr = method.GetCustomAttribute<HttpMethodAttribute>();
    var authAttr = method.GetCustomAttribute<AuthorizeAttribute>();

    if (httpAttr is not null)
    {
        string auth = authAttr is not null ? $" [Auth: {authAttr.Role}]" : "";
        Console.WriteLine($"  {httpAttr.Method,-6} {httpAttr.Path}{auth} -> {method.Name}()");
    }
}

// ============================================================
// 2. Reflection — Inspecting Types
// ============================================================

Console.WriteLine("\n=== Type Inspection ===");

Type type = typeof(SampleClass);
Console.WriteLine($"Type: {type.FullName}");
Console.WriteLine($"Is class: {type.IsClass}");
Console.WriteLine($"Is abstract: {type.IsAbstract}");
Console.WriteLine($"Base type: {type.BaseType?.Name}");
Console.WriteLine($"Interfaces: {string.Join(", ", type.GetInterfaces().Select(i => i.Name))}");

// Properties
Console.WriteLine("\nProperties:");
foreach (var prop in type.GetProperties())
{
    Console.WriteLine($"  {prop.PropertyType.Name} {prop.Name} (get={prop.CanRead}, set={prop.CanWrite})");
}

// Methods (excluding inherited Object methods)
Console.WriteLine("\nDeclared methods:");
foreach (var method in type.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly))
{
    var parameters = string.Join(", ", method.GetParameters().Select(p => $"{p.ParameterType.Name} {p.Name}"));
    Console.WriteLine($"  {method.ReturnType.Name} {method.Name}({parameters})");
}

// Fields
Console.WriteLine("\nFields (including private):");
foreach (var field in type.GetFields(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance))
{
    Console.WriteLine($"  {field.FieldType.Name} {field.Name} (public={field.IsPublic})");
}

// ============================================================
// 3. Dynamic Object Creation
// ============================================================

Console.WriteLine("\n=== Dynamic Object Creation ===");

// Create instance using Activator
object? obj = Activator.CreateInstance(typeof(SampleClass));
Console.WriteLine($"Created: {obj?.GetType().Name}");

// Create with constructor parameters
object? objWithArgs = Activator.CreateInstance(typeof(SampleClass), "dynamically created", 42);
Console.WriteLine($"Created with args: {objWithArgs}");

// Using ConstructorInfo
var ctor = typeof(SampleClass).GetConstructor(new[] { typeof(string), typeof(int) });
object? ctorObj = ctor?.Invoke(new object[] { "via ConstructorInfo", 99 });
Console.WriteLine($"Via ConstructorInfo: {ctorObj}");

// ============================================================
// 4. Dynamic Method Invocation
// ============================================================

Console.WriteLine("\n=== Dynamic Method Invocation ===");

var instance = new SampleClass("test", 10);

// Invoke a public method by name
MethodInfo? greetMethod = type.GetMethod("Greet");
object? result = greetMethod?.Invoke(instance, new object[] { "World" });
Console.WriteLine($"Greet result: {result}");

// Invoke a method with multiple parameters
MethodInfo? addMethod = type.GetMethod("Add");
object? sum = addMethod?.Invoke(instance, new object[] { 3, 4 });
Console.WriteLine($"Add result: {sum}");

// Set and get property values dynamically
PropertyInfo? nameProp = type.GetProperty("Name");
Console.WriteLine($"Before: Name = {nameProp?.GetValue(instance)}");
nameProp?.SetValue(instance, "modified");
Console.WriteLine($"After:  Name = {nameProp?.GetValue(instance)}");

// Access private field
FieldInfo? secretField = type.GetField("_secret", BindingFlags.NonPublic | BindingFlags.Instance);
Console.WriteLine($"Private _secret = {secretField?.GetValue(instance)}");
secretField?.SetValue(instance, "exposed!");
Console.WriteLine($"Modified _secret = {secretField?.GetValue(instance)}");

// ============================================================
// 5. Generic Type Reflection
// ============================================================

Console.WriteLine("\n=== Generic Type Reflection ===");

Type openGeneric = typeof(List<>);
Console.WriteLine($"Open generic: {openGeneric.Name}, IsGenericTypeDefinition={openGeneric.IsGenericTypeDefinition}");

Type closedGeneric = typeof(List<string>);
Console.WriteLine($"Closed generic: {closedGeneric.Name}, GenericArgs=[{string.Join(", ", closedGeneric.GetGenericArguments().Select(t => t.Name))}]");

// Create a closed generic type dynamically
Type constructed = openGeneric.MakeGenericType(typeof(int));
object? listInstance = Activator.CreateInstance(constructed);
MethodInfo? addItemMethod = constructed.GetMethod("Add");
addItemMethod?.Invoke(listInstance, new object[] { 42 });
addItemMethod?.Invoke(listInstance, new object[] { 99 });

PropertyInfo? countProp = constructed.GetProperty("Count");
Console.WriteLine($"Dynamic List<int> count: {countProp?.GetValue(listInstance)}");

// ============================================================
// 6. Attribute-Based Plugin Discovery
// ============================================================

Console.WriteLine("\n=== Plugin Discovery via Attributes ===");

// Find all types marked with [Plugin]
var pluginTypes = Assembly.GetExecutingAssembly()
    .GetTypes()
    .Where(t => t.GetCustomAttribute<PluginAttribute>() is not null && !t.IsAbstract);

Console.WriteLine("Discovered plugins:");
foreach (var pluginType in pluginTypes)
{
    var attr = pluginType.GetCustomAttribute<PluginAttribute>()!;
    Console.WriteLine($"  [{attr.Name} v{attr.Version}] {pluginType.Name}");

    // Instantiate and invoke
    if (Activator.CreateInstance(pluginType) is IPlugin plugin)
    {
        plugin.Execute();
    }
}

// ============================================================
// 7. Validation Framework Using Attributes
// ============================================================

Console.WriteLine("\n=== Attribute-Based Validation ===");

var validUser = new UserModel { Name = "Alice", Email = "alice@test.com", Age = 25 };
var invalidUser = new UserModel { Name = "", Email = "bad-email", Age = -5 };

Console.WriteLine($"Valid user:   {Validator.Validate(validUser)}");
Console.WriteLine($"Invalid user: {Validator.Validate(invalidUser)}");

var errors = Validator.GetErrors(invalidUser);
foreach (var error in errors)
    Console.WriteLine($"  Error: {error}");

// ============================================================
// Custom Attributes
// ============================================================

[AttributeUsage(AttributeTargets.Class)]
public class RouteAttribute : Attribute
{
    public string Path { get; }
    public RouteAttribute(string path) => Path = path;
}

[AttributeUsage(AttributeTargets.Method)]
public class HttpMethodAttribute : Attribute
{
    public string Method { get; }
    public string Path { get; }
    public HttpMethodAttribute(string method, string path) { Method = method; Path = path; }
}

[AttributeUsage(AttributeTargets.Method)]
public class AuthorizeAttribute : Attribute
{
    public string Role { get; }
    public AuthorizeAttribute(string role = "User") => Role = role;
}

[AttributeUsage(AttributeTargets.Class)]
public class PluginAttribute : Attribute
{
    public string Name { get; }
    public string Version { get; }
    public PluginAttribute(string name, string version) { Name = name; Version = version; }
}

// Validation attributes
[AttributeUsage(AttributeTargets.Property)]
public class RequiredAttribute : Attribute { }

[AttributeUsage(AttributeTargets.Property)]
public class RangeAttribute : Attribute
{
    public int Min { get; }
    public int Max { get; }
    public RangeAttribute(int min, int max) { Min = min; Max = max; }
}

[AttributeUsage(AttributeTargets.Property)]
public class EmailAttribute : Attribute { }

// ============================================================
// Sample Types
// ============================================================

[Route("/api/users")]
public class UserController
{
    [HttpMethod("GET", "/")]
    public string GetAll() => "all users";

    [HttpMethod("GET", "/{id}")]
    public string GetById(int id) => $"user {id}";

    [HttpMethod("POST", "/")]
    [Authorize("Admin")]
    public string Create() => "created";

    [HttpMethod("DELETE", "/{id}")]
    [Authorize("Admin")]
    public string Delete(int id) => $"deleted {id}";
}

public interface IGreetable { string Greet(string name); }

public class SampleClass : IGreetable
{
    private string _secret = "hidden";

    public string Name { get; set; }
    public int Value { get; }

    public SampleClass() { Name = "default"; Value = 0; }
    public SampleClass(string name, int value) { Name = name; Value = value; }

    public string Greet(string name) => $"Hello, {name}! I am {Name}.";
    public int Add(int a, int b) => a + b;
    public override string ToString() => $"SampleClass(Name={Name}, Value={Value})";
}

// Plugins
public interface IPlugin { void Execute(); }

[Plugin("Logger", "1.0")]
public class LoggerPlugin : IPlugin
{
    public void Execute() => Console.WriteLine("    LoggerPlugin executed.");
}

[Plugin("Metrics", "2.1")]
public class MetricsPlugin : IPlugin
{
    public void Execute() => Console.WriteLine("    MetricsPlugin executed.");
}

// Validation model
public class UserModel
{
    [Required]
    public string Name { get; set; } = "";

    [Required]
    [Email]
    public string Email { get; set; } = "";

    [Range(0, 150)]
    public int Age { get; set; }
}

// Simple validator using reflection
public static class Validator
{
    public static bool Validate(object obj) => GetErrors(obj).Count == 0;

    public static List<string> GetErrors(object obj)
    {
        var errors = new List<string>();
        foreach (var prop in obj.GetType().GetProperties())
        {
            var value = prop.GetValue(obj);

            if (prop.GetCustomAttribute<RequiredAttribute>() is not null)
            {
                if (value is null || (value is string s && string.IsNullOrWhiteSpace(s)))
                    errors.Add($"{prop.Name} is required");
            }

            if (prop.GetCustomAttribute<RangeAttribute>() is RangeAttribute range && value is int intVal)
            {
                if (intVal < range.Min || intVal > range.Max)
                    errors.Add($"{prop.Name} must be between {range.Min} and {range.Max}");
            }

            if (prop.GetCustomAttribute<EmailAttribute>() is not null && value is string email)
            {
                if (!email.Contains('@') || !email.Contains('.'))
                    errors.Add($"{prop.Name} must be a valid email");
            }
        }
        return errors;
    }
}
