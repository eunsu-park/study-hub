# Reflection and Attributes

**Previous**: [NuGet and Project System](./13_NuGet_and_Project_System.md) | **Next**: [Interop and Unsafe Code](./15_Interop_and_Unsafe.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Apply built-in attributes to control compiler behavior and metadata
2. Design and implement custom attributes with parameters and targets
3. Use the `Type` class to inspect type information at runtime
4. Enumerate properties, methods, fields, and constructors via reflection
5. Invoke methods and create instances dynamically
6. Discover attributes applied to types and members at runtime
7. Load and inspect assemblies programmatically
8. Understand the performance implications of reflection and when to avoid it
9. Recognize source generators as a compile-time alternative to reflection

---

Reflection is the ability of a program to examine and manipulate its own structure at runtime. Combined with attributes — declarative metadata tags — reflection enables powerful patterns like serialization frameworks, dependency injection containers, test discovery, ORM mapping, and validation systems. This lesson teaches you to read type metadata, invoke members dynamically, build custom attributes, and understand when reflection is the right tool versus when compile-time alternatives are better.

## 1. Built-in Attributes

### 1.1 The [Obsolete] Attribute

The `[Obsolete]` attribute marks members that should no longer be used, generating compiler warnings or errors:

```csharp
public class LegacyService
{
    // Generates a compiler warning
    [Obsolete("Use GetDataAsync instead.")]
    public string GetData() => "old data";

    // Generates a compiler ERROR (second parameter = true)
    [Obsolete("This method will be removed in v3.0. Use ProcessAsync.", true)]
    public void Process() { }

    public Task<string> GetDataAsync() => Task.FromResult("new data");
    public Task ProcessAsync() => Task.CompletedTask;
}

// Usage:
var svc = new LegacyService();
// svc.GetData();   // Warning CS0618: 'GetData' is obsolete
// svc.Process();   // Error CS0619: 'Process' is obsolete
```

### 1.2 The [Conditional] Attribute

`[Conditional]` causes the compiler to omit calls to the method unless a specified symbol is defined:

```csharp
using System.Diagnostics;

public static class Logger
{
    [Conditional("DEBUG")]
    public static void DebugLog(string message)
    {
        Console.WriteLine($"[DEBUG {DateTime.Now:HH:mm:ss}] {message}");
    }

    [Conditional("TRACE")]
    public static void TraceLog(string message)
    {
        Console.WriteLine($"[TRACE {DateTime.Now:HH:mm:ss}] {message}");
    }
}

// In Debug builds (DEBUG symbol defined):
Logger.DebugLog("Starting operation");  // This call is compiled in

// In Release builds (DEBUG symbol NOT defined):
Logger.DebugLog("Starting operation");  // This call is completely removed by the compiler
```

### 1.3 Serialization Attributes

```csharp
using System.Text.Json.Serialization;

public class UserDto
{
    [JsonPropertyName("user_name")]
    public required string UserName { get; set; }

    [JsonPropertyName("email_address")]
    public required string Email { get; set; }

    [JsonIgnore]
    public string? InternalToken { get; set; }

    [JsonConverter(typeof(JsonStringEnumConverter))]
    public UserRole Role { get; set; }

    [JsonPropertyOrder(1)]
    public int Id { get; set; }
}

public enum UserRole { Admin, User, Guest }

// Serialization:
var user = new UserDto { UserName = "alice", Email = "alice@example.com", Role = UserRole.Admin, Id = 1 };
string json = JsonSerializer.Serialize(user, new JsonSerializerOptions { WriteIndented = true });
// Output:
// {
//   "id": 1,
//   "user_name": "alice",
//   "email_address": "alice@example.com",
//   "role": "Admin"
// }
// Note: InternalToken is excluded due to [JsonIgnore]
```

### 1.4 Other Common Built-in Attributes

```csharp
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

public class DemoAttributes
{
    // Compiler-generated caller info
    public void Log(
        string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        Console.WriteLine($"{filePath}:{lineNumber} [{memberName}] {message}");
    }

    // Data annotations for validation
    [Required]
    [StringLength(100, MinimumLength = 2)]
    public string Name { get; set; } = "";

    [Range(0, 150)]
    public int Age { get; set; }

    [EmailAddress]
    public string Email { get; set; } = "";

    // Description for tooling and documentation
    [Description("The maximum number of retry attempts")]
    [DefaultValue(3)]
    public int MaxRetries { get; set; } = 3;
}
```

## 2. Creating Custom Attributes

### 2.1 Basic Custom Attribute

Custom attributes are classes that inherit from `System.Attribute`:

```csharp
// Define a custom attribute
[AttributeUsage(AttributeTargets.Class | AttributeTargets.Method)]
public class AuthorAttribute : Attribute
{
    public string Name { get; }
    public string? Email { get; set; }
    public double Version { get; set; } = 1.0;

    public AuthorAttribute(string name)
    {
        Name = name;
    }
}

// Apply the custom attribute
[Author("Alice Smith", Email = "alice@example.com", Version = 2.1)]
public class PaymentProcessor
{
    [Author("Bob Jones")]
    public void ProcessPayment(decimal amount)
    {
        // ...
    }
}
```

### 2.2 AttributeUsage Options

```csharp
// AllowMultiple: Can the attribute be applied more than once?
[AttributeUsage(
    AttributeTargets.Class | AttributeTargets.Struct,  // Valid targets
    AllowMultiple = true,                               // Can apply multiple times
    Inherited = false                                   // Not inherited by derived classes
)]
public class TagAttribute : Attribute
{
    public string Value { get; }
    public TagAttribute(string value) => Value = value;
}

// Multiple applications allowed:
[Tag("serializable")]
[Tag("auditable")]
[Tag("cacheable")]
public class Order
{
    public int Id { get; set; }
    public decimal Total { get; set; }
}
```

### 2.3 AttributeTargets Enumeration

```csharp
// AttributeTargets values:
// Assembly, Module, Class, Struct, Enum, Constructor, Method,
// Property, Field, Event, Interface, Parameter, Delegate,
// ReturnValue, GenericParameter, All

// Target-specific examples:
[AttributeUsage(AttributeTargets.Property)]
public class ColumnNameAttribute : Attribute
{
    public string Name { get; }
    public ColumnNameAttribute(string name) => Name = name;
}

[AttributeUsage(AttributeTargets.Parameter)]
public class NotNullAttribute : Attribute { }

[AttributeUsage(AttributeTargets.ReturnValue)]
public class MustDisposeAttribute : Attribute { }

// Usage:
public class Customer
{
    [ColumnName("customer_name")]
    public string Name { get; set; } = "";

    public void Save([NotNull] string connectionString) { }
}
```

## 3. Reflection Basics

### 3.1 Getting Type Information

The `System.Type` class is the gateway to all reflection operations:

```csharp
using System;

// Three ways to get a Type object:

// 1. typeof() operator (compile-time, no instance needed)
Type stringType = typeof(string);
Console.WriteLine(stringType.FullName);  // System.String

// 2. GetType() instance method (runtime)
string greeting = "Hello";
Type greetingType = greeting.GetType();
Console.WriteLine(greetingType.Name);  // String

// 3. Type.GetType() static method (by string name)
Type? intType = Type.GetType("System.Int32");
Console.WriteLine(intType?.Name);  // Int32

// Type comparison
Console.WriteLine(stringType == greetingType);  // True
Console.WriteLine(greeting is string);          // True (preferred pattern)
```

### 3.2 Examining Type Properties

```csharp
Type type = typeof(List<int>);

Console.WriteLine($"Name:            {type.Name}");            // List`1
Console.WriteLine($"FullName:        {type.FullName}");        // System.Collections.Generic.List`1[[System.Int32, ...]]
Console.WriteLine($"Namespace:       {type.Namespace}");       // System.Collections.Generic
Console.WriteLine($"Assembly:        {type.Assembly.GetName().Name}");  // System.Private.CoreLib
Console.WriteLine($"IsClass:         {type.IsClass}");         // True
Console.WriteLine($"IsAbstract:      {type.IsAbstract}");      // False
Console.WriteLine($"IsSealed:        {type.IsSealed}");        // False
Console.WriteLine($"IsGenericType:   {type.IsGenericType}");   // True
Console.WriteLine($"IsValueType:     {type.IsValueType}");     // False
Console.WriteLine($"BaseType:        {type.BaseType?.Name}");  // Object

// Generic type arguments
Type[] genericArgs = type.GetGenericArguments();
foreach (var arg in genericArgs)
    Console.WriteLine($"Generic arg: {arg.Name}");  // Int32

// Interfaces implemented
Type[] interfaces = type.GetInterfaces();
foreach (var iface in interfaces)
    Console.WriteLine($"Implements: {iface.Name}");
// IList`1, ICollection`1, IEnumerable`1, IEnumerable, IReadOnlyList`1, ...
```

## 4. Inspecting Types

### 4.1 Getting Properties

```csharp
public class Person
{
    public int Id { get; set; }
    public string FirstName { get; set; } = "";
    public string LastName { get; set; } = "";
    public DateTime BirthDate { get; private set; }
    internal string? Nickname { get; set; }

    public string FullName => $"{FirstName} {LastName}";
}

Type personType = typeof(Person);

// Get all public instance properties
var publicProps = personType.GetProperties(BindingFlags.Public | BindingFlags.Instance);
Console.WriteLine("Public Properties:");
foreach (var prop in publicProps)
{
    string accessors = "";
    if (prop.CanRead) accessors += "get; ";
    if (prop.CanWrite) accessors += "set; ";
    Console.WriteLine($"  {prop.PropertyType.Name} {prop.Name} {{ {accessors}}}");
}
// Output:
//   Int32 Id { get; set; }
//   String FirstName { get; set; }
//   String LastName { get; set; }
//   DateTime BirthDate { get; }       (set is private, so CanWrite is still true)
//   String FullName { get; }

// Get non-public properties too
var allProps = personType.GetProperties(
    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
Console.WriteLine($"\nAll properties count: {allProps.Length}");  // includes Nickname
```

### 4.2 Getting Methods

```csharp
public class Calculator
{
    public int Add(int a, int b) => a + b;
    public double Add(double a, double b) => a + b;
    public static int Multiply(int a, int b) => a * b;
    private int Secret() => 42;
}

Type calcType = typeof(Calculator);

// All public methods (includes inherited from Object)
var methods = calcType.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
foreach (var method in methods)
{
    var parameters = method.GetParameters();
    string paramList = string.Join(", ",
        parameters.Select(p => $"{p.ParameterType.Name} {p.Name}"));
    Console.WriteLine($"  {method.ReturnType.Name} {method.Name}({paramList})");
}
// Output:
//   Int32 Add(Int32 a, Int32 b)
//   Double Add(Double a, Double b)

// Get static methods
var staticMethods = calcType.GetMethods(BindingFlags.Public | BindingFlags.Static);
foreach (var m in staticMethods)
    Console.WriteLine($"  static {m.Name}");  // Multiply

// Get private methods
var privateMethods = calcType.GetMethods(BindingFlags.NonPublic | BindingFlags.Instance);
foreach (var m in privateMethods)
    Console.WriteLine($"  private {m.Name}");  // Secret (plus some inherited)
```

### 4.3 Getting Fields and Constructors

```csharp
public class DatabaseConnection
{
    private readonly string _connectionString;
    private int _timeout = 30;
    public static int PoolSize = 10;

    public DatabaseConnection(string connectionString)
    {
        _connectionString = connectionString;
    }

    public DatabaseConnection(string connectionString, int timeout)
    {
        _connectionString = connectionString;
        _timeout = timeout;
    }
}

Type dbType = typeof(DatabaseConnection);

// Fields
Console.WriteLine("Fields:");
var fields = dbType.GetFields(BindingFlags.Public | BindingFlags.NonPublic |
                               BindingFlags.Instance | BindingFlags.Static);
foreach (var field in fields)
{
    string access = field.IsPublic ? "public" : field.IsPrivate ? "private" : "other";
    string modifier = field.IsStatic ? " static" : "";
    string readOnly = field.IsInitOnly ? " readonly" : "";
    Console.WriteLine($"  {access}{modifier}{readOnly} {field.FieldType.Name} {field.Name}");
}
// private readonly String _connectionString
// private Int32 _timeout
// public static Int32 PoolSize

// Constructors
Console.WriteLine("\nConstructors:");
var ctors = dbType.GetConstructors();
foreach (var ctor in ctors)
{
    var parameters = ctor.GetParameters();
    string paramList = string.Join(", ",
        parameters.Select(p => $"{p.ParameterType.Name} {p.Name}"));
    Console.WriteLine($"  {dbType.Name}({paramList})");
}
// DatabaseConnection(String connectionString)
// DatabaseConnection(String connectionString, Int32 timeout)
```

## 5. Invoking Methods Dynamically

### 5.1 Calling Instance Methods

```csharp
public class Greeter
{
    public string SayHello(string name) => $"Hello, {name}!";
    public string SayHello(string name, string language)
    {
        return language switch
        {
            "es" => $"Hola, {name}!",
            "de" => $"Hallo, {name}!",
            "ja" => $"こんにちは、{name}!",
            _ => $"Hello, {name}!"
        };
    }
    private int GetMagicNumber() => 42;
}

// Create instance and invoke methods via reflection
var greeter = new Greeter();
Type greeterType = typeof(Greeter);

// Invoke public method with one parameter
MethodInfo? sayHello1 = greeterType.GetMethod("SayHello", new[] { typeof(string) });
object? result1 = sayHello1?.Invoke(greeter, new object[] { "Alice" });
Console.WriteLine(result1);  // Hello, Alice!

// Invoke overloaded method with two parameters
MethodInfo? sayHello2 = greeterType.GetMethod("SayHello", new[] { typeof(string), typeof(string) });
object? result2 = sayHello2?.Invoke(greeter, new object[] { "Alice", "es" });
Console.WriteLine(result2);  // Hola, Alice!

// Invoke private method
MethodInfo? secret = greeterType.GetMethod("GetMagicNumber",
    BindingFlags.NonPublic | BindingFlags.Instance);
object? result3 = secret?.Invoke(greeter, null);
Console.WriteLine(result3);  // 42
```

### 5.2 Calling Static Methods

```csharp
public static class MathHelper
{
    public static double Hypotenuse(double a, double b) => Math.Sqrt(a * a + b * b);
}

Type mathType = typeof(MathHelper);
MethodInfo? hypMethod = mathType.GetMethod("Hypotenuse");

// For static methods, the first argument to Invoke is null
object? result = hypMethod?.Invoke(null, new object[] { 3.0, 4.0 });
Console.WriteLine(result);  // 5
```

### 5.3 Setting and Getting Property Values

```csharp
public class Config
{
    public string AppName { get; set; } = "Default";
    public int MaxRetries { get; set; } = 3;
    public bool Verbose { get; set; }
}

var config = new Config();
Type configType = typeof(Config);

// Set a property value
PropertyInfo? appNameProp = configType.GetProperty("AppName");
appNameProp?.SetValue(config, "MyApplication");

PropertyInfo? retriesProp = configType.GetProperty("MaxRetries");
retriesProp?.SetValue(config, 5);

// Get a property value
object? name = appNameProp?.GetValue(config);
Console.WriteLine(name);  // MyApplication

// Dynamically set properties from a dictionary
var settings = new Dictionary<string, object>
{
    ["AppName"] = "ProductionApp",
    ["MaxRetries"] = 10,
    ["Verbose"] = true
};

foreach (var (key, value) in settings)
{
    PropertyInfo? prop = configType.GetProperty(key);
    if (prop != null && prop.CanWrite)
    {
        object converted = Convert.ChangeType(value, prop.PropertyType);
        prop.SetValue(config, converted);
    }
}

Console.WriteLine($"{config.AppName}, Retries={config.MaxRetries}, Verbose={config.Verbose}");
// ProductionApp, Retries=10, Verbose=True
```

## 6. Creating Instances with Activator

### 6.1 Basic Instance Creation

```csharp
public class Logger
{
    public string Name { get; }
    public LogLevel Level { get; }

    public Logger() : this("Default", LogLevel.Info) { }
    public Logger(string name) : this(name, LogLevel.Info) { }
    public Logger(string name, LogLevel level)
    {
        Name = name;
        Level = level;
    }

    public void Log(string message) => Console.WriteLine($"[{Level}] {Name}: {message}");
}

public enum LogLevel { Debug, Info, Warning, Error }

// Create with parameterless constructor
object? logger1 = Activator.CreateInstance(typeof(Logger));
((Logger)logger1!).Log("Hello");  // [Info] Default: Hello

// Create with specific constructor parameters
object? logger2 = Activator.CreateInstance(typeof(Logger), "AppLogger", LogLevel.Debug);
((Logger)logger2!).Log("Debug info");  // [Debug] AppLogger: Debug info

// Create from type name string
Type? loggerType = Type.GetType("MyNamespace.Logger");
if (loggerType != null)
{
    object? logger3 = Activator.CreateInstance(loggerType, "DynamicLogger");
}
```

### 6.2 Generic Instance Creation

```csharp
public class Repository<T> where T : class, new()
{
    private readonly List<T> _items = new();

    public void Add(T item) => _items.Add(item);
    public IReadOnlyList<T> GetAll() => _items.AsReadOnly();
    public override string ToString() => $"Repository<{typeof(T).Name}> with {_items.Count} items";
}

public class Product
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
}

// Create a generic type at runtime
Type openGenericType = typeof(Repository<>);  // Open generic type
Type closedGenericType = openGenericType.MakeGenericType(typeof(Product));  // Repository<Product>

object? repo = Activator.CreateInstance(closedGenericType);

// Invoke methods on the generic instance
MethodInfo? addMethod = closedGenericType.GetMethod("Add");
var product = new Product { Id = 1, Name = "Widget" };
addMethod?.Invoke(repo, new object[] { product });

Console.WriteLine(repo);  // Repository<Product> with 1 items
```

### 6.3 Factory Pattern with Activator

```csharp
public interface IMessageHandler
{
    void Handle(string message);
}

public class EmailHandler : IMessageHandler
{
    public void Handle(string message) => Console.WriteLine($"Email: {message}");
}

public class SmsHandler : IMessageHandler
{
    public void Handle(string message) => Console.WriteLine($"SMS: {message}");
}

public class SlackHandler : IMessageHandler
{
    public void Handle(string message) => Console.WriteLine($"Slack: {message}");
}

// Dynamic factory using reflection
public static class HandlerFactory
{
    private static readonly Dictionary<string, Type> _handlers = new();

    public static void Register(string name, Type type)
    {
        if (!typeof(IMessageHandler).IsAssignableFrom(type))
            throw new ArgumentException($"{type.Name} does not implement IMessageHandler");
        _handlers[name] = type;
    }

    public static IMessageHandler Create(string name)
    {
        if (!_handlers.TryGetValue(name, out Type? type))
            throw new KeyNotFoundException($"No handler registered for '{name}'");
        return (IMessageHandler)Activator.CreateInstance(type)!;
    }
}

// Usage:
HandlerFactory.Register("email", typeof(EmailHandler));
HandlerFactory.Register("sms", typeof(SmsHandler));
HandlerFactory.Register("slack", typeof(SlackHandler));

IMessageHandler handler = HandlerFactory.Create("slack");
handler.Handle("Build succeeded");  // Slack: Build succeeded
```

## 7. Attribute Discovery at Runtime

### 7.1 Reading Attributes from Types

```csharp
[AttributeUsage(AttributeTargets.Class, AllowMultiple = false)]
public class ApiEndpointAttribute : Attribute
{
    public string Route { get; }
    public string Description { get; set; } = "";
    public ApiEndpointAttribute(string route) => Route = route;
}

[ApiEndpoint("/api/users", Description = "User management")]
public class UserController { }

[ApiEndpoint("/api/orders", Description = "Order management")]
public class OrderController { }

// Discover all controllers with [ApiEndpoint]
var controllerTypes = typeof(Program).Assembly.GetTypes()
    .Where(t => t.GetCustomAttribute<ApiEndpointAttribute>() != null);

foreach (var type in controllerTypes)
{
    var attr = type.GetCustomAttribute<ApiEndpointAttribute>()!;
    Console.WriteLine($"  {type.Name}: {attr.Route} - {attr.Description}");
}
// UserController: /api/users - User management
// OrderController: /api/orders - Order management
```

### 7.2 Reading Attributes from Members

```csharp
[AttributeUsage(AttributeTargets.Property)]
public class DisplayNameAttribute : Attribute
{
    public string Name { get; }
    public DisplayNameAttribute(string name) => Name = name;
}

[AttributeUsage(AttributeTargets.Property)]
public class MaxLengthAttribute : Attribute
{
    public int Length { get; }
    public MaxLengthAttribute(int length) => Length = length;
}

public class Employee
{
    [DisplayName("Employee ID")]
    public int Id { get; set; }

    [DisplayName("Full Name")]
    [MaxLength(100)]
    public string Name { get; set; } = "";

    [DisplayName("Department")]
    [MaxLength(50)]
    public string Department { get; set; } = "";
}

// Print a formatted header using display names
Type empType = typeof(Employee);
var properties = empType.GetProperties();

foreach (var prop in properties)
{
    var displayAttr = prop.GetCustomAttribute<DisplayNameAttribute>();
    var maxLenAttr = prop.GetCustomAttribute<MaxLengthAttribute>();

    string displayName = displayAttr?.Name ?? prop.Name;
    string constraint = maxLenAttr != null ? $" (max {maxLenAttr.Length} chars)" : "";
    Console.WriteLine($"  {displayName}: {prop.PropertyType.Name}{constraint}");
}
// Employee ID: Int32
// Full Name: String (max 100 chars)
// Department: String (max 50 chars)
```

### 7.3 Checking for Attribute Presence

```csharp
// IsDefined is faster when you only need to check existence
bool hasObsolete = typeof(LegacyService).IsDefined(typeof(ObsoleteAttribute), false);

// GetCustomAttributes returns all attributes
var allAttributes = typeof(Order).GetCustomAttributes(true);
foreach (var attr in allAttributes)
    Console.WriteLine($"  {attr.GetType().Name}");

// GetCustomAttributes<T> with generic type filter
var tags = typeof(Order).GetCustomAttributes<TagAttribute>();
foreach (var tag in tags)
    Console.WriteLine($"  Tag: {tag.Value}");
```

## 8. Assembly Inspection

### 8.1 Loading and Examining Assemblies

```csharp
using System.Reflection;

// Get the currently executing assembly
Assembly currentAssembly = Assembly.GetExecutingAssembly();
Console.WriteLine($"Name: {currentAssembly.GetName().Name}");
Console.WriteLine($"Version: {currentAssembly.GetName().Version}");
Console.WriteLine($"Location: {currentAssembly.Location}");

// Get all types in the assembly
Type[] allTypes = currentAssembly.GetTypes();
Console.WriteLine($"Total types: {allTypes.Length}");

// Find all public classes
var publicClasses = allTypes.Where(t => t.IsClass && t.IsPublic);
foreach (var cls in publicClasses)
    Console.WriteLine($"  Class: {cls.FullName}");

// Find types implementing a specific interface
var handlers = allTypes
    .Where(t => typeof(IMessageHandler).IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract);
foreach (var handler in handlers)
    Console.WriteLine($"  Handler: {handler.Name}");
```

### 8.2 Loading External Assemblies

```csharp
// Load by name (from the application's probe path)
Assembly? byName = Assembly.Load("System.Text.Json");

// Load from a specific file path
Assembly fromFile = Assembly.LoadFrom("/path/to/MyPlugin.dll");

// Examine exported types from an external assembly
var exportedTypes = fromFile.GetExportedTypes();
foreach (var type in exportedTypes)
{
    Console.WriteLine($"  {type.FullName}");
}

// Find a specific type
Type? pluginType = fromFile.GetType("MyPlugin.DataProcessor");
if (pluginType != null)
{
    object? instance = Activator.CreateInstance(pluginType);
    MethodInfo? runMethod = pluginType.GetMethod("Run");
    runMethod?.Invoke(instance, null);
}
```

### 8.3 Assembly Metadata

```csharp
Assembly asm = Assembly.GetExecutingAssembly();

// Read assembly-level attributes
var title = asm.GetCustomAttribute<AssemblyTitleAttribute>()?.Title;
var company = asm.GetCustomAttribute<AssemblyCompanyAttribute>()?.Company;
var version = asm.GetCustomAttribute<AssemblyFileVersionAttribute>()?.Version;
var informational = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

Console.WriteLine($"Title: {title}");
Console.WriteLine($"Company: {company}");
Console.WriteLine($"File Version: {version}");
Console.WriteLine($"Informational Version: {informational}");

// Read custom metadata (added by build system)
var metadata = asm.GetCustomAttributes<AssemblyMetadataAttribute>();
foreach (var m in metadata)
    Console.WriteLine($"  {m.Key} = {m.Value}");
// Example output: BuildTimestamp = 2025-01-15T10:30:00Z
```

## 9. Performance Cost of Reflection

### 9.1 Benchmarking Reflection vs Direct Calls

```csharp
using System.Diagnostics;
using System.Reflection;

public class MathService
{
    public int Square(int x) => x * x;
}

// Benchmark: Direct call vs Reflection
var service = new MathService();
Type type = typeof(MathService);
MethodInfo method = type.GetMethod("Square")!;
const int iterations = 1_000_000;

// Direct call
var sw = Stopwatch.StartNew();
for (int i = 0; i < iterations; i++)
{
    _ = service.Square(42);
}
sw.Stop();
Console.WriteLine($"Direct call:      {sw.ElapsedMilliseconds} ms");

// Reflection invoke
sw.Restart();
for (int i = 0; i < iterations; i++)
{
    _ = method.Invoke(service, new object[] { 42 });
}
sw.Stop();
Console.WriteLine($"Reflection invoke: {sw.ElapsedMilliseconds} ms");

// Typical results:
// Direct call:       ~2 ms
// Reflection invoke: ~500 ms
// Reflection is ~100-250x slower
```

### 9.2 Caching Reflection Results

```csharp
// WRONG: Looking up the method every time
public static object? SlowInvoke(object target, string methodName, params object[] args)
{
    // GetMethod is called on every invocation - expensive!
    MethodInfo? method = target.GetType().GetMethod(methodName);
    return method?.Invoke(target, args);
}

// BETTER: Cache MethodInfo objects
public class ReflectionCache
{
    private static readonly Dictionary<(Type, string), MethodInfo> _methodCache = new();

    public static MethodInfo? GetMethod(Type type, string methodName)
    {
        var key = (type, methodName);
        if (!_methodCache.TryGetValue(key, out var method))
        {
            method = type.GetMethod(methodName);
            if (method != null)
                _methodCache[key] = method;
        }
        return method;
    }
}

// BEST: Compile to delegate for repeated calls
public static class DelegateCache
{
    public static Func<object, object[], object?> CreateInvoker(MethodInfo method)
    {
        // Cache the delegate - subsequent calls avoid reflection overhead
        return (target, args) => method.Invoke(target, args);
    }
}
```

### 9.3 Compiled Expressions for Maximum Performance

```csharp
using System.Linq.Expressions;
using System.Reflection;

public static class FastPropertyAccess
{
    public static Func<T, TProperty> CreateGetter<T, TProperty>(string propertyName)
    {
        var parameter = Expression.Parameter(typeof(T), "obj");
        var property = Expression.Property(parameter, propertyName);
        var lambda = Expression.Lambda<Func<T, TProperty>>(property, parameter);
        return lambda.Compile();  // Compiles to IL - nearly as fast as direct access
    }

    public static Action<T, TProperty> CreateSetter<T, TProperty>(string propertyName)
    {
        var objParam = Expression.Parameter(typeof(T), "obj");
        var valueParam = Expression.Parameter(typeof(TProperty), "value");
        var property = Expression.Property(objParam, propertyName);
        var assign = Expression.Assign(property, valueParam);
        var lambda = Expression.Lambda<Action<T, TProperty>>(assign, objParam, valueParam);
        return lambda.Compile();
    }
}

// Usage:
var getName = FastPropertyAccess.CreateGetter<Person, string>("FirstName");
var setName = FastPropertyAccess.CreateSetter<Person, string>("FirstName");

var person = new Person { FirstName = "Alice" };
Console.WriteLine(getName(person));  // Alice
setName(person, "Bob");
Console.WriteLine(getName(person));  // Bob
// Performance is nearly identical to direct property access
```

## 10. Source Generators as a Compile-Time Alternative

### 10.1 Why Source Generators?

Source generators run during compilation and emit C# code, achieving what reflection does but with zero runtime cost:

```csharp
// Traditional reflection-based approach (runtime cost):
public static string ToQueryString(object obj)
{
    var type = obj.GetType();
    var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
    var pairs = properties
        .Where(p => p.GetValue(obj) != null)
        .Select(p => $"{p.Name}={Uri.EscapeDataString(p.GetValue(obj)!.ToString()!)}");
    return string.Join("&", pairs);
}

// Source-generator approach (compile-time, zero runtime reflection):
// The generator inspects [QueryString]-annotated classes at compile time
// and emits optimized code like:
//
// partial class SearchParams
// {
//     public string ToQueryString()
//     {
//         var sb = new StringBuilder();
//         if (Query != null) sb.Append($"Query={Uri.EscapeDataString(Query)}&");
//         if (Page != null) sb.Append($"Page={Page}&");
//         return sb.ToString().TrimEnd('&');
//     }
// }
```

### 10.2 Well-Known Source Generators in .NET

```csharp
// System.Text.Json source generation
[JsonSerializable(typeof(WeatherForecast))]
public partial class AppJsonContext : JsonSerializerContext { }

// Generated at compile time - no reflection needed for serialization
var json = JsonSerializer.Serialize(forecast, AppJsonContext.Default.WeatherForecast);

// Regex source generation
public partial class Validators
{
    [GeneratedRegex(@"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")]
    public static partial Regex EmailRegex();
}

// LibraryImport source generation (P/Invoke)
public static partial class NativeMethods
{
    [LibraryImport("user32.dll", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int MessageBox(IntPtr hWnd, string text, string caption, int type);
}
```

## 11. Practical Example: Attribute-Based Validation Framework

### 11.1 Defining Validation Attributes

```csharp
// Base validation attribute
[AttributeUsage(AttributeTargets.Property, AllowMultiple = true)]
public abstract class ValidationAttribute : Attribute
{
    public string? ErrorMessage { get; set; }
    public abstract bool IsValid(object? value);
    public virtual string GetErrorMessage(string propertyName)
        => ErrorMessage ?? $"Validation failed for {propertyName}";
}

// [Required]
public class RequiredAttribute : ValidationAttribute
{
    public override bool IsValid(object? value)
        => value switch
        {
            null => false,
            string s => !string.IsNullOrWhiteSpace(s),
            _ => true
        };

    public override string GetErrorMessage(string propertyName)
        => ErrorMessage ?? $"{propertyName} is required";
}

// [StringLength]
public class StringLengthAttribute : ValidationAttribute
{
    public int MinLength { get; }
    public int MaxLength { get; }

    public StringLengthAttribute(int maxLength, int minLength = 0)
    {
        MaxLength = maxLength;
        MinLength = minLength;
    }

    public override bool IsValid(object? value)
    {
        if (value is not string s) return true;  // null handled by Required
        return s.Length >= MinLength && s.Length <= MaxLength;
    }

    public override string GetErrorMessage(string propertyName)
        => ErrorMessage ?? $"{propertyName} must be between {MinLength} and {MaxLength} characters";
}

// [Range]
public class RangeAttribute : ValidationAttribute
{
    public double Min { get; }
    public double Max { get; }

    public RangeAttribute(double min, double max)
    {
        Min = min;
        Max = max;
    }

    public override bool IsValid(object? value)
    {
        if (value is null) return true;
        double numericValue = Convert.ToDouble(value);
        return numericValue >= Min && numericValue <= Max;
    }

    public override string GetErrorMessage(string propertyName)
        => ErrorMessage ?? $"{propertyName} must be between {Min} and {Max}";
}

// [EmailAddress]
public class EmailAddressAttribute : ValidationAttribute
{
    public override bool IsValid(object? value)
    {
        if (value is not string email) return true;
        int atIndex = email.IndexOf('@');
        return atIndex > 0 && email.IndexOf('.', atIndex) > atIndex + 1;
    }

    public override string GetErrorMessage(string propertyName)
        => ErrorMessage ?? $"{propertyName} must be a valid email address";
}
```

### 11.2 The Validation Engine

```csharp
public class ValidationResult
{
    public bool IsValid { get; init; }
    public IReadOnlyList<string> Errors { get; init; } = [];

    public static ValidationResult Success() => new() { IsValid = true, Errors = [] };
    public static ValidationResult Failure(IEnumerable<string> errors)
        => new() { IsValid = false, Errors = errors.ToList() };
}

public static class Validator
{
    // Cache property info and attributes for performance
    private static readonly Dictionary<Type, (PropertyInfo Prop, ValidationAttribute[] Attrs)[]> _cache = new();

    public static ValidationResult Validate(object obj)
    {
        ArgumentNullException.ThrowIfNull(obj);
        Type type = obj.GetType();

        // Get or cache the validation metadata
        if (!_cache.TryGetValue(type, out var validationInfo))
        {
            validationInfo = type.GetProperties(BindingFlags.Public | BindingFlags.Instance)
                .Select(p => (Prop: p, Attrs: p.GetCustomAttributes<ValidationAttribute>().ToArray()))
                .Where(x => x.Attrs.Length > 0)
                .ToArray();
            _cache[type] = validationInfo;
        }

        var errors = new List<string>();

        foreach (var (prop, attrs) in validationInfo)
        {
            object? value = prop.GetValue(obj);
            foreach (var attr in attrs)
            {
                if (!attr.IsValid(value))
                {
                    errors.Add(attr.GetErrorMessage(prop.Name));
                }
            }
        }

        return errors.Count == 0
            ? ValidationResult.Success()
            : ValidationResult.Failure(errors);
    }
}
```

### 11.3 Using the Validation Framework

```csharp
public class CreateUserRequest
{
    [Required]
    [StringLength(50, minLength: 2)]
    public string? Username { get; set; }

    [Required]
    [EmailAddress]
    public string? Email { get; set; }

    [Required]
    [StringLength(100, minLength: 8, ErrorMessage = "Password must be 8-100 characters")]
    public string? Password { get; set; }

    [Range(13, 120, ErrorMessage = "Age must be between 13 and 120")]
    public int Age { get; set; }
}

// Valid request
var validRequest = new CreateUserRequest
{
    Username = "alice",
    Email = "alice@example.com",
    Password = "SecurePass123",
    Age = 30
};

var result1 = Validator.Validate(validRequest);
Console.WriteLine($"Valid: {result1.IsValid}");  // True

// Invalid request
var invalidRequest = new CreateUserRequest
{
    Username = "a",          // Too short
    Email = "not-an-email",  // Invalid format
    Password = "short",      // Too short
    Age = 5                  // Below minimum
};

var result2 = Validator.Validate(invalidRequest);
Console.WriteLine($"Valid: {result2.IsValid}");  // False
foreach (var error in result2.Errors)
    Console.WriteLine($"  - {error}");
// - Username must be between 2 and 50 characters
// - Email must be a valid email address
// - Password must be 8-100 characters
// - Age must be between 13 and 120
```

## 12. Practice Problems

1. **Custom Attribute Design**: Create an `[Auditable]` attribute that can be applied to classes. It should store the `Author`, `CreatedDate` (string), and an optional `Description`. Write code that scans all types in the current assembly and prints a report of all auditable classes with their metadata.

2. **Dynamic Object Mapper**: Write a method `T MapTo<T>(object source) where T : new()` that copies property values from `source` to a new instance of `T` by matching property names. Handle type mismatches gracefully. Test it by mapping a `UserEntity` to a `UserDto` where both share `Id`, `Name`, and `Email` properties.

3. **Plugin Loader**: Build a simple plugin system. Define an `IPlugin` interface with `Name`, `Version`, and `Execute()`. Use `Assembly.LoadFrom()` to load a DLL, find all types implementing `IPlugin`, instantiate them, and call `Execute()`. Describe how you would handle the case where the DLL does not exist or contains no plugins.

4. **Reflection Performance Test**: Create a class with 5 properties. Write three benchmarks that set all 5 properties 100,000 times using: (a) direct property access, (b) `PropertyInfo.SetValue`, and (c) compiled expression delegates. Compare the timings and explain the results.

5. **Attribute-Based Router**: Create a `[Route]` attribute (with HTTP method and path) and a `[QueryParam]` attribute. Annotate several methods in a controller class. Write a discovery function that scans the class and builds a routing table: a `Dictionary<(string Method, string Path), MethodInfo>`. Print the routing table.
