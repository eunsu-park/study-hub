# 리플렉션과 어트리뷰트

**이전**: [NuGet과 프로젝트 시스템](./13_NuGet_and_Project_System.md) | **다음**: [상호 운용과 안전하지 않은 코드](./15_Interop_and_Unsafe.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 내장 어트리뷰트(Attribute)를 적용하여 컴파일러 동작과 메타데이터를 제어할 수 있다
2. 매개변수와 대상이 있는 사용자 지정 어트리뷰트를 설계하고 구현할 수 있다
3. `Type` 클래스를 사용하여 런타임에 타입 정보를 검사할 수 있다
4. 리플렉션(Reflection)을 통해 속성, 메서드, 필드, 생성자를 열거할 수 있다
5. 메서드를 동적으로 호출하고 인스턴스를 생성할 수 있다
6. 런타임에 타입과 멤버에 적용된 어트리뷰트를 발견할 수 있다
7. 프로그래밍 방식으로 어셈블리를 로드하고 검사할 수 있다
8. 리플렉션의 성능 영향과 피해야 할 시점을 이해할 수 있다
9. 소스 생성기(Source Generator)를 리플렉션의 컴파일 타임 대안으로 인식할 수 있다

---

리플렉션(Reflection)은 프로그램이 런타임에 자신의 구조를 검사하고 조작하는 능력입니다. 어트리뷰트(선언적 메타데이터 태그)와 결합하면, 리플렉션은 직렬화 프레임워크, 의존성 주입 컨테이너, 테스트 발견, ORM 매핑, 유효성 검사 시스템과 같은 강력한 패턴을 가능하게 합니다. 이 레슨에서는 타입 메타데이터를 읽고, 멤버를 동적으로 호출하고, 사용자 지정 어트리뷰트를 만들고, 리플렉션이 올바른 도구인 경우와 컴파일 타임 대안이 더 나은 경우를 이해하는 방법을 가르칩니다.

## 1. 내장 어트리뷰트

### 1.1 [Obsolete] 어트리뷰트

`[Obsolete]` 어트리뷰트는 더 이상 사용하면 안 되는 멤버를 표시하여 컴파일러 경고 또는 오류를 생성합니다:

```csharp
public class LegacyService
{
    // 컴파일러 경고 생성
    [Obsolete("Use GetDataAsync instead.")]
    public string GetData() => "old data";

    // 컴파일러 오류 생성 (두 번째 매개변수 = true)
    [Obsolete("This method will be removed in v3.0. Use ProcessAsync.", true)]
    public void Process() { }

    public Task<string> GetDataAsync() => Task.FromResult("new data");
    public Task ProcessAsync() => Task.CompletedTask;
}

// 사용:
var svc = new LegacyService();
// svc.GetData();   // 경고 CS0618: 'GetData'는 사용되지 않음
// svc.Process();   // 오류 CS0619: 'Process'는 사용되지 않음
```

### 1.2 [Conditional] 어트리뷰트

`[Conditional]`은 지정된 심볼이 정의되지 않은 경우 컴파일러가 해당 메서드 호출을 생략하도록 합니다:

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

// Debug 빌드에서 (DEBUG 심볼 정의됨):
Logger.DebugLog("Starting operation");  // 이 호출은 컴파일에 포함됨

// Release 빌드에서 (DEBUG 심볼 정의되지 않음):
Logger.DebugLog("Starting operation");  // 이 호출은 컴파일러에 의해 완전히 제거됨
```

### 1.3 직렬화 어트리뷰트

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

// 직렬화:
var user = new UserDto { UserName = "alice", Email = "alice@example.com", Role = UserRole.Admin, Id = 1 };
string json = JsonSerializer.Serialize(user, new JsonSerializerOptions { WriteIndented = true });
// 출력:
// {
//   "id": 1,
//   "user_name": "alice",
//   "email_address": "alice@example.com",
//   "role": "Admin"
// }
// 참고: InternalToken은 [JsonIgnore]로 인해 제외됨
```

### 1.4 기타 일반적인 내장 어트리뷰트

```csharp
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Runtime.CompilerServices;

public class DemoAttributes
{
    // 컴파일러 생성 호출자 정보
    public void Log(
        string message,
        [CallerMemberName] string memberName = "",
        [CallerFilePath] string filePath = "",
        [CallerLineNumber] int lineNumber = 0)
    {
        Console.WriteLine($"{filePath}:{lineNumber} [{memberName}] {message}");
    }

    // 유효성 검사를 위한 데이터 어노테이션
    [Required]
    [StringLength(100, MinimumLength = 2)]
    public string Name { get; set; } = "";

    [Range(0, 150)]
    public int Age { get; set; }

    [EmailAddress]
    public string Email { get; set; } = "";

    // 도구 및 문서를 위한 설명
    [Description("The maximum number of retry attempts")]
    [DefaultValue(3)]
    public int MaxRetries { get; set; } = 3;
}
```

## 2. 사용자 지정 어트리뷰트 만들기

### 2.1 기본 사용자 지정 어트리뷰트

사용자 지정 어트리뷰트는 `System.Attribute`를 상속하는 클래스입니다:

```csharp
// 사용자 지정 어트리뷰트 정의
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

// 사용자 지정 어트리뷰트 적용
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

### 2.2 AttributeUsage 옵션

```csharp
// AllowMultiple: 어트리뷰트를 여러 번 적용할 수 있는가?
[AttributeUsage(
    AttributeTargets.Class | AttributeTargets.Struct,  // 유효한 대상
    AllowMultiple = true,                               // 여러 번 적용 가능
    Inherited = false                                   // 파생 클래스에 상속되지 않음
)]
public class TagAttribute : Attribute
{
    public string Value { get; }
    public TagAttribute(string value) => Value = value;
}

// 다중 적용 허용:
[Tag("serializable")]
[Tag("auditable")]
[Tag("cacheable")]
public class Order
{
    public int Id { get; set; }
    public decimal Total { get; set; }
}
```

### 2.3 AttributeTargets 열거형

```csharp
// AttributeTargets 값:
// Assembly, Module, Class, Struct, Enum, Constructor, Method,
// Property, Field, Event, Interface, Parameter, Delegate,
// ReturnValue, GenericParameter, All

// 대상별 예제:
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

// 사용:
public class Customer
{
    [ColumnName("customer_name")]
    public string Name { get; set; } = "";

    public void Save([NotNull] string connectionString) { }
}
```

## 3. 리플렉션 기초

### 3.1 타입 정보 가져오기

`System.Type` 클래스는 모든 리플렉션 작업의 진입점입니다:

```csharp
using System;

// Type 객체를 가져오는 세 가지 방법:

// 1. typeof() 연산자 (컴파일 타임, 인스턴스 필요 없음)
Type stringType = typeof(string);
Console.WriteLine(stringType.FullName);  // System.String

// 2. GetType() 인스턴스 메서드 (런타임)
string greeting = "Hello";
Type greetingType = greeting.GetType();
Console.WriteLine(greetingType.Name);  // String

// 3. Type.GetType() 정적 메서드 (문자열 이름으로)
Type? intType = Type.GetType("System.Int32");
Console.WriteLine(intType?.Name);  // Int32

// 타입 비교
Console.WriteLine(stringType == greetingType);  // True
Console.WriteLine(greeting is string);          // True (선호되는 패턴)
```

### 3.2 타입 속성 검사

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

// 제네릭 타입 인수
Type[] genericArgs = type.GetGenericArguments();
foreach (var arg in genericArgs)
    Console.WriteLine($"Generic arg: {arg.Name}");  // Int32

// 구현된 인터페이스
Type[] interfaces = type.GetInterfaces();
foreach (var iface in interfaces)
    Console.WriteLine($"Implements: {iface.Name}");
// IList`1, ICollection`1, IEnumerable`1, IEnumerable, IReadOnlyList`1, ...
```

## 4. 타입 검사

### 4.1 속성 가져오기

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

// 모든 public 인스턴스 속성 가져오기
var publicProps = personType.GetProperties(BindingFlags.Public | BindingFlags.Instance);
Console.WriteLine("Public Properties:");
foreach (var prop in publicProps)
{
    string accessors = "";
    if (prop.CanRead) accessors += "get; ";
    if (prop.CanWrite) accessors += "set; ";
    Console.WriteLine($"  {prop.PropertyType.Name} {prop.Name} {{ {accessors}}}");
}
// 출력:
//   Int32 Id { get; set; }
//   String FirstName { get; set; }
//   String LastName { get; set; }
//   DateTime BirthDate { get; }       (set은 private이므로 CanWrite는 여전히 true)
//   String FullName { get; }

// 비공개 속성도 가져오기
var allProps = personType.GetProperties(
    BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance);
Console.WriteLine($"\nAll properties count: {allProps.Length}");  // Nickname 포함
```

### 4.2 메서드 가져오기

```csharp
public class Calculator
{
    public int Add(int a, int b) => a + b;
    public double Add(double a, double b) => a + b;
    public static int Multiply(int a, int b) => a * b;
    private int Secret() => 42;
}

Type calcType = typeof(Calculator);

// 모든 public 메서드 (Object에서 상속된 것 포함)
var methods = calcType.GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly);
foreach (var method in methods)
{
    var parameters = method.GetParameters();
    string paramList = string.Join(", ",
        parameters.Select(p => $"{p.ParameterType.Name} {p.Name}"));
    Console.WriteLine($"  {method.ReturnType.Name} {method.Name}({paramList})");
}
// 출력:
//   Int32 Add(Int32 a, Int32 b)
//   Double Add(Double a, Double b)

// 정적 메서드 가져오기
var staticMethods = calcType.GetMethods(BindingFlags.Public | BindingFlags.Static);
foreach (var m in staticMethods)
    Console.WriteLine($"  static {m.Name}");  // Multiply

// 비공개 메서드 가져오기
var privateMethods = calcType.GetMethods(BindingFlags.NonPublic | BindingFlags.Instance);
foreach (var m in privateMethods)
    Console.WriteLine($"  private {m.Name}");  // Secret (상속된 것 일부 포함)
```

### 4.3 필드와 생성자 가져오기

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

// 필드
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

// 생성자
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

## 5. 동적 메서드 호출

### 5.1 인스턴스 메서드 호출

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

// 인스턴스를 만들고 리플렉션을 통해 메서드 호출
var greeter = new Greeter();
Type greeterType = typeof(Greeter);

// 매개변수가 하나인 public 메서드 호출
MethodInfo? sayHello1 = greeterType.GetMethod("SayHello", new[] { typeof(string) });
object? result1 = sayHello1?.Invoke(greeter, new object[] { "Alice" });
Console.WriteLine(result1);  // Hello, Alice!

// 매개변수가 두 개인 오버로드된 메서드 호출
MethodInfo? sayHello2 = greeterType.GetMethod("SayHello", new[] { typeof(string), typeof(string) });
object? result2 = sayHello2?.Invoke(greeter, new object[] { "Alice", "es" });
Console.WriteLine(result2);  // Hola, Alice!

// 비공개 메서드 호출
MethodInfo? secret = greeterType.GetMethod("GetMagicNumber",
    BindingFlags.NonPublic | BindingFlags.Instance);
object? result3 = secret?.Invoke(greeter, null);
Console.WriteLine(result3);  // 42
```

### 5.2 정적 메서드 호출

```csharp
public static class MathHelper
{
    public static double Hypotenuse(double a, double b) => Math.Sqrt(a * a + b * b);
}

Type mathType = typeof(MathHelper);
MethodInfo? hypMethod = mathType.GetMethod("Hypotenuse");

// 정적 메서드의 경우 Invoke의 첫 번째 인수는 null
object? result = hypMethod?.Invoke(null, new object[] { 3.0, 4.0 });
Console.WriteLine(result);  // 5
```

### 5.3 속성 값 설정 및 가져오기

```csharp
public class Config
{
    public string AppName { get; set; } = "Default";
    public int MaxRetries { get; set; } = 3;
    public bool Verbose { get; set; }
}

var config = new Config();
Type configType = typeof(Config);

// 속성 값 설정
PropertyInfo? appNameProp = configType.GetProperty("AppName");
appNameProp?.SetValue(config, "MyApplication");

PropertyInfo? retriesProp = configType.GetProperty("MaxRetries");
retriesProp?.SetValue(config, 5);

// 속성 값 가져오기
object? name = appNameProp?.GetValue(config);
Console.WriteLine(name);  // MyApplication

// 딕셔너리에서 동적으로 속성 설정
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

## 6. Activator로 인스턴스 생성

### 6.1 기본 인스턴스 생성

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

// 매개변수 없는 생성자로 생성
object? logger1 = Activator.CreateInstance(typeof(Logger));
((Logger)logger1!).Log("Hello");  // [Info] Default: Hello

// 특정 생성자 매개변수로 생성
object? logger2 = Activator.CreateInstance(typeof(Logger), "AppLogger", LogLevel.Debug);
((Logger)logger2!).Log("Debug info");  // [Debug] AppLogger: Debug info

// 타입 이름 문자열로 생성
Type? loggerType = Type.GetType("MyNamespace.Logger");
if (loggerType != null)
{
    object? logger3 = Activator.CreateInstance(loggerType, "DynamicLogger");
}
```

### 6.2 제네릭 인스턴스 생성

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

// 런타임에 제네릭 타입 생성
Type openGenericType = typeof(Repository<>);  // 열린 제네릭 타입
Type closedGenericType = openGenericType.MakeGenericType(typeof(Product));  // Repository<Product>

object? repo = Activator.CreateInstance(closedGenericType);

// 제네릭 인스턴스에서 메서드 호출
MethodInfo? addMethod = closedGenericType.GetMethod("Add");
var product = new Product { Id = 1, Name = "Widget" };
addMethod?.Invoke(repo, new object[] { product });

Console.WriteLine(repo);  // Repository<Product> with 1 items
```

### 6.3 Activator를 사용한 팩토리 패턴

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

// 리플렉션을 사용한 동적 팩토리
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

// 사용:
HandlerFactory.Register("email", typeof(EmailHandler));
HandlerFactory.Register("sms", typeof(SmsHandler));
HandlerFactory.Register("slack", typeof(SlackHandler));

IMessageHandler handler = HandlerFactory.Create("slack");
handler.Handle("Build succeeded");  // Slack: Build succeeded
```

## 7. 런타임에서의 어트리뷰트 발견

### 7.1 타입에서 어트리뷰트 읽기

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

// [ApiEndpoint]가 있는 모든 컨트롤러 발견
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

### 7.2 멤버에서 어트리뷰트 읽기

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

// 표시 이름을 사용하여 형식화된 헤더 출력
Type empType = typeof(Employee);
var properties = empType.GetProperties();

foreach (var prop in properties)
{
    var displayAttr = prop.GetCustomAttribute<DisplayNameAttribute>();
    var maxLenAttr = prop.GetCustomAttribute<MaxLengthAttribute>();

    string displayName = displayAttr?.Name ?? prop.Name;
    string constraint = maxLenAttr != null ? $" (최대 {maxLenAttr.Length}자)" : "";
    Console.WriteLine($"  {displayName}: {prop.PropertyType.Name}{constraint}");
}
// Employee ID: Int32
// Full Name: String (최대 100자)
// Department: String (최대 50자)
```

### 7.3 어트리뷰트 존재 여부 확인

```csharp
// IsDefined는 존재 여부만 확인할 때 더 빠름
bool hasObsolete = typeof(LegacyService).IsDefined(typeof(ObsoleteAttribute), false);

// GetCustomAttributes는 모든 어트리뷰트를 반환
var allAttributes = typeof(Order).GetCustomAttributes(true);
foreach (var attr in allAttributes)
    Console.WriteLine($"  {attr.GetType().Name}");

// 제네릭 타입 필터가 있는 GetCustomAttributes<T>
var tags = typeof(Order).GetCustomAttributes<TagAttribute>();
foreach (var tag in tags)
    Console.WriteLine($"  Tag: {tag.Value}");
```

## 8. 어셈블리 검사

### 8.1 어셈블리 로드 및 검사

```csharp
using System.Reflection;

// 현재 실행 중인 어셈블리 가져오기
Assembly currentAssembly = Assembly.GetExecutingAssembly();
Console.WriteLine($"Name: {currentAssembly.GetName().Name}");
Console.WriteLine($"Version: {currentAssembly.GetName().Version}");
Console.WriteLine($"Location: {currentAssembly.Location}");

// 어셈블리의 모든 타입 가져오기
Type[] allTypes = currentAssembly.GetTypes();
Console.WriteLine($"Total types: {allTypes.Length}");

// 모든 public 클래스 찾기
var publicClasses = allTypes.Where(t => t.IsClass && t.IsPublic);
foreach (var cls in publicClasses)
    Console.WriteLine($"  Class: {cls.FullName}");

// 특정 인터페이스를 구현하는 타입 찾기
var handlers = allTypes
    .Where(t => typeof(IMessageHandler).IsAssignableFrom(t) && !t.IsInterface && !t.IsAbstract);
foreach (var handler in handlers)
    Console.WriteLine($"  Handler: {handler.Name}");
```

### 8.2 외부 어셈블리 로드

```csharp
// 이름으로 로드 (애플리케이션의 탐색 경로에서)
Assembly? byName = Assembly.Load("System.Text.Json");

// 특정 파일 경로에서 로드
Assembly fromFile = Assembly.LoadFrom("/path/to/MyPlugin.dll");

// 외부 어셈블리에서 내보낸 타입 검사
var exportedTypes = fromFile.GetExportedTypes();
foreach (var type in exportedTypes)
{
    Console.WriteLine($"  {type.FullName}");
}

// 특정 타입 찾기
Type? pluginType = fromFile.GetType("MyPlugin.DataProcessor");
if (pluginType != null)
{
    object? instance = Activator.CreateInstance(pluginType);
    MethodInfo? runMethod = pluginType.GetMethod("Run");
    runMethod?.Invoke(instance, null);
}
```

### 8.3 어셈블리 메타데이터

```csharp
Assembly asm = Assembly.GetExecutingAssembly();

// 어셈블리 수준 어트리뷰트 읽기
var title = asm.GetCustomAttribute<AssemblyTitleAttribute>()?.Title;
var company = asm.GetCustomAttribute<AssemblyCompanyAttribute>()?.Company;
var version = asm.GetCustomAttribute<AssemblyFileVersionAttribute>()?.Version;
var informational = asm.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;

Console.WriteLine($"Title: {title}");
Console.WriteLine($"Company: {company}");
Console.WriteLine($"File Version: {version}");
Console.WriteLine($"Informational Version: {informational}");

// 사용자 지정 메타데이터 읽기 (빌드 시스템에 의해 추가됨)
var metadata = asm.GetCustomAttributes<AssemblyMetadataAttribute>();
foreach (var m in metadata)
    Console.WriteLine($"  {m.Key} = {m.Value}");
// 예시 출력: BuildTimestamp = 2025-01-15T10:30:00Z
```

## 9. 리플렉션의 성능 비용

### 9.1 리플렉션 vs 직접 호출 벤치마킹

```csharp
using System.Diagnostics;
using System.Reflection;

public class MathService
{
    public int Square(int x) => x * x;
}

// 벤치마크: 직접 호출 vs 리플렉션
var service = new MathService();
Type type = typeof(MathService);
MethodInfo method = type.GetMethod("Square")!;
const int iterations = 1_000_000;

// 직접 호출
var sw = Stopwatch.StartNew();
for (int i = 0; i < iterations; i++)
{
    _ = service.Square(42);
}
sw.Stop();
Console.WriteLine($"직접 호출:      {sw.ElapsedMilliseconds} ms");

// 리플렉션 호출
sw.Restart();
for (int i = 0; i < iterations; i++)
{
    _ = method.Invoke(service, new object[] { 42 });
}
sw.Stop();
Console.WriteLine($"리플렉션 호출: {sw.ElapsedMilliseconds} ms");

// 일반적인 결과:
// 직접 호출:       ~2 ms
// 리플렉션 호출:   ~500 ms
// 리플렉션은 ~100-250배 느림
```

### 9.2 리플렉션 결과 캐싱

```csharp
// 잘못된 방법: 매번 메서드를 조회
public static object? SlowInvoke(object target, string methodName, params object[] args)
{
    // 매 호출마다 GetMethod가 호출됨 - 비용이 높음!
    MethodInfo? method = target.GetType().GetMethod(methodName);
    return method?.Invoke(target, args);
}

// 더 나은 방법: MethodInfo 객체 캐싱
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

// 최선의 방법: 반복 호출을 위해 대리자로 컴파일
public static class DelegateCache
{
    public static Func<object, object[], object?> CreateInvoker(MethodInfo method)
    {
        // 대리자를 캐싱 - 이후 호출에서 리플렉션 오버헤드 방지
        return (target, args) => method.Invoke(target, args);
    }
}
```

### 9.3 최대 성능을 위한 컴파일된 식

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
        return lambda.Compile();  // IL로 컴파일 - 직접 접근과 거의 같은 속도
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

// 사용:
var getName = FastPropertyAccess.CreateGetter<Person, string>("FirstName");
var setName = FastPropertyAccess.CreateSetter<Person, string>("FirstName");

var person = new Person { FirstName = "Alice" };
Console.WriteLine(getName(person));  // Alice
setName(person, "Bob");
Console.WriteLine(getName(person));  // Bob
// 성능은 직접 속성 접근과 거의 동일
```

## 10. 컴파일 타임 대안으로서의 소스 생성기

### 10.1 소스 생성기를 사용하는 이유

소스 생성기(Source Generator)는 컴파일 중에 실행되어 C# 코드를 생성하므로, 리플렉션이 하는 일을 런타임 비용 없이 달성합니다:

```csharp
// 전통적인 리플렉션 기반 접근 방식 (런타임 비용):
public static string ToQueryString(object obj)
{
    var type = obj.GetType();
    var properties = type.GetProperties(BindingFlags.Public | BindingFlags.Instance);
    var pairs = properties
        .Where(p => p.GetValue(obj) != null)
        .Select(p => $"{p.Name}={Uri.EscapeDataString(p.GetValue(obj)!.ToString()!)}");
    return string.Join("&", pairs);
}

// 소스 생성기 접근 방식 (컴파일 타임, 런타임 리플렉션 제로):
// 생성기가 컴파일 타임에 [QueryString] 어노테이션이 달린 클래스를 검사하고
// 최적화된 코드를 다음과 같이 생성합니다:
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

### 10.2 .NET의 잘 알려진 소스 생성기

```csharp
// System.Text.Json 소스 생성
[JsonSerializable(typeof(WeatherForecast))]
public partial class AppJsonContext : JsonSerializerContext { }

// 컴파일 타임에 생성 - 직렬화에 리플렉션이 필요 없음
var json = JsonSerializer.Serialize(forecast, AppJsonContext.Default.WeatherForecast);

// Regex 소스 생성
public partial class Validators
{
    [GeneratedRegex(@"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")]
    public static partial Regex EmailRegex();
}

// LibraryImport 소스 생성 (P/Invoke)
public static partial class NativeMethods
{
    [LibraryImport("user32.dll", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int MessageBox(IntPtr hWnd, string text, string caption, int type);
}
```

## 11. 실전 예제: 어트리뷰트 기반 유효성 검사 프레임워크

### 11.1 유효성 검사 어트리뷰트 정의

```csharp
// 기본 유효성 검사 어트리뷰트
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
        if (value is not string s) return true;  // null은 Required에서 처리
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

### 11.2 유효성 검사 엔진

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
    // 성능을 위해 속성 정보와 어트리뷰트를 캐싱
    private static readonly Dictionary<Type, (PropertyInfo Prop, ValidationAttribute[] Attrs)[]> _cache = new();

    public static ValidationResult Validate(object obj)
    {
        ArgumentNullException.ThrowIfNull(obj);
        Type type = obj.GetType();

        // 유효성 검사 메타데이터 가져오기 또는 캐싱
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

### 11.3 유효성 검사 프레임워크 사용

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

// 유효한 요청
var validRequest = new CreateUserRequest
{
    Username = "alice",
    Email = "alice@example.com",
    Password = "SecurePass123",
    Age = 30
};

var result1 = Validator.Validate(validRequest);
Console.WriteLine($"Valid: {result1.IsValid}");  // True

// 유효하지 않은 요청
var invalidRequest = new CreateUserRequest
{
    Username = "a",          // 너무 짧음
    Email = "not-an-email",  // 잘못된 형식
    Password = "short",      // 너무 짧음
    Age = 5                  // 최솟값 미만
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

## 12. 연습 문제

1. **사용자 지정 어트리뷰트 설계**: 클래스에 적용할 수 있는 `[Auditable]` 어트리뷰트를 만드세요. `Author`, `CreatedDate` (문자열), 선택적 `Description`을 저장해야 합니다. 현재 어셈블리의 모든 타입을 스캔하고 감사 가능한(auditable) 모든 클래스와 메타데이터를 출력하는 코드를 작성하세요.

2. **동적 객체 매퍼**: 속성 이름을 매칭하여 `source`의 속성 값을 새 `T` 인스턴스에 복사하는 `T MapTo<T>(object source) where T : new()` 메서드를 작성하세요. 타입 불일치를 우아하게 처리하세요. `Id`, `Name`, `Email` 속성을 공유하는 `UserEntity`를 `UserDto`로 매핑하여 테스트하세요.

3. **플러그인 로더**: 간단한 플러그인 시스템을 구축하세요. `Name`, `Version`, `Execute()`가 있는 `IPlugin` 인터페이스를 정의하세요. `Assembly.LoadFrom()`을 사용하여 DLL을 로드하고, `IPlugin`을 구현하는 모든 타입을 찾고, 인스턴스를 만들어 `Execute()`를 호출하세요. DLL이 존재하지 않거나 플러그인이 없는 경우를 어떻게 처리할지 설명하세요.

4. **리플렉션 성능 테스트**: 5개의 속성이 있는 클래스를 만드세요. (a) 직접 속성 접근, (b) `PropertyInfo.SetValue`, (c) 컴파일된 식 대리자를 사용하여 5개 속성 모두를 100,000번 설정하는 세 가지 벤치마크를 작성하세요. 시간을 비교하고 결과를 설명하세요.

5. **어트리뷰트 기반 라우터**: HTTP 메서드와 경로가 있는 `[Route]` 어트리뷰트와 `[QueryParam]` 어트리뷰트를 만드세요. 컨트롤러 클래스의 여러 메서드에 어노테이션을 붙이세요. 클래스를 스캔하고 라우팅 테이블(`Dictionary<(string Method, string Path), MethodInfo>`)을 구축하는 발견 함수를 작성하세요. 라우팅 테이블을 출력하세요.
