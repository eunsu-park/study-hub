# Dependency Injection

**Previous**: [Spans and Memory](./09_Spans_and_Memory.md) | **Next**: [Serialization](./11_Serialization.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the Dependency Injection (DI) principle and its benefits
2. Apply constructor injection to decouple components
3. Configure the Microsoft.Extensions.DependencyInjection container
4. Choose the correct service lifetime (Transient, Scoped, Singleton)
5. Use the Options pattern to inject strongly-typed configuration
6. Register keyed services and factory-based registrations
7. Recognize and avoid common DI anti-patterns
8. Design modular applications using DI composition roots

---

As applications grow, classes inevitably depend on other classes: a controller needs a service, a service needs a repository, a repository needs a database connection. Without a disciplined approach, these dependencies become tangled, hard to test, and difficult to change. Dependency Injection (DI) inverts the control: instead of a class creating its own dependencies, they are provided from the outside. C# and .NET have first-class support for DI through `Microsoft.Extensions.DependencyInjection`, which powers ASP.NET Core, worker services, and any .NET application.

## 1. What Is DI and Why Use It

### 1.1 The Problem: Tight Coupling

```csharp
// Tightly coupled — OrderService creates its own dependencies
public class OrderService
{
    private readonly SqlOrderRepository _repository;
    private readonly SmtpEmailSender _emailSender;

    public OrderService()
    {
        // Hard-coded dependencies
        _repository = new SqlOrderRepository("Server=localhost;Database=Orders;...");
        _emailSender = new SmtpEmailSender("smtp.example.com", 587);
    }

    public void PlaceOrder(Order order)
    {
        _repository.Save(order);
        _emailSender.Send(order.CustomerEmail, "Order confirmed", $"Order {order.Id}");
    }
}
```

Problems with this approach:
- Cannot swap `SqlOrderRepository` for an in-memory repository in tests
- Cannot replace `SmtpEmailSender` with a mock to avoid sending real emails
- Connection strings are buried inside the class
- Adding a new dependency requires modifying the constructor

### 1.2 The Solution: Dependency Injection

```csharp
// Loosely coupled — dependencies are injected through the constructor
public interface IOrderRepository
{
    void Save(Order order);
    Order? GetById(int id);
}

public interface IEmailSender
{
    void Send(string to, string subject, string body);
}

public class OrderService
{
    private readonly IOrderRepository _repository;
    private readonly IEmailSender _emailSender;

    // Dependencies are provided externally
    public OrderService(IOrderRepository repository, IEmailSender emailSender)
    {
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
        _emailSender = emailSender ?? throw new ArgumentNullException(nameof(emailSender));
    }

    public void PlaceOrder(Order order)
    {
        _repository.Save(order);
        _emailSender.Send(order.CustomerEmail, "Order confirmed", $"Order {order.Id}");
    }
}
```

Benefits:
- **Testability**: Inject mock implementations in unit tests
- **Flexibility**: Swap implementations without changing consuming code
- **Separation of concerns**: Each class does one thing
- **Explicit dependencies**: Constructor signature documents what a class needs

## 2. Constructor Injection

Constructor injection is the recommended pattern in .NET. All required dependencies are declared as constructor parameters.

### 2.1 Single Responsibility

```csharp
public interface ILogger
{
    void Log(string message);
}

public interface IInventoryService
{
    bool Reserve(string productId, int quantity);
    void Release(string productId, int quantity);
}

public interface IPaymentGateway
{
    PaymentResult Charge(string customerId, decimal amount);
}

public class CheckoutService
{
    private readonly IInventoryService _inventory;
    private readonly IPaymentGateway _payment;
    private readonly IOrderRepository _orders;
    private readonly ILogger _logger;

    public CheckoutService(
        IInventoryService inventory,
        IPaymentGateway payment,
        IOrderRepository orders,
        ILogger logger)
    {
        _inventory = inventory;
        _payment = payment;
        _orders = orders;
        _logger = logger;
    }

    public CheckoutResult Checkout(Cart cart)
    {
        _logger.Log($"Starting checkout for cart {cart.Id}");

        foreach (var item in cart.Items)
        {
            if (!_inventory.Reserve(item.ProductId, item.Quantity))
            {
                _logger.Log($"Insufficient inventory for {item.ProductId}");
                return CheckoutResult.InsufficientInventory;
            }
        }

        PaymentResult paymentResult = _payment.Charge(cart.CustomerId, cart.Total);
        if (!paymentResult.Success)
        {
            foreach (var item in cart.Items)
                _inventory.Release(item.ProductId, item.Quantity);

            return CheckoutResult.PaymentFailed;
        }

        var order = new Order(cart);
        _orders.Save(order);
        _logger.Log($"Order {order.Id} placed successfully");
        return CheckoutResult.Success;
    }
}
```

### 2.2 Primary Constructors (C# 12)

```csharp
// C# 12 primary constructor — more concise
public class CheckoutService(
    IInventoryService inventory,
    IPaymentGateway payment,
    IOrderRepository orders,
    ILogger logger)
{
    public CheckoutResult Checkout(Cart cart)
    {
        logger.Log($"Starting checkout for cart {cart.Id}");
        // ... use inventory, payment, orders directly
        return CheckoutResult.Success;
    }
}
```

## 3. Microsoft.Extensions.DependencyInjection

The built-in DI container in .NET is lightweight and sufficient for most applications.

### 3.1 Setting Up the Container

```csharp
using Microsoft.Extensions.DependencyInjection;

// Create and configure the service collection
var services = new ServiceCollection();

// Register services
services.AddTransient<IEmailSender, SmtpEmailSender>();
services.AddScoped<IOrderRepository, SqlOrderRepository>();
services.AddSingleton<ILogger, ConsoleLogger>();
services.AddScoped<OrderService>();

// Build the service provider (composition root)
ServiceProvider provider = services.BuildServiceProvider();

// Resolve a service
using var scope = provider.CreateScope();
var orderService = scope.ServiceProvider.GetRequiredService<OrderService>();
orderService.PlaceOrder(new Order { CustomerEmail = "user@example.com" });
```

### 3.2 In ASP.NET Core

```csharp
var builder = WebApplication.CreateBuilder(args);

// Register services in Program.cs
builder.Services.AddScoped<IOrderRepository, SqlOrderRepository>();
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();
builder.Services.AddScoped<OrderService>();
builder.Services.AddControllers();

var app = builder.Build();
app.MapControllers();
app.Run();
```

```csharp
// Controller receives dependencies automatically
[ApiController]
[Route("api/[controller]")]
public class OrdersController : ControllerBase
{
    private readonly OrderService _orderService;

    public OrdersController(OrderService orderService)
    {
        _orderService = orderService;
    }

    [HttpPost]
    public IActionResult PlaceOrder([FromBody] OrderRequest request)
    {
        var order = new Order { CustomerEmail = request.Email };
        _orderService.PlaceOrder(order);
        return Ok(new { order.Id });
    }
}
```

### 3.3 GetService vs GetRequiredService

```csharp
// GetService returns null if not registered
IEmailSender? sender = provider.GetService<IEmailSender>();
if (sender is not null)
{
    sender.Send("test@example.com", "Subject", "Body");
}

// GetRequiredService throws InvalidOperationException if not registered
IEmailSender requiredSender = provider.GetRequiredService<IEmailSender>();
```

## 4. Service Lifetimes: Transient, Scoped, Singleton

Choosing the correct lifetime is critical for correctness and performance.

### 4.1 Transient

A new instance is created every time the service is requested.

```csharp
services.AddTransient<IEmailSender, SmtpEmailSender>();

// Each resolution creates a new instance
var sender1 = provider.GetRequiredService<IEmailSender>();
var sender2 = provider.GetRequiredService<IEmailSender>();
Console.WriteLine(ReferenceEquals(sender1, sender2)); // False
```

**Use for**: Lightweight, stateless services. Services that hold no shared state.

### 4.2 Scoped

One instance per scope. In ASP.NET Core, each HTTP request creates a scope.

```csharp
services.AddScoped<IOrderRepository, SqlOrderRepository>();

using var scope1 = provider.CreateScope();
var repo1a = scope1.ServiceProvider.GetRequiredService<IOrderRepository>();
var repo1b = scope1.ServiceProvider.GetRequiredService<IOrderRepository>();
Console.WriteLine(ReferenceEquals(repo1a, repo1b)); // True — same scope

using var scope2 = provider.CreateScope();
var repo2 = scope2.ServiceProvider.GetRequiredService<IOrderRepository>();
Console.WriteLine(ReferenceEquals(repo1a, repo2)); // False — different scope
```

**Use for**: Database contexts (DbContext), unit-of-work patterns, per-request state.

### 4.3 Singleton

One instance for the entire application lifetime.

```csharp
services.AddSingleton<ILogger, ConsoleLogger>();

var logger1 = provider.GetRequiredService<ILogger>();
var logger2 = provider.GetRequiredService<ILogger>();
Console.WriteLine(ReferenceEquals(logger1, logger2)); // True
```

**Use for**: Configuration, caches, HTTP client factories, thread-safe shared state.

### 4.4 Lifetime Comparison Table

```csharp
public class LifetimeDemo
{
    private static int _counter;

    public class MyService
    {
        public int Id { get; } = Interlocked.Increment(ref _counter);
        public override string ToString() => $"Instance #{Id}";
    }

    public static void Demonstrate()
    {
        var services = new ServiceCollection();
        services.AddTransient<MyService>();    // Change to test different lifetimes
        var provider = services.BuildServiceProvider();

        using var scope1 = provider.CreateScope();
        var a = scope1.ServiceProvider.GetRequiredService<MyService>();
        var b = scope1.ServiceProvider.GetRequiredService<MyService>();

        using var scope2 = provider.CreateScope();
        var c = scope2.ServiceProvider.GetRequiredService<MyService>();

        // Transient: a=#1, b=#2, c=#3 (all different)
        // Scoped:    a=#1, b=#1, c=#2 (same within scope)
        // Singleton: a=#1, b=#1, c=#1 (always same)
        Console.WriteLine($"a={a}, b={b}, c={c}");
    }
}
```

## 5. IServiceProvider and Service Resolution

### 5.1 Resolving Multiple Implementations

```csharp
// Register multiple implementations of the same interface
services.AddTransient<INotificationChannel, EmailChannel>();
services.AddTransient<INotificationChannel, SmsChannel>();
services.AddTransient<INotificationChannel, PushChannel>();

// Resolve all implementations
IEnumerable<INotificationChannel> channels =
    provider.GetServices<INotificationChannel>();

foreach (var channel in channels)
{
    channel.Send("Hello, World!");
}
```

### 5.2 Creating a Notification Dispatcher

```csharp
public class NotificationDispatcher
{
    private readonly IEnumerable<INotificationChannel> _channels;

    public NotificationDispatcher(IEnumerable<INotificationChannel> channels)
    {
        _channels = channels;
    }

    public void Broadcast(string message)
    {
        foreach (var channel in _channels)
        {
            try
            {
                channel.Send(message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{channel.GetType().Name} failed: {ex.Message}");
            }
        }
    }
}

// Registration
services.AddTransient<INotificationChannel, EmailChannel>();
services.AddTransient<INotificationChannel, SmsChannel>();
services.AddTransient<NotificationDispatcher>();
```

## 6. Options Pattern (IOptions&lt;T&gt;)

The Options pattern provides strongly-typed access to configuration sections.

### 6.1 Defining Options Classes

```csharp
public class SmtpOptions
{
    public const string SectionName = "Smtp";

    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 587;
    public string Username { get; set; } = "";
    public string Password { get; set; } = "";
    public bool UseSsl { get; set; } = true;
}

public class DatabaseOptions
{
    public const string SectionName = "Database";

    public string ConnectionString { get; set; } = "";
    public int MaxRetries { get; set; } = 3;
    public int CommandTimeoutSeconds { get; set; } = 30;
}
```

### 6.2 Binding Configuration

```csharp
// appsettings.json
// {
//   "Smtp": {
//     "Host": "smtp.example.com",
//     "Port": 587,
//     "Username": "noreply@example.com",
//     "Password": "secret",
//     "UseSsl": true
//   },
//   "Database": {
//     "ConnectionString": "Server=localhost;Database=MyApp;...",
//     "MaxRetries": 5
//   }
// }

var builder = WebApplication.CreateBuilder(args);

// Bind options from configuration
builder.Services.Configure<SmtpOptions>(
    builder.Configuration.GetSection(SmtpOptions.SectionName));

builder.Services.Configure<DatabaseOptions>(
    builder.Configuration.GetSection(DatabaseOptions.SectionName));
```

### 6.3 Injecting Options

```csharp
public class SmtpEmailSender : IEmailSender
{
    private readonly SmtpOptions _options;

    // IOptions<T>: read once at startup
    public SmtpEmailSender(IOptions<SmtpOptions> options)
    {
        _options = options.Value;
    }

    public void Send(string to, string subject, string body)
    {
        Console.WriteLine($"Sending via {_options.Host}:{_options.Port} to {to}");
        // ... actual SMTP implementation
    }
}
```

### 6.4 IOptionsSnapshot vs IOptionsMonitor

```csharp
public class ConfigAwareService
{
    private readonly IOptionsMonitor<DatabaseOptions> _dbOptions;

    // IOptionsMonitor<T>: reloads when configuration changes (Singleton-safe)
    public ConfigAwareService(IOptionsMonitor<DatabaseOptions> dbOptions)
    {
        _dbOptions = dbOptions;
        _dbOptions.OnChange(newOptions =>
        {
            Console.WriteLine($"Config changed: MaxRetries = {newOptions.MaxRetries}");
        });
    }

    public string GetConnectionString()
    {
        return _dbOptions.CurrentValue.ConnectionString;
    }
}

// IOptionsSnapshot<T>: scoped — new snapshot per request (Scoped services only)
public class RequestScopedService
{
    private readonly DatabaseOptions _options;

    public RequestScopedService(IOptionsSnapshot<DatabaseOptions> options)
    {
        _options = options.Value;
    }
}
```

### 6.5 Options Validation

```csharp
builder.Services.AddOptions<SmtpOptions>()
    .Bind(builder.Configuration.GetSection(SmtpOptions.SectionName))
    .Validate(o => !string.IsNullOrEmpty(o.Host), "SMTP host is required")
    .Validate(o => o.Port is > 0 and < 65536, "Port must be between 1 and 65535")
    .ValidateOnStart(); // Fail fast at startup if invalid
```

## 7. Keyed Services

Keyed services (introduced in .NET 8) let you register and resolve multiple implementations of the same interface using string or enum keys.

```csharp
public enum StorageType { Local, S3, Azure }

services.AddKeyedSingleton<IStorageProvider, LocalStorageProvider>(StorageType.Local);
services.AddKeyedSingleton<IStorageProvider, S3StorageProvider>(StorageType.S3);
services.AddKeyedSingleton<IStorageProvider, AzureBlobStorageProvider>(StorageType.Azure);

// Inject a specific keyed service
public class FileUploadService
{
    private readonly IStorageProvider _storage;

    public FileUploadService(
        [FromKeyedServices(StorageType.S3)] IStorageProvider storage)
    {
        _storage = storage;
    }

    public async Task UploadAsync(Stream file, string path)
    {
        await _storage.SaveAsync(file, path);
    }
}
```

```csharp
// Resolve keyed services manually
var provider = services.BuildServiceProvider();
var s3 = provider.GetRequiredKeyedService<IStorageProvider>(StorageType.S3);
var local = provider.GetRequiredKeyedService<IStorageProvider>(StorageType.Local);
```

## 8. Factory-Based Registrations

Sometimes you need more control over how a service is created.

### 8.1 Simple Factory

```csharp
services.AddTransient<IEmailSender>(provider =>
{
    var options = provider.GetRequiredService<IOptions<SmtpOptions>>().Value;
    var logger = provider.GetRequiredService<ILogger<SmtpEmailSender>>();
    return new SmtpEmailSender(options, logger);
});
```

### 8.2 Conditional Registration

```csharp
services.AddScoped<IOrderRepository>(provider =>
{
    var config = provider.GetRequiredService<IConfiguration>();
    string dbType = config["Database:Type"] ?? "sql";

    return dbType.ToLowerInvariant() switch
    {
        "sql" => new SqlOrderRepository(
            provider.GetRequiredService<IOptions<DatabaseOptions>>()),
        "mongo" => new MongoOrderRepository(
            provider.GetRequiredService<IOptions<MongoOptions>>()),
        "memory" => new InMemoryOrderRepository(),
        _ => throw new InvalidOperationException($"Unknown database type: {dbType}")
    };
});
```

### 8.3 Decorator Pattern with DI

```csharp
// Register base implementation
services.AddScoped<SqlOrderRepository>();

// Register decorator that wraps the base implementation
services.AddScoped<IOrderRepository>(provider =>
{
    var inner = provider.GetRequiredService<SqlOrderRepository>();
    var logger = provider.GetRequiredService<ILogger<LoggingOrderRepository>>();
    var cache = provider.GetRequiredService<IMemoryCache>();

    // Wrap with logging decorator
    IOrderRepository logged = new LoggingOrderRepository(inner, logger);
    // Wrap with caching decorator
    IOrderRepository cached = new CachingOrderRepository(logged, cache);

    return cached;
});
```

```csharp
public class LoggingOrderRepository : IOrderRepository
{
    private readonly IOrderRepository _inner;
    private readonly ILogger _logger;

    public LoggingOrderRepository(IOrderRepository inner, ILogger logger)
    {
        _inner = inner;
        _logger = logger;
    }

    public void Save(Order order)
    {
        _logger.LogInformation("Saving order {OrderId}", order.Id);
        _inner.Save(order);
        _logger.LogInformation("Saved order {OrderId}", order.Id);
    }

    public Order? GetById(int id)
    {
        _logger.LogInformation("Retrieving order {OrderId}", id);
        return _inner.GetById(id);
    }
}
```

## 9. DI Best Practices

### 9.1 Avoid the Service Locator Anti-Pattern

```csharp
// BAD: Service Locator — hides dependencies, hard to test
public class BadService
{
    private readonly IServiceProvider _provider;

    public BadService(IServiceProvider provider)
    {
        _provider = provider;
    }

    public void DoWork()
    {
        // Dependency is hidden — not visible in the constructor
        var repo = _provider.GetRequiredService<IOrderRepository>();
        repo.Save(new Order());
    }
}

// GOOD: Explicit constructor injection
public class GoodService
{
    private readonly IOrderRepository _repository;

    public GoodService(IOrderRepository repository)
    {
        _repository = repository; // Dependency is explicit
    }

    public void DoWork()
    {
        _repository.Save(new Order());
    }
}
```

### 9.2 Avoid Captive Dependencies

A captive dependency occurs when a longer-lived service holds a reference to a shorter-lived service.

```csharp
// DANGEROUS: Singleton captures a Scoped service
services.AddSingleton<MySingletonService>();  // Lives forever
services.AddScoped<MyDbContext>();              // Lives per request

public class MySingletonService
{
    private readonly MyDbContext _db; // BUG: This DbContext outlives its scope!

    public MySingletonService(MyDbContext db)
    {
        _db = db; // This DbContext was created in a scope that may have been disposed
    }
}
```

```csharp
// FIX: Use IServiceScopeFactory to create scopes on demand
public class MySingletonService
{
    private readonly IServiceScopeFactory _scopeFactory;

    public MySingletonService(IServiceScopeFactory scopeFactory)
    {
        _scopeFactory = scopeFactory;
    }

    public void DoWork()
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<MyDbContext>();
        // Use db within this scope — it will be properly disposed
    }
}
```

### 9.3 Register Interfaces, Not Implementations

```csharp
// BAD: registering concrete type as itself
services.AddScoped<SqlOrderRepository>();
// Consumers must depend on SqlOrderRepository directly

// GOOD: register behind an interface
services.AddScoped<IOrderRepository, SqlOrderRepository>();
// Consumers depend on IOrderRepository — can swap implementations
```

### 9.4 Composition Root

All DI registration should happen in a single location — the composition root. In ASP.NET Core, this is `Program.cs`. Avoid scattering registrations across the codebase.

```csharp
// Program.cs — the composition root
var builder = WebApplication.CreateBuilder(args);

// Group registrations by feature
builder.Services.AddOrderingModule(builder.Configuration);
builder.Services.AddNotificationModule(builder.Configuration);
builder.Services.AddInventoryModule(builder.Configuration);

var app = builder.Build();
```

```csharp
// Extension method for modular registration
public static class OrderingModuleExtensions
{
    public static IServiceCollection AddOrderingModule(
        this IServiceCollection services, IConfiguration config)
    {
        services.Configure<DatabaseOptions>(config.GetSection("Database"));
        services.AddScoped<IOrderRepository, SqlOrderRepository>();
        services.AddScoped<OrderService>();
        services.AddScoped<CheckoutService>();
        return services;
    }
}
```

## 10. Practical Example: Modular Application with DI

This example builds a modular console application with a clean DI setup: interfaces, implementations, options, and a composition root.

```csharp
// --- Interfaces ---
public interface IWeatherProvider
{
    Task<WeatherData> GetCurrentWeatherAsync(string city);
}

public interface IWeatherCache
{
    WeatherData? Get(string city);
    void Set(string city, WeatherData data, TimeSpan expiry);
}

public interface IWeatherFormatter
{
    string Format(WeatherData data);
}

public record WeatherData(string City, double Temperature, string Condition, DateTime Timestamp);
```

```csharp
// --- Options ---
public class WeatherApiOptions
{
    public string ApiKey { get; set; } = "";
    public string BaseUrl { get; set; } = "https://api.weather.example.com";
    public int TimeoutSeconds { get; set; } = 10;
}

public class CacheOptions
{
    public int ExpirationMinutes { get; set; } = 15;
}
```

```csharp
// --- Implementations ---
public class HttpWeatherProvider : IWeatherProvider
{
    private readonly HttpClient _client;
    private readonly WeatherApiOptions _options;

    public HttpWeatherProvider(HttpClient client, IOptions<WeatherApiOptions> options)
    {
        _client = client;
        _options = options.Value;
        _client.BaseAddress = new Uri(_options.BaseUrl);
        _client.Timeout = TimeSpan.FromSeconds(_options.TimeoutSeconds);
    }

    public async Task<WeatherData> GetCurrentWeatherAsync(string city)
    {
        var response = await _client.GetStringAsync($"/current?city={city}&key={_options.ApiKey}");
        // Parse response (simplified)
        return new WeatherData(city, 22.5, "Sunny", DateTime.UtcNow);
    }
}

public class InMemoryWeatherCache : IWeatherCache
{
    private readonly ConcurrentDictionary<string, (WeatherData Data, DateTime Expiry)> _cache = new();

    public WeatherData? Get(string city)
    {
        if (_cache.TryGetValue(city, out var entry) && entry.Expiry > DateTime.UtcNow)
            return entry.Data;
        return null;
    }

    public void Set(string city, WeatherData data, TimeSpan expiry)
    {
        _cache[city] = (data, DateTime.UtcNow.Add(expiry));
    }
}

public class ConsoleWeatherFormatter : IWeatherFormatter
{
    public string Format(WeatherData data)
    {
        return $"{data.City}: {data.Temperature}C, {data.Condition} (as of {data.Timestamp:HH:mm})";
    }
}
```

```csharp
// --- Application Service ---
public class WeatherApp
{
    private readonly IWeatherProvider _provider;
    private readonly IWeatherCache _cache;
    private readonly IWeatherFormatter _formatter;
    private readonly CacheOptions _cacheOptions;

    public WeatherApp(
        IWeatherProvider provider,
        IWeatherCache cache,
        IWeatherFormatter formatter,
        IOptions<CacheOptions> cacheOptions)
    {
        _provider = provider;
        _cache = cache;
        _formatter = formatter;
        _cacheOptions = cacheOptions.Value;
    }

    public async Task<string> GetWeatherReportAsync(string city)
    {
        WeatherData? data = _cache.Get(city);

        if (data is null)
        {
            data = await _provider.GetCurrentWeatherAsync(city);
            _cache.Set(city, data, TimeSpan.FromMinutes(_cacheOptions.ExpirationMinutes));
        }

        return _formatter.Format(data);
    }
}
```

```csharp
// --- Composition Root (Program.cs) ---
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

var services = new ServiceCollection();

// Configuration
services.Configure<WeatherApiOptions>(config.GetSection("WeatherApi"));
services.Configure<CacheOptions>(config.GetSection("Cache"));

// Services
services.AddHttpClient<IWeatherProvider, HttpWeatherProvider>();
services.AddSingleton<IWeatherCache, InMemoryWeatherCache>();
services.AddTransient<IWeatherFormatter, ConsoleWeatherFormatter>();
services.AddTransient<WeatherApp>();

var provider = services.BuildServiceProvider();

// Run the application
var app = provider.GetRequiredService<WeatherApp>();
string[] cities = { "Seattle", "London", "Tokyo", "Sydney" };

foreach (string city in cities)
{
    string report = await app.GetWeatherReportAsync(city);
    Console.WriteLine(report);
}
```

## 11. Practice Problems

1. **Plugin System**: Design an `IPlugin` interface with `string Name`, `int Priority`, and `Task ExecuteAsync()`. Register 5 different plugin implementations with varying priorities. Create a `PluginRunner` service that resolves all `IPlugin` instances via `IEnumerable<IPlugin>`, sorts them by priority, and executes them sequentially. Add a configuration option to disable specific plugins by name.

2. **Service Lifetime Tracker**: Create a service `LifetimeTracker` that logs its creation and disposal (implement `IDisposable`). Register it as Transient, Scoped, and Singleton (using different interface markers like `ITransientTracker`, `IScopedTracker`, `ISingletonTracker`). Write a console program that creates 3 scopes, resolves each tracker type twice per scope, and observes the creation/disposal pattern.

3. **Configuration Hot Reload**: Build a service that reads settings from a JSON file using `IOptionsMonitor<T>`. The service should log whenever the configuration changes. Write a test that modifies the JSON file at runtime and verifies the service picks up the new values within 5 seconds.

4. **Decorator Chain**: Implement the decorator pattern for an `IMessageSender` interface. Create three decorators: `LoggingMessageSender`, `RetryMessageSender` (retries 3 times on failure), and `CircuitBreakerMessageSender` (stops sending after 5 consecutive failures). Register them as a decorator chain using factory registrations. Write a test that simulates failures and verifies each decorator's behavior.

5. **Modular Registration**: Split a hypothetical e-commerce application into 4 modules (Catalog, Orders, Payments, Notifications). Each module has its own interfaces, implementations, and options. Create an extension method for each module (`AddCatalogModule`, `AddOrdersModule`, etc.) that registers all its services. Write a composition root that assembles the full application and resolves a top-level service that depends on services from all 4 modules.
