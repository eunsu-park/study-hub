// Lesson 10: Dependency Injection
// Run: dotnet run
// Note: Requires NuGet packages:
//   dotnet add package Microsoft.Extensions.DependencyInjection
//   dotnet add package Microsoft.Extensions.Options
//   dotnet add package Microsoft.Extensions.Configuration
//   dotnet add package Microsoft.Extensions.Configuration.Json

using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;

// ============================================================
// 1. Manual DI (Without Container)
// ============================================================

Console.WriteLine("=== Manual Dependency Injection ===");

// Without DI: tightly coupled
// var service = new OrderService(); // Creates its own dependencies internally

// With DI: dependencies injected via constructor
ILogger logger = new ConsoleLogger();
IRepository repository = new InMemoryRepository();
var orderService = new OrderService(repository, logger);

orderService.PlaceOrder("Widget", 3);
orderService.PlaceOrder("Gadget", 1);

// ============================================================
// 2. DI Container Setup
// ============================================================

Console.WriteLine("\n=== DI Container (ServiceCollection) ===");

// Configure the DI container
var services = new ServiceCollection();

// Register services with different lifetimes
services.AddSingleton<ILogger, ConsoleLogger>();      // One instance for the app
services.AddScoped<IRepository, InMemoryRepository>(); // One per scope
services.AddTransient<OrderService>();                  // New instance each time

// Build the service provider
var provider = services.BuildServiceProvider();

// Resolve services
var resolvedService = provider.GetRequiredService<OrderService>();
resolvedService.PlaceOrder("Keyboard", 2);

// ============================================================
// 3. Service Lifetimes: Singleton vs Scoped vs Transient
// ============================================================

Console.WriteLine("\n=== Service Lifetimes ===");

var lifetimeServices = new ServiceCollection();
lifetimeServices.AddSingleton<SingletonService>();
lifetimeServices.AddScoped<ScopedService>();
lifetimeServices.AddTransient<TransientService>();
lifetimeServices.AddTransient<LifetimeDemo>();

var lifetimeProvider = lifetimeServices.BuildServiceProvider();

// Scope 1
Console.WriteLine("  --- Scope 1 ---");
using (var scope = lifetimeProvider.CreateScope())
{
    var demo1 = scope.ServiceProvider.GetRequiredService<LifetimeDemo>();
    var demo2 = scope.ServiceProvider.GetRequiredService<LifetimeDemo>();
    demo1.PrintIds("  Request 1");
    demo2.PrintIds("  Request 2");
}

// Scope 2 — scoped service gets a new instance
Console.WriteLine("  --- Scope 2 ---");
using (var scope = lifetimeProvider.CreateScope())
{
    var demo = scope.ServiceProvider.GetRequiredService<LifetimeDemo>();
    demo.PrintIds("  Request 3");
}

// ============================================================
// 4. IOptions<T> Pattern
// ============================================================

Console.WriteLine("\n=== IOptions<T> Pattern ===");

var optionServices = new ServiceCollection();

// Configure options directly
optionServices.Configure<EmailSettings>(opts =>
{
    opts.SmtpHost = "smtp.example.com";
    opts.SmtpPort = 587;
    opts.FromAddress = "noreply@example.com";
    opts.UseSsl = true;
});

optionServices.AddTransient<EmailService>();
var optionProvider = optionServices.BuildServiceProvider();

var emailService = optionProvider.GetRequiredService<EmailService>();
emailService.SendEmail("user@test.com", "Hello", "Welcome!");

// ============================================================
// 5. Factory Registration
// ============================================================

Console.WriteLine("\n=== Factory Registration ===");

var factoryServices = new ServiceCollection();
factoryServices.AddSingleton<ILogger, ConsoleLogger>();

// Register with a factory delegate for complex construction
factoryServices.AddTransient<INotificationService>(sp =>
{
    var log = sp.GetRequiredService<ILogger>();
    bool useEmail = true; // Could come from configuration
    return useEmail
        ? new EmailNotification(log)
        : new SmsNotification(log);
});

var factoryProvider = factoryServices.BuildServiceProvider();
var notification = factoryProvider.GetRequiredService<INotificationService>();
notification.Notify("Build succeeded!");

// ============================================================
// 6. Keyed Services (C# / .NET 8+)
// ============================================================

Console.WriteLine("\n=== Multiple Implementations ===");

var multiServices = new ServiceCollection();
multiServices.AddSingleton<ILogger, ConsoleLogger>();

// Register multiple implementations of the same interface
multiServices.AddTransient<INotificationService, EmailNotification>();
multiServices.AddTransient<INotificationService, SmsNotification>();

var multiProvider = multiServices.BuildServiceProvider();

// Resolve all implementations
var allNotifications = multiProvider.GetServices<INotificationService>();
foreach (var svc in allNotifications)
    svc.Notify("Deployed v2.0");

// ============================================================
// 7. Disposable Services
// ============================================================

Console.WriteLine("\n=== Disposable Services ===");

var disposableServices = new ServiceCollection();
disposableServices.AddScoped<DisposableResource>();

var disposableProvider = disposableServices.BuildServiceProvider();

using (var scope = disposableProvider.CreateScope())
{
    var resource = scope.ServiceProvider.GetRequiredService<DisposableResource>();
    resource.DoWork();
    Console.WriteLine("  Scope ending...");
} // DisposableResource.Dispose() is called here automatically
Console.WriteLine("  Scope ended.");

// ============================================================
// Interfaces and Implementations
// ============================================================

public interface ILogger
{
    void Log(string message);
}

public class ConsoleLogger : ILogger
{
    public void Log(string message) => Console.WriteLine($"  [LOG] {message}");
}

public interface IRepository
{
    void Save(string item);
    IEnumerable<string> GetAll();
}

public class InMemoryRepository : IRepository
{
    private readonly List<string> _items = new();
    public void Save(string item) => _items.Add(item);
    public IEnumerable<string> GetAll() => _items;
}

public class OrderService
{
    private readonly IRepository _repository;
    private readonly ILogger _logger;

    // Dependencies are injected via constructor
    public OrderService(IRepository repository, ILogger logger)
    {
        _repository = repository;
        _logger = logger;
    }

    public void PlaceOrder(string product, int quantity)
    {
        var order = $"{product} x{quantity}";
        _repository.Save(order);
        _logger.Log($"Order placed: {order}");
    }
}

// Lifetime demo services
public class SingletonService { public Guid Id { get; } = Guid.NewGuid(); }
public class ScopedService { public Guid Id { get; } = Guid.NewGuid(); }
public class TransientService { public Guid Id { get; } = Guid.NewGuid(); }

public class LifetimeDemo
{
    private readonly SingletonService _singleton;
    private readonly ScopedService _scoped;
    private readonly TransientService _transient;

    public LifetimeDemo(SingletonService s, ScopedService sc, TransientService t)
    {
        _singleton = s; _scoped = sc; _transient = t;
    }

    public void PrintIds(string label)
    {
        Console.WriteLine($"{label}:");
        Console.WriteLine($"    Singleton:  {_singleton.Id.ToString()[..8]}");
        Console.WriteLine($"    Scoped:     {_scoped.Id.ToString()[..8]}");
        Console.WriteLine($"    Transient:  {_transient.Id.ToString()[..8]}");
    }
}

// Options pattern
public class EmailSettings
{
    public string SmtpHost { get; set; } = "";
    public int SmtpPort { get; set; }
    public string FromAddress { get; set; } = "";
    public bool UseSsl { get; set; }
}

public class EmailService
{
    private readonly EmailSettings _settings;

    public EmailService(IOptions<EmailSettings> options)
    {
        _settings = options.Value;
    }

    public void SendEmail(string to, string subject, string body)
    {
        Console.WriteLine($"  Sending via {_settings.SmtpHost}:{_settings.SmtpPort} (SSL={_settings.UseSsl})");
        Console.WriteLine($"  From: {_settings.FromAddress} To: {to}");
        Console.WriteLine($"  Subject: {subject}");
    }
}

// Notification services
public interface INotificationService
{
    void Notify(string message);
}

public class EmailNotification : INotificationService
{
    private readonly ILogger _logger;
    public EmailNotification(ILogger logger) => _logger = logger;
    public void Notify(string message) => _logger.Log($"[Email] {message}");
}

public class SmsNotification : INotificationService
{
    private readonly ILogger _logger;
    public SmsNotification(ILogger logger) => _logger = logger;
    public void Notify(string message) => _logger.Log($"[SMS] {message}");
}

// Disposable resource
public class DisposableResource : IDisposable
{
    public void DoWork() => Console.WriteLine("  DisposableResource doing work...");
    public void Dispose() => Console.WriteLine("  DisposableResource.Dispose() called!");
}
