/*
 * Exercises for Lesson 10: Dependency Injection
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;

// ---------------------------------------------------------------------------
// Exercise 1: Manual DI — constructor injection
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Constructor Injection ===");

    ILogger logger = new ConsoleLogger();
    IRepository repo = new InMemoryRepository(logger);
    var service = new UserService(repo, logger);

    service.CreateUser("Alice");
    service.CreateUser("Bob");
    var user = service.GetUser("Alice");
    Console.WriteLine($"  Found: {user?.Name ?? "null"}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Simple DI container — service registration and resolution
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Simple DI Container ===");

    var container = new SimpleContainer();

    // Register services
    container.RegisterSingleton<ILogger>(() => new ConsoleLogger());
    container.RegisterTransient<IRepository>(() => new InMemoryRepository(container.Resolve<ILogger>()));
    container.RegisterTransient<UserService>(() =>
        new UserService(container.Resolve<IRepository>(), container.Resolve<ILogger>()));

    var service = container.Resolve<UserService>();
    service.CreateUser("Charlie");
    Console.WriteLine($"  Created user via DI container");

    // Verify singleton behavior
    var logger1 = container.Resolve<ILogger>();
    var logger2 = container.Resolve<ILogger>();
    Console.WriteLine($"  Singleton same instance: {ReferenceEquals(logger1, logger2)}");

    var repo1 = container.Resolve<IRepository>();
    var repo2 = container.Resolve<IRepository>();
    Console.WriteLine($"  Transient same instance: {ReferenceEquals(repo1, repo2)}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Lifetime management — transient vs singleton vs scoped
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Lifetime Demos ===");

    var container = new SimpleContainer();
    int transientCount = 0;
    int singletonCreated = 0;

    container.RegisterTransient<INotifier>(() =>
    {
        transientCount++;
        return new EmailNotifier($"Instance-{transientCount}");
    });

    container.RegisterSingleton<ICache>(() =>
    {
        singletonCreated++;
        return new MemoryCache($"Cache-{singletonCreated}");
    });

    // Transient: new instance each time
    var n1 = container.Resolve<INotifier>() as EmailNotifier;
    var n2 = container.Resolve<INotifier>() as EmailNotifier;
    Console.WriteLine($"  Transient: {n1?.Id}, {n2?.Id} (different={!ReferenceEquals(n1, n2)})");

    // Singleton: same instance
    var c1 = container.Resolve<ICache>() as MemoryCache;
    var c2 = container.Resolve<ICache>() as MemoryCache;
    Console.WriteLine($"  Singleton: {c1?.Id}, {c2?.Id} (same={ReferenceEquals(c1, c2)})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Interface segregation — multiple small interfaces
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Interface Segregation ===");

    IReadRepository readRepo = new SqlRepository();
    IWriteRepository writeRepo = new SqlRepository();

    writeRepo.Save("item-1", "Hello");
    writeRepo.Save("item-2", "World");
    string? val = readRepo.FindById("item-1");
    Console.WriteLine($"  Read item-1: {val}");
    Console.WriteLine($"  Read item-3: {readRepo.FindById("item-3") ?? "(not found)"}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Decorator pattern via DI — logging decorator
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Decorator Pattern ===");

    ILogger innerLogger = new ConsoleLogger();
    IRepository innerRepo = new InMemoryRepository(innerLogger);
    IRepository loggingRepo = new LoggingRepositoryDecorator(innerRepo, innerLogger);

    var service = new UserService(loggingRepo, innerLogger);
    service.CreateUser("Decorated-User");
    service.GetUser("Decorated-User");
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

interface ILogger { void Log(string message); }
interface IRepository { void Add(string name); UserRecord? Find(string name); }
interface INotifier { void Notify(string msg); }
interface ICache { void Set(string key, object value); }
interface IReadRepository { string? FindById(string id); }
interface IWriteRepository { void Save(string id, string data); }

record UserRecord(string Name);

class ConsoleLogger : ILogger
{
    public void Log(string message) => Console.WriteLine($"    [LOG] {message}");
}

class InMemoryRepository : IRepository
{
    private readonly Dictionary<string, UserRecord> _store = new();
    private readonly ILogger _logger;
    public InMemoryRepository(ILogger logger) => _logger = logger;
    public void Add(string name) { _store[name] = new UserRecord(name); }
    public UserRecord? Find(string name) => _store.GetValueOrDefault(name);
}

class UserService
{
    private readonly IRepository _repo;
    private readonly ILogger _logger;
    public UserService(IRepository repo, ILogger logger) { _repo = repo; _logger = logger; }
    public void CreateUser(string name) { _repo.Add(name); _logger.Log($"Created user: {name}"); }
    public UserRecord? GetUser(string name) => _repo.Find(name);
}

class EmailNotifier : INotifier
{
    public string Id { get; }
    public EmailNotifier(string id) => Id = id;
    public void Notify(string msg) => Console.WriteLine($"  Email: {msg}");
}

class MemoryCache : ICache
{
    public string Id { get; }
    public MemoryCache(string id) => Id = id;
    public void Set(string key, object value) { }
}

class SqlRepository : IReadRepository, IWriteRepository
{
    private readonly Dictionary<string, string> _data = new();
    public string? FindById(string id) => _data.GetValueOrDefault(id);
    public void Save(string id, string data) => _data[id] = data;
}

class LoggingRepositoryDecorator : IRepository
{
    private readonly IRepository _inner;
    private readonly ILogger _logger;
    public LoggingRepositoryDecorator(IRepository inner, ILogger logger)
    { _inner = inner; _logger = logger; }

    public void Add(string name)
    {
        _logger.Log($"[DECORATOR] Adding: {name}");
        _inner.Add(name);
    }

    public UserRecord? Find(string name)
    {
        _logger.Log($"[DECORATOR] Finding: {name}");
        return _inner.Find(name);
    }
}

class SimpleContainer
{
    private readonly Dictionary<Type, Func<object>> _transient = new();
    private readonly Dictionary<Type, Lazy<object>> _singleton = new();

    public void RegisterTransient<T>(Func<T> factory) where T : class =>
        _transient[typeof(T)] = () => factory();

    public void RegisterSingleton<T>(Func<T> factory) where T : class =>
        _singleton[typeof(T)] = new Lazy<object>(() => factory());

    public T Resolve<T>() where T : class
    {
        if (_singleton.TryGetValue(typeof(T), out var lazy))
            return (T)lazy.Value;
        if (_transient.TryGetValue(typeof(T), out var factory))
            return (T)factory();
        throw new InvalidOperationException($"No registration for {typeof(T).Name}");
    }
}
