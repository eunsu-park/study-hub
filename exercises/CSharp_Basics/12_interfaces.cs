/*
 * Exercises for Lesson 12: Interfaces
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Implement multiple interfaces — a multimedia player
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Multiple Interfaces ===");

    IPlayable[] playlist =
    {
        new Song("Bohemian Rhapsody", "Queen", TimeSpan.FromMinutes(5.55)),
        new Podcast("Tech Today", "Episode 42", TimeSpan.FromMinutes(45)),
        new Song("Hotel California", "Eagles", TimeSpan.FromMinutes(6.30)),
    };

    foreach (var item in playlist)
    {
        item.Play();
        Console.WriteLine($"  Duration: {item.Duration}");
        item.Pause();
        Console.WriteLine();
    }

    // ISearchable
    var searchables = playlist.OfType<ISearchable>().ToList();
    string query = "tech";
    foreach (var item in searchables)
        Console.WriteLine($"Matches \"{query}\": {item.Matches(query)} — {item}");
    Console.WriteLine();
}

// Exercise 2: Strategy pattern with interfaces
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Strategy Pattern — Sorting ===");

    var students = new List<Student>
    {
        new("Alice", 22, 3.8),
        new("Bob", 20, 3.2),
        new("Charlie", 23, 3.9),
        new("Diana", 21, 2.7),
        new("Eve", 20, 3.5)
    };

    ISortStrategy<Student>[] strategies =
    {
        new SortByName(),
        new SortByAge(),
        new SortByGpa()
    };

    foreach (var strategy in strategies)
    {
        var sorted = strategy.Sort(students);
        Console.WriteLine($"{strategy.Name}:");
        foreach (var s in sorted)
            Console.WriteLine($"  {s.Name,-10} age={s.Age} GPA={s.Gpa}");
        Console.WriteLine();
    }
}

// Exercise 3: Default interface methods
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Default Interface Methods ===");

    ILogger[] loggers = { new ConsoleLogger(), new FileLogger("app.log") };

    foreach (var logger in loggers)
    {
        Console.WriteLine($"Using {logger.GetType().Name}:");
        logger.LogInfo("Application started");
        logger.LogWarning("Low memory");
        logger.LogError("Connection failed");
        logger.Log(LogLevel.Debug, "Debug info here");
        Console.WriteLine();
    }
}

// Exercise 4: Interface-based dependency injection pattern
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Dependency Injection ===");

    // Swap implementations without changing the service
    IRepository<string> memoryRepo = new InMemoryRepository<string>();
    var service1 = new DataService<string>(memoryRepo);

    service1.Add("Hello");
    service1.Add("World");
    service1.Add("CSharp");

    Console.WriteLine($"InMemoryRepository — All items: [{string.Join(", ", service1.GetAll())}]");
    Console.WriteLine($"  GetById(1): {service1.GetById(1)}");
    Console.WriteLine($"  Count: {service1.Count}");

    service1.Remove(0);
    Console.WriteLine($"  After remove(0): [{string.Join(", ", service1.GetAll())}]");
    Console.WriteLine();
}

// Exercise 5: IComparable and IEquatable — a Priority Queue
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Priority Queue with IComparable ===");

    var tasks = new List<PriorityTask>
    {
        new("Fix critical bug", Priority.Critical),
        new("Update docs", Priority.Low),
        new("Deploy hotfix", Priority.High),
        new("Code review", Priority.Medium),
        new("Server crash", Priority.Critical),
        new("Add logging", Priority.Medium)
    };

    Console.WriteLine("Unsorted tasks:");
    foreach (var task in tasks)
        Console.WriteLine($"  [{task.Level}] {task.Description}");

    tasks.Sort();
    Console.WriteLine("\nSorted by priority (critical first):");
    foreach (var task in tasks)
        Console.WriteLine($"  [{task.Level}] {task.Description}");

    // Equality check
    var t1 = new PriorityTask("Fix bug", Priority.High);
    var t2 = new PriorityTask("Fix bug", Priority.High);
    var t3 = new PriorityTask("Fix bug", Priority.Low);
    Console.WriteLine($"\nt1 == t2 (same desc & priority): {t1.Equals(t2)}");
    Console.WriteLine($"t1 == t3 (same desc, diff priority): {t1.Equals(t3)}");
    Console.WriteLine();
}

// Supporting types

interface IPlayable
{
    string Title { get; }
    TimeSpan Duration { get; }
    void Play();
    void Pause();
}

interface ISearchable
{
    bool Matches(string query);
}

class Song : IPlayable, ISearchable
{
    public string Title { get; }
    public string Artist { get; }
    public TimeSpan Duration { get; }
    public Song(string title, string artist, TimeSpan duration) { Title = title; Artist = artist; Duration = duration; }
    public void Play() => Console.WriteLine($"  Playing song: {Title} by {Artist}");
    public void Pause() => Console.WriteLine($"  Paused: {Title}");
    public bool Matches(string query) =>
        Title.Contains(query, StringComparison.OrdinalIgnoreCase) ||
        Artist.Contains(query, StringComparison.OrdinalIgnoreCase);
    public override string ToString() => $"Song: {Title} by {Artist}";
}

class Podcast : IPlayable, ISearchable
{
    public string Title { get; }
    public string Episode { get; }
    public TimeSpan Duration { get; }
    public Podcast(string title, string episode, TimeSpan duration) { Title = title; Episode = episode; Duration = duration; }
    public void Play() => Console.WriteLine($"  Playing podcast: {Title} — {Episode}");
    public void Pause() => Console.WriteLine($"  Paused: {Title}");
    public bool Matches(string query) =>
        Title.Contains(query, StringComparison.OrdinalIgnoreCase) ||
        Episode.Contains(query, StringComparison.OrdinalIgnoreCase);
    public override string ToString() => $"Podcast: {Title} — {Episode}";
}

record Student(string Name, int Age, double Gpa);

interface ISortStrategy<T>
{
    string Name { get; }
    IEnumerable<T> Sort(IEnumerable<T> items);
}

class SortByName : ISortStrategy<Student>
{
    public string Name => "Sort by Name";
    public IEnumerable<Student> Sort(IEnumerable<Student> items) => items.OrderBy(s => s.Name);
}

class SortByAge : ISortStrategy<Student>
{
    public string Name => "Sort by Age";
    public IEnumerable<Student> Sort(IEnumerable<Student> items) => items.OrderBy(s => s.Age);
}

class SortByGpa : ISortStrategy<Student>
{
    public string Name => "Sort by GPA (desc)";
    public IEnumerable<Student> Sort(IEnumerable<Student> items) => items.OrderByDescending(s => s.Gpa);
}

enum LogLevel { Debug, Info, Warning, Error }

interface ILogger
{
    void Log(LogLevel level, string message);
    void LogInfo(string message) => Log(LogLevel.Info, message);
    void LogWarning(string message) => Log(LogLevel.Warning, message);
    void LogError(string message) => Log(LogLevel.Error, message);
}

class ConsoleLogger : ILogger
{
    public void Log(LogLevel level, string message) =>
        Console.WriteLine($"    [{level,-7}] {message}");
}

class FileLogger : ILogger
{
    private readonly string _path;
    public FileLogger(string path) => _path = path;
    public void Log(LogLevel level, string message) =>
        Console.WriteLine($"    [{level,-7}] -> {_path}: {message}");
}

interface IRepository<T>
{
    void Add(T item);
    T? GetById(int id);
    IEnumerable<T> GetAll();
    bool Remove(int id);
    int Count { get; }
}

class InMemoryRepository<T> : IRepository<T>
{
    private readonly List<T> _items = new();
    public void Add(T item) => _items.Add(item);
    public T? GetById(int id) => id >= 0 && id < _items.Count ? _items[id] : default;
    public IEnumerable<T> GetAll() => _items.AsReadOnly();
    public bool Remove(int id)
    {
        if (id < 0 || id >= _items.Count) return false;
        _items.RemoveAt(id);
        return true;
    }
    public int Count => _items.Count;
}

class DataService<T>
{
    private readonly IRepository<T> _repo;
    public DataService(IRepository<T> repo) => _repo = repo;
    public void Add(T item) => _repo.Add(item);
    public T? GetById(int id) => _repo.GetById(id);
    public IEnumerable<T> GetAll() => _repo.GetAll();
    public bool Remove(int id) => _repo.Remove(id);
    public int Count => _repo.Count;
}

enum Priority { Critical = 0, High = 1, Medium = 2, Low = 3 }

class PriorityTask : IComparable<PriorityTask>, IEquatable<PriorityTask>
{
    public string Description { get; }
    public Priority Level { get; }
    public PriorityTask(string desc, Priority level) { Description = desc; Level = level; }
    public int CompareTo(PriorityTask? other) => other is null ? 1 : Level.CompareTo(other.Level);
    public bool Equals(PriorityTask? other) => other is not null && Description == other.Description && Level == other.Level;
    public override bool Equals(object? obj) => Equals(obj as PriorityTask);
    public override int GetHashCode() => HashCode.Combine(Description, Level);
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
