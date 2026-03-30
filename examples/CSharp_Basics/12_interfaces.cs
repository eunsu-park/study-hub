// Lesson 12: Interfaces
// Run: dotnet run

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

// =============================================================================
// BASIC INTERFACE IMPLEMENTATION
// =============================================================================
Console.WriteLine("=== Basic Interface ===");

IShape circle = new Circle(5.0);
IShape rect = new Rectangle(4.0, 6.0);

Console.WriteLine($"Circle:    area={circle.Area():F2}, perimeter={circle.Perimeter():F2}");
Console.WriteLine($"Rectangle: area={rect.Area():F2}, perimeter={rect.Perimeter():F2}");

// Interface reference allows polymorphism
IShape[] shapes = { circle, rect, new Triangle(3.0, 4.0, 5.0) };
foreach (IShape shape in shapes)
{
    Console.WriteLine($"  {shape.GetType().Name}: {shape.Describe()}");
}

// =============================================================================
// MULTIPLE INTERFACE IMPLEMENTATION
// =============================================================================
Console.WriteLine("\n=== Multiple Interfaces ===");

var doc = new Document("report.pdf", 2048);
Console.WriteLine($"Document: {doc.Name}");

// Use as IPrintable
IPrintable printable = doc;
printable.Print();

// Use as ISaveable
ISaveable saveable = doc;
saveable.Save("/tmp/report.pdf");

// Use as ISearchable
ISearchable searchable = doc;
Console.WriteLine($"Search 'summary': {searchable.Search("summary")}");

// =============================================================================
// DEFAULT INTERFACE METHODS (C# 8+)
// =============================================================================
Console.WriteLine("\n=== Default Interface Methods ===");

ILogger consoleLogger = new ConsoleLogger();
ILogger fileLogger = new FileLogger("app.log");

consoleLogger.Log("Application started.");
consoleLogger.LogWarning("Low memory.");  // Default implementation
consoleLogger.LogError("Crash!");          // Default implementation

fileLogger.Log("Writing to file.");
fileLogger.LogWarning("Disk space low.");

// =============================================================================
// IComparable<T> — Sorting Custom Objects
// =============================================================================
Console.WriteLine("\n=== IComparable<T> ===");

var students = new List<Student>
{
    new Student("Charlie", 3.5),
    new Student("Alice", 3.9),
    new Student("Bob", 3.7),
    new Student("Diana", 3.2),
    new Student("Eve", 3.9)
};

students.Sort(); // Uses IComparable<Student>.CompareTo

Console.WriteLine("Students sorted by GPA (desc), then name:");
foreach (var s in students)
{
    Console.WriteLine($"  {s.Name}: GPA {s.GPA:F1}");
}

// =============================================================================
// IEnumerable<T> — Making Custom Types Iterable
// =============================================================================
Console.WriteLine("\n=== IEnumerable<T> ===");

var range = new NumberRange(1, 10);

Console.Write("NumberRange(1, 10): ");
foreach (int n in range)
{
    Console.Write($"{n} ");
}
Console.WriteLine();

// Works with LINQ because it implements IEnumerable<T>
var evens = range.Where(n => n % 2 == 0);
Console.Write("Evens: ");
foreach (int n in evens)
{
    Console.Write($"{n} ");
}
Console.WriteLine();

Console.WriteLine($"Sum: {range.Sum()}");
Console.WriteLine($"Count: {range.Count()}");

// =============================================================================
// IEquatable<T> — Value-Based Equality
// =============================================================================
Console.WriteLine("\n=== IEquatable<T> ===");

var p1 = new Point(3, 4);
var p2 = new Point(3, 4);
var p3 = new Point(1, 2);

Console.WriteLine($"p1 = {p1}");
Console.WriteLine($"p2 = {p2}");
Console.WriteLine($"p3 = {p3}");
Console.WriteLine($"p1.Equals(p2): {p1.Equals(p2)}");
Console.WriteLine($"p1.Equals(p3): {p1.Equals(p3)}");
Console.WriteLine($"p1 == p2: {p1 == p2}");

// Works correctly in HashSet/Dictionary
var pointSet = new HashSet<Point> { p1, p2, p3 };
Console.WriteLine($"HashSet count (p1, p2 are equal): {pointSet.Count}");

// =============================================================================
// EXPLICIT INTERFACE IMPLEMENTATION
// =============================================================================
Console.WriteLine("\n=== Explicit Interface Implementation ===");

var widget = new Widget();

// Must cast to the specific interface to call explicitly implemented methods
IClickable clickable = widget;
IDraggable draggable = widget;

clickable.Activate();  // Calls IClickable.Activate
draggable.Activate();  // Calls IDraggable.Activate

// widget.Activate(); // Compile error: ambiguous without cast

// =============================================================================
// GENERIC INTERFACES
// =============================================================================
Console.WriteLine("\n=== Generic Interface (IRepository<T>) ===");

var repo = new InMemoryRepository<string>();
repo.Add("alpha");
repo.Add("beta");
repo.Add("gamma");

Console.WriteLine($"Count: {repo.Count}");
Console.WriteLine($"GetById(1): {repo.GetById(1)}");

var all = repo.GetAll();
Console.WriteLine($"All: [{string.Join(", ", all)}]");

repo.Delete(0);
Console.WriteLine($"After delete(0): [{string.Join(", ", repo.GetAll())}]");

// =============================================================================
// INTERFACE DEFINITIONS
// =============================================================================

interface IShape
{
    double Area();
    double Perimeter();

    // Default method (C# 8+)
    string Describe() => $"{GetType().Name}: area={Area():F2}, perim={Perimeter():F2}";
}

interface IPrintable
{
    void Print();
}

interface ISaveable
{
    void Save(string path);
}

interface ISearchable
{
    bool Search(string query);
}

interface ILogger
{
    void Log(string message);

    // Default interface methods — implementations can override or use defaults
    void LogWarning(string message) => Log($"[WARN] {message}");
    void LogError(string message) => Log($"[ERROR] {message}");
}

interface IClickable
{
    void Activate();
}

interface IDraggable
{
    void Activate();
}

interface IRepository<T>
{
    void Add(T item);
    T? GetById(int id);
    IEnumerable<T> GetAll();
    void Delete(int id);
    int Count { get; }
}

// =============================================================================
// CLASS IMPLEMENTATIONS
// =============================================================================

class Circle : IShape
{
    public double Radius { get; }
    public Circle(double radius) => Radius = radius;

    public double Area() => Math.PI * Radius * Radius;
    public double Perimeter() => 2 * Math.PI * Radius;
}

class Rectangle : IShape
{
    public double Width { get; }
    public double Height { get; }

    public Rectangle(double w, double h) { Width = w; Height = h; }

    public double Area() => Width * Height;
    public double Perimeter() => 2 * (Width + Height);
}

class Triangle : IShape
{
    public double A { get; }
    public double B { get; }
    public double C { get; }

    public Triangle(double a, double b, double c) { A = a; B = b; C = c; }

    public double Area()
    {
        double s = (A + B + C) / 2;
        return Math.Sqrt(s * (s - A) * (s - B) * (s - C));
    }
    public double Perimeter() => A + B + C;
}

/// <summary>
/// Implements multiple interfaces.
/// </summary>
class Document : IPrintable, ISaveable, ISearchable
{
    public string Name { get; }
    public int SizeBytes { get; }

    public Document(string name, int size) { Name = name; SizeBytes = size; }

    public void Print() => Console.WriteLine($"  Printing '{Name}' ({SizeBytes} bytes)...");
    public void Save(string path) => Console.WriteLine($"  Saving '{Name}' to {path}...");
    public bool Search(string query) => Name.Contains(query, StringComparison.OrdinalIgnoreCase);
}

class ConsoleLogger : ILogger
{
    public void Log(string message) => Console.WriteLine($"  [Console] {message}");
}

class FileLogger : ILogger
{
    private readonly string _filename;
    public FileLogger(string filename) => _filename = filename;

    public void Log(string message) => Console.WriteLine($"  [File:{_filename}] {message}");
}

/// <summary>
/// IComparable: sort by GPA descending, then name ascending.
/// </summary>
class Student : IComparable<Student>
{
    public string Name { get; }
    public double GPA { get; }

    public Student(string name, double gpa) { Name = name; GPA = gpa; }

    public int CompareTo(Student? other)
    {
        if (other is null) return 1;
        int cmp = other.GPA.CompareTo(GPA); // Descending by GPA
        return cmp != 0 ? cmp : string.Compare(Name, other.Name, StringComparison.Ordinal);
    }
}

/// <summary>
/// IEnumerable: makes a custom type work with foreach and LINQ.
/// </summary>
class NumberRange : IEnumerable<int>
{
    private readonly int _start;
    private readonly int _end;

    public NumberRange(int start, int end) { _start = start; _end = end; }

    public IEnumerator<int> GetEnumerator()
    {
        for (int i = _start; i <= _end; i++)
            yield return i;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

/// <summary>
/// IEquatable: value-based equality for use in collections.
/// </summary>
class Point : IEquatable<Point>
{
    public int X { get; }
    public int Y { get; }

    public Point(int x, int y) { X = x; Y = y; }

    public bool Equals(Point? other)
        => other is not null && X == other.X && Y == other.Y;

    public override bool Equals(object? obj) => Equals(obj as Point);
    public override int GetHashCode() => HashCode.Combine(X, Y);

    public static bool operator ==(Point? a, Point? b)
        => a is null ? b is null : a.Equals(b);
    public static bool operator !=(Point? a, Point? b) => !(a == b);

    public override string ToString() => $"({X}, {Y})";
}

/// <summary>
/// Explicit interface implementation — resolves ambiguity.
/// </summary>
class Widget : IClickable, IDraggable
{
    // Explicit implementation: must be called through the interface
    void IClickable.Activate() => Console.WriteLine("  Widget clicked!");
    void IDraggable.Activate() => Console.WriteLine("  Widget dragged!");
}

/// <summary>
/// Generic interface implementation.
/// </summary>
class InMemoryRepository<T> : IRepository<T>
{
    private readonly List<T> _items = new();

    public int Count => _items.Count;

    public void Add(T item) => _items.Add(item);

    public T? GetById(int id)
        => id >= 0 && id < _items.Count ? _items[id] : default;

    public IEnumerable<T> GetAll() => _items.AsReadOnly();

    public void Delete(int id)
    {
        if (id >= 0 && id < _items.Count)
            _items.RemoveAt(id);
    }
}
