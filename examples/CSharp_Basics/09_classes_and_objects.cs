// Lesson 09: Classes and Objects
// Run: dotnet run

using System;
using System.Collections.Generic;

// =============================================================================
// BASIC CLASS USAGE
// =============================================================================
Console.WriteLine("=== Basic Class ===");

var p1 = new Person("Alice", 30);
var p2 = new Person("Bob", 25);

Console.WriteLine(p1);
Console.WriteLine(p2);
p1.Greet();

// =============================================================================
// CONSTRUCTORS: DEFAULT, PARAMETERIZED, CHAINING
// =============================================================================
Console.WriteLine("\n=== Constructors ===");

var r1 = new Rectangle();          // Default constructor
var r2 = new Rectangle(5.0, 3.0);  // Parameterized constructor
var r3 = new Rectangle(4.0);       // Square constructor (chains to parameterized)

Console.WriteLine($"r1 (default): {r1}");
Console.WriteLine($"r2 (5x3):     {r2}");
Console.WriteLine($"r3 (square):  {r3}");

// =============================================================================
// ACCESS MODIFIERS
// =============================================================================
Console.WriteLine("\n=== Access Modifiers ===");

var account = new BankAccount("Alice", 1000);
Console.WriteLine(account);

account.Deposit(500);
account.Withdraw(200);
Console.WriteLine($"Balance after transactions: {account.GetBalance():C}");

// account.balance — not accessible (private)
// account._accountId — not accessible (private)

// =============================================================================
// STATIC MEMBERS
// =============================================================================
Console.WriteLine("\n=== Static Members ===");

// Static field tracks all instances
var a1 = new BankAccount("Bob", 500);
var a2 = new BankAccount("Charlie", 750);
Console.WriteLine($"Total accounts created: {BankAccount.TotalAccounts}");

// Static method
double celsius = Temperature.FahrenheitToCelsius(98.6);
Console.WriteLine($"98.6°F = {celsius:F2}°C");

double fahrenheit = Temperature.CelsiusToFahrenheit(37);
Console.WriteLine($"37°C = {fahrenheit:F2}°F");

// =============================================================================
// THIS KEYWORD
// =============================================================================
Console.WriteLine("\n=== This Keyword ===");

var builder = new QueryBuilder()
    .Select("name, age")
    .From("users")
    .Where("age > 18")
    .OrderBy("name");

Console.WriteLine($"Query: {builder.Build()}");

// =============================================================================
// OBJECT INITIALIZER SYNTAX
// =============================================================================
Console.WriteLine("\n=== Object Initializer ===");

var config = new AppConfig
{
    AppName = "MyApp",
    Version = "2.1.0",
    MaxRetries = 3,
    Timeout = TimeSpan.FromSeconds(30)
};
Console.WriteLine(config);

// =============================================================================
// NESTED CLASSES
// =============================================================================
Console.WriteLine("\n=== Nested Classes ===");

var list = new SimpleLinkedList<int>();
list.Add(10);
list.Add(20);
list.Add(30);
Console.WriteLine($"LinkedList: {list}");
Console.WriteLine($"Count: {list.Count}");

// =============================================================================
// PARTIAL CLASSES (concept demo — normally split across files)
// =============================================================================
Console.WriteLine("\n=== Partial Classes ===");

var calc = new Calculator();
Console.WriteLine($"Add(3, 5): {calc.Add(3, 5)}");
Console.WriteLine($"Multiply(3, 5): {calc.Multiply(3, 5)}");

// =============================================================================
// RECORD TYPES (C# 9+)
// =============================================================================
Console.WriteLine("\n=== Records ===");

// Records provide value-based equality, immutability, and concise syntax
var student1 = new Student("Alice", "CS101", 3.8);
var student2 = new Student("Alice", "CS101", 3.8);
var student3 = student1 with { GPA = 3.9 }; // Non-destructive mutation

Console.WriteLine($"student1: {student1}");
Console.WriteLine($"student3 (mutated): {student3}");
Console.WriteLine($"student1 == student2: {student1 == student2}");  // true (value equality)
Console.WriteLine($"student1 == student3: {student1 == student3}");  // false

// Deconstruct
var (name, course, gpa) = student1;
Console.WriteLine($"Deconstructed: {name}, {course}, {gpa}");

// =============================================================================
// CLASS DEFINITIONS
// =============================================================================

/// <summary>
/// Basic class with constructor and methods.
/// </summary>
class Person
{
    // Fields (private by default in convention)
    private readonly string _name;
    private readonly int _age;

    // Constructor
    public Person(string name, int age)
    {
        _name = name;
        _age = age;
    }

    // Method
    public void Greet()
    {
        Console.WriteLine($"  Hi, I'm {_name} and I'm {_age} years old.");
    }

    // Override ToString for readable output
    public override string ToString() => $"Person({_name}, age {_age})";
}

/// <summary>
/// Demonstrates constructor chaining and multiple constructors.
/// </summary>
class Rectangle
{
    public double Width { get; }
    public double Height { get; }

    // Default constructor
    public Rectangle() : this(1.0, 1.0) { }

    // Square constructor chains to the full constructor
    public Rectangle(double side) : this(side, side) { }

    // Primary constructor
    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }

    public double Area() => Width * Height;
    public double Perimeter() => 2 * (Width + Height);

    public override string ToString()
        => $"Rectangle({Width}x{Height}, area={Area()}, perimeter={Perimeter()})";
}

/// <summary>
/// Demonstrates access modifiers and encapsulation.
/// </summary>
class BankAccount
{
    // Private fields — not accessible outside the class
    private static int _nextId = 1;
    private readonly int _accountId;
    private decimal _balance;

    // Public static property — accessible without an instance
    public static int TotalAccounts { get; private set; }

    // Public read-only property
    public string Owner { get; }

    // Constructor
    public BankAccount(string owner, decimal initialBalance)
    {
        _accountId = _nextId++;
        Owner = owner;
        _balance = initialBalance;
        TotalAccounts++;
    }

    // Public methods — controlled access to private state
    public decimal GetBalance() => _balance;

    public void Deposit(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("Deposit amount must be positive.");
        _balance += amount;
    }

    public bool Withdraw(decimal amount)
    {
        if (amount <= 0 || amount > _balance)
            return false;
        _balance -= amount;
        return true;
    }

    public override string ToString()
        => $"Account #{_accountId} ({Owner}): {_balance:C}";
}

/// <summary>
/// Demonstrates static methods (utility class pattern).
/// </summary>
static class Temperature
{
    public static double FahrenheitToCelsius(double f) => (f - 32) * 5.0 / 9.0;
    public static double CelsiusToFahrenheit(double c) => c * 9.0 / 5.0 + 32;
}

/// <summary>
/// Demonstrates fluent interface using 'this' keyword.
/// </summary>
class QueryBuilder
{
    private string _select = "*";
    private string _from = "";
    private string _where = "";
    private string _orderBy = "";

    public QueryBuilder Select(string columns) { _select = columns; return this; }
    public QueryBuilder From(string table)     { _from = table; return this; }
    public QueryBuilder Where(string condition) { _where = condition; return this; }
    public QueryBuilder OrderBy(string column)  { _orderBy = column; return this; }

    public string Build()
    {
        var query = $"SELECT {_select} FROM {_from}";
        if (!string.IsNullOrEmpty(_where)) query += $" WHERE {_where}";
        if (!string.IsNullOrEmpty(_orderBy)) query += $" ORDER BY {_orderBy}";
        return query;
    }
}

/// <summary>
/// Demonstrates object initializer pattern.
/// </summary>
class AppConfig
{
    public string AppName { get; set; } = "DefaultApp";
    public string Version { get; set; } = "1.0.0";
    public int MaxRetries { get; set; } = 1;
    public TimeSpan Timeout { get; set; } = TimeSpan.FromSeconds(10);

    public override string ToString()
        => $"Config({AppName} v{Version}, retries={MaxRetries}, timeout={Timeout.TotalSeconds}s)";
}

/// <summary>
/// Demonstrates nested class (Node is internal to the list).
/// </summary>
class SimpleLinkedList<T>
{
    // Nested class — only meaningful inside SimpleLinkedList
    private class Node
    {
        public T Value { get; }
        public Node? Next { get; set; }

        public Node(T value) { Value = value; }
    }

    private Node? _head;
    public int Count { get; private set; }

    public void Add(T value)
    {
        var newNode = new Node(value);
        if (_head == null)
        {
            _head = newNode;
        }
        else
        {
            var current = _head;
            while (current.Next != null)
                current = current.Next;
            current.Next = newNode;
        }
        Count++;
    }

    public override string ToString()
    {
        var parts = new List<string>();
        var current = _head;
        while (current != null)
        {
            parts.Add(current.Value?.ToString() ?? "null");
            current = current.Next;
        }
        return string.Join(" -> ", parts);
    }
}

// Partial classes — in practice these would be in separate files
partial class Calculator
{
    public int Add(int a, int b) => a + b;
    public int Subtract(int a, int b) => a - b;
}

partial class Calculator
{
    public int Multiply(int a, int b) => a * b;
    public double Divide(double a, double b) => b != 0 ? a / b : throw new DivideByZeroException();
}

// Record type — immutable reference type with value semantics
record Student(string Name, string Course, double GPA);
