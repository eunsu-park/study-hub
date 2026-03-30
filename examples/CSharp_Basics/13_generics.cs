// Lesson 13: Generics
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Linq;

// =============================================================================
// GENERIC METHODS
// =============================================================================
Console.WriteLine("=== Generic Methods ===");

// The compiler infers T from the arguments
Console.WriteLine($"Swap int:    {SwapDemo(1, 2)}");
Console.WriteLine($"Swap string: {SwapDemo("hello", "world")}");

// Identity function — works with any type
Console.WriteLine($"Identity(42):      {Identity(42)}");
Console.WriteLine($"Identity(\"abc\"):   {Identity("abc")}");

// Generic Print for arrays
int[] nums = { 1, 2, 3, 4, 5 };
string[] words = { "alpha", "beta", "gamma" };
Console.Write("int[]:    "); PrintArray(nums);
Console.Write("string[]: "); PrintArray(words);

// =============================================================================
// GENERIC CLASS — Stack<T>
// =============================================================================
Console.WriteLine("\n=== Generic Class: SimpleStack<T> ===");

var intStack = new SimpleStack<int>(5);
intStack.Push(10);
intStack.Push(20);
intStack.Push(30);
Console.WriteLine($"Int stack: {intStack}");
Console.WriteLine($"Pop: {intStack.Pop()}");
Console.WriteLine($"Peek: {intStack.Peek()}");
Console.WriteLine($"Count: {intStack.Count}");

var stringStack = new SimpleStack<string>(3);
stringStack.Push("first");
stringStack.Push("second");
Console.WriteLine($"String stack: {stringStack}");

// =============================================================================
// GENERIC CLASS — Pair<T1, T2>
// =============================================================================
Console.WriteLine("\n=== Generic Pair<T1, T2> ===");

var nameAge = new Pair<string, int>("Alice", 30);
var pointPair = new Pair<double, double>(3.14, 2.71);

Console.WriteLine($"Pair: {nameAge}");
Console.WriteLine($"Pair: {pointPair}");

// =============================================================================
// GENERIC CONSTRAINTS
// =============================================================================
Console.WriteLine("\n=== Generic Constraints ===");

// where T : struct (value type constraint)
Console.WriteLine($"WrapNullable(42):   {WrapNullable(42)}");
Console.WriteLine($"WrapNullable(3.14): {WrapNullable(3.14)}");

// where T : class (reference type constraint)
var result = CoalesceOrDefault("hello", "default");
Console.WriteLine($"Coalesce(\"hello\", \"default\"): {result}");
result = CoalesceOrDefault<string>(null, "fallback");
Console.WriteLine($"Coalesce(null, \"fallback\"): {result}");

// where T : IComparable<T> (interface constraint)
Console.WriteLine($"Max(3, 7): {GenericMax(3, 7)}");
Console.WriteLine($"Max(\"apple\", \"banana\"): {GenericMax("apple", "banana")}");

// where T : new() (parameterless constructor constraint)
var newInt = CreateDefault<int>();
var newList = CreateDefault<List<int>>();
Console.WriteLine($"CreateDefault<int>(): {newInt}");
Console.WriteLine($"CreateDefault<List<int>>(): Count={newList.Count}");

// Multiple constraints combined
var sortedRepo = new SortedRepository<Student>();
sortedRepo.Add(new Student("Charlie", 3.5));
sortedRepo.Add(new Student("Alice", 3.9));
sortedRepo.Add(new Student("Bob", 3.7));

Console.WriteLine("\nSorted repository:");
foreach (var s in sortedRepo.GetAllSorted())
{
    Console.WriteLine($"  {s}");
}

// =============================================================================
// GENERIC INTERFACE IMPLEMENTATION
// =============================================================================
Console.WriteLine("\n=== Generic Interface: ITransformer<TIn, TOut> ===");

ITransformer<string, int> lengthTransformer = new StringLengthTransformer();
ITransformer<int, string> hexTransformer = new IntToHexTransformer();

Console.WriteLine($"\"hello\" -> length: {lengthTransformer.Transform("hello")}");
Console.WriteLine($"255 -> hex: {hexTransformer.Transform(255)}");

// Using generic transformer with collections
string[] texts = { "a", "bb", "ccc", "dddd" };
int[] lengths = TransformAll(texts, lengthTransformer);
Console.Write("Lengths: ");
PrintArray(lengths);

// =============================================================================
// COVARIANCE AND CONTRAVARIANCE
// =============================================================================
Console.WriteLine("\n=== Covariance (out T) ===");

// IEnumerable<T> is covariant: IEnumerable<Derived> can be assigned to IEnumerable<Base>
IEnumerable<Animal> animals = new List<Dog>
{
    new Dog("Rex"),
    new Dog("Buddy")
};

foreach (Animal a in animals)
{
    Console.WriteLine($"  Animal: {a.Name}");
}

// Our custom covariant interface
IProducer<Animal> animalProducer = new DogProducer(); // Dog is Animal
Console.WriteLine($"Producer: {animalProducer.Produce().Name}");

Console.WriteLine("\n=== Contravariance (in T) ===");

// IComparer<T> is contravariant: IComparer<Base> can be used as IComparer<Derived>
IComparer<Animal> animalComparer = new AnimalNameComparer();
var dogs = new List<Dog> { new Dog("Zara"), new Dog("Alpha"), new Dog("Max") };
dogs.Sort(animalComparer); // Using IComparer<Animal> for List<Dog>

Console.Write("Sorted dogs: ");
Console.WriteLine(string.Join(", ", dogs.Select(d => d.Name)));

// =============================================================================
// GENERIC UTILITY — Result<T> (Success/Error Pattern)
// =============================================================================
Console.WriteLine("\n=== Result<T> Pattern ===");

var parseResult = SafeParse("42");
Console.WriteLine($"Parse \"42\": {parseResult}");

var failResult = SafeParse("abc");
Console.WriteLine($"Parse \"abc\": {failResult}");

// Using the result
if (parseResult.IsSuccess)
{
    Console.WriteLine($"Value doubled: {parseResult.Value * 2}");
}

// =============================================================================
// METHOD DEFINITIONS
// =============================================================================

static string SwapDemo<T>(T a, T b)
{
    (a, b) = (b, a);
    return $"({a}, {b})";
}

static T Identity<T>(T value) => value;

static void PrintArray<T>(T[] array)
    => Console.WriteLine($"[{string.Join(", ", array)}]");

// Constraints
static T? WrapNullable<T>(T value) where T : struct => value;

static T CoalesceOrDefault<T>(T? value, T fallback) where T : class
    => value ?? fallback;

static T GenericMax<T>(T a, T b) where T : IComparable<T>
    => a.CompareTo(b) >= 0 ? a : b;

static T CreateDefault<T>() where T : new() => new T();

static TOut[] TransformAll<TIn, TOut>(TIn[] items, ITransformer<TIn, TOut> transformer)
{
    var result = new TOut[items.Length];
    for (int i = 0; i < items.Length; i++)
        result[i] = transformer.Transform(items[i]);
    return result;
}

static Result<int> SafeParse(string input)
{
    if (int.TryParse(input, out int value))
        return Result<int>.Success(value);
    return Result<int>.Error($"Cannot parse '{input}' as int");
}

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

class SimpleStack<T>
{
    private readonly T[] _items;
    private int _top;

    public int Count => _top;
    public bool IsEmpty => _top == 0;
    public bool IsFull => _top == _items.Length;

    public SimpleStack(int capacity)
    {
        _items = new T[capacity];
        _top = 0;
    }

    public void Push(T item)
    {
        if (IsFull) throw new InvalidOperationException("Stack is full.");
        _items[_top++] = item;
    }

    public T Pop()
    {
        if (IsEmpty) throw new InvalidOperationException("Stack is empty.");
        return _items[--_top];
    }

    public T Peek()
    {
        if (IsEmpty) throw new InvalidOperationException("Stack is empty.");
        return _items[_top - 1];
    }

    public override string ToString()
    {
        var items = new T[_top];
        Array.Copy(_items, items, _top);
        Array.Reverse(items);
        return $"[{string.Join(", ", items)}] (top first)";
    }
}

class Pair<T1, T2>
{
    public T1 First { get; }
    public T2 Second { get; }

    public Pair(T1 first, T2 second) { First = first; Second = second; }
    public override string ToString() => $"({First}, {Second})";
}

// Multiple constraints: T must be comparable AND have a parameterless constructor
class SortedRepository<T> where T : IComparable<T>, new()
{
    private readonly List<T> _items = new();

    public void Add(T item) => _items.Add(item);

    public IEnumerable<T> GetAllSorted()
    {
        var sorted = new List<T>(_items);
        sorted.Sort();
        return sorted;
    }
}

class Student : IComparable<Student>
{
    public string Name { get; }
    public double GPA { get; }

    public Student() : this("Unknown", 0) { } // Required by new() constraint
    public Student(string name, double gpa) { Name = name; GPA = gpa; }

    public int CompareTo(Student? other)
        => other is null ? 1 : string.Compare(Name, other.Name, StringComparison.Ordinal);

    public override string ToString() => $"{Name} (GPA: {GPA:F1})";
}

// Generic interface
interface ITransformer<TIn, TOut>
{
    TOut Transform(TIn input);
}

class StringLengthTransformer : ITransformer<string, int>
{
    public int Transform(string input) => input.Length;
}

class IntToHexTransformer : ITransformer<int, string>
{
    public string Transform(int input) => $"0x{input:X}";
}

// Covariance demo types
class Animal
{
    public string Name { get; }
    public Animal(string name) => Name = name;
}

class Dog : Animal
{
    public Dog(string name) : base(name) { }
}

// Covariant interface (out T — can only return T)
interface IProducer<out T>
{
    T Produce();
}

class DogProducer : IProducer<Dog>
{
    public Dog Produce() => new Dog("Produced Dog");
}

// Contravariance helper
class AnimalNameComparer : IComparer<Animal>
{
    public int Compare(Animal? x, Animal? y)
        => string.Compare(x?.Name, y?.Name, StringComparison.Ordinal);
}

// Result<T> pattern — generic success/error wrapper
class Result<T>
{
    public bool IsSuccess { get; }
    public T Value { get; }
    public string ErrorMessage { get; }

    private Result(bool success, T value, string error)
    {
        IsSuccess = success;
        Value = value;
        ErrorMessage = error;
    }

    public static Result<T> Success(T value) => new(true, value, "");
    public static Result<T> Error(string message) => new(false, default!, message);

    public override string ToString()
        => IsSuccess ? $"Ok({Value})" : $"Error({ErrorMessage})";
}
