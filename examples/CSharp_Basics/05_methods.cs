// Lesson 05: Methods
// Run: dotnet run

using System;

// =============================================================================
// BASIC METHODS
// =============================================================================
Console.WriteLine("=== Basic Methods ===");

int sum = Add(3, 5);
Console.WriteLine($"Add(3, 5) = {sum}");

Greet("Alice");
Greet("Bob");

// =============================================================================
// DEFAULT PARAMETERS AND NAMED ARGUMENTS
// =============================================================================
Console.WriteLine("\n=== Default Parameters & Named Arguments ===");

PrintInfo("Alice");                        // Uses defaults
PrintInfo("Bob", age: 25);                 // Named argument
PrintInfo("Charlie", 30, "Engineer");      // Positional
PrintInfo(occupation: "Designer", name: "Diana"); // Named, reordered

// =============================================================================
// REF, OUT, AND IN PARAMETERS
// =============================================================================
Console.WriteLine("\n=== ref, out, in Parameters ===");

// ref — passes by reference; must be initialized before call
int value = 10;
Console.WriteLine($"Before DoubleIt: {value}");
DoubleIt(ref value);
Console.WriteLine($"After DoubleIt:  {value}");

// out — passes by reference; must be assigned inside the method
if (TryDivide(10, 3, out double quotient, out double remainder))
{
    Console.WriteLine($"10 / 3 = {quotient:F2}, remainder = {remainder:F2}");
}

// Inline out variable declaration
if (int.TryParse("42", out int parsed))
{
    Console.WriteLine($"Parsed: {parsed}");
}

// in — passes by read-only reference (no copies, cannot modify)
var largeStruct = new LargeData { X = 1.0, Y = 2.0, Z = 3.0 };
double mag = Magnitude(in largeStruct);
Console.WriteLine($"Magnitude: {mag:F4}");

// =============================================================================
// METHOD OVERLOADING
// =============================================================================
Console.WriteLine("\n=== Method Overloading ===");

Console.WriteLine($"Area(5):      {Area(5):F2}");         // Circle
Console.WriteLine($"Area(4, 6):   {Area(4, 6):F2}");      // Rectangle
Console.WriteLine($"Area(3,4,5):  {Area(3, 4, 5):F2}");   // Triangle (Heron)

// =============================================================================
// PARAMS (VARIABLE-LENGTH ARGUMENTS)
// =============================================================================
Console.WriteLine("\n=== Params ===");

Console.WriteLine($"Sum():          {Sum()}");
Console.WriteLine($"Sum(1):         {Sum(1)}");
Console.WriteLine($"Sum(1,2,3):     {Sum(1, 2, 3)}");
Console.WriteLine($"Sum(1..10):     {Sum(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)}");

// =============================================================================
// TUPLES AS RETURN VALUES
// =============================================================================
Console.WriteLine("\n=== Tuple Returns ===");

var (min, max, avg) = GetStats(new[] { 5, 2, 8, 1, 9, 3, 7 });
Console.WriteLine($"Stats: min={min}, max={max}, avg={avg:F2}");

// Named tuples
var person = GetPerson();
Console.WriteLine($"Person: {person.Name}, Age {person.Age}");

// =============================================================================
// LOCAL FUNCTIONS
// =============================================================================
Console.WriteLine("\n=== Local Functions ===");

int[] numbers = { 12, 45, 7, 23, 56, 89, 34 };
var sorted = BubbleSort(numbers);
Console.WriteLine($"Sorted: [{string.Join(", ", sorted)}]");

// Local function example: validation within a method
string result = ProcessOrder("Widget", 5);
Console.WriteLine(result);

// =============================================================================
// RECURSION
// =============================================================================
Console.WriteLine("\n=== Recursion ===");

Console.WriteLine($"Factorial(5) = {Factorial(5)}");
Console.WriteLine($"Factorial(10) = {Factorial(10)}");

Console.WriteLine($"Fibonacci(10) = {Fibonacci(10)}");

// Print first 15 Fibonacci numbers
Console.Write("Fibonacci sequence: ");
for (int i = 0; i < 15; i++)
{
    Console.Write($"{Fibonacci(i)} ");
}
Console.WriteLine();

// =============================================================================
// EXPRESSION-BODIED METHODS
// =============================================================================
Console.WriteLine("\n=== Expression-Bodied Methods ===");

Console.WriteLine($"Square(7) = {Square(7)}");
Console.WriteLine($"IsEven(4) = {IsEven(4)}");
Console.WriteLine($"IsEven(7) = {IsEven(7)}");
Console.WriteLine($"Max(3, 8) = {Max(3, 8)}");

// =============================================================================
// METHOD DEFINITIONS
// =============================================================================

// Basic methods
static int Add(int a, int b) => a + b;
static void Greet(string name) => Console.WriteLine($"  Hello, {name}!");

// Default parameters
static void PrintInfo(string name, int age = 0, string occupation = "Unknown")
{
    Console.WriteLine($"  Name: {name}, Age: {age}, Occupation: {occupation}");
}

// ref parameter
static void DoubleIt(ref int x) => x *= 2;

// out parameters
static bool TryDivide(double a, double b, out double quotient, out double remainder)
{
    if (b == 0)
    {
        quotient = 0;
        remainder = 0;
        return false;
    }
    quotient = a / b;
    remainder = a % b;
    return true;
}

// in parameter (read-only reference)
static double Magnitude(in LargeData data)
{
    // data.X = 0; // Compile error: cannot modify 'in' parameter
    return Math.Sqrt(data.X * data.X + data.Y * data.Y + data.Z * data.Z);
}

// Overloaded methods
static double Area(double radius) => Math.PI * radius * radius;
static double Area(double width, double height) => width * height;
static double Area(double a, double b, double c)
{
    // Heron's formula for triangle area
    double s = (a + b + c) / 2;
    return Math.Sqrt(s * (s - a) * (s - b) * (s - c));
}

// Params
static int Sum(params int[] numbers)
{
    int total = 0;
    foreach (int n in numbers) total += n;
    return total;
}

// Tuple returns
static (int Min, int Max, double Average) GetStats(int[] data)
{
    int min = data[0], max = data[0];
    double sum = 0;
    foreach (int n in data)
    {
        if (n < min) min = n;
        if (n > max) max = n;
        sum += n;
    }
    return (min, max, sum / data.Length);
}

static (string Name, int Age) GetPerson() => ("Alice", 30);

// Local functions inside a method
static int[] BubbleSort(int[] input)
{
    int[] arr = (int[])input.Clone();

    // Local function: only visible inside BubbleSort
    void Swap(int i, int j)
    {
        (arr[i], arr[j]) = (arr[j], arr[i]);
    }

    for (int i = 0; i < arr.Length - 1; i++)
    {
        for (int j = 0; j < arr.Length - 1 - i; j++)
        {
            if (arr[j] > arr[j + 1])
                Swap(j, j + 1);
        }
    }
    return arr;
}

static string ProcessOrder(string item, int quantity)
{
    // Local function for validation
    bool IsValid(string name, int qty)
        => !string.IsNullOrEmpty(name) && qty > 0 && qty <= 100;

    if (!IsValid(item, quantity))
        return "Invalid order.";

    return $"Order placed: {quantity}x {item}";
}

// Recursion
static long Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);

static int Fibonacci(int n)
{
    if (n <= 0) return 0;
    if (n == 1) return 1;
    return Fibonacci(n - 1) + Fibonacci(n - 2);
}

// Expression-bodied methods
static int Square(int x) => x * x;
static bool IsEven(int x) => x % 2 == 0;
static int Max(int a, int b) => a > b ? a : b;

// Helper struct for 'in' parameter demo
struct LargeData
{
    public double X, Y, Z;
}
