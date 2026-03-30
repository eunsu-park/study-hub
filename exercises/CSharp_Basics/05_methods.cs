/*
 * Exercises for Lesson 05: Methods
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: ref and out parameters
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: ref and out Parameters ===");

    // Swap two values using ref
    int a = 10, b = 20;
    Console.WriteLine($"Before swap: a={a}, b={b}");
    Swap(ref a, ref b);
    Console.WriteLine($"After swap:  a={a}, b={b}");

    // Parse with validation using out
    string[] inputs = { "42", "hello", "3.14", "-7" };
    foreach (string input in inputs)
    {
        if (TryParsePositiveInt(input, out int result, out string error))
            Console.WriteLine($"  \"{input}\" -> valid: {result}");
        else
            Console.WriteLine($"  \"{input}\" -> invalid: {error}");
    }

    // Clamp value using ref
    int value = 150;
    Clamp(ref value, 0, 100);
    Console.WriteLine($"Clamp(150, 0, 100) = {value}");
    Console.WriteLine();

    static void Swap(ref int x, ref int y) => (x, y) = (y, x);

    static bool TryParsePositiveInt(string s, out int result, out string error)
    {
        if (!int.TryParse(s, out result))
        {
            error = "Not a valid integer";
            return false;
        }
        if (result <= 0)
        {
            error = "Must be positive";
            return false;
        }
        error = string.Empty;
        return true;
    }

    static void Clamp(ref int val, int min, int max)
    {
        if (val < min) val = min;
        else if (val > max) val = max;
    }
}

// Exercise 2: Method overloading
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Method Overloading ===");

    Console.WriteLine($"Area(circle r=5):       {Area(5.0):F2}");
    Console.WriteLine($"Area(rect 4x6):         {Area(4.0, 6.0):F2}");
    Console.WriteLine($"Area(triangle 3,4,5):   {Area(3.0, 4.0, 5.0):F2}");

    Console.WriteLine($"Format(42):             {Format(42)}");
    Console.WriteLine($"Format(3.14159):        {Format(3.14159)}");
    Console.WriteLine($"Format(\"hello\"):        {Format("hello")}");
    Console.WriteLine($"Format(true):           {Format(true)}");
    Console.WriteLine();

    // Circle area
    static double Area(double radius) => Math.PI * radius * radius;
    // Rectangle area
    static double Area(double width, double height) => width * height;
    // Triangle area using Heron's formula
    static double Area(double a, double b, double c)
    {
        double s = (a + b + c) / 2;
        return Math.Sqrt(s * (s - a) * (s - b) * (s - c));
    }

    static string Format(int value) => $"Integer: {value:N0}";
    static string Format(double value) => $"Double: {value:F3}";
    static string Format(string value) => $"String: \"{value}\" (length={value.Length})";
    static string Format(bool value) => $"Boolean: {(value ? "Yes" : "No")}";
}

// Exercise 3: Recursion — classic problems
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Recursion ===");

    // Factorial
    for (int i = 0; i <= 10; i++)
        Console.Write($"{Factorial(i)} ");
    Console.WriteLine();

    // Fibonacci
    Console.Write("Fibonacci: ");
    for (int i = 0; i < 15; i++)
        Console.Write($"{Fibonacci(i)} ");
    Console.WriteLine();

    // Power function
    Console.WriteLine($"2^10 = {Power(2, 10)}");
    Console.WriteLine($"3^5  = {Power(3, 5)}");

    // GCD using Euclidean algorithm
    Console.WriteLine($"GCD(48, 18) = {GCD(48, 18)}");
    Console.WriteLine($"GCD(100, 75) = {GCD(100, 75)}");

    // Sum of digits
    Console.WriteLine($"DigitSum(12345) = {DigitSum(12345)}");
    Console.WriteLine();

    static long Factorial(int n) => n <= 1 ? 1 : n * Factorial(n - 1);

    static int Fibonacci(int n)
    {
        if (n <= 1) return n;
        int a = 0, b = 1;
        for (int i = 2; i <= n; i++)
            (a, b) = (b, a + b);
        return b;
    }

    static long Power(long baseVal, int exp)
    {
        if (exp == 0) return 1;
        if (exp % 2 == 0)
        {
            long half = Power(baseVal, exp / 2);
            return half * half;
        }
        return baseVal * Power(baseVal, exp - 1);
    }

    static int GCD(int a, int b) => b == 0 ? a : GCD(b, a % b);

    static int DigitSum(int n) => n < 10 ? n : n % 10 + DigitSum(n / 10);
}

// Exercise 4: Optional parameters, named arguments, and params
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Optional/Named Params ===");

    Console.WriteLine(CreateGreeting("Alice"));
    Console.WriteLine(CreateGreeting("Bob", greeting: "Hi"));
    Console.WriteLine(CreateGreeting("Charlie", punctuation: "!!!", greeting: "Hey"));

    Console.WriteLine($"Sum():          {Sum()}");
    Console.WriteLine($"Sum(1,2,3):     {Sum(1, 2, 3)}");
    Console.WriteLine($"Sum(10,20,...):  {Sum(10, 20, 30, 40, 50)}");

    PrintFormatted("Title", width: 30, fillChar: '=');
    PrintFormatted("Subtitle", width: 20);
    Console.WriteLine();

    static string CreateGreeting(string name, string greeting = "Hello", string punctuation = "!")
        => $"{greeting}, {name}{punctuation}";

    static int Sum(params int[] numbers)
    {
        int total = 0;
        foreach (int n in numbers) total += n;
        return total;
    }

    static void PrintFormatted(string text, int width = 40, char fillChar = '-')
    {
        int padding = Math.Max(0, width - text.Length - 2) / 2;
        string left = new string(fillChar, padding);
        string right = new string(fillChar, width - text.Length - 2 - padding);
        Console.WriteLine($"{left} {text} {right}");
    }
}

// Exercise 5: Expression-bodied members and local functions
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Expression Bodies & Local Functions ===");

    // Expression-bodied methods for math
    int[] values = { -5, 0, 3, -2, 7, -1, 4 };
    Console.WriteLine($"Values: [{string.Join(", ", values)}]");
    Console.WriteLine($"Absolute values: [{string.Join(", ", values.Select(Abs))}]");
    Console.WriteLine($"Positive count: {CountWhere(values, IsPositive)}");
    Console.WriteLine($"Negative count: {CountWhere(values, IsNegative)}");

    // Local function for validation pipeline
    var (isValid, message) = ValidateAge(25);
    Console.WriteLine($"ValidateAge(25): valid={isValid}, msg=\"{message}\"");
    (isValid, message) = ValidateAge(-1);
    Console.WriteLine($"ValidateAge(-1): valid={isValid}, msg=\"{message}\"");
    (isValid, message) = ValidateAge(200);
    Console.WriteLine($"ValidateAge(200): valid={isValid}, msg=\"{message}\"");
    Console.WriteLine();

    static int Abs(int x) => x < 0 ? -x : x;
    static bool IsPositive(int x) => x > 0;
    static bool IsNegative(int x) => x < 0;

    static int CountWhere(int[] arr, Func<int, bool> predicate)
    {
        int count = 0;
        foreach (int item in arr)
            if (predicate(item)) count++;
        return count;
    }

    static (bool isValid, string message) ValidateAge(int age)
    {
        if (age < 0) return (false, "Age cannot be negative");
        if (age > 150) return (false, "Age seems unrealistic");
        return (true, "Valid age");
    }
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
