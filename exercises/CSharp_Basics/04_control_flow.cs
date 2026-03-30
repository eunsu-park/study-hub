/*
 * Exercises for Lesson 04: Control Flow
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Switch expressions with pattern matching
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Switch Expressions ===");

    // Classify HTTP status codes using switch expression
    int[] statusCodes = { 200, 201, 301, 400, 403, 404, 500, 503 };
    foreach (int code in statusCodes)
    {
        string description = code switch
        {
            200 => "OK",
            201 => "Created",
            301 => "Moved Permanently",
            400 => "Bad Request",
            403 => "Forbidden",
            404 => "Not Found",
            500 => "Internal Server Error",
            503 => "Service Unavailable",
            >= 200 and < 300 => "Success",
            >= 300 and < 400 => "Redirection",
            >= 400 and < 500 => "Client Error",
            >= 500 => "Server Error",
            _ => "Unknown"
        };
        Console.WriteLine($"  HTTP {code}: {description}");
    }
    Console.WriteLine();
}

// Exercise 2: Various loop patterns — for, foreach, while, do-while
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Loop Patterns ===");

    // FizzBuzz using a for loop
    Console.Write("FizzBuzz (1-20): ");
    for (int i = 1; i <= 20; i++)
    {
        string output = (i % 3, i % 5) switch
        {
            (0, 0) => "FizzBuzz",
            (0, _) => "Fizz",
            (_, 0) => "Buzz",
            _ => i.ToString()
        };
        Console.Write($"{output} ");
    }
    Console.WriteLine();

    // Collatz sequence using while
    int n = 27;
    int steps = 0;
    Console.Write($"Collatz({n}): {n}");
    while (n != 1)
    {
        n = n % 2 == 0 ? n / 2 : 3 * n + 1;
        steps++;
        if (steps <= 10) Console.Write($" -> {n}");
    }
    Console.WriteLine($" ... ({steps} steps total)");

    // Pyramid with nested loops
    Console.WriteLine("Pyramid:");
    int height = 5;
    for (int row = 1; row <= height; row++)
    {
        Console.Write(new string(' ', height - row));
        Console.WriteLine(new string('*', 2 * row - 1));
    }
    Console.WriteLine();
}

// Exercise 3: Pattern matching with is, when, and relational patterns
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Pattern Matching ===");

    object[] items = { 42, "hello", 3.14, null!, true, 'A', new int[] { 1, 2, 3 } };

    foreach (object? item in items)
    {
        string desc = item switch
        {
            int i when i > 0 => $"Positive int: {i}",
            int i => $"Non-positive int: {i}",
            string { Length: > 5 } s => $"Long string: \"{s}\"",
            string s => $"Short string: \"{s}\"",
            double d => $"Double: {d:F2}",
            bool b => $"Boolean: {b}",
            char c => $"Char: '{c}'",
            int[] arr => $"Int array with {arr.Length} elements",
            null => "null value",
            _ => $"Unknown type: {item.GetType().Name}"
        };
        Console.WriteLine($"  {desc}");
    }

    // Relational and logical patterns
    Console.WriteLine("\nBMI Classification:");
    double[] bmis = { 16.5, 18.5, 22.0, 27.0, 32.5 };
    foreach (double bmi in bmis)
    {
        string category = bmi switch
        {
            < 18.5 => "Underweight",
            >= 18.5 and < 25.0 => "Normal",
            >= 25.0 and < 30.0 => "Overweight",
            >= 30.0 => "Obese",
            _ => "Invalid"
        };
        Console.WriteLine($"  BMI {bmi:F1} -> {category}");
    }
    Console.WriteLine();
}

// Exercise 4: break, continue, and labeled loops
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Loop Control ===");

    // Skip even numbers, stop at 15
    Console.Write("Odd numbers (stop at 15): ");
    for (int i = 1; i <= 20; i++)
    {
        if (i > 15) break;
        if (i % 2 == 0) continue;
        Console.Write($"{i} ");
    }
    Console.WriteLine();

    // Find first prime pair (twin primes)
    Console.Write("Twin primes under 50: ");
    for (int i = 2; i < 50; i++)
    {
        if (IsPrime(i) && IsPrime(i + 2))
            Console.Write($"({i},{i + 2}) ");
    }
    Console.WriteLine();

    // Nested loop with early exit using a flag
    Console.WriteLine("Finding first pair (i,j) where i*j == 42:");
    bool found = false;
    for (int i = 1; i <= 10 && !found; i++)
    {
        for (int j = 1; j <= 10 && !found; j++)
        {
            if (i * j == 42)
            {
                Console.WriteLine($"  Found: {i} * {j} = 42");
                found = true;
            }
        }
    }
    Console.WriteLine();

    static bool IsPrime(int n)
    {
        if (n < 2) return false;
        for (int i = 2; i * i <= n; i++)
            if (n % i == 0) return false;
        return true;
    }
}

// Exercise 5: Implement a simple menu-driven calculator using do-while and switch
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Menu-Driven Calculator (Simulated) ===");

    // Simulate a sequence of operations
    (string op, double a, double b)[] operations =
    {
        ("+", 10, 3),
        ("-", 25, 7),
        ("*", 6, 8),
        ("/", 100, 3),
        ("/", 5, 0),
        ("%", 17, 5)
    };

    foreach (var (op, a, b) in operations)
    {
        string result = op switch
        {
            "+" => $"{a} + {b} = {a + b}",
            "-" => $"{a} - {b} = {a - b}",
            "*" => $"{a} * {b} = {a * b}",
            "/" when b != 0 => $"{a} / {b} = {a / b:F4}",
            "/" => $"{a} / {b} = Error (division by zero)",
            "%" when b != 0 => $"{a} % {b} = {a % b}",
            _ => $"Unknown operator: {op}"
        };
        Console.WriteLine($"  {result}");
    }
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
