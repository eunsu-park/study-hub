/*
 * Exercises for Lesson 03: Operators and Expressions
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Demonstrate operator precedence and associativity
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Operator Precedence ===");

    int a = 2 + 3 * 4;
    int b = (2 + 3) * 4;
    Console.WriteLine($"2 + 3 * 4   = {a}  (multiplication first)");
    Console.WriteLine($"(2 + 3) * 4 = {b}  (parentheses override)");

    int c = 10 - 4 - 2;
    int d = 10 - (4 - 2);
    Console.WriteLine($"10 - 4 - 2   = {c}  (left-to-right)");
    Console.WriteLine($"10 - (4 - 2) = {d}  (right group first)");

    bool e = true || false && false;
    bool f = (true || false) && false;
    Console.WriteLine($"true || false && false   = {e}  (&& has higher precedence)");
    Console.WriteLine($"(true || false) && false = {f}");

    int g = 2 + 3 * 4 / 2 - 1;
    Console.WriteLine($"2 + 3 * 4 / 2 - 1 = {g}  (= 2 + 6 - 1 = 7)");
    Console.WriteLine();
}

// Exercise 2: Null-coalescing and null-conditional operators
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Null-Coalescing Operators ===");

    string? input = null;
    string result = input ?? "default";
    Console.WriteLine($"null ?? \"default\" = \"{result}\"");

    input = "hello";
    result = input ?? "default";
    Console.WriteLine($"\"hello\" ?? \"default\" = \"{result}\"");

    // Chained null-coalescing
    string? first = null;
    string? second = null;
    string? third = "found";
    string final = first ?? second ?? third ?? "none";
    Console.WriteLine($"null ?? null ?? \"found\" ?? \"none\" = \"{final}\"");

    // Null-coalescing assignment
    List<string>? items = null;
    items ??= new List<string>();
    items.Add("item1");
    Console.WriteLine($"After ??= assignment, count: {items.Count}");

    // Null-conditional with method chains
    string? text = "Hello, World!";
    int? len = text?.Trim()?.Length;
    Console.WriteLine($"\"Hello, World!\"?.Trim()?.Length = {len}");

    text = null;
    len = text?.Trim()?.Length;
    Console.WriteLine($"null?.Trim()?.Length = {len?.ToString() ?? "null"}");
    Console.WriteLine();
}

// Exercise 3: Checked and unchecked arithmetic
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Checked Arithmetic ===");

    // Unchecked overflow (default behavior)
    unchecked
    {
        int max = int.MaxValue;
        int overflowed = max + 1;
        Console.WriteLine($"Unchecked: {max} + 1 = {overflowed} (overflow!)");
    }

    // Checked overflow detection
    try
    {
        checked
        {
            int max = int.MaxValue;
            int overflowed = max + 1;
            Console.WriteLine($"This line won't execute: {overflowed}");
        }
    }
    catch (OverflowException ex)
    {
        Console.WriteLine($"Checked: OverflowException caught — {ex.Message}");
    }

    // Safe multiplication with checked
    try
    {
        checked
        {
            int big = 1_000_000;
            int product = big * big;
            Console.WriteLine($"This line won't execute: {product}");
        }
    }
    catch (OverflowException)
    {
        Console.WriteLine("Checked: 1,000,000 * 1,000,000 overflows int");
        long safeProd = 1_000_000L * 1_000_000L;
        Console.WriteLine($"Using long: {safeProd}");
    }
    Console.WriteLine();
}

// Exercise 4: Bitwise operations — flags and masking
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Bitwise Operations ===");

    int a = 0b1100;
    int b = 0b1010;

    Console.WriteLine($"a       = {Convert.ToString(a, 2).PadLeft(4, '0')} ({a})");
    Console.WriteLine($"b       = {Convert.ToString(b, 2).PadLeft(4, '0')} ({b})");
    Console.WriteLine($"a & b   = {Convert.ToString(a & b, 2).PadLeft(4, '0')} ({a & b})  AND");
    Console.WriteLine($"a | b   = {Convert.ToString(a | b, 2).PadLeft(4, '0')} ({a | b})  OR");
    Console.WriteLine($"a ^ b   = {Convert.ToString(a ^ b, 2).PadLeft(4, '0')} ({a ^ b})  XOR");
    Console.WriteLine($"~a      = {~a}  NOT (two's complement)");

    // Shift operations
    int val = 1;
    Console.WriteLine($"1 << 0 = {val << 0}");
    Console.WriteLine($"1 << 1 = {val << 1}");
    Console.WriteLine($"1 << 4 = {val << 4}");
    Console.WriteLine($"16 >> 2 = {16 >> 2}");

    // Practical: permission flags
    int read = 0b001, write = 0b010, execute = 0b100;
    int permissions = read | write;
    Console.WriteLine($"Permissions: {Convert.ToString(permissions, 2).PadLeft(3, '0')}");
    Console.WriteLine($"Can read?    {(permissions & read) != 0}");
    Console.WriteLine($"Can execute? {(permissions & execute) != 0}");
    Console.WriteLine();
}

// Exercise 5: Ternary operator and compound expressions
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Ternary and Compound Expressions ===");

    // Ternary for classification
    int[] scores = { 95, 82, 67, 45, 33 };
    foreach (int score in scores)
    {
        string grade = score >= 90 ? "A"
                     : score >= 80 ? "B"
                     : score >= 70 ? "C"
                     : score >= 60 ? "D"
                     : "F";
        Console.WriteLine($"Score {score} -> Grade {grade}");
    }

    // Compound assignment operators
    int x = 100;
    Console.WriteLine($"x = {x}");
    x += 10; Console.WriteLine($"x += 10 -> {x}");
    x -= 5;  Console.WriteLine($"x -= 5  -> {x}");
    x *= 2;  Console.WriteLine($"x *= 2  -> {x}");
    x /= 3;  Console.WriteLine($"x /= 3  -> {x}");
    x %= 13; Console.WriteLine($"x %= 13 -> {x}");

    // String interpolation with expressions
    double radius = 5.0;
    Console.WriteLine($"Circle: r={radius}, area={Math.PI * radius * radius:F2}, circumference={2 * Math.PI * radius:F2}");
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
