/*
 * Exercises for Lesson 06: Arrays and Strings
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Array manipulation — sorting, searching, slicing
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Array Manipulation ===");

    int[] numbers = { 38, 27, 43, 3, 9, 82, 10, 15, 72, 51 };
    Console.WriteLine($"Original: [{string.Join(", ", numbers)}]");

    // Manual bubble sort on a copy
    int[] sorted = (int[])numbers.Clone();
    for (int i = 0; i < sorted.Length - 1; i++)
        for (int j = 0; j < sorted.Length - 1 - i; j++)
            if (sorted[j] > sorted[j + 1])
                (sorted[j], sorted[j + 1]) = (sorted[j + 1], sorted[j]);
    Console.WriteLine($"Sorted:   [{string.Join(", ", sorted)}]");

    // Binary search on sorted array
    int target = 43;
    int index = Array.BinarySearch(sorted, target);
    Console.WriteLine($"BinarySearch({target}): index={index}");

    // Array slicing with ranges
    int[] slice = numbers[2..6];
    Console.WriteLine($"numbers[2..6]: [{string.Join(", ", slice)}]");
    Console.WriteLine($"Last 3: [{string.Join(", ", numbers[^3..])}]");

    // Reverse
    int[] reversed = (int[])numbers.Clone();
    Array.Reverse(reversed);
    Console.WriteLine($"Reversed: [{string.Join(", ", reversed)}]");
    Console.WriteLine();
}

// Exercise 2: Multi-dimensional and jagged arrays
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Multi-dimensional Arrays ===");

    // 2D matrix multiplication
    int[,] matA = { { 1, 2 }, { 3, 4 } };
    int[,] matB = { { 5, 6 }, { 7, 8 } };
    int[,] result = new int[2, 2];

    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 2; j++)
            for (int k = 0; k < 2; k++)
                result[i, j] += matA[i, k] * matB[k, j];

    Console.WriteLine("Matrix A * B =");
    for (int i = 0; i < 2; i++)
        Console.WriteLine($"  [{result[i, 0],4} {result[i, 1],4}]");

    // Jagged array — Pascal's triangle
    Console.WriteLine("\nPascal's Triangle:");
    int rows = 7;
    int[][] pascal = new int[rows][];
    for (int i = 0; i < rows; i++)
    {
        pascal[i] = new int[i + 1];
        pascal[i][0] = pascal[i][i] = 1;
        for (int j = 1; j < i; j++)
            pascal[i][j] = pascal[i - 1][j - 1] + pascal[i - 1][j];

        string padding = new string(' ', (rows - i - 1) * 2);
        Console.WriteLine($"{padding}{string.Join("  ", pascal[i].Select(n => n.ToString().PadLeft(2)))}");
    }
    Console.WriteLine();
}

// Exercise 3: String processing — search, replace, format
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: String Processing ===");

    string text = "  The Quick Brown Fox Jumps Over The Lazy Dog  ";
    Console.WriteLine($"Original: \"{text}\"");
    Console.WriteLine($"Trimmed:  \"{text.Trim()}\"");
    Console.WriteLine($"Lower:    \"{text.Trim().ToLower()}\"");
    Console.WriteLine($"Upper:    \"{text.Trim().ToUpper()}\"");

    // Word analysis
    string[] words = text.Trim().Split(' ', StringSplitOptions.RemoveEmptyEntries);
    Console.WriteLine($"Word count: {words.Length}");
    Console.WriteLine($"Longest word: {words.OrderByDescending(w => w.Length).First()}");

    // Character frequency
    var freq = text.Trim().ToLower()
        .Where(char.IsLetter)
        .GroupBy(c => c)
        .OrderByDescending(g => g.Count())
        .Take(5);
    Console.Write("Top 5 chars: ");
    foreach (var g in freq)
        Console.Write($"'{g.Key}'={g.Count()} ");
    Console.WriteLine();

    // Replace and contains
    string csv = "apple,banana,,cherry,,date";
    string cleaned = string.Join(",", csv.Split(',', StringSplitOptions.RemoveEmptyEntries));
    Console.WriteLine($"Cleaned CSV: {cleaned}");
    Console.WriteLine();
}

// Exercise 4: StringBuilder for efficient string concatenation
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: StringBuilder ===");

    // Build a formatted report
    var sb = new System.Text.StringBuilder();
    sb.AppendLine("+----------------------------+");
    sb.AppendLine("|     Monthly Sales Report   |");
    sb.AppendLine("+----------------------------+");

    (string month, double sales)[] data =
    {
        ("January", 12500.50),
        ("February", 11200.75),
        ("March", 15800.00),
        ("April", 13100.25),
        ("May", 16400.80)
    };

    double total = 0;
    foreach (var (month, sales) in data)
    {
        sb.AppendLine($"| {month,-12} ${sales,10:N2} |");
        total += sales;
    }
    sb.AppendLine("+----------------------------+");
    sb.AppendLine($"| {"Total",-12} ${total,10:N2} |");
    sb.AppendLine("+----------------------------+");

    Console.Write(sb.ToString());

    // Performance comparison note
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var sb2 = new System.Text.StringBuilder();
    for (int i = 0; i < 10000; i++)
        sb2.Append(i.ToString());
    sw.Stop();
    Console.WriteLine($"StringBuilder (10k appends): {sw.ElapsedTicks} ticks, length={sb2.Length}");
    Console.WriteLine();
}

// Exercise 5: String interpolation, verbatim strings, and raw strings
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: String Literals ===");

    // Verbatim string for file paths
    string path = @"C:\Users\Admin\Documents\file.txt";
    Console.WriteLine($"Verbatim path: {path}");

    // Escape sequences
    Console.WriteLine("Tab:\tHello\tWorld");
    Console.WriteLine("Newline in string: Line1\nLine2");

    // Multiline interpolated string
    string name = "Alice";
    int age = 30;
    string card = $"""
        ╔══════════════════╗
        ║  Name: {name,-10}║
        ║  Age:  {age,-10}║
        ╚══════════════════╝
        """;
    Console.WriteLine(card);

    // String comparison
    string a = "hello";
    string b = "HELLO";
    Console.WriteLine($"Ordinal: \"{a}\" == \"{b}\" -> {string.Equals(a, b, StringComparison.Ordinal)}");
    Console.WriteLine($"IgnoreCase: \"{a}\" == \"{b}\" -> {string.Equals(a, b, StringComparison.OrdinalIgnoreCase)}");

    // Char operations
    string sample = "Hello, World! 123";
    int letters = sample.Count(char.IsLetter);
    int digits = sample.Count(char.IsDigit);
    int spaces = sample.Count(c => c == ' ');
    Console.WriteLine($"\"{sample}\": {letters} letters, {digits} digits, {spaces} spaces");
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
