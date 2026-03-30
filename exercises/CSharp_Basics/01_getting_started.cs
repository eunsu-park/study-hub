/*
 * Exercises for Lesson 01: Getting Started
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Print a welcome message with your name and today's date
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Welcome Message ===");
    string name = "Alice";
    string date = DateTime.Now.ToString("yyyy-MM-dd");
    Console.WriteLine($"Welcome, {name}! Today is {date}.");
    Console.WriteLine($"You are running .NET {Environment.Version}");
    Console.WriteLine();
}

// Exercise 2: Display system information using Environment class
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: System Information ===");
    Console.WriteLine($"OS: {Environment.OSVersion}");
    Console.WriteLine($"Machine Name: {Environment.MachineName}");
    Console.WriteLine($"Processor Count: {Environment.ProcessorCount}");
    Console.WriteLine($"64-bit OS: {Environment.Is64BitOperatingSystem}");
    Console.WriteLine($"64-bit Process: {Environment.Is64BitProcess}");
    Console.WriteLine($"CLR Version: {Environment.Version}");
    Console.WriteLine($"Current Directory: {Environment.CurrentDirectory}");
    Console.WriteLine();
}

// Exercise 3: Command-line argument simulation — parse and display arguments
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Argument Parsing ===");
    string[] simulatedArgs = { "--name", "Bob", "--verbose", "--count", "5" };

    string? name = null;
    bool verbose = false;
    int count = 1;

    for (int i = 0; i < simulatedArgs.Length; i++)
    {
        switch (simulatedArgs[i])
        {
            case "--name":
                name = simulatedArgs[++i];
                break;
            case "--verbose":
                verbose = true;
                break;
            case "--count":
                count = int.Parse(simulatedArgs[++i]);
                break;
        }
    }

    Console.WriteLine($"Name: {name ?? "(not set)"}");
    Console.WriteLine($"Verbose: {verbose}");
    Console.WriteLine($"Count: {count}");

    if (verbose)
    {
        for (int i = 0; i < count; i++)
            Console.WriteLine($"  Iteration {i + 1}: Hello, {name}!");
    }
    Console.WriteLine();
}

// Exercise 4: Build a simple ASCII art banner generator
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: ASCII Banner ===");
    string message = "C# Rocks!";
    int width = message.Length + 6;

    Console.WriteLine(new string('*', width));
    Console.WriteLine($"*  {message}  *");
    Console.WriteLine(new string('*', width));
    Console.WriteLine();

    // Boxed version
    string top = "+" + new string('-', width - 2) + "+";
    Console.WriteLine(top);
    Console.WriteLine($"|  {message}  |");
    Console.WriteLine(top);
    Console.WriteLine();
}

// Exercise 5: Create a multiplication table using nested formatting
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Multiplication Table ===");
    int size = 5;

    // Header row
    Console.Write("     ");
    for (int j = 1; j <= size; j++)
        Console.Write($"{j,4}");
    Console.WriteLine();
    Console.WriteLine("    " + new string('-', size * 4 + 1));

    // Table body
    for (int i = 1; i <= size; i++)
    {
        Console.Write($"{i,3} |");
        for (int j = 1; j <= size; j++)
            Console.Write($"{i * j,4}");
        Console.WriteLine();
    }
    Console.WriteLine();
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
