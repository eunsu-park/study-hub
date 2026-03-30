// Lesson 01: Getting Started with C#
// Run: dotnet run

// This file demonstrates top-level statements (C# 9+).
// In top-level statements, the compiler generates a Main method automatically.

using System;

// --- Hello World ---
Console.WriteLine("Hello, World!");
Console.WriteLine("Welcome to C# Programming!");

// --- Console Output Formatting ---
Console.WriteLine("--- Console Output ---");
Console.Write("This does NOT add a newline. ");
Console.Write("So this appears on the same line.\n");

// String interpolation in output
string language = "C#";
int version = 12;
Console.WriteLine($"Language: {language}, Version: {version}");

// Composite formatting (older style)
Console.WriteLine("Language: {0}, Version: {1}", language, version);

// Format specifiers
double pi = 3.14159265;
Console.WriteLine($"Pi to 2 decimal places: {pi:F2}");
Console.WriteLine($"Pi in scientific notation: {pi:E3}");

// --- Console Input ---
Console.WriteLine("\n--- Console Input ---");
Console.Write("Enter your name (or press Enter for default): ");
string? name = Console.ReadLine();
if (string.IsNullOrWhiteSpace(name))
{
    name = "Developer";
}
Console.WriteLine($"Hello, {name}!");

// --- Command Line Arguments ---
// When using top-level statements, args is automatically available
Console.WriteLine("\n--- Command Line Arguments ---");
Console.WriteLine($"Number of arguments: {args.Length}");
for (int i = 0; i < args.Length; i++)
{
    Console.WriteLine($"  args[{i}] = {args[i]}");
}

if (args.Length == 0)
{
    Console.WriteLine("  (No arguments provided. Try: dotnet run -- arg1 arg2)");
}

// --- Environment Information ---
Console.WriteLine("\n--- Environment Information ---");
Console.WriteLine($"OS: {Environment.OSVersion}");
Console.WriteLine($".NET Version: {Environment.Version}");
Console.WriteLine($"Machine Name: {Environment.MachineName}");
Console.WriteLine($"Current Directory: {Environment.CurrentDirectory}");
Console.WriteLine($"Is 64-bit OS: {Environment.Is64BitOperatingSystem}");
Console.WriteLine($"Is 64-bit Process: {Environment.Is64BitProcess}");

// --- Exit Codes ---
// Return an exit code (0 = success)
Console.WriteLine("\n--- Exit Code ---");
Console.WriteLine("Program completed successfully (exit code 0).");
Environment.ExitCode = 0;

// --- Demonstrating a Traditional Main Method Style ---
// Note: In a real project, you would choose EITHER top-level statements
// OR a class with Main, not both. This is shown here for reference only.
//
// class Program
// {
//     static void Main(string[] args)
//     {
//         Console.WriteLine("Hello from Main!");
//         foreach (var arg in args)
//         {
//             Console.WriteLine($"Arg: {arg}");
//         }
//     }
// }

// --- Multiple Classes Can Coexist with Top-Level Statements ---
var greeter = new Greeter("World");
greeter.SayHello();

/// <summary>
/// A simple helper class to demonstrate that classes can be defined
/// alongside top-level statements.
/// </summary>
class Greeter
{
    private readonly string _name;

    public Greeter(string name)
    {
        _name = name;
    }

    public void SayHello()
    {
        Console.WriteLine($"\nGreeter says: Hello, {_name}!");
    }
}
