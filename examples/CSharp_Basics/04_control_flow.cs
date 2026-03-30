// Lesson 04: Control Flow
// Run: dotnet run

using System;
using System.Collections.Generic;

// =============================================================================
// IF / ELSE IF / ELSE
// =============================================================================
Console.WriteLine("=== If / Else ===");

int temperature = 25;

if (temperature > 35)
{
    Console.WriteLine("It's extremely hot!");
}
else if (temperature > 25)
{
    Console.WriteLine("It's warm.");
}
else if (temperature > 15)
{
    Console.WriteLine($"It's pleasant ({temperature}°C).");
}
else
{
    Console.WriteLine("It's cold.");
}

// Single-line if (no braces — use sparingly)
bool isRaining = true;
if (isRaining) Console.WriteLine("Bring an umbrella!");

// =============================================================================
// SWITCH STATEMENT
// =============================================================================
Console.WriteLine("\n=== Switch Statement ===");

string dayOfWeek = "Wednesday";

switch (dayOfWeek)
{
    case "Monday":
    case "Tuesday":
    case "Wednesday":
    case "Thursday":
    case "Friday":
        Console.WriteLine($"{dayOfWeek} is a weekday.");
        break;
    case "Saturday":
    case "Sunday":
        Console.WriteLine($"{dayOfWeek} is a weekend day.");
        break;
    default:
        Console.WriteLine("Unknown day.");
        break;
}

// =============================================================================
// SWITCH EXPRESSION (C# 8+)
// =============================================================================
Console.WriteLine("\n=== Switch Expression ===");

int httpStatus = 404;
string statusMessage = httpStatus switch
{
    200 => "OK",
    301 => "Moved Permanently",
    400 => "Bad Request",
    401 => "Unauthorized",
    403 => "Forbidden",
    404 => "Not Found",
    500 => "Internal Server Error",
    _   => $"Unknown Status ({httpStatus})"
};
Console.WriteLine($"HTTP {httpStatus}: {statusMessage}");

// Switch expression with range patterns
int score = 87;
string grade = score switch
{
    >= 90 => "A",
    >= 80 => "B",
    >= 70 => "C",
    >= 60 => "D",
    _     => "F"
};
Console.WriteLine($"Score {score} -> Grade {grade}");

// =============================================================================
// PATTERN MATCHING
// =============================================================================
Console.WriteLine("\n=== Pattern Matching ===");

// Type pattern
object[] items = { 42, "hello", 3.14, true, null!, new int[] { 1, 2, 3 } };

foreach (object item in items)
{
    string description = item switch
    {
        int n when n > 0    => $"Positive integer: {n}",
        int n               => $"Non-positive integer: {n}",
        string s            => $"String of length {s.Length}: \"{s}\"",
        double d            => $"Double: {d}",
        bool b              => $"Boolean: {b}",
        int[] arr           => $"Int array with {arr.Length} elements",
        null                => "Null value",
        _                   => $"Unknown type: {item.GetType().Name}"
    };
    Console.WriteLine($"  {description}");
}

// Property pattern
var point = new { X = 3, Y = 0 };
string location = point switch
{
    { X: 0, Y: 0 }     => "Origin",
    { X: _, Y: 0 }     => "On X-axis",
    { X: 0, Y: _ }     => "On Y-axis",
    _                   => $"Point ({point.X}, {point.Y})"
};
Console.WriteLine($"\nPoint location: {location}");

// =============================================================================
// FOR LOOP
// =============================================================================
Console.WriteLine("\n=== For Loop ===");

// Classic for loop
Console.Write("Counting: ");
for (int i = 1; i <= 10; i++)
{
    Console.Write($"{i} ");
}
Console.WriteLine();

// Counting down
Console.Write("Countdown: ");
for (int i = 5; i >= 1; i--)
{
    Console.Write($"{i} ");
}
Console.WriteLine("Go!");

// Step by 2
Console.Write("Even numbers: ");
for (int i = 2; i <= 20; i += 2)
{
    Console.Write($"{i} ");
}
Console.WriteLine();

// =============================================================================
// FOREACH LOOP
// =============================================================================
Console.WriteLine("\n=== Foreach Loop ===");

string[] fruits = { "Apple", "Banana", "Cherry", "Date", "Elderberry" };

foreach (string fruit in fruits)
{
    Console.WriteLine($"  Fruit: {fruit}");
}

// Foreach with index (using LINQ or manual counter)
Console.WriteLine("\nWith index:");
int index = 0;
foreach (string fruit in fruits)
{
    Console.WriteLine($"  [{index}] {fruit}");
    index++;
}

// =============================================================================
// WHILE AND DO-WHILE
// =============================================================================
Console.WriteLine("\n=== While Loop ===");

// While loop — may execute zero times
int counter = 1;
int sum = 0;
while (counter <= 100)
{
    sum += counter;
    counter++;
}
Console.WriteLine($"Sum of 1 to 100: {sum}");

// Do-while — executes at least once
Console.WriteLine("\nDo-While (simulated menu):");
int menuChoice;
int iteration = 0;
do
{
    iteration++;
    menuChoice = iteration >= 3 ? 0 : iteration; // Simulate: pick 1, 2, then exit
    switch (menuChoice)
    {
        case 1: Console.WriteLine("  Option 1 selected."); break;
        case 2: Console.WriteLine("  Option 2 selected."); break;
        case 0: Console.WriteLine("  Exiting menu."); break;
        default: Console.WriteLine("  Invalid option."); break;
    }
} while (menuChoice != 0);

// =============================================================================
// BREAK, CONTINUE, GOTO
// =============================================================================
Console.WriteLine("\n=== Break and Continue ===");

// Break — exit loop early
Console.Write("Break at 5: ");
for (int i = 1; i <= 10; i++)
{
    if (i == 5) break;
    Console.Write($"{i} ");
}
Console.WriteLine();

// Continue — skip current iteration
Console.Write("Skip odds: ");
for (int i = 1; i <= 10; i++)
{
    if (i % 2 != 0) continue;
    Console.Write($"{i} ");
}
Console.WriteLine();

// Labeled break with goto (use sparingly)
Console.WriteLine("\nNested loop break with goto:");
for (int i = 0; i < 3; i++)
{
    for (int j = 0; j < 3; j++)
    {
        if (i == 1 && j == 1)
        {
            Console.WriteLine($"  Breaking at ({i},{j})");
            goto AfterLoop;
        }
        Console.WriteLine($"  ({i},{j})");
    }
}
AfterLoop:
Console.WriteLine("  After nested loops.");

// =============================================================================
// TUPLE PATTERNS AND WHEN GUARDS IN SWITCH
// =============================================================================
Console.WriteLine("\n=== Tuple Patterns ===");

string rock = "rock", paper = "paper", scissors = "scissors";
(string, string)[] games = { (rock, scissors), (paper, rock), (scissors, scissors) };

foreach (var (p1, p2) in games)
{
    string result = (p1, p2) switch
    {
        (var a, var b) when a == b          => "Draw",
        ("rock", "scissors")               => "Player 1 wins",
        ("scissors", "paper")              => "Player 1 wins",
        ("paper", "rock")                  => "Player 1 wins",
        _                                  => "Player 2 wins"
    };
    Console.WriteLine($"  {p1} vs {p2}: {result}");
}
