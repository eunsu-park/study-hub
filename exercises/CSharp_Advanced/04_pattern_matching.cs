/*
 * Exercises for Lesson 04: Pattern Matching
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;

// ---------------------------------------------------------------------------
// Exercise 1: Type patterns — shape area calculator
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Type Pattern — Shape Areas ===");

    Shape[] shapes = { new Circle(5), new Rectangle(4, 6), new Triangle(3, 8) };

    foreach (var shape in shapes)
    {
        double area = shape switch
        {
            Circle c    => Math.PI * c.Radius * c.Radius,
            Rectangle r => r.Width * r.Height,
            Triangle t  => 0.5 * t.Base * t.Height,
            _           => throw new ArgumentException("Unknown shape")
        };
        Console.WriteLine($"  {shape.GetType().Name}: area = {area:F2}");
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Property patterns — access control
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Property Patterns — Access Control ===");

    var users = new List<User>
    {
        new("Alice", Role.Admin, true),
        new("Bob", Role.Editor, true),
        new("Charlie", Role.Viewer, true),
        new("Dave", Role.Admin, false),
        new("Eve", Role.Editor, false),
    };

    foreach (var user in users)
    {
        string access = user switch
        {
            { IsActive: false }                    => "DENIED (inactive)",
            { UserRole: Role.Admin }               => "FULL ACCESS",
            { UserRole: Role.Editor, IsActive: true } => "READ/WRITE",
            { UserRole: Role.Viewer }              => "READ ONLY",
            _                                       => "DENIED"
        };
        Console.WriteLine($"  {user.Name,-10} Role={user.UserRole,-7} Active={user.IsActive,-5} => {access}");
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Tuple patterns — state machine
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Tuple Patterns — Traffic Light ===");

    var transitions = new (TrafficLight Current, bool TimerExpired)[]
    {
        (TrafficLight.Red, true),
        (TrafficLight.Green, true),
        (TrafficLight.Yellow, true),
        (TrafficLight.Green, false),
    };

    foreach (var (current, expired) in transitions)
    {
        var next = (current, expired) switch
        {
            (TrafficLight.Red, true)    => TrafficLight.Green,
            (TrafficLight.Green, true)  => TrafficLight.Yellow,
            (TrafficLight.Yellow, true) => TrafficLight.Red,
            (_, false)                  => current,
            _                           => current,
        };
        Console.WriteLine($"  {current} (expired={expired}) => {next}");
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Command parser using patterns
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Command Parser ===");

    string[] commands = {
        "MOVE 10 20",
        "DRAW circle 5",
        "DRAW rect 4 6",
        "COLOR red",
        "QUIT",
        "UNKNOWN foo",
    };

    foreach (var cmd in commands)
    {
        string result = ParseCommand(cmd);
        Console.WriteLine($"  '{cmd}' => {result}");
    }
    Console.WriteLine();
}

string ParseCommand(string input)
{
    var parts = input.Split(' ');
    return parts switch
    {
        ["MOVE", var x, var y] when int.TryParse(x, out var px) && int.TryParse(y, out var py)
            => $"Move to ({px}, {py})",
        ["DRAW", "circle", var r] when double.TryParse(r, out var radius)
            => $"Draw circle radius={radius}",
        ["DRAW", "rect", var w, var h] when double.TryParse(w, out var dw) && double.TryParse(h, out var dh)
            => $"Draw rectangle {dw}x{dh}",
        ["COLOR", var c]
            => $"Set color to {c}",
        ["QUIT"]
            => "Exit program",
        _ => $"Unknown command: {input}"
    };
}

// ---------------------------------------------------------------------------
// Exercise 5: Relational and logical patterns — grading system
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Relational Patterns — Grades ===");

    int[] scores = { 97, 85, 72, 63, 55, 40, 101, -5 };

    foreach (var score in scores)
    {
        string grade = score switch
        {
            < 0 or > 100 => "INVALID",
            >= 90         => "A",
            >= 80         => "B",
            >= 70         => "C",
            >= 60         => "D",
            _             => "F"
        };
        Console.WriteLine($"  Score {score,4} => {grade}");
    }
    Console.WriteLine();
}

// ---- Run all exercises ----
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();

// ===========================================================================
// Supporting types
// ===========================================================================

abstract record Shape;
record Circle(double Radius) : Shape;
record Rectangle(double Width, double Height) : Shape;
record Triangle(double Base, double Height) : Shape;

enum Role { Admin, Editor, Viewer }
record User(string Name, Role UserRole, bool IsActive);

enum TrafficLight { Red, Yellow, Green }
