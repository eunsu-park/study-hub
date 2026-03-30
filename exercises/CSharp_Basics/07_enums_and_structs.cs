/*
 * Exercises for Lesson 07: Enums and Structs
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Enum basics and flags
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Enums and Flags ===");

    // Basic enum iteration
    Console.WriteLine("Days of the week:");
    foreach (DayOfWeek day in Enum.GetValues<DayOfWeek>())
        Console.WriteLine($"  {day} = {(int)day}");

    // Flags enum for file permissions
    var perms = FilePermission.Read | FilePermission.Write;
    Console.WriteLine($"\nPermissions: {perms}");
    Console.WriteLine($"Has Read:    {perms.HasFlag(FilePermission.Read)}");
    Console.WriteLine($"Has Execute: {perms.HasFlag(FilePermission.Execute)}");

    // Add execute permission
    perms |= FilePermission.Execute;
    Console.WriteLine($"After adding Execute: {perms}");

    // Remove write permission
    perms &= ~FilePermission.Write;
    Console.WriteLine($"After removing Write: {perms}");

    // Enum parsing
    if (Enum.TryParse<Season>("Summer", out var season))
        Console.WriteLine($"\nParsed: \"Summer\" -> {season} (value={((int)season)})");

    string[] inputs = { "Spring", "Autumn", "Invalid" };
    foreach (string input in inputs)
    {
        bool ok = Enum.TryParse<Season>(input, out var s);
        Console.WriteLine($"  TryParse(\"{input}\"): success={ok}, value={s}");
    }
    Console.WriteLine();
}

// Exercise 2: Struct basics — defining and using value types
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Struct Basics ===");

    // Create and use Point structs
    var p1 = new Point(3, 4);
    var p2 = new Point(7, 1);

    Console.WriteLine($"p1 = {p1}");
    Console.WriteLine($"p2 = {p2}");
    Console.WriteLine($"Distance from origin: p1={p1.DistanceFromOrigin():F2}, p2={p2.DistanceFromOrigin():F2}");
    Console.WriteLine($"Distance p1->p2: {p1.DistanceTo(p2):F2}");

    // Value type behavior — copy semantics
    var p3 = p1;
    Console.WriteLine($"\np3 = p1: p3={p3}");
    Console.WriteLine($"p1 equals p3? {p1.Equals(p3)}");

    // Color struct
    var red = new Color(255, 0, 0);
    var blue = new Color(0, 0, 255);
    var purple = Color.Mix(red, blue);
    Console.WriteLine($"\nRed:    {red}");
    Console.WriteLine($"Blue:   {blue}");
    Console.WriteLine($"Mixed:  {purple}");
    Console.WriteLine();
}

// Exercise 3: Struct vs class — demonstrating value vs reference semantics
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Struct vs Class ===");

    // Value type (struct) — copy is independent
    var sp1 = new StructPoint { X = 10, Y = 20 };
    var sp2 = sp1;
    sp2.X = 99;
    Console.WriteLine($"Struct: sp1=({sp1.X},{sp1.Y}), sp2=({sp2.X},{sp2.Y}) — independent copies");

    // Reference type (class) — copy shares data
    var cp1 = new ClassPoint { X = 10, Y = 20 };
    var cp2 = cp1;
    cp2.X = 99;
    Console.WriteLine($"Class:  cp1=({cp1.X},{cp1.Y}), cp2=({cp2.X},{cp2.Y}) — same reference!");

    // Struct in collections
    var points = new List<StructPoint>
    {
        new() { X = 1, Y = 2 },
        new() { X = 3, Y = 4 },
        new() { X = 5, Y = 6 }
    };
    Console.WriteLine($"\nStruct points: {string.Join(", ", points.Select(p => $"({p.X},{p.Y})"))}");

    // Readonly struct usage
    var range = new Range(1, 10);
    Console.WriteLine($"\nRange: {range}");
    Console.WriteLine($"Contains 5: {range.Contains(5)}");
    Console.WriteLine($"Contains 11: {range.Contains(11)}");
    Console.WriteLine($"Length: {range.Length}");
    Console.WriteLine();
}

// Exercise 4: Record structs and with-expressions
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Record Structs ===");

    var temp1 = new Temperature(20.0, TemperatureUnit.Celsius);
    Console.WriteLine($"Original: {temp1}");

    // with-expression creates a copy with modifications
    var temp2 = temp1 with { Value = 100.0 };
    Console.WriteLine($"Modified: {temp2}");

    // Value equality
    var temp3 = new Temperature(20.0, TemperatureUnit.Celsius);
    Console.WriteLine($"\ntemp1 == temp3: {temp1 == temp3}");
    Console.WriteLine($"temp1 == temp2: {temp1 == temp2}");

    // Temperature conversion
    Console.WriteLine($"\n{temp1} in Fahrenheit: {temp1.ToFahrenheit():F1}°F");
    Console.WriteLine($"{temp2} in Fahrenheit: {temp2.ToFahrenheit():F1}°F");

    var fahr = new Temperature(72.0, TemperatureUnit.Fahrenheit);
    Console.WriteLine($"{fahr} in Celsius: {fahr.ToCelsius():F1}°C");
    Console.WriteLine();
}

// Exercise 5: Combining enums and structs — card game model
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Card Game Model ===");

    // Create a deck of cards
    var deck = new List<Card>();
    foreach (Suit suit in Enum.GetValues<Suit>())
        foreach (Rank rank in Enum.GetValues<Rank>())
            deck.Add(new Card(rank, suit));

    Console.WriteLine($"Deck size: {deck.Count}");
    Console.WriteLine($"First 5: {string.Join(", ", deck.Take(5))}");
    Console.WriteLine($"Last 5: {string.Join(", ", deck.TakeLast(5))}");

    // Shuffle using Fisher-Yates
    var rng = new Random(42);
    for (int i = deck.Count - 1; i > 0; i--)
    {
        int j = rng.Next(i + 1);
        (deck[i], deck[j]) = (deck[j], deck[i]);
    }

    // Deal a hand of 5 cards
    var hand = deck.Take(5).OrderByDescending(c => c.Rank).ToList();
    Console.WriteLine($"\nDealt hand: {string.Join(", ", hand)}");

    // Check for pairs
    var groups = hand.GroupBy(c => c.Rank).Where(g => g.Count() >= 2);
    foreach (var g in groups)
        Console.WriteLine($"  Pair of {g.Key}s!");
    Console.WriteLine();
}

// Supporting types

[Flags]
enum FilePermission
{
    None    = 0,
    Read    = 1,
    Write   = 2,
    Execute = 4,
    All     = Read | Write | Execute
}

enum Season { Spring, Summer, Autumn, Winter }

readonly struct Point(double x, double y)
{
    public double X { get; } = x;
    public double Y { get; } = y;
    public double DistanceFromOrigin() => Math.Sqrt(X * X + Y * Y);
    public double DistanceTo(Point other) => Math.Sqrt(Math.Pow(X - other.X, 2) + Math.Pow(Y - other.Y, 2));
    public override string ToString() => $"({X}, {Y})";
}

readonly struct Color(byte r, byte g, byte b)
{
    public byte R { get; } = r;
    public byte G { get; } = g;
    public byte B { get; } = b;
    public static Color Mix(Color a, Color c) =>
        new((byte)((a.R + c.R) / 2), (byte)((a.G + c.G) / 2), (byte)((a.B + c.B) / 2));
    public override string ToString() => $"RGB({R}, {G}, {B})";
}

struct StructPoint { public int X; public int Y; }

class ClassPoint { public int X { get; set; } public int Y { get; set; } }

readonly struct Range(int min, int max)
{
    public int Min { get; } = min;
    public int Max { get; } = max;
    public bool Contains(int value) => value >= Min && value <= Max;
    public int Length => Max - Min;
    public override string ToString() => $"[{Min}..{Max}]";
}

enum TemperatureUnit { Celsius, Fahrenheit }

record struct Temperature(double Value, TemperatureUnit Unit)
{
    public double ToFahrenheit() => Unit == TemperatureUnit.Celsius ? Value * 9.0 / 5.0 + 32.0 : Value;
    public double ToCelsius() => Unit == TemperatureUnit.Fahrenheit ? (Value - 32.0) * 5.0 / 9.0 : Value;
    public override string ToString() => $"{Value:F1}°{(Unit == TemperatureUnit.Celsius ? "C" : "F")}";
}

enum Suit { Hearts, Diamonds, Clubs, Spades }
enum Rank { Two = 2, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace }

readonly record struct Card(Rank Rank, Suit Suit)
{
    public override string ToString() => $"{Rank} of {Suit}";
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
