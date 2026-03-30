/*
 * Exercises for Lesson 10: Properties and Indexers
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Auto-properties and init-only properties
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Auto and Init Properties ===");

    var config = new AppConfig
    {
        AppName = "MyApp",
        Version = "2.1.0",
        MaxRetries = 5,
        Timeout = TimeSpan.FromSeconds(30)
    };

    Console.WriteLine($"App: {config.AppName} v{config.Version}");
    Console.WriteLine($"MaxRetries: {config.MaxRetries}");
    Console.WriteLine($"Timeout: {config.Timeout.TotalSeconds}s");

    // init-only — cannot modify after construction
    // config.AppName = "Other"; // Compile error!

    // Required properties
    var user = new UserProfile { Username = "alice", DisplayName = "Alice Smith" };
    Console.WriteLine($"\nUser: {user.Username} ({user.DisplayName})");
    Console.WriteLine($"Created: {user.CreatedAt:yyyy-MM-dd HH:mm}");
    Console.WriteLine();
}

// Exercise 2: Property validation with backing fields
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Property Validation ===");

    var temp = new Temperature();
    temp.Celsius = 100;
    Console.WriteLine($"100°C = {temp.Fahrenheit:F1}°F = {temp.Kelvin:F1}K");

    temp.Fahrenheit = 72;
    Console.WriteLine($"72°F = {temp.Celsius:F1}°C = {temp.Kelvin:F1}K");

    try
    {
        temp.Kelvin = -10; // Below absolute zero
    }
    catch (ArgumentOutOfRangeException ex)
    {
        Console.WriteLine($"Setting Kelvin to -10: {ex.Message}");
    }

    // Percentage with clamping
    var progress = new ProgressBar();
    progress.Percentage = 75;
    Console.WriteLine($"\nProgress: {progress}");
    progress.Percentage = 150; // Should be clamped to 100
    Console.WriteLine($"Set to 150: {progress}");
    progress.Percentage = -20; // Should be clamped to 0
    Console.WriteLine($"Set to -20: {progress}");
    Console.WriteLine();
}

// Exercise 3: Computed properties and expression-bodied members
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Computed Properties ===");

    var rect = new Rectangle(10, 5);
    Console.WriteLine($"Rectangle: {rect.Width}x{rect.Height}");
    Console.WriteLine($"Area: {rect.Area}");
    Console.WriteLine($"Perimeter: {rect.Perimeter}");
    Console.WriteLine($"IsSquare: {rect.IsSquare}");
    Console.WriteLine($"Diagonal: {rect.Diagonal:F2}");

    var square = new Rectangle(7, 7);
    Console.WriteLine($"\nSquare: {square.Width}x{square.Height}");
    Console.WriteLine($"IsSquare: {square.IsSquare}");

    // DateRange with computed properties
    var range = new DateRange(new DateTime(2025, 1, 15), new DateTime(2025, 3, 20));
    Console.WriteLine($"\nDate range: {range}");
    Console.WriteLine($"Days: {range.TotalDays}");
    Console.WriteLine($"Is current: {range.IsCurrent}");
    Console.WriteLine($"Midpoint: {range.Midpoint:yyyy-MM-dd}");
    Console.WriteLine();
}

// Exercise 4: Custom indexer — a typed matrix class
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Custom Indexer — Matrix ===");

    var matrix = new Matrix(3, 3);
    // Fill with values
    int val = 1;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++)
            matrix[r, c] = val++;

    Console.WriteLine("Matrix:");
    matrix.Print();

    Console.WriteLine($"matrix[1,1] = {matrix[1, 1]}");
    Console.WriteLine($"matrix[2,0] = {matrix[2, 0]}");

    // Bounds checking
    try
    {
        _ = matrix[5, 5];
    }
    catch (IndexOutOfRangeException ex)
    {
        Console.WriteLine($"matrix[5,5]: {ex.Message}");
    }

    // String-keyed indexer — settings store
    Console.WriteLine("\nSettings store:");
    var settings = new Settings();
    settings["theme"] = "dark";
    settings["language"] = "en";
    settings["font_size"] = "14";

    Console.WriteLine($"theme: {settings["theme"]}");
    Console.WriteLine($"language: {settings["language"]}");
    Console.WriteLine($"missing: {settings["missing"] ?? "(null)"}");
    Console.WriteLine($"Count: {settings.Count}");
    Console.WriteLine();
}

// Exercise 5: Property change notification pattern
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Property Change Notification ===");

    var product = new ObservableProduct("Widget", 9.99m, 100);
    product.PropertyChanged += (name, oldVal, newVal) =>
        Console.WriteLine($"  [{name}] changed: {oldVal} -> {newVal}");

    Console.WriteLine("Changing product properties:");
    product.Name = "Super Widget";
    product.Price = 14.99m;
    product.Stock = 50;
    product.Price = 14.99m; // Same value — should not trigger

    Console.WriteLine($"\nProduct: {product}");
    Console.WriteLine();
}

// Supporting types

class AppConfig
{
    public required string AppName { get; init; }
    public required string Version { get; init; }
    public int MaxRetries { get; init; } = 3;
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(10);
}

class UserProfile
{
    public required string Username { get; init; }
    public required string DisplayName { get; init; }
    public DateTime CreatedAt { get; } = DateTime.Now;
}

class Temperature
{
    private double _celsius;

    public double Celsius
    {
        get => _celsius;
        set => _celsius = value;
    }

    public double Fahrenheit
    {
        get => _celsius * 9.0 / 5.0 + 32.0;
        set => _celsius = (value - 32.0) * 5.0 / 9.0;
    }

    public double Kelvin
    {
        get => _celsius + 273.15;
        set
        {
            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), "Kelvin cannot be negative");
            _celsius = value - 273.15;
        }
    }
}

class ProgressBar
{
    private int _percentage;
    public int Percentage
    {
        get => _percentage;
        set => _percentage = Math.Clamp(value, 0, 100);
    }

    public override string ToString()
    {
        int filled = _percentage / 5;
        return $"[{new string('#', filled)}{new string('.', 20 - filled)}] {_percentage}%";
    }
}

class Rectangle
{
    public double Width { get; }
    public double Height { get; }
    public Rectangle(double width, double height) { Width = width; Height = height; }
    public double Area => Width * Height;
    public double Perimeter => 2 * (Width + Height);
    public bool IsSquare => Math.Abs(Width - Height) < 1e-10;
    public double Diagonal => Math.Sqrt(Width * Width + Height * Height);
}

class DateRange
{
    public DateTime Start { get; }
    public DateTime End { get; }
    public DateRange(DateTime start, DateTime end) { Start = start; End = end; }
    public int TotalDays => (int)(End - Start).TotalDays;
    public bool IsCurrent => DateTime.Now >= Start && DateTime.Now <= End;
    public DateTime Midpoint => Start.AddDays(TotalDays / 2.0);
    public override string ToString() => $"{Start:yyyy-MM-dd} to {End:yyyy-MM-dd}";
}

class Matrix
{
    private readonly double[,] _data;
    public int Rows { get; }
    public int Cols { get; }

    public Matrix(int rows, int cols) { Rows = rows; Cols = cols; _data = new double[rows, cols]; }

    public double this[int row, int col]
    {
        get
        {
            ValidateIndex(row, col);
            return _data[row, col];
        }
        set
        {
            ValidateIndex(row, col);
            _data[row, col] = value;
        }
    }

    private void ValidateIndex(int row, int col)
    {
        if (row < 0 || row >= Rows || col < 0 || col >= Cols)
            throw new IndexOutOfRangeException($"Index [{row},{col}] out of range for {Rows}x{Cols} matrix");
    }

    public void Print()
    {
        for (int r = 0; r < Rows; r++)
        {
            for (int c = 0; c < Cols; c++)
                Console.Write($"{_data[r, c],6:F0}");
            Console.WriteLine();
        }
    }
}

class Settings
{
    private readonly Dictionary<string, string> _data = new();

    public string? this[string key]
    {
        get => _data.TryGetValue(key, out var val) ? val : null;
        set { if (value is not null) _data[key] = value; else _data.Remove(key); }
    }

    public int Count => _data.Count;
}

class ObservableProduct
{
    public delegate void PropertyChangedHandler(string propertyName, object? oldValue, object? newValue);
    public event PropertyChangedHandler? PropertyChanged;

    private string _name;
    private decimal _price;
    private int _stock;

    public ObservableProduct(string name, decimal price, int stock)
    {
        _name = name; _price = price; _stock = stock;
    }

    public string Name
    {
        get => _name;
        set { if (_name != value) { var old = _name; _name = value; PropertyChanged?.Invoke(nameof(Name), old, value); } }
    }

    public decimal Price
    {
        get => _price;
        set { if (_price != value) { var old = _price; _price = value; PropertyChanged?.Invoke(nameof(Price), old, value); } }
    }

    public int Stock
    {
        get => _stock;
        set { if (_stock != value) { var old = _stock; _stock = value; PropertyChanged?.Invoke(nameof(Stock), old, value); } }
    }

    public override string ToString() => $"{Name} (${Price}, stock={Stock})";
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
