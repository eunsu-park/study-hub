/*
 * Exercises for Lesson 15: File I/O
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System.Text;
using System.Text.Json;

// Exercise 1: Basic file read/write operations
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Basic File Operations ===");

    string dir = Path.Combine(Path.GetTempPath(), "csharp_exercises");
    Directory.CreateDirectory(dir);

    // Write text file
    string filePath = Path.Combine(dir, "sample.txt");
    string[] lines = { "Hello, File I/O!", "This is line 2.", "C# makes it easy.", "Line 4 here.", "Final line." };
    File.WriteAllLines(filePath, lines);
    Console.WriteLine($"Wrote {lines.Length} lines to {filePath}");

    // Read all text
    string content = File.ReadAllText(filePath);
    Console.WriteLine($"ReadAllText ({content.Length} chars):");
    Console.WriteLine($"  {content.Replace("\n", "\n  ")}");

    // Read line by line
    string[] readLines = File.ReadAllLines(filePath);
    Console.WriteLine($"ReadAllLines ({readLines.Length} lines):");
    for (int i = 0; i < readLines.Length; i++)
        Console.WriteLine($"  [{i}] {readLines[i]}");

    // Append to file
    File.AppendAllText(filePath, "Appended line.\n");
    Console.WriteLine($"\nAfter append: {File.ReadAllLines(filePath).Length} lines");

    // File info
    var info = new FileInfo(filePath);
    Console.WriteLine($"Size: {info.Length} bytes");
    Console.WriteLine($"Created: {info.CreationTime:yyyy-MM-dd HH:mm:ss}");
    Console.WriteLine($"Extension: {info.Extension}");
    Console.WriteLine();

    // Cleanup
    File.Delete(filePath);
}

// Exercise 2: Stream-based reading and writing
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Stream Operations ===");

    string dir = Path.Combine(Path.GetTempPath(), "csharp_exercises");
    Directory.CreateDirectory(dir);
    string filePath = Path.Combine(dir, "stream_test.txt");

    // Write with StreamWriter
    using (var writer = new StreamWriter(filePath, false, Encoding.UTF8))
    {
        for (int i = 1; i <= 100; i++)
            writer.WriteLine($"Line {i}: {(i % 15 == 0 ? "FizzBuzz" : i % 3 == 0 ? "Fizz" : i % 5 == 0 ? "Buzz" : i.ToString())}");
    }
    Console.WriteLine("Wrote 100 lines with StreamWriter");

    // Read with StreamReader — count specific lines
    int fizzCount = 0, buzzCount = 0, fizzBuzzCount = 0;
    using (var reader = new StreamReader(filePath))
    {
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Contains("FizzBuzz")) fizzBuzzCount++;
            else if (line.Contains("Fizz")) fizzCount++;
            else if (line.Contains("Buzz")) buzzCount++;
        }
    }
    Console.WriteLine($"Fizz: {fizzCount}, Buzz: {buzzCount}, FizzBuzz: {fizzBuzzCount}");

    // Binary write and read
    string binPath = Path.Combine(dir, "data.bin");
    using (var bw = new BinaryWriter(File.Open(binPath, FileMode.Create)))
    {
        bw.Write(42);
        bw.Write(3.14159);
        bw.Write("Hello Binary");
        bw.Write(true);
    }

    using (var br = new BinaryReader(File.Open(binPath, FileMode.Open)))
    {
        int intVal = br.ReadInt32();
        double dblVal = br.ReadDouble();
        string strVal = br.ReadString();
        bool boolVal = br.ReadBoolean();
        Console.WriteLine($"\nBinary read: int={intVal}, double={dblVal:F5}, string=\"{strVal}\", bool={boolVal}");
    }

    // Cleanup
    File.Delete(filePath);
    File.Delete(binPath);
    Console.WriteLine();
}

// Exercise 3: CSV parsing and generation
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: CSV Parsing ===");

    string dir = Path.Combine(Path.GetTempPath(), "csharp_exercises");
    Directory.CreateDirectory(dir);
    string csvPath = Path.Combine(dir, "employees.csv");

    // Generate CSV
    var employees = new List<(string Name, string Dept, decimal Salary)>
    {
        ("Alice Johnson", "Engineering", 95000),
        ("Bob Smith", "Marketing", 72000),
        ("Charlie Brown", "Engineering", 88000),
        ("Diana Prince", "HR", 68000),
        ("Eve Wilson", "Engineering", 105000),
        ("Frank Miller", "Marketing", 76000),
        ("Grace Lee", "HR", 71000)
    };

    var csvBuilder = new StringBuilder();
    csvBuilder.AppendLine("Name,Department,Salary");
    foreach (var (name, dept, salary) in employees)
        csvBuilder.AppendLine($"{name},{dept},{salary}");
    File.WriteAllText(csvPath, csvBuilder.ToString());
    Console.WriteLine($"Wrote CSV with {employees.Count} records");

    // Parse CSV
    string[] csvLines = File.ReadAllLines(csvPath);
    var parsed = new List<(string Name, string Dept, decimal Salary)>();
    foreach (string line in csvLines.Skip(1)) // skip header
    {
        string[] fields = line.Split(',');
        if (fields.Length == 3)
            parsed.Add((fields[0], fields[1], decimal.Parse(fields[2])));
    }

    // Analytics
    Console.WriteLine($"\nParsed {parsed.Count} employees:");
    var byDept = parsed.GroupBy(e => e.Dept);
    foreach (var dept in byDept.OrderBy(g => g.Key))
    {
        decimal avg = dept.Average(e => e.Salary);
        decimal max = dept.Max(e => e.Salary);
        Console.WriteLine($"  {dept.Key,-15} count={dept.Count()}, avg=${avg:N0}, max=${max:N0}");
    }

    decimal totalPayroll = parsed.Sum(e => e.Salary);
    Console.WriteLine($"\nTotal payroll: ${totalPayroll:N0}");
    Console.WriteLine($"Highest paid: {parsed.MaxBy(e => e.Salary).Name}");

    File.Delete(csvPath);
    Console.WriteLine();
}

// Exercise 4: JSON serialization and deserialization
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: JSON Operations ===");

    string dir = Path.Combine(Path.GetTempPath(), "csharp_exercises");
    Directory.CreateDirectory(dir);
    string jsonPath = Path.Combine(dir, "config.json");

    // Create and serialize a configuration object
    var config = new AppConfiguration
    {
        AppName = "MyApp",
        Version = "2.1.0",
        Debug = false,
        Database = new DatabaseConfig
        {
            Host = "localhost",
            Port = 5432,
            Name = "myapp_db"
        },
        Features = new[] { "auth", "logging", "caching" },
        Limits = new Dictionary<string, int>
        {
            ["max_connections"] = 100,
            ["timeout_seconds"] = 30,
            ["max_retries"] = 3
        }
    };

    var options = new JsonSerializerOptions { WriteIndented = true };
    string json = JsonSerializer.Serialize(config, options);
    File.WriteAllText(jsonPath, json);
    Console.WriteLine($"Serialized config to JSON ({json.Length} chars):");
    Console.WriteLine(json);

    // Deserialize
    string readJson = File.ReadAllText(jsonPath);
    var loaded = JsonSerializer.Deserialize<AppConfiguration>(readJson);
    Console.WriteLine($"\nDeserialized:");
    Console.WriteLine($"  App: {loaded?.AppName} v{loaded?.Version}");
    Console.WriteLine($"  DB: {loaded?.Database?.Host}:{loaded?.Database?.Port}/{loaded?.Database?.Name}");
    Console.WriteLine($"  Features: [{string.Join(", ", loaded?.Features ?? Array.Empty<string>())}]");
    Console.WriteLine($"  Max connections: {loaded?.Limits?.GetValueOrDefault("max_connections")}");

    File.Delete(jsonPath);
    Console.WriteLine();
}

// Exercise 5: Directory operations and file search
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Directory Operations ===");

    string baseDir = Path.Combine(Path.GetTempPath(), "csharp_exercises", "project");

    // Create directory structure
    string[] dirs = { "src", "src/models", "src/services", "tests", "docs" };
    foreach (string d in dirs)
        Directory.CreateDirectory(Path.Combine(baseDir, d));

    // Create sample files
    (string path, string content)[] files =
    {
        ("src/Program.cs", "// Main entry point\nclass Program { }"),
        ("src/models/User.cs", "// User model\nclass User { public string Name; }"),
        ("src/models/Product.cs", "// Product model\nclass Product { public decimal Price; }"),
        ("src/services/AuthService.cs", "// Auth service\nclass AuthService { }"),
        ("tests/UserTests.cs", "// User tests\nclass UserTests { }"),
        ("docs/README.md", "# Project\nSample project structure"),
        ("docs/API.md", "# API\nAPI documentation")
    };

    foreach (var (path, content) in files)
        File.WriteAllText(Path.Combine(baseDir, path), content);

    Console.WriteLine($"Created {dirs.Length} directories and {files.Length} files");

    // List all files recursively
    Console.WriteLine("\nAll files:");
    foreach (string file in Directory.GetFiles(baseDir, "*.*", SearchOption.AllDirectories))
    {
        string relative = Path.GetRelativePath(baseDir, file);
        long size = new FileInfo(file).Length;
        Console.WriteLine($"  {relative,-35} ({size} bytes)");
    }

    // Search for .cs files
    string[] csFiles = Directory.GetFiles(baseDir, "*.cs", SearchOption.AllDirectories);
    Console.WriteLine($"\n.cs files found: {csFiles.Length}");
    foreach (string f in csFiles)
        Console.WriteLine($"  {Path.GetFileName(f)}");

    // Search for files containing "model"
    Console.WriteLine("\nFiles containing 'model':");
    foreach (string file in Directory.GetFiles(baseDir, "*.*", SearchOption.AllDirectories))
    {
        string content = File.ReadAllText(file);
        if (content.Contains("model", StringComparison.OrdinalIgnoreCase))
            Console.WriteLine($"  {Path.GetRelativePath(baseDir, file)}");
    }

    // Directory size calculation
    long totalSize = Directory.GetFiles(baseDir, "*.*", SearchOption.AllDirectories)
        .Sum(f => new FileInfo(f).Length);
    Console.WriteLine($"\nTotal project size: {totalSize} bytes");

    // Cleanup
    Directory.Delete(baseDir, recursive: true);
    Console.WriteLine("Cleaned up temporary project directory.");
    Console.WriteLine();
}

// Supporting types for JSON exercise

class AppConfiguration
{
    public string AppName { get; set; } = "";
    public string Version { get; set; } = "";
    public bool Debug { get; set; }
    public DatabaseConfig? Database { get; set; }
    public string[]? Features { get; set; }
    public Dictionary<string, int>? Limits { get; set; }
}

class DatabaseConfig
{
    public string Host { get; set; } = "";
    public int Port { get; set; }
    public string Name { get; set; } = "";
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
