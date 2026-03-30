# File I/O

**Previous**: [Exception Handling](./14_Exception_Handling.md) | **Next**: None (Final Lesson)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `File` class static methods for quick file operations
2. Read and write text files with `StreamReader` and `StreamWriter`
3. Handle binary data with `FileStream`, `BinaryReader`, and `BinaryWriter`
4. Manage resources automatically with the `using` statement
5. Work with directories using the `Directory` and `Path` classes
6. Read and write CSV files programmatically
7. Serialize and deserialize JSON with `System.Text.Json`
8. Perform asynchronous file operations

---

File I/O is essential for any real-world application — reading configuration files, writing logs, processing data exports, and managing user-generated content. C# provides a rich set of classes in the `System.IO` namespace for working with files and directories. This lesson covers everything from simple one-line file reads to streaming binary data and async file operations, giving you the tools to handle any file-related task in your programs.

## 1. `File` Class Static Methods

The `System.IO.File` class provides convenient static methods for common file operations. These methods open the file, perform the operation, and close the file in a single call.

### 1.1 Reading Files

```csharp
using System.IO;

// Read entire file as a single string
string content = File.ReadAllText("example.txt");
Console.WriteLine(content);

// Read file as an array of lines
string[] lines = File.ReadAllLines("example.txt");
foreach (string line in lines)
{
    Console.WriteLine(line);
}
Console.WriteLine($"Total lines: {lines.Length}");

// Read file as bytes (for binary files)
byte[] bytes = File.ReadAllBytes("image.png");
Console.WriteLine($"File size: {bytes.Length} bytes");
```

### 1.2 Writing Files

```csharp
// Write a string to a file (creates or overwrites)
File.WriteAllText("output.txt", "Hello, World!\nThis is line 2.");

// Write an array of lines
string[] names = { "Alice", "Bob", "Charlie", "Diana" };
File.WriteAllLines("names.txt", names);

// Write bytes
byte[] data = { 0x48, 0x65, 0x6C, 0x6C, 0x6F };  // "Hello" in ASCII
File.WriteAllBytes("binary.dat", data);

// Append text (does not overwrite)
File.AppendAllText("log.txt", $"[{DateTime.Now}] Application started.\n");
File.AppendAllText("log.txt", $"[{DateTime.Now}] Processing data.\n");

// Append lines
File.AppendAllLines("log.txt", new[] { "Line A", "Line B" });
```

### 1.3 File Existence and Information

```csharp
string path = "example.txt";

// Check if file exists
if (File.Exists(path))
{
    Console.WriteLine($"File '{path}' exists.");

    // Get file info
    FileInfo info = new FileInfo(path);
    Console.WriteLine($"  Size: {info.Length} bytes");
    Console.WriteLine($"  Created: {info.CreationTime}");
    Console.WriteLine($"  Modified: {info.LastWriteTime}");
    Console.WriteLine($"  Read-only: {info.IsReadOnly}");
    Console.WriteLine($"  Extension: {info.Extension}");
}
else
{
    Console.WriteLine($"File '{path}' does not exist.");
}
```

### 1.4 File Operations: Copy, Move, Delete

```csharp
// Copy a file
File.Copy("source.txt", "destination.txt");
File.Copy("source.txt", "destination.txt", overwrite: true);

// Move (rename) a file
File.Move("old_name.txt", "new_name.txt");
File.Move("file.txt", @"archive\file.txt", overwrite: true);

// Delete a file
if (File.Exists("temp.txt"))
{
    File.Delete("temp.txt");
    Console.WriteLine("File deleted.");
}
```

### 1.5 Reading with Encoding

```csharp
using System.Text;

// Read with specific encoding
string utf8Content = File.ReadAllText("data.txt", Encoding.UTF8);
string latinContent = File.ReadAllText("legacy.txt", Encoding.Latin1);

// Write with specific encoding
File.WriteAllText("unicode.txt", "Hello in Unicode: Hola Welt", Encoding.UTF8);
File.WriteAllText("ascii.txt", "ASCII only", Encoding.ASCII);
```

## 2. `StreamReader` and `StreamWriter`

For large files or when you need more control, use `StreamReader` and `StreamWriter`. They process data line by line or in chunks, rather than loading the entire file into memory.

### 2.1 Reading with StreamReader

```csharp
using (StreamReader reader = new StreamReader("large_file.txt"))
{
    string line;
    int lineNumber = 0;
    while ((line = reader.ReadLine()) != null)
    {
        lineNumber++;
        Console.WriteLine($"{lineNumber}: {line}");
    }
}
```

### 2.2 StreamReader Properties and Methods

```csharp
using var reader = new StreamReader("data.txt");

// Read one character at a time
int charCode = reader.Read();
if (charCode != -1)
{
    char c = (char)charCode;
    Console.WriteLine($"First character: {c}");
}

// Read a block of characters
char[] buffer = new char[100];
int charsRead = reader.Read(buffer, 0, buffer.Length);
Console.WriteLine($"Read {charsRead} characters.");

// Peek without consuming
int nextChar = reader.Peek();
Console.WriteLine($"Next character: {(char)nextChar}");

// Read to end
string remaining = reader.ReadToEnd();
Console.WriteLine($"Remaining content length: {remaining.Length}");

// Check if at end of stream
Console.WriteLine($"End of stream: {reader.EndOfStream}");
```

### 2.3 Writing with StreamWriter

```csharp
using (StreamWriter writer = new StreamWriter("output.txt"))
{
    writer.WriteLine("First line");
    writer.WriteLine("Second line");
    writer.Write("No newline at end");
}

// Append mode
using (StreamWriter writer = new StreamWriter("output.txt", append: true))
{
    writer.WriteLine();  // Add newline after previous content
    writer.WriteLine("Appended line 1");
    writer.WriteLine("Appended line 2");
}
```

### 2.4 StreamWriter with Formatting

```csharp
using var writer = new StreamWriter("report.txt");

writer.WriteLine("=== Sales Report ===");
writer.WriteLine($"Date: {DateTime.Now:yyyy-MM-dd}");
writer.WriteLine();

// Write formatted table
writer.WriteLine($"{"Product",-20} {"Quantity",10} {"Price",10} {"Total",12}");
writer.WriteLine(new string('-', 54));

var products = new[]
{
    (Name: "Laptop", Qty: 5, Price: 999.99m),
    (Name: "Mouse", Qty: 50, Price: 29.99m),
    (Name: "Keyboard", Qty: 30, Price: 79.99m),
    (Name: "Monitor", Qty: 10, Price: 349.99m)
};

decimal grandTotal = 0;
foreach (var p in products)
{
    decimal total = p.Qty * p.Price;
    grandTotal += total;
    writer.WriteLine($"{p.Name,-20} {p.Qty,10} {p.Price,10:C} {total,12:C}");
}

writer.WriteLine(new string('-', 54));
writer.WriteLine($"{"Grand Total",-20} {"",10} {"",10} {grandTotal,12:C}");

writer.Flush();  // Ensure all data is written
```

### 2.5 Processing Large Files Efficiently

```csharp
public static void ProcessLargeFile(string inputPath, string outputPath)
{
    long lineCount = 0;
    long matchCount = 0;

    using var reader = new StreamReader(inputPath);
    using var writer = new StreamWriter(outputPath);

    string line;
    while ((line = reader.ReadLine()) != null)
    {
        lineCount++;

        // Example: filter lines containing "ERROR"
        if (line.Contains("ERROR", StringComparison.OrdinalIgnoreCase))
        {
            matchCount++;
            writer.WriteLine($"[Line {lineCount}] {line}");
        }
    }

    Console.WriteLine($"Processed {lineCount:N0} lines, found {matchCount:N0} matches.");
}
```

## 3. `FileStream` for Binary I/O

`FileStream` provides low-level byte-oriented access to files. Use it when you need precise control or are working with binary data.

### 3.1 Basic FileStream Usage

```csharp
// Writing bytes
using (FileStream fs = new FileStream("data.bin", FileMode.Create))
{
    byte[] data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
    fs.Write(data, 0, data.Length);
    Console.WriteLine($"Wrote {data.Length} bytes.");
}

// Reading bytes
using (FileStream fs = new FileStream("data.bin", FileMode.Open))
{
    byte[] buffer = new byte[fs.Length];
    int bytesRead = fs.Read(buffer, 0, buffer.Length);
    Console.WriteLine($"Read {bytesRead} bytes: {BitConverter.ToString(buffer)}");
    // Output: Read 5 bytes: 01-02-03-04-05
}
```

### 3.2 FileMode Options

```csharp
// FileMode.Create      — Create new or overwrite existing
// FileMode.CreateNew   — Create new; error if exists
// FileMode.Open        — Open existing; error if not found
// FileMode.OpenOrCreate — Open if exists, create if not
// FileMode.Append      — Open for appending; create if not found
// FileMode.Truncate    — Open and truncate to zero length

using var fs = new FileStream("log.bin", FileMode.OpenOrCreate, FileAccess.ReadWrite);
fs.Seek(0, SeekOrigin.End);  // Move to end for appending
byte[] newData = Encoding.UTF8.GetBytes("New entry\n");
fs.Write(newData, 0, newData.Length);
```

### 3.3 Seeking in a FileStream

```csharp
using var fs = new FileStream("data.bin", FileMode.Open, FileAccess.Read);

// Read from beginning
byte[] header = new byte[4];
fs.Read(header, 0, 4);
Console.WriteLine($"Header: {BitConverter.ToString(header)}");

// Seek to a specific position
fs.Seek(10, SeekOrigin.Begin);    // 10 bytes from start
fs.Seek(-5, SeekOrigin.End);      // 5 bytes from end
fs.Seek(3, SeekOrigin.Current);   // 3 bytes forward from current

Console.WriteLine($"Position: {fs.Position}");
Console.WriteLine($"Length: {fs.Length}");
```

## 4. `BinaryReader` and `BinaryWriter`

These classes wrap a stream and provide methods for reading and writing primitive types in binary format.

### 4.1 Writing with BinaryWriter

```csharp
using (FileStream fs = new FileStream("records.bin", FileMode.Create))
using (BinaryWriter writer = new BinaryWriter(fs))
{
    // Write different primitive types
    writer.Write(42);            // int (4 bytes)
    writer.Write(3.14159);       // double (8 bytes)
    writer.Write(true);          // bool (1 byte)
    writer.Write("Hello");       // string (length-prefixed)
    writer.Write('A');           // char (2 bytes in UTF-16)
    writer.Write(255L);          // long (8 bytes)
    writer.Write(9.99m);         // decimal (16 bytes)

    Console.WriteLine($"File size: {fs.Length} bytes");
}
```

### 4.2 Reading with BinaryReader

```csharp
using (FileStream fs = new FileStream("records.bin", FileMode.Open))
using (BinaryReader reader = new BinaryReader(fs))
{
    // Read in the SAME ORDER they were written
    int intVal = reader.ReadInt32();
    double doubleVal = reader.ReadDouble();
    bool boolVal = reader.ReadBoolean();
    string stringVal = reader.ReadString();
    char charVal = reader.ReadChar();
    long longVal = reader.ReadInt64();
    decimal decimalVal = reader.ReadDecimal();

    Console.WriteLine($"int: {intVal}");         // 42
    Console.WriteLine($"double: {doubleVal}");   // 3.14159
    Console.WriteLine($"bool: {boolVal}");       // True
    Console.WriteLine($"string: {stringVal}");   // Hello
    Console.WriteLine($"char: {charVal}");       // A
    Console.WriteLine($"long: {longVal}");       // 255
    Console.WriteLine($"decimal: {decimalVal}"); // 9.99
}
```

### 4.3 Storing Structured Data

```csharp
public class StudentRecord
{
    public int Id { get; set; }
    public string Name { get; set; }
    public double Gpa { get; set; }
    public bool IsActive { get; set; }

    public void WriteTo(BinaryWriter writer)
    {
        writer.Write(Id);
        writer.Write(Name);
        writer.Write(Gpa);
        writer.Write(IsActive);
    }

    public static StudentRecord ReadFrom(BinaryReader reader)
    {
        return new StudentRecord
        {
            Id = reader.ReadInt32(),
            Name = reader.ReadString(),
            Gpa = reader.ReadDouble(),
            IsActive = reader.ReadBoolean()
        };
    }

    public override string ToString() =>
        $"[{Id}] {Name}, GPA: {Gpa:F2}, Active: {IsActive}";
}
```

```csharp
// Write multiple records
List<StudentRecord> students = new List<StudentRecord>
{
    new StudentRecord { Id = 1, Name = "Alice", Gpa = 3.9, IsActive = true },
    new StudentRecord { Id = 2, Name = "Bob", Gpa = 3.5, IsActive = true },
    new StudentRecord { Id = 3, Name = "Charlie", Gpa = 2.8, IsActive = false }
};

using (var fs = new FileStream("students.bin", FileMode.Create))
using (var writer = new BinaryWriter(fs))
{
    writer.Write(students.Count);  // Write record count first
    foreach (var student in students)
    {
        student.WriteTo(writer);
    }
}

// Read them back
using (var fs = new FileStream("students.bin", FileMode.Open))
using (var reader = new BinaryReader(fs))
{
    int count = reader.ReadInt32();
    Console.WriteLine($"Reading {count} student records:");

    for (int i = 0; i < count; i++)
    {
        StudentRecord s = StudentRecord.ReadFrom(reader);
        Console.WriteLine($"  {s}");
    }
}
// Output:
// Reading 3 student records:
//   [1] Alice, GPA: 3.90, Active: True
//   [2] Bob, GPA: 3.50, Active: True
//   [3] Charlie, GPA: 2.80, Active: False
```

## 5. `using` Statement for Automatic Resource Cleanup

As covered in the Exception Handling lesson, the `using` statement ensures streams and other `IDisposable` objects are properly closed.

### 5.1 Traditional vs Modern Syntax

```csharp
// Traditional using statement (block scope)
using (StreamReader reader = new StreamReader("file.txt"))
{
    string content = reader.ReadToEnd();
    Console.WriteLine(content);
}
// reader is disposed here

// Modern using declaration (C# 8+, method scope)
using var reader2 = new StreamReader("file.txt");
string content2 = reader2.ReadToEnd();
Console.WriteLine(content2);
// reader2 is disposed at end of enclosing scope
```

### 5.2 Multiple Resources

```csharp
// Chained using statements
using var input = new StreamReader("input.txt");
using var output = new StreamWriter("output.txt");
using var errorLog = new StreamWriter("errors.txt");

string line;
int lineNum = 0;
while ((line = input.ReadLine()) != null)
{
    lineNum++;
    try
    {
        string processed = ProcessLine(line);
        output.WriteLine(processed);
    }
    catch (Exception ex)
    {
        errorLog.WriteLine($"Line {lineNum}: {ex.Message}");
    }
}
// All three streams are disposed in reverse order
```

### 5.3 Ensuring Cleanup on Exceptions

```csharp
public static void SafeFileCopy(string source, string destination)
{
    using var reader = new FileStream(source, FileMode.Open, FileAccess.Read);
    using var writer = new FileStream(destination, FileMode.Create, FileAccess.Write);

    byte[] buffer = new byte[8192];
    int bytesRead;
    long totalBytes = 0;

    while ((bytesRead = reader.Read(buffer, 0, buffer.Length)) > 0)
    {
        writer.Write(buffer, 0, bytesRead);
        totalBytes += bytesRead;
    }

    Console.WriteLine($"Copied {totalBytes:N0} bytes from '{source}' to '{destination}'.");
    // Both streams are properly closed even if an exception occurs
}
```

## 6. `Directory` Class

The `System.IO.Directory` class provides static methods for creating, moving, and listing directories.

### 6.1 Creating and Checking Directories

```csharp
string dirPath = @"output\reports\2026";

// Create directory (and all parent directories)
if (!Directory.Exists(dirPath))
{
    Directory.CreateDirectory(dirPath);
    Console.WriteLine($"Created: {dirPath}");
}

// Get directory info
DirectoryInfo dirInfo = new DirectoryInfo(dirPath);
Console.WriteLine($"Full path: {dirInfo.FullName}");
Console.WriteLine($"Created: {dirInfo.CreationTime}");
Console.WriteLine($"Parent: {dirInfo.Parent?.Name}");
```

### 6.2 Listing Files and Directories

```csharp
string searchDir = @"C:\Projects";

// Get all files in a directory
string[] files = Directory.GetFiles(searchDir);
foreach (string file in files)
{
    Console.WriteLine(file);
}

// Get files with a pattern
string[] txtFiles = Directory.GetFiles(searchDir, "*.txt");
string[] csFiles = Directory.GetFiles(searchDir, "*.cs", SearchOption.AllDirectories);
Console.WriteLine($"Found {csFiles.Length} .cs files recursively.");

// Get subdirectories
string[] subDirs = Directory.GetDirectories(searchDir);
foreach (string dir in subDirs)
{
    Console.WriteLine($"  Directory: {dir}");
}

// Enumerate (lazy, better for large directories)
foreach (string file in Directory.EnumerateFiles(searchDir, "*.*", SearchOption.AllDirectories))
{
    FileInfo fi = new FileInfo(file);
    if (fi.Length > 1024 * 1024)  // > 1MB
    {
        Console.WriteLine($"Large file: {fi.Name} ({fi.Length / 1024.0 / 1024.0:F2} MB)");
    }
}
```

### 6.3 Directory Operations

```csharp
// Move (rename) a directory
Directory.Move("old_folder", "new_folder");

// Delete a directory
Directory.Delete("temp_folder");                      // Must be empty
Directory.Delete("temp_folder", recursive: true);     // Delete all contents

// Get current and special directories
string currentDir = Directory.GetCurrentDirectory();
Console.WriteLine($"Current: {currentDir}");

string tempDir = Path.GetTempPath();
Console.WriteLine($"Temp: {tempDir}");

string desktopDir = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
Console.WriteLine($"Desktop: {desktopDir}");
```

## 7. `Path` Class

The `System.IO.Path` class provides static methods for manipulating file and directory path strings in a platform-safe way.

### 7.1 Path Manipulation

```csharp
string filePath = @"C:\Users\alice\Documents\report.pdf";

Console.WriteLine($"File name: {Path.GetFileName(filePath)}");
// "report.pdf"

Console.WriteLine($"File name without extension: {Path.GetFileNameWithoutExtension(filePath)}");
// "report"

Console.WriteLine($"Extension: {Path.GetExtension(filePath)}");
// ".pdf"

Console.WriteLine($"Directory: {Path.GetDirectoryName(filePath)}");
// "C:\Users\alice\Documents"

Console.WriteLine($"Full path: {Path.GetFullPath(filePath)}");
// "C:\Users\alice\Documents\report.pdf"

Console.WriteLine($"Root: {Path.GetPathRoot(filePath)}");
// "C:\"

Console.WriteLine($"Has extension: {Path.HasExtension(filePath)}");
// True
```

### 7.2 Combining Paths

Always use `Path.Combine` rather than string concatenation. It handles path separators correctly across platforms.

```csharp
// GOOD: Path.Combine handles separators
string outputDir = "output";
string fileName = "report.txt";
string fullPath = Path.Combine(outputDir, fileName);
Console.WriteLine(fullPath);  // "output/report.txt" or "output\report.txt"

// Multiple segments
string path = Path.Combine("root", "sub1", "sub2", "file.txt");
Console.WriteLine(path);  // "root/sub1/sub2/file.txt"

// BAD: manual concatenation — error-prone
// string badPath = outputDir + "/" + fileName;  // Won't work on all platforms
```

### 7.3 Changing File Extensions and Creating Temp Files

```csharp
// Change extension
string original = "document.txt";
string backup = Path.ChangeExtension(original, ".bak");
Console.WriteLine(backup);  // "document.bak"

// Create temporary file paths
string tempFile = Path.GetTempFileName();
Console.WriteLine($"Temp file: {tempFile}");
// e.g., "/tmp/tmpABC123.tmp"

string tempDir = Path.GetTempPath();
Console.WriteLine($"Temp dir: {tempDir}");

// Generate random file name
string randomName = Path.GetRandomFileName();
Console.WriteLine($"Random name: {randomName}");
// e.g., "4k2znmos.hsf"
```

### 7.4 Building a File Manager Utility

```csharp
public static class FileManager
{
    public static void OrganizeByExtension(string sourceDir, string targetDir)
    {
        if (!Directory.Exists(sourceDir))
            throw new DirectoryNotFoundException($"Source not found: {sourceDir}");

        string[] files = Directory.GetFiles(sourceDir);
        int movedCount = 0;

        foreach (string filePath in files)
        {
            string extension = Path.GetExtension(filePath).TrimStart('.').ToLower();
            if (string.IsNullOrEmpty(extension))
                extension = "no_extension";

            string destDir = Path.Combine(targetDir, extension);
            Directory.CreateDirectory(destDir);

            string destPath = Path.Combine(destDir, Path.GetFileName(filePath));

            // Handle duplicates
            if (File.Exists(destPath))
            {
                string nameWithoutExt = Path.GetFileNameWithoutExtension(filePath);
                string ext = Path.GetExtension(filePath);
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                destPath = Path.Combine(destDir, $"{nameWithoutExt}_{timestamp}{ext}");
            }

            File.Move(filePath, destPath);
            movedCount++;
            Console.WriteLine($"Moved: {Path.GetFileName(filePath)} -> {extension}/");
        }

        Console.WriteLine($"\nOrganized {movedCount} files into folders.");
    }
}
```

## 8. Reading and Writing CSV Files

CSV (Comma-Separated Values) is a common format for data exchange. Here is how to handle it in C# without external libraries.

### 8.1 Writing CSV

```csharp
public class CsvWriter
{
    public static void WriteCsv(string path, string[] headers, List<string[]> rows)
    {
        using var writer = new StreamWriter(path);

        // Write header
        writer.WriteLine(string.Join(",", headers));

        // Write data rows
        foreach (string[] row in rows)
        {
            string[] escapedFields = row.Select(EscapeCsvField).ToArray();
            writer.WriteLine(string.Join(",", escapedFields));
        }
    }

    private static string EscapeCsvField(string field)
    {
        if (field.Contains(',') || field.Contains('"') || field.Contains('\n'))
        {
            // Escape quotes by doubling them, wrap in quotes
            return $"\"{field.Replace("\"", "\"\"")}\"";
        }
        return field;
    }
}
```

```csharp
string[] headers = { "Name", "Age", "Email", "City" };
var rows = new List<string[]>
{
    new[] { "Alice", "30", "alice@example.com", "New York" },
    new[] { "Bob", "25", "bob@example.com", "San Francisco" },
    new[] { "Charlie", "35", "charlie@example.com", "Chicago" },
    new[] { "Diana, Jr.", "28", "diana@example.com", "Los Angeles" }  // Note the comma in name
};

CsvWriter.WriteCsv("people.csv", headers, rows);
// Output file:
// Name,Age,Email,City
// Alice,30,alice@example.com,New York
// Bob,25,bob@example.com,San Francisco
// Charlie,35,charlie@example.com,Chicago
// "Diana, Jr.",28,diana@example.com,Los Angeles
```

### 8.2 Reading CSV

```csharp
public class CsvReader
{
    public static (string[] Headers, List<string[]> Rows) ReadCsv(string path)
    {
        using var reader = new StreamReader(path);

        string headerLine = reader.ReadLine();
        if (headerLine == null)
            throw new InvalidOperationException("CSV file is empty.");

        string[] headers = ParseCsvLine(headerLine);
        List<string[]> rows = new List<string[]>();

        string line;
        while ((line = reader.ReadLine()) != null)
        {
            if (!string.IsNullOrWhiteSpace(line))
            {
                rows.Add(ParseCsvLine(line));
            }
        }

        return (headers, rows);
    }

    private static string[] ParseCsvLine(string line)
    {
        List<string> fields = new List<string>();
        bool inQuotes = false;
        int fieldStart = 0;
        var current = new System.Text.StringBuilder();

        for (int i = 0; i < line.Length; i++)
        {
            char c = line[i];

            if (inQuotes)
            {
                if (c == '"')
                {
                    if (i + 1 < line.Length && line[i + 1] == '"')
                    {
                        current.Append('"');
                        i++;  // Skip the escaped quote
                    }
                    else
                    {
                        inQuotes = false;
                    }
                }
                else
                {
                    current.Append(c);
                }
            }
            else
            {
                if (c == '"')
                {
                    inQuotes = true;
                }
                else if (c == ',')
                {
                    fields.Add(current.ToString());
                    current.Clear();
                }
                else
                {
                    current.Append(c);
                }
            }
        }

        fields.Add(current.ToString());
        return fields.ToArray();
    }
}
```

```csharp
var (headers, rows) = CsvReader.ReadCsv("people.csv");

Console.WriteLine("Headers: " + string.Join(" | ", headers));

foreach (string[] row in rows)
{
    Console.WriteLine(string.Join(" | ", row));
}
// Headers: Name | Age | Email | City
// Alice | 30 | alice@example.com | New York
// Bob | 25 | bob@example.com | San Francisco
// Charlie | 35 | charlie@example.com | Chicago
// Diana, Jr. | 28 | diana@example.com | Los Angeles
```

### 8.3 CSV to Objects

```csharp
public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }
    public string Email { get; set; }
    public string City { get; set; }

    public override string ToString() => $"{Name} ({Age}), {Email}, {City}";
}

public static List<Person> LoadPeople(string csvPath)
{
    var (headers, rows) = CsvReader.ReadCsv(csvPath);
    List<Person> people = new List<Person>();

    foreach (string[] row in rows)
    {
        people.Add(new Person
        {
            Name = row[0],
            Age = int.Parse(row[1]),
            Email = row[2],
            City = row[3]
        });
    }

    return people;
}
```

## 9. Basic JSON with `System.Text.Json`

`System.Text.Json` is the built-in JSON library in modern .NET. It provides serialization (object to JSON) and deserialization (JSON to object).

### 9.1 Serialization (Object to JSON)

```csharp
using System.Text.Json;

public class Product
{
    public int Id { get; set; }
    public string Name { get; set; }
    public decimal Price { get; set; }
    public string[] Tags { get; set; }
    public bool InStock { get; set; }
}

Product product = new Product
{
    Id = 1,
    Name = "Mechanical Keyboard",
    Price = 149.99m,
    Tags = new[] { "electronics", "peripherals", "gaming" },
    InStock = true
};

// Serialize to JSON string
string json = JsonSerializer.Serialize(product);
Console.WriteLine(json);
// {"Id":1,"Name":"Mechanical Keyboard","Price":149.99,"Tags":["electronics","peripherals","gaming"],"InStock":true}

// Pretty-printed JSON
var options = new JsonSerializerOptions { WriteIndented = true };
string prettyJson = JsonSerializer.Serialize(product, options);
Console.WriteLine(prettyJson);
```

### 9.2 Deserialization (JSON to Object)

```csharp
string jsonInput = @"{
    ""Id"": 2,
    ""Name"": ""Wireless Mouse"",
    ""Price"": 59.99,
    ""Tags"": [""electronics"", ""peripherals""],
    ""InStock"": true
}";

Product deserialized = JsonSerializer.Deserialize<Product>(jsonInput);
Console.WriteLine($"Product: {deserialized.Name}, Price: ${deserialized.Price}");
// Product: Wireless Mouse, Price: $59.99

// Deserialize a list
string jsonArray = @"[
    {""Id"": 1, ""Name"": ""Laptop"", ""Price"": 999.99, ""Tags"": [], ""InStock"": true},
    {""Id"": 2, ""Name"": ""Tablet"", ""Price"": 499.99, ""Tags"": [], ""InStock"": false}
]";

List<Product> products = JsonSerializer.Deserialize<List<Product>>(jsonArray);
foreach (Product p in products)
{
    Console.WriteLine($"  {p.Name}: ${p.Price} (In stock: {p.InStock})");
}
```

### 9.3 Reading and Writing JSON Files

```csharp
// Write JSON to file
public static void SaveToJson<T>(string path, T data)
{
    var options = new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase
    };

    string json = JsonSerializer.Serialize(data, options);
    File.WriteAllText(path, json);
}

// Read JSON from file
public static T LoadFromJson<T>(string path)
{
    string json = File.ReadAllText(path);
    var options = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true
    };
    return JsonSerializer.Deserialize<T>(json, options);
}
```

```csharp
// Save products
var products = new List<Product>
{
    new Product { Id = 1, Name = "Laptop", Price = 999.99m, Tags = new[] { "tech" }, InStock = true },
    new Product { Id = 2, Name = "Chair", Price = 249.99m, Tags = new[] { "furniture" }, InStock = true }
};

SaveToJson("products.json", products);

// Load them back
var loaded = LoadFromJson<List<Product>>("products.json");
foreach (var p in loaded)
{
    Console.WriteLine($"{p.Name}: ${p.Price}");
}
```

### 9.4 Customizing Serialization

```csharp
using System.Text.Json.Serialization;

public class Config
{
    [JsonPropertyName("app_name")]
    public string AppName { get; set; }

    [JsonPropertyName("version")]
    public string Version { get; set; }

    [JsonPropertyName("max_connections")]
    public int MaxConnections { get; set; }

    [JsonIgnore]  // Excluded from serialization
    public string InternalSecret { get; set; }

    [JsonPropertyName("features")]
    public Dictionary<string, bool> Features { get; set; }
}
```

```csharp
var config = new Config
{
    AppName = "MyApp",
    Version = "2.1.0",
    MaxConnections = 100,
    InternalSecret = "super-secret",
    Features = new Dictionary<string, bool>
    {
        ["dark_mode"] = true,
        ["beta_features"] = false
    }
};

string json = JsonSerializer.Serialize(config, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(json);
// {
//   "app_name": "MyApp",
//   "version": "2.1.0",
//   "max_connections": 100,
//   "features": {
//     "dark_mode": true,
//     "beta_features": false
//   }
// }
// Note: InternalSecret is not included (JsonIgnore)
```

## 10. Async File I/O

For applications that need to remain responsive (especially UI apps and web servers), async file operations prevent blocking the calling thread.

### 10.1 Async Read and Write

```csharp
using System.Threading.Tasks;

// Async read
public static async Task<string> ReadFileAsync(string path)
{
    string content = await File.ReadAllTextAsync(path);
    return content;
}

// Async write
public static async Task WriteFileAsync(string path, string content)
{
    await File.WriteAllTextAsync(path, content);
}

// Async read lines
public static async Task<string[]> ReadLinesAsync(string path)
{
    return await File.ReadAllLinesAsync(path);
}
```

### 10.2 Async Stream Operations

```csharp
public static async Task ProcessFileAsync(string inputPath, string outputPath)
{
    using var reader = new StreamReader(inputPath);
    using var writer = new StreamWriter(outputPath);

    string line;
    int lineCount = 0;

    while ((line = await reader.ReadLineAsync()) != null)
    {
        lineCount++;
        string processed = $"[{lineCount:D4}] {line.ToUpper()}";
        await writer.WriteLineAsync(processed);
    }

    Console.WriteLine($"Processed {lineCount} lines asynchronously.");
}
```

### 10.3 Async File Copy

```csharp
public static async Task CopyFileAsync(string source, string destination,
    int bufferSize = 81920)
{
    using var sourceStream = new FileStream(source, FileMode.Open,
        FileAccess.Read, FileShare.Read, bufferSize, useAsync: true);
    using var destStream = new FileStream(destination, FileMode.Create,
        FileAccess.Write, FileShare.None, bufferSize, useAsync: true);

    byte[] buffer = new byte[bufferSize];
    int bytesRead;
    long totalBytes = 0;

    while ((bytesRead = await sourceStream.ReadAsync(buffer, 0, buffer.Length)) > 0)
    {
        await destStream.WriteAsync(buffer, 0, bytesRead);
        totalBytes += bytesRead;
    }

    Console.WriteLine($"Copied {totalBytes:N0} bytes asynchronously.");
}
```

### 10.4 Calling Async Methods

```csharp
// In an async Main method (C# 7.1+)
static async Task Main(string[] args)
{
    // Write a file
    await File.WriteAllTextAsync("async_test.txt", "Hello from async!");

    // Read it back
    string content = await File.ReadAllTextAsync("async_test.txt");
    Console.WriteLine(content);

    // Process files
    await ProcessFileAsync("input.txt", "output.txt");

    // Copy a file
    await CopyFileAsync("large_file.dat", "large_file_backup.dat");

    // Read all lines
    string[] lines = await File.ReadAllLinesAsync("data.txt");
    Console.WriteLine($"Read {lines.Length} lines.");
}
```

### 10.5 When to Use Async File I/O

```csharp
// USE ASYNC when:
// - Building a web application (ASP.NET Core) — frees threads for other requests
// - Building a desktop/mobile UI — prevents freezing the UI thread
// - Processing multiple files concurrently

// Example: process multiple files in parallel
public static async Task ProcessMultipleFiles(string[] filePaths)
{
    var tasks = filePaths.Select(async path =>
    {
        string content = await File.ReadAllTextAsync(path);
        int wordCount = content.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
        Console.WriteLine($"{Path.GetFileName(path)}: {wordCount} words");
        return wordCount;
    });

    int[] results = await Task.WhenAll(tasks);
    Console.WriteLine($"Total words across all files: {results.Sum()}");
}

// DO NOT BOTHER with async for:
// - Simple console apps with sequential file operations
// - Very small files where the overhead of async is not worth it
```

## 11. Practice Problems

1. **Line Counter**: Write a program that takes a file path as a command-line argument and displays: total lines, blank lines, lines starting with "#" (comments), and the average line length. Handle the case where the file does not exist with a friendly error message. Use `StreamReader` for memory efficiency.

2. **File Backup Utility**: Create a program that backs up a directory. Given a source directory path: (a) create a backup directory named `backup_YYYYMMDD_HHmmss`, (b) recursively copy all files and subdirectories, (c) skip files larger than 10MB (print a warning), (d) print a summary showing files copied, files skipped, and total bytes copied. Use `Path.Combine` for all path construction.

3. **CSV Analyzer**: Write a program that reads a CSV file and provides an interactive menu: (a) display all records in a formatted table, (b) sort by any column (ascending or descending), (c) filter rows where a column matches a value, (d) add a new row, (e) save the modified data back to CSV. Handle quoted fields with commas inside them correctly.

4. **JSON Config Manager**: Create a configuration manager that reads/writes a JSON config file. It should support: (a) getting a value by key (supporting nested keys like "database.host"), (b) setting a value by key, (c) listing all keys, (d) deleting a key, (e) saving changes back to the JSON file. Use `System.Text.Json` with `JsonDocument` for dynamic access. The config file should be auto-created with defaults if it does not exist.

5. **Log File Analyzer**: Write an async program that processes multiple log files in parallel. Each log file has lines in the format `[TIMESTAMP] [LEVEL] Message`. The program should: (a) count occurrences of each log level (INFO, WARN, ERROR, DEBUG), (b) find all ERROR lines and write them to an `errors_summary.txt` file, (c) compute the time range covered by each file, (d) generate a combined report across all files. Use `async`/`await` and `Task.WhenAll` for parallel processing.
