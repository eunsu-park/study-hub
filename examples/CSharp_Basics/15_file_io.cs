// Lesson 15: File I/O
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

// Create a temporary working directory for demonstrations
string workDir = Path.Combine(Path.GetTempPath(), "csharp_fileio_demo");
Directory.CreateDirectory(workDir);
Console.WriteLine($"Working directory: {workDir}\n");

// =============================================================================
// FILE — Simple Read/Write (all-at-once)
// =============================================================================
Console.WriteLine("=== File Class (Simple I/O) ===");

// Write all text
string filePath = Path.Combine(workDir, "hello.txt");
File.WriteAllText(filePath, "Hello, C# File I/O!\nThis is line 2.\nThis is line 3.");
Console.WriteLine($"Wrote: {filePath}");

// Read all text
string content = File.ReadAllText(filePath);
Console.WriteLine($"ReadAllText:\n{content}\n");

// Read all lines (returns string[])
string[] lines = File.ReadAllLines(filePath);
Console.WriteLine($"ReadAllLines ({lines.Length} lines):");
for (int i = 0; i < lines.Length; i++)
{
    Console.WriteLine($"  [{i}] {lines[i]}");
}

// Write all lines
string linesPath = Path.Combine(workDir, "lines.txt");
string[] data = { "Apple", "Banana", "Cherry", "Date", "Elderberry" };
File.WriteAllLines(linesPath, data);
Console.WriteLine($"\nWrote {data.Length} lines to: {linesPath}");

// Append text
File.AppendAllText(linesPath, "Fig\nGrape\n");
Console.WriteLine("Appended 2 more lines.");
Console.WriteLine($"Total lines: {File.ReadAllLines(linesPath).Length}");

// =============================================================================
// STREAMWRITER AND STREAMREADER
// =============================================================================
Console.WriteLine("\n=== StreamWriter / StreamReader ===");

string logPath = Path.Combine(workDir, "log.txt");

// StreamWriter — write line by line with buffering
using (var writer = new StreamWriter(logPath))
{
    writer.WriteLine($"Log started: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
    for (int i = 1; i <= 5; i++)
    {
        writer.WriteLine($"[{i:D3}] Event: operation_{i} completed.");
    }
    writer.WriteLine($"Log ended: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
}
Console.WriteLine($"Wrote log to: {logPath}");

// StreamReader — read line by line
Console.WriteLine("Reading log:");
using (var reader = new StreamReader(logPath))
{
    string? line;
    while ((line = reader.ReadLine()) != null)
    {
        Console.WriteLine($"  {line}");
    }
}

// StreamWriter with append mode
using (var writer = new StreamWriter(logPath, append: true))
{
    writer.WriteLine("--- Appended entry ---");
}

// Using declaration syntax (C# 8+)
using var appendReader = new StreamReader(logPath);
int lineCount = 0;
while (appendReader.ReadLine() != null) lineCount++;
Console.WriteLine($"Total lines after append: {lineCount}");

// =============================================================================
// BINARY READ/WRITE
// =============================================================================
Console.WriteLine("\n=== Binary I/O ===");

string binPath = Path.Combine(workDir, "data.bin");

// Write binary data
using (var bw = new BinaryWriter(File.Open(binPath, FileMode.Create)))
{
    bw.Write(42);           // int (4 bytes)
    bw.Write(3.14159);      // double (8 bytes)
    bw.Write(true);         // bool (1 byte)
    bw.Write("Hello");      // length-prefixed string
}
Console.WriteLine($"Wrote binary data to: {binPath}");

// Read binary data
using (var br = new BinaryReader(File.Open(binPath, FileMode.Open)))
{
    int intVal = br.ReadInt32();
    double dblVal = br.ReadDouble();
    bool boolVal = br.ReadBoolean();
    string strVal = br.ReadString();

    Console.WriteLine($"  int: {intVal}");
    Console.WriteLine($"  double: {dblVal}");
    Console.WriteLine($"  bool: {boolVal}");
    Console.WriteLine($"  string: \"{strVal}\"");
}

// =============================================================================
// FILE INFO AND METADATA
// =============================================================================
Console.WriteLine("\n=== File Info ===");

var fileInfo = new FileInfo(filePath);
Console.WriteLine($"Name:          {fileInfo.Name}");
Console.WriteLine($"Full path:     {fileInfo.FullName}");
Console.WriteLine($"Extension:     {fileInfo.Extension}");
Console.WriteLine($"Size:          {fileInfo.Length} bytes");
Console.WriteLine($"Created:       {fileInfo.CreationTime}");
Console.WriteLine($"Last modified: {fileInfo.LastWriteTime}");
Console.WriteLine($"Exists:        {fileInfo.Exists}");
Console.WriteLine($"IsReadOnly:    {fileInfo.IsReadOnly}");

// Check existence
Console.WriteLine($"\nFile.Exists: {File.Exists(filePath)}");
Console.WriteLine($"File.Exists(fake): {File.Exists("/fake/path.txt")}");

// =============================================================================
// DIRECTORY OPERATIONS
// =============================================================================
Console.WriteLine("\n=== Directory Operations ===");

// Create nested directories
string nestedDir = Path.Combine(workDir, "level1", "level2", "level3");
Directory.CreateDirectory(nestedDir);
Console.WriteLine($"Created: {nestedDir}");

// Create some files in subdirectories
File.WriteAllText(Path.Combine(workDir, "level1", "a.txt"), "file a");
File.WriteAllText(Path.Combine(workDir, "level1", "b.cs"), "file b");
File.WriteAllText(Path.Combine(workDir, "level1", "level2", "c.txt"), "file c");

// List directory contents
Console.WriteLine($"\nContents of {workDir}:");
foreach (string entry in Directory.GetFileSystemEntries(workDir))
{
    bool isDir = Directory.Exists(entry);
    Console.WriteLine($"  {(isDir ? "[DIR]" : "[FILE]")} {Path.GetFileName(entry)}");
}

// Recursive file search
Console.WriteLine($"\nAll .txt files (recursive):");
foreach (string f in Directory.GetFiles(workDir, "*.txt", SearchOption.AllDirectories))
{
    Console.WriteLine($"  {Path.GetRelativePath(workDir, f)}");
}

// Directory info
var dirInfo = new DirectoryInfo(workDir);
Console.WriteLine($"\nDirectory info:");
Console.WriteLine($"  Name: {dirInfo.Name}");
Console.WriteLine($"  Full path: {dirInfo.FullName}");
Console.WriteLine($"  Parent: {dirInfo.Parent?.Name ?? "(none)"}");
Console.WriteLine($"  Files: {dirInfo.GetFiles().Length}");
Console.WriteLine($"  Subdirs: {dirInfo.GetDirectories().Length}");

// =============================================================================
// PATH OPERATIONS
// =============================================================================
Console.WriteLine("\n=== Path Operations ===");

string testPath = "/home/user/documents/report.pdf";

Console.WriteLine($"Path:           {testPath}");
Console.WriteLine($"FileName:       {Path.GetFileName(testPath)}");
Console.WriteLine($"FileNameNoExt:  {Path.GetFileNameWithoutExtension(testPath)}");
Console.WriteLine($"Extension:      {Path.GetExtension(testPath)}");
Console.WriteLine($"Directory:      {Path.GetDirectoryName(testPath)}");
Console.WriteLine($"IsRooted:       {Path.IsPathRooted(testPath)}");

// Combine paths safely
string combined = Path.Combine("home", "user", "docs", "file.txt");
Console.WriteLine($"\nCombine: {combined}");

// Change extension
string changed = Path.ChangeExtension(testPath, ".docx");
Console.WriteLine($"ChangeExt: {changed}");

// Temp paths
Console.WriteLine($"TempPath: {Path.GetTempPath()}");
Console.WriteLine($"TempFile: {Path.GetTempFileName()}");
Console.WriteLine($"RandomName: {Path.GetRandomFileName()}");

// =============================================================================
// FILE COPY, MOVE, DELETE
// =============================================================================
Console.WriteLine("\n=== File Copy / Move / Delete ===");

string srcFile = Path.Combine(workDir, "source.txt");
string copyFile = Path.Combine(workDir, "copy.txt");
string moveFile = Path.Combine(workDir, "moved.txt");

File.WriteAllText(srcFile, "Original content.");
Console.WriteLine($"Created: {Path.GetFileName(srcFile)}");

// Copy
File.Copy(srcFile, copyFile, overwrite: true);
Console.WriteLine($"Copied to: {Path.GetFileName(copyFile)}");

// Move (rename)
File.Move(copyFile, moveFile);
Console.WriteLine($"Moved to: {Path.GetFileName(moveFile)}");

// Delete
File.Delete(moveFile);
Console.WriteLine($"Deleted: {Path.GetFileName(moveFile)}");
Console.WriteLine($"Exists after delete: {File.Exists(moveFile)}");

// =============================================================================
// JSON SERIALIZATION (System.Text.Json)
// =============================================================================
Console.WriteLine("\n=== JSON Serialization ===");

// Serialize object to JSON
var person = new Person
{
    Name = "Alice",
    Age = 30,
    Email = "alice@example.com",
    Hobbies = new List<string> { "Reading", "Hiking", "Coding" }
};

var options = new JsonSerializerOptions
{
    WriteIndented = true,
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
};

string json = JsonSerializer.Serialize(person, options);
Console.WriteLine($"Serialized:\n{json}");

// Write JSON to file
string jsonPath = Path.Combine(workDir, "person.json");
File.WriteAllText(jsonPath, json);
Console.WriteLine($"\nWrote JSON to: {jsonPath}");

// Read and deserialize from file
string readJson = File.ReadAllText(jsonPath);
Person? deserialized = JsonSerializer.Deserialize<Person>(readJson, options);
Console.WriteLine($"Deserialized: {deserialized?.Name}, {deserialized?.Age}, " +
                  $"[{string.Join(", ", deserialized?.Hobbies ?? new())}]");

// Serialize a list
var people = new List<Person>
{
    new() { Name = "Bob", Age = 25, Email = "bob@test.com", Hobbies = new() { "Gaming" } },
    new() { Name = "Charlie", Age = 35, Email = "charlie@test.com", Hobbies = new() { "Music", "Art" } }
};

string listJson = JsonSerializer.Serialize(people, options);
string listPath = Path.Combine(workDir, "people.json");
File.WriteAllText(listPath, listJson);
Console.WriteLine($"\nSerialized list ({people.Count} people) to: {listPath}");

// Deserialize list
var readPeople = JsonSerializer.Deserialize<List<Person>>(File.ReadAllText(listPath), options);
Console.WriteLine("Deserialized list:");
foreach (var p in readPeople ?? new())
{
    Console.WriteLine($"  {p.Name}, age {p.Age}");
}

// =============================================================================
// CLEANUP
// =============================================================================
Console.WriteLine("\n=== Cleanup ===");
try
{
    Directory.Delete(workDir, recursive: true);
    Console.WriteLine($"Cleaned up: {workDir}");
}
catch (IOException ex)
{
    Console.WriteLine($"Cleanup note: {ex.Message}");
}

Console.WriteLine("\nFile I/O demo complete.");

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

class Person
{
    public string Name { get; set; } = "";
    public int Age { get; set; }
    public string Email { get; set; } = "";
    public List<string> Hobbies { get; set; } = new();
}
