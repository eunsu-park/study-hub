# 파일 I/O (File I/O)

**이전**: [예외 처리](./14_Exception_Handling.md) | **다음**: 없음 (마지막 레슨)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `File` 클래스의 정적 메서드를 사용하여 빠른 파일 작업을 수행할 수 있다
2. `StreamReader`와 `StreamWriter`로 텍스트 파일을 읽고 쓸 수 있다
3. `FileStream`, `BinaryReader`, `BinaryWriter`로 바이너리 데이터를 처리할 수 있다
4. `using` 문으로 리소스를 자동으로 관리할 수 있다
5. `Directory`와 `Path` 클래스를 사용하여 디렉토리를 다룰 수 있다
6. CSV 파일을 프로그래밍 방식으로 읽고 쓸 수 있다
7. `System.Text.Json`으로 JSON을 직렬화하고 역직렬화할 수 있다
8. 비동기 파일 작업을 수행할 수 있다

---

파일 I/O는 모든 실제 애플리케이션에 필수적입니다 — 설정 파일 읽기, 로그 쓰기, 데이터 내보내기 처리, 사용자 생성 콘텐츠 관리 등. C#은 `System.IO` 네임스페이스에서 파일과 디렉토리를 다루기 위한 풍부한 클래스 세트를 제공합니다. 이 레슨에서는 간단한 한 줄 파일 읽기부터 바이너리 데이터 스트리밍과 비동기 파일 작업까지 모든 것을 다루어, 프로그램에서 모든 파일 관련 작업을 처리할 수 있는 도구를 제공합니다.

## 1. `File` 클래스 정적 메서드

`System.IO.File` 클래스는 일반적인 파일 작업을 위한 편리한 정적 메서드를 제공합니다. 이 메서드들은 파일을 열고, 작업을 수행하고, 한 번의 호출로 파일을 닫습니다.

### 1.1 파일 읽기

```csharp
using System.IO;

// 전체 파일을 단일 문자열로 읽기
string content = File.ReadAllText("example.txt");
Console.WriteLine(content);

// 파일을 줄 배열로 읽기
string[] lines = File.ReadAllLines("example.txt");
foreach (string line in lines)
{
    Console.WriteLine(line);
}
Console.WriteLine($"Total lines: {lines.Length}");

// 파일을 바이트로 읽기 (바이너리 파일용)
byte[] bytes = File.ReadAllBytes("image.png");
Console.WriteLine($"File size: {bytes.Length} bytes");
```

### 1.2 파일 쓰기

```csharp
// 문자열을 파일에 쓰기 (생성 또는 덮어쓰기)
File.WriteAllText("output.txt", "Hello, World!\nThis is line 2.");

// 줄 배열 쓰기
string[] names = { "Alice", "Bob", "Charlie", "Diana" };
File.WriteAllLines("names.txt", names);

// 바이트 쓰기
byte[] data = { 0x48, 0x65, 0x6C, 0x6C, 0x6F };  // ASCII로 "Hello"
File.WriteAllBytes("binary.dat", data);

// 텍스트 추가 (덮어쓰지 않음)
File.AppendAllText("log.txt", $"[{DateTime.Now}] Application started.\n");
File.AppendAllText("log.txt", $"[{DateTime.Now}] Processing data.\n");

// 줄 추가
File.AppendAllLines("log.txt", new[] { "Line A", "Line B" });
```

### 1.3 파일 존재 여부와 정보

```csharp
string path = "example.txt";

// 파일 존재 여부 확인
if (File.Exists(path))
{
    Console.WriteLine($"File '{path}' exists.");

    // 파일 정보 가져오기
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

### 1.4 파일 작업: 복사, 이동, 삭제

```csharp
// 파일 복사
File.Copy("source.txt", "destination.txt");
File.Copy("source.txt", "destination.txt", overwrite: true);

// 파일 이동 (이름 바꾸기)
File.Move("old_name.txt", "new_name.txt");
File.Move("file.txt", @"archive\file.txt", overwrite: true);

// 파일 삭제
if (File.Exists("temp.txt"))
{
    File.Delete("temp.txt");
    Console.WriteLine("File deleted.");
}
```

### 1.5 인코딩을 지정한 읽기

```csharp
using System.Text;

// 특정 인코딩으로 읽기
string utf8Content = File.ReadAllText("data.txt", Encoding.UTF8);
string latinContent = File.ReadAllText("legacy.txt", Encoding.Latin1);

// 특정 인코딩으로 쓰기
File.WriteAllText("unicode.txt", "Hello in Unicode: Hola Welt", Encoding.UTF8);
File.WriteAllText("ascii.txt", "ASCII only", Encoding.ASCII);
```

## 2. `StreamReader`와 `StreamWriter`

대용량 파일이나 더 많은 제어가 필요한 경우 `StreamReader`와 `StreamWriter`를 사용합니다. 전체 파일을 메모리에 로드하는 대신 줄 단위 또는 청크 단위로 데이터를 처리합니다.

### 2.1 StreamReader로 읽기

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

### 2.2 StreamReader 속성과 메서드

```csharp
using var reader = new StreamReader("data.txt");

// 한 문자씩 읽기
int charCode = reader.Read();
if (charCode != -1)
{
    char c = (char)charCode;
    Console.WriteLine($"First character: {c}");
}

// 문자 블록 읽기
char[] buffer = new char[100];
int charsRead = reader.Read(buffer, 0, buffer.Length);
Console.WriteLine($"Read {charsRead} characters.");

// 소비하지 않고 미리 보기
int nextChar = reader.Peek();
Console.WriteLine($"Next character: {(char)nextChar}");

// 끝까지 읽기
string remaining = reader.ReadToEnd();
Console.WriteLine($"Remaining content length: {remaining.Length}");

// 스트림 끝인지 확인
Console.WriteLine($"End of stream: {reader.EndOfStream}");
```

### 2.3 StreamWriter로 쓰기

```csharp
using (StreamWriter writer = new StreamWriter("output.txt"))
{
    writer.WriteLine("First line");
    writer.WriteLine("Second line");
    writer.Write("No newline at end");
}

// 추가 모드
using (StreamWriter writer = new StreamWriter("output.txt", append: true))
{
    writer.WriteLine();  // 이전 내용 뒤에 줄바꿈 추가
    writer.WriteLine("Appended line 1");
    writer.WriteLine("Appended line 2");
}
```

### 2.4 서식이 있는 StreamWriter

```csharp
using var writer = new StreamWriter("report.txt");

writer.WriteLine("=== Sales Report ===");
writer.WriteLine($"Date: {DateTime.Now:yyyy-MM-dd}");
writer.WriteLine();

// 서식화된 테이블 쓰기
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

writer.Flush();  // 모든 데이터가 기록되었는지 확인
```

### 2.5 대용량 파일의 효율적 처리

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

        // 예: "ERROR"를 포함하는 줄 필터링
        if (line.Contains("ERROR", StringComparison.OrdinalIgnoreCase))
        {
            matchCount++;
            writer.WriteLine($"[Line {lineCount}] {line}");
        }
    }

    Console.WriteLine($"Processed {lineCount:N0} lines, found {matchCount:N0} matches.");
}
```

## 3. 바이너리 I/O를 위한 `FileStream`

`FileStream`은 파일에 대한 저수준 바이트 지향 접근을 제공합니다. 정밀한 제어가 필요하거나 바이너리 데이터를 다룰 때 사용합니다.

### 3.1 기본 FileStream 사용

```csharp
// 바이트 쓰기
using (FileStream fs = new FileStream("data.bin", FileMode.Create))
{
    byte[] data = { 0x01, 0x02, 0x03, 0x04, 0x05 };
    fs.Write(data, 0, data.Length);
    Console.WriteLine($"Wrote {data.Length} bytes.");
}

// 바이트 읽기
using (FileStream fs = new FileStream("data.bin", FileMode.Open))
{
    byte[] buffer = new byte[fs.Length];
    int bytesRead = fs.Read(buffer, 0, buffer.Length);
    Console.WriteLine($"Read {bytesRead} bytes: {BitConverter.ToString(buffer)}");
    // 출력: Read 5 bytes: 01-02-03-04-05
}
```

### 3.2 FileMode 옵션

```csharp
// FileMode.Create      — 새로 생성 또는 기존 파일 덮어쓰기
// FileMode.CreateNew   — 새로 생성; 존재하면 오류
// FileMode.Open        — 기존 파일 열기; 없으면 오류
// FileMode.OpenOrCreate — 존재하면 열기, 없으면 생성
// FileMode.Append      — 추가용으로 열기; 없으면 생성
// FileMode.Truncate    — 열고 길이를 0으로 자르기

using var fs = new FileStream("log.bin", FileMode.OpenOrCreate, FileAccess.ReadWrite);
fs.Seek(0, SeekOrigin.End);  // 추가를 위해 끝으로 이동
byte[] newData = Encoding.UTF8.GetBytes("New entry\n");
fs.Write(newData, 0, newData.Length);
```

### 3.3 FileStream에서의 탐색

```csharp
using var fs = new FileStream("data.bin", FileMode.Open, FileAccess.Read);

// 처음부터 읽기
byte[] header = new byte[4];
fs.Read(header, 0, 4);
Console.WriteLine($"Header: {BitConverter.ToString(header)}");

// 특정 위치로 탐색
fs.Seek(10, SeekOrigin.Begin);    // 시작에서 10바이트
fs.Seek(-5, SeekOrigin.End);      // 끝에서 5바이트
fs.Seek(3, SeekOrigin.Current);   // 현재에서 3바이트 앞으로

Console.WriteLine($"Position: {fs.Position}");
Console.WriteLine($"Length: {fs.Length}");
```

## 4. `BinaryReader`와 `BinaryWriter`

이 클래스들은 스트림을 감싸고 기본 타입을 바이너리 형식으로 읽고 쓰는 메서드를 제공합니다.

### 4.1 BinaryWriter로 쓰기

```csharp
using (FileStream fs = new FileStream("records.bin", FileMode.Create))
using (BinaryWriter writer = new BinaryWriter(fs))
{
    // 다양한 기본 타입 쓰기
    writer.Write(42);            // int (4바이트)
    writer.Write(3.14159);       // double (8바이트)
    writer.Write(true);          // bool (1바이트)
    writer.Write("Hello");       // string (길이 접두사 포함)
    writer.Write('A');           // char (UTF-16에서 2바이트)
    writer.Write(255L);          // long (8바이트)
    writer.Write(9.99m);         // decimal (16바이트)

    Console.WriteLine($"File size: {fs.Length} bytes");
}
```

### 4.2 BinaryReader로 읽기

```csharp
using (FileStream fs = new FileStream("records.bin", FileMode.Open))
using (BinaryReader reader = new BinaryReader(fs))
{
    // 쓴 것과 같은 순서로 읽기
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

### 4.3 구조화된 데이터 저장

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
// 여러 레코드 쓰기
List<StudentRecord> students = new List<StudentRecord>
{
    new StudentRecord { Id = 1, Name = "Alice", Gpa = 3.9, IsActive = true },
    new StudentRecord { Id = 2, Name = "Bob", Gpa = 3.5, IsActive = true },
    new StudentRecord { Id = 3, Name = "Charlie", Gpa = 2.8, IsActive = false }
};

using (var fs = new FileStream("students.bin", FileMode.Create))
using (var writer = new BinaryWriter(fs))
{
    writer.Write(students.Count);  // 먼저 레코드 수 쓰기
    foreach (var student in students)
    {
        student.WriteTo(writer);
    }
}

// 다시 읽기
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
// 출력:
// Reading 3 student records:
//   [1] Alice, GPA: 3.90, Active: True
//   [2] Bob, GPA: 3.50, Active: True
//   [3] Charlie, GPA: 2.80, Active: False
```

## 5. 자동 리소스 정리를 위한 `using` 문

예외 처리 레슨에서 다룬 것처럼, `using` 문은 스트림과 기타 `IDisposable` 객체가 올바르게 닫히도록 보장합니다.

### 5.1 전통적 구문 vs 현대적 구문

```csharp
// 전통적 using 문 (블록 스코프)
using (StreamReader reader = new StreamReader("file.txt"))
{
    string content = reader.ReadToEnd();
    Console.WriteLine(content);
}
// reader는 여기서 해제됨

// 현대적 using 선언 (C# 8+, 메서드 스코프)
using var reader2 = new StreamReader("file.txt");
string content2 = reader2.ReadToEnd();
Console.WriteLine(content2);
// reader2는 감싸는 스코프의 끝에서 해제됨
```

### 5.2 다중 리소스

```csharp
// 연쇄 using 문
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
// 세 스트림 모두 역순으로 해제됨
```

### 5.3 예외 시 정리 보장

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
    // 예외가 발생해도 두 스트림 모두 올바르게 닫힘
}
```

## 6. `Directory` 클래스

`System.IO.Directory` 클래스는 디렉토리를 생성, 이동, 나열하는 정적 메서드를 제공합니다.

### 6.1 디렉토리 생성과 확인

```csharp
string dirPath = @"output\reports\2026";

// 디렉토리 생성 (모든 상위 디렉토리 포함)
if (!Directory.Exists(dirPath))
{
    Directory.CreateDirectory(dirPath);
    Console.WriteLine($"Created: {dirPath}");
}

// 디렉토리 정보 가져오기
DirectoryInfo dirInfo = new DirectoryInfo(dirPath);
Console.WriteLine($"Full path: {dirInfo.FullName}");
Console.WriteLine($"Created: {dirInfo.CreationTime}");
Console.WriteLine($"Parent: {dirInfo.Parent?.Name}");
```

### 6.2 파일과 디렉토리 나열

```csharp
string searchDir = @"C:\Projects";

// 디렉토리의 모든 파일 가져오기
string[] files = Directory.GetFiles(searchDir);
foreach (string file in files)
{
    Console.WriteLine(file);
}

// 패턴으로 파일 가져오기
string[] txtFiles = Directory.GetFiles(searchDir, "*.txt");
string[] csFiles = Directory.GetFiles(searchDir, "*.cs", SearchOption.AllDirectories);
Console.WriteLine($"Found {csFiles.Length} .cs files recursively.");

// 하위 디렉토리 가져오기
string[] subDirs = Directory.GetDirectories(searchDir);
foreach (string dir in subDirs)
{
    Console.WriteLine($"  Directory: {dir}");
}

// 열거 (지연, 대용량 디렉토리에 더 나음)
foreach (string file in Directory.EnumerateFiles(searchDir, "*.*", SearchOption.AllDirectories))
{
    FileInfo fi = new FileInfo(file);
    if (fi.Length > 1024 * 1024)  // > 1MB
    {
        Console.WriteLine($"Large file: {fi.Name} ({fi.Length / 1024.0 / 1024.0:F2} MB)");
    }
}
```

### 6.3 디렉토리 작업

```csharp
// 디렉토리 이동 (이름 바꾸기)
Directory.Move("old_folder", "new_folder");

// 디렉토리 삭제
Directory.Delete("temp_folder");                      // 비어 있어야 함
Directory.Delete("temp_folder", recursive: true);     // 모든 내용 삭제

// 현재 디렉토리와 특수 디렉토리 가져오기
string currentDir = Directory.GetCurrentDirectory();
Console.WriteLine($"Current: {currentDir}");

string tempDir = Path.GetTempPath();
Console.WriteLine($"Temp: {tempDir}");

string desktopDir = Environment.GetFolderPath(Environment.SpecialFolder.Desktop);
Console.WriteLine($"Desktop: {desktopDir}");
```

## 7. `Path` 클래스

`System.IO.Path` 클래스는 플랫폼 안전한 방식으로 파일 및 디렉토리 경로 문자열을 조작하는 정적 메서드를 제공합니다.

### 7.1 경로 조작

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

### 7.2 경로 결합

문자열 연결 대신 항상 `Path.Combine`을 사용하세요. 플랫폼 간에 경로 구분자를 올바르게 처리합니다.

```csharp
// 좋음: Path.Combine은 구분자를 처리
string outputDir = "output";
string fileName = "report.txt";
string fullPath = Path.Combine(outputDir, fileName);
Console.WriteLine(fullPath);  // "output/report.txt" 또는 "output\report.txt"

// 여러 세그먼트
string path = Path.Combine("root", "sub1", "sub2", "file.txt");
Console.WriteLine(path);  // "root/sub1/sub2/file.txt"

// 나쁨: 수동 연결 — 오류 발생 가능
// string badPath = outputDir + "/" + fileName;  // 모든 플랫폼에서 작동하지 않음
```

### 7.3 파일 확장자 변경과 임시 파일 생성

```csharp
// 확장자 변경
string original = "document.txt";
string backup = Path.ChangeExtension(original, ".bak");
Console.WriteLine(backup);  // "document.bak"

// 임시 파일 경로 생성
string tempFile = Path.GetTempFileName();
Console.WriteLine($"Temp file: {tempFile}");
// 예: "/tmp/tmpABC123.tmp"

string tempDir = Path.GetTempPath();
Console.WriteLine($"Temp dir: {tempDir}");

// 랜덤 파일명 생성
string randomName = Path.GetRandomFileName();
Console.WriteLine($"Random name: {randomName}");
// 예: "4k2znmos.hsf"
```

### 7.4 파일 관리자 유틸리티 만들기

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

            // 중복 처리
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

## 8. CSV 파일 읽기와 쓰기

CSV(Comma-Separated Values)는 데이터 교환에 일반적인 형식입니다. 외부 라이브러리 없이 C#에서 처리하는 방법을 알아봅시다.

### 8.1 CSV 쓰기

```csharp
public class CsvWriter
{
    public static void WriteCsv(string path, string[] headers, List<string[]> rows)
    {
        using var writer = new StreamWriter(path);

        // 헤더 쓰기
        writer.WriteLine(string.Join(",", headers));

        // 데이터 행 쓰기
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
            // 따옴표를 두 배로 이스케이프하고 따옴표로 감싸기
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
    new[] { "Diana, Jr.", "28", "diana@example.com", "Los Angeles" }  // 이름에 쉼표 포함
};

CsvWriter.WriteCsv("people.csv", headers, rows);
// 출력 파일:
// Name,Age,Email,City
// Alice,30,alice@example.com,New York
// Bob,25,bob@example.com,San Francisco
// Charlie,35,charlie@example.com,Chicago
// "Diana, Jr.",28,diana@example.com,Los Angeles
```

### 8.2 CSV 읽기

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
                        i++;  // 이스케이프된 따옴표 건너뛰기
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

### 8.3 CSV에서 객체로

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

## 9. `System.Text.Json`을 사용한 기본 JSON

`System.Text.Json`은 최신 .NET의 내장 JSON 라이브러리입니다. 직렬화(serialization, 객체에서 JSON)와 역직렬화(deserialization, JSON에서 객체)를 제공합니다.

### 9.1 직렬화 (객체에서 JSON)

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

// JSON 문자열로 직렬화
string json = JsonSerializer.Serialize(product);
Console.WriteLine(json);
// {"Id":1,"Name":"Mechanical Keyboard","Price":149.99,"Tags":["electronics","peripherals","gaming"],"InStock":true}

// 보기 좋게 출력된 JSON
var options = new JsonSerializerOptions { WriteIndented = true };
string prettyJson = JsonSerializer.Serialize(product, options);
Console.WriteLine(prettyJson);
```

### 9.2 역직렬화 (JSON에서 객체)

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

// 리스트 역직렬화
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

### 9.3 JSON 파일 읽기와 쓰기

```csharp
// JSON을 파일에 쓰기
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

// 파일에서 JSON 읽기
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
// 제품 저장
var products = new List<Product>
{
    new Product { Id = 1, Name = "Laptop", Price = 999.99m, Tags = new[] { "tech" }, InStock = true },
    new Product { Id = 2, Name = "Chair", Price = 249.99m, Tags = new[] { "furniture" }, InStock = true }
};

SaveToJson("products.json", products);

// 다시 로드
var loaded = LoadFromJson<List<Product>>("products.json");
foreach (var p in loaded)
{
    Console.WriteLine($"{p.Name}: ${p.Price}");
}
```

### 9.4 직렬화 커스터마이징

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

    [JsonIgnore]  // 직렬화에서 제외
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
// 참고: InternalSecret은 포함되지 않음 (JsonIgnore)
```

## 10. 비동기 파일 I/O

반응성을 유지해야 하는 애플리케이션(특히 UI 앱과 웹 서버)에서는 비동기 파일 작업이 호출 스레드의 차단을 방지합니다.

### 10.1 비동기 읽기와 쓰기

```csharp
using System.Threading.Tasks;

// 비동기 읽기
public static async Task<string> ReadFileAsync(string path)
{
    string content = await File.ReadAllTextAsync(path);
    return content;
}

// 비동기 쓰기
public static async Task WriteFileAsync(string path, string content)
{
    await File.WriteAllTextAsync(path, content);
}

// 비동기 줄 읽기
public static async Task<string[]> ReadLinesAsync(string path)
{
    return await File.ReadAllLinesAsync(path);
}
```

### 10.2 비동기 스트림 작업

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

### 10.3 비동기 파일 복사

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

### 10.4 비동기 메서드 호출

```csharp
// 비동기 Main 메서드에서 (C# 7.1+)
static async Task Main(string[] args)
{
    // 파일 쓰기
    await File.WriteAllTextAsync("async_test.txt", "Hello from async!");

    // 다시 읽기
    string content = await File.ReadAllTextAsync("async_test.txt");
    Console.WriteLine(content);

    // 파일 처리
    await ProcessFileAsync("input.txt", "output.txt");

    // 파일 복사
    await CopyFileAsync("large_file.dat", "large_file_backup.dat");

    // 모든 줄 읽기
    string[] lines = await File.ReadAllLinesAsync("data.txt");
    Console.WriteLine($"Read {lines.Length} lines.");
}
```

### 10.5 비동기 파일 I/O를 사용해야 하는 경우

```csharp
// 비동기를 사용하는 경우:
// - 웹 애플리케이션(ASP.NET Core) 구축 시 — 다른 요청을 위해 스레드 해제
// - 데스크톱/모바일 UI 구축 시 — UI 스레드 동결 방지
// - 여러 파일을 동시에 처리할 때

// 예: 여러 파일을 병렬로 처리
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

// 비동기가 필요하지 않은 경우:
// - 순차적 파일 작업이 있는 간단한 콘솔 앱
// - 비동기의 오버헤드가 가치 없는 매우 작은 파일
```

## 11. 연습 문제

1. **줄 카운터**: 파일 경로를 명령줄 인수로 받아 표시하는 프로그램을 작성하세요: 총 줄 수, 빈 줄, "#"으로 시작하는 줄(주석), 평균 줄 길이. 파일이 존재하지 않는 경우 친절한 오류 메시지로 처리하세요. 메모리 효율성을 위해 `StreamReader`를 사용하세요.

2. **파일 백업 유틸리티**: 디렉토리를 백업하는 프로그램을 만드세요. 소스 디렉토리 경로가 주어지면: (a) `backup_YYYYMMDD_HHmmss`라는 이름의 백업 디렉토리를 생성, (b) 모든 파일과 하위 디렉토리를 재귀적으로 복사, (c) 10MB보다 큰 파일은 건너뛰기(경고 출력), (d) 복사된 파일 수, 건너뛴 파일 수, 복사된 총 바이트 수를 보여주는 요약을 출력하세요. 모든 경로 구성에 `Path.Combine`을 사용하세요.

3. **CSV 분석기**: CSV 파일을 읽고 대화형 메뉴를 제공하는 프로그램을 작성하세요: (a) 서식화된 테이블로 모든 레코드 표시, (b) 모든 열로 정렬(오름차순 또는 내림차순), (c) 열 값이 일치하는 행 필터링, (d) 새 행 추가, (e) 수정된 데이터를 CSV로 다시 저장. 내부에 쉼표가 있는 따옴표 필드를 올바르게 처리하세요.

4. **JSON 설정 관리자**: JSON 설정 파일을 읽고/쓰는 설정 관리자를 만드세요. 지원해야 할 기능: (a) 키로 값 가져오기(중첩된 키 지원: "database.host"), (b) 키로 값 설정, (c) 모든 키 나열, (d) 키 삭제, (e) 변경 사항을 JSON 파일에 다시 저장. 동적 접근을 위해 `System.Text.Json`과 `JsonDocument`를 사용하세요. 설정 파일이 존재하지 않으면 기본값으로 자동 생성되어야 합니다.

5. **로그 파일 분석기**: 여러 로그 파일을 병렬로 처리하는 비동기 프로그램을 작성하세요. 각 로그 파일의 줄은 `[TIMESTAMP] [LEVEL] Message` 형식입니다. 프로그램은: (a) 각 로그 레벨(INFO, WARN, ERROR, DEBUG)의 발생 횟수 카운트, (b) 모든 ERROR 줄을 찾아 `errors_summary.txt` 파일에 쓰기, (c) 각 파일이 커버하는 시간 범위 계산, (d) 모든 파일에 걸친 결합 보고서 생성. 병렬 처리를 위해 `async`/`await`와 `Task.WhenAll`을 사용하세요.
