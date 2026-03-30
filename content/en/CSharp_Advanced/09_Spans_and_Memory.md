# Spans and Memory

**Previous**: [Concurrency and Parallelism](./08_Concurrency_and_Parallelism.md) | **Next**: [Dependency Injection](./10_Dependency_Injection.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Explain the difference between stack and heap allocation and their performance implications
2. Use `Span<T>` and `ReadOnlySpan<T>` to work with contiguous memory without allocation
3. Understand `Span<T>` limitations and why it is a `ref struct`
4. Use `Memory<T>` and `ReadOnlyMemory<T>` when heap storage or async methods are needed
5. Allocate temporary buffers on the stack with `stackalloc`
6. Reuse buffers efficiently with `ArrayPool<T>`
7. Perform zero-allocation string parsing with spans
8. Apply `MemoryMarshal` for low-level memory reinterpretation

---

High-performance .NET code often bottlenecks on memory allocation and garbage collection. Every `new byte[]`, every `string.Substring()`, and every LINQ `.ToArray()` creates heap pressure. C# provides `Span<T>`, `Memory<T>`, `stackalloc`, and `ArrayPool<T>` as tools to reduce or eliminate allocations in hot paths. This lesson teaches you to think about memory as a first-class concern and write code that is both fast and safe.

## 1. Stack vs Heap Allocation Review

### 1.1 The Managed Heap

All reference types (`class`, `string`, arrays) live on the managed heap. The garbage collector (GC) reclaims them when no references remain. GC pressure increases with allocation rate — allocating millions of small objects per second can cause noticeable pauses.

```csharp
// Each call allocates a new string on the heap
string GetSubstring(string source, int start, int length)
{
    return source.Substring(start, length); // New heap allocation
}
```

### 1.2 The Stack

Value types (`struct`, `int`, `double`) declared as local variables live on the stack. Stack allocation is essentially free — it just moves the stack pointer. Stack memory is reclaimed automatically when the method returns.

```csharp
// No heap allocation — struct lives on the stack
public struct Point
{
    public double X, Y;
}

void Calculate()
{
    Point p = new Point { X = 1.0, Y = 2.0 }; // Stack-allocated
    double distance = Math.Sqrt(p.X * p.X + p.Y * p.Y);
}
```

### 1.3 Allocation Cost in Practice

```csharp
// Benchmark: heap allocation vs stackalloc
using System.Diagnostics;

void HeapAllocation(int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        byte[] buffer = new byte[256]; // Heap
        buffer[0] = (byte)i;
    }
}

void StackAllocation(int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        Span<byte> buffer = stackalloc byte[256]; // Stack
        buffer[0] = (byte)i;
    }
}
```

## 2. Span&lt;T&gt; and ReadOnlySpan&lt;T&gt;

`Span<T>` is a type-safe, memory-safe view over a contiguous region of memory. It can point to arrays, stack-allocated memory, or native memory — all through a unified API.

### 2.1 Creating Spans

```csharp
// From an array
int[] numbers = { 1, 2, 3, 4, 5 };
Span<int> span = numbers;               // Entire array
Span<int> slice = numbers.AsSpan(1, 3); // Elements 2, 3, 4

// From a string (ReadOnlySpan<char>)
string text = "Hello, World!";
ReadOnlySpan<char> greeting = text.AsSpan(0, 5); // "Hello"

// From stackalloc
Span<byte> buffer = stackalloc byte[128];
buffer.Fill(0xFF);

// From a pointer (unsafe)
unsafe
{
    int* ptr = stackalloc int[10];
    Span<int> fromPtr = new Span<int>(ptr, 10);
}
```

### 2.2 Slicing Without Allocation

The killer feature of `Span<T>` is zero-copy slicing. Unlike `Array.Copy` or `string.Substring`, slicing a span creates a new view of the same memory.

```csharp
byte[] data = new byte[1024];
FillData(data);

// Zero-copy slicing — no new arrays created
Span<byte> header = data.AsSpan(0, 16);
Span<byte> payload = data.AsSpan(16, data.Length - 16);

// Nested slicing
Span<byte> firstWord = header[..4];
Span<byte> secondWord = header[4..8];
```

### 2.3 Common Span Operations

```csharp
Span<int> span = stackalloc int[5];

// Fill
span.Fill(42);  // All elements = 42

// Clear
span.Clear();   // All elements = 0

// Copy
Span<int> source = stackalloc int[] { 1, 2, 3 };
Span<int> dest = stackalloc int[3];
source.CopyTo(dest);

// TryCopyTo (safe, returns false if dest too small)
bool success = source.TryCopyTo(dest);

// IndexOf
Span<byte> bytes = stackalloc byte[] { 10, 20, 30, 20, 10 };
int idx = bytes.IndexOf((byte)20); // 1

// Contains
bool has30 = bytes.Contains((byte)30); // true

// SequenceEqual
bool equal = source.SequenceEqual(dest); // true
```

### 2.4 ReadOnlySpan&lt;T&gt;

`ReadOnlySpan<T>` prevents modification of the underlying data. String slicing returns `ReadOnlySpan<char>`.

```csharp
public int CountWords(ReadOnlySpan<char> text)
{
    if (text.IsEmpty) return 0;

    int count = 0;
    bool inWord = false;

    foreach (char c in text)
    {
        if (char.IsWhiteSpace(c))
        {
            inWord = false;
        }
        else if (!inWord)
        {
            count++;
            inWord = true;
        }
    }
    return count;
}

// Usage — no string allocations
string sentence = "The quick brown fox jumps over the lazy dog";
int words = CountWords(sentence.AsSpan()); // 9
int firstThreeWords = CountWords(sentence.AsSpan(0, 15)); // 3
```

## 3. Span Limitations

`Span<T>` is a `ref struct`, which imposes strict rules.

### 3.1 What ref struct Means

A `ref struct` can only live on the stack. This prevents it from escaping to the heap, which is necessary because a span might point to stack-allocated memory that will be reclaimed when the method returns.

```csharp
// CANNOT do any of these:
class MyClass
{
    // Span<int> _field;                  // Error: cannot be a field of a class
}

// Span<int> BoxedSpan = (object)span;    // Error: cannot box a ref struct
// Task<Span<int>> task = ...;            // Error: cannot be a type argument
// List<Span<int>> list = ...;            // Error: cannot be a type argument

// CANNOT use in async methods:
// async Task ProcessAsync()
// {
//     Span<byte> buffer = stackalloc byte[256];
//     await SomeOperationAsync(); // Error: Span cannot cross await boundary
// }

// CANNOT use in lambdas or local functions that capture:
// Action action = () => { var s = span; }; // Error

// CAN do:
void ProcessSync(Span<byte> data) { /* OK in synchronous methods */ }
ref struct MyRefStruct { public Span<int> Data; } // OK: ref struct in ref struct
```

### 3.2 Choosing Between Span and Memory

| Feature | `Span<T>` | `Memory<T>` |
|---------|-----------|-------------|
| Can be a field | No (ref struct) | Yes |
| Can be used in async | No | Yes |
| Can be a generic arg | No | Yes |
| Points to stack memory | Yes | No |
| Performance | Fastest | Slightly slower |

## 4. Memory&lt;T&gt; and ReadOnlyMemory&lt;T&gt;

`Memory<T>` is the heap-safe counterpart to `Span<T>`. It can be stored in fields, used with async methods, and passed to generic APIs.

### 4.1 Basic Usage

```csharp
public class DataProcessor
{
    private Memory<byte> _buffer;  // OK: Memory<T> can be a field

    public DataProcessor(byte[] data)
    {
        _buffer = data.AsMemory();
    }

    public async Task ProcessAsync()
    {
        Memory<byte> header = _buffer[..16];
        Memory<byte> payload = _buffer[16..];

        // Memory<T> works across await boundaries
        await ProcessHeaderAsync(header);
        await ProcessPayloadAsync(payload);
    }

    private async Task ProcessHeaderAsync(Memory<byte> header)
    {
        // When you need Span, call .Span inside a synchronous scope
        ParseHeader(header.Span);
        await Task.CompletedTask;
    }

    private void ParseHeader(Span<byte> header)
    {
        // Fast, zero-allocation parsing
        int version = header[0];
        int flags = header[1];
    }

    private async Task ProcessPayloadAsync(Memory<byte> payload)
    {
        // Write to stream
        await using var stream = new MemoryStream();
        await stream.WriteAsync(payload);
    }
}
```

### 4.2 IMemoryOwner&lt;T&gt; and MemoryPool

```csharp
using System.Buffers;

// Rent memory from a pool (similar to ArrayPool but returns Memory<T>)
using IMemoryOwner<byte> owner = MemoryPool<byte>.Shared.Rent(4096);
Memory<byte> memory = owner.Memory[..4096]; // Slice to exact size needed

// Process the memory
FillData(memory.Span);
await SendAsync(memory);
// Memory returned to pool when owner is disposed
```

## 5. stackalloc Keyword

`stackalloc` allocates memory on the stack, avoiding GC pressure entirely. The memory is freed automatically when the method exits.

### 5.1 Basic stackalloc

```csharp
public bool IsValidUtf8(ReadOnlySpan<byte> data)
{
    // Stack-allocated lookup table — no heap allocation
    Span<bool> continuationValid = stackalloc bool[256];
    for (int i = 0x80; i <= 0xBF; i++)
        continuationValid[i] = true;

    int i2 = 0;
    while (i2 < data.Length)
    {
        byte b = data[i2];
        if (b < 0x80) { i2++; continue; }
        // ... validation logic using the lookup table
        i2++;
    }
    return true;
}
```

### 5.2 Conditional stackalloc

For variable-size buffers, use `stackalloc` for small sizes and fall back to `ArrayPool` for large ones.

```csharp
public string FormatNumber(double value)
{
    const int StackThreshold = 256;

    char[]? rented = null;
    Span<char> buffer = value.ToString().Length <= StackThreshold
        ? stackalloc char[StackThreshold]
        : (rented = ArrayPool<char>.Shared.Rent(1024));

    try
    {
        if (value.TryFormat(buffer, out int charsWritten, "N2"))
        {
            return new string(buffer[..charsWritten]);
        }
        return value.ToString("N2");
    }
    finally
    {
        if (rented is not null)
            ArrayPool<char>.Shared.Return(rented);
    }
}
```

### 5.3 stackalloc with Initializers

```csharp
// C# 8+: stackalloc with initializer
ReadOnlySpan<int> primes = stackalloc int[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

// Pattern matching with spans
ReadOnlySpan<byte> magic = stackalloc byte[] { 0x89, 0x50, 0x4E, 0x47 }; // PNG header
bool isPng = fileHeader.StartsWith(magic);
```

## 6. ref struct and Why Span Is One

### 6.1 Defining Your Own ref struct

```csharp
public ref struct TokenEnumerator
{
    private ReadOnlySpan<char> _remaining;
    private ReadOnlySpan<char> _current;
    private readonly char _separator;

    public TokenEnumerator(ReadOnlySpan<char> text, char separator)
    {
        _remaining = text;
        _current = default;
        _separator = separator;
    }

    public ReadOnlySpan<char> Current => _current;

    public bool MoveNext()
    {
        if (_remaining.IsEmpty)
            return false;

        int idx = _remaining.IndexOf(_separator);
        if (idx < 0)
        {
            _current = _remaining;
            _remaining = default;
        }
        else
        {
            _current = _remaining[..idx];
            _remaining = _remaining[(idx + 1)..];
        }
        return true;
    }

    // Enable foreach syntax
    public TokenEnumerator GetEnumerator() => this;
}
```

```csharp
// Usage — zero allocation string splitting
ReadOnlySpan<char> csv = "Alice,30,Engineer,Seattle";
var enumerator = new TokenEnumerator(csv, ',');

while (enumerator.MoveNext())
{
    // Each token is a ReadOnlySpan<char> — no string allocation
    Console.WriteLine(enumerator.Current.ToString());
}
```

### 6.2 Dispose Pattern for ref struct

```csharp
public ref struct SpanWriter
{
    private Span<byte> _buffer;
    private int _position;
    private byte[]? _rentedArray;

    public SpanWriter(int capacity)
    {
        _rentedArray = ArrayPool<byte>.Shared.Rent(capacity);
        _buffer = _rentedArray;
        _position = 0;
    }

    public void WriteByte(byte value)
    {
        _buffer[_position++] = value;
    }

    public ReadOnlySpan<byte> Written => _buffer[.._position];

    // ref struct cannot implement IDisposable, but can have a Dispose method
    public void Dispose()
    {
        if (_rentedArray is not null)
        {
            ArrayPool<byte>.Shared.Return(_rentedArray);
            _rentedArray = null;
        }
    }
}

// Usage with using statement (pattern-based)
using var writer = new SpanWriter(1024);
writer.WriteByte(0x01);
writer.WriteByte(0x02);
ReadOnlySpan<byte> data = writer.Written;
```

## 7. ArrayPool&lt;T&gt; for Buffer Reuse

`ArrayPool<T>` provides a pool of reusable arrays, eliminating repeated allocations for temporary buffers.

### 7.1 Basic Usage

```csharp
using System.Buffers;

public byte[] CompressData(ReadOnlySpan<byte> input)
{
    // Rent a buffer — may be larger than requested
    byte[] buffer = ArrayPool<byte>.Shared.Rent(input.Length * 2);

    try
    {
        int compressedLength = Compress(input, buffer);
        // Copy the exact result to a right-sized array
        return buffer.AsSpan(0, compressedLength).ToArray();
    }
    finally
    {
        // ALWAYS return rented arrays — pass clearArray: true for sensitive data
        ArrayPool<byte>.Shared.Return(buffer, clearArray: false);
    }
}
```

### 7.2 Custom Pool Configuration

```csharp
// Custom pool with specific size limits
ArrayPool<byte> customPool = ArrayPool<byte>.Create(
    maxArrayLength: 1024 * 1024,  // Max 1 MB arrays
    maxArraysPerBucket: 50        // Keep up to 50 arrays per size bucket
);

byte[] buf = customPool.Rent(8192);
try
{
    // Use buffer
}
finally
{
    customPool.Return(buf);
}
```

### 7.3 MemoryStream Alternative with ArrayPool

```csharp
public class PooledMemoryStream : IDisposable
{
    private byte[] _buffer;
    private int _length;
    private bool _disposed;

    public PooledMemoryStream(int initialCapacity = 256)
    {
        _buffer = ArrayPool<byte>.Shared.Rent(initialCapacity);
        _length = 0;
    }

    public void Write(ReadOnlySpan<byte> data)
    {
        EnsureCapacity(_length + data.Length);
        data.CopyTo(_buffer.AsSpan(_length));
        _length += data.Length;
    }

    public ReadOnlySpan<byte> WrittenSpan => _buffer.AsSpan(0, _length);

    private void EnsureCapacity(int required)
    {
        if (required <= _buffer.Length) return;

        int newSize = Math.Max(_buffer.Length * 2, required);
        byte[] newBuffer = ArrayPool<byte>.Shared.Rent(newSize);
        _buffer.AsSpan(0, _length).CopyTo(newBuffer);
        ArrayPool<byte>.Shared.Return(_buffer);
        _buffer = newBuffer;
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            ArrayPool<byte>.Shared.Return(_buffer);
            _disposed = true;
        }
    }
}
```

## 8. String Processing with Spans (Zero-Allocation Parsing)

String manipulation is one of the biggest sources of allocation in .NET. Spans let you parse strings without creating intermediate `string` objects.

### 8.1 Parsing Integers Without Allocation

```csharp
public bool TryParseCoordinate(ReadOnlySpan<char> input, out double lat, out double lon)
{
    lat = 0;
    lon = 0;

    int commaIndex = input.IndexOf(',');
    if (commaIndex < 0) return false;

    ReadOnlySpan<char> latPart = input[..commaIndex].Trim();
    ReadOnlySpan<char> lonPart = input[(commaIndex + 1)..].Trim();

    return double.TryParse(latPart, out lat) && double.TryParse(lonPart, out lon);
}

// Usage — no string allocations
string data = "47.6062, -122.3321";
if (TryParseCoordinate(data, out double lat, out double lon))
{
    Console.WriteLine($"Lat: {lat}, Lon: {lon}");
}
```

### 8.2 Splitting Without string.Split

```csharp
public static int SplitAndSum(ReadOnlySpan<char> csv)
{
    int sum = 0;
    while (!csv.IsEmpty)
    {
        int commaIdx = csv.IndexOf(',');
        ReadOnlySpan<char> token;

        if (commaIdx < 0)
        {
            token = csv;
            csv = default;
        }
        else
        {
            token = csv[..commaIdx];
            csv = csv[(commaIdx + 1)..];
        }

        if (int.TryParse(token.Trim(), out int value))
        {
            sum += value;
        }
    }
    return sum;
}

// Zero allocations for parsing "1, 2, 3, 4, 5"
int total = SplitAndSum("1, 2, 3, 4, 5"); // 15
```

### 8.3 Building Strings with Span (ISpanFormattable)

```csharp
public readonly struct IpAddress : ISpanFormattable
{
    private readonly uint _value;

    public IpAddress(byte a, byte b, byte c, byte d)
    {
        _value = (uint)(a << 24 | b << 16 | c << 8 | d);
    }

    public bool TryFormat(Span<char> destination, out int charsWritten,
        ReadOnlySpan<char> format, IFormatProvider? provider)
    {
        Span<char> buffer = stackalloc char[15]; // "255.255.255.255"
        int pos = 0;

        for (int i = 3; i >= 0; i--)
        {
            byte octet = (byte)(_value >> (i * 8));
            if (!octet.TryFormat(buffer[pos..], out int written))
            {
                charsWritten = 0;
                return false;
            }
            pos += written;
            if (i > 0) buffer[pos++] = '.';
        }

        if (pos > destination.Length)
        {
            charsWritten = 0;
            return false;
        }

        buffer[..pos].CopyTo(destination);
        charsWritten = pos;
        return true;
    }

    public string ToString(string? format, IFormatProvider? provider)
    {
        Span<char> buffer = stackalloc char[15];
        TryFormat(buffer, out int written, default, provider);
        return new string(buffer[..written]);
    }
}
```

## 9. MemoryMarshal Utilities

`MemoryMarshal` provides low-level operations for reinterpreting memory, reading/writing structured data, and converting between span types.

### 9.1 Cast Between Types

```csharp
using System.Runtime.InteropServices;

byte[] rawBytes = { 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00 };

// Reinterpret bytes as ints (same memory, no copy)
ReadOnlySpan<int> ints = MemoryMarshal.Cast<byte, int>(rawBytes);
Console.WriteLine(ints[0]); // 1
Console.WriteLine(ints[1]); // 2
```

### 9.2 Read and Write Structs

```csharp
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct PacketHeader
{
    public byte Version;
    public byte Type;
    public ushort Length;
    public uint SequenceNumber;
}

// Read a struct from raw bytes
byte[] networkData = ReceivePacket();
PacketHeader header = MemoryMarshal.Read<PacketHeader>(networkData);
Console.WriteLine($"Version: {header.Version}, Seq: {header.SequenceNumber}");

// Write a struct to bytes
var outHeader = new PacketHeader { Version = 1, Type = 2, Length = 64, SequenceNumber = 42 };
byte[] output = new byte[Marshal.SizeOf<PacketHeader>()];
MemoryMarshal.Write(output, in outHeader);
```

### 9.3 GetReference for Unsafe Fast Access

```csharp
// Get a direct reference to the first element (avoids bounds check)
Span<int> span = new int[] { 10, 20, 30, 40, 50 };
ref int first = ref MemoryMarshal.GetReference(span);

// Use Unsafe.Add for indexed access without bounds checking
ref int third = ref Unsafe.Add(ref first, 2);
Console.WriteLine(third); // 30
```

## 10. Practical Example: High-Performance CSV Parser

This example combines `Span<T>`, `stackalloc`, `ArrayPool<T>`, and zero-allocation parsing to build a fast CSV parser.

```csharp
using System.Buffers;

public ref struct CsvReader
{
    private ReadOnlySpan<char> _data;
    private int _lineNumber;

    public CsvReader(ReadOnlySpan<char> data)
    {
        _data = data;
        _lineNumber = 0;
    }

    public bool TryReadLine(out CsvLine line)
    {
        if (_data.IsEmpty)
        {
            line = default;
            return false;
        }

        int newlineIdx = _data.IndexOf('\n');
        ReadOnlySpan<char> rawLine;

        if (newlineIdx < 0)
        {
            rawLine = _data;
            _data = default;
        }
        else
        {
            rawLine = _data[..newlineIdx];
            _data = _data[(newlineIdx + 1)..];
        }

        // Strip carriage return
        if (rawLine.Length > 0 && rawLine[^1] == '\r')
            rawLine = rawLine[..^1];

        line = new CsvLine(rawLine, _lineNumber++);
        return true;
    }
}

public ref struct CsvLine
{
    private ReadOnlySpan<char> _data;
    public int LineNumber { get; }

    public CsvLine(ReadOnlySpan<char> data, int lineNumber)
    {
        _data = data;
        LineNumber = lineNumber;
    }

    public int FieldCount
    {
        get
        {
            if (_data.IsEmpty) return 0;
            int count = 1;
            foreach (char c in _data)
                if (c == ',') count++;
            return count;
        }
    }

    public ReadOnlySpan<char> GetField(int index)
    {
        ReadOnlySpan<char> remaining = _data;
        int current = 0;

        while (!remaining.IsEmpty)
        {
            int commaIdx = remaining.IndexOf(',');
            ReadOnlySpan<char> field;

            if (commaIdx < 0)
            {
                field = remaining;
                remaining = default;
            }
            else
            {
                field = remaining[..commaIdx];
                remaining = remaining[(commaIdx + 1)..];
            }

            if (current == index)
                return field.Trim();

            current++;
        }

        return default;
    }

    public bool TryGetInt(int index, out int value)
    {
        return int.TryParse(GetField(index), out value);
    }

    public bool TryGetDouble(int index, out double value)
    {
        return double.TryParse(GetField(index), out value);
    }
}
```

```csharp
// Full processing pipeline
public class CsvProcessor
{
    public record SalesRecord(string Product, int Quantity, double Price);

    public static List<SalesRecord> ParseSalesData(string csvContent)
    {
        var records = new List<SalesRecord>();
        var reader = new CsvReader(csvContent);

        // Skip header line
        reader.TryReadLine(out _);

        while (reader.TryReadLine(out CsvLine line))
        {
            ReadOnlySpan<char> product = line.GetField(0);

            if (line.TryGetInt(1, out int qty) && line.TryGetDouble(2, out double price))
            {
                records.Add(new SalesRecord(product.ToString(), qty, price));
            }
        }

        return records;
    }
}
```

```csharp
// Usage and benchmarking
string csv = """
    Product,Quantity,Price
    Widget A,100,19.99
    Widget B,250,9.50
    Gadget X,75,49.99
    Gadget Y,300,14.75
    """;

var records = CsvProcessor.ParseSalesData(csv);
foreach (var record in records)
{
    Console.WriteLine($"{record.Product}: {record.Quantity} x ${record.Price:F2} = ${record.Quantity * record.Price:F2}");
}

// Compare with traditional string.Split approach:
// string.Split allocates N string objects per line
// Span-based parser allocates strings only for the final SalesRecord
```

```csharp
// ArrayPool-based file reading for large CSVs
public static List<CsvProcessor.SalesRecord> ParseLargeFile(string filePath)
{
    byte[] rentedBuffer = ArrayPool<byte>.Shared.Rent((int)new FileInfo(filePath).Length);
    try
    {
        int bytesRead;
        using (var fs = File.OpenRead(filePath))
        {
            bytesRead = fs.Read(rentedBuffer);
        }

        // Convert bytes to chars
        ReadOnlySpan<byte> bytes = rentedBuffer.AsSpan(0, bytesRead);
        int charCount = System.Text.Encoding.UTF8.GetCharCount(bytes);

        char[] charBuffer = ArrayPool<char>.Shared.Rent(charCount);
        try
        {
            System.Text.Encoding.UTF8.GetChars(bytes, charBuffer);
            return CsvProcessor.ParseSalesData(new string(charBuffer, 0, charCount));
        }
        finally
        {
            ArrayPool<char>.Shared.Return(charBuffer);
        }
    }
    finally
    {
        ArrayPool<byte>.Shared.Return(rentedBuffer);
    }
}
```

## 11. Practice Problems

1. **Zero-Allocation Number Formatter**: Write a method `FormatWithCommas(long value, Span<char> buffer, out int charsWritten)` that formats a number with thousands separators (e.g., `1234567` becomes `1,234,567`) using only stack-allocated memory. Do not allocate any strings or arrays. Handle negative numbers.

2. **Binary Protocol Parser**: Define a simple binary protocol with a header (4-byte magic, 2-byte version, 4-byte payload length) and variable-length payload. Write a parser using `Span<T>` and `MemoryMarshal.Read<T>` that extracts the header fields and payload without any heap allocation. Include validation for the magic bytes and maximum payload size.

3. **Span-Based String Builder**: Implement a `ref struct StackStringBuilder` backed by `stackalloc` (with `ArrayPool` fallback for overflow). Support `Append(ReadOnlySpan<char>)`, `Append(int)`, `Append(char)`, and `ToString()`. The builder should not allocate until `ToString()` is called. Write a test that builds a 500-character string using only stack memory.

4. **Pooled Buffer Manager**: Create a `BufferManager` class that wraps `ArrayPool<byte>` with automatic tracking of rented buffers. Implement `IDisposable` so that all outstanding buffers are returned when the manager is disposed. Add a `BufferHandle` ref struct that represents a rented buffer with `Span<byte>` access and automatic return on dispose.

5. **HTTP Header Parser**: Write a zero-allocation HTTP header parser that takes a `ReadOnlySpan<byte>` containing raw HTTP response bytes and extracts: status code, content-type, content-length, and each header as key-value `ReadOnlySpan<char>` pairs. Use `Span` slicing and `Utf8Parser.TryParse` where possible. Validate it against a sample HTTP response.
