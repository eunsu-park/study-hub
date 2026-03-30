# Interop and Unsafe Code

**Previous**: [Reflection and Attributes](./14_Reflection_and_Attributes.md) | **Next**: [Performance and Profiling](./16_Performance_Profiling.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Call native C/C++ libraries from C# using Platform Invoke (P/Invoke)
2. Use both `DllImport` and the newer `LibraryImport` source-generated approach
3. Marshal strings, structs, and arrays between managed and unmanaged code
4. Write unsafe code blocks with pointer arithmetic
5. Use the `fixed` statement to pin managed objects in memory
6. Allocate memory on the stack with `stackalloc`
7. Apply `Span<T>` and `Memory<T>` as safe alternatives to raw pointers
8. Understand when unsafe code is justified and how to minimize its scope

---

C# is a managed language with automatic memory management, but real-world applications sometimes need to interact with native libraries, manipulate memory directly, or squeeze out every last bit of performance. The interop and unsafe features of C# provide controlled escape hatches from the managed runtime. This lesson covers P/Invoke for calling native code, the `unsafe` keyword for pointer operations, and modern safe alternatives like `Span<T>` that reduce the need for raw pointers.

## 1. Platform Invoke (P/Invoke) with DllImport

### 1.1 Basic P/Invoke

P/Invoke allows C# code to call functions exported from native (unmanaged) dynamic libraries:

```csharp
using System.Runtime.InteropServices;

public static class NativeMethods
{
    // Call the C standard library's strlen function
    [DllImport("libc", EntryPoint = "strlen")]
    public static extern nuint StringLength(string s);

    // Call Windows MessageBox
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);

    // Call a math function from the C runtime
    [DllImport("libm", EntryPoint = "sqrt")]
    public static extern double Sqrt(double x);
}

// Usage:
// nuint len = NativeMethods.StringLength("Hello");
// Console.WriteLine(len);  // 5

// double result = NativeMethods.Sqrt(16.0);
// Console.WriteLine(result);  // 4
```

### 1.2 DllImport Options

```csharp
public static class Win32
{
    [DllImport(
        "kernel32.dll",                      // Library name
        EntryPoint = "GetCurrentProcessId",  // Exact function name (if different from C# name)
        SetLastError = true,                 // Capture Win32 error code
        CharSet = CharSet.Unicode,           // String encoding
        ExactSpelling = true,                // Don't search for A/W suffix variants
        CallingConvention = CallingConvention.Winapi  // Calling convention
    )]
    public static extern uint GetCurrentProcessId();

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern IntPtr CreateFile(
        string lpFileName,
        uint dwDesiredAccess,
        uint dwShareMode,
        IntPtr lpSecurityAttributes,
        uint dwCreationDisposition,
        uint dwFlagsAndAttributes,
        IntPtr hTemplateFile);
}

// Check for errors after P/Invoke:
// IntPtr handle = Win32.CreateFile(...);
// if (handle == IntPtr.Zero)
// {
//     int errorCode = Marshal.GetLastWin32Error();
//     throw new System.ComponentModel.Win32Exception(errorCode);
// }
```

### 1.3 Cross-Platform P/Invoke

```csharp
public static class CrossPlatformNative
{
    // The runtime resolves the correct library name per platform:
    // Windows: "mylibrary.dll"
    // Linux:   "libmylibrary.so"
    // macOS:   "libmylibrary.dylib"

    // Using NativeLibrary for manual resolution
    public static double CallNativeSqrt(double value)
    {
        string libName = RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
            ? "msvcrt"
            : "libm";

        IntPtr libHandle = NativeLibrary.Load(libName);
        try
        {
            IntPtr funcPtr = NativeLibrary.GetExport(libHandle, "sqrt");
            var sqrtFunc = Marshal.GetDelegateForFunctionPointer<SqrtDelegate>(funcPtr);
            return sqrtFunc(value);
        }
        finally
        {
            NativeLibrary.Free(libHandle);
        }
    }

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate double SqrtDelegate(double x);
}
```

## 2. LibraryImport (.NET 7+ Source-Generated P/Invoke)

### 2.1 Basic LibraryImport

`LibraryImport` is the modern replacement for `DllImport`. It uses source generators for better performance and AOT compatibility:

```csharp
using System.Runtime.InteropServices;

public static partial class ModernNative
{
    // Note: the class must be partial, and the method must be static partial
    [LibraryImport("libc", EntryPoint = "abs")]
    public static partial int Abs(int value);

    [LibraryImport("libm", EntryPoint = "pow")]
    public static partial double Pow(double x, double y);

    [LibraryImport("libm", EntryPoint = "floor")]
    public static partial double Floor(double x);
}

// Usage:
// int absolute = ModernNative.Abs(-42);  // 42
// double result = ModernNative.Pow(2.0, 10.0);  // 1024.0
```

### 2.2 String Marshaling with LibraryImport

```csharp
public static partial class StringInterop
{
    // UTF-8 marshaling (default for LibraryImport)
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf8)]
    public static partial int ProcessUtf8String(string input);

    // UTF-16 marshaling (for Windows APIs)
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int ProcessUtf16String(string input);

    // Custom marshaler
    [LibraryImport("mylib")]
    public static partial int ProcessData(
        [MarshalAs(UnmanagedType.LPStr)] string ansiString);  // ANSI string

    // Output string parameter
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf8)]
    public static partial int GetName(
        [MarshalUsing(typeof(Utf8StringMarshaller))] out string name);
}
```

### 2.3 DllImport vs LibraryImport Comparison

```csharp
// OLD: DllImport (runtime stub generation)
public static class OldStyle
{
    [DllImport("mylib", CharSet = CharSet.Unicode)]
    public static extern int Calculate(string input, out int result);
}

// NEW: LibraryImport (compile-time source generation)
public static partial class NewStyle
{
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int Calculate(string input, out int result);
}

// Benefits of LibraryImport:
// 1. Source-generated marshaling code (visible, debuggable)
// 2. Better performance (no runtime stub generation)
// 3. AOT (Ahead-of-Time) compilation compatible
// 4. Trimming friendly
// 5. Analyzers catch errors at compile time
```

## 3. Marshaling Strings, Structs, and Arrays

### 3.1 Struct Marshaling

```csharp
using System.Runtime.InteropServices;

// Match the layout of a C struct:
// typedef struct {
//     int x;
//     int y;
//     char name[32];
// } Point;

[StructLayout(LayoutKind.Sequential)]
public struct Point
{
    public int X;
    public int Y;

    [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)]
    public string Name;
}

// Explicit layout (for unions or overlapping fields):
[StructLayout(LayoutKind.Explicit)]
public struct Variant
{
    [FieldOffset(0)] public int IntValue;
    [FieldOffset(0)] public float FloatValue;
    [FieldOffset(0)] public double DoubleValue;
    [FieldOffset(8)] public VariantType Type;
}

public enum VariantType : int
{
    Int = 0,
    Float = 1,
    Double = 2
}

// Sequential with packing control:
[StructLayout(LayoutKind.Sequential, Pack = 1)]  // No padding between fields
public struct PackedData
{
    public byte Flag;      // offset 0
    public int Value;      // offset 1 (normally would be offset 4 with default packing)
    public short Count;    // offset 5
}
// Total size with Pack=1: 7 bytes
// Total size with default packing: 12 bytes
```

### 3.2 Array Marshaling

```csharp
public static partial class ArrayInterop
{
    // Fixed-size array in a struct
    [StructLayout(LayoutKind.Sequential)]
    public struct Matrix3x3
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 9)]
        public float[] Elements;
    }

    // Passing an array to a native function
    // C signature: void process_data(int* data, int length);
    [LibraryImport("mylib")]
    public static partial void ProcessData(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] int[] data,
        int length);

    // Receiving an array from native code
    // C signature: int get_results(float* buffer, int bufferSize);
    [LibraryImport("mylib")]
    public static partial int GetResults(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] float[] buffer,
        int bufferSize);
}

// Usage:
// int[] data = { 1, 2, 3, 4, 5 };
// ArrayInterop.ProcessData(data, data.Length);

// float[] results = new float[100];
// int count = ArrayInterop.GetResults(results, results.Length);
```

### 3.3 Callback Marshaling (Function Pointers)

```csharp
// C signature: typedef int (*CompareFunc)(int a, int b);
//              void sort_array(int* arr, int len, CompareFunc cmp);

// Define a delegate matching the native callback signature
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate int CompareFunc(int a, int b);

public static class SortInterop
{
    [DllImport("mylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void sort_array(int[] arr, int len, CompareFunc cmp);
}

// Usage:
// int Ascending(int a, int b) => a.CompareTo(b);
// int[] array = { 5, 3, 8, 1, 9 };
// SortInterop.sort_array(array, array.Length, Ascending);
// Array is now: { 1, 3, 5, 8, 9 }

// .NET 5+ function pointer syntax (faster, no delegate allocation):
// [LibraryImport("mylib")]
// public static unsafe partial void sort_array(
//     int* arr, int len, delegate* unmanaged[Cdecl]<int, int, int> cmp);
```

## 4. Calling Native C Libraries from C#

### 4.1 Complete Example: Wrapping a C Math Library

Suppose we have a native C library `mathutils` with these functions:

```c
// mathutils.h
typedef struct {
    double x;
    double y;
} Vector2;

double vector2_length(Vector2 v);
Vector2 vector2_add(Vector2 a, Vector2 b);
Vector2 vector2_normalize(Vector2 v);
double vector2_dot(Vector2 a, Vector2 b);
```

The C# wrapper:

```csharp
using System.Runtime.InteropServices;

[StructLayout(LayoutKind.Sequential)]
public struct Vector2
{
    public double X;
    public double Y;

    public Vector2(double x, double y) { X = x; Y = y; }
    public override string ToString() => $"({X:F2}, {Y:F2})";
}

public static partial class MathUtils
{
    private const string LibName = "mathutils";

    [LibraryImport(LibName, EntryPoint = "vector2_length")]
    public static partial double Length(Vector2 v);

    [LibraryImport(LibName, EntryPoint = "vector2_add")]
    public static partial Vector2 Add(Vector2 a, Vector2 b);

    [LibraryImport(LibName, EntryPoint = "vector2_normalize")]
    public static partial Vector2 Normalize(Vector2 v);

    [LibraryImport(LibName, EntryPoint = "vector2_dot")]
    public static partial double Dot(Vector2 a, Vector2 b);
}

// Usage:
// var a = new Vector2(3.0, 4.0);
// var b = new Vector2(1.0, 2.0);
//
// double len = MathUtils.Length(a);          // 5.0
// Vector2 sum = MathUtils.Add(a, b);        // (4.0, 6.0)
// Vector2 norm = MathUtils.Normalize(a);    // (0.6, 0.8)
// double dot = MathUtils.Dot(a, b);         // 11.0
```

### 4.2 SafeHandle for Native Resources

```csharp
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

// C API:
// void* database_open(const char* path);
// int database_query(void* db, const char* sql, char* result, int resultSize);
// void database_close(void* db);

// SafeHandle prevents resource leaks even if exceptions occur
public class DatabaseSafeHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    private DatabaseSafeHandle() : base(true) { }

    protected override bool ReleaseHandle()
    {
        NativeDatabase.database_close(handle);
        return true;
    }
}

public static partial class NativeDatabase
{
    [LibraryImport("mydb", StringMarshaling = StringMarshaling.Utf8)]
    public static partial DatabaseSafeHandle database_open(string path);

    [LibraryImport("mydb", StringMarshaling = StringMarshaling.Utf8)]
    public static partial int database_query(
        DatabaseSafeHandle db,
        string sql,
        [Out] byte[] result,
        int resultSize);

    [LibraryImport("mydb")]
    internal static partial void database_close(IntPtr db);
}

// Usage with automatic cleanup:
// using var db = NativeDatabase.database_open("/path/to/data.db");
// byte[] buffer = new byte[1024];
// int bytesRead = NativeDatabase.database_query(db, "SELECT * FROM users", buffer, buffer.Length);
// string result = Encoding.UTF8.GetString(buffer, 0, bytesRead);
// db is automatically closed when disposed (even if an exception occurs)
```

## 5. The unsafe Keyword and Unsafe Context

### 5.1 Enabling Unsafe Code

To use unsafe code, you must enable it in the project file and mark code blocks:

```xml
<!-- In .csproj -->
<PropertyGroup>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>
```

```csharp
// Unsafe can be applied to methods, classes, or blocks
public class UnsafeDemo
{
    // Entire method is unsafe
    public unsafe void UnsafeMethod()
    {
        int x = 42;
        int* ptr = &x;
        Console.WriteLine(*ptr);  // 42
    }

    // Only a block is unsafe
    public void MixedMethod()
    {
        int value = 100;

        // Safe code here...

        unsafe
        {
            int* ptr = &value;
            *ptr = 200;
        }

        // value is now 200
        Console.WriteLine(value);
    }
}

// Unsafe struct
public unsafe struct UnsafeBuffer
{
    public fixed byte Data[256];  // Fixed-size buffer (no heap allocation)
    public int Length;
}
```

### 5.2 When to Use unsafe

```csharp
// VALID reasons to use unsafe:
// 1. Interop with native code that requires pointers
// 2. Performance-critical hot paths (image processing, math-heavy computation)
// 3. Working with fixed-size buffers in structs
// 4. Interfacing with hardware or memory-mapped I/O

// AVOID unsafe when:
// 1. Span<T> or Memory<T> can do the job (almost always)
// 2. You just want to skip bounds checking (use Span indexer instead)
// 3. The code is not performance-critical
// 4. You are not experienced with memory management

// In modern .NET, approximately 95% of cases where you might consider unsafe
// can be handled safely with Span<T>, stackalloc in safe contexts, or
// Unsafe.As/Unsafe.Add from System.Runtime.CompilerServices.
```

## 6. Pointer Types and Operations

### 6.1 Pointer Basics

```csharp
public unsafe class PointerBasics
{
    public static void Demonstrate()
    {
        // Declare and use pointers
        int value = 42;
        int* ptr = &value;     // Address-of operator

        Console.WriteLine($"Value: {value}");           // 42
        Console.WriteLine($"Address: {(nint)ptr:X}");   // Memory address in hex
        Console.WriteLine($"Dereferenced: {*ptr}");     // 42

        // Modify through pointer
        *ptr = 100;
        Console.WriteLine($"Value after: {value}");     // 100

        // Pointer to pointer
        int** ptrToPtr = &ptr;
        Console.WriteLine($"Double deref: {**ptrToPtr}"); // 100

        // Void pointer (type-erased)
        void* voidPtr = ptr;
        int* castBack = (int*)voidPtr;
        Console.WriteLine($"Cast back: {*castBack}");   // 100
    }
}
```

### 6.2 Pointer Arithmetic

```csharp
public unsafe class PointerArithmetic
{
    public static void Demonstrate()
    {
        // Stack-allocated array
        int* array = stackalloc int[5];
        for (int i = 0; i < 5; i++)
        {
            array[i] = (i + 1) * 10;  // 10, 20, 30, 40, 50
        }

        // Pointer indexing
        Console.WriteLine(array[0]);  // 10
        Console.WriteLine(array[4]);  // 50

        // Pointer arithmetic (advances by sizeof(int) = 4 bytes per step)
        int* p = array;
        Console.WriteLine(*p);       // 10
        p++;                          // Move to next int
        Console.WriteLine(*p);       // 20
        p += 2;                       // Skip two ints
        Console.WriteLine(*p);       // 40

        // Pointer difference
        int* start = array;
        int* end = array + 5;
        long count = end - start;     // 5 (number of elements, not bytes)
        Console.WriteLine($"Element count: {count}");

        // Iterate with pointers
        for (int* current = array; current < array + 5; current++)
        {
            Console.Write($"{*current} ");
        }
        // Output: 10 20 30 40 50
    }
}
```

### 6.3 Working with Byte Buffers

```csharp
public unsafe class ByteBufferOps
{
    // Write an integer into a byte buffer at a specific offset (little-endian)
    public static void WriteInt32(byte* buffer, int offset, int value)
    {
        *(int*)(buffer + offset) = value;
    }

    // Read an integer from a byte buffer at a specific offset
    public static int ReadInt32(byte* buffer, int offset)
    {
        return *(int*)(buffer + offset);
    }

    public static void Demonstrate()
    {
        byte* buffer = stackalloc byte[16];

        // Write values at specific offsets
        WriteInt32(buffer, 0, 42);
        WriteInt32(buffer, 4, 100);
        WriteInt32(buffer, 8, -1);
        WriteInt32(buffer, 12, int.MaxValue);

        // Read them back
        for (int i = 0; i < 4; i++)
        {
            int value = ReadInt32(buffer, i * 4);
            Console.WriteLine($"Offset {i * 4}: {value}");
        }
        // Offset 0: 42
        // Offset 4: 100
        // Offset 8: -1
        // Offset 12: 2147483647
    }
}
```

## 7. The fixed Statement for Pinning

### 7.1 Why Pinning Is Needed

The garbage collector can move objects in memory during compaction. If native code or pointer operations reference a managed object, it must be pinned so the GC does not move it:

```csharp
public unsafe class FixedDemo
{
    public static void PinArray()
    {
        int[] numbers = { 10, 20, 30, 40, 50 };

        // Pin the array so the GC won't move it
        fixed (int* ptr = numbers)
        {
            // Inside this block, 'numbers' is pinned at a fixed memory address
            for (int i = 0; i < numbers.Length; i++)
            {
                Console.Write($"{*(ptr + i)} ");
            }
            Console.WriteLine();

            // Modify through pointer
            *(ptr + 2) = 999;
        }
        // Array is unpinned here — GC can move it again

        Console.WriteLine(numbers[2]);  // 999
    }

    public static void PinString()
    {
        string text = "Hello, World!";

        fixed (char* ptr = text)
        {
            // Walk the string character by character using a pointer
            for (char* p = ptr; *p != '\0'; p++)
            {
                Console.Write($"{*p} ");
            }
            Console.WriteLine();
        }
        // H e l l o ,   W o r l d !
    }

    public static void PinStructField()
    {
        var point = new double[] { 3.0, 4.0, 5.0 };

        fixed (double* p = point)
        {
            // Compute Euclidean length using pointer access
            double sum = 0;
            for (int i = 0; i < 3; i++)
                sum += p[i] * p[i];
            double length = Math.Sqrt(sum);
            Console.WriteLine($"Length: {length:F4}");  // 7.0711
        }
    }
}
```

### 7.2 Fixed-Size Buffers in Structs

```csharp
// Fixed-size buffers avoid heap allocation entirely
public unsafe struct Packet
{
    public int Id;
    public fixed byte Payload[256];  // Inline 256-byte buffer
    public int PayloadLength;

    public void SetPayload(ReadOnlySpan<byte> data)
    {
        if (data.Length > 256)
            throw new ArgumentException("Payload too large");

        fixed (byte* dest = Payload)
        {
            data.CopyTo(new Span<byte>(dest, 256));
        }
        PayloadLength = data.Length;
    }

    public ReadOnlySpan<byte> GetPayload()
    {
        fixed (byte* src = Payload)
        {
            return new ReadOnlySpan<byte>(src, PayloadLength);
        }
    }
}
```

## 8. stackalloc

### 8.1 Unsafe stackalloc

`stackalloc` allocates memory on the stack instead of the heap. It is extremely fast but limited in size:

```csharp
public unsafe class StackAllocUnsafe
{
    public static void Demonstrate()
    {
        // Allocate 100 integers on the stack (400 bytes)
        int* buffer = stackalloc int[100];

        // Fill with squares
        for (int i = 0; i < 100; i++)
            buffer[i] = i * i;

        // Sum them
        long sum = 0;
        for (int i = 0; i < 100; i++)
            sum += buffer[i];

        Console.WriteLine($"Sum of squares 0-99: {sum}");  // 328350

        // Memory is automatically freed when the method returns
        // No GC pressure, no heap allocation
    }
}
```

### 8.2 Safe stackalloc with Span (C# 7.2+)

In modern C#, you can use `stackalloc` without `unsafe` by assigning to a `Span<T>`:

```csharp
public class SafeStackAlloc
{
    public static void Demonstrate()
    {
        // Safe stackalloc - no unsafe keyword needed!
        Span<int> buffer = stackalloc int[100];

        for (int i = 0; i < buffer.Length; i++)
            buffer[i] = i * i;

        long sum = 0;
        foreach (int val in buffer)
            sum += val;

        Console.WriteLine($"Sum: {sum}");  // 328350

        // Bounds-checked: buffer[100] would throw IndexOutOfRangeException
    }

    // Pattern: fall back to heap for large allocations
    public static double ComputeAverage(int count)
    {
        const int stackAllocThreshold = 256;

        Span<double> values = count <= stackAllocThreshold
            ? stackalloc double[count]    // Stack (fast, small)
            : new double[count];          // Heap (safe for large)

        // Fill with sample data
        for (int i = 0; i < values.Length; i++)
            values[i] = i * 1.5;

        double sum = 0;
        foreach (double v in values)
            sum += v;

        return sum / values.Length;
    }
}
```

## 9. The sizeof Operator

```csharp
public unsafe class SizeOfDemo
{
    public static void Demonstrate()
    {
        // sizeof for primitive types (works in safe context too)
        Console.WriteLine($"sizeof(byte):    {sizeof(byte)}");     // 1
        Console.WriteLine($"sizeof(short):   {sizeof(short)}");    // 2
        Console.WriteLine($"sizeof(int):     {sizeof(int)}");      // 4
        Console.WriteLine($"sizeof(long):    {sizeof(long)}");     // 8
        Console.WriteLine($"sizeof(float):   {sizeof(float)}");    // 4
        Console.WriteLine($"sizeof(double):  {sizeof(double)}");   // 8
        Console.WriteLine($"sizeof(decimal): {sizeof(decimal)}");  // 16
        Console.WriteLine($"sizeof(char):    {sizeof(char)}");     // 2 (UTF-16)
        Console.WriteLine($"sizeof(bool):    {sizeof(bool)}");     // 1
        Console.WriteLine($"sizeof(nint):    {sizeof(nint)}");     // 4 or 8 (platform-dependent)

        // sizeof for structs (requires unsafe for non-primitive types)
        Console.WriteLine($"sizeof(Point):   {sizeof(Point)}");    // 16 (two doubles)
        Console.WriteLine($"sizeof(Guid):    {sizeof(Guid)}");     // 16

        // Use Marshal.SizeOf for marshaled size (may differ due to padding)
        Console.WriteLine($"Marshal.SizeOf<Point>(): {Marshal.SizeOf<Point>()}");
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Point { public double X; public double Y; }
}
```

## 10. Span<T> as a Safe Alternative to Pointers

### 10.1 Span<T> Basics

`Span<T>` provides a type-safe, bounds-checked view over contiguous memory without requiring `unsafe`:

```csharp
public class SpanBasics
{
    public static void Demonstrate()
    {
        // Span over an array
        int[] array = { 1, 2, 3, 4, 5, 6, 7, 8 };
        Span<int> span = array;

        // Slice (no allocation!)
        Span<int> middle = span.Slice(2, 4);  // { 3, 4, 5, 6 }
        Console.WriteLine(string.Join(", ", middle.ToArray()));

        // Modify through span (modifies the original array)
        middle[0] = 99;
        Console.WriteLine(array[2]);  // 99

        // Span over stackalloc
        Span<byte> stackBuffer = stackalloc byte[64];
        stackBuffer.Fill(0xFF);
        Console.WriteLine(stackBuffer[0]);  // 255

        // Span over unmanaged memory
        // IntPtr unmanagedMem = Marshal.AllocHGlobal(100);
        // Span<byte> unmanagedSpan;
        // unsafe { unmanagedSpan = new Span<byte>((void*)unmanagedMem, 100); }
        // Marshal.FreeHGlobal(unmanagedMem);
    }
}
```

### 10.2 String Processing with ReadOnlySpan<char>

```csharp
public class SpanStringProcessing
{
    // Parse integers from a comma-separated string without any heap allocations
    public static int SumCsvInts(ReadOnlySpan<char> csv)
    {
        int sum = 0;

        while (!csv.IsEmpty)
        {
            int commaIndex = csv.IndexOf(',');
            ReadOnlySpan<char> token;

            if (commaIndex >= 0)
            {
                token = csv[..commaIndex].Trim();
                csv = csv[(commaIndex + 1)..];
            }
            else
            {
                token = csv.Trim();
                csv = ReadOnlySpan<char>.Empty;
            }

            if (int.TryParse(token, out int value))
                sum += value;
        }

        return sum;
    }

    // Extract a value from a "key=value" string
    public static ReadOnlySpan<char> GetValue(ReadOnlySpan<char> pair)
    {
        int eqIndex = pair.IndexOf('=');
        if (eqIndex < 0) return ReadOnlySpan<char>.Empty;
        return pair[(eqIndex + 1)..].Trim();
    }

    public static void Demonstrate()
    {
        int sum = SumCsvInts("10, 20, 30, 40, 50");
        Console.WriteLine($"Sum: {sum}");  // 150

        ReadOnlySpan<char> value = GetValue("timeout=30");
        Console.WriteLine($"Value: {value.ToString()}");  // 30
    }
}
```

### 10.3 Span<T> for Binary Data Processing

```csharp
public class BinaryProcessor
{
    // Read a little-endian int32 from a span
    public static int ReadInt32LittleEndian(ReadOnlySpan<byte> source)
    {
        return System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(source);
    }

    // Write a little-endian int32 to a span
    public static void WriteInt32LittleEndian(Span<byte> dest, int value)
    {
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(dest, value);
    }

    // Parse a simple binary header: [magic:4][version:4][length:4][payload:N]
    public static (int Version, ReadOnlySpan<byte> Payload) ParsePacket(ReadOnlySpan<byte> data)
    {
        const int MagicNumber = 0x4D594150;  // "MYAP"

        if (data.Length < 12)
            throw new ArgumentException("Packet too short");

        int magic = ReadInt32LittleEndian(data[..4]);
        if (magic != MagicNumber)
            throw new InvalidOperationException("Invalid magic number");

        int version = ReadInt32LittleEndian(data[4..8]);
        int length = ReadInt32LittleEndian(data[8..12]);

        if (data.Length < 12 + length)
            throw new ArgumentException("Packet truncated");

        return (version, data.Slice(12, length));
    }

    public static void Demonstrate()
    {
        // Build a packet
        Span<byte> packet = stackalloc byte[20];
        WriteInt32LittleEndian(packet[..4], 0x4D594150);    // Magic
        WriteInt32LittleEndian(packet[4..8], 2);            // Version 2
        WriteInt32LittleEndian(packet[8..12], 8);           // Payload length
        packet[12..20].Fill(0xAB);                          // Payload data

        var (version, payload) = ParsePacket(packet);
        Console.WriteLine($"Version: {version}, Payload length: {payload.Length}");
        // Version: 2, Payload length: 8
    }
}
```

### 10.4 Memory<T> for Async Scenarios

```csharp
public class MemoryDemo
{
    // Span<T> is a ref struct — it cannot be stored on the heap or used in async methods.
    // Memory<T> is the heap-friendly counterpart.

    public static async Task ProcessDataAsync(Memory<byte> buffer)
    {
        // Fill with data
        for (int i = 0; i < buffer.Length; i++)
            buffer.Span[i] = (byte)(i % 256);

        // Simulate async I/O
        await Task.Delay(10);

        // Process in chunks
        int chunkSize = 16;
        for (int offset = 0; offset < buffer.Length; offset += chunkSize)
        {
            int remaining = Math.Min(chunkSize, buffer.Length - offset);
            Memory<byte> chunk = buffer.Slice(offset, remaining);

            // Can pass Memory<T> to async operations
            await ProcessChunkAsync(chunk);
        }
    }

    private static async Task ProcessChunkAsync(Memory<byte> chunk)
    {
        // Access the underlying Span when you need it
        Span<byte> span = chunk.Span;
        int sum = 0;
        foreach (byte b in span)
            sum += b;

        await Task.Delay(1);
    }
}
```

## 11. Practical Example: Calling a Native Math Library

This example ties together P/Invoke, marshaling, and safe interop to wrap a hypothetical native statistics library:

```csharp
using System.Runtime.InteropServices;

// ---- Native API Wrapper ----

[StructLayout(LayoutKind.Sequential)]
public struct Statistics
{
    public double Mean;
    public double Median;
    public double StandardDeviation;
    public double Min;
    public double Max;
    public int Count;
}

public static partial class NativeStats
{
    // Hypothetical native functions:
    // Statistics compute_stats(const double* data, int count);
    // void sort_doubles(double* data, int count);
    // double percentile(const double* sorted_data, int count, double p);

    [LibraryImport("statslib")]
    public static partial Statistics compute_stats(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] double[] data,
        int count);

    [LibraryImport("statslib")]
    public static partial void sort_doubles(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] double[] data,
        int count);

    [LibraryImport("statslib")]
    public static partial double percentile(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] double[] sortedData,
        int count,
        double p);
}

// ---- Managed Wrapper for Safety ----

public class StatisticsService
{
    /// <summary>
    /// Computes descriptive statistics for the given data using the native library.
    /// Falls back to a managed implementation if the native library is unavailable.
    /// </summary>
    public Statistics Compute(ReadOnlySpan<double> data)
    {
        if (data.IsEmpty)
            throw new ArgumentException("Data cannot be empty");

        // Copy to an array for P/Invoke (P/Invoke needs a pinnable array)
        double[] array = data.ToArray();

        try
        {
            return NativeStats.compute_stats(array, array.Length);
        }
        catch (DllNotFoundException)
        {
            // Fallback to managed implementation
            return ComputeManaged(array);
        }
    }

    private static Statistics ComputeManaged(double[] data)
    {
        Array.Sort(data);
        double sum = 0;
        foreach (double d in data) sum += d;
        double mean = sum / data.Length;

        double varianceSum = 0;
        foreach (double d in data)
            varianceSum += (d - mean) * (d - mean);

        double median = data.Length % 2 == 0
            ? (data[data.Length / 2 - 1] + data[data.Length / 2]) / 2.0
            : data[data.Length / 2];

        return new Statistics
        {
            Mean = mean,
            Median = median,
            StandardDeviation = Math.Sqrt(varianceSum / data.Length),
            Min = data[0],
            Max = data[^1],
            Count = data.Length
        };
    }
}

// ---- Usage ----

// var service = new StatisticsService();
// double[] data = { 4.5, 2.3, 7.8, 1.1, 9.0, 3.4, 6.7, 5.2 };
// Statistics stats = service.Compute(data);
// Console.WriteLine($"Mean: {stats.Mean:F2}");
// Console.WriteLine($"Median: {stats.Median:F2}");
// Console.WriteLine($"StdDev: {stats.StandardDeviation:F2}");
// Console.WriteLine($"Range: [{stats.Min:F2}, {stats.Max:F2}]");
// Console.WriteLine($"Count: {stats.Count}");
```

## 12. Practice Problems

1. **P/Invoke Wrapper**: Write a C# wrapper for the POSIX `time` function (`time_t time(time_t *tloc)`) and the `ctime` function (`char *ctime(const time_t *timep)`). Call `time` to get the current Unix timestamp, then pass it to `ctime` to get a human-readable string. Handle the string marshaling properly.

2. **Struct Marshaling**: Define a C struct for a network packet header with fields: `version` (uint8), `type` (uint8), `length` (uint16), `sequence` (uint32), and `checksum` (uint32). Create the matching C# struct with `StructLayout`. Verify the total size is 12 bytes (with `Pack = 1`). Write methods to serialize and deserialize the header to/from a `Span<byte>`.

3. **Pointer-Based Image Processing**: Write an unsafe method that takes a `byte[]` representing grayscale pixel values and inverts each pixel (newValue = 255 - oldValue). Use pointer arithmetic to iterate through the array. Then write an equivalent safe version using `Span<byte>` and compare the readability.

4. **Safe stackalloc Histogram**: Write a method that computes a histogram of byte values (0-255) in a `ReadOnlySpan<byte>`. Use `stackalloc int[256]` (assigned to `Span<int>`) to avoid heap allocation for the histogram bins. Return the top 5 most frequent values.

5. **Memory<T> Pipeline**: Create an async data processing pipeline where: (a) a producer writes random doubles into a `Memory<double>` buffer, (b) a processor reads the buffer and computes a running average, and (c) a consumer prints results. All stages communicate through `Memory<T>` slices. Explain why you cannot use `Span<T>` here.
