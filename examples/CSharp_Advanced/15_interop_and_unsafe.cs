// Lesson 15: Interop and Unsafe Code
// Run: dotnet run
// Note: Requires <AllowUnsafeBlocks>true</AllowUnsafeBlocks> in .csproj

using System;
using System.Runtime.InteropServices;

// ============================================================
// 1. P/Invoke — Calling Native C Functions
// ============================================================

Console.WriteLine("=== P/Invoke Basics ===");

// Platform-specific P/Invoke examples
if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
{
    // Windows: call kernel32.dll
    Console.WriteLine("  Running on Windows");
    // uint tickCount = NativeWindows.GetTickCount();
    // Console.WriteLine($"  System uptime (ms): {tickCount}");
}
else if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) ||
         RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
{
    Console.WriteLine($"  Running on {RuntimeInformation.OSDescription}");

    // Call libc getpid()
    int pid = NativeUnix.getpid();
    Console.WriteLine($"  Process ID (via P/Invoke): {pid}");
    Console.WriteLine($"  Process ID (via .NET):     {Environment.ProcessId}");

    // Call libc strlen()
    nint len = NativeUnix.strlen("Hello, World!");
    Console.WriteLine($"  strlen(\"Hello, World!\") = {len}");
}

// ============================================================
// 2. StructLayout — Marshalling Structs
// ============================================================

Console.WriteLine("\n=== Struct Marshalling ===");

// Sequential layout — fields are in order, like C struct
var point = new NativePoint { X = 10, Y = 20 };
Console.WriteLine($"NativePoint: ({point.X}, {point.Y})");
Console.WriteLine($"NativePoint size: {Marshal.SizeOf<NativePoint>()} bytes");

// Explicit layout — control exact byte offsets (union-like)
var color = new ColorUnion { ARGB = 0xFF_80_40_20 };
Console.WriteLine($"\nColorUnion ARGB = 0x{color.ARGB:X8}");
Console.WriteLine($"  A={color.A}, R={color.R}, G={color.G}, B={color.B}");

// ============================================================
// 3. Unsafe Pointers
// ============================================================

Console.WriteLine("\n=== Unsafe Pointers ===");

unsafe
{
    // Pointer to a local variable
    int value = 42;
    int* ptr = &value;
    Console.WriteLine($"  value = {value}");
    Console.WriteLine($"  *ptr  = {*ptr}");
    Console.WriteLine($"  Address: 0x{(nint)ptr:X}");

    // Modify through pointer
    *ptr = 100;
    Console.WriteLine($"  After *ptr = 100: value = {value}");

    // Pointer arithmetic
    Console.WriteLine("\n  Pointer arithmetic:");
    int* arr = stackalloc int[5];
    for (int i = 0; i < 5; i++)
        arr[i] = i * 10;

    for (int i = 0; i < 5; i++)
        Console.WriteLine($"    arr[{i}] = {*(arr + i)}");
}

// ============================================================
// 4. Fixed Statement — Pinning Managed Objects
// ============================================================

Console.WriteLine("\n=== Fixed Statement ===");

int[] managedArray = { 10, 20, 30, 40, 50 };

unsafe
{
    // 'fixed' pins the array so the GC won't move it
    fixed (int* p = managedArray)
    {
        Console.WriteLine($"  Array pinned at 0x{(nint)p:X}");

        // Read using pointer
        for (int i = 0; i < managedArray.Length; i++)
            Console.Write($"  {p[i]}");
        Console.WriteLine();

        // Modify using pointer
        p[2] = 999;
    }
    // Array is unpinned here

    Console.WriteLine($"  After modification: [{string.Join(", ", managedArray)}]");

    // Fixed with strings
    string text = "Hello";
    fixed (char* charPtr = text)
    {
        Console.Write("  String chars: ");
        for (int i = 0; i < text.Length; i++)
            Console.Write($"{charPtr[i]} ");
        Console.WriteLine();
    }
}

// ============================================================
// 5. Stackalloc
// ============================================================

Console.WriteLine("\n=== Stackalloc ===");

unsafe
{
    // Allocate on the stack (no GC)
    byte* buffer = stackalloc byte[256];

    // Fill with pattern
    for (int i = 0; i < 256; i++)
        buffer[i] = (byte)(i & 0xFF);

    Console.WriteLine($"  buffer[0]={buffer[0]}, buffer[128]={buffer[128]}, buffer[255]={buffer[255]}");
}

// Safe stackalloc with Span (no unsafe needed)
Span<int> safeStack = stackalloc int[10];
for (int i = 0; i < safeStack.Length; i++)
    safeStack[i] = i * i;
Console.WriteLine($"  Safe stackalloc: [{string.Join(", ", safeStack.ToArray())}]");

// ============================================================
// 6. sizeof and Marshal.SizeOf
// ============================================================

Console.WriteLine("\n=== Size Information ===");

unsafe
{
    Console.WriteLine($"  sizeof(byte)   = {sizeof(byte)}");
    Console.WriteLine($"  sizeof(int)    = {sizeof(int)}");
    Console.WriteLine($"  sizeof(long)   = {sizeof(long)}");
    Console.WriteLine($"  sizeof(double) = {sizeof(double)}");
    Console.WriteLine($"  sizeof(nint)   = {sizeof(nint)} (pointer size)");
}

Console.WriteLine($"  Marshal.SizeOf<NativePoint>() = {Marshal.SizeOf<NativePoint>()}");
Console.WriteLine($"  Marshal.SizeOf<ColorUnion>()  = {Marshal.SizeOf<ColorUnion>()}");

// ============================================================
// 7. Memory Copy and Block Operations
// ============================================================

Console.WriteLine("\n=== Memory Operations ===");

unsafe
{
    int* src = stackalloc int[] { 1, 2, 3, 4, 5 };
    int* dst = stackalloc int[5];

    // Copy memory block
    Buffer.MemoryCopy(src, dst, 5 * sizeof(int), 5 * sizeof(int));

    Console.Write("  Copied: ");
    for (int i = 0; i < 5; i++)
        Console.Write($"{dst[i]} ");
    Console.WriteLine();
}

// ============================================================
// 8. Practical: Fast Array Sum with Unsafe
// ============================================================

Console.WriteLine("\n=== Performance: Unsafe Array Sum ===");

int[] data = new int[1_000_000];
var rng = new Random(42);
for (int i = 0; i < data.Length; i++)
    data[i] = rng.Next(100);

// Safe LINQ sum
var sw = System.Diagnostics.Stopwatch.StartNew();
long safeSum = 0;
for (int i = 0; i < data.Length; i++)
    safeSum += data[i];
sw.Stop();
Console.WriteLine($"  Safe sum:   {safeSum} ({sw.ElapsedTicks} ticks)");

// Unsafe pointer sum
sw.Restart();
long unsafeSum = UnsafeSum(data);
sw.Stop();
Console.WriteLine($"  Unsafe sum: {unsafeSum} ({sw.ElapsedTicks} ticks)");

static unsafe long UnsafeSum(int[] array)
{
    long sum = 0;
    fixed (int* ptr = array)
    {
        int* p = ptr;
        int* end = ptr + array.Length;
        while (p < end)
        {
            sum += *p;
            p++;
        }
    }
    return sum;
}

// ============================================================
// 9. Function Pointers (C# 9+)
// ============================================================

Console.WriteLine("\n=== Function Pointers ===");

unsafe
{
    // Managed function pointer
    delegate*<int, int, int> funcPtr = &MathAdd;
    int result = funcPtr(10, 20);
    Console.WriteLine($"  Function pointer result: {result}");

    delegate*<int, int, int> mulPtr = &MathMultiply;
    Console.WriteLine($"  Multiply via func ptr: {mulPtr(6, 7)}");
}

static int MathAdd(int a, int b) => a + b;
static int MathMultiply(int a, int b) => a * b;

// ============================================================
// Native Method Declarations
// ============================================================

static class NativeUnix
{
    [DllImport("libc")]
    public static extern int getpid();

    [DllImport("libc")]
    public static extern nint strlen(string s);
}

// Uncomment on Windows:
// static class NativeWindows
// {
//     [DllImport("kernel32.dll")]
//     public static extern uint GetTickCount();
// }

// ============================================================
// Marshalled Structs
// ============================================================

[StructLayout(LayoutKind.Sequential)]
struct NativePoint
{
    public int X;
    public int Y;
}

// Union-like struct using explicit layout
[StructLayout(LayoutKind.Explicit)]
struct ColorUnion
{
    [FieldOffset(0)] public uint ARGB;
    [FieldOffset(0)] public byte B;
    [FieldOffset(1)] public byte G;
    [FieldOffset(2)] public byte R;
    [FieldOffset(3)] public byte A;
}
