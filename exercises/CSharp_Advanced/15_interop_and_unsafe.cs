/*
 * Exercises for Lesson 15: Interop and Unsafe Code
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Runtime.InteropServices;
using System.Text;

// ---------------------------------------------------------------------------
// Exercise 1: P/Invoke — call native OS functions
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: P/Invoke Basics ===");

    // Environment information via managed wrappers (cross-platform safe)
    Console.WriteLine($"  OS: {RuntimeInformation.OSDescription}");
    Console.WriteLine($"  Architecture: {RuntimeInformation.OSArchitecture}");
    Console.WriteLine($"  Framework: {RuntimeInformation.FrameworkDescription}");
    Console.WriteLine($"  Process arch: {RuntimeInformation.ProcessArchitecture}");

    // Demonstrate StructLayout for interop struct definition
    var point = new NativePoint { X = 10, Y = 20 };
    int size = Marshal.SizeOf<NativePoint>();
    Console.WriteLine($"  NativePoint size: {size} bytes, value: ({point.X}, {point.Y})");

    var rect = new NativeRect { Left = 0, Top = 0, Right = 100, Bottom = 50 };
    Console.WriteLine($"  NativeRect size: {Marshal.SizeOf<NativeRect>()} bytes");
    Console.WriteLine($"  Rect: ({rect.Left},{rect.Top}) to ({rect.Right},{rect.Bottom})");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Marshal — string and memory marshalling
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Marshal Operations ===");

    // Allocate unmanaged memory and write a string
    string original = "Hello from unmanaged memory!";
    IntPtr ptr = Marshal.StringToHGlobalAnsi(original);
    try
    {
        string recovered = Marshal.PtrToStringAnsi(ptr)!;
        Console.WriteLine($"  Original : {original}");
        Console.WriteLine($"  Recovered: {recovered}");
        Console.WriteLine($"  Match: {original == recovered}");
    }
    finally
    {
        Marshal.FreeHGlobal(ptr);
        Console.WriteLine("  Memory freed.");
    }

    // Struct to unmanaged memory round-trip
    var original_pt = new NativePoint { X = 42, Y = 99 };
    int structSize = Marshal.SizeOf<NativePoint>();
    IntPtr structPtr = Marshal.AllocHGlobal(structSize);
    try
    {
        Marshal.StructureToPtr(original_pt, structPtr, false);
        var roundTrip = Marshal.PtrToStructure<NativePoint>(structPtr);
        Console.WriteLine($"  Struct round-trip: ({roundTrip.X}, {roundTrip.Y})");
    }
    finally
    {
        Marshal.FreeHGlobal(structPtr);
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Unsafe pointer arithmetic
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Unsafe Pointer Manipulation ===");

    int[] array = { 10, 20, 30, 40, 50 };

    unsafe
    {
        fixed (int* basePtr = array)
        {
            Console.WriteLine("  Array via pointers:");
            for (int i = 0; i < array.Length; i++)
            {
                int* current = basePtr + i;
                Console.WriteLine($"    [{i}] address offset={i * sizeof(int)}, value={*current}");
            }

            // Modify via pointer
            *(basePtr + 2) = 999;
        }
    }
    Console.WriteLine($"  After pointer write: array[2] = {array[2]}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Unsafe — swap and reverse with pointers
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Pointer-Based Swap and Reverse ===");

    int a = 10, b = 20;
    Console.WriteLine($"  Before swap: a={a}, b={b}");
    unsafe { UnsafeSwap(&a, &b); }
    Console.WriteLine($"  After swap : a={a}, b={b}");

    int[] data = { 1, 2, 3, 4, 5 };
    Console.WriteLine($"  Before reverse: [{string.Join(", ", data)}]");
    unsafe
    {
        fixed (int* ptr = data)
        {
            UnsafeReverse(ptr, data.Length);
        }
    }
    Console.WriteLine($"  After reverse : [{string.Join(", ", data)}]");
    Console.WriteLine();
}

unsafe void UnsafeSwap(int* a, int* b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}

unsafe void UnsafeReverse(int* arr, int length)
{
    int* left = arr;
    int* right = arr + length - 1;
    while (left < right)
    {
        int temp = *left;
        *left = *right;
        *right = temp;
        left++;
        right--;
    }
}

// ---------------------------------------------------------------------------
// Exercise 5: Stackalloc — stack-allocated buffer operations
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Stackalloc Buffer ===");

    // Compute Fibonacci on the stack
    const int count = 15;
    Span<long> fib = stackalloc long[count];
    fib[0] = 0;
    fib[1] = 1;
    for (int i = 2; i < count; i++)
        fib[i] = fib[i - 1] + fib[i - 2];

    Console.Write("  Fibonacci: ");
    for (int i = 0; i < count; i++)
        Console.Write($"{fib[i]} ");
    Console.WriteLine();

    // Stack-allocated byte buffer for encoding
    Span<byte> buffer = stackalloc byte[128];
    string message = "Stack-allocated encoding test";
    int bytesWritten = Encoding.UTF8.GetBytes(message.AsSpan(), buffer);
    string decoded = Encoding.UTF8.GetString(buffer[..bytesWritten]);
    Console.WriteLine($"  Encoded {bytesWritten} bytes on stack: \"{decoded}\"");
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

[StructLayout(LayoutKind.Sequential)]
struct NativePoint
{
    public int X;
    public int Y;
}

[StructLayout(LayoutKind.Sequential)]
struct NativeRect
{
    public int Left;
    public int Top;
    public int Right;
    public int Bottom;
}
