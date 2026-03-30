# 상호 운용과 안전하지 않은 코드

**이전**: [리플렉션과 어트리뷰트](./14_Reflection_and_Attributes.md) | **다음**: [성능과 프로파일링](./16_Performance_Profiling.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 플랫폼 호출(P/Invoke)을 사용하여 C#에서 네이티브 C/C++ 라이브러리를 호출할 수 있다
2. `DllImport`와 최신 `LibraryImport` 소스 생성 방식을 모두 사용할 수 있다
3. 관리 코드와 비관리 코드 간에 문자열, 구조체, 배열을 마샬링할 수 있다
4. 포인터 연산을 사용하는 unsafe 코드 블록을 작성할 수 있다
5. `fixed` 문을 사용하여 관리 객체를 메모리에 고정할 수 있다
6. `stackalloc`으로 스택에 메모리를 할당할 수 있다
7. 원시 포인터의 안전한 대안으로 `Span<T>`과 `Memory<T>`를 적용할 수 있다
8. unsafe 코드가 정당화되는 시점과 그 범위를 최소화하는 방법을 이해할 수 있다

---

C#은 자동 메모리 관리가 있는 관리 언어이지만, 실제 애플리케이션에서는 네이티브 라이브러리와 상호 작용하거나, 메모리를 직접 조작하거나, 마지막 성능까지 짜내야 할 때가 있습니다. C#의 상호 운용(Interop)과 unsafe 기능은 관리 런타임에서 제어된 탈출구를 제공합니다. 이 레슨에서는 네이티브 코드 호출을 위한 P/Invoke, 포인터 연산을 위한 `unsafe` 키워드, 그리고 원시 포인터의 필요성을 줄이는 `Span<T>` 같은 최신 안전 대안을 다룹니다.

## 1. DllImport를 사용한 플랫폼 호출 (P/Invoke)

### 1.1 기본 P/Invoke

P/Invoke는 C# 코드가 네이티브(비관리) 동적 라이브러리에서 내보낸 함수를 호출할 수 있게 합니다:

```csharp
using System.Runtime.InteropServices;

public static class NativeMethods
{
    // C 표준 라이브러리의 strlen 함수 호출
    [DllImport("libc", EntryPoint = "strlen")]
    public static extern nuint StringLength(string s);

    // Windows MessageBox 호출
    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    public static extern int MessageBox(IntPtr hWnd, string text, string caption, uint type);

    // C 런타임의 수학 함수 호출
    [DllImport("libm", EntryPoint = "sqrt")]
    public static extern double Sqrt(double x);
}

// 사용:
// nuint len = NativeMethods.StringLength("Hello");
// Console.WriteLine(len);  // 5

// double result = NativeMethods.Sqrt(16.0);
// Console.WriteLine(result);  // 4
```

### 1.2 DllImport 옵션

```csharp
public static class Win32
{
    [DllImport(
        "kernel32.dll",                      // 라이브러리 이름
        EntryPoint = "GetCurrentProcessId",  // 정확한 함수 이름 (C# 이름과 다른 경우)
        SetLastError = true,                 // Win32 오류 코드 캡처
        CharSet = CharSet.Unicode,           // 문자열 인코딩
        ExactSpelling = true,                // A/W 접미사 변형을 검색하지 않음
        CallingConvention = CallingConvention.Winapi  // 호출 규약
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

// P/Invoke 후 오류 확인:
// IntPtr handle = Win32.CreateFile(...);
// if (handle == IntPtr.Zero)
// {
//     int errorCode = Marshal.GetLastWin32Error();
//     throw new System.ComponentModel.Win32Exception(errorCode);
// }
```

### 1.3 크로스 플랫폼 P/Invoke

```csharp
public static class CrossPlatformNative
{
    // 런타임이 플랫폼별로 올바른 라이브러리 이름을 해결합니다:
    // Windows: "mylibrary.dll"
    // Linux:   "libmylibrary.so"
    // macOS:   "libmylibrary.dylib"

    // 수동 해결을 위해 NativeLibrary 사용
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

## 2. LibraryImport (.NET 7+ 소스 생성 P/Invoke)

### 2.1 기본 LibraryImport

`LibraryImport`는 `DllImport`의 최신 대체품입니다. 소스 생성기를 사용하여 더 나은 성능과 AOT 호환성을 제공합니다:

```csharp
using System.Runtime.InteropServices;

public static partial class ModernNative
{
    // 참고: 클래스는 partial이어야 하며 메서드는 static partial이어야 합니다
    [LibraryImport("libc", EntryPoint = "abs")]
    public static partial int Abs(int value);

    [LibraryImport("libm", EntryPoint = "pow")]
    public static partial double Pow(double x, double y);

    [LibraryImport("libm", EntryPoint = "floor")]
    public static partial double Floor(double x);
}

// 사용:
// int absolute = ModernNative.Abs(-42);  // 42
// double result = ModernNative.Pow(2.0, 10.0);  // 1024.0
```

### 2.2 LibraryImport에서의 문자열 마샬링

```csharp
public static partial class StringInterop
{
    // UTF-8 마샬링 (LibraryImport의 기본값)
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf8)]
    public static partial int ProcessUtf8String(string input);

    // UTF-16 마샬링 (Windows API용)
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int ProcessUtf16String(string input);

    // 사용자 지정 마샬러
    [LibraryImport("mylib")]
    public static partial int ProcessData(
        [MarshalAs(UnmanagedType.LPStr)] string ansiString);  // ANSI 문자열

    // 출력 문자열 매개변수
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf8)]
    public static partial int GetName(
        [MarshalUsing(typeof(Utf8StringMarshaller))] out string name);
}
```

### 2.3 DllImport vs LibraryImport 비교

```csharp
// 이전: DllImport (런타임 스텁 생성)
public static class OldStyle
{
    [DllImport("mylib", CharSet = CharSet.Unicode)]
    public static extern int Calculate(string input, out int result);
}

// 신규: LibraryImport (컴파일 타임 소스 생성)
public static partial class NewStyle
{
    [LibraryImport("mylib", StringMarshaling = StringMarshaling.Utf16)]
    public static partial int Calculate(string input, out int result);
}

// LibraryImport의 장점:
// 1. 소스 생성 마샬링 코드 (가시적, 디버그 가능)
// 2. 더 나은 성능 (런타임 스텁 생성 없음)
// 3. AOT (Ahead-of-Time) 컴파일 호환
// 4. 트리밍 친화적
// 5. 분석기가 컴파일 타임에 오류를 감지
```

## 3. 문자열, 구조체, 배열 마샬링

### 3.1 구조체 마샬링

```csharp
using System.Runtime.InteropServices;

// C 구조체의 레이아웃과 일치:
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

// 명시적 레이아웃 (공용체 또는 겹치는 필드용):
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

// 패킹 제어가 있는 순차적 레이아웃:
[StructLayout(LayoutKind.Sequential, Pack = 1)]  // 필드 간 패딩 없음
public struct PackedData
{
    public byte Flag;      // 오프셋 0
    public int Value;      // 오프셋 1 (기본 패킹에서는 보통 오프셋 4)
    public short Count;    // 오프셋 5
}
// Pack=1일 때 총 크기: 7바이트
// 기본 패킹일 때 총 크기: 12바이트
```

### 3.2 배열 마샬링

```csharp
public static partial class ArrayInterop
{
    // 구조체 내 고정 크기 배열
    [StructLayout(LayoutKind.Sequential)]
    public struct Matrix3x3
    {
        [MarshalAs(UnmanagedType.ByValArray, SizeConst = 9)]
        public float[] Elements;
    }

    // 네이티브 함수에 배열 전달
    // C 시그니처: void process_data(int* data, int length);
    [LibraryImport("mylib")]
    public static partial void ProcessData(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] int[] data,
        int length);

    // 네이티브 코드에서 배열 수신
    // C 시그니처: int get_results(float* buffer, int bufferSize);
    [LibraryImport("mylib")]
    public static partial int GetResults(
        [MarshalAs(UnmanagedType.LPArray, SizeParamIndex = 1)] float[] buffer,
        int bufferSize);
}

// 사용:
// int[] data = { 1, 2, 3, 4, 5 };
// ArrayInterop.ProcessData(data, data.Length);

// float[] results = new float[100];
// int count = ArrayInterop.GetResults(results, results.Length);
```

### 3.3 콜백 마샬링 (함수 포인터)

```csharp
// C 시그니처: typedef int (*CompareFunc)(int a, int b);
//              void sort_array(int* arr, int len, CompareFunc cmp);

// 네이티브 콜백 시그니처와 일치하는 대리자 정의
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
public delegate int CompareFunc(int a, int b);

public static class SortInterop
{
    [DllImport("mylib", CallingConvention = CallingConvention.Cdecl)]
    public static extern void sort_array(int[] arr, int len, CompareFunc cmp);
}

// 사용:
// int Ascending(int a, int b) => a.CompareTo(b);
// int[] array = { 5, 3, 8, 1, 9 };
// SortInterop.sort_array(array, array.Length, Ascending);
// 배열은 이제: { 1, 3, 5, 8, 9 }

// .NET 5+ 함수 포인터 구문 (더 빠름, 대리자 할당 없음):
// [LibraryImport("mylib")]
// public static unsafe partial void sort_array(
//     int* arr, int len, delegate* unmanaged[Cdecl]<int, int, int> cmp);
```

## 4. C#에서 네이티브 C 라이브러리 호출

### 4.1 완전한 예제: C 수학 라이브러리 래핑

다음 함수들이 있는 네이티브 C 라이브러리 `mathutils`가 있다고 가정합니다:

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

C# 래퍼:

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

// 사용:
// var a = new Vector2(3.0, 4.0);
// var b = new Vector2(1.0, 2.0);
//
// double len = MathUtils.Length(a);          // 5.0
// Vector2 sum = MathUtils.Add(a, b);        // (4.0, 6.0)
// Vector2 norm = MathUtils.Normalize(a);    // (0.6, 0.8)
// double dot = MathUtils.Dot(a, b);         // 11.0
```

### 4.2 네이티브 리소스를 위한 SafeHandle

```csharp
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

// C API:
// void* database_open(const char* path);
// int database_query(void* db, const char* sql, char* result, int resultSize);
// void database_close(void* db);

// SafeHandle은 예외가 발생하더라도 리소스 누수를 방지합니다
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

// 자동 정리와 함께 사용:
// using var db = NativeDatabase.database_open("/path/to/data.db");
// byte[] buffer = new byte[1024];
// int bytesRead = NativeDatabase.database_query(db, "SELECT * FROM users", buffer, buffer.Length);
// string result = Encoding.UTF8.GetString(buffer, 0, bytesRead);
// dispose될 때 db가 자동으로 닫힘 (예외가 발생하더라도)
```

## 5. unsafe 키워드와 안전하지 않은 컨텍스트

### 5.1 Unsafe 코드 활성화

unsafe 코드를 사용하려면 프로젝트 파일에서 활성화하고 코드 블록을 표시해야 합니다:

```xml
<!-- .csproj에서 -->
<PropertyGroup>
  <AllowUnsafeBlocks>true</AllowUnsafeBlocks>
</PropertyGroup>
```

```csharp
// unsafe는 메서드, 클래스 또는 블록에 적용할 수 있습니다
public class UnsafeDemo
{
    // 전체 메서드가 unsafe
    public unsafe void UnsafeMethod()
    {
        int x = 42;
        int* ptr = &x;
        Console.WriteLine(*ptr);  // 42
    }

    // 블록만 unsafe
    public void MixedMethod()
    {
        int value = 100;

        // 여기는 안전한 코드...

        unsafe
        {
            int* ptr = &value;
            *ptr = 200;
        }

        // value는 이제 200
        Console.WriteLine(value);
    }
}

// Unsafe 구조체
public unsafe struct UnsafeBuffer
{
    public fixed byte Data[256];  // 고정 크기 버퍼 (힙 할당 없음)
    public int Length;
}
```

### 5.2 unsafe를 사용해야 하는 경우

```csharp
// unsafe를 사용하는 유효한 이유:
// 1. 포인터가 필요한 네이티브 코드와의 상호 운용
// 2. 성능이 중요한 핫 경로 (이미지 처리, 수학 집약적 계산)
// 3. 구조체에서 고정 크기 버퍼 작업
// 4. 하드웨어 또는 메모리 매핑 I/O와의 인터페이스

// unsafe를 피해야 하는 경우:
// 1. Span<T> 또는 Memory<T>로 작업을 수행할 수 있을 때 (거의 항상)
// 2. 범위 검사만 건너뛰고 싶을 때 (대신 Span 인덱서 사용)
// 3. 코드가 성능에 중요하지 않을 때
// 4. 메모리 관리에 경험이 없을 때

// 최신 .NET에서 unsafe를 고려할 수 있는 경우의 약 95%는
// Span<T>, 안전한 컨텍스트의 stackalloc, 또는
// System.Runtime.CompilerServices의 Unsafe.As/Unsafe.Add로 안전하게 처리할 수 있습니다.
```

## 6. 포인터 타입과 연산

### 6.1 포인터 기초

```csharp
public unsafe class PointerBasics
{
    public static void Demonstrate()
    {
        // 포인터 선언 및 사용
        int value = 42;
        int* ptr = &value;     // 주소 연산자

        Console.WriteLine($"Value: {value}");           // 42
        Console.WriteLine($"Address: {(nint)ptr:X}");   // 16진수 메모리 주소
        Console.WriteLine($"Dereferenced: {*ptr}");     // 42

        // 포인터를 통해 수정
        *ptr = 100;
        Console.WriteLine($"Value after: {value}");     // 100

        // 포인터의 포인터
        int** ptrToPtr = &ptr;
        Console.WriteLine($"Double deref: {**ptrToPtr}"); // 100

        // void 포인터 (타입이 지워진)
        void* voidPtr = ptr;
        int* castBack = (int*)voidPtr;
        Console.WriteLine($"Cast back: {*castBack}");   // 100
    }
}
```

### 6.2 포인터 연산

```csharp
public unsafe class PointerArithmetic
{
    public static void Demonstrate()
    {
        // 스택 할당 배열
        int* array = stackalloc int[5];
        for (int i = 0; i < 5; i++)
        {
            array[i] = (i + 1) * 10;  // 10, 20, 30, 40, 50
        }

        // 포인터 인덱싱
        Console.WriteLine(array[0]);  // 10
        Console.WriteLine(array[4]);  // 50

        // 포인터 연산 (단계당 sizeof(int) = 4바이트씩 이동)
        int* p = array;
        Console.WriteLine(*p);       // 10
        p++;                          // 다음 int로 이동
        Console.WriteLine(*p);       // 20
        p += 2;                       // int 두 개 건너뜀
        Console.WriteLine(*p);       // 40

        // 포인터 차이
        int* start = array;
        int* end = array + 5;
        long count = end - start;     // 5 (바이트가 아닌 요소 수)
        Console.WriteLine($"Element count: {count}");

        // 포인터로 반복
        for (int* current = array; current < array + 5; current++)
        {
            Console.Write($"{*current} ");
        }
        // 출력: 10 20 30 40 50
    }
}
```

### 6.3 바이트 버퍼 작업

```csharp
public unsafe class ByteBufferOps
{
    // 특정 오프셋에서 바이트 버퍼에 정수 쓰기 (리틀 엔디안)
    public static void WriteInt32(byte* buffer, int offset, int value)
    {
        *(int*)(buffer + offset) = value;
    }

    // 특정 오프셋에서 바이트 버퍼에서 정수 읽기
    public static int ReadInt32(byte* buffer, int offset)
    {
        return *(int*)(buffer + offset);
    }

    public static void Demonstrate()
    {
        byte* buffer = stackalloc byte[16];

        // 특정 오프셋에 값 쓰기
        WriteInt32(buffer, 0, 42);
        WriteInt32(buffer, 4, 100);
        WriteInt32(buffer, 8, -1);
        WriteInt32(buffer, 12, int.MaxValue);

        // 다시 읽기
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

## 7. 고정을 위한 fixed 문

### 7.1 고정이 필요한 이유

가비지 컬렉터는 컴팩션 중에 메모리에서 객체를 이동할 수 있습니다. 네이티브 코드나 포인터 연산이 관리 객체를 참조하는 경우, GC가 이동하지 않도록 고정해야 합니다:

```csharp
public unsafe class FixedDemo
{
    public static void PinArray()
    {
        int[] numbers = { 10, 20, 30, 40, 50 };

        // GC가 이동하지 않도록 배열 고정
        fixed (int* ptr = numbers)
        {
            // 이 블록 안에서 'numbers'는 고정된 메모리 주소에 고정됨
            for (int i = 0; i < numbers.Length; i++)
            {
                Console.Write($"{*(ptr + i)} ");
            }
            Console.WriteLine();

            // 포인터를 통해 수정
            *(ptr + 2) = 999;
        }
        // 여기서 배열의 고정이 해제됨 — GC가 다시 이동할 수 있음

        Console.WriteLine(numbers[2]);  // 999
    }

    public static void PinString()
    {
        string text = "Hello, World!";

        fixed (char* ptr = text)
        {
            // 포인터를 사용하여 문자별로 문자열 순회
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
            // 포인터 접근을 사용하여 유클리드 거리 계산
            double sum = 0;
            for (int i = 0; i < 3; i++)
                sum += p[i] * p[i];
            double length = Math.Sqrt(sum);
            Console.WriteLine($"Length: {length:F4}");  // 7.0711
        }
    }
}
```

### 7.2 구조체의 고정 크기 버퍼

```csharp
// 고정 크기 버퍼는 힙 할당을 완전히 피합니다
public unsafe struct Packet
{
    public int Id;
    public fixed byte Payload[256];  // 인라인 256바이트 버퍼
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

`stackalloc`은 힙 대신 스택에 메모리를 할당합니다. 매우 빠르지만 크기가 제한됩니다:

```csharp
public unsafe class StackAllocUnsafe
{
    public static void Demonstrate()
    {
        // 스택에 100개의 정수 할당 (400바이트)
        int* buffer = stackalloc int[100];

        // 제곱으로 채우기
        for (int i = 0; i < 100; i++)
            buffer[i] = i * i;

        // 합계
        long sum = 0;
        for (int i = 0; i < 100; i++)
            sum += buffer[i];

        Console.WriteLine($"Sum of squares 0-99: {sum}");  // 328350

        // 메서드가 반환될 때 메모리가 자동으로 해제됨
        // GC 압력 없음, 힙 할당 없음
    }
}
```

### 8.2 Span을 사용한 안전한 stackalloc (C# 7.2+)

최신 C#에서는 `Span<T>`에 할당하여 `unsafe` 없이 `stackalloc`을 사용할 수 있습니다:

```csharp
public class SafeStackAlloc
{
    public static void Demonstrate()
    {
        // 안전한 stackalloc - unsafe 키워드가 필요 없음!
        Span<int> buffer = stackalloc int[100];

        for (int i = 0; i < buffer.Length; i++)
            buffer[i] = i * i;

        long sum = 0;
        foreach (int val in buffer)
            sum += val;

        Console.WriteLine($"Sum: {sum}");  // 328350

        // 범위 검사됨: buffer[100]은 IndexOutOfRangeException을 던짐
    }

    // 패턴: 대형 할당에 대해 힙으로 폴백
    public static double ComputeAverage(int count)
    {
        const int stackAllocThreshold = 256;

        Span<double> values = count <= stackAllocThreshold
            ? stackalloc double[count]    // 스택 (빠름, 작음)
            : new double[count];          // 힙 (대형에 안전)

        // 샘플 데이터로 채우기
        for (int i = 0; i < values.Length; i++)
            values[i] = i * 1.5;

        double sum = 0;
        foreach (double v in values)
            sum += v;

        return sum / values.Length;
    }
}
```

## 9. sizeof 연산자

```csharp
public unsafe class SizeOfDemo
{
    public static void Demonstrate()
    {
        // 원시 타입의 sizeof (안전한 컨텍스트에서도 동작)
        Console.WriteLine($"sizeof(byte):    {sizeof(byte)}");     // 1
        Console.WriteLine($"sizeof(short):   {sizeof(short)}");    // 2
        Console.WriteLine($"sizeof(int):     {sizeof(int)}");      // 4
        Console.WriteLine($"sizeof(long):    {sizeof(long)}");     // 8
        Console.WriteLine($"sizeof(float):   {sizeof(float)}");    // 4
        Console.WriteLine($"sizeof(double):  {sizeof(double)}");   // 8
        Console.WriteLine($"sizeof(decimal): {sizeof(decimal)}");  // 16
        Console.WriteLine($"sizeof(char):    {sizeof(char)}");     // 2 (UTF-16)
        Console.WriteLine($"sizeof(bool):    {sizeof(bool)}");     // 1
        Console.WriteLine($"sizeof(nint):    {sizeof(nint)}");     // 4 또는 8 (플랫폼 종속)

        // 구조체의 sizeof (비원시 타입에는 unsafe 필요)
        Console.WriteLine($"sizeof(Point):   {sizeof(Point)}");    // 16 (double 두 개)
        Console.WriteLine($"sizeof(Guid):    {sizeof(Guid)}");     // 16

        // 마샬링 크기에는 Marshal.SizeOf 사용 (패딩으로 인해 다를 수 있음)
        Console.WriteLine($"Marshal.SizeOf<Point>(): {Marshal.SizeOf<Point>()}");
    }

    [StructLayout(LayoutKind.Sequential)]
    struct Point { public double X; public double Y; }
}
```

## 10. 포인터의 안전한 대안으로서의 Span<T>

### 10.1 Span<T> 기초

`Span<T>`는 `unsafe` 없이 연속 메모리에 대한 타입 안전하고 범위 검사되는 뷰를 제공합니다:

```csharp
public class SpanBasics
{
    public static void Demonstrate()
    {
        // 배열 위의 Span
        int[] array = { 1, 2, 3, 4, 5, 6, 7, 8 };
        Span<int> span = array;

        // 슬라이스 (할당 없음!)
        Span<int> middle = span.Slice(2, 4);  // { 3, 4, 5, 6 }
        Console.WriteLine(string.Join(", ", middle.ToArray()));

        // span을 통해 수정 (원본 배열 수정)
        middle[0] = 99;
        Console.WriteLine(array[2]);  // 99

        // stackalloc 위의 Span
        Span<byte> stackBuffer = stackalloc byte[64];
        stackBuffer.Fill(0xFF);
        Console.WriteLine(stackBuffer[0]);  // 255

        // 비관리 메모리 위의 Span
        // IntPtr unmanagedMem = Marshal.AllocHGlobal(100);
        // Span<byte> unmanagedSpan;
        // unsafe { unmanagedSpan = new Span<byte>((void*)unmanagedMem, 100); }
        // Marshal.FreeHGlobal(unmanagedMem);
    }
}
```

### 10.2 ReadOnlySpan<char>을 사용한 문자열 처리

```csharp
public class SpanStringProcessing
{
    // 힙 할당 없이 쉼표로 구분된 문자열에서 정수를 파싱
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

    // "key=value" 문자열에서 값 추출
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

### 10.3 바이너리 데이터 처리를 위한 Span<T>

```csharp
public class BinaryProcessor
{
    // span에서 리틀 엔디안 int32 읽기
    public static int ReadInt32LittleEndian(ReadOnlySpan<byte> source)
    {
        return System.Buffers.Binary.BinaryPrimitives.ReadInt32LittleEndian(source);
    }

    // span에 리틀 엔디안 int32 쓰기
    public static void WriteInt32LittleEndian(Span<byte> dest, int value)
    {
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(dest, value);
    }

    // 간단한 바이너리 헤더 파싱: [magic:4][version:4][length:4][payload:N]
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
        // 패킷 구성
        Span<byte> packet = stackalloc byte[20];
        WriteInt32LittleEndian(packet[..4], 0x4D594150);    // 매직 넘버
        WriteInt32LittleEndian(packet[4..8], 2);            // 버전 2
        WriteInt32LittleEndian(packet[8..12], 8);           // 페이로드 길이
        packet[12..20].Fill(0xAB);                          // 페이로드 데이터

        var (version, payload) = ParsePacket(packet);
        Console.WriteLine($"Version: {version}, Payload length: {payload.Length}");
        // Version: 2, Payload length: 8
    }
}
```

### 10.4 비동기 시나리오를 위한 Memory<T>

```csharp
public class MemoryDemo
{
    // Span<T>는 ref struct입니다 — 힙에 저장하거나 async 메서드에서 사용할 수 없습니다.
    // Memory<T>는 힙 친화적인 대응물입니다.

    public static async Task ProcessDataAsync(Memory<byte> buffer)
    {
        // 데이터로 채우기
        for (int i = 0; i < buffer.Length; i++)
            buffer.Span[i] = (byte)(i % 256);

        // 비동기 I/O 시뮬레이션
        await Task.Delay(10);

        // 청크 단위로 처리
        int chunkSize = 16;
        for (int offset = 0; offset < buffer.Length; offset += chunkSize)
        {
            int remaining = Math.Min(chunkSize, buffer.Length - offset);
            Memory<byte> chunk = buffer.Slice(offset, remaining);

            // Memory<T>를 비동기 연산에 전달 가능
            await ProcessChunkAsync(chunk);
        }
    }

    private static async Task ProcessChunkAsync(Memory<byte> chunk)
    {
        // 필요할 때 기본 Span에 접근
        Span<byte> span = chunk.Span;
        int sum = 0;
        foreach (byte b in span)
            sum += b;

        await Task.Delay(1);
    }
}
```

## 11. 실전 예제: 네이티브 수학 라이브러리 호출

이 예제는 P/Invoke, 마샬링, 안전한 상호 운용을 결합하여 가상의 네이티브 통계 라이브러리를 래핑합니다:

```csharp
using System.Runtime.InteropServices;

// ---- 네이티브 API 래퍼 ----

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
    // 가상의 네이티브 함수:
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

// ---- 안전을 위한 관리 래퍼 ----

public class StatisticsService
{
    /// <summary>
    /// 네이티브 라이브러리를 사용하여 주어진 데이터의 기술 통계를 계산합니다.
    /// 네이티브 라이브러리를 사용할 수 없는 경우 관리 구현으로 폴백합니다.
    /// </summary>
    public Statistics Compute(ReadOnlySpan<double> data)
    {
        if (data.IsEmpty)
            throw new ArgumentException("Data cannot be empty");

        // P/Invoke를 위해 배열로 복사 (P/Invoke에는 고정 가능한 배열이 필요)
        double[] array = data.ToArray();

        try
        {
            return NativeStats.compute_stats(array, array.Length);
        }
        catch (DllNotFoundException)
        {
            // 관리 구현으로 폴백
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

// ---- 사용 ----

// var service = new StatisticsService();
// double[] data = { 4.5, 2.3, 7.8, 1.1, 9.0, 3.4, 6.7, 5.2 };
// Statistics stats = service.Compute(data);
// Console.WriteLine($"Mean: {stats.Mean:F2}");
// Console.WriteLine($"Median: {stats.Median:F2}");
// Console.WriteLine($"StdDev: {stats.StandardDeviation:F2}");
// Console.WriteLine($"Range: [{stats.Min:F2}, {stats.Max:F2}]");
// Console.WriteLine($"Count: {stats.Count}");
```

## 12. 연습 문제

1. **P/Invoke 래퍼**: POSIX `time` 함수(`time_t time(time_t *tloc)`)와 `ctime` 함수(`char *ctime(const time_t *timep)`)를 위한 C# 래퍼를 작성하세요. `time`을 호출하여 현재 Unix 타임스탬프를 가져온 다음, `ctime`에 전달하여 사람이 읽을 수 있는 문자열을 얻으세요. 문자열 마샬링을 적절히 처리하세요.

2. **구조체 마샬링**: 필드가 `version` (uint8), `type` (uint8), `length` (uint16), `sequence` (uint32), `checksum` (uint32)인 네트워크 패킷 헤더의 C 구조체를 정의하세요. `StructLayout`으로 일치하는 C# 구조체를 만드세요. `Pack = 1`로 총 크기가 12바이트인지 확인하세요. 헤더를 `Span<byte>`에서/으로 직렬화 및 역직렬화하는 메서드를 작성하세요.

3. **포인터 기반 이미지 처리**: 그레이스케일 픽셀 값을 나타내는 `byte[]`를 받아 각 픽셀을 반전(newValue = 255 - oldValue)하는 unsafe 메서드를 작성하세요. 포인터 연산을 사용하여 배열을 순회하세요. 그런 다음 `Span<byte>`를 사용한 동등한 안전 버전을 작성하고 가독성을 비교하세요.

4. **안전한 stackalloc 히스토그램**: `ReadOnlySpan<byte>`에서 바이트 값(0-255)의 히스토그램을 계산하는 메서드를 작성하세요. 히스토그램 빈에 대한 힙 할당을 피하기 위해 `stackalloc int[256]`(`Span<int>`에 할당)을 사용하세요. 가장 빈번한 상위 5개 값을 반환하세요.

5. **Memory<T> 파이프라인**: 비동기 데이터 처리 파이프라인을 만드세요: (a) 프로듀서가 `Memory<double>` 버퍼에 무작위 double을 작성, (b) 프로세서가 버퍼를 읽고 이동 평균을 계산, (c) 소비자가 결과를 출력합니다. 모든 단계는 `Memory<T>` 슬라이스를 통해 통신합니다. 여기서 `Span<T>`를 사용할 수 없는 이유를 설명하세요.
