# Span과 메모리

**이전**: [동시성과 병렬성](./08_Concurrency_and_Parallelism.md) | **다음**: [의존성 주입](./10_Dependency_Injection.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 스택과 힙 할당의 차이와 성능 영향 설명하기
2. `Span<T>`와 `ReadOnlySpan<T>`를 사용하여 할당 없이 연속 메모리 작업하기
3. `Span<T>` 제한 사항과 `ref struct`인 이유 이해하기
4. 힙 저장소나 비동기 메서드가 필요할 때 `Memory<T>`와 `ReadOnlyMemory<T>` 사용하기
5. `stackalloc`으로 스택에 임시 버퍼 할당하기
6. `ArrayPool<T>`로 버퍼를 효율적으로 재사용하기
7. 스팬(span)을 이용한 무할당 문자열 파싱 수행하기
8. 저수준 메모리 재해석을 위한 `MemoryMarshal` 적용하기

---

고성능 .NET 코드는 종종 메모리 할당과 가비지 컬렉션에서 병목이 발생합니다. 모든 `new byte[]`, 모든 `string.Substring()`, 모든 LINQ `.ToArray()`는 힙 압력을 생성합니다. C#은 핫 경로에서 할당을 줄이거나 제거하기 위한 도구로 `Span<T>`, `Memory<T>`, `stackalloc`, `ArrayPool<T>`를 제공합니다. 이 레슨에서는 메모리를 일급 관심사로 생각하고 빠르면서도 안전한 코드를 작성하는 방법을 배웁니다.

## 1. 스택 vs 힙 할당 복습

### 1.1 관리되는 힙

모든 참조 타입(`class`, `string`, 배열)은 관리되는 힙에 존재합니다. 가비지 컬렉터(GC)는 참조가 남아있지 않으면 이를 회수합니다. GC 압력은 할당률에 비례하여 증가합니다 — 초당 수백만 개의 작은 객체를 할당하면 눈에 띄는 일시 정지가 발생할 수 있습니다.

```csharp
// 각 호출마다 힙에 새 문자열을 할당
string GetSubstring(string source, int start, int length)
{
    return source.Substring(start, length); // 새 힙 할당
}
```

### 1.2 스택

지역 변수로 선언된 값 타입(`struct`, `int`, `double`)은 스택에 존재합니다. 스택 할당은 본질적으로 무료입니다 — 스택 포인터만 이동하면 됩니다. 스택 메모리는 메서드가 반환될 때 자동으로 회수됩니다.

```csharp
// 힙 할당 없음 — 구조체가 스택에 존재
public struct Point
{
    public double X, Y;
}

void Calculate()
{
    Point p = new Point { X = 1.0, Y = 2.0 }; // 스택 할당
    double distance = Math.Sqrt(p.X * p.X + p.Y * p.Y);
}
```

### 1.3 실제 할당 비용

```csharp
// 벤치마크: 힙 할당 vs stackalloc
using System.Diagnostics;

void HeapAllocation(int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        byte[] buffer = new byte[256]; // 힙
        buffer[0] = (byte)i;
    }
}

void StackAllocation(int iterations)
{
    for (int i = 0; i < iterations; i++)
    {
        Span<byte> buffer = stackalloc byte[256]; // 스택
        buffer[0] = (byte)i;
    }
}
```

## 2. Span&lt;T&gt;과 ReadOnlySpan&lt;T&gt;

`Span<T>`은 연속 메모리 영역에 대한 타입 안전하고 메모리 안전한 뷰입니다. 배열, 스택 할당 메모리, 네이티브 메모리를 모두 통합된 API로 가리킬 수 있습니다.

### 2.1 Span 생성

```csharp
// 배열로부터
int[] numbers = { 1, 2, 3, 4, 5 };
Span<int> span = numbers;               // 전체 배열
Span<int> slice = numbers.AsSpan(1, 3); // 요소 2, 3, 4

// 문자열로부터 (ReadOnlySpan<char>)
string text = "Hello, World!";
ReadOnlySpan<char> greeting = text.AsSpan(0, 5); // "Hello"

// stackalloc으로부터
Span<byte> buffer = stackalloc byte[128];
buffer.Fill(0xFF);

// 포인터로부터 (unsafe)
unsafe
{
    int* ptr = stackalloc int[10];
    Span<int> fromPtr = new Span<int>(ptr, 10);
}
```

### 2.2 할당 없는 슬라이싱

`Span<T>`의 핵심 기능은 제로카피 슬라이싱입니다. `Array.Copy`나 `string.Substring`과 달리, 스팬 슬라이싱은 같은 메모리의 새 뷰를 생성합니다.

```csharp
byte[] data = new byte[1024];
FillData(data);

// 제로카피 슬라이싱 — 새 배열이 생성되지 않음
Span<byte> header = data.AsSpan(0, 16);
Span<byte> payload = data.AsSpan(16, data.Length - 16);

// 중첩 슬라이싱
Span<byte> firstWord = header[..4];
Span<byte> secondWord = header[4..8];
```

### 2.3 일반적인 Span 연산

```csharp
Span<int> span = stackalloc int[5];

// 채우기
span.Fill(42);  // 모든 요소 = 42

// 지우기
span.Clear();   // 모든 요소 = 0

// 복사
Span<int> source = stackalloc int[] { 1, 2, 3 };
Span<int> dest = stackalloc int[3];
source.CopyTo(dest);

// TryCopyTo (안전, 대상이 너무 작으면 false 반환)
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

`ReadOnlySpan<T>`은 기본 데이터의 수정을 방지합니다. 문자열 슬라이싱은 `ReadOnlySpan<char>`을 반환합니다.

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

// 사용법 — 문자열 할당 없음
string sentence = "The quick brown fox jumps over the lazy dog";
int words = CountWords(sentence.AsSpan()); // 9
int firstThreeWords = CountWords(sentence.AsSpan(0, 15)); // 3
```

## 3. Span 제한 사항

`Span<T>`은 `ref struct`이므로 엄격한 규칙이 적용됩니다.

### 3.1 ref struct의 의미

`ref struct`는 스택에만 존재할 수 있습니다. 이는 메서드가 반환될 때 회수되는 스택 할당 메모리를 스팬이 가리킬 수 있기 때문에, 힙으로의 탈출을 방지하기 위해 필요합니다.

```csharp
// 다음 중 어느 것도 할 수 없음:
class MyClass
{
    // Span<int> _field;                  // 에러: 클래스의 필드가 될 수 없음
}

// Span<int> BoxedSpan = (object)span;    // 에러: ref struct를 박싱할 수 없음
// Task<Span<int>> task = ...;            // 에러: 타입 인수가 될 수 없음
// List<Span<int>> list = ...;            // 에러: 타입 인수가 될 수 없음

// 비동기 메서드에서 사용 불가:
// async Task ProcessAsync()
// {
//     Span<byte> buffer = stackalloc byte[256];
//     await SomeOperationAsync(); // 에러: Span은 await 경계를 넘을 수 없음
// }

// 캡처하는 람다나 로컬 함수에서 사용 불가:
// Action action = () => { var s = span; }; // 에러

// 할 수 있는 것:
void ProcessSync(Span<byte> data) { /* 동기 메서드에서 OK */ }
ref struct MyRefStruct { public Span<int> Data; } // OK: ref struct 안의 ref struct
```

### 3.2 Span과 Memory 간 선택

| 기능 | `Span<T>` | `Memory<T>` |
|------|-----------|-------------|
| 필드가 될 수 있음 | 아니오 (ref struct) | 예 |
| 비동기에서 사용 가능 | 아니오 | 예 |
| 제네릭 인수 가능 | 아니오 | 예 |
| 스택 메모리 가리킴 | 예 | 아니오 |
| 성능 | 가장 빠름 | 약간 느림 |

## 4. Memory&lt;T&gt;와 ReadOnlyMemory&lt;T&gt;

`Memory<T>`는 `Span<T>`의 힙 안전한 대응물입니다. 필드에 저장하고, 비동기 메서드에서 사용하고, 제네릭 API에 전달할 수 있습니다.

### 4.1 기본 사용법

```csharp
public class DataProcessor
{
    private Memory<byte> _buffer;  // OK: Memory<T>는 필드가 될 수 있음

    public DataProcessor(byte[] data)
    {
        _buffer = data.AsMemory();
    }

    public async Task ProcessAsync()
    {
        Memory<byte> header = _buffer[..16];
        Memory<byte> payload = _buffer[16..];

        // Memory<T>는 await 경계를 넘어 작동
        await ProcessHeaderAsync(header);
        await ProcessPayloadAsync(payload);
    }

    private async Task ProcessHeaderAsync(Memory<byte> header)
    {
        // Span이 필요하면 동기 범위 내에서 .Span을 호출
        ParseHeader(header.Span);
        await Task.CompletedTask;
    }

    private void ParseHeader(Span<byte> header)
    {
        // 빠른 무할당 파싱
        int version = header[0];
        int flags = header[1];
    }

    private async Task ProcessPayloadAsync(Memory<byte> payload)
    {
        // 스트림에 쓰기
        await using var stream = new MemoryStream();
        await stream.WriteAsync(payload);
    }
}
```

### 4.2 IMemoryOwner&lt;T&gt;와 MemoryPool

```csharp
using System.Buffers;

// 풀에서 메모리 대여 (ArrayPool과 유사하지만 Memory<T>를 반환)
using IMemoryOwner<byte> owner = MemoryPool<byte>.Shared.Rent(4096);
Memory<byte> memory = owner.Memory[..4096]; // 필요한 정확한 크기로 슬라이스

// 메모리 처리
FillData(memory.Span);
await SendAsync(memory);
// owner가 dispose될 때 메모리가 풀에 반환됨
```

## 5. stackalloc 키워드

`stackalloc`은 스택에 메모리를 할당하여 GC 압력을 완전히 피합니다. 메서드가 종료될 때 메모리가 자동으로 해제됩니다.

### 5.1 기본 stackalloc

```csharp
public bool IsValidUtf8(ReadOnlySpan<byte> data)
{
    // 스택 할당 룩업 테이블 — 힙 할당 없음
    Span<bool> continuationValid = stackalloc bool[256];
    for (int i = 0x80; i <= 0xBF; i++)
        continuationValid[i] = true;

    int i2 = 0;
    while (i2 < data.Length)
    {
        byte b = data[i2];
        if (b < 0x80) { i2++; continue; }
        // ... 룩업 테이블을 사용한 검증 로직
        i2++;
    }
    return true;
}
```

### 5.2 조건부 stackalloc

가변 크기 버퍼의 경우 작은 크기에는 `stackalloc`을 사용하고 큰 크기에는 `ArrayPool`로 폴백합니다.

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

### 5.3 이니셜라이저가 있는 stackalloc

```csharp
// C# 8+: 이니셜라이저가 있는 stackalloc
ReadOnlySpan<int> primes = stackalloc int[] { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29 };

// 스팬을 이용한 패턴 매칭
ReadOnlySpan<byte> magic = stackalloc byte[] { 0x89, 0x50, 0x4E, 0x47 }; // PNG 헤더
bool isPng = fileHeader.StartsWith(magic);
```

## 6. ref struct와 Span이 ref struct인 이유

### 6.1 자신만의 ref struct 정의

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

    // foreach 구문 활성화
    public TokenEnumerator GetEnumerator() => this;
}
```

```csharp
// 사용법 — 무할당 문자열 분할
ReadOnlySpan<char> csv = "Alice,30,Engineer,Seattle";
var enumerator = new TokenEnumerator(csv, ',');

while (enumerator.MoveNext())
{
    // 각 토큰은 ReadOnlySpan<char> — 문자열 할당 없음
    Console.WriteLine(enumerator.Current.ToString());
}
```

### 6.2 ref struct의 Dispose 패턴

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

    // ref struct는 IDisposable을 구현할 수 없지만 Dispose 메서드를 가질 수 있음
    public void Dispose()
    {
        if (_rentedArray is not null)
        {
            ArrayPool<byte>.Shared.Return(_rentedArray);
            _rentedArray = null;
        }
    }
}

// 패턴 기반 using 문으로 사용
using var writer = new SpanWriter(1024);
writer.WriteByte(0x01);
writer.WriteByte(0x02);
ReadOnlySpan<byte> data = writer.Written;
```

## 7. 버퍼 재사용을 위한 ArrayPool&lt;T&gt;

`ArrayPool<T>`는 재사용 가능한 배열 풀을 제공하여 임시 버퍼의 반복 할당을 제거합니다.

### 7.1 기본 사용법

```csharp
using System.Buffers;

public byte[] CompressData(ReadOnlySpan<byte> input)
{
    // 버퍼 대여 — 요청한 것보다 클 수 있음
    byte[] buffer = ArrayPool<byte>.Shared.Rent(input.Length * 2);

    try
    {
        int compressedLength = Compress(input, buffer);
        // 정확한 결과를 올바른 크기의 배열로 복사
        return buffer.AsSpan(0, compressedLength).ToArray();
    }
    finally
    {
        // 항상 대여한 배열을 반환 — 민감한 데이터는 clearArray: true 전달
        ArrayPool<byte>.Shared.Return(buffer, clearArray: false);
    }
}
```

### 7.2 커스텀 풀 구성

```csharp
// 특정 크기 제한이 있는 커스텀 풀
ArrayPool<byte> customPool = ArrayPool<byte>.Create(
    maxArrayLength: 1024 * 1024,  // 최대 1 MB 배열
    maxArraysPerBucket: 50        // 크기 버킷당 최대 50개 배열 유지
);

byte[] buf = customPool.Rent(8192);
try
{
    // 버퍼 사용
}
finally
{
    customPool.Return(buf);
}
```

### 7.3 ArrayPool을 이용한 MemoryStream 대안

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

## 8. 스팬을 이용한 문자열 처리 (무할당 파싱)

문자열 조작은 .NET에서 가장 큰 할당 원인 중 하나입니다. 스팬을 사용하면 중간 `string` 객체를 생성하지 않고 문자열을 파싱할 수 있습니다.

### 8.1 할당 없는 정수 파싱

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

// 사용법 — 문자열 할당 없음
string data = "47.6062, -122.3321";
if (TryParseCoordinate(data, out double lat, out double lon))
{
    Console.WriteLine($"위도: {lat}, 경도: {lon}");
}
```

### 8.2 string.Split 없는 분할

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

// "1, 2, 3, 4, 5" 파싱에 할당 없음
int total = SplitAndSum("1, 2, 3, 4, 5"); // 15
```

### 8.3 Span을 이용한 문자열 빌드 (ISpanFormattable)

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

## 9. MemoryMarshal 유틸리티

`MemoryMarshal`은 메모리 재해석, 구조화된 데이터 읽기/쓰기, 스팬 타입 간 변환을 위한 저수준 작업을 제공합니다.

### 9.1 타입 간 캐스트

```csharp
using System.Runtime.InteropServices;

byte[] rawBytes = { 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00 };

// 바이트를 int로 재해석 (같은 메모리, 복사 없음)
ReadOnlySpan<int> ints = MemoryMarshal.Cast<byte, int>(rawBytes);
Console.WriteLine(ints[0]); // 1
Console.WriteLine(ints[1]); // 2
```

### 9.2 구조체 읽기와 쓰기

```csharp
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct PacketHeader
{
    public byte Version;
    public byte Type;
    public ushort Length;
    public uint SequenceNumber;
}

// 원시 바이트에서 구조체 읽기
byte[] networkData = ReceivePacket();
PacketHeader header = MemoryMarshal.Read<PacketHeader>(networkData);
Console.WriteLine($"버전: {header.Version}, 시퀀스: {header.SequenceNumber}");

// 구조체를 바이트로 쓰기
var outHeader = new PacketHeader { Version = 1, Type = 2, Length = 64, SequenceNumber = 42 };
byte[] output = new byte[Marshal.SizeOf<PacketHeader>()];
MemoryMarshal.Write(output, in outHeader);
```

### 9.3 안전하지 않은 빠른 접근을 위한 GetReference

```csharp
// 첫 번째 요소에 대한 직접 참조 획득 (범위 검사 회피)
Span<int> span = new int[] { 10, 20, 30, 40, 50 };
ref int first = ref MemoryMarshal.GetReference(span);

// 범위 검사 없는 인덱스 접근을 위해 Unsafe.Add 사용
ref int third = ref Unsafe.Add(ref first, 2);
Console.WriteLine(third); // 30
```

## 10. 실전 예제: 고성능 CSV 파서

이 예제는 `Span<T>`, `stackalloc`, `ArrayPool<T>`, 무할당 파싱을 결합하여 빠른 CSV 파서를 구축합니다.

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

        // 캐리지 리턴 제거
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
// 전체 처리 파이프라인
public class CsvProcessor
{
    public record SalesRecord(string Product, int Quantity, double Price);

    public static List<SalesRecord> ParseSalesData(string csvContent)
    {
        var records = new List<SalesRecord>();
        var reader = new CsvReader(csvContent);

        // 헤더 줄 건너뛰기
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
// 사용법과 벤치마킹
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

// 전통적인 string.Split 접근법과 비교:
// string.Split은 줄당 N개의 string 객체를 할당
// Span 기반 파서는 최종 SalesRecord에 대해서만 문자열을 할당
```

```csharp
// 대용량 CSV를 위한 ArrayPool 기반 파일 읽기
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

        // 바이트를 문자로 변환
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

## 11. 연습 문제

1. **무할당 숫자 포매터**: `FormatWithCommas(long value, Span<char> buffer, out int charsWritten)` 메서드를 작성하여 천 단위 구분자로 숫자를 포맷하세요 (예: `1234567`이 `1,234,567`이 됨). 스택 할당 메모리만 사용하고 문자열이나 배열을 할당하지 마세요. 음수도 처리하세요.

2. **바이너리 프로토콜 파서**: 헤더(4바이트 매직, 2바이트 버전, 4바이트 페이로드 길이)와 가변 길이 페이로드가 있는 간단한 바이너리 프로토콜을 정의하세요. `Span<T>`와 `MemoryMarshal.Read<T>`를 사용하여 힙 할당 없이 헤더 필드와 페이로드를 추출하는 파서를 작성하세요. 매직 바이트와 최대 페이로드 크기에 대한 유효성 검사를 포함하세요.

3. **Span 기반 문자열 빌더**: `stackalloc` 기반(오버플로우 시 `ArrayPool` 폴백)의 `ref struct StackStringBuilder`를 구현하세요. `Append(ReadOnlySpan<char>)`, `Append(int)`, `Append(char)`, `ToString()`을 지원합니다. 빌더는 `ToString()`이 호출될 때까지 할당하지 않아야 합니다. 스택 메모리만 사용하여 500자 문자열을 빌드하는 테스트를 작성하세요.

4. **풀 버퍼 매니저**: 대여한 버퍼의 자동 추적 기능이 있는 `ArrayPool<byte>`를 래핑하는 `BufferManager` 클래스를 만드세요. 매니저가 dispose될 때 모든 미반환 버퍼가 반환되도록 `IDisposable`을 구현하세요. `Span<byte>` 접근과 dispose 시 자동 반환을 제공하는 `BufferHandle` ref struct를 추가하세요.

5. **HTTP 헤더 파서**: 원시 HTTP 응답 바이트를 포함하는 `ReadOnlySpan<byte>`를 받아 상태 코드, content-type, content-length, 각 헤더를 키-값 `ReadOnlySpan<char>` 쌍으로 추출하는 무할당 HTTP 헤더 파서를 작성하세요. 가능한 곳에서 `Span` 슬라이싱과 `Utf8Parser.TryParse`를 사용하세요. 샘플 HTTP 응답에 대해 검증하세요.
