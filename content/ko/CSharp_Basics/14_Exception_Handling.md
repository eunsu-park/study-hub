# 예외 처리 (Exception Handling)

**이전**: [제네릭](./13_Generics.md) | **다음**: [파일 I/O](./15_File_IO.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `try`, `catch`, `finally` 블록을 사용하여 예외를 처리할 수 있다
2. 특정 예외부터 일반 예외까지 여러 예외 타입을 잡을 수 있다
3. .NET 예외 계층 구조와 일반적인 예외 타입을 이해할 수 있다
4. 예외를 올바르게 던지고(throw) 다시 던질(rethrow) 수 있다
5. 사용자 정의 예외 클래스를 만들 수 있다
6. `when` 절을 사용한 예외 필터를 사용할 수 있다
7. `using` 문과 `IDisposable`로 리소스를 관리할 수 있다
8. 견고한 오류 처리를 위한 모범 사례를 적용할 수 있다

---

모든 비자명(non-trivial) 프로그램은 오류를 만납니다: 존재하지 않는 파일, 시간 초과되는 네트워크 연결, 잘못된 사용자 입력, 예상치 못한 null 참조 등. C#은 `try`, `catch`, `finally`, `throw`를 중심으로 구축된 구조적 예외 처리 모델을 사용합니다. 올바르게 사용하면 예외는 오류 처리 로직을 정상 흐름에서 분리하고, 의미 있는 진단을 생성하며, 애플리케이션이 비정상적으로 충돌하는 것을 방지합니다. 이 레슨에서는 전체 예외 처리 도구 모음과 전문 C# 개발자가 따르는 모범 사례를 다룹니다.

## 1. `try`, `catch`, `finally` 블록

`try-catch-finally` 구조는 C#에서 예외 처리의 기반입니다.

### 1.1 기본 try-catch

```csharp
try
{
    Console.Write("Enter a number: ");
    string input = Console.ReadLine();
    int number = int.Parse(input);  // FormatException 발생 가능
    int result = 100 / number;      // DivideByZeroException 발생 가능
    Console.WriteLine($"100 / {number} = {result}");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
```

### 1.2 `finally` 블록

`finally` 블록은 예외 발생 여부에 관계없이 항상 실행됩니다. 일반적으로 정리 작업에 사용됩니다.

```csharp
StreamReader reader = null;
try
{
    reader = new StreamReader("data.txt");
    string content = reader.ReadToEnd();
    Console.WriteLine(content);
}
catch (FileNotFoundException ex)
{
    Console.WriteLine($"File not found: {ex.FileName}");
}
catch (IOException ex)
{
    Console.WriteLine($"IO error: {ex.Message}");
}
finally
{
    // 예외가 던져져도 항상 실행됨
    if (reader != null)
    {
        reader.Close();
        Console.WriteLine("Reader closed.");
    }
}
```

### 1.3 catch 없이 try-finally

정리를 보장하면서 예외가 전파되도록 하려면 `catch` 블록 없이 `try-finally`를 사용할 수 있습니다.

```csharp
public void ProcessData()
{
    var connection = OpenDatabaseConnection();
    try
    {
        // 예외를 던질 수 있는 작업
        connection.Execute("INSERT INTO logs VALUES ('started')");
        // ... 추가 작업 ...
    }
    finally
    {
        // 예외가 전파되더라도 항상 연결 닫기
        connection.Close();
    }
    // 예외가 발생했다면 finally 실행 후 전파됨
}
```

### 1.4 중첩 try-catch

```csharp
try
{
    Console.WriteLine("Outer try");

    try
    {
        Console.WriteLine("Inner try");
        throw new InvalidOperationException("Something went wrong");
    }
    catch (InvalidOperationException ex)
    {
        Console.WriteLine($"Inner catch: {ex.Message}");
        // 처리하거나 다시 던짐
    }
    finally
    {
        Console.WriteLine("Inner finally");
    }

    Console.WriteLine("After inner try-catch");
}
catch (Exception ex)
{
    Console.WriteLine($"Outer catch: {ex.Message}");
}
finally
{
    Console.WriteLine("Outer finally");
}

// 출력:
// Outer try
// Inner try
// Inner catch: Something went wrong
// Inner finally
// After inner try-catch
// Outer finally
```

## 2. 다중 Catch 블록

여러 예외 타입이 가능한 경우, 가장 구체적인 것부터 가장 일반적인 것 순으로 catch 블록을 나열합니다. 런타임은 첫 번째로 일치하는 catch 블록을 사용합니다.

### 2.1 Catch 블록 순서

```csharp
try
{
    Console.Write("Enter an index (0-4): ");
    int index = int.Parse(Console.ReadLine());

    int[] numbers = { 10, 20, 30, 40, 50 };
    int value = numbers[index];
    int result = 1000 / value;

    Console.WriteLine($"Result: {result}");
}
catch (FormatException ex)
{
    Console.WriteLine($"Invalid format: {ex.Message}");
}
catch (IndexOutOfRangeException ex)
{
    Console.WriteLine($"Index out of range: {ex.Message}");
}
catch (DivideByZeroException ex)
{
    Console.WriteLine($"Division by zero: {ex.Message}");
}
catch (Exception ex)
{
    // 일반 catch — 항상 마지막에 배치
    Console.WriteLine($"Unexpected error: {ex.Message}");
}
```

### 2.2 하나의 블록에서 여러 타입 잡기 (C# 6+)

여러 예외 타입에 대해 동일한 처리 로직을 원하면 `when`을 사용하거나 개별적으로 처리합니다. 예외 필터 이전에는 `Exception`을 잡고 타입을 확인했습니다.

```csharp
try
{
    // 어떤 작업
}
catch (Exception ex) when (ex is FormatException || ex is OverflowException)
{
    Console.WriteLine($"Input error: {ex.Message}");
}
catch (Exception ex)
{
    Console.WriteLine($"Other error: {ex.Message}");
}
```

### 2.3 예외 상세 정보 접근

```csharp
try
{
    throw new ArgumentNullException("username", "Username cannot be null");
}
catch (ArgumentNullException ex)
{
    Console.WriteLine($"Message: {ex.Message}");
    Console.WriteLine($"Parameter: {ex.ParamName}");
    Console.WriteLine($"Type: {ex.GetType().Name}");
    Console.WriteLine($"Stack Trace:\n{ex.StackTrace}");
    Console.WriteLine($"Source: {ex.Source}");

    if (ex.InnerException != null)
    {
        Console.WriteLine($"Inner: {ex.InnerException.Message}");
    }
}
```

## 3. 예외 계층 구조

.NET의 모든 예외는 `System.Exception`에서 파생됩니다. 계층 구조를 이해하면 올바른 타입을 잡는 데 도움이 됩니다.

### 3.1 계층 구조

```
System.Exception
├── System.SystemException
│   ├── System.NullReferenceException
│   ├── System.IndexOutOfRangeException
│   ├── System.InvalidOperationException
│   ├── System.ArgumentException
│   │   ├── System.ArgumentNullException
│   │   └── System.ArgumentOutOfRangeException
│   ├── System.ArithmeticException
│   │   ├── System.DivideByZeroException
│   │   └── System.OverflowException
│   ├── System.FormatException
│   ├── System.NotSupportedException
│   ├── System.NotImplementedException
│   ├── System.InvalidCastException
│   ├── System.IO.IOException
│   │   ├── System.IO.FileNotFoundException
│   │   ├── System.IO.DirectoryNotFoundException
│   │   └── System.IO.EndOfStreamException
│   ├── System.OutOfMemoryException
│   └── System.StackOverflowException
└── System.ApplicationException (사용 중단, 사용하지 말 것)
```

### 3.2 일반적인 예외 타입

```csharp
// ArgumentException: 잘못된 인수
public void SetAge(int age)
{
    if (age < 0 || age > 150)
        throw new ArgumentOutOfRangeException(nameof(age), age,
            "Age must be between 0 and 150.");
}

// ArgumentNullException: 허용되지 않는 곳에 null
public void Greet(string name)
{
    if (name == null)
        throw new ArgumentNullException(nameof(name));
    Console.WriteLine($"Hello, {name}!");
}

// InvalidOperationException: 현재 상태에서 유효하지 않은 연산
public class OrderProcessor
{
    private bool _isInitialized = false;

    public void Process()
    {
        if (!_isInitialized)
            throw new InvalidOperationException(
                "Processor must be initialized before processing.");
    }
}

// NotSupportedException: 지원되지 않는 메서드
public class ReadOnlyList<T>
{
    public void Add(T item)
    {
        throw new NotSupportedException("This collection is read-only.");
    }
}

// KeyNotFoundException
public string GetConfig(Dictionary<string, string> config, string key)
{
    if (!config.ContainsKey(key))
        throw new KeyNotFoundException($"Configuration key '{key}' not found.");
    return config[key];
}
```

### 3.3 NullReferenceException

C# 프로그램에서 가장 흔한 예외입니다. null 참조의 멤버를 사용하려 할 때 발생합니다.

```csharp
string text = null;

try
{
    int len = text.Length;  // NullReferenceException
}
catch (NullReferenceException)
{
    Console.WriteLine("Object reference was null!");
}

// 예방: null 검사와 null 조건 연산자
int? safeLen = text?.Length;   // null, 예외 없음
int length = text?.Length ?? 0; // 0
```

## 4. 예외 던지기

### 4.1 `throw` 문

```csharp
public class BankAccount
{
    public decimal Balance { get; private set; }

    public BankAccount(decimal initialBalance)
    {
        if (initialBalance < 0)
            throw new ArgumentException(
                "Initial balance cannot be negative.",
                nameof(initialBalance));
        Balance = initialBalance;
    }

    public void Withdraw(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentOutOfRangeException(
                nameof(amount), amount, "Amount must be positive.");

        if (amount > Balance)
            throw new InvalidOperationException(
                $"Insufficient funds. Balance: {Balance:C}, Requested: {amount:C}");

        Balance -= amount;
    }
}
```

### 4.2 `throw` vs `throw ex` — 스택 트레이스 보존

이것은 C#에서 가장 중요한 예외 처리 규칙 중 하나입니다.

```csharp
// 잘못됨: throw ex — 스택 트레이스를 리셋
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    LogError(ex);
    throw ex;  // 나쁨: 스택 트레이스가 이 줄로 리셋됨
}

// 올바름: throw — 원본 스택 트레이스 보존
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    LogError(ex);
    throw;  // 좋음: 원본 스택 트레이스가 보존됨
}

// 역시 올바름: InnerException으로 새 예외에 감싸기
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    throw new ApplicationException("Operation failed", ex);
    // 원본 예외가 InnerException으로 보존됨
}
```

### 4.3 조건부 던지기

```csharp
public class Validator
{
    public static void EnsureNotNull<T>(T value, string paramName) where T : class
    {
        if (value == null)
            throw new ArgumentNullException(paramName);
    }

    public static void EnsureInRange(int value, int min, int max, string paramName)
    {
        if (value < min || value > max)
            throw new ArgumentOutOfRangeException(
                paramName, value, $"Value must be between {min} and {max}.");
    }

    public static void EnsureNotEmpty(string value, string paramName)
    {
        if (string.IsNullOrWhiteSpace(value))
            throw new ArgumentException("Value cannot be empty or whitespace.", paramName);
    }
}

// 사용법
public void RegisterUser(string username, string email, int age)
{
    Validator.EnsureNotEmpty(username, nameof(username));
    Validator.EnsureNotEmpty(email, nameof(email));
    Validator.EnsureInRange(age, 13, 120, nameof(age));

    // 유효한 데이터로 진행
    Console.WriteLine($"Registered: {username}, {email}, age {age}");
}
```

### 4.4 Throw 식 (C# 7+)

```csharp
public class Config
{
    private readonly string _connectionString;

    // 생성자에서 throw 식
    public Config(string connectionString)
    {
        _connectionString = connectionString
            ?? throw new ArgumentNullException(nameof(connectionString));
    }

    // null 병합에서의 throw 식
    public string GetSetting(Dictionary<string, string> settings, string key)
    {
        return settings.TryGetValue(key, out string value)
            ? value
            : throw new KeyNotFoundException($"Setting '{key}' not found.");
    }

    // 조건식에서의 throw 식
    public string Name { get; set; }
    public string DisplayName => Name ?? throw new InvalidOperationException("Name not set.");
}
```

## 5. 사용자 정의 예외 클래스

사용자 정의 예외를 만들면 애플리케이션 도메인에 특화된 의미 있는 오류 타입을 제공합니다.

### 5.1 기본 사용자 정의 예외

```csharp
public class InsufficientFundsException : Exception
{
    public decimal Balance { get; }
    public decimal RequestedAmount { get; }

    public InsufficientFundsException()
        : base("Insufficient funds for this operation.")
    {
    }

    public InsufficientFundsException(string message)
        : base(message)
    {
    }

    public InsufficientFundsException(string message, Exception innerException)
        : base(message, innerException)
    {
    }

    public InsufficientFundsException(decimal balance, decimal requestedAmount)
        : base($"Insufficient funds. Balance: {balance:C}, Requested: {requestedAmount:C}")
    {
        Balance = balance;
        RequestedAmount = requestedAmount;
    }
}
```

### 5.2 사용자 정의 예외 사용

```csharp
public class BankAccount
{
    public string AccountNumber { get; }
    public decimal Balance { get; private set; }

    public BankAccount(string accountNumber, decimal initialBalance)
    {
        AccountNumber = accountNumber;
        Balance = initialBalance;
    }

    public void Withdraw(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentOutOfRangeException(nameof(amount),
                "Withdrawal amount must be positive.");

        if (amount > Balance)
            throw new InsufficientFundsException(Balance, amount);

        Balance -= amount;
    }
}
```

```csharp
BankAccount account = new BankAccount("ACC-001", 500m);

try
{
    account.Withdraw(750m);
}
catch (InsufficientFundsException ex)
{
    Console.WriteLine(ex.Message);
    Console.WriteLine($"  Current balance: {ex.Balance:C}");
    Console.WriteLine($"  Requested: {ex.RequestedAmount:C}");
    Console.WriteLine($"  Shortfall: {ex.RequestedAmount - ex.Balance:C}");
}
// 출력:
// Insufficient funds. Balance: $500.00, Requested: $750.00
//   Current balance: $500.00
//   Requested: $750.00
//   Shortfall: $250.00
```

### 5.3 도메인에 대한 예외 계층 구조

```csharp
// 애플리케이션 기본 예외
public class AppException : Exception
{
    public string ErrorCode { get; }

    public AppException(string message, string errorCode = "UNKNOWN")
        : base(message)
    {
        ErrorCode = errorCode;
    }

    public AppException(string message, Exception inner, string errorCode = "UNKNOWN")
        : base(message, inner)
    {
        ErrorCode = errorCode;
    }
}

// 특정 도메인 예외
public class ValidationException : AppException
{
    public Dictionary<string, string[]> Errors { get; }

    public ValidationException(Dictionary<string, string[]> errors)
        : base("One or more validation errors occurred.", "VALIDATION_ERROR")
    {
        Errors = errors;
    }
}

public class EntityNotFoundException : AppException
{
    public string EntityType { get; }
    public object EntityId { get; }

    public EntityNotFoundException(string entityType, object id)
        : base($"{entityType} with ID '{id}' was not found.", "NOT_FOUND")
    {
        EntityType = entityType;
        EntityId = id;
    }
}

public class UnauthorizedException : AppException
{
    public UnauthorizedException(string action)
        : base($"Not authorized to perform: {action}", "UNAUTHORIZED")
    {
    }
}
```

```csharp
// 서비스에서의 사용
public class UserService
{
    private readonly Dictionary<int, string> _users = new Dictionary<int, string>
    {
        [1] = "Alice",
        [2] = "Bob"
    };

    public string GetUser(int id)
    {
        if (!_users.TryGetValue(id, out string name))
            throw new EntityNotFoundException("User", id);
        return name;
    }

    public void UpdateUser(int id, string newName, bool isAdmin)
    {
        if (!isAdmin)
            throw new UnauthorizedException("update user");

        if (string.IsNullOrWhiteSpace(newName))
        {
            var errors = new Dictionary<string, string[]>
            {
                ["name"] = new[] { "Name cannot be empty." }
            };
            throw new ValidationException(errors);
        }

        if (!_users.ContainsKey(id))
            throw new EntityNotFoundException("User", id);

        _users[id] = newName;
    }
}
```

## 6. 예외 필터 (`when` 절)

C# 6에서 도입된 예외 필터는 잡고 다시 던지지 않고도 catch 블록에 조건을 추가할 수 있게 합니다.

### 6.1 기본 예외 필터

```csharp
try
{
    MakeHttpRequest("https://api.example.com/data");
}
catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.NotFound)
{
    Console.WriteLine("Resource not found (404).");
}
catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.Unauthorized)
{
    Console.WriteLine("Authentication required (401).");
}
catch (HttpRequestException ex) when (ex.StatusCode == System.Net.HttpStatusCode.ServiceUnavailable)
{
    Console.WriteLine("Service temporarily unavailable (503). Retry later.");
}
catch (HttpRequestException ex)
{
    Console.WriteLine($"HTTP error: {ex.StatusCode} - {ex.Message}");
}
```

### 6.2 메시지나 속성으로 필터링

```csharp
try
{
    ProcessFile("important.dat");
}
catch (IOException ex) when (ex.Message.Contains("being used by another process"))
{
    Console.WriteLine("File is locked. Please close it and try again.");
}
catch (IOException ex) when (ex.Message.Contains("disk is full"))
{
    Console.WriteLine("Disk is full. Free up space and try again.");
}
catch (IOException ex)
{
    Console.WriteLine($"I/O error: {ex.Message}");
}
```

### 6.3 잡지 않고 로깅하기

강력한 기법: `when`을 사용하여 실제로 예외를 잡지 않고 로깅합니다.

```csharp
try
{
    RiskyOperation();
}
catch (Exception ex) when (LogException(ex))
{
    // LogException이 false를 반환하므로 이 블록은 실행되지 않음
}
catch (SpecificException ex)
{
    // 이 핸들러가 여전히 예외를 받음
    HandleSpecific(ex);
}

static bool LogException(Exception ex)
{
    Console.WriteLine($"[LOG] Exception occurred: {ex.GetType().Name}: {ex.Message}");
    return false;  // 절대 잡지 않음 — 로깅만 함
}
```

### 6.4 환경 기반 필터링

```csharp
bool isDevelopment = Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT") == "Development";

try
{
    ProcessOrder(orderId);
}
catch (Exception ex) when (isDevelopment)
{
    // 개발 환경에서는 전체 상세 정보 표시
    Console.WriteLine($"DEV ERROR: {ex}");
}
catch (Exception ex) when (!isDevelopment)
{
    // 프로덕션에서는 사용자 친화적 메시지 표시
    Console.WriteLine("An error occurred. Please contact support.");
    LogToFile(ex);
}
```

## 7. `using` 문과 `IDisposable`

`using` 문은 예외가 발생하더라도 `IDisposable` 객체가 올바르게 해제되도록 보장합니다. `try-finally` 블록의 문법적 설탕(syntactic sugar)입니다.

### 7.1 `IDisposable` 인터페이스

```csharp
public interface IDisposable
{
    void Dispose();
}
```

많은 .NET 클래스가 `IDisposable`을 구현합니다: 파일 스트림, 데이터베이스 연결, 네트워크 소켓, 타이머 등. 작업이 끝나면 항상 이러한 리소스를 해제하세요.

### 7.2 `using` 문

```csharp
// using 없이: 수동 try-finally
StreamReader reader = null;
try
{
    reader = new StreamReader("data.txt");
    string content = reader.ReadToEnd();
    Console.WriteLine(content);
}
finally
{
    reader?.Dispose();
}

// using 문 사용: 더 깔끔하고 안전
using (StreamReader reader2 = new StreamReader("data.txt"))
{
    string content = reader2.ReadToEnd();
    Console.WriteLine(content);
}
// reader2.Dispose()가 여기서 자동 호출됨
```

### 7.3 `using` 선언 (C# 8+)

```csharp
// using 선언: 감싸는 스코프의 끝에서 해제
public void ProcessFile(string path)
{
    using var reader = new StreamReader(path);  // 중괄호 불필요
    string line;
    while ((line = reader.ReadLine()) != null)
    {
        Console.WriteLine(line);
    }
    // reader는 여기, 메서드 끝에서 해제됨
}
```

### 7.4 다중 `using` 문

```csharp
// 중첩 using (전통적)
using (var input = new StreamReader("input.txt"))
using (var output = new StreamWriter("output.txt"))
{
    string line;
    while ((line = input.ReadLine()) != null)
    {
        output.WriteLine(line.ToUpper());
    }
}

// using 선언 (C# 8+) — 더 깔끔
public void CopyUpperCase(string inputPath, string outputPath)
{
    using var input = new StreamReader(inputPath);
    using var output = new StreamWriter(outputPath);

    string line;
    while ((line = input.ReadLine()) != null)
    {
        output.WriteLine(line.ToUpper());
    }
}
// 둘 다 여기서 해제됨
```

### 7.5 IDisposable 구현

```csharp
public class DatabaseConnection : IDisposable
{
    private bool _disposed = false;
    private bool _isOpen = false;

    public void Open()
    {
        _isOpen = true;
        Console.WriteLine("Connection opened.");
    }

    public void Execute(string query)
    {
        if (_disposed)
            throw new ObjectDisposedException(nameof(DatabaseConnection));
        if (!_isOpen)
            throw new InvalidOperationException("Connection is not open.");

        Console.WriteLine($"Executing: {query}");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            if (_isOpen)
            {
                _isOpen = false;
                Console.WriteLine("Connection closed.");
            }
            _disposed = true;
        }
    }
}
```

```csharp
using (var conn = new DatabaseConnection())
{
    conn.Open();
    conn.Execute("SELECT * FROM users");
    conn.Execute("UPDATE users SET active = 1 WHERE id = 5");
}
// 출력:
// Connection opened.
// Executing: SELECT * FROM users
// Executing: UPDATE users SET active = 1 WHERE id = 5
// Connection closed.
```

## 8. 리소스 정리를 위한 `try-finally`

`using` 문을 사용할 수 없는 경우(예: 리소스가 `IDisposable`이 아니거나 더 많은 제어가 필요한 경우)에도 `try-finally`는 정리를 보장합니다.

### 8.1 잠금 패턴

```csharp
public class ThreadSafeCounter
{
    private int _count = 0;
    private readonly object _lock = new object();

    public void Increment()
    {
        bool lockTaken = false;
        try
        {
            Monitor.Enter(_lock, ref lockTaken);
            _count++;
        }
        finally
        {
            if (lockTaken)
                Monitor.Exit(_lock);
        }
    }

    public int Count => _count;
}
```

### 8.2 임시 상태 변경

```csharp
public class ConsoleColorScope : IDisposable
{
    private readonly ConsoleColor _originalForeground;
    private readonly ConsoleColor _originalBackground;

    public ConsoleColorScope(ConsoleColor foreground,
        ConsoleColor? background = null)
    {
        _originalForeground = Console.ForegroundColor;
        _originalBackground = Console.BackgroundColor;

        Console.ForegroundColor = foreground;
        if (background.HasValue)
            Console.BackgroundColor = background.Value;
    }

    public void Dispose()
    {
        Console.ForegroundColor = _originalForeground;
        Console.BackgroundColor = _originalBackground;
    }
}
```

```csharp
Console.WriteLine("Normal text.");

using (new ConsoleColorScope(ConsoleColor.Red))
{
    Console.WriteLine("This is red!");
    Console.WriteLine("Still red.");
}

Console.WriteLine("Back to normal.");

using (new ConsoleColorScope(ConsoleColor.Green))
{
    Console.WriteLine("This is green!");
}
```

### 8.3 스톱워치 패턴

```csharp
public class TimedOperation : IDisposable
{
    private readonly string _operationName;
    private readonly System.Diagnostics.Stopwatch _stopwatch;

    public TimedOperation(string operationName)
    {
        _operationName = operationName;
        _stopwatch = System.Diagnostics.Stopwatch.StartNew();
        Console.WriteLine($"[START] {_operationName}");
    }

    public void Dispose()
    {
        _stopwatch.Stop();
        Console.WriteLine($"[END] {_operationName} completed in {_stopwatch.ElapsedMilliseconds}ms");
    }
}
```

```csharp
using (new TimedOperation("Data Processing"))
{
    // 작업 시뮬레이션
    Thread.Sleep(150);
    Console.WriteLine("Processing...");
}
// [START] Data Processing
// Processing...
// [END] Data Processing completed in 152ms
```

## 9. 모범 사례

### 9.1 필요하지 않으면 `Exception`을 잡지 마세요

```csharp
// 나쁨: 모든 것을 잡아 버그를 숨김
try
{
    ProcessOrder(order);
}
catch (Exception ex)
{
    Console.WriteLine("Something went wrong.");  // 모호, 실제 문제를 숨김
}

// 좋음: 특정 예외를 잡기
try
{
    ProcessOrder(order);
}
catch (PaymentDeclinedException ex)
{
    NotifyUser($"Payment declined: {ex.Reason}");
}
catch (InventoryException ex)
{
    NotifyUser($"Item unavailable: {ex.ItemName}");
}
// 예상치 못한 예외는 전역 핸들러로 전파되도록 함
```

### 9.2 예외를 삼키지 마세요

```csharp
// 나쁨: 예외를 삼킴 — 버그가 보이지 않게 됨
try
{
    SaveToDatabase(data);
}
catch (Exception)
{
    // 조용히 무시됨! 아무 표시 없이 데이터 손실.
}

// 좋음: 최소한 오류를 로깅
try
{
    SaveToDatabase(data);
}
catch (DbException ex)
{
    Logger.Error($"Failed to save data: {ex.Message}", ex);
    throw;  // 호출자가 알 수 있도록 다시 던짐
}
```

### 9.3 `throw ex`가 아닌 `throw`를 사용하세요

```csharp
// 나쁨: 스택 트레이스 손실
catch (Exception ex)
{
    Log(ex);
    throw ex;  // 스택 트레이스가 원래 위치가 아닌 여기를 가리킴
}

// 좋음: 스택 트레이스 보존
catch (Exception ex)
{
    Log(ex);
    throw;  // 스택 트레이스 보존됨
}

// 역시 좋음: inner exception으로 감싸기
catch (Exception ex)
{
    throw new ServiceException("Order processing failed", ex);
}
```

### 9.4 일찍 검증하고, 일찍 던지세요

```csharp
// 나쁨: 지연된 실패
public void SendEmail(string to, string subject, string body)
{
    // SMTP 연결을 열고, 메시지를 구성... 그런 다음 실패
    var smtp = new SmtpClient();
    smtp.Connect();
    var msg = new Message(to, subject, body);  // 여기서 NullReferenceException!
    smtp.Send(msg);
}

// 좋음: 매개변수를 즉시 검증
public void SendEmail(string to, string subject, string body)
{
    if (string.IsNullOrWhiteSpace(to))
        throw new ArgumentException("Recipient required.", nameof(to));
    if (string.IsNullOrWhiteSpace(subject))
        throw new ArgumentException("Subject required.", nameof(subject));
    if (body == null)
        throw new ArgumentNullException(nameof(body));

    // 이제 비용이 드는 작업을 안전하게 진행
    var smtp = new SmtpClient();
    smtp.Connect();
    smtp.Send(new Message(to, subject, body));
}
```

### 9.5 Parse + catch보다 `TryParse`를 선호하세요

```csharp
// 나쁨: 제어 흐름에 예외 사용
try
{
    int number = int.Parse(input);
    ProcessNumber(number);
}
catch (FormatException)
{
    Console.WriteLine("Invalid number.");
}

// 좋음: TryParse 사용
if (int.TryParse(input, out int number))
{
    ProcessNumber(number);
}
else
{
    Console.WriteLine("Invalid number.");
}
```

### 9.6 모범 사례 요약

```csharp
// 1. Exception이 아닌 특정 예외를 잡으세요
// 2. 예외를 조용히 삼키지 마세요
// 3. throw ex가 아닌 throw를 사용하세요
// 4. 입력을 일찍 검증하세요
// 5. 흐름 제어에 예외를 사용하지 마세요
// 6. IDisposable 리소스는 항상 해제하세요 (using 문)
// 7. 예외에 의미 있는 메시지를 포함하세요
// 8. 도메인별 오류에 사용자 정의 예외를 사용하세요
// 9. 다시 던지기 전에 예외를 로깅하세요
// 10. 조건부 처리에 예외 필터 (when)를 사용하세요
```

## 10. 실전 예제: 사용자 정의 예외를 사용한 입력 유효성 검사

사용자 정의 예외, 유효성 검사, 올바른 오류 처리를 갖춘 완전한 사용자 등록 시스템을 만들어 봅시다.

### 10.1 사용자 정의 예외

```csharp
public class ValidationException : Exception
{
    public List<ValidationError> Errors { get; }

    public ValidationException(List<ValidationError> errors)
        : base("Validation failed.")
    {
        Errors = errors;
    }
}

public class ValidationError
{
    public string Field { get; }
    public string Message { get; }

    public ValidationError(string field, string message)
    {
        Field = field;
        Message = message;
    }

    public override string ToString() => $"{Field}: {Message}";
}

public class DuplicateUserException : Exception
{
    public string Username { get; }

    public DuplicateUserException(string username)
        : base($"User '{username}' already exists.")
    {
        Username = username;
    }
}
```

### 10.2 유효성 검사기와 서비스

```csharp
public class UserRegistrationRequest
{
    public string Username { get; set; }
    public string Email { get; set; }
    public string Password { get; set; }
    public int Age { get; set; }
}

public static class UserValidator
{
    public static void Validate(UserRegistrationRequest request)
    {
        List<ValidationError> errors = new List<ValidationError>();

        if (string.IsNullOrWhiteSpace(request.Username))
            errors.Add(new ValidationError("Username", "Username is required."));
        else if (request.Username.Length < 3)
            errors.Add(new ValidationError("Username", "Username must be at least 3 characters."));
        else if (request.Username.Length > 20)
            errors.Add(new ValidationError("Username", "Username cannot exceed 20 characters."));

        if (string.IsNullOrWhiteSpace(request.Email))
            errors.Add(new ValidationError("Email", "Email is required."));
        else if (!request.Email.Contains("@") || !request.Email.Contains("."))
            errors.Add(new ValidationError("Email", "Email format is invalid."));

        if (string.IsNullOrWhiteSpace(request.Password))
            errors.Add(new ValidationError("Password", "Password is required."));
        else
        {
            if (request.Password.Length < 8)
                errors.Add(new ValidationError("Password", "Password must be at least 8 characters."));
            if (!request.Password.Any(char.IsUpper))
                errors.Add(new ValidationError("Password", "Password must contain an uppercase letter."));
            if (!request.Password.Any(char.IsDigit))
                errors.Add(new ValidationError("Password", "Password must contain a digit."));
        }

        if (request.Age < 13)
            errors.Add(new ValidationError("Age", "Must be at least 13 years old."));
        if (request.Age > 120)
            errors.Add(new ValidationError("Age", "Invalid age."));

        if (errors.Count > 0)
            throw new ValidationException(errors);
    }
}

public class UserService
{
    private readonly Dictionary<string, UserRegistrationRequest> _users = new();

    public void Register(UserRegistrationRequest request)
    {
        // 1단계: 유효성 검사
        UserValidator.Validate(request);

        // 2단계: 중복 확인
        if (_users.ContainsKey(request.Username.ToLower()))
            throw new DuplicateUserException(request.Username);

        // 3단계: 저장
        _users[request.Username.ToLower()] = request;
        Console.WriteLine($"User '{request.Username}' registered successfully.");
    }
}
```

### 10.3 모든 것 조합하기

```csharp
class Program
{
    static void Main()
    {
        UserService service = new UserService();

        // 테스트 1: 유효한 등록
        TryRegister(service, new UserRegistrationRequest
        {
            Username = "alice",
            Email = "alice@example.com",
            Password = "Secure123",
            Age = 25
        });

        // 테스트 2: 유효성 검사 오류
        TryRegister(service, new UserRegistrationRequest
        {
            Username = "ab",
            Email = "invalid-email",
            Password = "weak",
            Age = 10
        });

        // 테스트 3: 중복 사용자
        TryRegister(service, new UserRegistrationRequest
        {
            Username = "alice",
            Email = "alice2@example.com",
            Password = "AnotherPass1",
            Age = 30
        });
    }

    static void TryRegister(UserService service, UserRegistrationRequest request)
    {
        Console.WriteLine($"\n--- Registering: {request.Username} ---");
        try
        {
            service.Register(request);
        }
        catch (ValidationException ex)
        {
            Console.WriteLine("Registration failed — validation errors:");
            foreach (ValidationError error in ex.Errors)
            {
                Console.WriteLine($"  - {error}");
            }
        }
        catch (DuplicateUserException ex)
        {
            Console.WriteLine($"Registration failed: {ex.Message}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"Unexpected error: {ex.Message}");
        }
    }
}
```

출력:
```
--- Registering: alice ---
User 'alice' registered successfully.

--- Registering: ab ---
Registration failed — validation errors:
  - Username: Username must be at least 3 characters.
  - Email: Email format is invalid.
  - Password: Password must be at least 8 characters.
  - Password: Password must contain an uppercase letter.
  - Password: Password must contain a digit.
  - Age: Must be at least 13 years old.

--- Registering: alice ---
Registration failed: User 'alice' already exists.
```

## 11. 연습 문제

1. **오류 처리가 있는 계산기**: 사용자로부터 "10 / 3"과 같은 표현식을 읽는 명령줄 계산기를 만드세요. `FormatException`(숫자가 아닌 입력), `DivideByZeroException`, `OverflowException`(예: `int.MaxValue * 2`), 알 수 없는 연산자를 특정 메시지로 처리하세요. 사용자가 "quit"을 입력할 때까지 표현식을 계속 입력할 수 있도록 `while` 루프를 사용하세요. 프로그램이 절대 충돌하지 않게 하세요.

2. **사용자 정의 예외 계층 구조**: `ShoppingCartException` 기본 클래스를 만드세요. `ItemNotFoundException`, `InsufficientStockException`, `InvalidQuantityException`을 파생시키세요. 이 특정 예외들을 던지는 `AddItem`, `RemoveItem`, `UpdateQuantity` 메서드가 있는 `ShoppingCart` 클래스를 구현하세요. 모든 예외 경로를 실행하고 각각을 특정 처리로 잡는 테스트 코드를 작성하세요.

3. **파일 처리 파이프라인**: CSV 파일을 읽고, 각 행을 `Person` 객체(Name, Age, Email)로 파싱하고, 각 필드를 검증하고, 유효한 레코드를 출력 파일에 쓰는 프로그램을 작성하세요. 파싱 오류에 사용자 정의 예외를 사용하세요. 오류가 있는 줄 번호를 추적하세요. 마지막에 "Processed N records, M errors"를 보고하고 각 오류를 줄 번호와 함께 나열하세요. 오류 발생 시에도 모든 파일 핸들이 올바르게 닫히도록 하세요(`using` 사용).

4. **예외 필터 로거**: 예외 필터를 사용하여 예외를 분류하는 미들웨어 스타일 예외 핸들러를 만드세요. `catch...when`을 사용하는 `HandleException` 메서드를 작성하세요: (a) 일시적 오류를 로깅하고 재시도(시뮬레이션된 `TimeoutException` 최대 3회), (b) 보안 예외 알림(`UnauthorizedAccessException`), (c) 예상된 예외 무시(사용자 정의 `IgnorableException`), (d) 그 외 모든 것에 대한 catch-all. 각 경로를 시연하세요.

5. **리소스 관리자**: `T : IDisposable, new()`인 `ResourcePool<T>` 클래스를 구현하세요. N개의 리소스를 미리 생성하고, `Acquire()`와 `Release(T resource)` 연산을 허용하며, 모든 리소스를 해제하는 `IDisposable`을 자체 구현해야 합니다. `try-finally`를 사용하여 획득한 리소스가 항상 풀에 반환되도록 하세요. 리소스를 획득하고, 작업을 수행하고(예외를 던질 가능성 있음), 리소스가 반환됨을 보장하는 `using` 블록을 작성하세요.
