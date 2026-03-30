# Exception Handling

**Previous**: [Generics](./13_Generics.md) | **Next**: [File I/O](./15_File_IO.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Use `try`, `catch`, and `finally` blocks to handle exceptions
2. Catch multiple exception types from specific to general
3. Navigate the .NET exception hierarchy and common exception types
4. Throw and rethrow exceptions correctly
5. Create custom exception classes
6. Use exception filters with `when` clauses
7. Manage resources with the `using` statement and `IDisposable`
8. Apply best practices for robust error handling

---

Every non-trivial program encounters errors: files that do not exist, network connections that time out, invalid user input, and unexpected null references. C# uses a structured exception handling model built around `try`, `catch`, `finally`, and `throw`. When used correctly, exceptions separate error-handling logic from normal flow, produce meaningful diagnostics, and prevent your application from crashing ungracefully. This lesson covers the full exception handling toolkit and the best practices that professional C# developers follow.

## 1. `try`, `catch`, `finally` Blocks

The `try-catch-finally` construct is the foundation of exception handling in C#.

### 1.1 Basic try-catch

```csharp
try
{
    Console.Write("Enter a number: ");
    string input = Console.ReadLine();
    int number = int.Parse(input);  // May throw FormatException
    int result = 100 / number;      // May throw DivideByZeroException
    Console.WriteLine($"100 / {number} = {result}");
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
```

### 1.2 The `finally` Block

The `finally` block always executes, regardless of whether an exception occurred. It is typically used for cleanup.

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
    // Always executes, even if an exception is thrown
    if (reader != null)
    {
        reader.Close();
        Console.WriteLine("Reader closed.");
    }
}
```

### 1.3 try-finally Without catch

You can use `try-finally` without a `catch` block when you want to ensure cleanup but let exceptions propagate.

```csharp
public void ProcessData()
{
    var connection = OpenDatabaseConnection();
    try
    {
        // Do work that might throw
        connection.Execute("INSERT INTO logs VALUES ('started')");
        // ... more operations ...
    }
    finally
    {
        // Always close the connection, even if an exception propagates
        connection.Close();
    }
    // If an exception occurred, it will propagate after finally runs
}
```

### 1.4 Nested try-catch

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
        // Handle or rethrow
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

// Output:
// Outer try
// Inner try
// Inner catch: Something went wrong
// Inner finally
// After inner try-catch
// Outer finally
```

## 2. Multiple Catch Blocks

When multiple exception types are possible, list catch blocks from most specific to most general. The runtime uses the first matching catch block.

### 2.1 Ordering Catch Blocks

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
    // General catch — always put last
    Console.WriteLine($"Unexpected error: {ex.Message}");
}
```

### 2.2 Catching Multiple Types in One Block (C# 6+)

If you want the same handling logic for multiple exception types, use `when` or handle them individually. Prior to exception filters, you would catch `Exception` and check the type.

```csharp
try
{
    // Some operation
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

### 2.3 Accessing Exception Details

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

## 3. Exception Hierarchy

All exceptions in .NET derive from `System.Exception`. Understanding the hierarchy helps you catch the right types.

### 3.1 The Hierarchy

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
└── System.ApplicationException (deprecated, do not use)
```

### 3.2 Common Exception Types

```csharp
// ArgumentException: invalid argument
public void SetAge(int age)
{
    if (age < 0 || age > 150)
        throw new ArgumentOutOfRangeException(nameof(age), age,
            "Age must be between 0 and 150.");
}

// ArgumentNullException: null where not allowed
public void Greet(string name)
{
    if (name == null)
        throw new ArgumentNullException(nameof(name));
    Console.WriteLine($"Hello, {name}!");
}

// InvalidOperationException: operation not valid for current state
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

// NotSupportedException: method not supported
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

The most common exception in C# programs. It occurs when you try to use a member of a null reference.

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

// Prevention: null checks and null-conditional operator
int? safeLen = text?.Length;   // null, no exception
int length = text?.Length ?? 0; // 0
```

## 4. Throwing Exceptions

### 4.1 The `throw` Statement

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

### 4.2 `throw` vs `throw ex` — Preserving Stack Trace

This is one of the most important exception handling rules in C#.

```csharp
// WRONG: throw ex — resets the stack trace
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    LogError(ex);
    throw ex;  // BAD: Stack trace is reset to this line
}

// CORRECT: throw — preserves the original stack trace
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    LogError(ex);
    throw;  // GOOD: Original stack trace is preserved
}

// ALSO CORRECT: wrap in a new exception with InnerException
try
{
    SomeRiskyOperation();
}
catch (Exception ex)
{
    throw new ApplicationException("Operation failed", ex);
    // The original exception is preserved as InnerException
}
```

### 4.3 Conditional Throw

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

// Usage
public void RegisterUser(string username, string email, int age)
{
    Validator.EnsureNotEmpty(username, nameof(username));
    Validator.EnsureNotEmpty(email, nameof(email));
    Validator.EnsureInRange(age, 13, 120, nameof(age));

    // Proceed with valid data
    Console.WriteLine($"Registered: {username}, {email}, age {age}");
}
```

### 4.4 Throw Expression (C# 7+)

```csharp
public class Config
{
    private readonly string _connectionString;

    // Throw expression in constructor
    public Config(string connectionString)
    {
        _connectionString = connectionString
            ?? throw new ArgumentNullException(nameof(connectionString));
    }

    // Throw expression in null-coalescing
    public string GetSetting(Dictionary<string, string> settings, string key)
    {
        return settings.TryGetValue(key, out string value)
            ? value
            : throw new KeyNotFoundException($"Setting '{key}' not found.");
    }

    // Throw expression in conditional
    public string Name { get; set; }
    public string DisplayName => Name ?? throw new InvalidOperationException("Name not set.");
}
```

## 5. Custom Exception Classes

Creating custom exceptions provides meaningful error types specific to your application's domain.

### 5.1 Basic Custom Exception

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

### 5.2 Using Custom Exceptions

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
// Output:
// Insufficient funds. Balance: $500.00, Requested: $750.00
//   Current balance: $500.00
//   Requested: $750.00
//   Shortfall: $250.00
```

### 5.3 Exception Hierarchy for a Domain

```csharp
// Base exception for the application
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

// Specific domain exceptions
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
// Usage in a service
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

## 6. Exception Filters (`when` Clause)

Exception filters, introduced in C# 6, allow you to add conditions to catch blocks without catching and rethrowing.

### 6.1 Basic Exception Filters

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

### 6.2 Filtering by Message or Properties

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

### 6.3 Logging Without Catching

A powerful technique: use `when` to log the exception without actually catching it.

```csharp
try
{
    RiskyOperation();
}
catch (Exception ex) when (LogException(ex))
{
    // This block never executes because LogException returns false
}
catch (SpecificException ex)
{
    // This handler still gets the exception
    HandleSpecific(ex);
}

static bool LogException(Exception ex)
{
    Console.WriteLine($"[LOG] Exception occurred: {ex.GetType().Name}: {ex.Message}");
    return false;  // Never catches — just logs
}
```

### 6.4 Environment-Based Filtering

```csharp
bool isDevelopment = Environment.GetEnvironmentVariable("DOTNET_ENVIRONMENT") == "Development";

try
{
    ProcessOrder(orderId);
}
catch (Exception ex) when (isDevelopment)
{
    // In development, show full details
    Console.WriteLine($"DEV ERROR: {ex}");
}
catch (Exception ex) when (!isDevelopment)
{
    // In production, show user-friendly message
    Console.WriteLine("An error occurred. Please contact support.");
    LogToFile(ex);
}
```

## 7. `using` Statement and `IDisposable`

The `using` statement ensures that `IDisposable` objects are properly disposed, even if an exception occurs. It is syntactic sugar for a `try-finally` block.

### 7.1 The `IDisposable` Interface

```csharp
public interface IDisposable
{
    void Dispose();
}
```

Many .NET classes implement `IDisposable`: file streams, database connections, network sockets, timers, and more. Always dispose these resources when done.

### 7.2 The `using` Statement

```csharp
// Without using: manual try-finally
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

// With using statement: cleaner and safer
using (StreamReader reader2 = new StreamReader("data.txt"))
{
    string content = reader2.ReadToEnd();
    Console.WriteLine(content);
}
// reader2.Dispose() is called automatically here
```

### 7.3 The `using` Declaration (C# 8+)

```csharp
// using declaration: disposes at end of enclosing scope
public void ProcessFile(string path)
{
    using var reader = new StreamReader(path);  // No braces needed
    string line;
    while ((line = reader.ReadLine()) != null)
    {
        Console.WriteLine(line);
    }
    // reader is disposed here, at end of method
}
```

### 7.4 Multiple `using` Statements

```csharp
// Nested using (traditional)
using (var input = new StreamReader("input.txt"))
using (var output = new StreamWriter("output.txt"))
{
    string line;
    while ((line = input.ReadLine()) != null)
    {
        output.WriteLine(line.ToUpper());
    }
}

// Using declarations (C# 8+) — cleaner
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
// Both disposed here
```

### 7.5 Implementing IDisposable

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
// Output:
// Connection opened.
// Executing: SELECT * FROM users
// Executing: UPDATE users SET active = 1 WHERE id = 5
// Connection closed.
```

## 8. `try-finally` for Resource Cleanup

Even when you cannot use the `using` statement (e.g., the resource is not `IDisposable`, or you need more control), `try-finally` ensures cleanup.

### 8.1 Locking Pattern

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

### 8.2 Temporary State Changes

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

### 8.3 Stopwatch Pattern

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
    // Simulate work
    Thread.Sleep(150);
    Console.WriteLine("Processing...");
}
// [START] Data Processing
// Processing...
// [END] Data Processing completed in 152ms
```

## 9. Best Practices

### 9.1 Do Not Catch `Exception` Unless Necessary

```csharp
// BAD: catches everything, masks bugs
try
{
    ProcessOrder(order);
}
catch (Exception ex)
{
    Console.WriteLine("Something went wrong.");  // Vague, hides the real issue
}

// GOOD: catch specific exceptions
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
// Let unexpected exceptions propagate to a global handler
```

### 9.2 Do Not Swallow Exceptions

```csharp
// BAD: swallows the exception — bugs become invisible
try
{
    SaveToDatabase(data);
}
catch (Exception)
{
    // Silently ignored! Data loss with no indication.
}

// GOOD: at minimum, log the error
try
{
    SaveToDatabase(data);
}
catch (DbException ex)
{
    Logger.Error($"Failed to save data: {ex.Message}", ex);
    throw;  // Rethrow so the caller knows
}
```

### 9.3 Use `throw` Not `throw ex`

```csharp
// BAD: loses stack trace
catch (Exception ex)
{
    Log(ex);
    throw ex;  // Stack trace points here, not the original location
}

// GOOD: preserves stack trace
catch (Exception ex)
{
    Log(ex);
    throw;  // Stack trace preserved
}

// ALSO GOOD: wrap with inner exception
catch (Exception ex)
{
    throw new ServiceException("Order processing failed", ex);
}
```

### 9.4 Validate Early, Throw Early

```csharp
// BAD: delayed failure
public void SendEmail(string to, string subject, string body)
{
    // Opens SMTP connection, composes message... then fails
    var smtp = new SmtpClient();
    smtp.Connect();
    var msg = new Message(to, subject, body);  // NullReferenceException here!
    smtp.Send(msg);
}

// GOOD: validate parameters immediately
public void SendEmail(string to, string subject, string body)
{
    if (string.IsNullOrWhiteSpace(to))
        throw new ArgumentException("Recipient required.", nameof(to));
    if (string.IsNullOrWhiteSpace(subject))
        throw new ArgumentException("Subject required.", nameof(subject));
    if (body == null)
        throw new ArgumentNullException(nameof(body));

    // Now safe to proceed with expensive operations
    var smtp = new SmtpClient();
    smtp.Connect();
    smtp.Send(new Message(to, subject, body));
}
```

### 9.5 Prefer `TryParse` Over Parse + catch

```csharp
// BAD: using exceptions for control flow
try
{
    int number = int.Parse(input);
    ProcessNumber(number);
}
catch (FormatException)
{
    Console.WriteLine("Invalid number.");
}

// GOOD: use TryParse
if (int.TryParse(input, out int number))
{
    ProcessNumber(number);
}
else
{
    Console.WriteLine("Invalid number.");
}
```

### 9.6 Summary of Best Practices

```csharp
// 1. Catch specific exceptions, not Exception
// 2. Never swallow exceptions silently
// 3. Use throw, not throw ex
// 4. Validate inputs early
// 5. Don't use exceptions for flow control
// 6. Always dispose IDisposable resources (using statement)
// 7. Include meaningful messages in exceptions
// 8. Use custom exceptions for domain-specific errors
// 9. Log exceptions before rethrowing
// 10. Use exception filters (when) for conditional handling
```

## 10. Practical Example: Input Validation with Custom Exceptions

Let us build a complete user registration system with custom exceptions, validation, and proper error handling.

### 10.1 Custom Exceptions

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

### 10.2 Validator and Service

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
        // Step 1: Validate
        UserValidator.Validate(request);

        // Step 2: Check for duplicates
        if (_users.ContainsKey(request.Username.ToLower()))
            throw new DuplicateUserException(request.Username);

        // Step 3: Save
        _users[request.Username.ToLower()] = request;
        Console.WriteLine($"User '{request.Username}' registered successfully.");
    }
}
```

### 10.3 Putting It All Together

```csharp
class Program
{
    static void Main()
    {
        UserService service = new UserService();

        // Test 1: Valid registration
        TryRegister(service, new UserRegistrationRequest
        {
            Username = "alice",
            Email = "alice@example.com",
            Password = "Secure123",
            Age = 25
        });

        // Test 2: Validation errors
        TryRegister(service, new UserRegistrationRequest
        {
            Username = "ab",
            Email = "invalid-email",
            Password = "weak",
            Age = 10
        });

        // Test 3: Duplicate user
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

Output:
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

## 11. Practice Problems

1. **Calculator with Error Handling**: Build a command-line calculator that reads expressions like "10 / 3" from the user. Handle `FormatException` (non-numeric input), `DivideByZeroException`, `OverflowException` (e.g., `int.MaxValue * 2`), and unknown operators with specific messages. Use a `while` loop so the user can keep entering expressions until they type "quit". Never let the program crash.

2. **Custom Exception Hierarchy**: Create a `ShoppingCartException` base class. Derive `ItemNotFoundException`, `InsufficientStockException`, and `InvalidQuantityException`. Implement a `ShoppingCart` class with `AddItem`, `RemoveItem`, and `UpdateQuantity` methods that throw these specific exceptions. Write test code that exercises all exception paths and catches each one with specific handling.

3. **File Processing Pipeline**: Write a program that reads a CSV file, parses each row into a `Person` object (Name, Age, Email), validates each field, and writes valid records to an output file. Use custom exceptions for parsing errors. Track which line numbers had errors. At the end, report: "Processed N records, M errors" and list each error with its line number. Ensure all file handles are properly closed even on error (use `using`).

4. **Exception Filter Logger**: Create a middleware-style exception handler that uses exception filters to categorize exceptions. Write a `HandleException` method that uses `catch...when` to: (a) log and retry transient errors (simulated `TimeoutException` up to 3 times), (b) alert on security exceptions (`UnauthorizedAccessException`), (c) ignore expected exceptions (custom `IgnorableException`), and (d) catch-all for everything else. Demonstrate each path.

5. **Resource Manager**: Implement a `ResourcePool<T>` class where `T : IDisposable, new()`. It should pre-create N resources, allow `Acquire()` and `Release(T resource)` operations, and implement `IDisposable` itself to dispose all resources. Use `try-finally` to ensure that acquired resources are always released back to the pool. Write a `using` block that acquires a resource, does work (possibly throwing an exception), and guarantees the resource is returned.
