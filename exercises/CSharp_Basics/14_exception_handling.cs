/*
 * Exercises for Lesson 14: Exception Handling
 * Topic: CSharp_Basics
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

// Exercise 1: Try-catch-finally basics
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Try-Catch-Finally ===");

    // Division with exception handling
    (int a, int b)[] divisions = { (10, 3), (100, 0), (42, 7), (-15, 5) };

    foreach (var (a, b) in divisions)
    {
        try
        {
            int result = SafeDivide(a, b);
            Console.WriteLine($"  {a} / {b} = {result}");
        }
        catch (DivideByZeroException)
        {
            Console.WriteLine($"  {a} / {b} = Error: division by zero");
        }
    }

    // Multiple catch blocks
    Console.WriteLine("\nMultiple exception types:");
    string[] inputs = { "42", "abc", "", null! };
    foreach (string? input in inputs)
    {
        try
        {
            int val = ParsePositive(input!);
            Console.WriteLine($"  \"{input}\" -> {val}");
        }
        catch (ArgumentNullException)
        {
            Console.WriteLine($"  null -> ArgumentNullException");
        }
        catch (ArgumentException ex)
        {
            Console.WriteLine($"  \"{input}\" -> ArgumentException: {ex.Message}");
        }
        catch (FormatException)
        {
            Console.WriteLine($"  \"{input}\" -> FormatException");
        }
    }

    // Finally block
    Console.WriteLine("\nFinally block:");
    try
    {
        Console.WriteLine("  In try block");
        throw new InvalidOperationException("test");
    }
    catch (InvalidOperationException)
    {
        Console.WriteLine("  In catch block");
    }
    finally
    {
        Console.WriteLine("  In finally block (always runs)");
    }
    Console.WriteLine();

    static int SafeDivide(int a, int b) => b == 0 ? throw new DivideByZeroException() : a / b;

    static int ParsePositive(string s)
    {
        if (s is null) throw new ArgumentNullException(nameof(s));
        if (s.Length == 0) throw new ArgumentException("String cannot be empty", nameof(s));
        return int.Parse(s);
    }
}

// Exercise 2: Custom exceptions with hierarchy
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Custom Exceptions ===");

    var validator = new UserValidator();
    string[][] testCases =
    {
        new[] { "alice", "alice@example.com", "Str0ngP@ss!" },
        new[] { "", "alice@example.com", "password" },
        new[] { "bob", "not-an-email", "Str0ngP@ss!" },
        new[] { "charlie", "charlie@test.com", "weak" },
        new[] { "x", "x@x.com", "Str0ngP@ss!" }
    };

    foreach (var tc in testCases)
    {
        try
        {
            validator.Validate(tc[0], tc[1], tc[2]);
            Console.WriteLine($"  ({tc[0]}, {tc[1]}) -> Valid");
        }
        catch (ValidationException ex)
        {
            Console.WriteLine($"  ({tc[0]}, {tc[1]}) -> {ex.GetType().Name}: {ex.Message} [field={ex.FieldName}]");
        }
    }
    Console.WriteLine();
}

// Exercise 3: Exception filters with 'when'
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Exception Filters ===");

    // Filter by exception properties
    int[] errorCodes = { 404, 500, 401, 403, 503 };
    foreach (int code in errorCodes)
    {
        try
        {
            throw new HttpException(code, $"HTTP Error {code}");
        }
        catch (HttpException ex) when (ex.StatusCode == 401 || ex.StatusCode == 403)
        {
            Console.WriteLine($"  {code}: Authentication/Authorization error — {ex.Message}");
        }
        catch (HttpException ex) when (ex.StatusCode >= 500)
        {
            Console.WriteLine($"  {code}: Server error — {ex.Message}");
        }
        catch (HttpException ex) when (ex.StatusCode == 404)
        {
            Console.WriteLine($"  {code}: Not found — {ex.Message}");
        }
        catch (HttpException ex)
        {
            Console.WriteLine($"  {code}: Other HTTP error — {ex.Message}");
        }
    }

    // Retry pattern with exception filter
    Console.WriteLine("\nRetry pattern:");
    int attempt = 0;
    bool success = false;
    while (!success && attempt < 5)
    {
        try
        {
            attempt++;
            SimulateFlaky(attempt);
            Console.WriteLine($"  Attempt {attempt}: Success!");
            success = true;
        }
        catch (TimeoutException) when (attempt < 5)
        {
            Console.WriteLine($"  Attempt {attempt}: Timeout, retrying...");
        }
        catch (TimeoutException)
        {
            Console.WriteLine($"  Attempt {attempt}: Timeout, giving up.");
        }
    }
    Console.WriteLine();

    static void SimulateFlaky(int attempt)
    {
        if (attempt < 3) throw new TimeoutException($"Attempt {attempt} timed out");
    }
}

// Exercise 4: Inner exceptions and AggregateException
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Inner Exceptions ===");

    // Wrapping exceptions
    try
    {
        ProcessOrder(0);
    }
    catch (OrderProcessingException ex)
    {
        Console.WriteLine($"OrderProcessingException: {ex.Message}");
        Console.WriteLine($"  Inner: {ex.InnerException?.GetType().Name}: {ex.InnerException?.Message}");
        Console.WriteLine($"  Order ID: {ex.OrderId}");
    }

    // AggregateException — multiple failures
    Console.WriteLine("\nAggregateException:");
    var errors = new List<Exception>();
    string[] records = { "valid:100", "invalid", "valid:200", "error:fail", "valid:300" };

    foreach (string record in records)
    {
        try
        {
            ProcessRecord(record);
            Console.WriteLine($"  \"{record}\" -> OK");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  \"{record}\" -> Failed: {ex.Message}");
            errors.Add(ex);
        }
    }

    if (errors.Count > 0)
    {
        var aggregate = new AggregateException("Batch processing had errors", errors);
        Console.WriteLine($"\nTotal errors: {aggregate.InnerExceptions.Count}");
        foreach (var ex in aggregate.InnerExceptions)
            Console.WriteLine($"  - {ex.Message}");
    }
    Console.WriteLine();

    static void ProcessOrder(int orderId)
    {
        try
        {
            if (orderId <= 0) throw new ArgumentException("Invalid order ID");
        }
        catch (Exception ex)
        {
            throw new OrderProcessingException(orderId, "Failed to process order", ex);
        }
    }

    static void ProcessRecord(string record)
    {
        var parts = record.Split(':');
        if (parts[0] == "invalid") throw new FormatException("Invalid record format");
        if (parts[0] == "error") throw new InvalidOperationException($"Processing error: {parts[1]}");
    }
}

// Exercise 5: Disposable pattern and using statements
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Disposable Pattern ===");

    // using statement ensures Dispose is called
    Console.WriteLine("Using statement:");
    using (var resource = new ManagedResource("Connection-1"))
    {
        resource.DoWork();
        Console.WriteLine($"  Resource is active: {resource.IsActive}");
    }
    Console.WriteLine("  After using block — resource disposed.\n");

    // using declaration (C# 8.0+)
    Console.WriteLine("Using declaration:");
    DoWorkWithResource();

    // Exception in using block — Dispose still called
    Console.WriteLine("\nException in using block:");
    try
    {
        using var res = new ManagedResource("Connection-3");
        throw new InvalidOperationException("Something went wrong");
    }
    catch (InvalidOperationException ex)
    {
        Console.WriteLine($"  Caught: {ex.Message}");
        Console.WriteLine("  Resource was still disposed (check output above)");
    }
    Console.WriteLine();

    static void DoWorkWithResource()
    {
        using var resource = new ManagedResource("Connection-2");
        resource.DoWork();
        Console.WriteLine($"  Resource is active: {resource.IsActive}");
        // Dispose called at end of method scope
    }
}

// Supporting types

class ValidationException : Exception
{
    public string FieldName { get; }
    public ValidationException(string field, string message) : base(message) => FieldName = field;
}

class UsernameException : ValidationException
{
    public UsernameException(string message) : base("username", message) { }
}

class EmailException : ValidationException
{
    public EmailException(string message) : base("email", message) { }
}

class PasswordException : ValidationException
{
    public PasswordException(string message) : base("password", message) { }
}

class UserValidator
{
    public void Validate(string username, string email, string password)
    {
        if (string.IsNullOrWhiteSpace(username)) throw new UsernameException("Username is required");
        if (username.Length < 3) throw new UsernameException("Username must be at least 3 characters");
        if (!email.Contains('@')) throw new EmailException("Invalid email format");
        if (password.Length < 8) throw new PasswordException("Password must be at least 8 characters");
    }
}

class HttpException : Exception
{
    public int StatusCode { get; }
    public HttpException(int statusCode, string message) : base(message) => StatusCode = statusCode;
}

class OrderProcessingException : Exception
{
    public int OrderId { get; }
    public OrderProcessingException(int orderId, string message, Exception inner) : base(message, inner)
        => OrderId = orderId;
}

class ManagedResource : IDisposable
{
    public string Name { get; }
    public bool IsActive { get; private set; }

    public ManagedResource(string name)
    {
        Name = name;
        IsActive = true;
        Console.WriteLine($"  [{Name}] Acquired");
    }

    public void DoWork() => Console.WriteLine($"  [{Name}] Working...");

    public void Dispose()
    {
        if (IsActive)
        {
            IsActive = false;
            Console.WriteLine($"  [{Name}] Disposed");
        }
    }
}

// Main
Exercise1();
Exercise2();
Exercise3();
Exercise4();
Exercise5();
