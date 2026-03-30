// Lesson 14: Exception Handling
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.IO;

// =============================================================================
// BASIC TRY / CATCH / FINALLY
// =============================================================================
Console.WriteLine("=== Basic Try/Catch/Finally ===");

try
{
    int result = 10 / 0;
    Console.WriteLine($"Result: {result}"); // Never reached
}
catch (DivideByZeroException ex)
{
    Console.WriteLine($"Caught: {ex.GetType().Name} — {ex.Message}");
}
finally
{
    Console.WriteLine("Finally block always executes.");
}

// =============================================================================
// MULTIPLE CATCH BLOCKS
// =============================================================================
Console.WriteLine("\n=== Multiple Catch Blocks ===");

object[] testCases = { "42", "abc", null!, new int[] { 1, 2, 3 } };

foreach (object item in testCases)
{
    try
    {
        // Force different exception types
        if (item is null)
            throw new ArgumentNullException(nameof(item));

        if (item is string s)
        {
            int value = int.Parse(s); // May throw FormatException
            Console.WriteLine($"  Parsed: {value}");
        }
        else if (item is int[] arr)
        {
            Console.WriteLine($"  arr[10]: {arr[10]}"); // IndexOutOfRangeException
        }
    }
    catch (ArgumentNullException ex)
    {
        Console.WriteLine($"  Null: {ex.Message}");
    }
    catch (FormatException ex)
    {
        Console.WriteLine($"  Format: {ex.Message}");
    }
    catch (IndexOutOfRangeException ex)
    {
        Console.WriteLine($"  Index: {ex.Message}");
    }
    catch (Exception ex)
    {
        // Catch-all (always put last)
        Console.WriteLine($"  General: {ex.GetType().Name} — {ex.Message}");
    }
}

// =============================================================================
// EXCEPTION FILTERS (when)
// =============================================================================
Console.WriteLine("\n=== Exception Filters (when) ===");

int[] errorCodes = { 404, 500, 403, 200 };

foreach (int code in errorCodes)
{
    try
    {
        if (code >= 400)
            throw new HttpException(code, $"HTTP Error {code}");
        Console.WriteLine($"  Code {code}: OK");
    }
    catch (HttpException ex) when (ex.StatusCode == 404)
    {
        Console.WriteLine($"  Code {code}: Not Found — {ex.Message}");
    }
    catch (HttpException ex) when (ex.StatusCode >= 500)
    {
        Console.WriteLine($"  Code {code}: Server Error — {ex.Message}");
    }
    catch (HttpException ex) when (ex.StatusCode >= 400)
    {
        Console.WriteLine($"  Code {code}: Client Error — {ex.Message}");
    }
}

// =============================================================================
// CUSTOM EXCEPTIONS
// =============================================================================
Console.WriteLine("\n=== Custom Exceptions ===");

var account = new BankAccount(1000);
Console.WriteLine($"Balance: {account.Balance:C}");

try
{
    account.Withdraw(500);
    Console.WriteLine($"After $500 withdrawal: {account.Balance:C}");

    account.Withdraw(600); // Should fail
}
catch (InsufficientFundsException ex)
{
    Console.WriteLine($"Error: {ex.Message}");
    Console.WriteLine($"  Attempted: {ex.Amount:C}, Available: {ex.Balance:C}");
    Console.WriteLine($"  Shortfall: {ex.Shortfall:C}");
}

// Nested custom exception
try
{
    ValidateAge(-5);
}
catch (ValidationException ex)
{
    Console.WriteLine($"\nValidation error: {ex.Message}");
    Console.WriteLine($"  Field: {ex.FieldName}, Value: {ex.InvalidValue}");
}

// =============================================================================
// THROW AND RE-THROW
// =============================================================================
Console.WriteLine("\n=== Throw and Re-throw ===");

try
{
    ProcessData(null);
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"Caught at top level: {ex.Message}");
    if (ex.InnerException != null)
    {
        Console.WriteLine($"  Inner exception: {ex.InnerException.Message}");
    }
}

// Re-throw preserving stack trace
try
{
    try
    {
        int.Parse("not_a_number");
    }
    catch (FormatException)
    {
        Console.WriteLine("  Caught FormatException, re-throwing...");
        throw; // Re-throw preserves stack trace (not 'throw ex')
    }
}
catch (FormatException ex)
{
    Console.WriteLine($"  Re-caught: {ex.Message}");
}

// =============================================================================
// USING STATEMENT AND IDisposable
// =============================================================================
Console.WriteLine("\n=== Using Statement (IDisposable) ===");

// Traditional using statement
Console.WriteLine("Traditional using:");
using (var resource = new ManagedResource("ResourceA"))
{
    resource.DoWork();
    Console.WriteLine("  Inside using block.");
} // Dispose() called automatically here
Console.WriteLine("  After using block.");

// Using declaration (C# 8+) — disposed at end of scope
Console.WriteLine("\nUsing declaration:");
{
    using var res = new ManagedResource("ResourceB");
    res.DoWork();
    Console.WriteLine("  Inside scope.");
} // Disposed here

// Multiple using statements
Console.WriteLine("\nMultiple resources:");
using (var r1 = new ManagedResource("R1"))
using (var r2 = new ManagedResource("R2"))
{
    r1.DoWork();
    r2.DoWork();
} // Both disposed in reverse order

// =============================================================================
// EXCEPTION IN USING BLOCK
// =============================================================================
Console.WriteLine("\n=== Exception in Using Block ===");

try
{
    using var res = new ManagedResource("CrashTest");
    res.DoWork();
    throw new InvalidOperationException("Something went wrong!");
}
catch (InvalidOperationException ex)
{
    Console.WriteLine($"  Caught: {ex.Message}");
    Console.WriteLine("  (Resource was still disposed before catch!)");
}

// =============================================================================
// AGGREGATE EXCEPTION (multiple errors)
// =============================================================================
Console.WriteLine("\n=== AggregateException ===");

var errors = new List<Exception>();

string[] inputs = { "10", "abc", "20", "", "30" };
var results = new List<int>();

foreach (string input in inputs)
{
    try
    {
        if (string.IsNullOrEmpty(input))
            throw new ArgumentException("Input cannot be empty.");
        results.Add(int.Parse(input));
    }
    catch (Exception ex)
    {
        errors.Add(ex);
    }
}

Console.WriteLine($"Successfully parsed: [{string.Join(", ", results)}]");

if (errors.Count > 0)
{
    var aggregate = new AggregateException("Multiple parse errors", errors);
    Console.WriteLine($"Errors ({aggregate.InnerExceptions.Count}):");
    foreach (var ex in aggregate.InnerExceptions)
    {
        Console.WriteLine($"  - {ex.GetType().Name}: {ex.Message}");
    }
}

// =============================================================================
// BEST PRACTICES SUMMARY
// =============================================================================
Console.WriteLine("\n=== Best Practices ===");
Console.WriteLine("1. Catch specific exceptions, not just 'Exception'.");
Console.WriteLine("2. Use 'throw;' to re-throw (preserves stack trace).");
Console.WriteLine("3. Use 'using' for IDisposable resources.");
Console.WriteLine("4. Create custom exceptions for domain-specific errors.");
Console.WriteLine("5. Use exception filters (when) for conditional catching.");
Console.WriteLine("6. Don't use exceptions for normal control flow.");

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

class HttpException : Exception
{
    public int StatusCode { get; }

    public HttpException(int statusCode, string message)
        : base(message)
    {
        StatusCode = statusCode;
    }
}

class InsufficientFundsException : Exception
{
    public decimal Amount { get; }
    public decimal Balance { get; }
    public decimal Shortfall => Amount - Balance;

    public InsufficientFundsException(decimal amount, decimal balance)
        : base($"Insufficient funds: tried to withdraw {amount:C} but only {balance:C} available.")
    {
        Amount = amount;
        Balance = balance;
    }
}

class ValidationException : Exception
{
    public string FieldName { get; }
    public object? InvalidValue { get; }

    public ValidationException(string field, object? value, string message)
        : base(message)
    {
        FieldName = field;
        InvalidValue = value;
    }
}

class BankAccount
{
    public decimal Balance { get; private set; }

    public BankAccount(decimal initial) => Balance = initial;

    public void Withdraw(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("Amount must be positive.", nameof(amount));
        if (amount > Balance)
            throw new InsufficientFundsException(amount, Balance);
        Balance -= amount;
    }
}

static void ValidateAge(int age)
{
    if (age < 0 || age > 150)
        throw new ValidationException("Age", age, $"Age {age} is out of valid range (0-150).");
}

static void ProcessData(string? data)
{
    try
    {
        if (data is null)
            throw new ArgumentNullException(nameof(data));
        Console.WriteLine($"Processing: {data}");
    }
    catch (ArgumentNullException ex)
    {
        // Wrap in a new exception, preserving the original as InnerException
        throw new InvalidOperationException("Data processing failed.", ex);
    }
}

/// <summary>
/// IDisposable implementation for using statement demo.
/// </summary>
class ManagedResource : IDisposable
{
    private readonly string _name;
    private bool _disposed;

    public ManagedResource(string name)
    {
        _name = name;
        Console.WriteLine($"  [{_name}] Created.");
    }

    public void DoWork()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Console.WriteLine($"  [{_name}] Working...");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            Console.WriteLine($"  [{_name}] Disposed.");
            _disposed = true;
        }
    }
}
