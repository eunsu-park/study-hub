// Lesson 12: Testing with xUnit
// Run: dotnet test
// Note: This file demonstrates xUnit test patterns.
//   To run, create a test project:
//     dotnet new xunit -n MyTests
//     dotnet add package FluentAssertions  (optional)
//   Then place this file in the test project.

using System;
using System.Collections.Generic;
using System.Linq;

// ============================================================
// Production Code (System Under Test)
// ============================================================

/// <summary>Simple calculator for demonstration.</summary>
public class Calculator
{
    public int Add(int a, int b) => a + b;
    public int Subtract(int a, int b) => a - b;
    public double Divide(double a, double b)
    {
        if (b == 0) throw new DivideByZeroException("Cannot divide by zero");
        return a / b;
    }
    public bool IsEven(int n) => n % 2 == 0;
}

/// <summary>String utilities for demonstration.</summary>
public static class StringUtils
{
    public static string Reverse(string input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        return new string(input.Reverse().ToArray());
    }

    public static bool IsPalindrome(string input)
    {
        if (input is null) throw new ArgumentNullException(nameof(input));
        var cleaned = input.ToLower().Replace(" ", "");
        return cleaned == Reverse(cleaned);
    }

    public static int WordCount(string input)
    {
        if (string.IsNullOrWhiteSpace(input)) return 0;
        return input.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length;
    }
}

/// <summary>User service with dependency for testing DI patterns.</summary>
public interface IUserRepository
{
    User? GetById(int id);
    void Save(User user);
}

public record User(int Id, string Name, string Email);

public class UserService
{
    private readonly IUserRepository _repository;

    public UserService(IUserRepository repository)
    {
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
    }

    public User? GetUser(int id) => _repository.GetById(id);

    public void CreateUser(string name, string email)
    {
        if (string.IsNullOrWhiteSpace(name))
            throw new ArgumentException("Name is required", nameof(name));
        if (!email.Contains('@'))
            throw new ArgumentException("Invalid email", nameof(email));

        var user = new User(0, name, email);
        _repository.Save(user);
    }
}

// ============================================================
// xUnit Test Examples
// ============================================================

// NOTE: In a real project, these would be in a separate test project
// with `using Xunit;` and xUnit NuGet package references.
// The code below shows the patterns with the attribute names.

/*
// -------------------------------------------------------
// 1. [Fact] — Simple Test Cases
// -------------------------------------------------------

public class CalculatorTests
{
    private readonly Calculator _calc = new();

    [Fact]
    public void Add_TwoPositiveNumbers_ReturnsSum()
    {
        // Arrange
        int a = 3, b = 4;

        // Act
        int result = _calc.Add(a, b);

        // Assert
        Assert.Equal(7, result);
    }

    [Fact]
    public void Subtract_LargerFromSmaller_ReturnsNegative()
    {
        int result = _calc.Subtract(3, 10);
        Assert.Equal(-7, result);
    }

    [Fact]
    public void Divide_ByZero_ThrowsDivideByZeroException()
    {
        var ex = Assert.Throws<DivideByZeroException>(
            () => _calc.Divide(10, 0)
        );
        Assert.Contains("Cannot divide by zero", ex.Message);
    }

    [Fact]
    public void IsEven_WithEvenNumber_ReturnsTrue()
    {
        Assert.True(_calc.IsEven(4));
    }

    [Fact]
    public void IsEven_WithOddNumber_ReturnsFalse()
    {
        Assert.False(_calc.IsEven(7));
    }
}

// -------------------------------------------------------
// 2. [Theory] — Parameterized Tests
// -------------------------------------------------------

public class CalculatorTheoryTests
{
    private readonly Calculator _calc = new();

    [Theory]
    [InlineData(1, 1, 2)]
    [InlineData(0, 0, 0)]
    [InlineData(-1, 1, 0)]
    [InlineData(100, -50, 50)]
    [InlineData(int.MaxValue, 0, int.MaxValue)]
    public void Add_VariousInputs_ReturnsExpected(int a, int b, int expected)
    {
        Assert.Equal(expected, _calc.Add(a, b));
    }

    [Theory]
    [InlineData(10.0, 2.0, 5.0)]
    [InlineData(7.0, 2.0, 3.5)]
    [InlineData(-10.0, 4.0, -2.5)]
    public void Divide_ValidInputs_ReturnsQuotient(double a, double b, double expected)
    {
        Assert.Equal(expected, _calc.Divide(a, b), precision: 10);
    }
}

// -------------------------------------------------------
// 3. String Utility Tests
// -------------------------------------------------------

public class StringUtilsTests
{
    [Fact]
    public void Reverse_NormalString_ReturnsReversed()
    {
        Assert.Equal("olleh", StringUtils.Reverse("hello"));
    }

    [Fact]
    public void Reverse_NullInput_ThrowsArgumentNullException()
    {
        Assert.Throws<ArgumentNullException>(() => StringUtils.Reverse(null!));
    }

    [Theory]
    [InlineData("racecar", true)]
    [InlineData("madam", true)]
    [InlineData("hello", false)]
    [InlineData("A man a plan a canal Panama", true)]
    public void IsPalindrome_VariousInputs_ReturnsExpected(string input, bool expected)
    {
        Assert.Equal(expected, StringUtils.IsPalindrome(input));
    }

    [Theory]
    [InlineData("hello world", 2)]
    [InlineData("one", 1)]
    [InlineData("", 0)]
    [InlineData("  multiple   spaces  ", 2)]
    public void WordCount_VariousInputs_ReturnsExpected(string input, int expected)
    {
        Assert.Equal(expected, StringUtils.WordCount(input));
    }
}

// -------------------------------------------------------
// 4. Assert Patterns
// -------------------------------------------------------

public class AssertPatternTests
{
    [Fact]
    public void Assert_EqualityPatterns()
    {
        // Value equality
        Assert.Equal(42, 42);
        Assert.NotEqual(1, 2);

        // String equality
        Assert.Equal("hello", "hello");
        Assert.Equal("HELLO", "hello", ignoreCase: true);

        // Collection equality
        Assert.Equal(new[] { 1, 2, 3 }, new[] { 1, 2, 3 });
    }

    [Fact]
    public void Assert_BooleanPatterns()
    {
        Assert.True(1 < 2);
        Assert.False(1 > 2);
    }

    [Fact]
    public void Assert_NullPatterns()
    {
        string? nullStr = null;
        string nonNull = "hello";

        Assert.Null(nullStr);
        Assert.NotNull(nonNull);
    }

    [Fact]
    public void Assert_CollectionPatterns()
    {
        var list = new List<int> { 1, 2, 3, 4, 5 };

        Assert.Contains(3, list);
        Assert.DoesNotContain(6, list);
        Assert.All(list, item => Assert.True(item > 0));
        Assert.Single(list.Where(x => x == 3));
        Assert.Empty(list.Where(x => x > 100));

        // Collection contains element matching predicate
        Assert.Contains(list, item => item % 2 == 0);
    }

    [Fact]
    public void Assert_TypePatterns()
    {
        object obj = "hello";

        Assert.IsType<string>(obj);
        Assert.IsAssignableFrom<IComparable>(obj);
    }

    [Fact]
    public void Assert_RangePatterns()
    {
        double value = 3.14;

        Assert.InRange(value, 3.0, 4.0);
    }
}

// -------------------------------------------------------
// 5. Testing with Mocks (Manual Stub)
// -------------------------------------------------------

public class UserServiceTests
{
    [Fact]
    public void GetUser_ExistingId_ReturnsUser()
    {
        // Arrange — manual stub/mock
        var mockRepo = new StubUserRepository();
        mockRepo.Users[1] = new User(1, "Alice", "alice@test.com");
        var service = new UserService(mockRepo);

        // Act
        var user = service.GetUser(1);

        // Assert
        Assert.NotNull(user);
        Assert.Equal("Alice", user.Name);
    }

    [Fact]
    public void GetUser_NonExistentId_ReturnsNull()
    {
        var service = new UserService(new StubUserRepository());
        Assert.Null(service.GetUser(999));
    }

    [Fact]
    public void CreateUser_ValidInput_SavesUser()
    {
        var mockRepo = new StubUserRepository();
        var service = new UserService(mockRepo);

        service.CreateUser("Bob", "bob@test.com");

        Assert.Single(mockRepo.SavedUsers);
        Assert.Equal("Bob", mockRepo.SavedUsers[0].Name);
    }

    [Fact]
    public void CreateUser_EmptyName_ThrowsArgumentException()
    {
        var service = new UserService(new StubUserRepository());
        Assert.Throws<ArgumentException>(() => service.CreateUser("", "a@b.com"));
    }

    [Fact]
    public void CreateUser_InvalidEmail_ThrowsArgumentException()
    {
        var service = new UserService(new StubUserRepository());
        var ex = Assert.Throws<ArgumentException>(
            () => service.CreateUser("Bob", "invalid")
        );
        Assert.Contains("Invalid email", ex.Message);
    }
}

// Manual stub implementation for testing
public class StubUserRepository : IUserRepository
{
    public Dictionary<int, User> Users { get; } = new();
    public List<User> SavedUsers { get; } = new();

    public User? GetById(int id) => Users.GetValueOrDefault(id);
    public void Save(User user) => SavedUsers.Add(user);
}

// -------------------------------------------------------
// 6. [Theory] with MemberData and ClassData
// -------------------------------------------------------

public class MemberDataTests
{
    public static IEnumerable<object[]> AddTestData =>
        new List<object[]>
        {
            new object[] { 1, 2, 3 },
            new object[] { -1, -1, -2 },
            new object[] { 0, 0, 0 },
        };

    [Theory]
    [MemberData(nameof(AddTestData))]
    public void Add_MemberData_ReturnsExpected(int a, int b, int expected)
    {
        var calc = new Calculator();
        Assert.Equal(expected, calc.Add(a, b));
    }
}
*/

// ============================================================
// Runnable Demo (prints test-like output without xUnit runner)
// ============================================================

Console.WriteLine("=== xUnit Test Pattern Demonstrations ===");
Console.WriteLine("(See comments above for actual xUnit test code)\n");

var calc = new Calculator();

// Fact-style assertions
Console.WriteLine("[Fact] Add(3, 4) == 7: " + (calc.Add(3, 4) == 7 ? "PASS" : "FAIL"));
Console.WriteLine("[Fact] IsEven(4) == true: " + (calc.IsEven(4) ? "PASS" : "FAIL"));

try { calc.Divide(10, 0); Console.WriteLine("[Fact] Divide by zero: FAIL"); }
catch (DivideByZeroException) { Console.WriteLine("[Fact] Divide by zero throws: PASS"); }

// Theory-style parameterized
(int a, int b, int expected)[] testCases = { (1, 1, 2), (0, 0, 0), (-1, 1, 0), (100, -50, 50) };
foreach (var (a, b, expected) in testCases)
{
    bool pass = calc.Add(a, b) == expected;
    Console.WriteLine($"[Theory] Add({a}, {b}) == {expected}: {(pass ? "PASS" : "FAIL")}");
}

// String utils
Console.WriteLine($"\n[Fact] Reverse(\"hello\") == \"olleh\": {(StringUtils.Reverse("hello") == "olleh" ? "PASS" : "FAIL")}");
Console.WriteLine($"[Fact] IsPalindrome(\"racecar\"): {(StringUtils.IsPalindrome("racecar") ? "PASS" : "FAIL")}");
Console.WriteLine($"[Fact] WordCount(\"hello world\"): {(StringUtils.WordCount("hello world") == 2 ? "PASS" : "FAIL")}");

// Stub/mock pattern
var stubRepo = new StubUserRepository();
stubRepo.Users[1] = new User(1, "Alice", "alice@test.com");
var svc = new UserService(stubRepo);
Console.WriteLine($"\n[Fact] GetUser(1) returns Alice: {(svc.GetUser(1)?.Name == "Alice" ? "PASS" : "FAIL")}");
Console.WriteLine($"[Fact] GetUser(999) returns null: {(svc.GetUser(999) is null ? "PASS" : "FAIL")}");

Console.WriteLine("\nAll demonstrations complete. See source for full xUnit patterns.");

// Stub implementation used by runnable demo
public class StubUserRepository : IUserRepository
{
    public Dictionary<int, User> Users { get; } = new();
    public List<User> SavedUsers { get; } = new();
    public User? GetById(int id) => Users.GetValueOrDefault(id);
    public void Save(User user) => SavedUsers.Add(user);
}
