/*
 * Exercises for Lesson 12: Testing
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Linq;

// ---------------------------------------------------------------------------
// Exercise 1: Unit test framework basics — Assert methods
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Assert Methods ===");

    // Simulated test assertions
    Assert.AreEqual(4, Calculator.Add(2, 2), "2+2 should be 4");
    Assert.AreEqual(0, Calculator.Add(-1, 1), "-1+1 should be 0");
    Assert.AreEqual(-5, Calculator.Subtract(3, 8), "3-8 should be -5");
    Assert.IsTrue(Calculator.IsPositive(1), "1 is positive");
    Assert.IsFalse(Calculator.IsPositive(-1), "-1 is not positive");
    Assert.AreNotEqual(0, Calculator.Multiply(3, 4), "3*4 is not 0");

    Console.WriteLine("  All basic assertions passed!");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Testing exceptions
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Exception Testing ===");

    Assert.Throws<DivideByZeroException>(() => Calculator.Divide(10, 0),
        "Division by zero should throw");

    Assert.Throws<ArgumentException>(() => Calculator.Factorial(-1),
        "Negative factorial should throw");

    Assert.AreEqual(120, Calculator.Factorial(5), "5! = 120");
    Assert.AreEqual(1, Calculator.Factorial(0), "0! = 1");

    Console.WriteLine("  All exception tests passed!");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Parameterized tests — test multiple inputs
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Parameterized Tests ===");

    var testCases = new (string Input, bool Expected)[]
    {
        ("", false),
        ("a", false),
        ("aa", true),
        ("aba", true),
        ("abba", true),
        ("abc", false),
        ("racecar", true),
        ("A man a plan a canal Panama", false), // case/space sensitive
    };

    int passed = 0;
    foreach (var (input, expected) in testCases)
    {
        bool result = StringUtils.IsPalindrome(input);
        bool ok = result == expected;
        if (ok) passed++;
        Console.WriteLine($"  {(ok ? "PASS" : "FAIL")}: IsPalindrome(\"{input}\") = {result} (expected {expected})");
    }
    Console.WriteLine($"  {passed}/{testCases.Length} passed");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Mock setup — test service with fake dependencies
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Mock Dependencies ===");

    // Create mock/fake implementations
    var mockRepo = new MockUserRepository();
    mockRepo.SetupFind("alice", new UserDto("alice", "Alice Smith", "alice@test.com"));
    mockRepo.SetupFind("bob", null); // simulate not found

    var mockNotifier = new MockNotifier();
    var service = new AccountService(mockRepo, mockNotifier);

    // Test: successful lookup
    var result1 = service.GetUserProfile("alice");
    Assert.AreEqual("Alice Smith", result1?.DisplayName, "Should find Alice");

    // Test: not found
    var result2 = service.GetUserProfile("bob");
    Assert.IsTrue(result2 == null, "Bob should not be found");

    // Test: notification sent on password reset
    service.RequestPasswordReset("alice");
    Assert.AreEqual(1, mockNotifier.SentMessages.Count, "One notification sent");
    Assert.IsTrue(mockNotifier.SentMessages[0].Contains("alice"), "Notification mentions user");

    Console.WriteLine($"  Mock repo calls: {mockRepo.FindCallCount}");
    Console.WriteLine($"  Notifications sent: {mockNotifier.SentMessages.Count}");
    Console.WriteLine("  All mock tests passed!");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Test-driven design — implement from tests
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: TDD — Shopping Cart ===");

    var cart = new ShoppingCart();

    // Test empty cart
    Assert.AreEqual(0m, cart.Total, "Empty cart total is 0");
    Assert.AreEqual(0, cart.ItemCount, "Empty cart count is 0");

    // Test adding items
    cart.AddItem("Widget", 9.99m, 2);
    Assert.AreEqual(19.98m, cart.Total, "2 widgets = 19.98");
    Assert.AreEqual(1, cart.ItemCount, "1 line item");

    // Test adding more items
    cart.AddItem("Gadget", 24.99m, 1);
    Assert.AreEqual(44.97m, cart.Total, "Total with gadget");
    Assert.AreEqual(2, cart.ItemCount, "2 line items");

    // Test removing item
    cart.RemoveItem("Widget");
    Assert.AreEqual(24.99m, cart.Total, "After removal");
    Assert.AreEqual(1, cart.ItemCount, "1 item left");

    // Test discount
    cart.ApplyDiscount(0.10m); // 10%
    Assert.AreEqual(22.491m, cart.Total, "10% discount applied");

    Console.WriteLine("  All TDD tests passed!");
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

static class Calculator
{
    public static int Add(int a, int b) => a + b;
    public static int Subtract(int a, int b) => a - b;
    public static int Multiply(int a, int b) => a * b;
    public static int Divide(int a, int b) =>
        b == 0 ? throw new DivideByZeroException() : a / b;
    public static bool IsPositive(int n) => n > 0;
    public static long Factorial(int n)
    {
        if (n < 0) throw new ArgumentException("Negative input");
        long result = 1;
        for (int i = 2; i <= n; i++) result *= i;
        return result;
    }
}

static class StringUtils
{
    public static bool IsPalindrome(string s)
    {
        if (s.Length <= 1) return s.Length == 1;
        for (int i = 0; i < s.Length / 2; i++)
            if (s[i] != s[s.Length - 1 - i]) return false;
        return true;
    }
}

// Simple assertion helper (mimics test framework)
static class Assert
{
    public static void AreEqual<T>(T expected, T actual, string msg = "")
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new Exception($"ASSERT FAIL: expected={expected}, actual={actual}. {msg}");
    }
    public static void AreNotEqual<T>(T notExpected, T actual, string msg = "")
    {
        if (EqualityComparer<T>.Default.Equals(notExpected, actual))
            throw new Exception($"ASSERT FAIL: values should differ. {msg}");
    }
    public static void IsTrue(bool condition, string msg = "")
    { if (!condition) throw new Exception($"ASSERT FAIL: expected true. {msg}"); }
    public static void IsFalse(bool condition, string msg = "")
    { if (condition) throw new Exception($"ASSERT FAIL: expected false. {msg}"); }
    public static void Throws<TEx>(Action action, string msg = "") where TEx : Exception
    {
        try { action(); throw new Exception($"ASSERT FAIL: expected {typeof(TEx).Name}. {msg}"); }
        catch (TEx) { /* expected */ }
    }
}

record UserDto(string Id, string DisplayName, string Email);

interface IUserRepository { UserDto? FindById(string id); }
interface INotifier { void Send(string message); }

class MockUserRepository : IUserRepository
{
    private readonly Dictionary<string, UserDto?> _data = new();
    public int FindCallCount { get; private set; }
    public void SetupFind(string id, UserDto? result) => _data[id] = result;
    public UserDto? FindById(string id) { FindCallCount++; return _data.GetValueOrDefault(id); }
}

class MockNotifier : INotifier
{
    public List<string> SentMessages { get; } = new();
    public void Send(string message) => SentMessages.Add(message);
}

class AccountService
{
    private readonly IUserRepository _repo;
    private readonly INotifier _notifier;
    public AccountService(IUserRepository repo, INotifier notifier) { _repo = repo; _notifier = notifier; }
    public UserDto? GetUserProfile(string id) => _repo.FindById(id);
    public void RequestPasswordReset(string id)
    {
        var user = _repo.FindById(id);
        if (user != null) _notifier.Send($"Password reset requested for {user.Id}");
    }
}

class ShoppingCart
{
    private readonly List<CartItem> _items = new();
    private decimal _discount = 0m;
    public decimal Total => _items.Sum(i => i.Price * i.Quantity) * (1 - _discount);
    public int ItemCount => _items.Count;
    public void AddItem(string name, decimal price, int qty) => _items.Add(new CartItem(name, price, qty));
    public void RemoveItem(string name) => _items.RemoveAll(i => i.Name == name);
    public void ApplyDiscount(decimal rate) => _discount = rate;
}

record CartItem(string Name, decimal Price, int Quantity);
