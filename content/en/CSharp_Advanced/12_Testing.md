# Testing

**Previous**: [Serialization](./11_Serialization.md) | **Next**: [Entity Framework Core](./13_Entity_Framework_Core.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Distinguish between unit tests, integration tests, and end-to-end tests
2. Set up and write tests with the xUnit framework
3. Use `[Fact]`, `[Theory]`, and `[InlineData]` attributes effectively
4. Apply the Arrange-Act-Assert pattern for clear, maintainable tests
5. Manage test lifecycle with class and collection fixtures
6. Create mock objects with NSubstitute for isolated unit testing
7. Test asynchronous code correctly
8. Write integration tests for ASP.NET Core using `WebApplicationFactory`
9. Measure and improve code coverage with Coverlet
10. Organize tests following best practices for naming, structure, and maintainability

---

Testing is not an afterthought — it is a core engineering practice that directly impacts code quality, refactoring confidence, and development speed. A well-tested codebase lets you change behavior fearlessly. This lesson covers the full spectrum of testing in C#: from isolated unit tests with xUnit and mocking, through integration tests with `WebApplicationFactory`, to coverage analysis. By the end, you will have a complete testing toolkit.

## 1. Testing Fundamentals

### 1.1 The Testing Pyramid

```
        /  E2E  \        Few, slow, expensive
       /----------\
      / Integration \    Medium count, moderate speed
     /----------------\
    /    Unit Tests     \  Many, fast, cheap
   /--------------------\
```

### 1.2 Unit Tests

Unit tests verify a single unit of behavior in isolation. They are fast (milliseconds), deterministic, and independent of external systems.

```csharp
// Unit under test
public class PriceCalculator
{
    public decimal CalculateTotal(decimal unitPrice, int quantity, decimal taxRate)
    {
        if (unitPrice < 0) throw new ArgumentException("Price cannot be negative");
        if (quantity < 0) throw new ArgumentException("Quantity cannot be negative");

        decimal subtotal = unitPrice * quantity;
        decimal tax = subtotal * taxRate;
        return Math.Round(subtotal + tax, 2);
    }
}
```

### 1.3 Integration Tests

Integration tests verify that multiple components work together correctly. They may involve databases, file systems, or HTTP endpoints.

### 1.4 End-to-End Tests

E2E tests verify the entire application from the user's perspective. They are the most realistic but also the slowest and most brittle.

## 2. xUnit Framework Setup

### 2.1 Creating a Test Project

```bash
# Create a test project
dotnet new xunit -n MyApp.Tests

# Add reference to the project under test
dotnet add MyApp.Tests/MyApp.Tests.csproj reference MyApp/MyApp.csproj

# Run tests
dotnet test

# Run with verbosity
dotnet test --verbosity normal

# Run specific tests
dotnet test --filter "ClassName=PriceCalculatorTests"
dotnet test --filter "DisplayName~Calculate"
```

### 2.2 Project Structure

```
MyApp/
├── MyApp/
│   ├── Models/
│   ├── Services/
│   └── MyApp.csproj
└── MyApp.Tests/
    ├── Models/
    │   └── PersonTests.cs
    ├── Services/
    │   └── PriceCalculatorTests.cs
    ├── Fixtures/
    │   └── DatabaseFixture.cs
    ├── Helpers/
    │   └── TestDataBuilder.cs
    └── MyApp.Tests.csproj
```

### 2.3 Test Project Dependencies

```xml
<!-- MyApp.Tests.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <TargetFramework>net8.0</TargetFramework>
    <IsPackable>false</IsPackable>
  </PropertyGroup>

  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="17.*" />
    <PackageReference Include="xunit" Version="2.*" />
    <PackageReference Include="xunit.runner.visualstudio" Version="2.*" />
    <PackageReference Include="NSubstitute" Version="5.*" />
    <PackageReference Include="coverlet.collector" Version="6.*" />
  </ItemGroup>

  <ItemGroup>
    <ProjectReference Include="..\MyApp\MyApp.csproj" />
  </ItemGroup>
</Project>
```

## 3. Test Methods: [Fact], [Theory], [InlineData]

### 3.1 [Fact] — Single Test Case

A `[Fact]` represents a test that is always true — a single, specific scenario.

```csharp
public class PriceCalculatorTests
{
    private readonly PriceCalculator _calculator = new();

    [Fact]
    public void CalculateTotal_WithValidInputs_ReturnsCorrectTotal()
    {
        // Arrange
        decimal unitPrice = 10.00m;
        int quantity = 3;
        decimal taxRate = 0.08m;

        // Act
        decimal total = _calculator.CalculateTotal(unitPrice, quantity, taxRate);

        // Assert
        Assert.Equal(32.40m, total);
    }

    [Fact]
    public void CalculateTotal_WithZeroQuantity_ReturnsZero()
    {
        decimal total = _calculator.CalculateTotal(25.00m, 0, 0.10m);
        Assert.Equal(0.00m, total);
    }

    [Fact]
    public void CalculateTotal_WithNegativePrice_ThrowsArgumentException()
    {
        Assert.Throws<ArgumentException>(() =>
            _calculator.CalculateTotal(-5.00m, 1, 0.10m));
    }
}
```

### 3.2 [Theory] with [InlineData] — Parameterized Tests

A `[Theory]` runs the same test logic with multiple sets of data.

```csharp
public class PriceCalculatorTheoryTests
{
    private readonly PriceCalculator _calculator = new();

    [Theory]
    [InlineData(10.00, 1, 0.00, 10.00)]   // No tax
    [InlineData(10.00, 1, 0.10, 11.00)]   // 10% tax
    [InlineData(10.00, 3, 0.08, 32.40)]   // Multiple items
    [InlineData(99.99, 1, 0.0725, 107.24)] // Complex tax rate
    [InlineData(0.00, 5, 0.10, 0.00)]     // Free item
    public void CalculateTotal_VariousInputs_ReturnsExpected(
        double price, int quantity, double tax, double expected)
    {
        decimal result = _calculator.CalculateTotal(
            (decimal)price, quantity, (decimal)tax);

        Assert.Equal((decimal)expected, result);
    }
}
```

### 3.3 [Theory] with [MemberData] and [ClassData]

```csharp
public class StringHelperTests
{
    // MemberData references a static property or method
    public static IEnumerable<object[]> TruncateTestData()
    {
        yield return new object[] { "Hello, World!", 5, "Hello..." };
        yield return new object[] { "Hi", 5, "Hi" };
        yield return new object[] { "", 5, "" };
        yield return new object[] { "Testing", 7, "Testing" };
    }

    [Theory]
    [MemberData(nameof(TruncateTestData))]
    public void Truncate_VariousInputs_ReturnsExpected(
        string input, int maxLength, string expected)
    {
        string result = StringHelper.Truncate(input, maxLength);
        Assert.Equal(expected, result);
    }
}

// ClassData for complex test data
public class CalculatorTestData : IEnumerable<object[]>
{
    public IEnumerator<object[]> GetEnumerator()
    {
        yield return new object[] { 1, 2, 3 };
        yield return new object[] { -1, -1, -2 };
        yield return new object[] { int.MaxValue, 0, int.MaxValue };
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

[Theory]
[ClassData(typeof(CalculatorTestData))]
public void Add_VariousInputs_ReturnsSum(int a, int b, int expected)
{
    Assert.Equal(expected, Calculator.Add(a, b));
}
```

## 4. Assertions

### 4.1 Equality and Comparison

```csharp
[Fact]
public void Assertion_Examples()
{
    // Equality
    Assert.Equal(42, GetAnswer());
    Assert.NotEqual(0, GetAnswer());

    // Approximate equality (for floating point)
    Assert.Equal(3.14159, Math.PI, precision: 5);

    // Reference equality
    var obj = new object();
    var same = obj;
    Assert.Same(obj, same);
    Assert.NotSame(obj, new object());

    // Null checks
    Assert.Null(GetNullableValue());
    Assert.NotNull(GetNonNullValue());

    // Boolean
    Assert.True(IsValid());
    Assert.False(IsExpired());
}
```

### 4.2 Collection Assertions

```csharp
[Fact]
public void Collection_Assertions()
{
    var numbers = new List<int> { 1, 2, 3, 4, 5 };

    // Contains
    Assert.Contains(3, numbers);
    Assert.DoesNotContain(6, numbers);

    // All elements match a predicate
    Assert.All(numbers, n => Assert.True(n > 0));

    // Count
    Assert.Equal(5, numbers.Count);
    Assert.Empty(new List<int>());
    Assert.NotEmpty(numbers);
    Assert.Single(new List<int> { 42 });

    // Contains with predicate
    Assert.Contains(numbers, n => n % 2 == 0);

    // Collection equality (order matters)
    Assert.Equal(new[] { 1, 2, 3, 4, 5 }, numbers);
}
```

### 4.3 String Assertions

```csharp
[Fact]
public void String_Assertions()
{
    string greeting = "Hello, World!";

    Assert.Equal("Hello, World!", greeting);
    Assert.StartsWith("Hello", greeting);
    Assert.EndsWith("World!", greeting);
    Assert.Contains("World", greeting);
    Assert.DoesNotContain("Goodbye", greeting);
    Assert.Matches(@"Hello,\s\w+!", greeting);  // Regex
    Assert.Equal("hello, world!", greeting, ignoreCase: true);
}
```

### 4.4 Exception Assertions

```csharp
[Fact]
public void Exception_Assertions()
{
    // Assert specific exception type
    var ex = Assert.Throws<ArgumentNullException>(() =>
        ProcessData(null!));
    Assert.Equal("data", ex.ParamName);

    // Assert exception message
    var ex2 = Assert.Throws<InvalidOperationException>(() =>
        DoSomethingInvalid());
    Assert.Contains("not allowed", ex2.Message);

    // Assert async exception
    // (covered in section 8)
}
```

### 4.5 Type Assertions

```csharp
[Fact]
public void Type_Assertions()
{
    object obj = new List<int>();

    Assert.IsType<List<int>>(obj);          // Exact type
    Assert.IsAssignableFrom<IList<int>>(obj); // Type or subtype

    var result = Assert.IsType<List<int>>(obj);
    Assert.Empty(result); // Can use the typed result
}
```

## 5. Test Lifecycle

### 5.1 Constructor and Dispose (Per-Test Setup/Teardown)

xUnit creates a new instance of the test class for every test method. The constructor serves as setup, `IDisposable.Dispose` as teardown.

```csharp
public class FileProcessorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly FileProcessor _processor;

    public FileProcessorTests()
    {
        // Runs before EACH test
        _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);
        _processor = new FileProcessor(_tempDir);
    }

    [Fact]
    public void ProcessFile_CreatesOutput()
    {
        File.WriteAllText(Path.Combine(_tempDir, "input.txt"), "data");
        _processor.Process("input.txt");
        Assert.True(File.Exists(Path.Combine(_tempDir, "output.txt")));
    }

    [Fact]
    public void ProcessFile_EmptyInput_ThrowsException()
    {
        File.WriteAllText(Path.Combine(_tempDir, "empty.txt"), "");
        Assert.Throws<InvalidOperationException>(() =>
            _processor.Process("empty.txt"));
    }

    public void Dispose()
    {
        // Runs after EACH test
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}
```

### 5.2 IClassFixture — Shared Setup Across a Test Class

When setup is expensive (e.g., starting a database), share it across all tests in a class.

```csharp
public class DatabaseFixture : IDisposable
{
    public string ConnectionString { get; }

    public DatabaseFixture()
    {
        // Expensive setup — runs ONCE for the entire test class
        ConnectionString = "Server=localhost;Database=TestDb;...";
        InitializeDatabase();
    }

    private void InitializeDatabase()
    {
        // Create tables, seed data
    }

    public void Dispose()
    {
        // Cleanup — runs ONCE after all tests in the class
        DropDatabase();
    }

    private void DropDatabase() { }
}

public class UserRepositoryTests : IClassFixture<DatabaseFixture>
{
    private readonly DatabaseFixture _fixture;

    public UserRepositoryTests(DatabaseFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void GetUser_ExistingId_ReturnsUser()
    {
        var repo = new UserRepository(_fixture.ConnectionString);
        var user = repo.GetById(1);
        Assert.NotNull(user);
    }
}
```

### 5.3 ICollectionFixture — Shared Across Multiple Test Classes

```csharp
// Define the collection
[CollectionDefinition("Database")]
public class DatabaseCollection : ICollectionFixture<DatabaseFixture>
{
    // This class has no code — it just links the fixture to the collection name
}

// Any test class in this collection shares the fixture
[Collection("Database")]
public class OrderRepositoryTests
{
    private readonly DatabaseFixture _fixture;

    public OrderRepositoryTests(DatabaseFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void CreateOrder_ValidData_Succeeds()
    {
        var repo = new OrderRepository(_fixture.ConnectionString);
        var order = new Order { CustomerId = 1, Total = 99.99m };
        repo.Create(order);
        Assert.True(order.Id > 0);
    }
}
```

## 6. Arrange-Act-Assert Pattern

The AAA pattern provides a consistent structure for every test.

```csharp
public class ShoppingCartTests
{
    [Fact]
    public void AddItem_NewItem_IncreasesItemCount()
    {
        // Arrange — set up the test scenario
        var cart = new ShoppingCart();
        var item = new CartItem("SKU-001", "Widget", 9.99m, 2);

        // Act — perform the action being tested
        cart.AddItem(item);

        // Assert — verify the expected outcome
        Assert.Equal(1, cart.ItemCount);
        Assert.Equal(19.98m, cart.Total);
    }

    [Fact]
    public void AddItem_ExistingItem_IncreasesQuantity()
    {
        // Arrange
        var cart = new ShoppingCart();
        cart.AddItem(new CartItem("SKU-001", "Widget", 9.99m, 1));

        // Act
        cart.AddItem(new CartItem("SKU-001", "Widget", 9.99m, 2));

        // Assert
        Assert.Equal(1, cart.ItemCount);      // Still one distinct item
        Assert.Equal(3, cart.GetQuantity("SKU-001")); // Quantity combined
        Assert.Equal(29.97m, cart.Total);
    }

    [Fact]
    public void RemoveItem_ExistingItem_DecreasesTotal()
    {
        // Arrange
        var cart = new ShoppingCart();
        cart.AddItem(new CartItem("SKU-001", "Widget", 10.00m, 3));
        cart.AddItem(new CartItem("SKU-002", "Gadget", 25.00m, 1));

        // Act
        cart.RemoveItem("SKU-001");

        // Assert
        Assert.Equal(1, cart.ItemCount);
        Assert.Equal(25.00m, cart.Total);
        Assert.DoesNotContain(cart.Items, i => i.Sku == "SKU-001");
    }
}
```

## 7. Mocking with NSubstitute

Mocking replaces real dependencies with controlled substitutes, isolating the unit under test.

### 7.1 Basic Mocking

```csharp
using NSubstitute;

public class OrderServiceTests
{
    private readonly IOrderRepository _repository;
    private readonly IEmailSender _emailSender;
    private readonly ILogger _logger;
    private readonly OrderService _service;

    public OrderServiceTests()
    {
        // Create substitutes (mocks)
        _repository = Substitute.For<IOrderRepository>();
        _emailSender = Substitute.For<IEmailSender>();
        _logger = Substitute.For<ILogger>();

        // Inject mocks into the service
        _service = new OrderService(_repository, _emailSender, _logger);
    }

    [Fact]
    public void PlaceOrder_ValidOrder_SavesAndSendsEmail()
    {
        // Arrange
        var order = new Order { Id = 1, CustomerEmail = "user@example.com" };

        // Act
        _service.PlaceOrder(order);

        // Assert — verify interactions
        _repository.Received(1).Save(order);
        _emailSender.Received(1).Send(
            "user@example.com",
            Arg.Any<string>(),
            Arg.Is<string>(body => body.Contains("Order 1")));
    }
}
```

### 7.2 Configuring Return Values

```csharp
[Fact]
public void GetOrder_ExistingId_ReturnsOrder()
{
    // Arrange
    var expected = new Order { Id = 42, CustomerEmail = "alice@test.com" };
    _repository.GetById(42).Returns(expected);

    // Act
    Order? result = _service.GetOrder(42);

    // Assert
    Assert.NotNull(result);
    Assert.Equal(42, result.Id);
    Assert.Equal("alice@test.com", result.CustomerEmail);
}

[Fact]
public void GetOrder_NonExistentId_ReturnsNull()
{
    _repository.GetById(999).Returns((Order?)null);

    Order? result = _service.GetOrder(999);

    Assert.Null(result);
}
```

### 7.3 Configuring Exceptions

```csharp
[Fact]
public void PlaceOrder_RepositoryThrows_LogsError()
{
    // Arrange
    var order = new Order { Id = 1 };
    _repository.When(r => r.Save(Arg.Any<Order>()))
        .Do(_ => throw new InvalidOperationException("DB connection failed"));

    // Act & Assert
    Assert.Throws<InvalidOperationException>(() => _service.PlaceOrder(order));
    _logger.Received(1).Log(Arg.Is<string>(msg => msg.Contains("Error")));
}
```

### 7.4 Argument Matchers

```csharp
[Fact]
public void ProcessOrders_FiltersCorrectly()
{
    // Arg.Any<T>() — matches any value of type T
    _repository.Save(Arg.Any<Order>());

    // Arg.Is<T>(predicate) — matches values satisfying the predicate
    _emailSender.Received().Send(
        Arg.Is<string>(email => email.Contains("@")),
        Arg.Any<string>(),
        Arg.Any<string>());

    // Arg.Do<T>(action) — captures the argument for inspection
    Order? capturedOrder = null;
    _repository.When(r => r.Save(Arg.Do<Order>(o => capturedOrder = o)))
        .Do(_ => { });

    _service.PlaceOrder(new Order { Id = 99 });
    Assert.Equal(99, capturedOrder?.Id);
}
```

## 8. Testing Async Code

### 8.1 Async Fact and Theory

```csharp
public class AsyncServiceTests
{
    private readonly IHttpClientWrapper _httpClient;
    private readonly DataFetcher _fetcher;

    public AsyncServiceTests()
    {
        _httpClient = Substitute.For<IHttpClientWrapper>();
        _fetcher = new DataFetcher(_httpClient);
    }

    [Fact]
    public async Task FetchData_SuccessfulResponse_ReturnsData()
    {
        // Arrange
        _httpClient.GetStringAsync("https://api.example.com/data")
            .Returns(Task.FromResult("""{"name":"Alice","age":30}"""));

        // Act
        Person? result = await _fetcher.FetchPersonAsync("https://api.example.com/data");

        // Assert
        Assert.NotNull(result);
        Assert.Equal("Alice", result.Name);
    }

    [Fact]
    public async Task FetchData_Timeout_ThrowsOperationCanceledException()
    {
        // Arrange
        _httpClient.GetStringAsync(Arg.Any<string>())
            .Returns<string>(x => throw new TaskCanceledException("Timeout"));

        // Act & Assert
        await Assert.ThrowsAsync<TaskCanceledException>(() =>
            _fetcher.FetchPersonAsync("https://api.example.com/data"));
    }

    [Theory]
    [InlineData("https://api.example.com/users/1", "Alice")]
    [InlineData("https://api.example.com/users/2", "Bob")]
    public async Task FetchData_DifferentUrls_ReturnsCorrectPerson(
        string url, string expectedName)
    {
        _httpClient.GetStringAsync(url)
            .Returns(Task.FromResult($$$"""{"name":"{{{expectedName}}}","age":30}"""));

        Person? result = await _fetcher.FetchPersonAsync(url);

        Assert.Equal(expectedName, result?.Name);
    }
}
```

### 8.2 Testing Cancellation

```csharp
[Fact]
public async Task LongRunningOperation_WhenCancelled_ThrowsOperationCanceled()
{
    var service = new LongRunningService();
    using var cts = new CancellationTokenSource();

    // Cancel after a short delay
    cts.CancelAfter(TimeSpan.FromMilliseconds(50));

    await Assert.ThrowsAsync<OperationCanceledException>(() =>
        service.ProcessAsync(cts.Token));
}
```

### 8.3 Testing IAsyncEnumerable

```csharp
[Fact]
public async Task StreamData_ReturnsAllItems()
{
    var service = new StreamingService();

    var results = new List<int>();
    await foreach (int item in service.GenerateAsync(5))
    {
        results.Add(item);
    }

    Assert.Equal(5, results.Count);
    Assert.Equal(new[] { 0, 1, 2, 3, 4 }, results);
}
```

## 9. Integration Testing with WebApplicationFactory

`WebApplicationFactory` creates an in-memory test server for your ASP.NET Core application, enabling real HTTP-level integration tests without a network.

### 9.1 Basic Setup

```csharp
using Microsoft.AspNetCore.Mvc.Testing;
using System.Net;
using System.Net.Http.Json;

public class ApiIntegrationTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ApiIntegrationTests(WebApplicationFactory<Program> factory)
    {
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task GetWeather_ReturnsOk()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/weather");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var forecasts = await response.Content.ReadFromJsonAsync<List<WeatherForecast>>();
        Assert.NotNull(forecasts);
        Assert.NotEmpty(forecasts);
    }

    [Fact]
    public async Task PostOrder_ValidData_ReturnsCreated()
    {
        var order = new { CustomerEmail = "test@example.com", Total = 99.99 };

        HttpResponseMessage response = await _client.PostAsJsonAsync("/api/orders", order);

        Assert.Equal(HttpStatusCode.Created, response.StatusCode);
    }

    [Fact]
    public async Task GetOrder_NotFound_Returns404()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/orders/99999");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}
```

### 9.2 Custom Factory with Test Services

```csharp
public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureServices(services =>
        {
            // Remove the real database context
            var descriptor = services.SingleOrDefault(
                d => d.ServiceType == typeof(DbContextOptions<AppDbContext>));
            if (descriptor != null)
                services.Remove(descriptor);

            // Add an in-memory database for testing
            services.AddDbContext<AppDbContext>(options =>
                options.UseInMemoryDatabase("TestDb"));

            // Replace real email sender with a fake
            services.AddSingleton<IEmailSender, FakeEmailSender>();
        });

        builder.UseEnvironment("Testing");
    }
}

public class FakeEmailSender : IEmailSender
{
    public List<(string To, string Subject, string Body)> SentEmails { get; } = new();

    public void Send(string to, string subject, string body)
    {
        SentEmails.Add((to, subject, body));
    }
}

// Use the custom factory
public class OrderApiTests : IClassFixture<CustomWebApplicationFactory>
{
    private readonly HttpClient _client;
    private readonly CustomWebApplicationFactory _factory;

    public OrderApiTests(CustomWebApplicationFactory factory)
    {
        _factory = factory;
        _client = factory.CreateClient();
    }

    [Fact]
    public async Task PlaceOrder_SendsConfirmationEmail()
    {
        // Arrange
        var order = new { CustomerEmail = "test@test.com", Total = 50.00 };

        // Act
        await _client.PostAsJsonAsync("/api/orders", order);

        // Assert — check the fake email sender
        using var scope = _factory.Services.CreateScope();
        var emailSender = scope.ServiceProvider.GetRequiredService<IEmailSender>()
            as FakeEmailSender;

        Assert.NotNull(emailSender);
        Assert.Single(emailSender!.SentEmails);
        Assert.Equal("test@test.com", emailSender.SentEmails[0].To);
    }
}
```

## 10. Code Coverage with Coverlet

### 10.1 Running Coverage

```bash
# Collect coverage
dotnet test --collect:"XPlat Code Coverage"

# Generate HTML report (install reportgenerator first)
dotnet tool install -g dotnet-reportgenerator-globaltool

reportgenerator \
    -reports:"**/coverage.cobertura.xml" \
    -targetdir:"coveragereport" \
    -reporttypes:Html

# Open the report
open coveragereport/index.html
```

### 10.2 Coverage Configuration

```xml
<!-- In the test project's .csproj or a coverlet.runsettings file -->
<PropertyGroup>
    <CollectCoverage>true</CollectCoverage>
    <CoverletOutputFormat>cobertura</CoverletOutputFormat>
    <Threshold>80</Threshold>
    <ThresholdType>line,branch</ThresholdType>
    <ThresholdStat>total</ThresholdStat>
    <ExcludeByFile>**/Migrations/**</ExcludeByFile>
</PropertyGroup>
```

### 10.3 Excluding Code from Coverage

```csharp
using System.Diagnostics.CodeAnalysis;

[ExcludeFromCodeCoverage]
public class AutoGeneratedDto
{
    public string Name { get; set; } = "";
    public int Value { get; set; }
}

public class MyService
{
    [ExcludeFromCodeCoverage] // Exclude specific methods
    public void DebugDump()
    {
        // Debug-only code, not worth testing
    }
}
```

## 11. Test Organization Best Practices

### 11.1 Naming Conventions

```csharp
// Pattern: MethodName_Scenario_ExpectedBehavior
public class UserServiceTests
{
    [Fact]
    public void CreateUser_ValidEmail_ReturnsNewUser() { }

    [Fact]
    public void CreateUser_DuplicateEmail_ThrowsConflictException() { }

    [Fact]
    public void CreateUser_NullName_ThrowsArgumentNullException() { }

    [Fact]
    public void GetUser_ExistingId_ReturnsUser() { }

    [Fact]
    public void GetUser_DeletedUser_ReturnsNull() { }
}
```

### 11.2 Test Data Builders

```csharp
public class OrderBuilder
{
    private int _id = 1;
    private string _email = "test@example.com";
    private decimal _total = 100.00m;
    private OrderStatus _status = OrderStatus.Pending;
    private readonly List<OrderItem> _items = new();

    public OrderBuilder WithId(int id) { _id = id; return this; }
    public OrderBuilder WithEmail(string email) { _email = email; return this; }
    public OrderBuilder WithTotal(decimal total) { _total = total; return this; }
    public OrderBuilder WithStatus(OrderStatus status) { _status = status; return this; }

    public OrderBuilder WithItem(string sku, decimal price, int qty = 1)
    {
        _items.Add(new OrderItem { Sku = sku, Price = price, Quantity = qty });
        return this;
    }

    public Order Build() => new()
    {
        Id = _id,
        CustomerEmail = _email,
        Total = _total,
        Status = _status,
        Items = _items
    };
}

// Usage in tests
[Fact]
public void ProcessOrder_HighValueOrder_AppliesDiscount()
{
    var order = new OrderBuilder()
        .WithTotal(500.00m)
        .WithItem("SKU-001", 250.00m, 2)
        .Build();

    _service.Process(order);

    Assert.True(order.DiscountApplied);
}
```

### 11.3 Avoiding Common Mistakes

```csharp
// BAD: Test depends on execution order
// BAD: Test depends on external state (database, file system, network)
// BAD: Multiple acts in one test
// BAD: Testing implementation details instead of behavior
// BAD: Overly specific assertions that break on irrelevant changes

// GOOD: Each test is independent and self-contained
[Fact]
public void CalculateDiscount_GoldMember_Gets20Percent()
{
    // Arrange — all setup is within the test
    var calculator = new DiscountCalculator();
    var customer = new Customer { Tier = MemberTier.Gold };
    decimal originalPrice = 100.00m;

    // Act — one logical action
    decimal discount = calculator.Calculate(customer, originalPrice);

    // Assert — verify behavior, not internals
    Assert.Equal(20.00m, discount);
}
```

## 12. Practical Example: Testing a Service Layer

This example brings everything together: unit tests with mocks, parameterized theories, async testing, and clear test organization.

```csharp
// --- Service Under Test ---
public interface IProductRepository
{
    Task<Product?> GetByIdAsync(int id);
    Task<List<Product>> GetAllAsync();
    Task<Product> CreateAsync(Product product);
    Task UpdateAsync(Product product);
    Task DeleteAsync(int id);
}

public interface IPricingEngine
{
    decimal CalculatePrice(Product product, string? couponCode);
}

public class ProductService
{
    private readonly IProductRepository _repository;
    private readonly IPricingEngine _pricingEngine;

    public ProductService(IProductRepository repository, IPricingEngine pricingEngine)
    {
        _repository = repository;
        _pricingEngine = pricingEngine;
    }

    public async Task<ProductDto?> GetProductAsync(int id, string? couponCode = null)
    {
        Product? product = await _repository.GetByIdAsync(id);
        if (product is null) return null;

        decimal finalPrice = _pricingEngine.CalculatePrice(product, couponCode);
        return new ProductDto(product.Id, product.Name, product.BasePrice, finalPrice);
    }

    public async Task<ProductDto> CreateProductAsync(CreateProductRequest request)
    {
        if (string.IsNullOrWhiteSpace(request.Name))
            throw new ArgumentException("Product name is required");

        if (request.BasePrice <= 0)
            throw new ArgumentException("Price must be positive");

        var product = new Product
        {
            Name = request.Name,
            BasePrice = request.BasePrice,
            Category = request.Category
        };

        Product created = await _repository.CreateAsync(product);
        decimal price = _pricingEngine.CalculatePrice(created, null);
        return new ProductDto(created.Id, created.Name, created.BasePrice, price);
    }

    public async Task<List<ProductDto>> GetAllProductsAsync()
    {
        var products = await _repository.GetAllAsync();
        return products.Select(p => new ProductDto(
            p.Id, p.Name, p.BasePrice,
            _pricingEngine.CalculatePrice(p, null))).ToList();
    }
}

public record ProductDto(int Id, string Name, decimal BasePrice, decimal FinalPrice);
public record CreateProductRequest(string Name, decimal BasePrice, string Category);
public class Product
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public decimal BasePrice { get; set; }
    public string Category { get; set; } = "";
}
```

```csharp
// --- Complete Test Suite ---
public class ProductServiceTests
{
    private readonly IProductRepository _repository;
    private readonly IPricingEngine _pricingEngine;
    private readonly ProductService _service;

    public ProductServiceTests()
    {
        _repository = Substitute.For<IProductRepository>();
        _pricingEngine = Substitute.For<IPricingEngine>();
        _service = new ProductService(_repository, _pricingEngine);
    }

    // --- GetProductAsync Tests ---

    [Fact]
    public async Task GetProduct_ExistingProduct_ReturnsDto()
    {
        // Arrange
        var product = new Product { Id = 1, Name = "Widget", BasePrice = 25.00m };
        _repository.GetByIdAsync(1).Returns(product);
        _pricingEngine.CalculatePrice(product, null).Returns(25.00m);

        // Act
        ProductDto? result = await _service.GetProductAsync(1);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(1, result.Id);
        Assert.Equal("Widget", result.Name);
        Assert.Equal(25.00m, result.FinalPrice);
    }

    [Fact]
    public async Task GetProduct_WithCoupon_AppliesDiscount()
    {
        // Arrange
        var product = new Product { Id = 1, Name = "Widget", BasePrice = 100.00m };
        _repository.GetByIdAsync(1).Returns(product);
        _pricingEngine.CalculatePrice(product, "SAVE20").Returns(80.00m);

        // Act
        ProductDto? result = await _service.GetProductAsync(1, "SAVE20");

        // Assert
        Assert.NotNull(result);
        Assert.Equal(100.00m, result.BasePrice);
        Assert.Equal(80.00m, result.FinalPrice);
    }

    [Fact]
    public async Task GetProduct_NonExistentId_ReturnsNull()
    {
        _repository.GetByIdAsync(999).Returns((Product?)null);

        ProductDto? result = await _service.GetProductAsync(999);

        Assert.Null(result);
    }

    // --- CreateProductAsync Tests ---

    [Fact]
    public async Task CreateProduct_ValidRequest_ReturnsNewProduct()
    {
        // Arrange
        var request = new CreateProductRequest("Gadget", 49.99m, "Electronics");
        _repository.CreateAsync(Arg.Any<Product>()).Returns(callInfo =>
        {
            var p = callInfo.Arg<Product>();
            p.Id = 42;
            return p;
        });
        _pricingEngine.CalculatePrice(Arg.Any<Product>(), null).Returns(49.99m);

        // Act
        ProductDto result = await _service.CreateProductAsync(request);

        // Assert
        Assert.Equal(42, result.Id);
        Assert.Equal("Gadget", result.Name);
        await _repository.Received(1).CreateAsync(Arg.Is<Product>(p =>
            p.Name == "Gadget" && p.BasePrice == 49.99m));
    }

    [Theory]
    [InlineData("", 10.00)]
    [InlineData("  ", 10.00)]
    [InlineData(null, 10.00)]
    public async Task CreateProduct_EmptyName_ThrowsArgumentException(
        string? name, decimal price)
    {
        var request = new CreateProductRequest(name!, price, "Category");

        await Assert.ThrowsAsync<ArgumentException>(() =>
            _service.CreateProductAsync(request));

        await _repository.DidNotReceive().CreateAsync(Arg.Any<Product>());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-99.99)]
    public async Task CreateProduct_InvalidPrice_ThrowsArgumentException(decimal price)
    {
        var request = new CreateProductRequest("Valid Name", price, "Category");

        await Assert.ThrowsAsync<ArgumentException>(() =>
            _service.CreateProductAsync(request));
    }

    // --- GetAllProductsAsync Tests ---

    [Fact]
    public async Task GetAllProducts_ReturnsAllWithPricing()
    {
        // Arrange
        var products = new List<Product>
        {
            new() { Id = 1, Name = "A", BasePrice = 10.00m },
            new() { Id = 2, Name = "B", BasePrice = 20.00m }
        };
        _repository.GetAllAsync().Returns(products);
        _pricingEngine.CalculatePrice(Arg.Any<Product>(), null)
            .Returns(callInfo => callInfo.Arg<Product>().BasePrice * 1.1m);

        // Act
        List<ProductDto> results = await _service.GetAllProductsAsync();

        // Assert
        Assert.Equal(2, results.Count);
        Assert.Equal(11.00m, results[0].FinalPrice);
        Assert.Equal(22.00m, results[1].FinalPrice);
    }

    [Fact]
    public async Task GetAllProducts_EmptyRepository_ReturnsEmptyList()
    {
        _repository.GetAllAsync().Returns(new List<Product>());

        List<ProductDto> results = await _service.GetAllProductsAsync();

        Assert.Empty(results);
    }
}
```

## 13. Practice Problems

1. **Calculator Test Suite**: Write a comprehensive test suite for a `ScientificCalculator` class with methods for `Add`, `Subtract`, `Multiply`, `Divide`, `Power`, `SquareRoot`, and `Factorial`. Use `[Theory]` with `[InlineData]` for at least 5 test cases per method. Include edge cases: division by zero, negative square root, overflow, and factorial of negative numbers. Aim for 100% branch coverage.

2. **Mock Verification**: Create an `INotificationService` with `SendEmail`, `SendSms`, and `SendPush` methods. Build a `UserRegistrationService` that calls different notification methods based on user preferences. Write tests using NSubstitute that verify: the correct notification method is called, arguments are correct, no unexpected calls are made, and errors in one notification channel don't prevent others.

3. **Integration Test Suite**: Using `WebApplicationFactory`, write integration tests for a simple REST API with CRUD endpoints for a `Book` entity (`GET /api/books`, `GET /api/books/{id}`, `POST /api/books`, `PUT /api/books/{id}`, `DELETE /api/books/{id}`). Replace the database with an in-memory provider. Test happy paths, validation errors, not-found scenarios, and concurrent access.

4. **Test Data Builder Library**: Create a fluent test data builder system. Implement builders for `Customer`, `Order`, `OrderItem`, and `Address` that support: default values, method chaining, nested builders (e.g., `OrderBuilder.WithCustomer(c => c.WithName("Alice"))`), and a `BuildMany(int count)` method that generates unique instances.

5. **Async Retry Tester**: Write a `RetryService` that retries a `Func<Task<T>>` up to N times with exponential backoff. Write a test suite that: mocks the inner function to fail K times then succeed, verifies the correct number of retries, verifies the delay between retries increases exponentially, tests cancellation during retry wait, and tests the case where all retries are exhausted.
