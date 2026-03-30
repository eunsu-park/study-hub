# 테스팅

**이전**: [직렬화](./11_Serialization.md) | **다음**: [Entity Framework Core](./13_Entity_Framework_Core.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 단위 테스트, 통합 테스트, 종단 간 테스트를 구분하기
2. xUnit 프레임워크로 테스트를 설정하고 작성하기
3. `[Fact]`, `[Theory]`, `[InlineData]` 어트리뷰트를 효과적으로 사용하기
4. 명확하고 유지보수하기 쉬운 테스트를 위한 Arrange-Act-Assert 패턴 적용하기
5. 클래스와 컬렉션 픽스처로 테스트 생명주기 관리하기
6. NSubstitute로 격리된 단위 테스트를 위한 모의 객체 만들기
7. 비동기 코드를 올바르게 테스트하기
8. `WebApplicationFactory`를 사용하여 ASP.NET Core 통합 테스트 작성하기
9. Coverlet으로 코드 커버리지 측정하고 개선하기
10. 명명, 구조, 유지보수성에 대한 모범 사례를 따라 테스트 조직하기

---

테스팅은 사후 작업이 아닙니다 — 코드 품질, 리팩토링 자신감, 개발 속도에 직접적으로 영향을 미치는 핵심 엔지니어링 실천 사항입니다. 잘 테스트된 코드베이스는 두려움 없이 동작을 변경할 수 있게 해줍니다. 이 레슨에서는 C#에서의 테스팅 전체 범위를 다룹니다: xUnit과 모킹을 사용한 격리된 단위 테스트부터, `WebApplicationFactory`를 사용한 통합 테스트, 커버리지 분석까지. 마지막에는 완전한 테스팅 도구 키트를 갖추게 됩니다.

## 1. 테스팅 기초

### 1.1 테스팅 피라미드

```
        /  E2E  \        적고, 느리고, 비싼
       /----------\
      / 통합 테스트 \    중간 수량, 중간 속도
     /----------------\
    /    단위 테스트     \  많고, 빠르고, 저렴한
   /--------------------\
```

### 1.2 단위 테스트

단위 테스트(unit test)는 격리된 상태에서 단일 동작 단위를 검증합니다. 빠르고(밀리초), 결정적이며, 외부 시스템에 독립적입니다.

```csharp
// 테스트 대상 유닛
public class PriceCalculator
{
    public decimal CalculateTotal(decimal unitPrice, int quantity, decimal taxRate)
    {
        if (unitPrice < 0) throw new ArgumentException("가격은 음수일 수 없습니다");
        if (quantity < 0) throw new ArgumentException("수량은 음수일 수 없습니다");

        decimal subtotal = unitPrice * quantity;
        decimal tax = subtotal * taxRate;
        return Math.Round(subtotal + tax, 2);
    }
}
```

### 1.3 통합 테스트

통합 테스트(integration test)는 여러 컴포넌트가 함께 올바르게 작동하는지 검증합니다. 데이터베이스, 파일 시스템, HTTP 엔드포인트를 포함할 수 있습니다.

### 1.4 종단 간 테스트

E2E 테스트는 사용자 관점에서 전체 애플리케이션을 검증합니다. 가장 현실적이지만 가장 느리고 취약합니다.

## 2. xUnit 프레임워크 설정

### 2.1 테스트 프로젝트 생성

```bash
# 테스트 프로젝트 생성
dotnet new xunit -n MyApp.Tests

# 테스트 대상 프로젝트에 대한 참조 추가
dotnet add MyApp.Tests/MyApp.Tests.csproj reference MyApp/MyApp.csproj

# 테스트 실행
dotnet test

# 상세 출력으로 실행
dotnet test --verbosity normal

# 특정 테스트 실행
dotnet test --filter "ClassName=PriceCalculatorTests"
dotnet test --filter "DisplayName~Calculate"
```

### 2.2 프로젝트 구조

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

### 2.3 테스트 프로젝트 의존성

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

## 3. 테스트 메서드: [Fact], [Theory], [InlineData]

### 3.1 [Fact] — 단일 테스트 케이스

`[Fact]`는 항상 참인 테스트를 나타냅니다 — 하나의 특정 시나리오.

```csharp
public class PriceCalculatorTests
{
    private readonly PriceCalculator _calculator = new();

    [Fact]
    public void CalculateTotal_유효한입력_올바른합계반환()
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
    public void CalculateTotal_수량0_0반환()
    {
        decimal total = _calculator.CalculateTotal(25.00m, 0, 0.10m);
        Assert.Equal(0.00m, total);
    }

    [Fact]
    public void CalculateTotal_음수가격_ArgumentException던짐()
    {
        Assert.Throws<ArgumentException>(() =>
            _calculator.CalculateTotal(-5.00m, 1, 0.10m));
    }
}
```

### 3.2 [Theory]와 [InlineData] — 매개변수화된 테스트

`[Theory]`는 여러 데이터 세트로 동일한 테스트 로직을 실행합니다.

```csharp
public class PriceCalculatorTheoryTests
{
    private readonly PriceCalculator _calculator = new();

    [Theory]
    [InlineData(10.00, 1, 0.00, 10.00)]   // 세금 없음
    [InlineData(10.00, 1, 0.10, 11.00)]   // 10% 세금
    [InlineData(10.00, 3, 0.08, 32.40)]   // 여러 항목
    [InlineData(99.99, 1, 0.0725, 107.24)] // 복잡한 세율
    [InlineData(0.00, 5, 0.10, 0.00)]     // 무료 항목
    public void CalculateTotal_다양한입력_예상결과반환(
        double price, int quantity, double tax, double expected)
    {
        decimal result = _calculator.CalculateTotal(
            (decimal)price, quantity, (decimal)tax);

        Assert.Equal((decimal)expected, result);
    }
}
```

### 3.3 [Theory]와 [MemberData] 및 [ClassData]

```csharp
public class StringHelperTests
{
    // MemberData는 정적 속성 또는 메서드를 참조
    public static IEnumerable<object[]> TruncateTestData()
    {
        yield return new object[] { "Hello, World!", 5, "Hello..." };
        yield return new object[] { "Hi", 5, "Hi" };
        yield return new object[] { "", 5, "" };
        yield return new object[] { "Testing", 7, "Testing" };
    }

    [Theory]
    [MemberData(nameof(TruncateTestData))]
    public void Truncate_다양한입력_예상결과반환(
        string input, int maxLength, string expected)
    {
        string result = StringHelper.Truncate(input, maxLength);
        Assert.Equal(expected, result);
    }
}

// 복잡한 테스트 데이터를 위한 ClassData
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
public void Add_다양한입력_합계반환(int a, int b, int expected)
{
    Assert.Equal(expected, Calculator.Add(a, b));
}
```

## 4. 어설션

### 4.1 동등성과 비교

```csharp
[Fact]
public void 어설션_예제()
{
    // 동등성
    Assert.Equal(42, GetAnswer());
    Assert.NotEqual(0, GetAnswer());

    // 근사 동등성 (부동소수점용)
    Assert.Equal(3.14159, Math.PI, precision: 5);

    // 참조 동등성
    var obj = new object();
    var same = obj;
    Assert.Same(obj, same);
    Assert.NotSame(obj, new object());

    // Null 검사
    Assert.Null(GetNullableValue());
    Assert.NotNull(GetNonNullValue());

    // 불리언
    Assert.True(IsValid());
    Assert.False(IsExpired());
}
```

### 4.2 컬렉션 어설션

```csharp
[Fact]
public void 컬렉션_어설션()
{
    var numbers = new List<int> { 1, 2, 3, 4, 5 };

    // 포함
    Assert.Contains(3, numbers);
    Assert.DoesNotContain(6, numbers);

    // 모든 요소가 조건 충족
    Assert.All(numbers, n => Assert.True(n > 0));

    // 카운트
    Assert.Equal(5, numbers.Count);
    Assert.Empty(new List<int>());
    Assert.NotEmpty(numbers);
    Assert.Single(new List<int> { 42 });

    // 조건으로 포함
    Assert.Contains(numbers, n => n % 2 == 0);

    // 컬렉션 동등성 (순서 중요)
    Assert.Equal(new[] { 1, 2, 3, 4, 5 }, numbers);
}
```

### 4.3 문자열 어설션

```csharp
[Fact]
public void 문자열_어설션()
{
    string greeting = "Hello, World!";

    Assert.Equal("Hello, World!", greeting);
    Assert.StartsWith("Hello", greeting);
    Assert.EndsWith("World!", greeting);
    Assert.Contains("World", greeting);
    Assert.DoesNotContain("Goodbye", greeting);
    Assert.Matches(@"Hello,\s\w+!", greeting);  // 정규식
    Assert.Equal("hello, world!", greeting, ignoreCase: true);
}
```

### 4.4 예외 어설션

```csharp
[Fact]
public void 예외_어설션()
{
    // 특정 예외 타입 검증
    var ex = Assert.Throws<ArgumentNullException>(() =>
        ProcessData(null!));
    Assert.Equal("data", ex.ParamName);

    // 예외 메시지 검증
    var ex2 = Assert.Throws<InvalidOperationException>(() =>
        DoSomethingInvalid());
    Assert.Contains("허용되지 않음", ex2.Message);

    // 비동기 예외 검증
    // (섹션 8에서 다룸)
}
```

### 4.5 타입 어설션

```csharp
[Fact]
public void 타입_어설션()
{
    object obj = new List<int>();

    Assert.IsType<List<int>>(obj);          // 정확한 타입
    Assert.IsAssignableFrom<IList<int>>(obj); // 타입 또는 하위 타입

    var result = Assert.IsType<List<int>>(obj);
    Assert.Empty(result); // 타입이 지정된 결과 사용 가능
}
```

## 5. 테스트 생명주기

### 5.1 생성자와 Dispose (테스트별 설정/해제)

xUnit은 모든 테스트 메서드에 대해 테스트 클래스의 새 인스턴스를 생성합니다. 생성자가 설정, `IDisposable.Dispose`가 해제 역할을 합니다.

```csharp
public class FileProcessorTests : IDisposable
{
    private readonly string _tempDir;
    private readonly FileProcessor _processor;

    public FileProcessorTests()
    {
        // 각 테스트 전에 실행됨
        _tempDir = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString());
        Directory.CreateDirectory(_tempDir);
        _processor = new FileProcessor(_tempDir);
    }

    [Fact]
    public void ProcessFile_출력생성()
    {
        File.WriteAllText(Path.Combine(_tempDir, "input.txt"), "data");
        _processor.Process("input.txt");
        Assert.True(File.Exists(Path.Combine(_tempDir, "output.txt")));
    }

    [Fact]
    public void ProcessFile_빈입력_예외던짐()
    {
        File.WriteAllText(Path.Combine(_tempDir, "empty.txt"), "");
        Assert.Throws<InvalidOperationException>(() =>
            _processor.Process("empty.txt"));
    }

    public void Dispose()
    {
        // 각 테스트 후에 실행됨
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }
}
```

### 5.2 IClassFixture — 테스트 클래스 전체에서 공유 설정

설정이 비용이 많이 드는 경우(예: 데이터베이스 시작), 클래스 내 모든 테스트에서 공유합니다.

```csharp
public class DatabaseFixture : IDisposable
{
    public string ConnectionString { get; }

    public DatabaseFixture()
    {
        // 비용이 큰 설정 — 전체 테스트 클래스에 대해 한 번 실행
        ConnectionString = "Server=localhost;Database=TestDb;...";
        InitializeDatabase();
    }

    private void InitializeDatabase()
    {
        // 테이블 생성, 데이터 시드
    }

    public void Dispose()
    {
        // 정리 — 클래스의 모든 테스트 후 한 번 실행
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
    public void GetUser_존재하는ID_사용자반환()
    {
        var repo = new UserRepository(_fixture.ConnectionString);
        var user = repo.GetById(1);
        Assert.NotNull(user);
    }
}
```

### 5.3 ICollectionFixture — 여러 테스트 클래스에서 공유

```csharp
// 컬렉션 정의
[CollectionDefinition("Database")]
public class DatabaseCollection : ICollectionFixture<DatabaseFixture>
{
    // 이 클래스에는 코드가 없음 — 픽스처를 컬렉션 이름에 연결만 함
}

// 이 컬렉션의 모든 테스트 클래스가 픽스처를 공유
[Collection("Database")]
public class OrderRepositoryTests
{
    private readonly DatabaseFixture _fixture;

    public OrderRepositoryTests(DatabaseFixture fixture)
    {
        _fixture = fixture;
    }

    [Fact]
    public void CreateOrder_유효한데이터_성공()
    {
        var repo = new OrderRepository(_fixture.ConnectionString);
        var order = new Order { CustomerId = 1, Total = 99.99m };
        repo.Create(order);
        Assert.True(order.Id > 0);
    }
}
```

## 6. Arrange-Act-Assert 패턴

AAA 패턴은 모든 테스트에 일관된 구조를 제공합니다.

```csharp
public class ShoppingCartTests
{
    [Fact]
    public void AddItem_새항목_항목수증가()
    {
        // Arrange — 테스트 시나리오 설정
        var cart = new ShoppingCart();
        var item = new CartItem("SKU-001", "위젯", 9.99m, 2);

        // Act — 테스트 대상 행동 수행
        cart.AddItem(item);

        // Assert — 예상 결과 검증
        Assert.Equal(1, cart.ItemCount);
        Assert.Equal(19.98m, cart.Total);
    }

    [Fact]
    public void AddItem_기존항목_수량증가()
    {
        // Arrange
        var cart = new ShoppingCart();
        cart.AddItem(new CartItem("SKU-001", "위젯", 9.99m, 1));

        // Act
        cart.AddItem(new CartItem("SKU-001", "위젯", 9.99m, 2));

        // Assert
        Assert.Equal(1, cart.ItemCount);      // 여전히 하나의 고유 항목
        Assert.Equal(3, cart.GetQuantity("SKU-001")); // 수량이 합쳐짐
        Assert.Equal(29.97m, cart.Total);
    }

    [Fact]
    public void RemoveItem_기존항목_합계감소()
    {
        // Arrange
        var cart = new ShoppingCart();
        cart.AddItem(new CartItem("SKU-001", "위젯", 10.00m, 3));
        cart.AddItem(new CartItem("SKU-002", "가젯", 25.00m, 1));

        // Act
        cart.RemoveItem("SKU-001");

        // Assert
        Assert.Equal(1, cart.ItemCount);
        Assert.Equal(25.00m, cart.Total);
        Assert.DoesNotContain(cart.Items, i => i.Sku == "SKU-001");
    }
}
```

## 7. NSubstitute를 이용한 모킹

모킹(mocking)은 실제 의존성을 제어된 대체물로 교체하여 테스트 대상 유닛을 격리합니다.

### 7.1 기본 모킹

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
        // 대체물(모의 객체) 생성
        _repository = Substitute.For<IOrderRepository>();
        _emailSender = Substitute.For<IEmailSender>();
        _logger = Substitute.For<ILogger>();

        // 서비스에 모의 객체 주입
        _service = new OrderService(_repository, _emailSender, _logger);
    }

    [Fact]
    public void PlaceOrder_유효한주문_저장하고이메일전송()
    {
        // Arrange
        var order = new Order { Id = 1, CustomerEmail = "user@example.com" };

        // Act
        _service.PlaceOrder(order);

        // Assert — 상호작용 검증
        _repository.Received(1).Save(order);
        _emailSender.Received(1).Send(
            "user@example.com",
            Arg.Any<string>(),
            Arg.Is<string>(body => body.Contains("Order 1")));
    }
}
```

### 7.2 반환 값 구성

```csharp
[Fact]
public void GetOrder_존재하는ID_주문반환()
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
public void GetOrder_존재하지않는ID_Null반환()
{
    _repository.GetById(999).Returns((Order?)null);

    Order? result = _service.GetOrder(999);

    Assert.Null(result);
}
```

### 7.3 예외 구성

```csharp
[Fact]
public void PlaceOrder_리포지토리예외_에러로깅()
{
    // Arrange
    var order = new Order { Id = 1 };
    _repository.When(r => r.Save(Arg.Any<Order>()))
        .Do(_ => throw new InvalidOperationException("DB 연결 실패"));

    // Act & Assert
    Assert.Throws<InvalidOperationException>(() => _service.PlaceOrder(order));
    _logger.Received(1).Log(Arg.Is<string>(msg => msg.Contains("Error")));
}
```

### 7.4 인수 매처

```csharp
[Fact]
public void ProcessOrders_올바르게필터링()
{
    // Arg.Any<T>() — T 타입의 모든 값과 매칭
    _repository.Save(Arg.Any<Order>());

    // Arg.Is<T>(predicate) — 조건을 만족하는 값과 매칭
    _emailSender.Received().Send(
        Arg.Is<string>(email => email.Contains("@")),
        Arg.Any<string>(),
        Arg.Any<string>());

    // Arg.Do<T>(action) — 검사를 위해 인수를 캡처
    Order? capturedOrder = null;
    _repository.When(r => r.Save(Arg.Do<Order>(o => capturedOrder = o)))
        .Do(_ => { });

    _service.PlaceOrder(new Order { Id = 99 });
    Assert.Equal(99, capturedOrder?.Id);
}
```

## 8. 비동기 코드 테스팅

### 8.1 비동기 Fact와 Theory

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
    public async Task FetchData_성공적응답_데이터반환()
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
    public async Task FetchData_타임아웃_OperationCanceledException던짐()
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
    public async Task FetchData_다른URL_올바른사람반환(
        string url, string expectedName)
    {
        _httpClient.GetStringAsync(url)
            .Returns(Task.FromResult($$$"""{"name":"{{{expectedName}}}","age":30}"""));

        Person? result = await _fetcher.FetchPersonAsync(url);

        Assert.Equal(expectedName, result?.Name);
    }
}
```

### 8.2 취소 테스팅

```csharp
[Fact]
public async Task LongRunningOperation_취소시_OperationCanceled던짐()
{
    var service = new LongRunningService();
    using var cts = new CancellationTokenSource();

    // 짧은 지연 후 취소
    cts.CancelAfter(TimeSpan.FromMilliseconds(50));

    await Assert.ThrowsAsync<OperationCanceledException>(() =>
        service.ProcessAsync(cts.Token));
}
```

### 8.3 IAsyncEnumerable 테스팅

```csharp
[Fact]
public async Task StreamData_모든항목반환()
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

## 9. WebApplicationFactory를 이용한 통합 테스팅

`WebApplicationFactory`는 ASP.NET Core 애플리케이션을 위한 인메모리 테스트 서버를 생성하여 네트워크 없이 실제 HTTP 수준 통합 테스트를 가능하게 합니다.

### 9.1 기본 설정

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
    public async Task GetWeather_OK반환()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/weather");

        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        var forecasts = await response.Content.ReadFromJsonAsync<List<WeatherForecast>>();
        Assert.NotNull(forecasts);
        Assert.NotEmpty(forecasts);
    }

    [Fact]
    public async Task PostOrder_유효한데이터_Created반환()
    {
        var order = new { CustomerEmail = "test@example.com", Total = 99.99 };

        HttpResponseMessage response = await _client.PostAsJsonAsync("/api/orders", order);

        Assert.Equal(HttpStatusCode.Created, response.StatusCode);
    }

    [Fact]
    public async Task GetOrder_없는경우_404반환()
    {
        HttpResponseMessage response = await _client.GetAsync("/api/orders/99999");

        Assert.Equal(HttpStatusCode.NotFound, response.StatusCode);
    }
}
```

### 9.2 테스트 서비스가 있는 커스텀 팩토리

```csharp
public class CustomWebApplicationFactory : WebApplicationFactory<Program>
{
    protected override void ConfigureWebHost(IWebHostBuilder builder)
    {
        builder.ConfigureServices(services =>
        {
            // 실제 데이터베이스 컨텍스트 제거
            var descriptor = services.SingleOrDefault(
                d => d.ServiceType == typeof(DbContextOptions<AppDbContext>));
            if (descriptor != null)
                services.Remove(descriptor);

            // 테스트용 인메모리 데이터베이스 추가
            services.AddDbContext<AppDbContext>(options =>
                options.UseInMemoryDatabase("TestDb"));

            // 실제 이메일 발송기를 가짜로 교체
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

// 커스텀 팩토리 사용
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
    public async Task PlaceOrder_확인이메일전송()
    {
        // Arrange
        var order = new { CustomerEmail = "test@test.com", Total = 50.00 };

        // Act
        await _client.PostAsJsonAsync("/api/orders", order);

        // Assert — 가짜 이메일 발송기 확인
        using var scope = _factory.Services.CreateScope();
        var emailSender = scope.ServiceProvider.GetRequiredService<IEmailSender>()
            as FakeEmailSender;

        Assert.NotNull(emailSender);
        Assert.Single(emailSender!.SentEmails);
        Assert.Equal("test@test.com", emailSender.SentEmails[0].To);
    }
}
```

## 10. Coverlet을 이용한 코드 커버리지

### 10.1 커버리지 실행

```bash
# 커버리지 수집
dotnet test --collect:"XPlat Code Coverage"

# HTML 리포트 생성 (먼저 reportgenerator 설치)
dotnet tool install -g dotnet-reportgenerator-globaltool

reportgenerator \
    -reports:"**/coverage.cobertura.xml" \
    -targetdir:"coveragereport" \
    -reporttypes:Html

# 리포트 열기
open coveragereport/index.html
```

### 10.2 커버리지 구성

```xml
<!-- 테스트 프로젝트의 .csproj 또는 coverlet.runsettings 파일에서 -->
<PropertyGroup>
    <CollectCoverage>true</CollectCoverage>
    <CoverletOutputFormat>cobertura</CoverletOutputFormat>
    <Threshold>80</Threshold>
    <ThresholdType>line,branch</ThresholdType>
    <ThresholdStat>total</ThresholdStat>
    <ExcludeByFile>**/Migrations/**</ExcludeByFile>
</PropertyGroup>
```

### 10.3 커버리지에서 코드 제외

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
    [ExcludeFromCodeCoverage] // 특정 메서드 제외
    public void DebugDump()
    {
        // 디버그 전용 코드, 테스트할 가치 없음
    }
}
```

## 11. 테스트 조직 모범 사례

### 11.1 명명 규칙

```csharp
// 패턴: 메서드명_시나리오_예상동작
public class UserServiceTests
{
    [Fact]
    public void CreateUser_유효한이메일_새사용자반환() { }

    [Fact]
    public void CreateUser_중복이메일_ConflictException던짐() { }

    [Fact]
    public void CreateUser_Null이름_ArgumentNullException던짐() { }

    [Fact]
    public void GetUser_존재하는ID_사용자반환() { }

    [Fact]
    public void GetUser_삭제된사용자_Null반환() { }
}
```

### 11.2 테스트 데이터 빌더

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

// 테스트에서의 사용법
[Fact]
public void ProcessOrder_고가주문_할인적용()
{
    var order = new OrderBuilder()
        .WithTotal(500.00m)
        .WithItem("SKU-001", 250.00m, 2)
        .Build();

    _service.Process(order);

    Assert.True(order.DiscountApplied);
}
```

### 11.3 일반적인 실수 피하기

```csharp
// 나쁜 예: 테스트가 실행 순서에 의존
// 나쁜 예: 테스트가 외부 상태(데이터베이스, 파일 시스템, 네트워크)에 의존
// 나쁜 예: 하나의 테스트에 여러 Act
// 나쁜 예: 동작 대신 구현 세부 사항을 테스트
// 나쁜 예: 관련 없는 변경에 깨지는 지나치게 구체적인 어설션

// 좋은 예: 각 테스트가 독립적이고 자체 포함적
[Fact]
public void CalculateDiscount_골드회원_20퍼센트받음()
{
    // Arrange — 모든 설정이 테스트 내에 존재
    var calculator = new DiscountCalculator();
    var customer = new Customer { Tier = MemberTier.Gold };
    decimal originalPrice = 100.00m;

    // Act — 하나의 논리적 행동
    decimal discount = calculator.Calculate(customer, originalPrice);

    // Assert — 내부가 아닌 동작을 검증
    Assert.Equal(20.00m, discount);
}
```

## 12. 실전 예제: 서비스 레이어 테스팅

이 예제는 모든 것을 종합합니다: 모의 객체를 사용한 단위 테스트, 매개변수화된 이론, 비동기 테스팅, 깔끔한 테스트 조직.

```csharp
// --- 테스트 대상 서비스 ---
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
            throw new ArgumentException("상품명이 필요합니다");

        if (request.BasePrice <= 0)
            throw new ArgumentException("가격은 양수여야 합니다");

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
// --- 전체 테스트 스위트 ---
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

    // --- GetProductAsync 테스트 ---

    [Fact]
    public async Task GetProduct_존재하는상품_Dto반환()
    {
        // Arrange
        var product = new Product { Id = 1, Name = "위젯", BasePrice = 25.00m };
        _repository.GetByIdAsync(1).Returns(product);
        _pricingEngine.CalculatePrice(product, null).Returns(25.00m);

        // Act
        ProductDto? result = await _service.GetProductAsync(1);

        // Assert
        Assert.NotNull(result);
        Assert.Equal(1, result.Id);
        Assert.Equal("위젯", result.Name);
        Assert.Equal(25.00m, result.FinalPrice);
    }

    [Fact]
    public async Task GetProduct_쿠폰적용_할인적용()
    {
        // Arrange
        var product = new Product { Id = 1, Name = "위젯", BasePrice = 100.00m };
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
    public async Task GetProduct_존재하지않는ID_Null반환()
    {
        _repository.GetByIdAsync(999).Returns((Product?)null);

        ProductDto? result = await _service.GetProductAsync(999);

        Assert.Null(result);
    }

    // --- CreateProductAsync 테스트 ---

    [Fact]
    public async Task CreateProduct_유효한요청_새상품반환()
    {
        // Arrange
        var request = new CreateProductRequest("가젯", 49.99m, "전자제품");
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
        Assert.Equal("가젯", result.Name);
        await _repository.Received(1).CreateAsync(Arg.Is<Product>(p =>
            p.Name == "가젯" && p.BasePrice == 49.99m));
    }

    [Theory]
    [InlineData("", 10.00)]
    [InlineData("  ", 10.00)]
    [InlineData(null, 10.00)]
    public async Task CreateProduct_빈이름_ArgumentException던짐(
        string? name, decimal price)
    {
        var request = new CreateProductRequest(name!, price, "카테고리");

        await Assert.ThrowsAsync<ArgumentException>(() =>
            _service.CreateProductAsync(request));

        await _repository.DidNotReceive().CreateAsync(Arg.Any<Product>());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(-99.99)]
    public async Task CreateProduct_유효하지않은가격_ArgumentException던짐(decimal price)
    {
        var request = new CreateProductRequest("유효한 이름", price, "카테고리");

        await Assert.ThrowsAsync<ArgumentException>(() =>
            _service.CreateProductAsync(request));
    }

    // --- GetAllProductsAsync 테스트 ---

    [Fact]
    public async Task GetAllProducts_모든상품가격포함반환()
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
    public async Task GetAllProducts_빈리포지토리_빈리스트반환()
    {
        _repository.GetAllAsync().Returns(new List<Product>());

        List<ProductDto> results = await _service.GetAllProductsAsync();

        Assert.Empty(results);
    }
}
```

## 13. 연습 문제

1. **계산기 테스트 스위트**: `Add`, `Subtract`, `Multiply`, `Divide`, `Power`, `SquareRoot`, `Factorial` 메서드가 있는 `ScientificCalculator` 클래스에 대한 포괄적인 테스트 스위트를 작성하세요. 메서드당 최소 5개의 테스트 케이스에 `[Theory]`와 `[InlineData]`를 사용하세요. 엣지 케이스를 포함하세요: 0으로 나누기, 음수 제곱근, 오버플로우, 음수의 팩토리얼. 100% 분기 커버리지를 목표로 하세요.

2. **모의 검증**: `SendEmail`, `SendSms`, `SendPush` 메서드가 있는 `INotificationService`를 만드세요. 사용자 선호도에 따라 다른 알림 방법을 호출하는 `UserRegistrationService`를 구축하세요. NSubstitute를 사용하여 올바른 알림 방법이 호출되는지, 인수가 올바른지, 예상치 못한 호출이 없는지, 한 알림 채널의 오류가 다른 채널을 방해하지 않는지 검증하는 테스트를 작성하세요.

3. **통합 테스트 스위트**: `WebApplicationFactory`를 사용하여 `Book` 엔티티에 대한 CRUD 엔드포인트가 있는 간단한 REST API의 통합 테스트를 작성하세요(`GET /api/books`, `GET /api/books/{id}`, `POST /api/books`, `PUT /api/books/{id}`, `DELETE /api/books/{id}`). 데이터베이스를 인메모리 프로바이더로 교체하세요. 정상 경로, 유효성 검사 오류, not-found 시나리오, 동시 접근을 테스트하세요.

4. **테스트 데이터 빌더 라이브러리**: 유연한 테스트 데이터 빌더 시스템을 만드세요. `Customer`, `Order`, `OrderItem`, `Address`에 대한 빌더를 구현하되, 기본값, 메서드 체이닝, 중첩 빌더(예: `OrderBuilder.WithCustomer(c => c.WithName("Alice"))`), 고유한 인스턴스를 생성하는 `BuildMany(int count)` 메서드를 지원하세요.

5. **비동기 재시도 테스터**: `Func<Task<T>>`를 지수 백오프로 N번까지 재시도하는 `RetryService`를 작성하세요. 내부 함수를 모킹하여 K번 실패 후 성공하도록 하고, 올바른 재시도 횟수를 검증하고, 재시도 간 지연이 지수적으로 증가하는지 확인하고, 재시도 대기 중 취소를 테스트하고, 모든 재시도가 소진되는 경우를 테스트하는 테스트 스위트를 작성하세요.
