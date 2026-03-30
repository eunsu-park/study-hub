# 의존성 주입

**이전**: [Span과 메모리](./09_Spans_and_Memory.md) | **다음**: [직렬화](./11_Serialization.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 의존성 주입(Dependency Injection, DI) 원칙과 이점 설명하기
2. 생성자 주입을 적용하여 컴포넌트 분리하기
3. Microsoft.Extensions.DependencyInjection 컨테이너 구성하기
4. 올바른 서비스 수명(Transient, Scoped, Singleton) 선택하기
5. Options 패턴을 사용하여 강타입 구성 주입하기
6. 키 서비스(keyed service)와 팩토리 기반 등록 사용하기
7. 일반적인 DI 안티패턴 인식하고 피하기
8. DI 합성 루트(composition root)를 사용하여 모듈식 애플리케이션 설계하기

---

애플리케이션이 성장함에 따라 클래스는 불가피하게 다른 클래스에 의존하게 됩니다: 컨트롤러는 서비스가 필요하고, 서비스는 리포지토리가 필요하며, 리포지토리는 데이터베이스 연결이 필요합니다. 규율 잡힌 접근 방식 없이는 이러한 의존성이 얽히게 되어 테스트하기 어렵고 변경하기 어려워집니다. 의존성 주입(DI)은 제어를 역전시킵니다: 클래스가 자체 의존성을 생성하는 대신 외부에서 제공됩니다. C#과 .NET은 `Microsoft.Extensions.DependencyInjection`을 통해 DI를 일급으로 지원하며, 이는 ASP.NET Core, 워커 서비스, 모든 .NET 애플리케이션을 구동합니다.

## 1. DI란 무엇이며 왜 사용하는가

### 1.1 문제: 강한 결합

```csharp
// 강하게 결합됨 — OrderService가 자체 의존성을 생성
public class OrderService
{
    private readonly SqlOrderRepository _repository;
    private readonly SmtpEmailSender _emailSender;

    public OrderService()
    {
        // 하드코딩된 의존성
        _repository = new SqlOrderRepository("Server=localhost;Database=Orders;...");
        _emailSender = new SmtpEmailSender("smtp.example.com", 587);
    }

    public void PlaceOrder(Order order)
    {
        _repository.Save(order);
        _emailSender.Send(order.CustomerEmail, "주문 확인", $"주문 {order.Id}");
    }
}
```

이 접근 방식의 문제점:
- 테스트에서 `SqlOrderRepository`를 인메모리 리포지토리로 교체할 수 없음
- 실제 이메일 전송을 피하기 위해 `SmtpEmailSender`를 모의 객체로 교체할 수 없음
- 연결 문자열이 클래스 내부에 묻혀 있음
- 새 의존성을 추가하려면 생성자를 수정해야 함

### 1.2 해결책: 의존성 주입

```csharp
// 느슨하게 결합됨 — 의존성이 생성자를 통해 주입됨
public interface IOrderRepository
{
    void Save(Order order);
    Order? GetById(int id);
}

public interface IEmailSender
{
    void Send(string to, string subject, string body);
}

public class OrderService
{
    private readonly IOrderRepository _repository;
    private readonly IEmailSender _emailSender;

    // 의존성이 외부에서 제공됨
    public OrderService(IOrderRepository repository, IEmailSender emailSender)
    {
        _repository = repository ?? throw new ArgumentNullException(nameof(repository));
        _emailSender = emailSender ?? throw new ArgumentNullException(nameof(emailSender));
    }

    public void PlaceOrder(Order order)
    {
        _repository.Save(order);
        _emailSender.Send(order.CustomerEmail, "주문 확인", $"주문 {order.Id}");
    }
}
```

이점:
- **테스트 용이성**: 단위 테스트에서 모의 구현 주입 가능
- **유연성**: 소비 코드를 변경하지 않고 구현 교체 가능
- **관심사 분리**: 각 클래스가 하나의 일만 수행
- **명시적 의존성**: 생성자 시그니처가 클래스가 필요로 하는 것을 문서화

## 2. 생성자 주입

생성자 주입(constructor injection)은 .NET에서 권장되는 패턴입니다. 모든 필수 의존성은 생성자 매개변수로 선언됩니다.

### 2.1 단일 책임

```csharp
public interface ILogger
{
    void Log(string message);
}

public interface IInventoryService
{
    bool Reserve(string productId, int quantity);
    void Release(string productId, int quantity);
}

public interface IPaymentGateway
{
    PaymentResult Charge(string customerId, decimal amount);
}

public class CheckoutService
{
    private readonly IInventoryService _inventory;
    private readonly IPaymentGateway _payment;
    private readonly IOrderRepository _orders;
    private readonly ILogger _logger;

    public CheckoutService(
        IInventoryService inventory,
        IPaymentGateway payment,
        IOrderRepository orders,
        ILogger logger)
    {
        _inventory = inventory;
        _payment = payment;
        _orders = orders;
        _logger = logger;
    }

    public CheckoutResult Checkout(Cart cart)
    {
        _logger.Log($"카트 {cart.Id}에 대한 결제 시작");

        foreach (var item in cart.Items)
        {
            if (!_inventory.Reserve(item.ProductId, item.Quantity))
            {
                _logger.Log($"{item.ProductId}의 재고 부족");
                return CheckoutResult.InsufficientInventory;
            }
        }

        PaymentResult paymentResult = _payment.Charge(cart.CustomerId, cart.Total);
        if (!paymentResult.Success)
        {
            foreach (var item in cart.Items)
                _inventory.Release(item.ProductId, item.Quantity);

            return CheckoutResult.PaymentFailed;
        }

        var order = new Order(cart);
        _orders.Save(order);
        _logger.Log($"주문 {order.Id} 성공적으로 생성됨");
        return CheckoutResult.Success;
    }
}
```

### 2.2 기본 생성자 (C# 12)

```csharp
// C# 12 기본 생성자 — 더 간결함
public class CheckoutService(
    IInventoryService inventory,
    IPaymentGateway payment,
    IOrderRepository orders,
    ILogger logger)
{
    public CheckoutResult Checkout(Cart cart)
    {
        logger.Log($"카트 {cart.Id}에 대한 결제 시작");
        // ... inventory, payment, orders를 직접 사용
        return CheckoutResult.Success;
    }
}
```

## 3. Microsoft.Extensions.DependencyInjection

.NET 내장 DI 컨테이너는 가볍고 대부분의 애플리케이션에 충분합니다.

### 3.1 컨테이너 설정

```csharp
using Microsoft.Extensions.DependencyInjection;

// 서비스 컬렉션 생성 및 구성
var services = new ServiceCollection();

// 서비스 등록
services.AddTransient<IEmailSender, SmtpEmailSender>();
services.AddScoped<IOrderRepository, SqlOrderRepository>();
services.AddSingleton<ILogger, ConsoleLogger>();
services.AddScoped<OrderService>();

// 서비스 프로바이더 빌드 (합성 루트)
ServiceProvider provider = services.BuildServiceProvider();

// 서비스 해석
using var scope = provider.CreateScope();
var orderService = scope.ServiceProvider.GetRequiredService<OrderService>();
orderService.PlaceOrder(new Order { CustomerEmail = "user@example.com" });
```

### 3.2 ASP.NET Core에서

```csharp
var builder = WebApplication.CreateBuilder(args);

// Program.cs에서 서비스 등록
builder.Services.AddScoped<IOrderRepository, SqlOrderRepository>();
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();
builder.Services.AddScoped<OrderService>();
builder.Services.AddControllers();

var app = builder.Build();
app.MapControllers();
app.Run();
```

```csharp
// 컨트롤러가 자동으로 의존성을 받음
[ApiController]
[Route("api/[controller]")]
public class OrdersController : ControllerBase
{
    private readonly OrderService _orderService;

    public OrdersController(OrderService orderService)
    {
        _orderService = orderService;
    }

    [HttpPost]
    public IActionResult PlaceOrder([FromBody] OrderRequest request)
    {
        var order = new Order { CustomerEmail = request.Email };
        _orderService.PlaceOrder(order);
        return Ok(new { order.Id });
    }
}
```

### 3.3 GetService vs GetRequiredService

```csharp
// GetService는 등록되지 않은 경우 null 반환
IEmailSender? sender = provider.GetService<IEmailSender>();
if (sender is not null)
{
    sender.Send("test@example.com", "제목", "본문");
}

// GetRequiredService는 등록되지 않은 경우 InvalidOperationException 던짐
IEmailSender requiredSender = provider.GetRequiredService<IEmailSender>();
```

## 4. 서비스 수명: Transient, Scoped, Singleton

올바른 수명을 선택하는 것은 정확성과 성능에 매우 중요합니다.

### 4.1 Transient

서비스가 요청될 때마다 새 인스턴스가 생성됩니다.

```csharp
services.AddTransient<IEmailSender, SmtpEmailSender>();

// 각 해석마다 새 인스턴스 생성
var sender1 = provider.GetRequiredService<IEmailSender>();
var sender2 = provider.GetRequiredService<IEmailSender>();
Console.WriteLine(ReferenceEquals(sender1, sender2)); // False
```

**사용 대상**: 가볍고 상태가 없는 서비스. 공유 상태를 보유하지 않는 서비스.

### 4.2 Scoped

스코프당 하나의 인스턴스. ASP.NET Core에서는 각 HTTP 요청이 스코프를 생성합니다.

```csharp
services.AddScoped<IOrderRepository, SqlOrderRepository>();

using var scope1 = provider.CreateScope();
var repo1a = scope1.ServiceProvider.GetRequiredService<IOrderRepository>();
var repo1b = scope1.ServiceProvider.GetRequiredService<IOrderRepository>();
Console.WriteLine(ReferenceEquals(repo1a, repo1b)); // True — 같은 스코프

using var scope2 = provider.CreateScope();
var repo2 = scope2.ServiceProvider.GetRequiredService<IOrderRepository>();
Console.WriteLine(ReferenceEquals(repo1a, repo2)); // False — 다른 스코프
```

**사용 대상**: 데이터베이스 컨텍스트(DbContext), 작업 단위 패턴, 요청별 상태.

### 4.3 Singleton

전체 애플리케이션 수명 동안 하나의 인스턴스.

```csharp
services.AddSingleton<ILogger, ConsoleLogger>();

var logger1 = provider.GetRequiredService<ILogger>();
var logger2 = provider.GetRequiredService<ILogger>();
Console.WriteLine(ReferenceEquals(logger1, logger2)); // True
```

**사용 대상**: 구성, 캐시, HTTP 클라이언트 팩토리, 스레드 안전한 공유 상태.

### 4.4 수명 비교 표

```csharp
public class LifetimeDemo
{
    private static int _counter;

    public class MyService
    {
        public int Id { get; } = Interlocked.Increment(ref _counter);
        public override string ToString() => $"인스턴스 #{Id}";
    }

    public static void Demonstrate()
    {
        var services = new ServiceCollection();
        services.AddTransient<MyService>();    // 다른 수명을 테스트하려면 변경
        var provider = services.BuildServiceProvider();

        using var scope1 = provider.CreateScope();
        var a = scope1.ServiceProvider.GetRequiredService<MyService>();
        var b = scope1.ServiceProvider.GetRequiredService<MyService>();

        using var scope2 = provider.CreateScope();
        var c = scope2.ServiceProvider.GetRequiredService<MyService>();

        // Transient: a=#1, b=#2, c=#3 (모두 다름)
        // Scoped:    a=#1, b=#1, c=#2 (스코프 내 동일)
        // Singleton: a=#1, b=#1, c=#1 (항상 동일)
        Console.WriteLine($"a={a}, b={b}, c={c}");
    }
}
```

## 5. IServiceProvider와 서비스 해석

### 5.1 여러 구현 해석

```csharp
// 같은 인터페이스의 여러 구현 등록
services.AddTransient<INotificationChannel, EmailChannel>();
services.AddTransient<INotificationChannel, SmsChannel>();
services.AddTransient<INotificationChannel, PushChannel>();

// 모든 구현 해석
IEnumerable<INotificationChannel> channels =
    provider.GetServices<INotificationChannel>();

foreach (var channel in channels)
{
    channel.Send("안녕하세요!");
}
```

### 5.2 알림 디스패처 만들기

```csharp
public class NotificationDispatcher
{
    private readonly IEnumerable<INotificationChannel> _channels;

    public NotificationDispatcher(IEnumerable<INotificationChannel> channels)
    {
        _channels = channels;
    }

    public void Broadcast(string message)
    {
        foreach (var channel in _channels)
        {
            try
            {
                channel.Send(message);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"{channel.GetType().Name} 실패: {ex.Message}");
            }
        }
    }
}

// 등록
services.AddTransient<INotificationChannel, EmailChannel>();
services.AddTransient<INotificationChannel, SmsChannel>();
services.AddTransient<NotificationDispatcher>();
```

## 6. Options 패턴 (IOptions&lt;T&gt;)

Options 패턴은 구성 섹션에 대한 강타입 접근을 제공합니다.

### 6.1 옵션 클래스 정의

```csharp
public class SmtpOptions
{
    public const string SectionName = "Smtp";

    public string Host { get; set; } = "localhost";
    public int Port { get; set; } = 587;
    public string Username { get; set; } = "";
    public string Password { get; set; } = "";
    public bool UseSsl { get; set; } = true;
}

public class DatabaseOptions
{
    public const string SectionName = "Database";

    public string ConnectionString { get; set; } = "";
    public int MaxRetries { get; set; } = 3;
    public int CommandTimeoutSeconds { get; set; } = 30;
}
```

### 6.2 구성 바인딩

```csharp
// appsettings.json
// {
//   "Smtp": {
//     "Host": "smtp.example.com",
//     "Port": 587,
//     "Username": "noreply@example.com",
//     "Password": "secret",
//     "UseSsl": true
//   },
//   "Database": {
//     "ConnectionString": "Server=localhost;Database=MyApp;...",
//     "MaxRetries": 5
//   }
// }

var builder = WebApplication.CreateBuilder(args);

// 구성에서 옵션 바인딩
builder.Services.Configure<SmtpOptions>(
    builder.Configuration.GetSection(SmtpOptions.SectionName));

builder.Services.Configure<DatabaseOptions>(
    builder.Configuration.GetSection(DatabaseOptions.SectionName));
```

### 6.3 옵션 주입

```csharp
public class SmtpEmailSender : IEmailSender
{
    private readonly SmtpOptions _options;

    // IOptions<T>: 시작 시 한 번 읽음
    public SmtpEmailSender(IOptions<SmtpOptions> options)
    {
        _options = options.Value;
    }

    public void Send(string to, string subject, string body)
    {
        Console.WriteLine($"{_options.Host}:{_options.Port}를 통해 {to}에게 전송");
        // ... 실제 SMTP 구현
    }
}
```

### 6.4 IOptionsSnapshot vs IOptionsMonitor

```csharp
public class ConfigAwareService
{
    private readonly IOptionsMonitor<DatabaseOptions> _dbOptions;

    // IOptionsMonitor<T>: 구성이 변경되면 다시 로드 (Singleton 안전)
    public ConfigAwareService(IOptionsMonitor<DatabaseOptions> dbOptions)
    {
        _dbOptions = dbOptions;
        _dbOptions.OnChange(newOptions =>
        {
            Console.WriteLine($"구성 변경됨: MaxRetries = {newOptions.MaxRetries}");
        });
    }

    public string GetConnectionString()
    {
        return _dbOptions.CurrentValue.ConnectionString;
    }
}

// IOptionsSnapshot<T>: 스코프 — 요청마다 새 스냅샷 (Scoped 서비스만)
public class RequestScopedService
{
    private readonly DatabaseOptions _options;

    public RequestScopedService(IOptionsSnapshot<DatabaseOptions> options)
    {
        _options = options.Value;
    }
}
```

### 6.5 옵션 유효성 검사

```csharp
builder.Services.AddOptions<SmtpOptions>()
    .Bind(builder.Configuration.GetSection(SmtpOptions.SectionName))
    .Validate(o => !string.IsNullOrEmpty(o.Host), "SMTP 호스트가 필요합니다")
    .Validate(o => o.Port is > 0 and < 65536, "포트는 1과 65535 사이여야 합니다")
    .ValidateOnStart(); // 유효하지 않으면 시작 시 즉시 실패
```

## 7. 키 서비스

키 서비스(keyed service, .NET 8에서 도입)를 사용하면 문자열 또는 열거형 키를 사용하여 같은 인터페이스의 여러 구현을 등록하고 해석할 수 있습니다.

```csharp
public enum StorageType { Local, S3, Azure }

services.AddKeyedSingleton<IStorageProvider, LocalStorageProvider>(StorageType.Local);
services.AddKeyedSingleton<IStorageProvider, S3StorageProvider>(StorageType.S3);
services.AddKeyedSingleton<IStorageProvider, AzureBlobStorageProvider>(StorageType.Azure);

// 특정 키 서비스 주입
public class FileUploadService
{
    private readonly IStorageProvider _storage;

    public FileUploadService(
        [FromKeyedServices(StorageType.S3)] IStorageProvider storage)
    {
        _storage = storage;
    }

    public async Task UploadAsync(Stream file, string path)
    {
        await _storage.SaveAsync(file, path);
    }
}
```

```csharp
// 키 서비스 수동 해석
var provider = services.BuildServiceProvider();
var s3 = provider.GetRequiredKeyedService<IStorageProvider>(StorageType.S3);
var local = provider.GetRequiredKeyedService<IStorageProvider>(StorageType.Local);
```

## 8. 팩토리 기반 등록

서비스 생성 방식에 대한 더 많은 제어가 필요한 경우가 있습니다.

### 8.1 간단한 팩토리

```csharp
services.AddTransient<IEmailSender>(provider =>
{
    var options = provider.GetRequiredService<IOptions<SmtpOptions>>().Value;
    var logger = provider.GetRequiredService<ILogger<SmtpEmailSender>>();
    return new SmtpEmailSender(options, logger);
});
```

### 8.2 조건부 등록

```csharp
services.AddScoped<IOrderRepository>(provider =>
{
    var config = provider.GetRequiredService<IConfiguration>();
    string dbType = config["Database:Type"] ?? "sql";

    return dbType.ToLowerInvariant() switch
    {
        "sql" => new SqlOrderRepository(
            provider.GetRequiredService<IOptions<DatabaseOptions>>()),
        "mongo" => new MongoOrderRepository(
            provider.GetRequiredService<IOptions<MongoOptions>>()),
        "memory" => new InMemoryOrderRepository(),
        _ => throw new InvalidOperationException($"알 수 없는 데이터베이스 타입: {dbType}")
    };
});
```

### 8.3 DI를 이용한 데코레이터 패턴

```csharp
// 기본 구현 등록
services.AddScoped<SqlOrderRepository>();

// 기본 구현을 래핑하는 데코레이터 등록
services.AddScoped<IOrderRepository>(provider =>
{
    var inner = provider.GetRequiredService<SqlOrderRepository>();
    var logger = provider.GetRequiredService<ILogger<LoggingOrderRepository>>();
    var cache = provider.GetRequiredService<IMemoryCache>();

    // 로깅 데코레이터로 래핑
    IOrderRepository logged = new LoggingOrderRepository(inner, logger);
    // 캐싱 데코레이터로 래핑
    IOrderRepository cached = new CachingOrderRepository(logged, cache);

    return cached;
});
```

```csharp
public class LoggingOrderRepository : IOrderRepository
{
    private readonly IOrderRepository _inner;
    private readonly ILogger _logger;

    public LoggingOrderRepository(IOrderRepository inner, ILogger logger)
    {
        _inner = inner;
        _logger = logger;
    }

    public void Save(Order order)
    {
        _logger.LogInformation("주문 {OrderId} 저장 중", order.Id);
        _inner.Save(order);
        _logger.LogInformation("주문 {OrderId} 저장됨", order.Id);
    }

    public Order? GetById(int id)
    {
        _logger.LogInformation("주문 {OrderId} 조회 중", id);
        return _inner.GetById(id);
    }
}
```

## 9. DI 모범 사례

### 9.1 서비스 로케이터 안티패턴 피하기

```csharp
// 나쁜 예: 서비스 로케이터 — 의존성을 숨기고 테스트하기 어렵게 만듦
public class BadService
{
    private readonly IServiceProvider _provider;

    public BadService(IServiceProvider provider)
    {
        _provider = provider;
    }

    public void DoWork()
    {
        // 의존성이 숨겨져 있음 — 생성자에서 보이지 않음
        var repo = _provider.GetRequiredService<IOrderRepository>();
        repo.Save(new Order());
    }
}

// 좋은 예: 명시적 생성자 주입
public class GoodService
{
    private readonly IOrderRepository _repository;

    public GoodService(IOrderRepository repository)
    {
        _repository = repository; // 의존성이 명시적
    }

    public void DoWork()
    {
        _repository.Save(new Order());
    }
}
```

### 9.2 포획된 의존성 피하기

포획된 의존성(captive dependency)은 더 긴 수명의 서비스가 더 짧은 수명의 서비스에 대한 참조를 보유할 때 발생합니다.

```csharp
// 위험: Singleton이 Scoped 서비스를 포획
services.AddSingleton<MySingletonService>();  // 영원히 존재
services.AddScoped<MyDbContext>();              // 요청당 존재

public class MySingletonService
{
    private readonly MyDbContext _db; // 버그: 이 DbContext는 스코프보다 오래 살아남음!

    public MySingletonService(MyDbContext db)
    {
        _db = db; // 이 DbContext는 dispose되었을 수 있는 스코프에서 생성됨
    }
}
```

```csharp
// 해결: IServiceScopeFactory를 사용하여 필요할 때 스코프 생성
public class MySingletonService
{
    private readonly IServiceScopeFactory _scopeFactory;

    public MySingletonService(IServiceScopeFactory scopeFactory)
    {
        _scopeFactory = scopeFactory;
    }

    public void DoWork()
    {
        using var scope = _scopeFactory.CreateScope();
        var db = scope.ServiceProvider.GetRequiredService<MyDbContext>();
        // 이 스코프 내에서 db 사용 — 제대로 dispose됨
    }
}
```

### 9.3 인터페이스를 등록하고, 구현을 등록하지 않기

```csharp
// 나쁜 예: 구체 타입 자체를 등록
services.AddScoped<SqlOrderRepository>();
// 소비자가 SqlOrderRepository에 직접 의존해야 함

// 좋은 예: 인터페이스 뒤에 등록
services.AddScoped<IOrderRepository, SqlOrderRepository>();
// 소비자가 IOrderRepository에 의존 — 구현 교체 가능
```

### 9.4 합성 루트

모든 DI 등록은 합성 루트(composition root)라는 단일 위치에서 이루어져야 합니다. ASP.NET Core에서는 `Program.cs`입니다. 코드베이스 전체에 등록을 분산시키지 마세요.

```csharp
// Program.cs — 합성 루트
var builder = WebApplication.CreateBuilder(args);

// 기능별로 등록 그룹화
builder.Services.AddOrderingModule(builder.Configuration);
builder.Services.AddNotificationModule(builder.Configuration);
builder.Services.AddInventoryModule(builder.Configuration);

var app = builder.Build();
```

```csharp
// 모듈식 등록을 위한 확장 메서드
public static class OrderingModuleExtensions
{
    public static IServiceCollection AddOrderingModule(
        this IServiceCollection services, IConfiguration config)
    {
        services.Configure<DatabaseOptions>(config.GetSection("Database"));
        services.AddScoped<IOrderRepository, SqlOrderRepository>();
        services.AddScoped<OrderService>();
        services.AddScoped<CheckoutService>();
        return services;
    }
}
```

## 10. 실전 예제: DI를 사용한 모듈식 애플리케이션

이 예제는 깔끔한 DI 설정으로 모듈식 콘솔 애플리케이션을 구축합니다: 인터페이스, 구현, 옵션, 합성 루트.

```csharp
// --- 인터페이스 ---
public interface IWeatherProvider
{
    Task<WeatherData> GetCurrentWeatherAsync(string city);
}

public interface IWeatherCache
{
    WeatherData? Get(string city);
    void Set(string city, WeatherData data, TimeSpan expiry);
}

public interface IWeatherFormatter
{
    string Format(WeatherData data);
}

public record WeatherData(string City, double Temperature, string Condition, DateTime Timestamp);
```

```csharp
// --- 옵션 ---
public class WeatherApiOptions
{
    public string ApiKey { get; set; } = "";
    public string BaseUrl { get; set; } = "https://api.weather.example.com";
    public int TimeoutSeconds { get; set; } = 10;
}

public class CacheOptions
{
    public int ExpirationMinutes { get; set; } = 15;
}
```

```csharp
// --- 구현 ---
public class HttpWeatherProvider : IWeatherProvider
{
    private readonly HttpClient _client;
    private readonly WeatherApiOptions _options;

    public HttpWeatherProvider(HttpClient client, IOptions<WeatherApiOptions> options)
    {
        _client = client;
        _options = options.Value;
        _client.BaseAddress = new Uri(_options.BaseUrl);
        _client.Timeout = TimeSpan.FromSeconds(_options.TimeoutSeconds);
    }

    public async Task<WeatherData> GetCurrentWeatherAsync(string city)
    {
        var response = await _client.GetStringAsync($"/current?city={city}&key={_options.ApiKey}");
        // 응답 파싱 (간략화)
        return new WeatherData(city, 22.5, "맑음", DateTime.UtcNow);
    }
}

public class InMemoryWeatherCache : IWeatherCache
{
    private readonly ConcurrentDictionary<string, (WeatherData Data, DateTime Expiry)> _cache = new();

    public WeatherData? Get(string city)
    {
        if (_cache.TryGetValue(city, out var entry) && entry.Expiry > DateTime.UtcNow)
            return entry.Data;
        return null;
    }

    public void Set(string city, WeatherData data, TimeSpan expiry)
    {
        _cache[city] = (data, DateTime.UtcNow.Add(expiry));
    }
}

public class ConsoleWeatherFormatter : IWeatherFormatter
{
    public string Format(WeatherData data)
    {
        return $"{data.City}: {data.Temperature}C, {data.Condition} ({data.Timestamp:HH:mm} 기준)";
    }
}
```

```csharp
// --- 애플리케이션 서비스 ---
public class WeatherApp
{
    private readonly IWeatherProvider _provider;
    private readonly IWeatherCache _cache;
    private readonly IWeatherFormatter _formatter;
    private readonly CacheOptions _cacheOptions;

    public WeatherApp(
        IWeatherProvider provider,
        IWeatherCache cache,
        IWeatherFormatter formatter,
        IOptions<CacheOptions> cacheOptions)
    {
        _provider = provider;
        _cache = cache;
        _formatter = formatter;
        _cacheOptions = cacheOptions.Value;
    }

    public async Task<string> GetWeatherReportAsync(string city)
    {
        WeatherData? data = _cache.Get(city);

        if (data is null)
        {
            data = await _provider.GetCurrentWeatherAsync(city);
            _cache.Set(city, data, TimeSpan.FromMinutes(_cacheOptions.ExpirationMinutes));
        }

        return _formatter.Format(data);
    }
}
```

```csharp
// --- 합성 루트 (Program.cs) ---
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Configuration;

var config = new ConfigurationBuilder()
    .AddJsonFile("appsettings.json", optional: true)
    .AddEnvironmentVariables()
    .Build();

var services = new ServiceCollection();

// 구성
services.Configure<WeatherApiOptions>(config.GetSection("WeatherApi"));
services.Configure<CacheOptions>(config.GetSection("Cache"));

// 서비스
services.AddHttpClient<IWeatherProvider, HttpWeatherProvider>();
services.AddSingleton<IWeatherCache, InMemoryWeatherCache>();
services.AddTransient<IWeatherFormatter, ConsoleWeatherFormatter>();
services.AddTransient<WeatherApp>();

var provider = services.BuildServiceProvider();

// 애플리케이션 실행
var app = provider.GetRequiredService<WeatherApp>();
string[] cities = { "서울", "런던", "도쿄", "시드니" };

foreach (string city in cities)
{
    string report = await app.GetWeatherReportAsync(city);
    Console.WriteLine(report);
}
```

## 11. 연습 문제

1. **플러그인 시스템**: `string Name`, `int Priority`, `Task ExecuteAsync()`가 있는 `IPlugin` 인터페이스를 설계하세요. 다양한 우선순위를 가진 5개의 플러그인 구현을 등록하세요. `IEnumerable<IPlugin>`을 통해 모든 `IPlugin` 인스턴스를 해석하고, 우선순위별로 정렬하여 순차적으로 실행하는 `PluginRunner` 서비스를 만드세요. 이름으로 특정 플러그인을 비활성화하는 구성 옵션을 추가하세요.

2. **서비스 수명 추적기**: 생성과 소멸을 로깅하는 `LifetimeTracker` 서비스를 만드세요(`IDisposable` 구현). Transient, Scoped, Singleton으로 등록하세요(`ITransientTracker`, `IScopedTracker`, `ISingletonTracker` 같은 다른 인터페이스 마커 사용). 3개의 스코프를 생성하고, 각 스코프에서 각 추적기 타입을 두 번 해석하여 생성/소멸 패턴을 관찰하는 콘솔 프로그램을 작성하세요.

3. **구성 핫 리로드**: `IOptionsMonitor<T>`를 사용하여 JSON 파일에서 설정을 읽는 서비스를 구축하세요. 서비스는 구성이 변경될 때마다 로깅해야 합니다. 런타임에 JSON 파일을 수정하고 서비스가 5초 이내에 새 값을 감지하는지 확인하는 테스트를 작성하세요.

4. **데코레이터 체인**: `IMessageSender` 인터페이스에 대한 데코레이터 패턴을 구현하세요. `LoggingMessageSender`, `RetryMessageSender`(실패 시 3번 재시도), `CircuitBreakerMessageSender`(연속 5번 실패 후 전송 중지) 세 가지 데코레이터를 만드세요. 팩토리 등록을 사용하여 데코레이터 체인으로 등록하세요. 실패를 시뮬레이션하고 각 데코레이터의 동작을 확인하는 테스트를 작성하세요.

5. **모듈식 등록**: 가상의 전자상거래 애플리케이션을 4개 모듈(카탈로그, 주문, 결제, 알림)로 분할하세요. 각 모듈에는 자체 인터페이스, 구현, 옵션이 있습니다. 각 모듈에 대한 확장 메서드(`AddCatalogModule`, `AddOrdersModule` 등)를 만들어 모든 서비스를 등록하세요. 전체 애플리케이션을 조립하고 4개 모듈의 서비스에 의존하는 최상위 서비스를 해석하는 합성 루트를 작성하세요.
