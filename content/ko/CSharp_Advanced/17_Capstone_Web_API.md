# 캡스톤: Minimal Web API

**이전**: [성능과 프로파일링](./16_Performance_Profiling.md) | **다음**: [CSharp Advanced 개요](./00_Overview.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. ASP.NET Core를 사용하여 Minimal Web API 프로젝트를 생성할 수 있다
2. `MapGet`, `MapPost`, `MapPut`, `MapDelete`로 RESTful 엔드포인트를 정의할 수 있다
3. 라우트 매개변수, 쿼리 문자열, JSON 요청/응답 본문을 처리할 수 있다
4. 내장 DI 컨테이너를 사용하여 종속성을 등록하고 주입할 수 있다
5. SQLite와 Entity Framework Core로 데이터 지속성을 설정할 수 있다
6. 미들웨어로 입력 유효성 검사와 오류 처리를 구현할 수 있다
7. 기본 JWT 인증을 구현할 수 있다
8. `WebApplicationFactory`를 사용하여 통합 테스트를 작성할 수 있다
9. 프로덕션 품질의 API 프로젝트 구조를 구성할 수 있다

---

이 캡스톤 레슨은 전체 과정의 개념을 결합하여 완전한 작동하는 Web API를 구축합니다. .NET 6에서 도입된 Minimal API는 컨트롤러의 번거로움 없이 HTTP 서비스를 구축하는 간소화된 방법을 제공합니다. 처음부터 Todo API를 구축합니다: 엔드포인트 정의, Entity Framework Core와 SQLite로 데이터 지속, 입력 유효성 검사, 오류 처리, 인증 추가, 테스트 작성까지. 마지막에는 실제 API 개발 패턴을 보여주는 참조 구현을 갖게 됩니다.

## 1. ASP.NET Core Minimal API 개요

### 1.1 Minimal API란?

Minimal API는 최소한의 보일러플레이트로 HTTP 엔드포인트를 정의할 수 있게 합니다. 컨트롤러, 어트리뷰트, 복잡한 시작 클래스 대신 `Program.cs`에서 직접 라우트를 매핑합니다:

```csharp
// 가장 간단한 ASP.NET Core 애플리케이션
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "Hello, World!");

app.Run();
```

이것은 완전히 기능하는 웹 서버입니다. `builder`는 서비스(의존성 주입, 로깅, 구성)를 구성하고, `app`은 요청 파이프라인(미들웨어와 엔드포인트)을 정의합니다.

### 1.2 Minimal API vs 컨트롤러

```csharp
// Minimal API 스타일:
app.MapGet("/api/products/{id}", async (int id, ProductService service) =>
{
    var product = await service.GetByIdAsync(id);
    return product is not null ? Results.Ok(product) : Results.NotFound();
});

// 컨트롤러 스타일 (비교용 — 이 레슨에서는 사용하지 않음):
// [ApiController]
// [Route("api/[controller]")]
// public class ProductsController : ControllerBase
// {
//     [HttpGet("{id}")]
//     public async Task<IActionResult> Get(int id, [FromServices] ProductService service)
//     {
//         var product = await service.GetByIdAsync(id);
//         return product is not null ? Ok(product) : NotFound();
//     }
// }

// Minimal API를 사용하는 경우:
// - 적은 수의 엔드포인트를 가진 마이크로서비스
// - 빠른 프로토타이핑
// - 간단한 CRUD API
// - 규약 기반보다 명시적 라우팅을 선호할 때

// 컨트롤러가 더 나을 수 있는 경우:
// - 많은 엔드포인트를 가진 대규모 API
// - 필터(액션 필터, 결과 필터)가 필요할 때
// - 팀이 MVC 패턴에 익숙할 때
```

## 2. 새 Web API 프로젝트 생성

### 2.1 프로젝트 설정

```bash
# 솔루션 구조 생성
mkdir TodoApi && cd TodoApi
dotnet new sln

# API 프로젝트 생성
dotnet new webapi -o src/TodoApi --use-minimal-apis
dotnet sln add src/TodoApi

# 테스트 프로젝트 생성
dotnet new xunit -o tests/TodoApi.Tests
dotnet sln add tests/TodoApi.Tests
dotnet add tests/TodoApi.Tests reference src/TodoApi

# API 프로젝트에 필요한 패키지 추가
cd src/TodoApi
dotnet add package Microsoft.EntityFrameworkCore.Sqlite
dotnet add package Microsoft.EntityFrameworkCore.Design
dotnet add package Microsoft.AspNetCore.Authentication.JwtBearer
dotnet add package FluentValidation

# 테스트 패키지 추가
cd ../../tests/TodoApi.Tests
dotnet add package Microsoft.AspNetCore.Mvc.Testing
dotnet add package FluentAssertions
```

### 2.2 프로젝트 구조

```
TodoApi/
├── TodoApi.sln
├── src/
│   └── TodoApi/
│       ├── TodoApi.csproj
│       ├── Program.cs
│       ├── appsettings.json
│       ├── Data/
│       │   ├── TodoDbContext.cs
│       │   └── Migrations/
│       ├── Models/
│       │   ├── Todo.cs
│       │   └── User.cs
│       ├── DTOs/
│       │   ├── CreateTodoRequest.cs
│       │   ├── UpdateTodoRequest.cs
│       │   └── TodoResponse.cs
│       ├── Services/
│       │   ├── ITodoService.cs
│       │   ├── TodoService.cs
│       │   ├── IAuthService.cs
│       │   └── AuthService.cs
│       ├── Validators/
│       │   ├── CreateTodoValidator.cs
│       │   └── UpdateTodoValidator.cs
│       └── Endpoints/
│           ├── TodoEndpoints.cs
│           └── AuthEndpoints.cs
└── tests/
    └── TodoApi.Tests/
        ├── TodoApi.Tests.csproj
        └── TodoEndpointTests.cs
```

## 3. 엔드포인트 정의

### 3.1 기본 CRUD 엔드포인트

```csharp
// Endpoints/TodoEndpoints.cs
using TodoApi.DTOs;
using TodoApi.Services;

namespace TodoApi.Endpoints;

public static class TodoEndpoints
{
    public static void MapTodoEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/todos")
            .WithTags("Todos");

        // GET /api/todos
        group.MapGet("/", async (ITodoService service) =>
        {
            var todos = await service.GetAllAsync();
            return Results.Ok(todos);
        })
        .WithName("GetAllTodos")
        .Produces<List<TodoResponse>>(StatusCodes.Status200OK);

        // GET /api/todos/{id}
        group.MapGet("/{id:int}", async (int id, ITodoService service) =>
        {
            var todo = await service.GetByIdAsync(id);
            return todo is not null
                ? Results.Ok(todo)
                : Results.NotFound(new { message = $"Todo with id {id} not found" });
        })
        .WithName("GetTodoById")
        .Produces<TodoResponse>(StatusCodes.Status200OK)
        .Produces(StatusCodes.Status404NotFound);

        // POST /api/todos
        group.MapPost("/", async (CreateTodoRequest request, ITodoService service) =>
        {
            var todo = await service.CreateAsync(request);
            return Results.Created($"/api/todos/{todo.Id}", todo);
        })
        .WithName("CreateTodo")
        .Produces<TodoResponse>(StatusCodes.Status201Created)
        .Produces(StatusCodes.Status400BadRequest);

        // PUT /api/todos/{id}
        group.MapPut("/{id:int}", async (int id, UpdateTodoRequest request, ITodoService service) =>
        {
            var todo = await service.UpdateAsync(id, request);
            return todo is not null
                ? Results.Ok(todo)
                : Results.NotFound(new { message = $"Todo with id {id} not found" });
        })
        .WithName("UpdateTodo")
        .Produces<TodoResponse>(StatusCodes.Status200OK)
        .Produces(StatusCodes.Status404NotFound);

        // DELETE /api/todos/{id}
        group.MapDelete("/{id:int}", async (int id, ITodoService service) =>
        {
            var deleted = await service.DeleteAsync(id);
            return deleted
                ? Results.NoContent()
                : Results.NotFound(new { message = $"Todo with id {id} not found" });
        })
        .WithName("DeleteTodo")
        .Produces(StatusCodes.Status204NoContent)
        .Produces(StatusCodes.Status404NotFound);
    }
}
```

### 3.2 라우트 매개변수와 쿼리 문자열

```csharp
// Minimal API의 다양한 매개변수 바인딩 소스:

// 라우트 매개변수: /api/todos/{id}
app.MapGet("/api/todos/{id:int}", (int id) => $"Todo {id}");

// 쿼리 문자열: /api/todos?completed=true&page=2
app.MapGet("/api/todos", (bool? completed, int page = 1, int pageSize = 10) =>
{
    return Results.Ok(new
    {
        completed,
        page,
        pageSize,
        message = $"Filtering by completed={completed}, page {page}, size {pageSize}"
    });
});

// 헤더 바인딩
app.MapGet("/api/info", (
    [FromHeader(Name = "X-Request-Id")] string? requestId) =>
{
    return Results.Ok(new { requestId });
});

// 결합: 라우트 + 쿼리 + 본문 + 서비스
app.MapPut("/api/todos/{id:int}",
    async (
        int id,                        // 라우트에서
        [FromQuery] bool notify,       // 쿼리 문자열에서
        UpdateTodoRequest body,        // JSON 본문에서 (POST/PUT에서 자동 감지)
        ITodoService service           // DI 컨테이너에서
    ) =>
{
    var result = await service.UpdateAsync(id, body);
    if (notify && result is not null)
    {
        // 알림 전송...
    }
    return result is not null ? Results.Ok(result) : Results.NotFound();
});
```

## 4. JSON을 사용한 요청/응답

### 4.1 데이터 전송 객체 (DTO)

```csharp
// Models/Todo.cs — 데이터베이스 엔티티
namespace TodoApi.Models;

public class Todo
{
    public int Id { get; set; }
    public required string Title { get; set; }
    public string? Description { get; set; }
    public bool IsCompleted { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public DateTime? CompletedAt { get; set; }
    public int? UserId { get; set; }
    public User? User { get; set; }
}
```

```csharp
// DTOs/CreateTodoRequest.cs
namespace TodoApi.DTOs;

public record CreateTodoRequest(
    string Title,
    string? Description = null
);
```

```csharp
// DTOs/UpdateTodoRequest.cs
namespace TodoApi.DTOs;

public record UpdateTodoRequest(
    string? Title = null,
    string? Description = null,
    bool? IsCompleted = null
);
```

```csharp
// DTOs/TodoResponse.cs
namespace TodoApi.DTOs;

public record TodoResponse(
    int Id,
    string Title,
    string? Description,
    bool IsCompleted,
    DateTime CreatedAt,
    DateTime? CompletedAt
);
```

### 4.2 JSON 구성

```csharp
// Program.cs에서 — JSON 직렬화를 전역으로 구성
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
});

// 자동 직렬화/역직렬화:
// - POST 본문 JSON은 매개변수 타입으로 자동 역직렬화됨
// - Results.Ok(object)는 자동으로 JSON으로 직렬화됨
// - Content-Type: application/json이 자동으로 설정됨

// 요청 예시:
// POST /api/todos
// Content-Type: application/json
// {
//     "title": "Buy groceries",
//     "description": "Milk, eggs, bread"
// }

// 응답 예시:
// HTTP/1.1 201 Created
// Content-Type: application/json
// Location: /api/todos/1
// {
//     "id": 1,
//     "title": "Buy groceries",
//     "description": "Milk, eggs, bread",
//     "is_completed": false,
//     "created_at": "2025-01-15T10:30:00Z"
// }
```

## 5. Minimal API에서의 의존성 주입

### 5.1 서비스 등록

```csharp
// Program.cs
var builder = WebApplication.CreateBuilder(args);

// DI 컨테이너에 서비스 등록
builder.Services.AddScoped<ITodoService, TodoService>();
builder.Services.AddScoped<IAuthService, AuthService>();

// DbContext 등록
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// 유효성 검사기 등록
builder.Services.AddScoped<IValidator<CreateTodoRequest>, CreateTodoValidator>();
builder.Services.AddScoped<IValidator<UpdateTodoRequest>, UpdateTodoValidator>();

// 인증 등록 (11절에서 다룸)
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer();

builder.Services.AddAuthorization();

var app = builder.Build();
```

### 5.2 엔드포인트에서 서비스 주입

```csharp
// 서비스는 매개변수로 주입됨 — DI 컨테이너가 자동으로 해결

// 단일 서비스 주입
app.MapGet("/api/todos", async (ITodoService service) =>
    Results.Ok(await service.GetAllAsync()));

// 다중 서비스 주입
app.MapPost("/api/todos", async (
    CreateTodoRequest request,
    ITodoService todoService,
    IValidator<CreateTodoRequest> validator,
    ILogger<Program> logger) =>
{
    var validation = await validator.ValidateAsync(request);
    if (!validation.IsValid)
    {
        return Results.BadRequest(validation.Errors.Select(e => e.ErrorMessage));
    }

    logger.LogInformation("Creating todo: {Title}", request.Title);
    var todo = await todoService.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
});

// 필요할 때 HttpContext 주입
app.MapGet("/api/me", (HttpContext context) =>
{
    var userId = context.User.FindFirst("sub")?.Value;
    return Results.Ok(new { userId });
});
```

## 6. Entity Framework Core 기초

### 6.1 DbContext 설정

```csharp
// Data/TodoDbContext.cs
using Microsoft.EntityFrameworkCore;
using TodoApi.Models;

namespace TodoApi.Data;

public class TodoDbContext : DbContext
{
    public TodoDbContext(DbContextOptions<TodoDbContext> options) : base(options) { }

    public DbSet<Todo> Todos => Set<Todo>();
    public DbSet<User> Users => Set<User>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        modelBuilder.Entity<Todo>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Title).IsRequired().HasMaxLength(200);
            entity.Property(e => e.Description).HasMaxLength(1000);
            entity.Property(e => e.CreatedAt).HasDefaultValueSql("datetime('now')");

            entity.HasOne(e => e.User)
                .WithMany(u => u.Todos)
                .HasForeignKey(e => e.UserId)
                .OnDelete(DeleteBehavior.SetNull);

            entity.HasIndex(e => e.IsCompleted);
            entity.HasIndex(e => e.UserId);
        });

        modelBuilder.Entity<User>(entity =>
        {
            entity.HasKey(e => e.Id);
            entity.Property(e => e.Username).IsRequired().HasMaxLength(50);
            entity.Property(e => e.PasswordHash).IsRequired();
            entity.HasIndex(e => e.Username).IsUnique();
        });
    }
}
```

### 6.2 User 모델

```csharp
// Models/User.cs
namespace TodoApi.Models;

public class User
{
    public int Id { get; set; }
    public required string Username { get; set; }
    public required string PasswordHash { get; set; }
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public List<Todo> Todos { get; set; } = new();
}
```

### 6.3 마이그레이션

```bash
# 초기 마이그레이션 생성
dotnet ef migrations add InitialCreate --project src/TodoApi

# 마이그레이션 적용 (데이터베이스 생성)
dotnet ef database update --project src/TodoApi

# 모델이 발전함에 따라 마이그레이션 추가
dotnet ef migrations add AddUserTable --project src/TodoApi
```

### 6.4 Todo 서비스 (CRUD 연산)

```csharp
// Services/ITodoService.cs
namespace TodoApi.Services;

using TodoApi.DTOs;

public interface ITodoService
{
    Task<List<TodoResponse>> GetAllAsync();
    Task<TodoResponse?> GetByIdAsync(int id);
    Task<TodoResponse> CreateAsync(CreateTodoRequest request);
    Task<TodoResponse?> UpdateAsync(int id, UpdateTodoRequest request);
    Task<bool> DeleteAsync(int id);
}
```

```csharp
// Services/TodoService.cs
using Microsoft.EntityFrameworkCore;
using TodoApi.Data;
using TodoApi.DTOs;
using TodoApi.Models;

namespace TodoApi.Services;

public class TodoService : ITodoService
{
    private readonly TodoDbContext _db;

    public TodoService(TodoDbContext db) => _db = db;

    public async Task<List<TodoResponse>> GetAllAsync()
    {
        return await _db.Todos
            .OrderByDescending(t => t.CreatedAt)
            .Select(t => MapToResponse(t))
            .ToListAsync();
    }

    public async Task<TodoResponse?> GetByIdAsync(int id)
    {
        var todo = await _db.Todos.FindAsync(id);
        return todo is not null ? MapToResponse(todo) : null;
    }

    public async Task<TodoResponse> CreateAsync(CreateTodoRequest request)
    {
        var todo = new Todo
        {
            Title = request.Title,
            Description = request.Description,
            IsCompleted = false,
            CreatedAt = DateTime.UtcNow
        };

        _db.Todos.Add(todo);
        await _db.SaveChangesAsync();

        return MapToResponse(todo);
    }

    public async Task<TodoResponse?> UpdateAsync(int id, UpdateTodoRequest request)
    {
        var todo = await _db.Todos.FindAsync(id);
        if (todo is null) return null;

        if (request.Title is not null)
            todo.Title = request.Title;

        if (request.Description is not null)
            todo.Description = request.Description;

        if (request.IsCompleted.HasValue)
        {
            todo.IsCompleted = request.IsCompleted.Value;
            todo.CompletedAt = request.IsCompleted.Value ? DateTime.UtcNow : null;
        }

        await _db.SaveChangesAsync();
        return MapToResponse(todo);
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var todo = await _db.Todos.FindAsync(id);
        if (todo is null) return false;

        _db.Todos.Remove(todo);
        await _db.SaveChangesAsync();
        return true;
    }

    private static TodoResponse MapToResponse(Todo todo) => new(
        Id: todo.Id,
        Title: todo.Title,
        Description: todo.Description,
        IsCompleted: todo.IsCompleted,
        CreatedAt: todo.CreatedAt,
        CompletedAt: todo.CompletedAt
    );
}
```

## 7. 개발 데이터베이스로서의 SQLite

### 7.1 구성

```json
// appsettings.json
{
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=todo.db"
  },
  "Jwt": {
    "Key": "YourSuperSecretKeyThatIsAtLeast32CharactersLong!",
    "Issuer": "TodoApi",
    "Audience": "TodoApiUsers",
    "ExpirationMinutes": 60
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning",
      "Microsoft.EntityFrameworkCore.Database.Command": "Information"
    }
  }
}
```

```csharp
// Program.cs에서 — SQLite로 DbContext 등록
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));
```

### 7.2 자동 데이터베이스 생성

```csharp
// Program.cs에서 — 시작 시 데이터베이스가 생성되도록 보장 (개발 전용)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<TodoDbContext>();
    db.Database.EnsureCreated();  // DB와 테이블이 없으면 생성
    // 프로덕션에서는 다음을 사용: db.Database.Migrate();
}
```

## 8. 입력 유효성 검사

### 8.1 FluentValidation 유효성 검사기

```csharp
// Validators/CreateTodoValidator.cs
using FluentValidation;
using TodoApi.DTOs;

namespace TodoApi.Validators;

public class CreateTodoValidator : AbstractValidator<CreateTodoRequest>
{
    public CreateTodoValidator()
    {
        RuleFor(x => x.Title)
            .NotEmpty().WithMessage("Title is required")
            .MaximumLength(200).WithMessage("Title cannot exceed 200 characters")
            .MinimumLength(1).WithMessage("Title must be at least 1 character");

        RuleFor(x => x.Description)
            .MaximumLength(1000).WithMessage("Description cannot exceed 1000 characters")
            .When(x => x.Description is not null);
    }
}
```

```csharp
// Validators/UpdateTodoValidator.cs
using FluentValidation;
using TodoApi.DTOs;

namespace TodoApi.Validators;

public class UpdateTodoValidator : AbstractValidator<UpdateTodoRequest>
{
    public UpdateTodoValidator()
    {
        RuleFor(x => x.Title)
            .MaximumLength(200).WithMessage("Title cannot exceed 200 characters")
            .MinimumLength(1).WithMessage("Title must be at least 1 character")
            .When(x => x.Title is not null);

        RuleFor(x => x.Description)
            .MaximumLength(1000).WithMessage("Description cannot exceed 1000 characters")
            .When(x => x.Description is not null);
    }
}
```

### 8.2 엔드포인트에서의 유효성 검사

```csharp
// 생성 엔드포인트에 유효성 검사 추가:
group.MapPost("/", async (
    CreateTodoRequest request,
    IValidator<CreateTodoRequest> validator,
    ITodoService service) =>
{
    var validation = await validator.ValidateAsync(request);
    if (!validation.IsValid)
    {
        var errors = validation.Errors
            .GroupBy(e => e.PropertyName)
            .ToDictionary(
                g => g.Key,
                g => g.Select(e => e.ErrorMessage).ToArray()
            );
        return Results.ValidationProblem(errors);
    }

    var todo = await service.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
});

// Results.ValidationProblem은 표준 RFC 7807 응답을 생성:
// HTTP/1.1 400 Bad Request
// Content-Type: application/problem+json
// {
//     "type": "https://tools.ietf.org/html/rfc9110#section-15.5.1",
//     "title": "One or more validation errors occurred.",
//     "status": 400,
//     "errors": {
//         "Title": ["Title is required"]
//     }
// }
```

### 8.3 유효성 검사 필터 (재사용 가능)

```csharp
// 유효성 검사 로직 반복을 피하기 위한 재사용 가능한 유효성 검사 필터:
public static class ValidationFilter
{
    public static RouteHandlerBuilder WithValidation<T>(this RouteHandlerBuilder builder)
        where T : class
    {
        builder.AddEndpointFilter(async (context, next) =>
        {
            var validator = context.HttpContext.RequestServices.GetService<IValidator<T>>();
            if (validator is null)
                return await next(context);

            // T 타입의 매개변수 찾기
            var argument = context.Arguments.OfType<T>().FirstOrDefault();
            if (argument is null)
                return await next(context);

            var result = await validator.ValidateAsync(argument);
            if (!result.IsValid)
            {
                var errors = result.Errors
                    .GroupBy(e => e.PropertyName)
                    .ToDictionary(
                        g => g.Key,
                        g => g.Select(e => e.ErrorMessage).ToArray()
                    );
                return Results.ValidationProblem(errors);
            }

            return await next(context);
        });

        return builder;
    }
}

// 사용:
group.MapPost("/", async (CreateTodoRequest request, ITodoService service) =>
{
    var todo = await service.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
})
.WithValidation<CreateTodoRequest>();  // 깔끔!
```

## 9. 오류 처리 미들웨어

### 9.1 전역 예외 핸들러

```csharp
// Program.cs에서 — 전역 오류 처리 미들웨어 추가

app.UseExceptionHandler(errorApp =>
{
    errorApp.Run(async context =>
    {
        context.Response.ContentType = "application/problem+json";

        var exceptionFeature = context.Features.Get<IExceptionHandlerFeature>();
        var exception = exceptionFeature?.Error;

        var (statusCode, title, detail) = exception switch
        {
            ArgumentException argEx =>
                (StatusCodes.Status400BadRequest, "Bad Request", argEx.Message),
            KeyNotFoundException notFoundEx =>
                (StatusCodes.Status404NotFound, "Not Found", notFoundEx.Message),
            UnauthorizedAccessException =>
                (StatusCodes.Status401Unauthorized, "Unauthorized", "Authentication required"),
            _ =>
                (StatusCodes.Status500InternalServerError, "Internal Server Error",
                 "An unexpected error occurred")
        };

        context.Response.StatusCode = statusCode;

        var problem = new
        {
            type = $"https://httpstatuses.com/{statusCode}",
            title,
            status = statusCode,
            detail,
            traceId = context.TraceIdentifier
        };

        // 예외 로깅
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
        if (statusCode >= 500)
            logger.LogError(exception, "Unhandled exception: {Message}", exception?.Message);
        else
            logger.LogWarning("Handled exception: {StatusCode} {Message}", statusCode, detail);

        await context.Response.WriteAsJsonAsync(problem);
    });
});
```

### 9.2 요청 로깅 미들웨어

```csharp
// 요청/응답 로깅을 위한 사용자 지정 미들웨어
app.Use(async (context, next) =>
{
    var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
    var stopwatch = System.Diagnostics.Stopwatch.StartNew();

    logger.LogInformation("Request: {Method} {Path}",
        context.Request.Method, context.Request.Path);

    await next();

    stopwatch.Stop();
    logger.LogInformation("Response: {Method} {Path} => {StatusCode} ({Elapsed}ms)",
        context.Request.Method,
        context.Request.Path,
        context.Response.StatusCode,
        stopwatch.ElapsedMilliseconds);
});
```

## 10. JWT를 사용한 인증

### 10.1 JWT 구성

```csharp
// Program.cs에서 — JWT 인증 구성
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.IdentityModel.Tokens;
using System.Text;

var jwtSettings = builder.Configuration.GetSection("Jwt");
var key = Encoding.UTF8.GetBytes(jwtSettings["Key"]!);

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = jwtSettings["Issuer"],
            ValidAudience = jwtSettings["Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(key),
            ClockSkew = TimeSpan.Zero  // 만료된 토큰에 대한 허용 오차 없음
        };
    });

builder.Services.AddAuthorization();

// ... 앱 빌드 후:
app.UseAuthentication();
app.UseAuthorization();
```

### 10.2 인증 서비스

```csharp
// Services/IAuthService.cs
namespace TodoApi.Services;

public interface IAuthService
{
    Task<string?> RegisterAsync(string username, string password);
    Task<string?> LoginAsync(string username, string password);
}
```

```csharp
// Services/AuthService.cs
using System.IdentityModel.Tokens.Jwt;
using System.Security.Claims;
using System.Security.Cryptography;
using System.Text;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using TodoApi.Data;
using TodoApi.Models;

namespace TodoApi.Services;

public class AuthService : IAuthService
{
    private readonly TodoDbContext _db;
    private readonly IConfiguration _config;

    public AuthService(TodoDbContext db, IConfiguration config)
    {
        _db = db;
        _config = config;
    }

    public async Task<string?> RegisterAsync(string username, string password)
    {
        if (await _db.Users.AnyAsync(u => u.Username == username))
            return null;  // 사용자 이름이 이미 존재

        var user = new User
        {
            Username = username,
            PasswordHash = HashPassword(password)
        };

        _db.Users.Add(user);
        await _db.SaveChangesAsync();

        return GenerateToken(user);
    }

    public async Task<string?> LoginAsync(string username, string password)
    {
        var user = await _db.Users.FirstOrDefaultAsync(u => u.Username == username);
        if (user is null || !VerifyPassword(password, user.PasswordHash))
            return null;

        return GenerateToken(user);
    }

    private string GenerateToken(User user)
    {
        var key = new SymmetricSecurityKey(
            Encoding.UTF8.GetBytes(_config["Jwt:Key"]!));
        var credentials = new SigningCredentials(key, SecurityAlgorithms.HmacSha256);

        var claims = new[]
        {
            new Claim(JwtRegisteredClaimNames.Sub, user.Id.ToString()),
            new Claim(JwtRegisteredClaimNames.UniqueName, user.Username),
            new Claim(JwtRegisteredClaimNames.Jti, Guid.NewGuid().ToString())
        };

        var expiration = int.Parse(_config["Jwt:ExpirationMinutes"] ?? "60");

        var token = new JwtSecurityToken(
            issuer: _config["Jwt:Issuer"],
            audience: _config["Jwt:Audience"],
            claims: claims,
            expires: DateTime.UtcNow.AddMinutes(expiration),
            signingCredentials: credentials
        );

        return new JwtSecurityTokenHandler().WriteToken(token);
    }

    private static string HashPassword(string password)
    {
        var salt = RandomNumberGenerator.GetBytes(16);
        var hash = Rfc2898DeriveBytes.Pbkdf2(
            Encoding.UTF8.GetBytes(password), salt, 100_000, HashAlgorithmName.SHA256, 32);
        return $"{Convert.ToBase64String(salt)}.{Convert.ToBase64String(hash)}";
    }

    private static bool VerifyPassword(string password, string storedHash)
    {
        var parts = storedHash.Split('.');
        if (parts.Length != 2) return false;
        var salt = Convert.FromBase64String(parts[0]);
        var hash = Convert.FromBase64String(parts[1]);
        var computedHash = Rfc2898DeriveBytes.Pbkdf2(
            Encoding.UTF8.GetBytes(password), salt, 100_000, HashAlgorithmName.SHA256, 32);
        return CryptographicOperations.FixedTimeEquals(hash, computedHash);
    }
}
```

### 10.3 인증 엔드포인트

```csharp
// Endpoints/AuthEndpoints.cs
namespace TodoApi.Endpoints;

using TodoApi.Services;

public static class AuthEndpoints
{
    public static void MapAuthEndpoints(this WebApplication app)
    {
        var group = app.MapGroup("/api/auth").WithTags("Authentication");

        group.MapPost("/register", async (RegisterRequest request, IAuthService auth) =>
        {
            if (string.IsNullOrWhiteSpace(request.Username) || request.Username.Length < 3)
                return Results.BadRequest(new { message = "Username must be at least 3 characters" });
            if (string.IsNullOrWhiteSpace(request.Password) || request.Password.Length < 8)
                return Results.BadRequest(new { message = "Password must be at least 8 characters" });

            var token = await auth.RegisterAsync(request.Username, request.Password);
            return token is not null
                ? Results.Ok(new { token })
                : Results.Conflict(new { message = "Username already exists" });
        });

        group.MapPost("/login", async (LoginRequest request, IAuthService auth) =>
        {
            var token = await auth.LoginAsync(request.Username, request.Password);
            return token is not null
                ? Results.Ok(new { token })
                : Results.Unauthorized();
        });
    }
}

public record RegisterRequest(string Username, string Password);
public record LoginRequest(string Username, string Password);
```

### 10.4 엔드포인트 보호

```csharp
// 특정 엔드포인트에 인증 요구:
group.MapPost("/", async (CreateTodoRequest request, ITodoService service) =>
{
    var todo = await service.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
})
.RequireAuthorization();  // 유효한 JWT 필요

// 또는 전체 그룹 보호:
var protectedGroup = app.MapGroup("/api/todos")
    .WithTags("Todos")
    .RequireAuthorization();

// 특정 엔드포인트에 익명 접근 허용:
protectedGroup.MapGet("/", async (ITodoService service) =>
    Results.Ok(await service.GetAllAsync()))
.AllowAnonymous();  // 이 엔드포인트에 대해 그룹 수준 인증 재정의
```

## 11. WebApplicationFactory로 API 테스트

### 11.1 테스트 설정

```csharp
// tests/TodoApi.Tests/TodoEndpointTests.cs
using System.Net;
using System.Net.Http.Json;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using TodoApi.Data;
using TodoApi.DTOs;

namespace TodoApi.Tests;

public class TodoEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;
    private readonly WebApplicationFactory<Program> _factory;

    public TodoEndpointTests(WebApplicationFactory<Program> factory)
    {
        // 테스트 서버 커스터마이징
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                // 실제 데이터베이스 등록 제거
                var descriptor = services.SingleOrDefault(
                    d => d.ServiceType == typeof(DbContextOptions<TodoDbContext>));
                if (descriptor != null) services.Remove(descriptor);

                // 테스트용 인메모리 데이터베이스 추가
                services.AddDbContext<TodoDbContext>(options =>
                    options.UseInMemoryDatabase("TestDb_" + Guid.NewGuid()));
            });
        });

        _client = _factory.CreateClient();
    }
}
```

### 11.2 CRUD 테스트

```csharp
public class TodoEndpointTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;
    private readonly WebApplicationFactory<Program> _factory;

    public TodoEndpointTests(WebApplicationFactory<Program> factory)
    {
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                var descriptor = services.SingleOrDefault(
                    d => d.ServiceType == typeof(DbContextOptions<TodoDbContext>));
                if (descriptor != null) services.Remove(descriptor);

                services.AddDbContext<TodoDbContext>(options =>
                    options.UseInMemoryDatabase("TestDb_" + Guid.NewGuid()));
            });
        });

        _client = _factory.CreateClient();
    }

    [Fact]
    public async Task GetAll_ReturnsEmptyList_WhenNoTodos()
    {
        var response = await _client.GetAsync("/api/todos");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var todos = await response.Content.ReadFromJsonAsync<List<TodoResponse>>();
        todos.Should().BeEmpty();
    }

    [Fact]
    public async Task Create_ReturnsCreatedTodo()
    {
        var request = new CreateTodoRequest("Buy groceries", "Milk, eggs, bread");

        var response = await _client.PostAsJsonAsync("/api/todos", request);

        response.StatusCode.Should().Be(HttpStatusCode.Created);
        var todo = await response.Content.ReadFromJsonAsync<TodoResponse>();
        todo.Should().NotBeNull();
        todo!.Title.Should().Be("Buy groceries");
        todo.Description.Should().Be("Milk, eggs, bread");
        todo.IsCompleted.Should().BeFalse();
        todo.Id.Should().BeGreaterThan(0);

        // Location 헤더 확인
        response.Headers.Location.Should().NotBeNull();
        response.Headers.Location!.PathAndQuery.Should().Be($"/api/todos/{todo.Id}");
    }

    [Fact]
    public async Task Create_ReturnsBadRequest_WhenTitleIsEmpty()
    {
        var request = new CreateTodoRequest("");

        var response = await _client.PostAsJsonAsync("/api/todos", request);

        response.StatusCode.Should().Be(HttpStatusCode.BadRequest);
    }

    [Fact]
    public async Task GetById_ReturnsNotFound_WhenTodoDoesNotExist()
    {
        var response = await _client.GetAsync("/api/todos/999");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task GetById_ReturnsTodo_WhenExists()
    {
        // 먼저 todo 생성
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("Test Todo"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // 조회
        var response = await _client.GetAsync($"/api/todos/{created!.Id}");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var todo = await response.Content.ReadFromJsonAsync<TodoResponse>();
        todo!.Title.Should().Be("Test Todo");
    }

    [Fact]
    public async Task Update_ModifiesTodo()
    {
        // 생성
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("Original Title"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // 수정
        var updateRequest = new UpdateTodoRequest(
            Title: "Updated Title",
            IsCompleted: true);
        var response = await _client.PutAsJsonAsync($"/api/todos/{created!.Id}", updateRequest);

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var updated = await response.Content.ReadFromJsonAsync<TodoResponse>();
        updated!.Title.Should().Be("Updated Title");
        updated.IsCompleted.Should().BeTrue();
        updated.CompletedAt.Should().NotBeNull();
    }

    [Fact]
    public async Task Delete_RemovesTodo()
    {
        // 생성
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("To Delete"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // 삭제
        var deleteResponse = await _client.DeleteAsync($"/api/todos/{created!.Id}");
        deleteResponse.StatusCode.Should().Be(HttpStatusCode.NoContent);

        // 삭제되었는지 확인
        var getResponse = await _client.GetAsync($"/api/todos/{created.Id}");
        getResponse.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }

    [Fact]
    public async Task Delete_ReturnsNotFound_WhenTodoDoesNotExist()
    {
        var response = await _client.DeleteAsync("/api/todos/999");

        response.StatusCode.Should().Be(HttpStatusCode.NotFound);
    }
}
```

## 12. 프로젝트 구조 모범 사례

### 12.1 완전한 Program.cs

```csharp
// Program.cs — 전체 애플리케이션 연결
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using FluentValidation;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.AspNetCore.Diagnostics;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using TodoApi.Data;
using TodoApi.DTOs;
using TodoApi.Endpoints;
using TodoApi.Services;
using TodoApi.Validators;

var builder = WebApplication.CreateBuilder(args);

// --- 서비스 ---

// 데이터베이스
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// 애플리케이션 서비스
builder.Services.AddScoped<ITodoService, TodoService>();
builder.Services.AddScoped<IAuthService, AuthService>();

// 유효성 검사
builder.Services.AddScoped<IValidator<CreateTodoRequest>, CreateTodoValidator>();
builder.Services.AddScoped<IValidator<UpdateTodoRequest>, UpdateTodoValidator>();

// JSON 옵션
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

// 인증
var jwtKey = builder.Configuration["Jwt:Key"]
    ?? throw new InvalidOperationException("JWT Key not configured");

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(Encoding.UTF8.GetBytes(jwtKey)),
            ClockSkew = TimeSpan.Zero
        };
    });

builder.Services.AddAuthorization();

// --- 앱 ---
var app = builder.Build();

// 미들웨어 파이프라인
app.UseExceptionHandler(errorApp =>
{
    errorApp.Run(async context =>
    {
        context.Response.ContentType = "application/problem+json";
        var exception = context.Features.Get<IExceptionHandlerFeature>()?.Error;
        var statusCode = exception switch
        {
            ArgumentException => StatusCodes.Status400BadRequest,
            KeyNotFoundException => StatusCodes.Status404NotFound,
            _ => StatusCodes.Status500InternalServerError
        };
        context.Response.StatusCode = statusCode;
        await context.Response.WriteAsJsonAsync(new
        {
            status = statusCode,
            title = exception?.GetType().Name ?? "Error",
            detail = statusCode < 500 ? exception?.Message : "An unexpected error occurred",
            traceId = context.TraceIdentifier
        });
    });
});

app.UseAuthentication();
app.UseAuthorization();

// 엔드포인트
app.MapAuthEndpoints();
app.MapTodoEndpoints();

// 헬스 체크
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

// 데이터베이스 생성 보장
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<TodoDbContext>();
    db.Database.EnsureCreated();
}

app.Run();

// 테스트에서 WebApplicationFactory를 위해 Program 접근 가능하게 함
public partial class Program { }
```

### 12.2 API 실행

```bash
# API 시작
cd src/TodoApi
dotnet run

# curl로 테스트:

# 헬스 체크
curl http://localhost:5000/health

# 사용자 등록
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "password123"}'

# 로그인 후 토큰 얻기
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "password123"}' | jq -r '.token')

# Todo 생성 (인증 포함)
curl -X POST http://localhost:5000/api/todos \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title": "Learn C#", "description": "Complete the advanced course"}'

# 모든 Todo 조회
curl http://localhost:5000/api/todos

# Todo 수정
curl -X PUT http://localhost:5000/api/todos/1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"isCompleted": true}'

# Todo 삭제
curl -X DELETE http://localhost:5000/api/todos/1 \
  -H "Authorization: Bearer $TOKEN"
```

### 12.3 테스트 실행

```bash
# 모든 테스트 실행
cd tests/TodoApi.Tests
dotnet test

# 자세한 출력으로 실행
dotnet test --verbosity normal

# 특정 테스트 실행
dotnet test --filter "Create_ReturnsCreatedTodo"
```

## 13. 연습 문제

1. **페이지네이션 엔드포인트**: `GET /api/todos` 엔드포인트에 페이지네이션을 추가하세요. `page` (기본값 1)와 `pageSize` (기본값 10, 최대 100) 쿼리 매개변수를 받아야 합니다. `items`, `totalCount`, `page`, `pageSize`, `totalPages`가 포함된 응답을 반환하세요. 25개의 todo를 생성하고 크기 10의 페이지 2가 정확히 10개의 항목을 반환하는지 확인하는 테스트를 작성하세요.

2. **필터링과 정렬**: `GET /api/todos` 엔드포인트를 확장하여 선택적 쿼리 매개변수를 받으세요: `completed` (bool), `search` (string, 제목 일치), `sortBy` (string: "created", "title"). 서비스 메서드와 최소 두 개의 통합 테스트를 작성하세요.

3. **속도 제한**: IP 주소당 분당 최대 100개의 요청을 허용하는 간단한 인메모리 속도 제한 미들웨어를 추가하세요. 제한을 초과하면 `429 Too Many Requests`를 반환하세요. 101개의 요청을 보내고 101번째가 거부되는지 확인하는 테스트를 작성하세요.

4. **일괄 연산**: `CreateTodoRequest` 객체의 목록(최대 50개)을 받아 단일 데이터베이스 트랜잭션에서 모두 생성하는 `POST /api/todos/batch` 엔드포인트를 추가하세요. 유효성 검사에 실패하면 아무것도 생성되지 않아야 합니다. 생성된 todo 목록을 반환하세요. 성공 및 유효성 검사 실패 케이스 모두에 대한 테스트를 작성하세요.

5. **전체 기능 확장**: Todo와 다대다 관계를 가진 `Tag` 모델을 추가하세요. 다음 엔드포인트를 생성하세요: (a) todo에 태그 추가 (`POST /api/todos/{id}/tags`), (b) todo에서 태그 제거 (`DELETE /api/todos/{id}/tags/{tagId}`), (c) 태그로 todo 필터링 (`GET /api/todos?tag=urgent`). EF Core 구성, 마이그레이션 명령, 서비스 메서드, 최소 세 개의 통합 테스트를 포함하세요.
