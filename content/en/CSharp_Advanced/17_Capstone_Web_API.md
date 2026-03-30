# Capstone: Minimal Web API

**Previous**: [Performance and Profiling](./16_Performance_Profiling.md) | **Next**: [CSharp Advanced Overview](./00_Overview.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Create a minimal Web API project using ASP.NET Core
2. Define RESTful endpoints with `MapGet`, `MapPost`, `MapPut`, and `MapDelete`
3. Handle route parameters, query strings, and JSON request/response bodies
4. Register and inject dependencies using the built-in DI container
5. Set up Entity Framework Core with SQLite for data persistence
6. Validate input and handle errors with middleware
7. Implement basic JWT authentication
8. Write integration tests using `WebApplicationFactory`
9. Organize a production-quality API project structure

---

This capstone lesson brings together concepts from the entire course to build a complete, working Web API. Minimal APIs, introduced in .NET 6, provide a streamlined way to build HTTP services without the ceremony of controllers. We will build a Todo API from scratch: defining endpoints, persisting data with Entity Framework Core and SQLite, validating input, handling errors, adding authentication, and writing tests. By the end, you will have a reference implementation that demonstrates real-world API development patterns.

## 1. ASP.NET Core Minimal API Overview

### 1.1 What Are Minimal APIs?

Minimal APIs allow you to define HTTP endpoints with minimal boilerplate. Instead of controllers, attributes, and complex startup classes, you map routes directly in `Program.cs`:

```csharp
// The simplest possible ASP.NET Core application
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "Hello, World!");

app.Run();
```

This is a fully functional web server. The `builder` configures services (dependency injection, logging, configuration), and `app` defines the request pipeline (middleware and endpoints).

### 1.2 Minimal APIs vs Controllers

```csharp
// Minimal API style:
app.MapGet("/api/products/{id}", async (int id, ProductService service) =>
{
    var product = await service.GetByIdAsync(id);
    return product is not null ? Results.Ok(product) : Results.NotFound();
});

// Controller style (for comparison — NOT used in this lesson):
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

// When to use minimal APIs:
// - Microservices with a small number of endpoints
// - Rapid prototyping
// - Simple CRUD APIs
// - When you prefer explicit routing over convention-based

// When controllers might be better:
// - Large APIs with many endpoints
// - When you need filters (action filters, result filters)
// - When the team is familiar with MVC patterns
```

## 2. Creating a New Web API Project

### 2.1 Project Setup

```bash
# Create the solution structure
mkdir TodoApi && cd TodoApi
dotnet new sln

# Create the API project
dotnet new webapi -o src/TodoApi --use-minimal-apis
dotnet sln add src/TodoApi

# Create the test project
dotnet new xunit -o tests/TodoApi.Tests
dotnet sln add tests/TodoApi.Tests
dotnet add tests/TodoApi.Tests reference src/TodoApi

# Add required packages to the API project
cd src/TodoApi
dotnet add package Microsoft.EntityFrameworkCore.Sqlite
dotnet add package Microsoft.EntityFrameworkCore.Design
dotnet add package Microsoft.AspNetCore.Authentication.JwtBearer
dotnet add package FluentValidation

# Add test packages
cd ../../tests/TodoApi.Tests
dotnet add package Microsoft.AspNetCore.Mvc.Testing
dotnet add package FluentAssertions
```

### 2.2 Project Structure

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

## 3. Defining Endpoints

### 3.1 Basic CRUD Endpoints

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

### 3.2 Route Parameters and Query Strings

```csharp
// Different parameter binding sources in minimal APIs:

// Route parameters: /api/todos/{id}
app.MapGet("/api/todos/{id:int}", (int id) => $"Todo {id}");

// Query strings: /api/todos?completed=true&page=2
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

// Header binding
app.MapGet("/api/info", (
    [FromHeader(Name = "X-Request-Id")] string? requestId) =>
{
    return Results.Ok(new { requestId });
});

// Combined: route + query + body + service
app.MapPut("/api/todos/{id:int}",
    async (
        int id,                        // From route
        [FromQuery] bool notify,       // From query string
        UpdateTodoRequest body,        // From JSON body (auto-detected for POST/PUT)
        ITodoService service           // From DI container
    ) =>
{
    var result = await service.UpdateAsync(id, body);
    if (notify && result is not null)
    {
        // Send notification...
    }
    return result is not null ? Results.Ok(result) : Results.NotFound();
});
```

## 4. Request/Response with JSON

### 4.1 Data Transfer Objects (DTOs)

```csharp
// Models/Todo.cs — Database entity
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

### 4.2 JSON Configuration

```csharp
// In Program.cs — Configure JSON serialization globally
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower;
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
    options.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
});

// Automatic serialization/deserialization:
// - POST body JSON is automatically deserialized to the parameter type
// - Results.Ok(object) automatically serializes to JSON
// - Content-Type: application/json is set automatically

// Example request:
// POST /api/todos
// Content-Type: application/json
// {
//     "title": "Buy groceries",
//     "description": "Milk, eggs, bread"
// }

// Example response:
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

## 5. Dependency Injection in Minimal APIs

### 5.1 Registering Services

```csharp
// Program.cs
var builder = WebApplication.CreateBuilder(args);

// Register services with the DI container
builder.Services.AddScoped<ITodoService, TodoService>();
builder.Services.AddScoped<IAuthService, AuthService>();

// Register DbContext
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// Register validators
builder.Services.AddScoped<IValidator<CreateTodoRequest>, CreateTodoValidator>();
builder.Services.AddScoped<IValidator<UpdateTodoRequest>, UpdateTodoValidator>();

// Register authentication (covered in section 11)
builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer();

builder.Services.AddAuthorization();

var app = builder.Build();
```

### 5.2 Injecting Services in Endpoints

```csharp
// Services are injected as parameters — the DI container resolves them automatically

// Single service injection
app.MapGet("/api/todos", async (ITodoService service) =>
    Results.Ok(await service.GetAllAsync()));

// Multiple service injection
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

// Inject HttpContext when needed
app.MapGet("/api/me", (HttpContext context) =>
{
    var userId = context.User.FindFirst("sub")?.Value;
    return Results.Ok(new { userId });
});
```

## 6. Entity Framework Core Basics

### 6.1 DbContext Setup

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

### 6.2 User Model

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

### 6.3 Migrations

```bash
# Create an initial migration
dotnet ef migrations add InitialCreate --project src/TodoApi

# Apply the migration (creates the database)
dotnet ef database update --project src/TodoApi

# Add more migrations as the model evolves
dotnet ef migrations add AddUserTable --project src/TodoApi
```

### 6.4 The Todo Service (CRUD Operations)

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

## 7. SQLite as Development Database

### 7.1 Configuration

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
// In Program.cs — register the DbContext with SQLite
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));
```

### 7.2 Automatic Database Creation

```csharp
// In Program.cs — ensure the database is created on startup (development only)
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<TodoDbContext>();
    db.Database.EnsureCreated();  // Creates DB and tables if they don't exist
    // In production, use: db.Database.Migrate();
}
```

## 8. Input Validation

### 8.1 FluentValidation Validators

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

### 8.2 Validation in Endpoints

```csharp
// Adding validation to the create endpoint:
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

// Results.ValidationProblem produces a standard RFC 7807 response:
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

### 8.3 Validation Filter (Reusable)

```csharp
// A reusable validation filter to avoid repeating validation logic:
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

            // Find the parameter of type T
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

// Usage:
group.MapPost("/", async (CreateTodoRequest request, ITodoService service) =>
{
    var todo = await service.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
})
.WithValidation<CreateTodoRequest>();  // Clean!
```

## 9. Error Handling Middleware

### 9.1 Global Exception Handler

```csharp
// In Program.cs — add global error handling middleware

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

        // Log the exception
        var logger = context.RequestServices.GetRequiredService<ILogger<Program>>();
        if (statusCode >= 500)
            logger.LogError(exception, "Unhandled exception: {Message}", exception?.Message);
        else
            logger.LogWarning("Handled exception: {StatusCode} {Message}", statusCode, detail);

        await context.Response.WriteAsJsonAsync(problem);
    });
});
```

### 9.2 Request Logging Middleware

```csharp
// Custom middleware for request/response logging
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

## 10. Authentication with JWT

### 10.1 JWT Configuration

```csharp
// In Program.cs — configure JWT authentication
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
            ClockSkew = TimeSpan.Zero  // No tolerance for expired tokens
        };
    });

builder.Services.AddAuthorization();

// ... after building the app:
app.UseAuthentication();
app.UseAuthorization();
```

### 10.2 Auth Service

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
            return null;  // Username already exists

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

### 10.3 Auth Endpoints

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

### 10.4 Protecting Endpoints

```csharp
// Require authentication on specific endpoints:
group.MapPost("/", async (CreateTodoRequest request, ITodoService service) =>
{
    var todo = await service.CreateAsync(request);
    return Results.Created($"/api/todos/{todo.Id}", todo);
})
.RequireAuthorization();  // Requires a valid JWT

// Or protect the entire group:
var protectedGroup = app.MapGroup("/api/todos")
    .WithTags("Todos")
    .RequireAuthorization();

// Allow anonymous access for specific endpoints:
protectedGroup.MapGet("/", async (ITodoService service) =>
    Results.Ok(await service.GetAllAsync()))
.AllowAnonymous();  // Override group-level auth for this endpoint
```

## 11. Testing the API with WebApplicationFactory

### 11.1 Test Setup

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
        // Customize the test server
        _factory = factory.WithWebHostBuilder(builder =>
        {
            builder.ConfigureServices(services =>
            {
                // Remove the real database registration
                var descriptor = services.SingleOrDefault(
                    d => d.ServiceType == typeof(DbContextOptions<TodoDbContext>));
                if (descriptor != null) services.Remove(descriptor);

                // Add in-memory database for testing
                services.AddDbContext<TodoDbContext>(options =>
                    options.UseInMemoryDatabase("TestDb_" + Guid.NewGuid()));
            });
        });

        _client = _factory.CreateClient();
    }
}
```

### 11.2 CRUD Tests

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

        // Verify Location header
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
        // Create a todo first
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("Test Todo"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // Fetch it
        var response = await _client.GetAsync($"/api/todos/{created!.Id}");

        response.StatusCode.Should().Be(HttpStatusCode.OK);
        var todo = await response.Content.ReadFromJsonAsync<TodoResponse>();
        todo!.Title.Should().Be("Test Todo");
    }

    [Fact]
    public async Task Update_ModifiesTodo()
    {
        // Create
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("Original Title"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // Update
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
        // Create
        var createResponse = await _client.PostAsJsonAsync("/api/todos",
            new CreateTodoRequest("To Delete"));
        var created = await createResponse.Content.ReadFromJsonAsync<TodoResponse>();

        // Delete
        var deleteResponse = await _client.DeleteAsync($"/api/todos/{created!.Id}");
        deleteResponse.StatusCode.Should().Be(HttpStatusCode.NoContent);

        // Verify it is gone
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

## 12. Project Structure Best Practices

### 12.1 Complete Program.cs

```csharp
// Program.cs — the entire application wired together
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

// --- Services ---

// Database
builder.Services.AddDbContext<TodoDbContext>(options =>
    options.UseSqlite(builder.Configuration.GetConnectionString("DefaultConnection")));

// Application services
builder.Services.AddScoped<ITodoService, TodoService>();
builder.Services.AddScoped<IAuthService, AuthService>();

// Validation
builder.Services.AddScoped<IValidator<CreateTodoRequest>, CreateTodoValidator>();
builder.Services.AddScoped<IValidator<UpdateTodoRequest>, UpdateTodoValidator>();

// JSON options
builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});

// Authentication
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

// --- App ---
var app = builder.Build();

// Middleware pipeline
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

// Endpoints
app.MapAuthEndpoints();
app.MapTodoEndpoints();

// Health check
app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

// Ensure database is created
using (var scope = app.Services.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<TodoDbContext>();
    db.Database.EnsureCreated();
}

app.Run();

// Make Program accessible for WebApplicationFactory in tests
public partial class Program { }
```

### 12.2 Running the API

```bash
# Start the API
cd src/TodoApi
dotnet run

# Test with curl:

# Health check
curl http://localhost:5000/health

# Register a user
curl -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "password123"}'

# Login and get a token
TOKEN=$(curl -s -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "alice", "password": "password123"}' | jq -r '.token')

# Create a todo (with auth)
curl -X POST http://localhost:5000/api/todos \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title": "Learn C#", "description": "Complete the advanced course"}'

# Get all todos
curl http://localhost:5000/api/todos

# Update a todo
curl -X PUT http://localhost:5000/api/todos/1 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"isCompleted": true}'

# Delete a todo
curl -X DELETE http://localhost:5000/api/todos/1 \
  -H "Authorization: Bearer $TOKEN"
```

### 12.3 Running Tests

```bash
# Run all tests
cd tests/TodoApi.Tests
dotnet test

# Run with verbose output
dotnet test --verbosity normal

# Run specific test
dotnet test --filter "Create_ReturnsCreatedTodo"
```

## 13. Practice Problems

1. **Pagination Endpoint**: Add pagination to the `GET /api/todos` endpoint. It should accept `page` (default 1) and `pageSize` (default 10, max 100) query parameters. Return a response with `items`, `totalCount`, `page`, `pageSize`, and `totalPages`. Write a test that creates 25 todos and verifies that page 2 with size 10 returns exactly 10 items.

2. **Filtering and Sorting**: Extend the `GET /api/todos` endpoint to accept optional query parameters: `completed` (bool), `search` (string, matches title), and `sortBy` (string: "created", "title"). Write the service method and at least two integration tests.

3. **Rate Limiting**: Add a simple in-memory rate limiter middleware that allows at most 100 requests per minute per IP address. Return `429 Too Many Requests` when the limit is exceeded. Write a test that sends 101 requests and verifies the 101st is rejected.

4. **Batch Operations**: Add a `POST /api/todos/batch` endpoint that accepts a list of `CreateTodoRequest` objects (max 50) and creates them all in a single database transaction. If any validation fails, none should be created. Return the list of created todos. Write tests for both success and validation failure cases.

5. **Full Feature Extension**: Add a `Tag` model with a many-to-many relationship to `Todo`. Create endpoints to: (a) add tags to a todo (`POST /api/todos/{id}/tags`), (b) remove a tag from a todo (`DELETE /api/todos/{id}/tags/{tagId}`), and (c) filter todos by tag (`GET /api/todos?tag=urgent`). Include EF Core configuration, migration commands, service methods, and at least three integration tests.
