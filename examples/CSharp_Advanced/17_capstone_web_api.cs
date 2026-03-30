// Lesson 17: Capstone — Minimal Web API
// Run: dotnet run
// Note: Requires the Web SDK. Create project with:
//   dotnet new web -n CapstoneApi
//   Then replace Program.cs with this file.
//
// Alternatively, this file runs as a standalone demo of the patterns.
// For the full Minimal API, use Microsoft.NET.Sdk.Web project.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;

// ============================================================
// Since this example may run without ASP.NET SDK,
// we demonstrate the patterns with a simulation layer.
// In a real Minimal API project, replace the simulation
// with the actual WebApplication code shown in comments.
// ============================================================

Console.WriteLine("=== Capstone: Minimal Web API Patterns ===\n");

// ============================================================
// 1. Minimal API Structure (actual code for Web SDK project)
// ============================================================

Console.WriteLine("--- Minimal API Code (for Microsoft.NET.Sdk.Web) ---");
Console.WriteLine("""
    // Program.cs — complete Minimal API

    var builder = WebApplication.CreateBuilder(args);

    // Register services (DI)
    builder.Services.AddSingleton<ITodoRepository, InMemoryTodoRepository>();
    builder.Services.AddEndpointsApiExplorer();
    builder.Services.AddSwaggerGen();

    var app = builder.Build();

    // Middleware pipeline
    if (app.Environment.IsDevelopment())
    {
        app.UseSwagger();
        app.UseSwaggerUI();
    }
    app.UseHttpsRedirection();

    // Map endpoints
    var todos = app.MapGroup("/api/todos").WithTags("Todos");

    todos.MapGet("/", (ITodoRepository repo) =>
        Results.Ok(repo.GetAll()));

    todos.MapGet("/{id:int}", (int id, ITodoRepository repo) =>
        repo.GetById(id) is Todo todo
            ? Results.Ok(todo)
            : Results.NotFound());

    todos.MapPost("/", (CreateTodoRequest req, ITodoRepository repo) =>
    {
        var todo = repo.Create(req.Title, req.DueDate);
        return Results.Created($"/api/todos/{todo.Id}", todo);
    });

    todos.MapPut("/{id:int}", (int id, UpdateTodoRequest req, ITodoRepository repo) =>
        repo.Update(id, req.Title, req.IsComplete, req.DueDate)
            ? Results.NoContent()
            : Results.NotFound());

    todos.MapDelete("/{id:int}", (int id, ITodoRepository repo) =>
        repo.Delete(id) ? Results.NoContent() : Results.NotFound());

    // Health check endpoint
    app.MapGet("/health", () => Results.Ok(new { status = "healthy", timestamp = DateTime.UtcNow }));

    app.Run();
    """);

// ============================================================
// 2. Domain Models
// ============================================================

Console.WriteLine("\n--- Domain Models & Repository ---\n");

var repo = new InMemoryTodoRepository();

// Create some todos
var todo1 = repo.Create("Learn C# Minimal APIs", DateTime.Now.AddDays(7));
var todo2 = repo.Create("Build a REST API", DateTime.Now.AddDays(14));
var todo3 = repo.Create("Deploy to Azure", DateTime.Now.AddDays(30));

Console.WriteLine("Created todos:");
foreach (var t in repo.GetAll())
    Console.WriteLine($"  {t}");

// ============================================================
// 3. Simulating API Requests
// ============================================================

Console.WriteLine("\n--- Simulating API Endpoints ---\n");

// GET /api/todos
Console.WriteLine("[GET /api/todos]");
var allTodos = repo.GetAll();
Console.WriteLine($"  Response: {JsonSerializer.Serialize(allTodos, jsonOptions)}");

// GET /api/todos/1
Console.WriteLine("\n[GET /api/todos/1]");
var found = repo.GetById(1);
if (found is not null)
    Console.WriteLine($"  200 OK: {JsonSerializer.Serialize(found, jsonOptions)}");
else
    Console.WriteLine("  404 Not Found");

// POST /api/todos
Console.WriteLine("\n[POST /api/todos]");
var newTodo = repo.Create("Write unit tests", DateTime.Now.AddDays(3));
Console.WriteLine($"  201 Created: {JsonSerializer.Serialize(newTodo, jsonOptions)}");

// PUT /api/todos/1
Console.WriteLine("\n[PUT /api/todos/1]");
bool updated = repo.Update(1, "Learn C# Minimal APIs (done)", true, null);
Console.WriteLine($"  {(updated ? "204 No Content" : "404 Not Found")}");
Console.WriteLine($"  Updated: {JsonSerializer.Serialize(repo.GetById(1), jsonOptions)}");

// DELETE /api/todos/2
Console.WriteLine("\n[DELETE /api/todos/2]");
bool deleted = repo.Delete(2);
Console.WriteLine($"  {(deleted ? "204 No Content" : "404 Not Found")}");

// GET /api/todos (after modifications)
Console.WriteLine("\n[GET /api/todos] (final state)");
foreach (var t in repo.GetAll())
    Console.WriteLine($"  {JsonSerializer.Serialize(t, jsonOptions)}");

// GET /api/todos/999 (not found)
Console.WriteLine("\n[GET /api/todos/999]");
Console.WriteLine($"  {(repo.GetById(999) is not null ? "200 OK" : "404 Not Found")}");

// ============================================================
// 4. Middleware Pattern
// ============================================================

Console.WriteLine("\n--- Middleware Pipeline Pattern ---\n");

// Simulate middleware pipeline
var pipeline = new MiddlewarePipeline();
pipeline.Use("Logging", ctx =>
{
    Console.WriteLine($"  [Logging] {ctx.Method} {ctx.Path}");
    return true; // Continue to next middleware
});
pipeline.Use("Authentication", ctx =>
{
    if (ctx.Path.StartsWith("/admin") && !ctx.Headers.ContainsKey("Authorization"))
    {
        Console.WriteLine("  [Auth] 401 Unauthorized");
        return false; // Short-circuit
    }
    Console.WriteLine("  [Auth] OK");
    return true;
});
pipeline.Use("Endpoint", ctx =>
{
    Console.WriteLine($"  [Endpoint] Handling {ctx.Path}");
    return true;
});

Console.WriteLine("Request 1: GET /api/todos");
pipeline.Execute(new RequestContext("GET", "/api/todos"));

Console.WriteLine("\nRequest 2: GET /admin/users (no auth)");
pipeline.Execute(new RequestContext("GET", "/admin/users"));

Console.WriteLine("\nRequest 3: GET /admin/users (with auth)");
var authedCtx = new RequestContext("GET", "/admin/users");
authedCtx.Headers["Authorization"] = "Bearer token123";
pipeline.Execute(authedCtx);

// ============================================================
// 5. Validation Pattern
// ============================================================

Console.WriteLine("\n--- Request Validation ---\n");

var validRequest = new CreateTodoRequest("Buy groceries", DateTime.Now.AddDays(1));
var invalidRequest = new CreateTodoRequest("", null);

Console.WriteLine($"Valid request:   {Validate(validRequest)}");
Console.WriteLine($"Invalid request: {Validate(invalidRequest)}");

string Validate(CreateTodoRequest req)
{
    var errors = new List<string>();
    if (string.IsNullOrWhiteSpace(req.Title))
        errors.Add("Title is required");
    if (req.Title?.Length > 200)
        errors.Add("Title must be under 200 characters");
    if (req.DueDate.HasValue && req.DueDate.Value < DateTime.Now)
        errors.Add("Due date must be in the future");

    return errors.Count == 0
        ? "Valid"
        : $"Invalid: [{string.Join("; ", errors)}]";
}

// ============================================================
// 6. Error Handling Pattern
// ============================================================

Console.WriteLine("\n--- Global Error Handling ---\n");

// In a real API, this would be middleware
void HandleRequest(string path, Action action)
{
    try
    {
        action();
    }
    catch (KeyNotFoundException)
    {
        Console.WriteLine($"  404 Not Found: {path}");
    }
    catch (ArgumentException ex)
    {
        Console.WriteLine($"  400 Bad Request: {ex.Message}");
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  500 Internal Server Error: {ex.Message}");
    }
}

HandleRequest("/api/todos/999", () => throw new KeyNotFoundException());
HandleRequest("/api/todos", () => throw new ArgumentException("Title is required"));
HandleRequest("/api/crash", () => throw new InvalidOperationException("Unexpected error"));

// ============================================================
// JSON options for pretty output
// ============================================================

var jsonOptions = new JsonSerializerOptions
{
    WriteIndented = false,
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
};

// ============================================================
// Domain Types
// ============================================================

public record Todo
{
    public int Id { get; init; }
    public string Title { get; set; } = "";
    public bool IsComplete { get; set; }
    public DateTime CreatedAt { get; init; }
    public DateTime? DueDate { get; set; }

    public override string ToString() =>
        $"[{Id}] {Title} (complete={IsComplete}, due={DueDate?.ToShortDateString() ?? "none"})";
}

public record CreateTodoRequest(string Title, DateTime? DueDate);
public record UpdateTodoRequest(string? Title, bool? IsComplete, DateTime? DueDate);

// ============================================================
// Repository Interface and Implementation
// ============================================================

public interface ITodoRepository
{
    IReadOnlyList<Todo> GetAll();
    Todo? GetById(int id);
    Todo Create(string title, DateTime? dueDate);
    bool Update(int id, string? title, bool? isComplete, DateTime? dueDate);
    bool Delete(int id);
}

public class InMemoryTodoRepository : ITodoRepository
{
    private readonly ConcurrentDictionary<int, Todo> _todos = new();
    private int _nextId;

    public IReadOnlyList<Todo> GetAll() =>
        _todos.Values.OrderBy(t => t.Id).ToList();

    public Todo? GetById(int id) =>
        _todos.GetValueOrDefault(id);

    public Todo Create(string title, DateTime? dueDate)
    {
        int id = System.Threading.Interlocked.Increment(ref _nextId);
        var todo = new Todo
        {
            Id = id,
            Title = title,
            IsComplete = false,
            CreatedAt = DateTime.UtcNow,
            DueDate = dueDate
        };
        _todos[id] = todo;
        return todo;
    }

    public bool Update(int id, string? title, bool? isComplete, DateTime? dueDate)
    {
        if (!_todos.TryGetValue(id, out var todo)) return false;
        if (title is not null) todo.Title = title;
        if (isComplete.HasValue) todo.IsComplete = isComplete.Value;
        if (dueDate.HasValue) todo.DueDate = dueDate;
        return true;
    }

    public bool Delete(int id) => _todos.TryRemove(id, out _);
}

// ============================================================
// Middleware Simulation
// ============================================================

public class RequestContext
{
    public string Method { get; }
    public string Path { get; }
    public Dictionary<string, string> Headers { get; } = new();

    public RequestContext(string method, string path)
    {
        Method = method;
        Path = path;
    }
}

public class MiddlewarePipeline
{
    private readonly List<(string Name, Func<RequestContext, bool> Handler)> _middlewares = new();

    public void Use(string name, Func<RequestContext, bool> handler)
        => _middlewares.Add((name, handler));

    public void Execute(RequestContext ctx)
    {
        foreach (var (name, handler) in _middlewares)
        {
            if (!handler(ctx))
                return; // Short-circuit
        }
    }
}
