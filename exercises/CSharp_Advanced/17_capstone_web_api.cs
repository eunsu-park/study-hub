/*
 * Exercises for Lesson 17: Capstone — Web API Design
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.Json;
using System.Text.Json.Serialization;

// ---------------------------------------------------------------------------
// Exercise 1: REST endpoint design — CRUD for a resource
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: REST Endpoint Design ===");

    var controller = new TodoController(new InMemoryTodoRepository());

    // CREATE
    var created = controller.Create(new CreateTodoRequest("Buy groceries", "High"));
    Console.WriteLine($"  Created: {created.Id} - {created.Title} [{created.Priority}]");

    var created2 = controller.Create(new CreateTodoRequest("Read book", "Low"));
    Console.WriteLine($"  Created: {created2.Id} - {created2.Title}");

    // READ all
    var all = controller.GetAll();
    Console.WriteLine($"  All items: {all.Count}");

    // READ one
    var found = controller.GetById(created.Id);
    Console.WriteLine($"  Found: {found?.Title}");

    // UPDATE
    var updated = controller.Update(created.Id, new UpdateTodoRequest(true));
    Console.WriteLine($"  Updated: {updated?.Title} completed={updated?.IsCompleted}");

    // DELETE
    bool deleted = controller.Delete(created2.Id);
    Console.WriteLine($"  Deleted: {deleted}, remaining: {controller.GetAll().Count}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Request validation — middleware simulation
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Request Validation ===");

    var validator = new RequestValidator();

    var validReq = new ApiRequest("POST", "/api/todos",
        new Dictionary<string, string> { ["Content-Type"] = "application/json", ["Authorization"] = "Bearer token123" },
        """{"title":"Test","priority":"High"}""");

    var noAuthReq = new ApiRequest("POST", "/api/todos",
        new Dictionary<string, string> { ["Content-Type"] = "application/json" },
        """{"title":"Test"}""");

    var emptyBodyReq = new ApiRequest("POST", "/api/todos",
        new Dictionary<string, string> { ["Content-Type"] = "application/json", ["Authorization"] = "Bearer x" },
        "");

    Console.WriteLine($"  Valid request   : {validator.Validate(validReq)}");
    Console.WriteLine($"  No auth header  : {validator.Validate(noAuthReq)}");
    Console.WriteLine($"  Empty body POST : {validator.Validate(emptyBodyReq)}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Response formatting — standardized API responses
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Standardized API Responses ===");

    var options = new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    var success = ApiResponse<TodoItem>.Ok(
        new TodoItem(1, "Test", "High", false));
    Console.WriteLine($"  Success:\n{JsonSerializer.Serialize(success, options)}");

    var error = ApiResponse<TodoItem>.Error(404, "Todo item not found",
        new Dictionary<string, string> { ["id"] = "999" });
    Console.WriteLine($"  Error:\n{JsonSerializer.Serialize(error, options)}");

    var list = ApiResponse<List<TodoItem>>.Ok(
        new List<TodoItem> { new(1, "A", "High", false), new(2, "B", "Low", true) });
    Console.WriteLine($"  List:\n{JsonSerializer.Serialize(list, options)}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Rate limiter — token bucket algorithm
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Rate Limiter ===");

    var limiter = new TokenBucketRateLimiter(capacity: 5, refillRate: 2.0);

    Console.WriteLine("  Sending 8 rapid requests:");
    for (int i = 1; i <= 8; i++)
    {
        bool allowed = limiter.TryConsume();
        Console.WriteLine($"    Request {i}: {(allowed ? "ALLOWED" : "RATE LIMITED")} (tokens={limiter.CurrentTokens:F1})");
    }

    // Simulate time passing (refill)
    limiter.SimulateTimePassing(2.0);
    Console.WriteLine($"  After 2 seconds (refill): tokens={limiter.CurrentTokens:F1}");

    for (int i = 1; i <= 3; i++)
    {
        bool allowed = limiter.TryConsume();
        Console.WriteLine($"    Request {i}: {(allowed ? "ALLOWED" : "RATE LIMITED")}");
    }
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Integration test simulation — test the full request pipeline
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Integration Test ===");

    var pipeline = new ApiPipeline();
    pipeline.AddMiddleware(new LoggingMiddleware());
    pipeline.AddMiddleware(new AuthMiddleware());
    pipeline.AddMiddleware(new RateLimitMiddleware(maxRequests: 3));

    var requests = new[]
    {
        new ApiRequest("GET", "/api/todos", new Dictionary<string, string> { ["Authorization"] = "Bearer valid" }, ""),
        new ApiRequest("GET", "/api/todos", new Dictionary<string, string> { ["Authorization"] = "Bearer valid" }, ""),
        new ApiRequest("GET", "/api/todos", new Dictionary<string, string>(), ""), // no auth
        new ApiRequest("GET", "/api/todos", new Dictionary<string, string> { ["Authorization"] = "Bearer valid" }, ""),
        new ApiRequest("GET", "/api/todos", new Dictionary<string, string> { ["Authorization"] = "Bearer valid" }, ""),
    };

    for (int i = 0; i < requests.Length; i++)
    {
        var response = pipeline.Process(requests[i]);
        Console.WriteLine($"  Request {i + 1}: {response.StatusCode} — {response.Message}");
    }
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

record TodoItem(int Id, string Title, string Priority, bool IsCompleted);
record CreateTodoRequest(string Title, string Priority);
record UpdateTodoRequest(bool IsCompleted);

interface ITodoRepository
{
    TodoItem Add(TodoItem item);
    List<TodoItem> GetAll();
    TodoItem? GetById(int id);
    TodoItem? Update(int id, UpdateTodoRequest req);
    bool Delete(int id);
}

class InMemoryTodoRepository : ITodoRepository
{
    private readonly List<TodoItem> _items = new();
    private int _nextId = 1;

    public TodoItem Add(TodoItem item)
    {
        var newItem = item with { Id = _nextId++ };
        _items.Add(newItem);
        return newItem;
    }
    public List<TodoItem> GetAll() => _items.ToList();
    public TodoItem? GetById(int id) => _items.FirstOrDefault(i => i.Id == id);
    public TodoItem? Update(int id, UpdateTodoRequest req)
    {
        var idx = _items.FindIndex(i => i.Id == id);
        if (idx < 0) return null;
        _items[idx] = _items[idx] with { IsCompleted = req.IsCompleted };
        return _items[idx];
    }
    public bool Delete(int id) => _items.RemoveAll(i => i.Id == id) > 0;
}

class TodoController
{
    private readonly ITodoRepository _repo;
    public TodoController(ITodoRepository repo) => _repo = repo;
    public TodoItem Create(CreateTodoRequest req) =>
        _repo.Add(new TodoItem(0, req.Title, req.Priority, false));
    public List<TodoItem> GetAll() => _repo.GetAll();
    public TodoItem? GetById(int id) => _repo.GetById(id);
    public TodoItem? Update(int id, UpdateTodoRequest req) => _repo.Update(id, req);
    public bool Delete(int id) => _repo.Delete(id);
}

record ApiRequest(string Method, string Path, Dictionary<string, string> Headers, string Body);

class ValidationResult
{
    public bool IsValid { get; init; }
    public string? Error { get; init; }
    public override string ToString() => IsValid ? "VALID" : $"INVALID: {Error}";
}

class RequestValidator
{
    public ValidationResult Validate(ApiRequest req)
    {
        if (!req.Headers.ContainsKey("Authorization"))
            return new() { IsValid = false, Error = "Missing Authorization header" };
        if (req.Method == "POST" && string.IsNullOrWhiteSpace(req.Body))
            return new() { IsValid = false, Error = "POST requires a body" };
        return new() { IsValid = true };
    }
}

class ApiResponse<T>
{
    public int StatusCode { get; init; }
    public bool Success { get; init; }
    public T? Data { get; init; }
    public string? ErrorMessage { get; init; }
    public Dictionary<string, string>? ErrorDetails { get; init; }

    public static ApiResponse<T> Ok(T data) => new() { StatusCode = 200, Success = true, Data = data };
    public static ApiResponse<T> Error(int code, string msg, Dictionary<string, string>? details = null) =>
        new() { StatusCode = code, Success = false, ErrorMessage = msg, ErrorDetails = details };
}

class TokenBucketRateLimiter
{
    private double _tokens;
    private readonly int _capacity;
    private readonly double _refillRate;

    public double CurrentTokens => _tokens;

    public TokenBucketRateLimiter(int capacity, double refillRate)
    {
        _capacity = capacity;
        _refillRate = refillRate;
        _tokens = capacity;
    }

    public bool TryConsume()
    {
        if (_tokens < 1) return false;
        _tokens--;
        return true;
    }

    public void SimulateTimePassing(double seconds)
    {
        _tokens = Math.Min(_capacity, _tokens + _refillRate * seconds);
    }
}

// ---- Middleware pipeline ----

record PipelineResponse(int StatusCode, string Message);

interface IMiddleware
{
    PipelineResponse? Process(ApiRequest request);
}

class LoggingMiddleware : IMiddleware
{
    public PipelineResponse? Process(ApiRequest request)
    {
        Console.WriteLine($"    [LOG] {request.Method} {request.Path}");
        return null; // pass through
    }
}

class AuthMiddleware : IMiddleware
{
    public PipelineResponse? Process(ApiRequest request)
    {
        if (!request.Headers.ContainsKey("Authorization"))
            return new PipelineResponse(401, "Unauthorized");
        return null;
    }
}

class RateLimitMiddleware : IMiddleware
{
    private int _remaining;
    public RateLimitMiddleware(int maxRequests) => _remaining = maxRequests;
    public PipelineResponse? Process(ApiRequest request)
    {
        if (_remaining <= 0)
            return new PipelineResponse(429, "Too Many Requests");
        _remaining--;
        return null;
    }
}

class ApiPipeline
{
    private readonly List<IMiddleware> _middleware = new();
    public void AddMiddleware(IMiddleware mw) => _middleware.Add(mw);
    public PipelineResponse Process(ApiRequest request)
    {
        foreach (var mw in _middleware)
        {
            var result = mw.Process(request);
            if (result != null) return result;
        }
        return new PipelineResponse(200, "OK");
    }
}
