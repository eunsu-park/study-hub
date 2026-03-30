# Serialization (System.Text.Json)

**Previous**: [Dependency Injection](./10_Dependency_Injection.md) | **Next**: [Testing](./12_Testing.md)

---

## Learning Objectives

After completing this lesson, you will be able to:

1. Serialize and deserialize C# objects to/from JSON using `System.Text.Json`
2. Customize serialization behavior with `JsonSerializerOptions`
3. Apply property attributes to control JSON output
4. Write custom `JsonConverter<T>` implementations for complex types
5. Use polymorphic serialization with `[JsonDerivedType]`
6. Generate source-based serializers for AOT and performance using `[JsonSerializable]`
7. Process JSON efficiently with `Utf8JsonReader`, `Utf8JsonWriter`, `JsonDocument`, and `JsonNode`
8. Migrate existing code from Newtonsoft.Json to System.Text.Json

---

Serialization is the process of converting objects into a format suitable for storage or transmission, and deserialization is the reverse. JSON (JavaScript Object Notation) is the dominant data interchange format for web APIs, configuration files, and document databases. Since .NET Core 3.0, `System.Text.Json` is the built-in, high-performance JSON library. This lesson covers everything from basic serialization to advanced topics like custom converters, polymorphism, source generators, and low-level readers/writers.

## 1. JSON Serialization Basics

### 1.1 Serialize and Deserialize

```csharp
using System.Text.Json;

public class Person
{
    public string Name { get; set; } = "";
    public int Age { get; set; }
    public string Email { get; set; } = "";
}

// Serialize to JSON string
var person = new Person { Name = "Alice", Age = 30, Email = "alice@example.com" };
string json = JsonSerializer.Serialize(person);
// {"Name":"Alice","Age":30,"Email":"alice@example.com"}

// Deserialize from JSON string
Person? deserialized = JsonSerializer.Deserialize<Person>(json);
Console.WriteLine(deserialized?.Name); // Alice
```

### 1.2 Serialize to Bytes (UTF-8)

```csharp
// Serialize directly to UTF-8 bytes (faster than going through a string)
byte[] utf8Json = JsonSerializer.SerializeToUtf8Bytes(person);

// Deserialize from UTF-8 bytes
Person? fromBytes = JsonSerializer.Deserialize<Person>(utf8Json);
```

### 1.3 Serialize to/from Streams

```csharp
// Write to a file
await using (var stream = File.Create("person.json"))
{
    await JsonSerializer.SerializeAsync(stream, person);
}

// Read from a file
await using (var stream = File.OpenRead("person.json"))
{
    Person? fromFile = await JsonSerializer.DeserializeAsync<Person>(stream);
    Console.WriteLine(fromFile?.Name);
}
```

## 2. JsonSerializerOptions

`JsonSerializerOptions` controls virtually every aspect of serialization behavior.

### 2.1 Common Options

```csharp
var options = new JsonSerializerOptions
{
    // Property naming
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,  // "Name" -> "name"

    // Formatting
    WriteIndented = true,                                // Pretty-print

    // Null handling
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,

    // Number handling
    NumberHandling = JsonNumberHandling.AllowReadingFromString,

    // Property matching
    PropertyNameCaseInsensitive = true,                  // Match "name" to "Name"

    // Encoder (allow non-ASCII characters)
    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,

    // Max depth
    MaxDepth = 64,

    // Unknown properties
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Skip // or Disallow
};

string json = JsonSerializer.Serialize(person, options);
// {
//   "name": "Alice",
//   "age": 30,
//   "email": "alice@example.com"
// }
```

### 2.2 Naming Policies

```csharp
// Built-in naming policies
JsonNamingPolicy.CamelCase;      // "FirstName" -> "firstName"
JsonNamingPolicy.SnakeCaseLower; // "FirstName" -> "first_name"
JsonNamingPolicy.SnakeCaseUpper; // "FirstName" -> "FIRST_NAME"
JsonNamingPolicy.KebabCaseLower; // "FirstName" -> "first-name"
JsonNamingPolicy.KebabCaseUpper; // "FirstName" -> "FIRST-NAME"

// Custom naming policy
public class UpperCaseNamingPolicy : JsonNamingPolicy
{
    public override string ConvertName(string name) => name.ToUpperInvariant();
}

var options = new JsonSerializerOptions
{
    PropertyNamingPolicy = new UpperCaseNamingPolicy()
};
// "Name" -> "NAME"
```

### 2.3 Reusing Options

```csharp
// Create once, reuse everywhere — options caches metadata internally
public static class JsonDefaults
{
    public static readonly JsonSerializerOptions Web = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        PropertyNameCaseInsensitive = true,
        WriteIndented = false
    };

    public static readonly JsonSerializerOptions Indented = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true
    };
}

// Usage
string json = JsonSerializer.Serialize(person, JsonDefaults.Web);
```

## 3. Property Attributes

### 3.1 JsonPropertyName

```csharp
public class GitHubUser
{
    [JsonPropertyName("login")]
    public string Username { get; set; } = "";

    [JsonPropertyName("avatar_url")]
    public string AvatarUrl { get; set; } = "";

    [JsonPropertyName("public_repos")]
    public int PublicRepoCount { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }
}
```

### 3.2 JsonIgnore

```csharp
public class User
{
    public string Name { get; set; } = "";

    [JsonIgnore]
    public string PasswordHash { get; set; } = "";

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? MiddleName { get; set; }

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
    public int Score { get; set; } // Omitted when 0
}
```

### 3.3 JsonInclude

By default, only public properties are serialized. Use `[JsonInclude]` to include fields or private setters.

```csharp
public class Product
{
    [JsonInclude]
    public string _sku; // Field (not property) — included via attribute

    public string Name { get; set; } = "";

    [JsonInclude]
    public decimal Price { get; private set; } // Private setter included

    [JsonConstructor]
    public Product(string name, decimal price, string sku)
    {
        Name = name;
        Price = price;
        _sku = sku;
    }
}
```

### 3.4 JsonConstructor and Required Properties

```csharp
public class Config
{
    [JsonRequired] // .NET 7+: deserialization fails if property is missing
    public string AppName { get; set; } = "";

    public string Version { get; set; } = "1.0";

    [JsonConstructor]
    public Config(string appName, string version)
    {
        AppName = appName;
        Version = version;
    }
}

// C# 11 required members also work:
public class StrictConfig
{
    public required string AppName { get; set; }
    public required string ConnectionString { get; set; }
    public int MaxRetries { get; set; } = 3;
}
```

### 3.5 JsonPropertyOrder

```csharp
public class ApiResponse<T>
{
    [JsonPropertyOrder(-2)]
    public string Status { get; set; } = "ok";

    [JsonPropertyOrder(-1)]
    public string Message { get; set; } = "";

    [JsonPropertyOrder(0)]
    public T? Data { get; set; }

    [JsonPropertyOrder(1)]
    public Dictionary<string, string[]>? Errors { get; set; }
}
```

## 4. Custom Converters (JsonConverter&lt;T&gt;)

When the default serialization behavior doesn't match your needs, write a custom converter.

### 4.1 Simple Converter: Unix Timestamp

```csharp
public class UnixTimestampConverter : JsonConverter<DateTime>
{
    public override DateTime Read(ref Utf8JsonReader reader, Type typeToConvert,
        JsonSerializerOptions options)
    {
        long unixTime = reader.GetInt64();
        return DateTimeOffset.FromUnixTimeSeconds(unixTime).UtcDateTime;
    }

    public override void Write(Utf8JsonWriter writer, DateTime value,
        JsonSerializerOptions options)
    {
        long unixTime = new DateTimeOffset(value).ToUnixTimeSeconds();
        writer.WriteNumberValue(unixTime);
    }
}

// Usage via attribute
public class LogEntry
{
    public string Message { get; set; } = "";

    [JsonConverter(typeof(UnixTimestampConverter))]
    public DateTime Timestamp { get; set; }
}

// Or via options
var options = new JsonSerializerOptions();
options.Converters.Add(new UnixTimestampConverter());
```

### 4.2 Converter for Custom Types

```csharp
public readonly struct Money
{
    public decimal Amount { get; }
    public string Currency { get; }

    public Money(decimal amount, string currency)
    {
        Amount = amount;
        Currency = currency;
    }

    public override string ToString() => $"{Amount} {Currency}";
}

public class MoneyConverter : JsonConverter<Money>
{
    public override Money Read(ref Utf8JsonReader reader, Type typeToConvert,
        JsonSerializerOptions options)
    {
        if (reader.TokenType == JsonTokenType.String)
        {
            // Parse "99.99 USD" format
            string value = reader.GetString()!;
            int spaceIdx = value.IndexOf(' ');
            decimal amount = decimal.Parse(value[..spaceIdx]);
            string currency = value[(spaceIdx + 1)..];
            return new Money(amount, currency);
        }

        // Parse {"amount": 99.99, "currency": "USD"} format
        if (reader.TokenType != JsonTokenType.StartObject)
            throw new JsonException("Expected string or object");

        decimal amt = 0;
        string cur = "USD";

        while (reader.Read() && reader.TokenType != JsonTokenType.EndObject)
        {
            string prop = reader.GetString()!;
            reader.Read();

            if (prop == "amount") amt = reader.GetDecimal();
            else if (prop == "currency") cur = reader.GetString()!;
        }

        return new Money(amt, cur);
    }

    public override void Write(Utf8JsonWriter writer, Money value,
        JsonSerializerOptions options)
    {
        writer.WriteStringValue($"{value.Amount} {value.Currency}");
    }
}
```

### 4.3 Converter Factory

```csharp
public class StringEnumConverterFactory : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert) =>
        typeToConvert.IsEnum;

    public override JsonConverter CreateConverter(Type typeToConvert,
        JsonSerializerOptions options)
    {
        Type converterType = typeof(StringEnumConverter<>).MakeGenericType(typeToConvert);
        return (JsonConverter)Activator.CreateInstance(converterType)!;
    }

    private class StringEnumConverter<T> : JsonConverter<T> where T : struct, Enum
    {
        public override T Read(ref Utf8JsonReader reader, Type typeToConvert,
            JsonSerializerOptions options)
        {
            string? value = reader.GetString();
            return Enum.TryParse<T>(value, ignoreCase: true, out var result)
                ? result
                : throw new JsonException($"Unable to parse '{value}' as {typeof(T).Name}");
        }

        public override void Write(Utf8JsonWriter writer, T value,
            JsonSerializerOptions options)
        {
            writer.WriteStringValue(value.ToString());
        }
    }
}
```

## 5. Polymorphic Serialization

### 5.1 JsonDerivedType Attribute (.NET 7+)

```csharp
[JsonDerivedType(typeof(Circle), "circle")]
[JsonDerivedType(typeof(Rectangle), "rectangle")]
[JsonDerivedType(typeof(Triangle), "triangle")]
public abstract class Shape
{
    public string Color { get; set; } = "Black";
    public abstract double Area();
}

public class Circle : Shape
{
    public double Radius { get; set; }
    public override double Area() => Math.PI * Radius * Radius;
}

public class Rectangle : Shape
{
    public double Width { get; set; }
    public double Height { get; set; }
    public override double Area() => Width * Height;
}

public class Triangle : Shape
{
    public double Base { get; set; }
    public double Height { get; set; }
    public override double Area() => 0.5 * Base * Height;
}
```

```csharp
// Serialize with type discriminator
Shape circle = new Circle { Radius = 5, Color = "Red" };
string json = JsonSerializer.Serialize(circle);
// {"$type":"circle","Color":"Red","Radius":5}

// Deserialize polymorphically
Shape? shape = JsonSerializer.Deserialize<Shape>(json);
Console.WriteLine(shape?.GetType().Name); // Circle
Console.WriteLine(shape?.Area());          // 78.54...
```

### 5.2 Polymorphic Collections

```csharp
List<Shape> shapes = new()
{
    new Circle { Radius = 3, Color = "Blue" },
    new Rectangle { Width = 4, Height = 6, Color = "Green" },
    new Triangle { Base = 5, Height = 8, Color = "Yellow" }
};

var options = new JsonSerializerOptions { WriteIndented = true };
string json = JsonSerializer.Serialize(shapes, options);

List<Shape>? restored = JsonSerializer.Deserialize<List<Shape>>(json);
foreach (Shape s in restored!)
{
    Console.WriteLine($"{s.GetType().Name}: Area = {s.Area():F2}, Color = {s.Color}");
}
```

## 6. JSON Source Generators

Source generators pre-compute serialization metadata at compile time, eliminating runtime reflection. This is critical for AOT (Ahead-of-Time) compilation and improves startup performance.

### 6.1 Basic Source Generator Setup

```csharp
// Define a partial context class
[JsonSourceGenerationOptions(
    PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
    WriteIndented = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(Person))]
[JsonSerializable(typeof(List<Person>))]
[JsonSerializable(typeof(ApiResponse<Person>))]
public partial class AppJsonContext : JsonSerializerContext
{
}
```

```csharp
// Usage with generated context
var person = new Person { Name = "Alice", Age = 30, Email = "alice@example.com" };

// Type-safe, AOT-compatible serialization
string json = JsonSerializer.Serialize(person, AppJsonContext.Default.Person);

// Deserialization
Person? result = JsonSerializer.Deserialize(json, AppJsonContext.Default.Person);

// Collection serialization
var people = new List<Person> { person };
string listJson = JsonSerializer.Serialize(people, AppJsonContext.Default.ListPerson);
```

### 6.2 Performance Benefits

```csharp
// Benchmark comparison (conceptual)
// Without source generator: 100μs first call (reflection), 5μs subsequent
// With source generator:    5μs first call (no reflection), 5μs subsequent
// AOT: Without generator -> fails at runtime. With generator -> works.
```

### 6.3 Registering with ASP.NET Core

```csharp
var builder = WebApplication.CreateBuilder(args);

builder.Services.ConfigureHttpJsonOptions(options =>
{
    options.SerializerOptions.TypeInfoResolverChain.Insert(0, AppJsonContext.Default);
});

var app = builder.Build();

app.MapGet("/person", () => new Person { Name = "Alice", Age = 30 });
app.MapPost("/person", (Person p) => Results.Ok(p));
```

## 7. Low-Level: Utf8JsonReader and Utf8JsonWriter

For maximum performance or streaming scenarios, use the low-level reader/writer directly.

### 7.1 Utf8JsonWriter

```csharp
using var stream = new MemoryStream();
using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions
{
    Indented = true,
    SkipValidation = false // Set true for performance in trusted code
});

writer.WriteStartObject();
writer.WriteString("name", "Alice");
writer.WriteNumber("age", 30);
writer.WriteBoolean("active", true);

writer.WriteStartArray("scores");
writer.WriteNumberValue(95);
writer.WriteNumberValue(87);
writer.WriteNumberValue(92);
writer.WriteEndArray();

writer.WriteStartObject("address");
writer.WriteString("city", "Seattle");
writer.WriteString("state", "WA");
writer.WriteEndObject();

writer.WriteNull("middleName");
writer.WriteEndObject();

writer.Flush();

string json = System.Text.Encoding.UTF8.GetString(stream.ToArray());
Console.WriteLine(json);
```

### 7.2 Utf8JsonReader

```csharp
byte[] jsonBytes = System.Text.Encoding.UTF8.GetBytes(
    """{"name":"Alice","age":30,"scores":[95,87,92]}""");

var reader = new Utf8JsonReader(jsonBytes);
string? name = null;
int age = 0;
var scores = new List<int>();

while (reader.Read())
{
    if (reader.TokenType == JsonTokenType.PropertyName)
    {
        string prop = reader.GetString()!;
        reader.Read(); // Move to value

        switch (prop)
        {
            case "name":
                name = reader.GetString();
                break;
            case "age":
                age = reader.GetInt32();
                break;
            case "scores":
                while (reader.Read() && reader.TokenType != JsonTokenType.EndArray)
                {
                    scores.Add(reader.GetInt32());
                }
                break;
        }
    }
}

Console.WriteLine($"{name}, {age}, scores: [{string.Join(", ", scores)}]");
```

## 8. JsonNode and JsonDocument for Dynamic JSON

### 8.1 JsonDocument (Read-Only)

`JsonDocument` provides read-only access to parsed JSON without mapping to a specific type.

```csharp
string json = """
{
    "users": [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25}
    ],
    "total": 2
}
""";

using JsonDocument doc = JsonDocument.Parse(json);
JsonElement root = doc.RootElement;

int total = root.GetProperty("total").GetInt32();
Console.WriteLine($"Total: {total}");

foreach (JsonElement user in root.GetProperty("users").EnumerateArray())
{
    string name = user.GetProperty("name").GetString()!;
    int age = user.GetProperty("age").GetInt32();
    Console.WriteLine($"  {name}, age {age}");
}
```

### 8.2 JsonNode (Mutable)

`JsonNode` provides a mutable DOM for building, modifying, and querying JSON.

```csharp
using System.Text.Json.Nodes;

// Build JSON dynamically
var node = new JsonObject
{
    ["name"] = "Alice",
    ["age"] = 30,
    ["tags"] = new JsonArray("developer", "speaker"),
    ["address"] = new JsonObject
    {
        ["city"] = "Seattle",
        ["zip"] = "98101"
    }
};

// Modify
node["age"] = 31;
node["tags"]!.AsArray().Add("author");
node["phone"] = "555-1234";

// Query
string city = node["address"]!["city"]!.GetValue<string>();
Console.WriteLine(city); // Seattle

// Serialize to string
string json = node.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(json);
```

### 8.3 Parsing and Modifying Existing JSON

```csharp
string input = """{"config": {"debug": false, "logLevel": "info", "maxRetries": 3}}""";

JsonNode? root = JsonNode.Parse(input);
JsonObject config = root!["config"]!.AsObject();

// Update values
config["debug"] = true;
config["logLevel"] = "debug";
config["newSetting"] = "value";

// Remove a property
config.Remove("maxRetries");

Console.WriteLine(root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
```

## 9. Handling Enums, Dates, and Nulls

### 9.1 Enum Serialization

```csharp
public enum Status { Active, Inactive, Suspended }

// Default: serializes as number
// {"Status":0}

// As string:
var options = new JsonSerializerOptions
{
    Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
};
// {"status":"active"}
```

### 9.2 Date and Time

```csharp
public class Event
{
    public string Name { get; set; } = "";
    public DateTime StartTime { get; set; }           // ISO 8601 by default
    public DateTimeOffset CreatedAt { get; set; }      // ISO 8601 with offset
    public DateOnly Date { get; set; }                 // "2024-01-15"
    public TimeOnly Time { get; set; }                 // "14:30:00"
    public TimeSpan Duration { get; set; }             // NOT supported by default
}

// TimeSpan converter
public class TimeSpanConverter : JsonConverter<TimeSpan>
{
    public override TimeSpan Read(ref Utf8JsonReader reader, Type typeToConvert,
        JsonSerializerOptions options)
    {
        return TimeSpan.Parse(reader.GetString()!);
    }

    public override void Write(Utf8JsonWriter writer, TimeSpan value,
        JsonSerializerOptions options)
    {
        writer.WriteStringValue(value.ToString());
    }
}
```

### 9.3 Null Handling Strategies

```csharp
var options = new JsonSerializerOptions
{
    // Strategy 1: Ignore all null properties globally
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
};

// Strategy 2: Per-property control
public class UserProfile
{
    public string Name { get; set; } = "";

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Bio { get; set; }

    [JsonIgnore(Condition = JsonIgnoreCondition.Never)] // Always include, even if null
    public string? Website { get; set; }
}
```

## 10. Migration Tips from Newtonsoft.Json

| Newtonsoft.Json | System.Text.Json |
|---|---|
| `JsonConvert.SerializeObject()` | `JsonSerializer.Serialize()` |
| `JsonConvert.DeserializeObject<T>()` | `JsonSerializer.Deserialize<T>()` |
| `[JsonProperty("name")]` | `[JsonPropertyName("name")]` |
| `JObject.Parse()` | `JsonNode.Parse()` / `JsonDocument.Parse()` |
| `JObject`, `JArray` | `JsonObject`, `JsonArray` / `JsonElement` |
| `NullValueHandling.Ignore` | `DefaultIgnoreCondition = WhenWritingNull` |
| `JsonSerializerSettings` | `JsonSerializerOptions` |
| `JsonConverter (Newtonsoft)` | `JsonConverter<T> (STJ)` |
| `TypeNameHandling.Auto` | `[JsonDerivedType]` |
| `DefaultValueHandling` | `[JsonIgnore(Condition = ...)]` |

```csharp
// Newtonsoft
// var settings = new JsonSerializerSettings
// {
//     NullValueHandling = NullValueHandling.Ignore,
//     Formatting = Formatting.Indented,
//     ContractResolver = new CamelCasePropertyNamesContractResolver()
// };
// string json = JsonConvert.SerializeObject(obj, settings);

// System.Text.Json equivalent
var options = new JsonSerializerOptions
{
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    WriteIndented = true,
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
};
string json = JsonSerializer.Serialize(obj, options);
```

## 11. Practical Example: Configuration File Reader/Writer

This example builds a complete configuration file manager that reads, writes, validates, and watches JSON configuration files.

```csharp
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;

public class AppConfiguration
{
    [JsonRequired]
    public string AppName { get; set; } = "";

    public string Version { get; set; } = "1.0.0";

    public DatabaseConfig Database { get; set; } = new();
    public LoggingConfig Logging { get; set; } = new();
    public List<FeatureFlag> Features { get; set; } = new();
}

public class DatabaseConfig
{
    [JsonRequired]
    public string ConnectionString { get; set; } = "";
    public int MaxPoolSize { get; set; } = 100;
    public int CommandTimeoutSeconds { get; set; } = 30;
    public bool EnableRetry { get; set; } = true;
}

public class LoggingConfig
{
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public LogLevel Level { get; set; } = LogLevel.Information;
    public string OutputPath { get; set; } = "logs/app.log";
    public long MaxFileSizeBytes { get; set; } = 10 * 1024 * 1024;
}

public enum LogLevel { Trace, Debug, Information, Warning, Error, Critical }

public class FeatureFlag
{
    public string Name { get; set; } = "";
    public bool Enabled { get; set; }
    public Dictionary<string, string> Parameters { get; set; } = new();
}
```

```csharp
public class ConfigManager
{
    private static readonly JsonSerializerOptions _options = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        WriteIndented = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
    };

    private readonly string _filePath;
    private FileSystemWatcher? _watcher;

    public event Action<AppConfiguration>? ConfigChanged;

    public ConfigManager(string filePath)
    {
        _filePath = filePath;
    }

    public async Task<AppConfiguration> LoadAsync()
    {
        if (!File.Exists(_filePath))
            throw new FileNotFoundException($"Configuration file not found: {_filePath}");

        await using var stream = File.OpenRead(_filePath);
        var config = await JsonSerializer.DeserializeAsync<AppConfiguration>(stream, _options);

        return config ?? throw new JsonException("Failed to deserialize configuration");
    }

    public async Task SaveAsync(AppConfiguration config)
    {
        string directory = Path.GetDirectoryName(_filePath)!;
        if (!Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        // Write to temp file first, then rename (atomic operation)
        string tempPath = _filePath + ".tmp";
        await using (var stream = File.Create(tempPath))
        {
            await JsonSerializer.SerializeAsync(stream, config, _options);
        }
        File.Move(tempPath, _filePath, overwrite: true);
    }

    public async Task MergeAsync(string partialJson)
    {
        // Load current config as JsonNode for merging
        JsonNode? current;
        if (File.Exists(_filePath))
        {
            string existing = await File.ReadAllTextAsync(_filePath);
            current = JsonNode.Parse(existing);
        }
        else
        {
            current = new JsonObject();
        }

        JsonNode? patch = JsonNode.Parse(partialJson);
        MergeNodes(current!.AsObject(), patch!.AsObject());

        await File.WriteAllTextAsync(_filePath,
            current.ToJsonString(_options));
    }

    private static void MergeNodes(JsonObject target, JsonObject source)
    {
        foreach (var prop in source)
        {
            if (prop.Value is JsonObject sourceObj
                && target[prop.Key] is JsonObject targetObj)
            {
                MergeNodes(targetObj, sourceObj);
            }
            else
            {
                target[prop.Key] = prop.Value?.DeepClone();
            }
        }
    }

    public void StartWatching()
    {
        string dir = Path.GetDirectoryName(_filePath)!;
        string file = Path.GetFileName(_filePath);

        _watcher = new FileSystemWatcher(dir, file)
        {
            NotifyFilter = NotifyFilters.LastWrite
        };

        _watcher.Changed += async (_, _) =>
        {
            await Task.Delay(100); // Debounce
            try
            {
                var config = await LoadAsync();
                ConfigChanged?.Invoke(config);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error reloading config: {ex.Message}");
            }
        };

        _watcher.EnableRaisingEvents = true;
    }

    public void StopWatching()
    {
        _watcher?.Dispose();
        _watcher = null;
    }
}
```

```csharp
// Usage
var manager = new ConfigManager("config/appsettings.json");

// Create initial configuration
var config = new AppConfiguration
{
    AppName = "MyApp",
    Version = "2.1.0",
    Database = new DatabaseConfig
    {
        ConnectionString = "Server=localhost;Database=MyDb;Trusted_Connection=true;",
        MaxPoolSize = 50
    },
    Logging = new LoggingConfig { Level = LogLevel.Debug },
    Features = new List<FeatureFlag>
    {
        new() { Name = "dark_mode", Enabled = true },
        new() { Name = "beta_features", Enabled = false,
                Parameters = new() { ["rollout_percent"] = "10" } }
    }
};

await manager.SaveAsync(config);

// Load and use
var loaded = await manager.LoadAsync();
Console.WriteLine($"App: {loaded.AppName} v{loaded.Version}");
Console.WriteLine($"DB Pool: {loaded.Database.MaxPoolSize}");
Console.WriteLine($"Log Level: {loaded.Logging.Level}");

// Merge partial update
await manager.MergeAsync("""{"logging": {"level": "warning"}}""");

// Watch for changes
manager.ConfigChanged += cfg =>
    Console.WriteLine($"Config reloaded: LogLevel = {cfg.Logging.Level}");

manager.StartWatching();
```

## 12. Practice Problems

1. **Generic API Response Wrapper**: Create a generic `ApiResponse<T>` class with `Status` (enum: Success, Error, NotFound), `Data` (T?), `Errors` (List<string>?), and `Timestamp`. Write a custom `JsonConverter<ApiResponse<T>>` that serializes the timestamp as a Unix epoch and omits empty error lists. Write tests that round-trip various payloads.

2. **JSON Schema Validator**: Using `JsonDocument`, write a simple validator that checks a JSON document against a schema definition (e.g., required fields, type checks for string/number/boolean, min/max for numbers, minLength/maxLength for strings). The schema itself should be loaded from a JSON file. Validate at least 5 different JSON samples and report all validation errors.

3. **JSON Diff Tool**: Write a method `IReadOnlyList<JsonDiff> ComputeDiff(string jsonA, string jsonB)` that compares two JSON documents and returns a list of differences. Each diff should contain the JSON path (e.g., `$.users[0].name`), the old value, the new value, and the type of change (Added, Removed, Modified). Use `JsonDocument` for efficient comparison without deserialization.

4. **Streaming JSON Parser**: Using `Utf8JsonReader`, parse a large JSON array file (simulated as a 10MB JSON array of objects) in a streaming fashion. Process each object individually without loading the entire file into memory. Count records, compute average of a numeric field, and find the max value. Demonstrate memory usage stays constant regardless of file size.

5. **Polyglot Serializer**: Create a `DocumentStore` class that can serialize/deserialize objects in multiple formats. Support `System.Text.Json`, `Utf8JsonWriter` (manual), and a simple custom text format. Use an `IDocumentSerializer` interface with three implementations. Register all three as keyed DI services. Write a roundtrip test for each format with the same complex object graph (including nested objects, collections, and enums).
