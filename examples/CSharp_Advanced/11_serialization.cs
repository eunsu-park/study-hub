// Lesson 11: Serialization with System.Text.Json
// Run: dotnet run

using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

// ============================================================
// 1. Basic Serialization and Deserialization
// ============================================================

Console.WriteLine("=== Basic Serialize / Deserialize ===");

var person = new Person { Name = "Alice", Age = 30, Email = "alice@example.com" };

// Serialize to JSON string
string json = JsonSerializer.Serialize(person);
Console.WriteLine($"Serialized: {json}");

// Deserialize back to object
Person? deserialized = JsonSerializer.Deserialize<Person>(json);
Console.WriteLine($"Deserialized: Name={deserialized?.Name}, Age={deserialized?.Age}");

// ============================================================
// 2. JsonSerializerOptions
// ============================================================

Console.WriteLine("\n=== JsonSerializerOptions ===");

var options = new JsonSerializerOptions
{
    WriteIndented = true,                                  // Pretty-print
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,     // camelCase keys
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull, // Skip nulls
    PropertyNameCaseInsensitive = true,                    // Case-insensitive read
};

var product = new Product
{
    Id = 1,
    Name = "Widget",
    Price = 9.99m,
    Description = null,
    Tags = new List<string> { "electronics", "sale" }
};

string prettyJson = JsonSerializer.Serialize(product, options);
Console.WriteLine($"Pretty JSON:\n{prettyJson}");

// Deserialize with case-insensitive matching
string inputJson = """{"ID": 2, "NAME": "Gadget", "PRICE": 24.99}""";
Product? parsed = JsonSerializer.Deserialize<Product>(inputJson, options);
Console.WriteLine($"\nParsed: {parsed?.Name}, ${parsed?.Price}");

// ============================================================
// 3. Attributes for Customization
// ============================================================

Console.WriteLine("\n=== Serialization Attributes ===");

var user = new UserDto
{
    UserId = 42,
    Username = "alice",
    PasswordHash = "secret_hash_123",
    CreatedAt = new DateTime(2024, 1, 15),
    Role = UserRole.Admin,
    Preferences = new Dictionary<string, string>
    {
        ["theme"] = "dark",
        ["language"] = "en"
    }
};

var attrOptions = new JsonSerializerOptions { WriteIndented = true };
string userJson = JsonSerializer.Serialize(user, attrOptions);
Console.WriteLine($"User JSON:\n{userJson}");

// Deserialize with renamed properties
string responseJson = """
{
    "user_id": 99,
    "username": "bob",
    "created_at": "2024-06-01T00:00:00",
    "role": "Moderator",
    "preferences": {"theme": "light"}
}
""";
UserDto? responseUser = JsonSerializer.Deserialize<UserDto>(responseJson, attrOptions);
Console.WriteLine($"\nDeserialized: {responseUser?.Username}, Role={responseUser?.Role}");

// ============================================================
// 4. Collections and Nested Objects
// ============================================================

Console.WriteLine("\n=== Collections and Nesting ===");

var order = new Order
{
    OrderId = "ORD-1001",
    Customer = new Customer { Name = "Alice", Address = "123 Main St" },
    Items = new List<OrderItem>
    {
        new() { Product = "Widget", Quantity = 2, UnitPrice = 9.99m },
        new() { Product = "Gadget", Quantity = 1, UnitPrice = 24.99m },
    }
};

string orderJson = JsonSerializer.Serialize(order, new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine($"Order:\n{orderJson}");

// Round-trip
Order? roundTrip = JsonSerializer.Deserialize<Order>(orderJson);
Console.WriteLine($"\nRound-trip items: {roundTrip?.Items.Count}");

// ============================================================
// 5. Polymorphic Serialization
// ============================================================

Console.WriteLine("\n=== Polymorphic Serialization ===");

Shape[] shapes =
{
    new CircleShape { Radius = 5.0 },
    new RectangleShape { Width = 4.0, Height = 6.0 },
    new CircleShape { Radius = 3.0 },
};

var polyOptions = new JsonSerializerOptions { WriteIndented = true };
string shapesJson = JsonSerializer.Serialize(shapes, polyOptions);
Console.WriteLine($"Shapes:\n{shapesJson}");

// Deserialize polymorphic types
Shape[]? deserializedShapes = JsonSerializer.Deserialize<Shape[]>(shapesJson, polyOptions);
foreach (var s in deserializedShapes ?? Array.Empty<Shape>())
{
    Console.WriteLine($"  {s.GetType().Name}: Area = {s.Area:F2}");
}

// ============================================================
// 6. Custom JsonConverter
// ============================================================

Console.WriteLine("\n=== Custom JsonConverter ===");

var config = new AppConfig
{
    Name = "MyApp",
    Version = new Version(2, 1, 3),
    LaunchDate = DateOnly.FromDateTime(new DateTime(2024, 3, 15))
};

var converterOptions = new JsonSerializerOptions
{
    WriteIndented = true,
    Converters = { new VersionConverter(), new DateOnlyConverter() }
};

string configJson = JsonSerializer.Serialize(config, converterOptions);
Console.WriteLine($"Config:\n{configJson}");

AppConfig? parsedConfig = JsonSerializer.Deserialize<AppConfig>(configJson, converterOptions);
Console.WriteLine($"\nParsed version: {parsedConfig?.Version}");

// ============================================================
// 7. JsonDocument and JsonElement (DOM Access)
// ============================================================

Console.WriteLine("\n=== JsonDocument (DOM) ===");

string rawJson = """
{
    "name": "test-api",
    "version": "1.0",
    "endpoints": [
        {"path": "/users", "method": "GET"},
        {"path": "/users", "method": "POST"},
        {"path": "/health", "method": "GET"}
    ]
}
""";

using var doc = JsonDocument.Parse(rawJson);
JsonElement root = doc.RootElement;

string apiName = root.GetProperty("name").GetString() ?? "";
Console.WriteLine($"API: {apiName}");

// Iterate array elements
Console.WriteLine("Endpoints:");
foreach (var endpoint in root.GetProperty("endpoints").EnumerateArray())
{
    string path = endpoint.GetProperty("path").GetString() ?? "";
    string method = endpoint.GetProperty("method").GetString() ?? "";
    Console.WriteLine($"  {method} {path}");
}

// ============================================================
// Supporting Types
// ============================================================

public class Person
{
    public string Name { get; set; } = "";
    public int Age { get; set; }
    public string? Email { get; set; }
}

public class Product
{
    public int Id { get; set; }
    public string Name { get; set; } = "";
    public decimal Price { get; set; }
    public string? Description { get; set; }
    public List<string> Tags { get; set; } = new();
}

public class UserDto
{
    [JsonPropertyName("user_id")]
    public int UserId { get; set; }

    [JsonPropertyName("username")]
    public string Username { get; set; } = "";

    [JsonIgnore] // Never serialize password hash
    public string? PasswordHash { get; set; }

    [JsonPropertyName("created_at")]
    public DateTime CreatedAt { get; set; }

    [JsonPropertyName("role")]
    [JsonConverter(typeof(JsonStringEnumConverter))]
    public UserRole Role { get; set; }

    [JsonPropertyName("preferences")]
    public Dictionary<string, string> Preferences { get; set; } = new();
}

public enum UserRole { User, Moderator, Admin }

public class Order
{
    public string OrderId { get; set; } = "";
    public Customer Customer { get; set; } = new();
    public List<OrderItem> Items { get; set; } = new();
}

public class Customer
{
    public string Name { get; set; } = "";
    public string Address { get; set; } = "";
}

public class OrderItem
{
    public string Product { get; set; } = "";
    public int Quantity { get; set; }
    public decimal UnitPrice { get; set; }
}

// Polymorphic types with type discriminator
[JsonDerivedType(typeof(CircleShape), "circle")]
[JsonDerivedType(typeof(RectangleShape), "rectangle")]
public abstract class Shape
{
    public abstract double Area { get; }
}

public class CircleShape : Shape
{
    public double Radius { get; set; }
    public override double Area => Math.PI * Radius * Radius;
}

public class RectangleShape : Shape
{
    public double Width { get; set; }
    public double Height { get; set; }
    public override double Area => Width * Height;
}

public class AppConfig
{
    public string Name { get; set; } = "";
    public Version Version { get; set; } = new();
    public DateOnly LaunchDate { get; set; }
}

// Custom converter for System.Version
public class VersionConverter : JsonConverter<Version>
{
    public override Version Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        => Version.Parse(reader.GetString() ?? "0.0");

    public override void Write(Utf8JsonWriter writer, Version value, JsonSerializerOptions options)
        => writer.WriteStringValue(value.ToString());
}

// Custom converter for DateOnly
public class DateOnlyConverter : JsonConverter<DateOnly>
{
    private const string Format = "yyyy-MM-dd";

    public override DateOnly Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        => DateOnly.ParseExact(reader.GetString() ?? "", Format);

    public override void Write(Utf8JsonWriter writer, DateOnly value, JsonSerializerOptions options)
        => writer.WriteStringValue(value.ToString(Format));
}
