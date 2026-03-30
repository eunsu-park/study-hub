/*
 * Exercises for Lesson 11: Serialization
 * Topic: CSharp_Advanced
 * Solutions to practice problems from the lesson.
 * Run: dotnet run
 */

using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Text.Json.Serialization;

// ---------------------------------------------------------------------------
// Exercise 1: Basic JSON serialization/deserialization
// ---------------------------------------------------------------------------
void Exercise1()
{
    Console.WriteLine("=== Exercise 1: Basic JSON Serialization ===");

    var person = new Person("Alice", 30, "alice@example.com");
    var options = new JsonSerializerOptions { WriteIndented = true };

    string json = JsonSerializer.Serialize(person, options);
    Console.WriteLine($"  Serialized:\n{json}");

    var deserialized = JsonSerializer.Deserialize<Person>(json);
    Console.WriteLine($"  Deserialized: {deserialized}");
    Console.WriteLine($"  Equal: {person == deserialized}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 2: Custom naming policy and property attributes
// ---------------------------------------------------------------------------
void Exercise2()
{
    Console.WriteLine("=== Exercise 2: Naming Policy and Attributes ===");

    var config = new AppConfig
    {
        DatabaseHost = "localhost",
        DatabasePort = 5432,
        MaxRetries = 3,
        EnableLogging = true,
        SecretKey = "super-secret"
    };

    var options = new JsonSerializerOptions
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower
    };

    string json = JsonSerializer.Serialize(config, options);
    Console.WriteLine($"  Serialized (snake_case):\n{json}");

    // SecretKey should be ignored
    Console.WriteLine($"  Contains 'secret': {json.Contains("secret", StringComparison.OrdinalIgnoreCase)}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 3: Custom JsonConverter — date-only converter
// ---------------------------------------------------------------------------
void Exercise3()
{
    Console.WriteLine("=== Exercise 3: Custom JsonConverter ===");

    var evt = new CalendarEvent
    {
        Title = "Team Meeting",
        Date = new DateTime(2025, 6, 15),
        Duration = TimeSpan.FromHours(1.5)
    };

    var options = new JsonSerializerOptions
    {
        WriteIndented = true,
        Converters = { new DateOnlyConverter(), new TimeSpanToMinutesConverter() }
    };

    string json = JsonSerializer.Serialize(evt, options);
    Console.WriteLine($"  Serialized:\n{json}");

    var deserialized = JsonSerializer.Deserialize<CalendarEvent>(json, options);
    Console.WriteLine($"  Deserialized: {deserialized?.Title}, {deserialized?.Date:yyyy-MM-dd}, {deserialized?.Duration}");
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 4: Polymorphic serialization with type discriminator
// ---------------------------------------------------------------------------
void Exercise4()
{
    Console.WriteLine("=== Exercise 4: Polymorphic Serialization ===");

    var notifications = new List<Notification>
    {
        new EmailNotification { Recipient = "alice@test.com", Subject = "Hello" },
        new SmsNotification { PhoneNumber = "+1234567890", Message = "Hi there" },
        new PushNotification { DeviceToken = "abc123", Title = "Alert", Badge = 5 },
    };

    var options = new JsonSerializerOptions
    {
        WriteIndented = true,
        Converters = { new NotificationConverter() }
    };

    string json = JsonSerializer.Serialize(notifications, options);
    Console.WriteLine($"  Serialized:\n{json}");

    var back = JsonSerializer.Deserialize<List<Notification>>(json, options);
    Console.WriteLine($"  Deserialized {back?.Count} notifications:");
    back?.ForEach(n => Console.WriteLine($"    {n.GetType().Name}: {n.Describe()}"));
    Console.WriteLine();
}

// ---------------------------------------------------------------------------
// Exercise 5: Handling missing/extra fields gracefully
// ---------------------------------------------------------------------------
void Exercise5()
{
    Console.WriteLine("=== Exercise 5: Resilient Deserialization ===");

    string jsonWithExtra = """{"Name":"Bob","Age":25,"Email":"bob@test.com","Nickname":"Bobby","Score":100}""";
    string jsonWithMissing = """{"Name":"Charlie"}""";

    var opts = new JsonSerializerOptions
    {
        PropertyNameCaseInsensitive = true,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
    };

    var p1 = JsonSerializer.Deserialize<Person>(jsonWithExtra, opts);
    Console.WriteLine($"  Extra fields: {p1}");

    var p2 = JsonSerializer.Deserialize<Person>(jsonWithMissing, opts);
    Console.WriteLine($"  Missing fields: {p2}");

    // Re-serialize with nulls omitted
    string compact = JsonSerializer.Serialize(p2, opts);
    Console.WriteLine($"  Compact: {compact}");
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

record Person(string Name, int Age, string? Email = null);

class AppConfig
{
    public string DatabaseHost { get; set; } = "";
    public int DatabasePort { get; set; }
    public int MaxRetries { get; set; }
    public bool EnableLogging { get; set; }
    [JsonIgnore]
    public string SecretKey { get; set; } = "";
}

class CalendarEvent
{
    public string Title { get; set; } = "";
    public DateTime Date { get; set; }
    public TimeSpan Duration { get; set; }
}

class DateOnlyConverter : JsonConverter<DateTime>
{
    public override DateTime Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions opts)
        => DateTime.Parse(reader.GetString()!);
    public override void Write(Utf8JsonWriter writer, DateTime value, JsonSerializerOptions opts)
        => writer.WriteStringValue(value.ToString("yyyy-MM-dd"));
}

class TimeSpanToMinutesConverter : JsonConverter<TimeSpan>
{
    public override TimeSpan Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions opts)
        => TimeSpan.FromMinutes(reader.GetDouble());
    public override void Write(Utf8JsonWriter writer, TimeSpan value, JsonSerializerOptions opts)
        => writer.WriteNumberValue(value.TotalMinutes);
}

abstract class Notification
{
    public abstract string Type { get; }
    public abstract string Describe();
}

class EmailNotification : Notification
{
    public override string Type => "email";
    public string Recipient { get; set; } = "";
    public string Subject { get; set; } = "";
    public override string Describe() => $"Email to {Recipient}: {Subject}";
}

class SmsNotification : Notification
{
    public override string Type => "sms";
    public string PhoneNumber { get; set; } = "";
    public string Message { get; set; } = "";
    public override string Describe() => $"SMS to {PhoneNumber}: {Message}";
}

class PushNotification : Notification
{
    public override string Type => "push";
    public string DeviceToken { get; set; } = "";
    public string Title { get; set; } = "";
    public int Badge { get; set; }
    public override string Describe() => $"Push '{Title}' badge={Badge}";
}

class NotificationConverter : JsonConverter<Notification>
{
    public override Notification Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions opts)
    {
        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;
        string type = root.GetProperty("Type").GetString()!;
        string raw = root.GetRawText();
        return type switch
        {
            "email" => JsonSerializer.Deserialize<EmailNotification>(raw)!,
            "sms" => JsonSerializer.Deserialize<SmsNotification>(raw)!,
            "push" => JsonSerializer.Deserialize<PushNotification>(raw)!,
            _ => throw new JsonException($"Unknown type: {type}")
        };
    }

    public override void Write(Utf8JsonWriter writer, Notification value, JsonSerializerOptions opts)
    {
        JsonSerializer.Serialize(writer, value, value.GetType(), opts);
    }
}
