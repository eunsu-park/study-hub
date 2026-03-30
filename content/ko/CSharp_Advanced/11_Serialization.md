# 직렬화 (System.Text.Json)

**이전**: [의존성 주입](./10_Dependency_Injection.md) | **다음**: [테스팅](./12_Testing.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. `System.Text.Json`을 사용하여 C# 객체를 JSON으로 직렬화/역직렬화하기
2. `JsonSerializerOptions`로 직렬화 동작 커스터마이즈하기
3. 속성 어트리뷰트를 적용하여 JSON 출력 제어하기
4. 복잡한 타입을 위한 커스텀 `JsonConverter<T>` 구현 작성하기
5. `[JsonDerivedType]`으로 다형성 직렬화 사용하기
6. `[JsonSerializable]`을 사용하여 AOT 및 성능을 위한 소스 기반 시리얼라이저 생성하기
7. `Utf8JsonReader`, `Utf8JsonWriter`, `JsonDocument`, `JsonNode`로 JSON을 효율적으로 처리하기
8. 기존 코드를 Newtonsoft.Json에서 System.Text.Json으로 마이그레이션하기

---

직렬화(serialization)는 객체를 저장이나 전송에 적합한 형식으로 변환하는 과정이며, 역직렬화(deserialization)는 그 반대입니다. JSON(JavaScript Object Notation)은 웹 API, 구성 파일, 문서 데이터베이스에서 지배적인 데이터 교환 형식입니다. .NET Core 3.0부터 `System.Text.Json`은 내장된 고성능 JSON 라이브러리입니다. 이 레슨은 기본 직렬화부터 커스텀 컨버터, 다형성, 소스 제너레이터, 저수준 리더/라이터 같은 고급 주제까지 다룹니다.

## 1. JSON 직렬화 기초

### 1.1 직렬화와 역직렬화

```csharp
using System.Text.Json;

public class Person
{
    public string Name { get; set; } = "";
    public int Age { get; set; }
    public string Email { get; set; } = "";
}

// JSON 문자열로 직렬화
var person = new Person { Name = "Alice", Age = 30, Email = "alice@example.com" };
string json = JsonSerializer.Serialize(person);
// {"Name":"Alice","Age":30,"Email":"alice@example.com"}

// JSON 문자열에서 역직렬화
Person? deserialized = JsonSerializer.Deserialize<Person>(json);
Console.WriteLine(deserialized?.Name); // Alice
```

### 1.2 바이트로 직렬화 (UTF-8)

```csharp
// UTF-8 바이트로 직접 직렬화 (문자열을 거치는 것보다 빠름)
byte[] utf8Json = JsonSerializer.SerializeToUtf8Bytes(person);

// UTF-8 바이트에서 역직렬화
Person? fromBytes = JsonSerializer.Deserialize<Person>(utf8Json);
```

### 1.3 스트림으로/에서 직렬화

```csharp
// 파일에 쓰기
await using (var stream = File.Create("person.json"))
{
    await JsonSerializer.SerializeAsync(stream, person);
}

// 파일에서 읽기
await using (var stream = File.OpenRead("person.json"))
{
    Person? fromFile = await JsonSerializer.DeserializeAsync<Person>(stream);
    Console.WriteLine(fromFile?.Name);
}
```

## 2. JsonSerializerOptions

`JsonSerializerOptions`는 직렬화 동작의 거의 모든 측면을 제어합니다.

### 2.1 일반적인 옵션

```csharp
var options = new JsonSerializerOptions
{
    // 속성 명명
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase,  // "Name" -> "name"

    // 포맷팅
    WriteIndented = true,                                // 예쁘게 출력

    // null 처리
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,

    // 숫자 처리
    NumberHandling = JsonNumberHandling.AllowReadingFromString,

    // 속성 매칭
    PropertyNameCaseInsensitive = true,                  // "name"을 "Name"에 매칭

    // 인코더 (비ASCII 문자 허용)
    Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping,

    // 최대 깊이
    MaxDepth = 64,

    // 알 수 없는 속성
    UnmappedMemberHandling = JsonUnmappedMemberHandling.Skip // 또는 Disallow
};

string json = JsonSerializer.Serialize(person, options);
// {
//   "name": "Alice",
//   "age": 30,
//   "email": "alice@example.com"
// }
```

### 2.2 명명 정책

```csharp
// 내장 명명 정책
JsonNamingPolicy.CamelCase;      // "FirstName" -> "firstName"
JsonNamingPolicy.SnakeCaseLower; // "FirstName" -> "first_name"
JsonNamingPolicy.SnakeCaseUpper; // "FirstName" -> "FIRST_NAME"
JsonNamingPolicy.KebabCaseLower; // "FirstName" -> "first-name"
JsonNamingPolicy.KebabCaseUpper; // "FirstName" -> "FIRST-NAME"

// 커스텀 명명 정책
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

### 2.3 옵션 재사용

```csharp
// 한 번 생성하고 모든 곳에서 재사용 — 옵션은 내부적으로 메타데이터를 캐시함
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

// 사용법
string json = JsonSerializer.Serialize(person, JsonDefaults.Web);
```

## 3. 속성 어트리뷰트

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
    public int Score { get; set; } // 0일 때 생략됨
}
```

### 3.3 JsonInclude

기본적으로 공개 속성만 직렬화됩니다. `[JsonInclude]`를 사용하여 필드나 private setter를 포함합니다.

```csharp
public class Product
{
    [JsonInclude]
    public string _sku; // 필드 (속성이 아님) — 어트리뷰트로 포함됨

    public string Name { get; set; } = "";

    [JsonInclude]
    public decimal Price { get; private set; } // private setter 포함됨

    [JsonConstructor]
    public Product(string name, decimal price, string sku)
    {
        Name = name;
        Price = price;
        _sku = sku;
    }
}
```

### 3.4 JsonConstructor와 Required 속성

```csharp
public class Config
{
    [JsonRequired] // .NET 7+: 속성이 누락되면 역직렬화 실패
    public string AppName { get; set; } = "";

    public string Version { get; set; } = "1.0";

    [JsonConstructor]
    public Config(string appName, string version)
    {
        AppName = appName;
        Version = version;
    }
}

// C# 11 required 멤버도 작동:
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

## 4. 커스텀 컨버터 (JsonConverter&lt;T&gt;)

기본 직렬화 동작이 요구 사항과 맞지 않을 때 커스텀 컨버터를 작성합니다.

### 4.1 간단한 컨버터: Unix 타임스탬프

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

// 어트리뷰트를 통한 사용법
public class LogEntry
{
    public string Message { get; set; } = "";

    [JsonConverter(typeof(UnixTimestampConverter))]
    public DateTime Timestamp { get; set; }
}

// 또는 옵션을 통해
var options = new JsonSerializerOptions();
options.Converters.Add(new UnixTimestampConverter());
```

### 4.2 커스텀 타입용 컨버터

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
            // "99.99 USD" 형식 파싱
            string value = reader.GetString()!;
            int spaceIdx = value.IndexOf(' ');
            decimal amount = decimal.Parse(value[..spaceIdx]);
            string currency = value[(spaceIdx + 1)..];
            return new Money(amount, currency);
        }

        // {"amount": 99.99, "currency": "USD"} 형식 파싱
        if (reader.TokenType != JsonTokenType.StartObject)
            throw new JsonException("문자열 또는 객체가 예상됨");

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

### 4.3 컨버터 팩토리

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
                : throw new JsonException($"'{value}'를 {typeof(T).Name}으로 파싱할 수 없음");
        }

        public override void Write(Utf8JsonWriter writer, T value,
            JsonSerializerOptions options)
        {
            writer.WriteStringValue(value.ToString());
        }
    }
}
```

## 5. 다형성 직렬화

### 5.1 JsonDerivedType 어트리뷰트 (.NET 7+)

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
// 타입 판별자와 함께 직렬화
Shape circle = new Circle { Radius = 5, Color = "Red" };
string json = JsonSerializer.Serialize(circle);
// {"$type":"circle","Color":"Red","Radius":5}

// 다형적으로 역직렬화
Shape? shape = JsonSerializer.Deserialize<Shape>(json);
Console.WriteLine(shape?.GetType().Name); // Circle
Console.WriteLine(shape?.Area());          // 78.54...
```

### 5.2 다형성 컬렉션

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
    Console.WriteLine($"{s.GetType().Name}: 면적 = {s.Area():F2}, 색상 = {s.Color}");
}
```

## 6. JSON 소스 제너레이터

소스 제너레이터(source generator)는 컴파일 시간에 직렬화 메타데이터를 미리 계산하여 런타임 리플렉션을 제거합니다. 이는 AOT(Ahead-of-Time) 컴파일에 필수적이며 시작 성능을 향상시킵니다.

### 6.1 기본 소스 제너레이터 설정

```csharp
// partial 컨텍스트 클래스 정의
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
// 생성된 컨텍스트와 함께 사용
var person = new Person { Name = "Alice", Age = 30, Email = "alice@example.com" };

// 타입 안전하고 AOT 호환 직렬화
string json = JsonSerializer.Serialize(person, AppJsonContext.Default.Person);

// 역직렬화
Person? result = JsonSerializer.Deserialize(json, AppJsonContext.Default.Person);

// 컬렉션 직렬화
var people = new List<Person> { person };
string listJson = JsonSerializer.Serialize(people, AppJsonContext.Default.ListPerson);
```

### 6.2 성능 이점

```csharp
// 벤치마크 비교 (개념적)
// 소스 제너레이터 없이: 첫 호출 100μs (리플렉션), 이후 5μs
// 소스 제너레이터와 함께: 첫 호출 5μs (리플렉션 없음), 이후 5μs
// AOT: 제너레이터 없이 -> 런타임에 실패. 제너레이터와 함께 -> 작동.
```

### 6.3 ASP.NET Core에 등록

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

## 7. 저수준: Utf8JsonReader와 Utf8JsonWriter

최대 성능이나 스트리밍 시나리오를 위해 저수준 리더/라이터를 직접 사용합니다.

### 7.1 Utf8JsonWriter

```csharp
using var stream = new MemoryStream();
using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions
{
    Indented = true,
    SkipValidation = false // 신뢰할 수 있는 코드에서는 성능을 위해 true로 설정
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
        reader.Read(); // 값으로 이동

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

Console.WriteLine($"{name}, {age}, 점수: [{string.Join(", ", scores)}]");
```

## 8. 동적 JSON을 위한 JsonNode와 JsonDocument

### 8.1 JsonDocument (읽기 전용)

`JsonDocument`는 특정 타입에 매핑하지 않고 파싱된 JSON에 대한 읽기 전용 접근을 제공합니다.

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
Console.WriteLine($"전체: {total}");

foreach (JsonElement user in root.GetProperty("users").EnumerateArray())
{
    string name = user.GetProperty("name").GetString()!;
    int age = user.GetProperty("age").GetInt32();
    Console.WriteLine($"  {name}, 나이 {age}");
}
```

### 8.2 JsonNode (변경 가능)

`JsonNode`는 JSON을 빌드, 수정, 쿼리하기 위한 변경 가능한 DOM을 제공합니다.

```csharp
using System.Text.Json.Nodes;

// 동적으로 JSON 빌드
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

// 수정
node["age"] = 31;
node["tags"]!.AsArray().Add("author");
node["phone"] = "555-1234";

// 쿼리
string city = node["address"]!["city"]!.GetValue<string>();
Console.WriteLine(city); // Seattle

// 문자열로 직렬화
string json = node.ToJsonString(new JsonSerializerOptions { WriteIndented = true });
Console.WriteLine(json);
```

### 8.3 기존 JSON 파싱과 수정

```csharp
string input = """{"config": {"debug": false, "logLevel": "info", "maxRetries": 3}}""";

JsonNode? root = JsonNode.Parse(input);
JsonObject config = root!["config"]!.AsObject();

// 값 업데이트
config["debug"] = true;
config["logLevel"] = "debug";
config["newSetting"] = "value";

// 속성 제거
config.Remove("maxRetries");

Console.WriteLine(root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
```

## 9. 열거형, 날짜, Null 처리

### 9.1 열거형 직렬화

```csharp
public enum Status { Active, Inactive, Suspended }

// 기본값: 숫자로 직렬화
// {"Status":0}

// 문자열로:
var options = new JsonSerializerOptions
{
    Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) }
};
// {"status":"active"}
```

### 9.2 날짜와 시간

```csharp
public class Event
{
    public string Name { get; set; } = "";
    public DateTime StartTime { get; set; }           // 기본적으로 ISO 8601
    public DateTimeOffset CreatedAt { get; set; }      // 오프셋 포함 ISO 8601
    public DateOnly Date { get; set; }                 // "2024-01-15"
    public TimeOnly Time { get; set; }                 // "14:30:00"
    public TimeSpan Duration { get; set; }             // 기본적으로 지원되지 않음
}

// TimeSpan 컨버터
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

### 9.3 Null 처리 전략

```csharp
var options = new JsonSerializerOptions
{
    // 전략 1: 전역적으로 모든 null 속성 무시
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
};

// 전략 2: 속성별 제어
public class UserProfile
{
    public string Name { get; set; } = "";

    [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
    public string? Bio { get; set; }

    [JsonIgnore(Condition = JsonIgnoreCondition.Never)] // null이어도 항상 포함
    public string? Website { get; set; }
}
```

## 10. Newtonsoft.Json에서의 마이그레이션 팁

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

// System.Text.Json 동등한 코드
var options = new JsonSerializerOptions
{
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
    WriteIndented = true,
    PropertyNamingPolicy = JsonNamingPolicy.CamelCase
};
string json = JsonSerializer.Serialize(obj, options);
```

## 11. 실전 예제: 구성 파일 리더/라이터

이 예제는 JSON 구성 파일을 읽고, 쓰고, 유효성을 검사하고, 감시하는 완전한 구성 파일 관리자를 구축합니다.

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
            throw new FileNotFoundException($"구성 파일을 찾을 수 없음: {_filePath}");

        await using var stream = File.OpenRead(_filePath);
        var config = await JsonSerializer.DeserializeAsync<AppConfiguration>(stream, _options);

        return config ?? throw new JsonException("구성 역직렬화 실패");
    }

    public async Task SaveAsync(AppConfiguration config)
    {
        string directory = Path.GetDirectoryName(_filePath)!;
        if (!Directory.Exists(directory))
            Directory.CreateDirectory(directory);

        // 먼저 임시 파일에 쓴 다음 이름 변경 (원자적 작업)
        string tempPath = _filePath + ".tmp";
        await using (var stream = File.Create(tempPath))
        {
            await JsonSerializer.SerializeAsync(stream, config, _options);
        }
        File.Move(tempPath, _filePath, overwrite: true);
    }

    public async Task MergeAsync(string partialJson)
    {
        // 병합을 위해 현재 구성을 JsonNode로 로드
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
            await Task.Delay(100); // 디바운스
            try
            {
                var config = await LoadAsync();
                ConfigChanged?.Invoke(config);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"구성 다시 로드 오류: {ex.Message}");
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
// 사용법
var manager = new ConfigManager("config/appsettings.json");

// 초기 구성 생성
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

// 로드하여 사용
var loaded = await manager.LoadAsync();
Console.WriteLine($"앱: {loaded.AppName} v{loaded.Version}");
Console.WriteLine($"DB 풀: {loaded.Database.MaxPoolSize}");
Console.WriteLine($"로그 레벨: {loaded.Logging.Level}");

// 부분 업데이트 병합
await manager.MergeAsync("""{"logging": {"level": "warning"}}""");

// 변경 감시
manager.ConfigChanged += cfg =>
    Console.WriteLine($"구성 다시 로드됨: LogLevel = {cfg.Logging.Level}");

manager.StartWatching();
```

## 12. 연습 문제

1. **제네릭 API 응답 래퍼**: `Status` (열거형: Success, Error, NotFound), `Data` (T?), `Errors` (List<string>?), `Timestamp`를 가진 제네릭 `ApiResponse<T>` 클래스를 만드세요. 타임스탬프를 Unix 에포크로 직렬화하고 빈 에러 리스트를 생략하는 커스텀 `JsonConverter<ApiResponse<T>>`를 작성하세요. 다양한 페이로드의 왕복 테스트를 작성하세요.

2. **JSON 스키마 유효성 검사기**: `JsonDocument`를 사용하여 스키마 정의(필수 필드, string/number/boolean 타입 검사, 숫자의 min/max, 문자열의 minLength/maxLength)에 대해 JSON 문서를 검사하는 간단한 유효성 검사기를 작성하세요. 스키마 자체도 JSON 파일에서 로드해야 합니다. 최소 5개의 다른 JSON 샘플을 유효성 검사하고 모든 유효성 검사 오류를 보고하세요.

3. **JSON Diff 도구**: 두 JSON 문서를 비교하고 차이점 목록을 반환하는 `IReadOnlyList<JsonDiff> ComputeDiff(string jsonA, string jsonB)` 메서드를 작성하세요. 각 diff에는 JSON 경로(예: `$.users[0].name`), 이전 값, 새 값, 변경 유형(Added, Removed, Modified)을 포함해야 합니다. 역직렬화 없이 효율적인 비교를 위해 `JsonDocument`를 사용하세요.

4. **스트리밍 JSON 파서**: `Utf8JsonReader`를 사용하여 대용량 JSON 배열 파일(10MB JSON 객체 배열로 시뮬레이션)을 스트리밍 방식으로 파싱하세요. 전체 파일을 메모리에 로드하지 않고 각 객체를 개별적으로 처리합니다. 레코드 수를 세고, 숫자 필드의 평균을 계산하고, 최댓값을 찾으세요. 파일 크기에 관계없이 메모리 사용량이 일정하게 유지됨을 증명하세요.

5. **다국어 시리얼라이저**: 여러 형식으로 객체를 직렬화/역직렬화할 수 있는 `DocumentStore` 클래스를 만드세요. `System.Text.Json`, `Utf8JsonWriter` (수동), 간단한 커스텀 텍스트 형식을 지원합니다. 세 가지 구현이 있는 `IDocumentSerializer` 인터페이스를 사용하세요. 세 가지 모두 키 DI 서비스로 등록하세요. 동일한 복잡한 객체 그래프(중첩 객체, 컬렉션, 열거형 포함)에 대해 각 형식의 왕복 테스트를 작성하세요.
