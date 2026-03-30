# 속성과 인덱서

**이전**: [클래스와 객체](./09_Classes_and_Objects.md) | **다음**: [상속](./11_Inheritance.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 명시적 백킹 필드가 있는 전체 속성 구문 작성하기
2. 간결한 선언을 위한 자동 구현 속성 사용하기
3. 읽기 전용(Read-Only)과 초기화 전용(Init-Only) 속성 만들기
4. 식 본문을 사용한 계산된 속성 정의하기
5. 속성 설정자에 유효성 검사 논리 추가하기
6. 사용자 정의 컬렉션 유사 접근을 위한 인덱서 구현하기
7. 정적 속성과 `required` 키워드 사용하기

---

속성(Property)은 객체의 데이터에 대한 제어된 접근을 제공하는 C#의 핵심 기능입니다. 호출자에게는 필드처럼 보이지만 내부적으로는 메서드처럼 동작하여 유효성 검사, 계산, 캡슐화를 허용합니다. 인덱서(Indexer)는 이 개념을 확장하여 대괄호를 사용한 배열 유사 접근을 제공합니다. 속성과 인덱서는 잘 설계된 C# 클래스에서 데이터를 노출하는 표준 방법을 형성합니다.

## 1. 전체 속성 구문

전체(수동 구현) 속성은 `get`과 `set` 접근자가 있는 명시적 백킹 필드(Backing Field)를 사용합니다.

### 1.1 기본 Get/Set 속성

```csharp
class Person
{
    // 백킹 필드
    private string _name;

    // 명시적 get과 set이 있는 속성
    public string Name
    {
        get
        {
            return _name;
        }
        set
        {
            _name = value; // 'value'는 set의 암시적 매개변수
        }
    }

    // 백킹 필드가 있는 또 다른 속성
    private int _age;

    public int Age
    {
        get { return _age; }
        set { _age = value; }
    }

    public Person(string name, int age)
    {
        _name = name;
        _age = age;
    }
}

Person p = new Person("Alice", 30);
Console.WriteLine(p.Name); // get 접근자 호출 -> "Alice"
p.Age = 31;                // set 접근자 호출
Console.WriteLine(p.Age);  // 31
```

### 1.2 읽기 전용과 쓰기 전용 속성

```csharp
class Temperature
{
    private double _celsius;

    // 읽기 전용 속성 (get 접근자만)
    public double Celsius
    {
        get { return _celsius; }
    }

    // 쓰기 전용 속성 (set 접근자만) — 드물지만 유효
    public double SetCelsius
    {
        set { _celsius = value; }
    }

    public Temperature(double celsius)
    {
        _celsius = celsius;
    }
}

Temperature t = new Temperature(100);
Console.WriteLine(t.Celsius); // 100
// t.Celsius = 200;          // 컴파일 오류: 읽기 전용 속성

t.SetCelsius = 200;          // OK: 쓰기 전용 속성
Console.WriteLine(t.Celsius); // 200
```

### 1.3 접근자의 접근 한정자

하나의 접근자를 속성 자체보다 더 제한적으로 만들 수 있습니다:

```csharp
class Account
{
    private decimal _balance;

    // public get, private set
    public decimal Balance
    {
        get { return _balance; }
        private set { _balance = value; } // 이 클래스 내에서만 접근 가능
    }

    public Account(decimal initialBalance)
    {
        Balance = initialBalance; // private set 사용
    }

    public void Deposit(decimal amount)
    {
        Balance += amount; // 내부적으로 private set 사용
    }
}

Account acc = new Account(1000);
Console.WriteLine(acc.Balance); // 1000 (public get)
acc.Deposit(500);
Console.WriteLine(acc.Balance); // 1500
// acc.Balance = 9999;          // 컴파일 오류: set은 private
```

## 2. 자동 구현 속성

get이나 set 접근자에 특별한 논리가 필요 없을 때, 자동 속성은 간결한 구문을 제공합니다. 컴파일러가 숨겨진 백킹 필드를 자동으로 생성합니다.

### 2.1 기본 자동 속성

```csharp
class Car
{
    // 자동 구현 속성
    public string Make { get; set; }
    public string Model { get; set; }
    public int Year { get; set; }
    public double Mileage { get; set; }

    public Car(string make, string model, int year)
    {
        Make = make;
        Model = model;
        Year = year;
        Mileage = 0;
    }

    public override string ToString()
        => $"{Year} {Make} {Model} ({Mileage:N0} 마일)";
}

Car car = new Car("Toyota", "Camry", 2023);
car.Mileage = 15000;
Console.WriteLine(car); // "2023 Toyota Camry (15,000 마일)"
```

### 2.2 기본값이 있는 자동 속성

```csharp
class Settings
{
    public string Theme { get; set; } = "Light";
    public int FontSize { get; set; } = 14;
    public bool ShowLineNumbers { get; set; } = true;
    public List<string> RecentFiles { get; set; } = new();

    public override string ToString()
        => $"Theme={Theme}, FontSize={FontSize}, Lines={ShowLineNumbers}";
}

Settings s = new Settings();
Console.WriteLine(s.Theme);    // "Light" (기본값)
Console.WriteLine(s.FontSize); // 14 (기본값)

s.Theme = "Dark";
Console.WriteLine(s); // "Theme=Dark, FontSize=14, Lines=True"
```

### 2.3 Private Set 자동 속성

```csharp
class Order
{
    public string OrderId { get; private set; }
    public DateTime CreatedAt { get; private set; }
    public string Status { get; private set; }

    public Order(string orderId)
    {
        OrderId = orderId;
        CreatedAt = DateTime.Now;
        Status = "대기 중";
    }

    public void Ship()
    {
        Status = "배송됨"; // 허용됨: 클래스 내에서
    }

    public void Deliver()
    {
        Status = "배달 완료";
    }
}

Order order = new Order("ORD-001");
Console.WriteLine(order.Status); // "대기 중"
order.Ship();
Console.WriteLine(order.Status); // "배송됨"
// order.Status = "취소됨";   // 컴파일 오류: private set
```

## 3. 읽기 전용 속성

`get` 접근자만 있는 속성은 읽기 전용이며 생성 시에만 설정할 수 있습니다.

### 3.1 Get 전용 자동 속성

```csharp
class ImmutablePoint
{
    // Get 전용: 생성자나 초기화자에서만 설정 가능
    public double X { get; }
    public double Y { get; }

    public ImmutablePoint(double x, double y)
    {
        X = x; // 생성자에서 허용
        Y = y;
    }

    public double DistanceTo(ImmutablePoint other)
    {
        double dx = X - other.X;
        double dy = Y - other.Y;
        return Math.Sqrt(dx * dx + dy * dy);
    }
}

ImmutablePoint p = new ImmutablePoint(3, 4);
Console.WriteLine(p.X); // 3
// p.X = 10;            // 컴파일 오류: get 전용 속성
```

### 3.2 백킹 필드가 있는 Readonly

```csharp
class Circle
{
    private readonly double _radius;

    public double Radius
    {
        get { return _radius; }
    }

    // 읽기 전용 계산된 속성 (백킹 필드 불필요)
    public double Area
    {
        get { return Math.PI * _radius * _radius; }
    }

    public double Circumference
    {
        get { return 2 * Math.PI * _radius; }
    }

    public Circle(double radius)
    {
        if (radius < 0)
            throw new ArgumentException("반지름은 음수가 될 수 없습니다.");
        _radius = radius;
    }
}

Circle c = new Circle(5);
Console.WriteLine($"반지름: {c.Radius}");           // 5
Console.WriteLine($"면적: {c.Area:F2}");             // 78.54
Console.WriteLine($"둘레: {c.Circumference:F2}"); // 31.42
```

## 4. 초기화 전용 속성

C# 9에서 `init` 접근자가 도입되었습니다. 이는 객체 초기화(생성자 또는 객체 초기화자) 중에만 속성을 설정할 수 있게 하며, 이후에는 설정할 수 없습니다.

### 4.1 기본 초기화 전용 속성

```csharp
class UserProfile
{
    public string Username { get; init; }
    public string Email { get; init; }
    public DateTime JoinDate { get; init; }
    public string Bio { get; set; } = ""; // 생성 후에도 변경 가능

    public UserProfile() { }
}

// 초기화 중에 설정 가능
UserProfile profile = new UserProfile
{
    Username = "alice",
    Email = "alice@example.com",
    JoinDate = DateTime.Now
};

// 변경 가능한 속성은 여전히 수정 가능
profile.Bio = "소프트웨어 개발자";

// 초기화 후에는 init 전용 속성을 수정할 수 없음
// profile.Username = "bob";    // 컴파일 오류
// profile.Email = "new@email"; // 컴파일 오류
```

### 4.2 생성자와 함께 초기화 전용

```csharp
class Product
{
    public string Name { get; init; }
    public decimal Price { get; init; }
    public string Category { get; init; }

    // 생성자도 init 전용 속성을 설정할 수 있음
    public Product(string name, decimal price, string category)
    {
        Name = name;
        Price = price;
        Category = category;
    }
}

Product p = new Product("노트북", 999.99m, "전자제품");
// p.Price = 899.99m; // 컴파일 오류: init 전용
```

### 4.3 레코드와 함께 초기화 전용

초기화 전용 속성은 불변 데이터를 위한 레코드와 자연스럽게 작동합니다:

```csharp
record class Customer
{
    public string Name { get; init; }
    public string Email { get; init; }
    public string Tier { get; init; } = "Standard";
}

Customer c = new Customer { Name = "Alice", Email = "alice@ex.com" };

// 'with' 표현식으로 수정된 복사본 생성
Customer upgraded = c with { Tier = "Premium" };

Console.WriteLine(c.Tier);        // "Standard"
Console.WriteLine(upgraded.Tier); // "Premium"
```

## 5. 계산된 속성

계산된(Computed) 속성은 값을 직접 저장하지 않고 다른 데이터에서 값을 도출합니다.

### 5.1 식 본문 속성

```csharp
class Rectangle
{
    public double Width { get; set; }
    public double Height { get; set; }

    // 식 본문 읽기 전용 속성 (접근할 때마다 계산)
    public double Area => Width * Height;
    public double Perimeter => 2 * (Width + Height);
    public double Diagonal => Math.Sqrt(Width * Width + Height * Height);
    public bool IsSquare => Width == Height;
    public string Summary => $"{Width}x{Height} (면적={Area:F1})";

    public Rectangle(double width, double height)
    {
        Width = width;
        Height = height;
    }
}

Rectangle r = new Rectangle(4, 3);
Console.WriteLine(r.Area);      // 12
Console.WriteLine(r.Diagonal);  // 5
Console.WriteLine(r.IsSquare);  // false
Console.WriteLine(r.Summary);   // "4x3 (면적=12.0)"

r.Width = 3;
Console.WriteLine(r.IsSquare);  // true (다시 계산됨)
```

### 5.2 식 본문 메서드 vs 속성

```csharp
class DateRange
{
    public DateTime Start { get; init; }
    public DateTime End { get; init; }

    // 속성: 저렴하고 상태에서 파생된 값에 사용
    public int TotalDays => (End - Start).Days;
    public bool IsActive => DateTime.Now >= Start && DateTime.Now <= End;

    // 메서드: 작업을 수행하거나, 부작용이 있거나, 매개변수를 받을 때 사용
    public bool Contains(DateTime date) => date >= Start && date <= End;
    public DateRange ExtendBy(int days) => new DateRange { Start = Start, End = End.AddDays(days) };
}
```

### 5.3 논리가 있는 전체 계산된 속성

```csharp
class TemperatureConverter
{
    private double _celsius;

    public double Celsius
    {
        get => _celsius;
        set => _celsius = value;
    }

    // 전체 get/set 논리가 있는 계산된 속성
    public double Fahrenheit
    {
        get => _celsius * 9.0 / 5.0 + 32.0;
        set => _celsius = (value - 32.0) * 5.0 / 9.0;
    }

    public double Kelvin
    {
        get => _celsius + 273.15;
        set => _celsius = value - 273.15;
    }
}

TemperatureConverter t = new TemperatureConverter();

t.Celsius = 100;
Console.WriteLine(t.Fahrenheit); // 212
Console.WriteLine(t.Kelvin);    // 373.15

t.Fahrenheit = 32;
Console.WriteLine(t.Celsius);   // 0
Console.WriteLine(t.Kelvin);    // 273.15
```

## 6. 속성 유효성 검사

속성이 필드보다 나은 주요 장점 중 하나는 설정자에서 값을 검증할 수 있다는 것입니다.

### 6.1 Set 접근자에서의 유효성 검사

```csharp
class Student
{
    private string _name;
    private int _age;
    private double _gpa;

    public string Name
    {
        get => _name;
        set
        {
            if (string.IsNullOrWhiteSpace(value))
                throw new ArgumentException("이름은 비어 있을 수 없습니다.");
            if (value.Length > 100)
                throw new ArgumentException("이름은 100자를 초과할 수 없습니다.");
            _name = value.Trim();
        }
    }

    public int Age
    {
        get => _age;
        set
        {
            if (value < 0 || value > 150)
                throw new ArgumentOutOfRangeException(nameof(value), "나이는 0-150이어야 합니다.");
            _age = value;
        }
    }

    public double GPA
    {
        get => _gpa;
        set
        {
            if (value < 0.0 || value > 4.0)
                throw new ArgumentOutOfRangeException(nameof(value), "GPA는 0.0-4.0이어야 합니다.");
            _gpa = value;
        }
    }

    public Student(string name, int age, double gpa)
    {
        Name = name;   // 설정자에서 유효성 검사 실행
        Age = age;
        GPA = gpa;
    }
}

Student s = new Student("Alice", 20, 3.8);
Console.WriteLine(s.Name); // "Alice"

try
{
    s.Age = -5; // ArgumentOutOfRangeException 발생
}
catch (ArgumentOutOfRangeException ex)
{
    Console.WriteLine(ex.Message);
}
```

### 6.2 변경 알림

```csharp
class ObservableValue
{
    private int _value;

    public int Value
    {
        get => _value;
        set
        {
            if (_value != value)
            {
                int oldValue = _value;
                _value = value;
                Console.WriteLine($"값 변경: {oldValue} -> {value}");
            }
        }
    }
}

ObservableValue ov = new ObservableValue();
ov.Value = 10; // "값 변경: 0 -> 10"
ov.Value = 10; // 출력 없음 (같은 값)
ov.Value = 20; // "값 변경: 10 -> 20"
```

### 6.3 값 클램핑

```csharp
class AudioPlayer
{
    private int _volume;

    public int Volume
    {
        get => _volume;
        set => _volume = Math.Clamp(value, 0, 100); // 범위 내로 자동 제한
    }

    private double _playbackSpeed;

    public double PlaybackSpeed
    {
        get => _playbackSpeed;
        set => _playbackSpeed = Math.Clamp(value, 0.25, 4.0);
    }

    public AudioPlayer()
    {
        Volume = 50;
        PlaybackSpeed = 1.0;
    }
}

AudioPlayer player = new AudioPlayer();
player.Volume = 200;     // 100으로 제한
Console.WriteLine(player.Volume); // 100

player.Volume = -10;     // 0으로 제한
Console.WriteLine(player.Volume); // 0
```

## 7. 인덱서

인덱서(Indexer)를 사용하면 배열처럼 대괄호 표기법으로 객체에 접근할 수 있습니다.

### 7.1 기본 인덱서

```csharp
class Sentence
{
    private string[] _words;

    public Sentence(string text)
    {
        _words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    }

    // 인덱서
    public string this[int index]
    {
        get
        {
            if (index < 0 || index >= _words.Length)
                throw new IndexOutOfRangeException();
            return _words[index];
        }
        set
        {
            if (index < 0 || index >= _words.Length)
                throw new IndexOutOfRangeException();
            _words[index] = value;
        }
    }

    public int WordCount => _words.Length;

    public override string ToString() => string.Join(" ", _words);
}

Sentence s = new Sentence("The quick brown fox");
Console.WriteLine(s[0]);     // "The"
Console.WriteLine(s[2]);     // "brown"

s[3] = "dog";
Console.WriteLine(s);        // "The quick brown dog"
Console.WriteLine(s.WordCount); // 4
```

### 7.2 문자열 키 인덱서

인덱서는 정수 키에 제한되지 않습니다:

```csharp
class JsonObject
{
    private Dictionary<string, object> _data = new();

    // 문자열 인덱서
    public object this[string key]
    {
        get
        {
            if (!_data.ContainsKey(key))
                throw new KeyNotFoundException($"키 '{key}'를 찾을 수 없습니다.");
            return _data[key];
        }
        set
        {
            _data[key] = value;
        }
    }

    public bool HasKey(string key) => _data.ContainsKey(key);
    public int Count => _data.Count;
}

JsonObject obj = new JsonObject();
obj["name"] = "Alice";
obj["age"] = 30;
obj["active"] = true;

Console.WriteLine(obj["name"]); // "Alice"
Console.WriteLine(obj["age"]);  // 30
Console.WriteLine(obj.Count);   // 3
```

### 7.3 읽기 전용 인덱서

```csharp
class FibonacciSequence
{
    private Dictionary<int, long> _cache = new() { [0] = 0, [1] = 1 };

    // 읽기 전용 인덱서 (식 본문)
    public long this[int n] => GetFibonacci(n);

    private long GetFibonacci(int n)
    {
        if (n < 0)
            throw new ArgumentOutOfRangeException(nameof(n));
        if (_cache.TryGetValue(n, out long cached))
            return cached;
        long result = GetFibonacci(n - 1) + GetFibonacci(n - 2);
        _cache[n] = result;
        return result;
    }
}

FibonacciSequence fib = new FibonacciSequence();
Console.WriteLine(fib[0]);   // 0
Console.WriteLine(fib[1]);   // 1
Console.WriteLine(fib[10]);  // 55
Console.WriteLine(fib[20]);  // 6765
```

## 8. 다중 매개변수 인덱서

인덱서는 여러 매개변수를 받을 수 있어, 그리드나 행렬 유사 접근에 유용합니다.

### 8.1 2차원 인덱서

```csharp
class Grid<T>
{
    private T[,] _data;

    public int Rows { get; }
    public int Columns { get; }

    public Grid(int rows, int columns)
    {
        Rows = rows;
        Columns = columns;
        _data = new T[rows, columns];
    }

    // 다중 매개변수 인덱서
    public T this[int row, int col]
    {
        get
        {
            ValidateBounds(row, col);
            return _data[row, col];
        }
        set
        {
            ValidateBounds(row, col);
            _data[row, col] = value;
        }
    }

    private void ValidateBounds(int row, int col)
    {
        if (row < 0 || row >= Rows || col < 0 || col >= Columns)
            throw new IndexOutOfRangeException(
                $"({row},{col})은 {Rows}x{Columns} 그리드의 범위를 벗어납니다.");
    }

    public void Fill(T value)
    {
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Columns; c++)
                _data[r, c] = value;
    }
}

Grid<int> grid = new Grid<int>(3, 4);
grid[0, 0] = 1;
grid[1, 2] = 42;
grid[2, 3] = 99;

Console.WriteLine(grid[1, 2]); // 42
```

### 8.2 혼합 타입 인덱서 매개변수

```csharp
class SpreadSheet
{
    private Dictionary<(int Row, string Col), string> _cells = new();

    // int 행과 string 열이 있는 인덱서 ("A1", "B3" 같은)
    public string this[int row, string col]
    {
        get => _cells.TryGetValue((row, col), out string? val) ? val : "";
        set => _cells[(row, col)] = value;
    }

    public int CellCount => _cells.Count;
}

SpreadSheet sheet = new SpreadSheet();
sheet[1, "A"] = "이름";
sheet[1, "B"] = "나이";
sheet[2, "A"] = "Alice";
sheet[2, "B"] = "30";

Console.WriteLine(sheet[1, "A"]); // "이름"
Console.WriteLine(sheet[2, "B"]); // "30"
Console.WriteLine(sheet[3, "C"]); // "" (빈 기본값)
```

## 9. 정적 속성

정적 속성은 인스턴스가 아닌 클래스에 속합니다.

### 9.1 싱글턴 패턴과 정적 속성

```csharp
class AppLogger
{
    // private 정적 인스턴스
    private static AppLogger? _instance;
    private static readonly object _lock = new();

    private List<string> _logs = new();

    // private 생성자가 외부 인스턴스화를 방지
    private AppLogger() { }

    // 단일 인스턴스를 제공하는 정적 속성
    public static AppLogger Instance
    {
        get
        {
            if (_instance is null)
            {
                lock (_lock)
                {
                    _instance ??= new AppLogger();
                }
            }
            return _instance;
        }
    }

    public int LogCount => _logs.Count;

    public void Log(string message)
    {
        _logs.Add($"[{DateTime.Now:HH:mm:ss}] {message}");
    }

    public IReadOnlyList<string> GetLogs() => _logs.AsReadOnly();
}

AppLogger.Instance.Log("애플리케이션 시작");
AppLogger.Instance.Log("사용자 로그인");
Console.WriteLine(AppLogger.Instance.LogCount); // 2
```

### 9.2 정적 속성으로 설정 구성

```csharp
class AppConfig
{
    public static string AppName { get; set; } = "MyApp";
    public static string Version { get; } = "2.0.0";
    public static bool IsDebug { get; set; } = false;
    public static int MaxRetries { get; set; } = 3;

    // 계산된 정적 속성
    public static string FullVersion => $"{AppName} v{Version}" + (IsDebug ? " (디버그)" : "");
}

AppConfig.IsDebug = true;
Console.WriteLine(AppConfig.FullVersion); // "MyApp v2.0.0 (디버그)"
```

## 10. 필수 멤버

C# 11에서 호출자가 초기화 중에 특정 속성을 반드시 설정하도록 강제하는 `required` 한정자가 도입되었습니다.

### 10.1 필수 속성

```csharp
class Employee
{
    public required string Name { get; set; }
    public required string Department { get; set; }
    public required string EmployeeId { get; init; }

    // 기본값이 있는 선택적 속성
    public string Title { get; set; } = "직원";
    public DateTime HireDate { get; init; } = DateTime.Now;
}

// 모든 required 속성을 반드시 설정해야 함
Employee emp = new Employee
{
    Name = "Alice Johnson",
    Department = "엔지니어링",
    EmployeeId = "EMP-001"
};

// 선택적 속성은 설정하거나 기본값 유지 가능
Employee emp2 = new Employee
{
    Name = "Bob Smith",
    Department = "마케팅",
    EmployeeId = "EMP-002",
    Title = "매니저"
};

// 이것은 컴파일되지 않음 (required 멤버 누락):
// Employee bad = new Employee { Name = "Charlie" };
```

### 10.2 생성자와 함께 Required

```csharp
class ApiClient
{
    public required string BaseUrl { get; init; }
    public required string ApiKey { get; init; }
    public int TimeoutSeconds { get; init; } = 30;
    public bool EnableLogging { get; init; } = false;

    // SetsRequiredMembers 특성은 생성자가 모든 required 멤버를 설정함을 나타냄
    [System.Diagnostics.CodeAnalysis.SetsRequiredMembers]
    public ApiClient(string baseUrl, string apiKey)
    {
        BaseUrl = baseUrl;
        ApiKey = apiKey;
    }

    // 매개변수 없는 생성자는 여전히 초기화자로 멤버를 설정해야 함
    public ApiClient() { }

    public override string ToString()
        => $"ApiClient({BaseUrl}, timeout={TimeoutSeconds}초, log={EnableLogging})";
}

// 생성자 사용 (required 멤버를 다시 설정할 필요 없음)
ApiClient client1 = new ApiClient("https://api.example.com", "key-123");

// 객체 초기화자 사용 (required 멤버를 반드시 설정)
ApiClient client2 = new ApiClient
{
    BaseUrl = "https://api.example.com",
    ApiKey = "key-456",
    TimeoutSeconds = 60,
    EnableLogging = true
};
```

### 10.3 계층 구조에서의 Required

```csharp
class BaseEntity
{
    public required int Id { get; init; }
    public required DateTime CreatedAt { get; init; }
}

class Customer : BaseEntity
{
    public required string Name { get; set; }
    public required string Email { get; set; }
    public string? Phone { get; set; }
}

// 기본 클래스와 파생 클래스 모두의 required 멤버를 설정해야 함
Customer c = new Customer
{
    Id = 1,
    CreatedAt = DateTime.Now,
    Name = "Alice",
    Email = "alice@example.com",
    Phone = "555-0123" // 선택적
};
```

## 11. 연습 문제

1. **온도 클래스**: `Celsius` 속성을 가진 `Temperature` 클래스를 만드세요. 내부적으로 섭씨에서/로 변환하는 `get`과 `set` 접근자를 모두 가진 계산된 속성 `Fahrenheit`와 `Kelvin`을 추가하세요. 절대 영도(-273.15 C) 아래로 온도를 설정하지 못하도록 유효성 검사를 추가하세요.

2. **SafeArray 인덱서**: 배열을 래핑하고 인덱서를 제공하는 제네릭 클래스 `SafeArray<T>`를 만드세요. 범위를 벗어난 인덱스로 읽기 접근하면 예외를 던지지 않고 `default(T)`를 반환하세요. 범위를 벗어난 인덱스로 쓰기 시 내부 배열을 자동으로 크기 조정하세요. `Length` 속성을 추가하세요.

3. **유효성 검사가 있는 Config 빌더**: `Host`(비어 있으면 안 됨)와 `Port`(1-65535)에 대해 required와 init 전용 속성을 사용하는 `ServerConfig` 클래스를 설계하세요. 선택적 속성 `UseSsl`(기본 false), `MaxConnections`(기본 100, 0보다 커야 함), `Timeout`(기본 30, 0보다 커야 함)을 추가하세요. 잘못된 입력 시 예외를 던지는 속성 설정자를 사용하여 제약 조건을 검증하세요.

4. **행렬 클래스**: 두 매개변수 인덱서 `this[int row, int col]`을 가진 `Matrix` 클래스를 만드세요. 2D 배열로부터 생성을 지원하세요. 계산된 속성 `Rows`, `Columns`, `IsSquare`를 추가하세요. 새 Matrix를 반환하는 `Transpose()` 메서드를 추가하세요.

5. **속성 변경 추적기**: 모든 set이 속성 이름, 이전 값, 새 값을 기록하는 내부 `List<string>`에 변경 사항을 로깅하는 속성 `Name`, `Value`, `Description`을 가진 클래스 `TrackedObject`를 만드세요. 리스트를 반환하는 읽기 전용 속성 `ChangeHistory`를 추가하세요. 이를 사용하여 속성 설정자가 감사 추적을 가능하게 하는 방법을 시연하세요.
