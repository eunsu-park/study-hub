# 클래스와 객체

**이전**: [컬렉션](./08_Collections.md) | **다음**: [속성과 인덱서](./10_Properties_and_Indexers.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 필드와 적절한 접근 한정자로 클래스 선언하기
2. 기본, 매개변수화, 체이닝 생성자를 포함한 생성자 작성하기
3. 간결한 객체 생성을 위한 객체 초기화자 사용하기
4. 정적 멤버와 정적 클래스 정의 및 사용하기
5. `this` 키워드의 목적과 사용법 이해하기
6. 결정적 리소스 정리를 위한 `IDisposable` 구현하기
7. 부분(Partial) 클래스와 중첩(Nested) 클래스 다루기
8. 참조 동등성과 값 동등성 구분하기

---

클래스는 C#에서 객체 지향 프로그래밍의 핵심 구성 요소입니다. 클래스는 데이터(필드와 속성)와 동작(메서드)을 캡슐화하는 객체의 청사진을 정의합니다. 클래스를 효과적으로 설계하고 사용하는 방법을 이해하는 것은 잘 구조화된 C# 애플리케이션을 작성하는 데 핵심적입니다.

## 1. 클래스 선언

클래스는 관련된 데이터와 동작을 단일 단위로 그룹화합니다.

### 1.1 기본 클래스 구조

```csharp
class Dog
{
    // 필드 (데이터)
    public string Name;
    public string Breed;
    public int Age;

    // 메서드 (동작)
    public string Bark()
    {
        return $"{Name}가 말합니다: 멍!";
    }

    public string GetInfo()
    {
        return $"{Name}은(는) {Age}살 {Breed}입니다.";
    }
}

// 객체 생성 (클래스의 인스턴스)
Dog myDog = new Dog();
myDog.Name = "Rex";
myDog.Breed = "저먼 셰퍼드";
myDog.Age = 5;

Console.WriteLine(myDog.Bark());    // "Rex가 말합니다: 멍!"
Console.WriteLine(myDog.GetInfo()); // "Rex은(는) 5살 저먼 셰퍼드입니다."
```

### 1.2 클래스는 참조 타입

하나의 클래스 변수를 다른 변수에 대입하면, 두 변수 모두 메모리의 같은 객체를 참조합니다:

```csharp
class Counter
{
    public int Value;
}

Counter a = new Counter { Value = 10 };
Counter b = a; // b는 같은 객체를 참조

b.Value = 99;
Console.WriteLine(a.Value); // 99 (같은 객체)

// null은 참조 타입의 유효한 값
Counter? c = null;
// c.Value는 NullReferenceException을 던짐
```

## 2. 필드와 접근 한정자

접근 한정자(Access Modifier)는 클래스 멤버의 가시성을 제어합니다.

### 2.1 접근 한정자 요약

| 한정자 | 접근 범위 |
|--------|---------|
| `public` | 어디서든 접근 가능 |
| `private` | 선언한 클래스 내에서만 접근 가능 |
| `protected` | 클래스 및 하위 클래스 내에서 접근 가능 |
| `internal` | 같은 어셈블리(프로젝트) 내에서 접근 가능 |
| `protected internal` | 같은 어셈블리 내에서 또는 하위 클래스에서 접근 가능 |
| `private protected` | 같은 어셈블리 내의 클래스 또는 하위 클래스에서 접근 가능 |

### 2.2 필드 선언

```csharp
class BankAccount
{
    // private 필드 (구현 세부 사항)
    private string _accountNumber;
    private decimal _balance;
    private static int _nextId = 1000;

    // public 필드 (일반적으로 권장하지 않음; 속성 선호)
    public string OwnerName;

    // readonly 필드 (생성자나 선언에서만 설정 가능)
    private readonly DateTime _createdAt;

    // 상수 필드 (컴파일 시간 상수, 암시적으로 static)
    public const decimal MinimumBalance = 100.00m;

    // static readonly (런타임 상수)
    public static readonly string BankName = "국민은행";

    public BankAccount(string owner, string accountNumber)
    {
        OwnerName = owner;
        _accountNumber = accountNumber;
        _balance = 0;
        _createdAt = DateTime.Now;
        _nextId++;
    }

    public decimal GetBalance() => _balance;

    public void Deposit(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("금액은 양수여야 합니다.");
        _balance += amount;
    }

    public void Withdraw(decimal amount)
    {
        if (amount <= 0)
            throw new ArgumentException("금액은 양수여야 합니다.");
        if (_balance - amount < MinimumBalance)
            throw new InvalidOperationException("잔액이 부족합니다.");
        _balance -= amount;
    }
}
```

### 2.3 명명 규칙

```csharp
class StyleExample
{
    // private 필드: _camelCase (밑줄 접두사)
    private int _count;
    private string _name;

    // public 필드 (드묾): PascalCase
    public int Id;

    // 상수: PascalCase
    public const int MaxRetries = 3;

    // 메서드: PascalCase
    public void DoSomething() { }

    // 매개변수와 지역 변수: camelCase
    public void Process(int itemCount)
    {
        int localVar = itemCount * 2;
    }
}
```

## 3. 생성자

생성자(Constructor)는 `new` 키워드로 객체를 생성할 때 객체를 초기화합니다.

### 3.1 기본 생성자

어떤 생성자도 정의하지 않으면, C#은 모든 필드를 기본값으로 설정하는 기본 매개변수 없는 생성자를 제공합니다:

```csharp
class SimpleClass
{
    public int Number;     // 기본값: 0
    public string? Text;   // 기본값: null
    public bool Flag;      // 기본값: false
}

SimpleClass obj = new SimpleClass();
Console.WriteLine(obj.Number); // 0
Console.WriteLine(obj.Text);   // null (빈 출력)
Console.WriteLine(obj.Flag);   // false
```

### 3.2 매개변수화 생성자

```csharp
class Person
{
    public string Name;
    public int Age;
    public string Email;

    // 매개변수화 생성자
    public Person(string name, int age, string email)
    {
        Name = name;
        Age = age;
        Email = email;
    }

    public override string ToString()
        => $"Person({Name}, {Age}, {Email})";
}

// 정의된 생성자를 사용해야 함
Person alice = new Person("Alice", 30, "alice@example.com");

// 이것은 컴파일되지 않음 (매개변수 없는 생성자가 정의되지 않음):
// Person bob = new Person();
```

### 3.3 생성자 오버로딩

```csharp
class Product
{
    public string Name;
    public double Price;
    public string Category;

    // 전체 생성자
    public Product(string name, double price, string category)
    {
        Name = name;
        Price = price;
        Category = category;
    }

    // 기본 카테고리를 가진 부분 생성자
    public Product(string name, double price)
    {
        Name = name;
        Price = price;
        Category = "일반";
    }

    // 최소 생성자
    public Product(string name)
    {
        Name = name;
        Price = 0.0;
        Category = "미정";
    }
}

Product p1 = new Product("노트북", 999.99, "전자제품");
Product p2 = new Product("위젯", 4.99);
Product p3 = new Product("미스터리");
```

### 3.4 :this()를 사용한 생성자 체이닝

생성자 체이닝은 초기화 로직의 중복을 방지합니다:

```csharp
class Employee
{
    public string Name;
    public string Department;
    public double Salary;
    public DateTime HireDate;

    // 주 생성자 (모든 매개변수)
    public Employee(string name, string department, double salary, DateTime hireDate)
    {
        Name = name;
        Department = department;
        Salary = salary;
        HireDate = hireDate;
    }

    // 주 생성자에 체이닝: 기본 입사일을 오늘로
    public Employee(string name, string department, double salary)
        : this(name, department, salary, DateTime.Now)
    {
    }

    // 추가 체이닝: 기본 급여
    public Employee(string name, string department)
        : this(name, department, 50000.0)
    {
    }

    // 추가 체이닝: 기본 부서
    public Employee(string name)
        : this(name, "미배정")
    {
    }

    public override string ToString()
        => $"{Name} ({Department}) - ${Salary:N0}, 입사일 {HireDate:d}";
}

Employee e1 = new Employee("Alice", "엔지니어링", 120000, new DateTime(2020, 3, 15));
Employee e2 = new Employee("Bob", "마케팅", 85000);
Employee e3 = new Employee("Charlie", "영업");
Employee e4 = new Employee("Diana");
```

### 3.5 기본 생성자 (C# 12)

C# 12에서는 클래스에 직접 생성자 매개변수를 선언할 수 있습니다:

```csharp
class Point(double x, double y)
{
    // x와 y는 클래스 전체에서 매개변수로 사용 가능
    public double X => x;
    public double Y => y;

    public double DistanceFromOrigin()
        => Math.Sqrt(x * x + y * y);

    public override string ToString() => $"({x}, {y})";
}

Point p = new Point(3, 4);
Console.WriteLine(p.DistanceFromOrigin()); // 5
```

## 4. 객체 초기화

### 4.1 객체 초기화자

객체 초기화자를 사용하면 일치하는 생성자 없이도 생성 시에 public 필드와 속성을 설정할 수 있습니다:

```csharp
class Config
{
    public string Host = "localhost";
    public int Port = 8080;
    public bool UseSsl = false;
    public string? ApiKey;
    public int TimeoutSeconds = 30;
}

// 재정의하고 싶은 필드만 설정
Config config = new Config
{
    Host = "api.example.com",
    Port = 443,
    UseSsl = true,
    ApiKey = "secret-key-123"
    // TimeoutSeconds는 기본값 30 유지
};
```

### 4.2 생성자와 초기화자 결합

```csharp
class HttpRequest
{
    public string Url;
    public string Method;
    public Dictionary<string, string> Headers;
    public string? Body;
    public int TimeoutMs;

    public HttpRequest(string url, string method = "GET")
    {
        Url = url;
        Method = method;
        Headers = new Dictionary<string, string>();
        TimeoutMs = 5000;
    }
}

// 생성자 + 객체 초기화자
HttpRequest request = new HttpRequest("https://api.example.com/data")
{
    Method = "POST",
    Body = "{\"key\": \"value\"}",
    TimeoutMs = 10000,
    Headers = { ["Content-Type"] = "application/json", ["Authorization"] = "Bearer token123" }
};
```

### 4.3 필수 멤버 (C# 11)

`required` 키워드는 호출자가 특정 멤버를 설정하도록 강제합니다:

```csharp
class User
{
    public required string Username;
    public required string Email;
    public string DisplayName = "";
    public DateTime CreatedAt = DateTime.Now;
}

// 초기화자에서 required 멤버를 반드시 설정해야 함
User user = new User
{
    Username = "alice",
    Email = "alice@example.com"
    // DisplayName과 CreatedAt은 선택적
};

// 이것은 컴파일되지 않음:
// User bad = new User(); // required 멤버 누락
```

## 5. 정적 멤버와 정적 클래스

### 5.1 정적 필드와 메서드

정적 멤버는 인스턴스가 아닌 클래스 자체에 속합니다:

```csharp
class MathHelper
{
    // 정적 필드
    public static readonly double Pi = 3.14159265358979;
    private static int _callCount = 0;

    // 정적 메서드
    public static double CircleArea(double radius)
    {
        _callCount++;
        return Pi * radius * radius;
    }

    public static double CircleCircumference(double radius)
    {
        _callCount++;
        return 2 * Pi * radius;
    }

    public static int GetCallCount() => _callCount;
}

// 인스턴스가 아닌 클래스에서 호출
double area = MathHelper.CircleArea(5.0); // 78.54...
double circ = MathHelper.CircleCircumference(5.0);
Console.WriteLine(MathHelper.GetCallCount()); // 2
```

### 5.2 정적 생성자

정적 생성자는 클래스가 처음 사용되기 전에 한 번 실행됩니다:

```csharp
class AppConfig
{
    public static string Environment;
    public static string Version;

    // 정적 생성자: 접근 한정자 없음, 매개변수 없음
    static AppConfig()
    {
        // 환경 또는 설정 파일에서 초기화
        Environment = System.Environment.GetEnvironmentVariable("APP_ENV") ?? "development";
        Version = "1.0.0";
        Console.WriteLine("AppConfig가 초기화되었습니다.");
    }
}

// 정적 생성자는 첫 접근 시 자동으로 실행
Console.WriteLine(AppConfig.Environment); // "development"
```

### 5.3 정적 클래스

정적 클래스는 인스턴스화할 수 없으며 정적 멤버만 포함할 수 있습니다:

```csharp
static class StringExtensions
{
    public static string Truncate(string input, int maxLength)
    {
        if (string.IsNullOrEmpty(input) || input.Length <= maxLength)
            return input;
        return input[..maxLength] + "...";
    }

    public static string Repeat(string input, int count)
    {
        return string.Concat(Enumerable.Repeat(input, count));
    }

    public static bool IsNumeric(string input)
    {
        return double.TryParse(input, out _);
    }
}

string truncated = StringExtensions.Truncate("Hello, World!", 5); // "Hello..."
string repeated = StringExtensions.Repeat("Ha", 3);               // "HaHaHa"
bool isNum = StringExtensions.IsNumeric("42.5");                   // true
```

## 6. this 키워드

`this` 키워드는 클래스의 현재 인스턴스를 참조합니다.

### 6.1 필드와 매개변수 구분

```csharp
class Rectangle
{
    private double _width;
    private double _height;

    // 매개변수 이름이 필드 이름과 같을 때 'this'로 구분
    // (_ 접두사 규칙이 대부분 이 필요성을 없앰)
    public Rectangle(double width, double height)
    {
        this._width = width;   // 명시적이지만, _ 접두사가 이미 명확하게 함
        this._height = height;
    }

    public double Area() => _width * _height;
}

// 더 일반적인 시나리오: 매개변수 이름이 필드를 가릴 때
class Circle
{
    public double Radius;

    public Circle(double Radius)
    {
        this.Radius = Radius; // 'this.Radius'는 필드, 'Radius'는 매개변수
    }
}
```

### 6.2 플루언트 API를 위한 this 반환

```csharp
class QueryBuilder
{
    private string _table = "";
    private string _where = "";
    private string _orderBy = "";
    private int _limit = 0;

    public QueryBuilder From(string table)
    {
        _table = table;
        return this; // 체이닝 활성화
    }

    public QueryBuilder Where(string condition)
    {
        _where = condition;
        return this;
    }

    public QueryBuilder OrderBy(string column)
    {
        _orderBy = column;
        return this;
    }

    public QueryBuilder Limit(int count)
    {
        _limit = count;
        return this;
    }

    public string Build()
    {
        string query = $"SELECT * FROM {_table}";
        if (!string.IsNullOrEmpty(_where)) query += $" WHERE {_where}";
        if (!string.IsNullOrEmpty(_orderBy)) query += $" ORDER BY {_orderBy}";
        if (_limit > 0) query += $" LIMIT {_limit}";
        return query;
    }
}

// 플루언트 메서드 체이닝
string sql = new QueryBuilder()
    .From("users")
    .Where("age > 18")
    .OrderBy("name")
    .Limit(10)
    .Build();
// "SELECT * FROM users WHERE age > 18 ORDER BY name LIMIT 10"
```

### 6.3 다른 메서드에 this 전달

```csharp
class Node
{
    public string Name;
    public Node? Parent;

    public Node(string name)
    {
        Name = name;
    }

    public Node AddChild(string childName)
    {
        Node child = new Node(childName);
        child.Parent = this; // 현재 노드를 부모로 전달
        return child;
    }
}

Node root = new Node("루트");
Node child = root.AddChild("자식1");
Console.WriteLine(child.Parent?.Name); // "루트"
```

## 7. 종료자와 IDisposable

### 7.1 종료자

종료자(Finalizer, 소멸자)는 가비지 컬렉터가 객체를 회수할 때 실행됩니다. 비관리 리소스 정리에 사용되지만, 직접 필요한 경우는 드뭅니다:

```csharp
class ResourceHolder
{
    private IntPtr _nativeHandle;

    public ResourceHolder()
    {
        _nativeHandle = AllocateNativeResource();
        Console.WriteLine("리소스가 할당되었습니다.");
    }

    // 종료자 (GC에 의해 호출, 비결정적 타이밍)
    ~ResourceHolder()
    {
        FreeNativeResource(_nativeHandle);
        Console.WriteLine("종료자에 의해 리소스가 해제되었습니다.");
    }

    private static IntPtr AllocateNativeResource() => IntPtr.Zero; // 플레이스홀더
    private static void FreeNativeResource(IntPtr handle) { }      // 플레이스홀더
}
```

### 7.2 IDisposable 패턴

결정적 정리를 위해 `IDisposable`을 구현하고 `using` 문을 사용합니다:

```csharp
class DatabaseConnection : IDisposable
{
    private bool _disposed = false;
    private string _connectionString;

    public DatabaseConnection(string connectionString)
    {
        _connectionString = connectionString;
        Console.WriteLine($"연결됨: {connectionString}");
    }

    public void ExecuteQuery(string sql)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Console.WriteLine($"실행: {sql}");
    }

    // IDisposable 구현
    public void Dispose()
    {
        if (!_disposed)
        {
            // 관리 리소스 해제
            Console.WriteLine("연결이 닫혔습니다.");
            _disposed = true;
        }
        GC.SuppressFinalize(this); // Dispose가 호출되면 종료자 불필요
    }

    ~DatabaseConnection()
    {
        Dispose();
    }
}

// using 문은 예외가 발생해도 Dispose가 호출되도록 보장
using (var conn = new DatabaseConnection("Server=localhost;Database=test"))
{
    conn.ExecuteQuery("SELECT * FROM users");
} // 여기서 Dispose()가 자동 호출

// using 선언 (C# 8+): 감싸는 범위 끝에서 해제
void ProcessData()
{
    using var conn = new DatabaseConnection("Server=localhost;Database=test");
    conn.ExecuteQuery("SELECT * FROM orders");
    // 메서드 종료 시 conn.Dispose() 호출
}
```

## 8. 부분 클래스

`partial` 키워드를 사용하면 클래스 정의를 여러 파일에 분할할 수 있습니다. 이는 코드 생성기(예: WinForms 디자이너, 소스 생성기)에서 일반적으로 사용됩니다.

### 8.1 클래스 분할

```csharp
// 파일: User.cs
partial class User
{
    public string Name;
    public string Email;

    public User(string name, string email)
    {
        Name = name;
        Email = email;
    }
}

// 파일: User.Validation.cs
partial class User
{
    public bool IsValid()
    {
        return !string.IsNullOrWhiteSpace(Name)
            && !string.IsNullOrWhiteSpace(Email)
            && Email.Contains('@');
    }

    public List<string> GetValidationErrors()
    {
        List<string> errors = new();
        if (string.IsNullOrWhiteSpace(Name))
            errors.Add("이름은 필수입니다.");
        if (string.IsNullOrWhiteSpace(Email))
            errors.Add("이메일은 필수입니다.");
        else if (!Email.Contains('@'))
            errors.Add("이메일에 @가 포함되어야 합니다.");
        return errors;
    }
}

// 파일: User.Display.cs
partial class User
{
    public override string ToString()
        => $"{Name} <{Email}>";

    public string ToJson()
        => $"{{\"name\": \"{Name}\", \"email\": \"{Email}\"}}";
}

// 사용 (모든 부분이 하나의 클래스로 결합)
User user = new User("Alice", "alice@example.com");
Console.WriteLine(user.IsValid());   // true
Console.WriteLine(user.ToString());  // "Alice <alice@example.com>"
Console.WriteLine(user.ToJson());
```

### 8.2 부분 메서드

부분 메서드를 사용하면 클래스의 한 부분이 메서드 시그니처를 선언하고 다른 부분이 구현을 제공할 수 있습니다:

```csharp
// 생성된 코드
partial class Order
{
    public decimal Total;

    public void Process()
    {
        // 부분 메서드 호출 (구현되지 않으면 아무 동작 없음)
        OnProcessing();
        Console.WriteLine($"주문 처리 중: ${Total}");
        OnProcessed();
    }

    // 부분 메서드 선언
    partial void OnProcessing();
    partial void OnProcessed();
}

// 사용자 정의 코드
partial class Order
{
    partial void OnProcessing()
    {
        Console.WriteLine("주문을 처리하려고 합니다...");
    }

    partial void OnProcessed()
    {
        Console.WriteLine("주문이 성공적으로 처리되었습니다.");
    }
}
```

## 9. 중첩 클래스

클래스를 다른 클래스 안에 정의할 수 있습니다. 중첩 클래스는 외부 클래스의 private 멤버에 접근할 수 있습니다.

### 9.1 기본 중첩 클래스

```csharp
class LinkedList
{
    // 중첩 클래스: 외부에 노출되지 않는 구현 세부 사항
    private class Node
    {
        public int Value;
        public Node? Next;

        public Node(int value)
        {
            Value = value;
            Next = null;
        }
    }

    private Node? _head;
    private int _count;

    public void AddFirst(int value)
    {
        Node newNode = new Node(value);
        newNode.Next = _head;
        _head = newNode;
        _count++;
    }

    public void PrintAll()
    {
        Node? current = _head;
        while (current != null)
        {
            Console.Write($"{current.Value} -> ");
            current = current.Next;
        }
        Console.WriteLine("null");
    }

    public int Count => _count;
}

LinkedList list = new LinkedList();
list.AddFirst(3);
list.AddFirst(2);
list.AddFirst(1);
list.PrintAll(); // 1 -> 2 -> 3 -> null
```

### 9.2 공개 중첩 클래스 (빌더 패턴)

```csharp
class Pizza
{
    public string Size { get; }
    public string Crust { get; }
    public List<string> Toppings { get; }

    // private 생성자: Builder만 Pizza를 생성할 수 있음
    private Pizza(string size, string crust, List<string> toppings)
    {
        Size = size;
        Crust = crust;
        Toppings = toppings;
    }

    public override string ToString()
        => $"{Size} 피자, {Crust} 크러스트, 토핑: {string.Join(", ", Toppings)}";

    // 공개 중첩 빌더 클래스
    public class Builder
    {
        private string _size = "미디엄";
        private string _crust = "레귤러";
        private List<string> _toppings = new();

        public Builder SetSize(string size) { _size = size; return this; }
        public Builder SetCrust(string crust) { _crust = crust; return this; }
        public Builder AddTopping(string topping) { _toppings.Add(topping); return this; }

        // Build는 Pizza의 private 생성자에 접근
        public Pizza Build() => new Pizza(_size, _crust, new List<string>(_toppings));
    }
}

Pizza pizza = new Pizza.Builder()
    .SetSize("라지")
    .SetCrust("씬")
    .AddTopping("모짜렐라")
    .AddTopping("페퍼로니")
    .AddTopping("버섯")
    .Build();

Console.WriteLine(pizza);
// "라지 피자, 씬 크러스트, 토핑: 모짜렐라, 페퍼로니, 버섯"
```

## 10. 참조 동등성 vs 값 동등성

### 10.1 기본 참조 동등성

기본적으로 클래스의 `==` 연산자와 `Equals` 메서드는 두 변수가 같은 객체를 가리키는지(참조 동등성) 확인합니다:

```csharp
class Point
{
    public int X, Y;
    public Point(int x, int y) { X = x; Y = y; }
}

Point a = new Point(3, 4);
Point b = new Point(3, 4);
Point c = a;

Console.WriteLine(a == b);          // false (다른 객체)
Console.WriteLine(a == c);          // true (같은 객체)
Console.WriteLine(a.Equals(b));     // false (기본: 참조 동등성)
Console.WriteLine(ReferenceEquals(a, b)); // false
Console.WriteLine(ReferenceEquals(a, c)); // true
```

### 10.2 값 동등성 구현

값 기반 동등성을 제공하려면 `Equals`, `GetHashCode`, 선택적으로 `==`/`!=`를 재정의합니다:

```csharp
class Coordinate : IEquatable<Coordinate>
{
    public double Latitude { get; }
    public double Longitude { get; }

    public Coordinate(double latitude, double longitude)
    {
        Latitude = latitude;
        Longitude = longitude;
    }

    // IEquatable<Coordinate> 구현
    public bool Equals(Coordinate? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        return Latitude == other.Latitude && Longitude == other.Longitude;
    }

    // Object.Equals 재정의
    public override bool Equals(object? obj)
        => Equals(obj as Coordinate);

    // Equals를 재정의할 때 GetHashCode도 반드시 재정의
    public override int GetHashCode()
        => HashCode.Combine(Latitude, Longitude);

    // 연산자 오버로드
    public static bool operator ==(Coordinate? left, Coordinate? right)
        => left is null ? right is null : left.Equals(right);

    public static bool operator !=(Coordinate? left, Coordinate? right)
        => !(left == right);

    public override string ToString()
        => $"({Latitude}, {Longitude})";
}

Coordinate nyc1 = new Coordinate(40.7128, -74.0060);
Coordinate nyc2 = new Coordinate(40.7128, -74.0060);
Coordinate la = new Coordinate(34.0522, -118.2437);

Console.WriteLine(nyc1 == nyc2);         // true (값 동등성)
Console.WriteLine(nyc1 == la);           // false
Console.WriteLine(nyc1.GetHashCode() == nyc2.GetHashCode()); // true

// 컬렉션에서 올바르게 작동
HashSet<Coordinate> visited = new() { nyc1 };
Console.WriteLine(visited.Contains(nyc2)); // true (같은 값)
```

### 10.3 레코드 클래스 (값 동등성의 지름길)

레코드 클래스는 보일러플레이트 없이 내장된 값 동등성을 제공합니다:

```csharp
record class Coordinate(double Latitude, double Longitude);

Coordinate a = new Coordinate(40.7128, -74.0060);
Coordinate b = new Coordinate(40.7128, -74.0060);

Console.WriteLine(a == b);      // true (값 동등성, 내장)
Console.WriteLine(a.GetHashCode() == b.GetHashCode()); // true

// with 표현식으로 비파괴적 변경
Coordinate moved = a with { Latitude = 41.0 };
Console.WriteLine(moved); // Coordinate { Latitude = 41, Longitude = -74.006 }
```

## 11. 연습 문제

1. **도서관 책 클래스**: `Title`, `Author`, `ISBN`, `IsCheckedOut` 필드를 가진 `Book` 클래스를 만드세요. 생성자(체이닝을 포함한 매개변수화 생성자), `CheckOut()` 메서드, `Return()` 메서드, `ToString()` 재정의를 추가하세요. ISBN은 생성 후 readonly여야 합니다.

2. **정적 추적이 있는 카운터**: 각 인스턴스가 자체 `Count` 값과 `Increment()` 메서드를 가진 `Counter` 클래스를 만드세요. 모든 Counter 인스턴스에서 수행된 총 증가 횟수를 추적하는 정적 필드를 사용하세요. 정적 메서드 `GetGlobalCount()`를 추가하세요.

3. **플루언트 이메일 빌더**: 중첩 `Builder` 클래스가 있는 `EmailMessage` 클래스를 설계하세요. 빌더는 체이닝 메서드를 지원해야 합니다: `From(string)`, `To(string)`, `Subject(string)`, `Body(string)`, `AddAttachment(string)`, `Build()`. 빌드 전에 From, To, Subject가 설정되었는지 검증하세요.

4. **Disposable 임시 파일**: `IDisposable`을 구현하는 `TempFile` 클래스를 만드세요. 생성자는 임시 파일을 생성하고(`Path.GetTempFileName()` 사용), `Dispose`는 파일을 삭제합니다. `Write(string content)`와 `ReadAll()` 메서드를 추가하세요. `using` 문으로 사용법을 시연하세요.

5. **카드 놀이의 값 동등성**: `Suit`(열거형: Hearts, Diamonds, Clubs, Spades)와 `Rank`(열거형: Ace부터 King까지)를 가진 `Card` 클래스를 만드세요. `IEquatable<Card>`를 구현하고 `Equals`, `GetHashCode`, `==`/`!=` 연산자를 재정의하세요. 같은 Suit와 Rank를 가진 두 카드가 동등하다고 간주되는지, `HashSet<Card>`에서 올바르게 작동하는지 확인하세요.
