# 상속 (Inheritance)

**이전**: [구조체와 열거형](./10_Structs_and_Enums.md) | **다음**: [인터페이스](./12_Interfaces.md)

---

## 학습 목표

이 레슨을 완료하면 다음을 할 수 있습니다:

1. 기본 클래스와 파생 클래스를 사용하여 클래스 계층 구조를 생성할 수 있다
2. `base` 키워드로 생성자를 연결할 수 있다
3. `virtual`과 `override`를 사용한 메서드 재정의와 `new`를 사용한 메서드 숨기기를 이해할 수 있다
4. 추상 클래스(abstract class)를 설계하고 추상 메서드를 구현할 수 있다
5. `sealed` 클래스와 메서드로 상속을 제한할 수 있다
6. `is`와 `as`를 사용하여 타입 검사 및 캐스팅을 수행할 수 있다
7. 업캐스팅(upcasting), 다운캐스팅(downcasting), 그리고 `protected` 접근 한정자를 이해할 수 있다
8. `Object` 클래스의 핵심 메서드를 재정의할 수 있다

---

상속(Inheritance)은 객체 지향 프로그래밍의 네 가지 기둥 중 하나입니다. 기존 클래스를 기반으로 새 클래스를 정의하여, 멤버와 동작을 상속받으면서 기능을 추가하거나 수정할 수 있습니다. C#에서 상속은 코드 재사용을 가능하게 하고, 자연스러운 계층 구조를 설정하며, 다형성(polymorphism)의 기반을 제공합니다. 이 레슨에서는 기본 클래스 파생부터 추상 클래스와 봉인 클래스(sealed class)까지 모든 것을 다룹니다.

## 1. 기본 클래스와 파생 클래스

C#에서 클래스는 콜론(`:`) 구문을 사용하여 다른 클래스를 상속할 수 있습니다. 상속되는 클래스를 **기본 클래스**(base class, 부모 클래스)라 하고, 상속받는 클래스를 **파생 클래스**(derived class, 자식 클래스)라 합니다.

### 1.1 기본 상속 구문

```csharp
// 기본 클래스
public class Animal
{
    public string Name { get; set; }
    public int Age { get; set; }

    public void Eat()
    {
        Console.WriteLine($"{Name} is eating.");
    }

    public void Sleep()
    {
        Console.WriteLine($"{Name} is sleeping.");
    }
}

// 파생 클래스
public class Dog : Animal
{
    public string Breed { get; set; }

    public void Bark()
    {
        Console.WriteLine($"{Name} says: Woof!");
    }
}
```

```csharp
Dog dog = new Dog();
dog.Name = "Rex";       // Animal에서 상속
dog.Age = 5;            // Animal에서 상속
dog.Breed = "Labrador"; // Dog에 정의됨

dog.Eat();   // 상속된 메서드: "Rex is eating."
dog.Sleep(); // 상속된 메서드: "Rex is sleeping."
dog.Bark();  // Dog 전용 메서드: "Rex says: Woof!"
```

### 1.2 단일 상속만 가능

C#은 클래스에 대해 단일 상속(single inheritance)만 지원합니다. 하나의 클래스는 하나의 기본 클래스만 상속할 수 있지만, 여러 인터페이스(interface)를 구현할 수 있습니다.

```csharp
// 유효: 단일 상속
public class GuideDog : Dog
{
    public string HandlerName { get; set; }

    public void Guide()
    {
        Console.WriteLine($"{Name} is guiding {HandlerName}.");
    }
}

// 유효하지 않음: C#에서 다중 상속은 허용되지 않음
// public class Hybrid : Dog, Cat { }  // 컴파일 오류
```

### 1.3 상속되는 것과 상속되지 않는 것

파생 클래스는 모든 public 및 protected 멤버를 상속합니다. private 멤버는 직접 접근할 수 없습니다(메모리에는 여전히 존재합니다). 생성자와 종료자(finalizer)는 상속되지 않지만, `base` 키워드를 사용하여 기본 클래스 생성자를 호출할 수 있습니다.

```csharp
public class Vehicle
{
    public string Make { get; set; }
    protected int year;          // 파생 클래스에서 접근 가능
    private string vin;          // 파생 클래스에서 접근 불가

    public Vehicle(string make, int year, string vin)
    {
        Make = make;
        this.year = year;
        this.vin = vin;
    }

    public void DisplayInfo()
    {
        Console.WriteLine($"{Make}, Year: {year}");
    }
}

public class Car : Vehicle
{
    public int Doors { get; set; }

    public Car(string make, int year, string vin, int doors)
        : base(make, year, vin)  // 기본 클래스 생성자 호출
    {
        Doors = doors;
    }

    public void ShowDetails()
    {
        Console.WriteLine($"{Make}, Year: {year}, Doors: {Doors}");
        // Console.WriteLine(vin); // 오류: 보호 수준으로 인해 'vin'에 접근할 수 없음
    }
}
```

## 2. `base`를 사용한 생성자 연결

생성자는 상속되지 않으므로, 기본 클래스에 매개변수 없는 생성자가 없는 경우 파생 클래스에서 명시적으로 기본 클래스 생성자를 호출해야 합니다.

### 2.1 기본 클래스 생성자 호출

```csharp
public class Person
{
    public string Name { get; set; }
    public int Age { get; set; }

    public Person(string name, int age)
    {
        Name = name;
        Age = age;
        Console.WriteLine("Person constructor called.");
    }
}

public class Student : Person
{
    public string School { get; set; }

    // Person에 매개변수 없는 생성자가 없으므로 base 생성자를 호출해야 함
    public Student(string name, int age, string school)
        : base(name, age)
    {
        School = school;
        Console.WriteLine("Student constructor called.");
    }
}
```

```csharp
Student s = new Student("Alice", 20, "MIT");
// 출력:
// Person constructor called.
// Student constructor called.

Console.WriteLine($"{s.Name}, {s.Age}, {s.School}");
// 출력: Alice, 20, MIT
```

### 2.2 다중 생성자 오버로드

파생 클래스는 여러 생성자를 가질 수 있으며, 각각 다른 기본 클래스 생성자나 `this`를 사용하여 같은 클래스의 다른 생성자에 연결할 수 있습니다.

```csharp
public class Employee : Person
{
    public string Company { get; set; }
    public decimal Salary { get; set; }

    // base로 생성자 연결
    public Employee(string name, int age, string company, decimal salary)
        : base(name, age)
    {
        Company = company;
        Salary = salary;
    }

    // 같은 클래스의 다른 생성자로 연결
    public Employee(string name, int age, string company)
        : this(name, age, company, 50000m)
    {
    }

    // 최소 생성자
    public Employee(string name)
        : base(name, 0)
    {
        Company = "Unknown";
        Salary = 0m;
    }
}
```

### 2.3 생성자 실행 순서

생성자는 계층 구조의 위에서 아래로 실행됩니다. 기본 클래스 생성자가 파생 클래스 생성자보다 먼저 실행됩니다.

```csharp
public class A
{
    public A() { Console.WriteLine("A constructor"); }
}

public class B : A
{
    public B() { Console.WriteLine("B constructor"); }
}

public class C : B
{
    public C() { Console.WriteLine("C constructor"); }
}

// new C() 출력:
// A constructor
// B constructor
// C constructor
```

## 3. 메서드 재정의: `virtual`, `override`, `new`

메서드 재정의(method overriding)는 파생 클래스가 기본 클래스에 정의된 메서드에 대해 특정 구현을 제공할 수 있게 합니다. C#은 `virtual`과 `override` 키워드를 사용하여 진정한 다형적 재정의를 수행합니다.

### 3.1 Virtual과 Override

기본 클래스 메서드에 `virtual`을 표시하면 파생 클래스에서 재정의할 수 있습니다. 파생 클래스에서 `override`를 사용하여 새로운 구현을 제공합니다.

```csharp
public class Shape
{
    public virtual double Area()
    {
        return 0;
    }

    public virtual string Describe()
    {
        return "I am a shape.";
    }
}

public class Circle : Shape
{
    public double Radius { get; set; }

    public Circle(double radius)
    {
        Radius = radius;
    }

    public override double Area()
    {
        return Math.PI * Radius * Radius;
    }

    public override string Describe()
    {
        return $"I am a circle with radius {Radius:F2}.";
    }
}
```

```csharp
Shape shape = new Circle(5);
Console.WriteLine(shape.Area());      // 78.54 (Circle의 override가 호출됨)
Console.WriteLine(shape.Describe());  // "I am a circle with radius 5.00."
```

### 3.2 `new`를 사용한 메서드 숨기기

파생 클래스에서 `override`를 사용하지 않고 기본 클래스 메서드와 같은 이름의 메서드를 정의하면, 기본 메서드를 **숨기는**(hiding) 것입니다. `new` 키워드를 명시적으로 사용하지 않으면 컴파일러가 경고를 발생시킵니다.

```csharp
public class BaseLogger
{
    public void Log(string message)
    {
        Console.WriteLine($"[Base] {message}");
    }
}

public class DerivedLogger : BaseLogger
{
    // BaseLogger.Log를 숨김 — 다형적이 아님
    public new void Log(string message)
    {
        Console.WriteLine($"[Derived] {message}");
    }
}
```

```csharp
DerivedLogger derived = new DerivedLogger();
derived.Log("Hello");            // [Derived] Hello

BaseLogger baseRef = derived;
baseRef.Log("Hello");            // [Base] Hello — 숨기기는 기본 버전이 호출됨을 의미
```

### 3.3 Override vs New: 핵심 차이점

```csharp
public class Animal
{
    public virtual void Speak() => Console.WriteLine("...");
}

public class Cat : Animal
{
    public override void Speak() => Console.WriteLine("Meow!");
}

public class SilentCat : Animal
{
    public new void Speak() => Console.WriteLine("(silent meow)");
}

Animal cat1 = new Cat();
Animal cat2 = new SilentCat();

cat1.Speak();  // "Meow!"          — override: 파생 버전이 호출됨
cat2.Speak();  // "..."            — new: 기본 참조를 통해 기본 버전이 호출됨
```

다형적 동작을 원할 때는 `override`를 사용하세요. `new`는 의도적으로 다형적 체인을 끊고 싶을 때만 사용하세요(실제로는 드문 경우입니다).

## 4. 추상 클래스와 추상 메서드

추상 클래스(abstract class)는 직접 인스턴스화할 수 없습니다. 계약을 정의하는 기본 클래스 역할을 하며, 파생 클래스가 모든 추상 멤버를 구현해야 합니다.

### 4.1 추상 클래스 선언

```csharp
public abstract class Vehicle
{
    public string Make { get; set; }
    public string Model { get; set; }

    // 추상 메서드: 구현 없음, 반드시 재정의해야 함
    public abstract void StartEngine();

    // 추상 속성
    public abstract int MaxSpeed { get; }

    // 구체적 메서드: 그대로 사용하거나 virtual이면 재정의 가능
    public void DisplayInfo()
    {
        Console.WriteLine($"{Make} {Model}, Max Speed: {MaxSpeed} km/h");
    }
}
```

### 4.2 추상 멤버 구현

```csharp
public class ElectricCar : Vehicle
{
    public int BatteryCapacity { get; set; }

    public override int MaxSpeed => 200;

    public override void StartEngine()
    {
        Console.WriteLine("Electric motor whirring...");
    }
}

public class GasCar : Vehicle
{
    public double EngineSize { get; set; }

    public override int MaxSpeed => 250;

    public override void StartEngine()
    {
        Console.WriteLine("Vroom vroom!");
    }
}
```

```csharp
// Vehicle v = new Vehicle();  // 오류: 추상 클래스를 인스턴스화할 수 없음

Vehicle car1 = new ElectricCar { Make = "Tesla", Model = "Model 3", BatteryCapacity = 75 };
Vehicle car2 = new GasCar { Make = "BMW", Model = "M3", EngineSize = 3.0 };

car1.StartEngine();   // "Electric motor whirring..."
car2.StartEngine();   // "Vroom vroom!"
car1.DisplayInfo();   // "Tesla Model 3, Max Speed: 200 km/h"
```

### 4.3 추상 클래스의 추상 멤버와 구체적 멤버

추상 클래스는 추상 멤버와 비추상 멤버를 혼합하여 포함할 수 있습니다. 이것이 (C# 8 이전의) 인터페이스와의 핵심 차이점입니다.

```csharp
public abstract class DatabaseConnection
{
    // 추상 멤버 — 파생 클래스가 반드시 구현해야 함
    public abstract string ConnectionString { get; }
    public abstract void Connect();
    public abstract void Disconnect();

    // 구체적 멤버 — 공유 동작
    public bool IsConnected { get; protected set; }

    public void ExecuteQuery(string query)
    {
        if (!IsConnected)
        {
            Console.WriteLine("Error: Not connected.");
            return;
        }
        Console.WriteLine($"Executing: {query}");
    }
}

public class SqlServerConnection : DatabaseConnection
{
    public override string ConnectionString => "Server=localhost;Database=mydb;";

    public override void Connect()
    {
        IsConnected = true;
        Console.WriteLine("Connected to SQL Server.");
    }

    public override void Disconnect()
    {
        IsConnected = false;
        Console.WriteLine("Disconnected from SQL Server.");
    }
}
```

## 5. 봉인 클래스와 봉인 메서드 (Sealed)

`sealed` 키워드는 클래스가 상속되거나 메서드가 더 이상 재정의되는 것을 방지합니다.

### 5.1 봉인 클래스

```csharp
public sealed class MathHelper
{
    public static double CircleArea(double radius) => Math.PI * radius * radius;
    public static double RectangleArea(double width, double height) => width * height;
}

// 오류: 봉인 타입 'MathHelper'에서 파생할 수 없음
// public class ExtendedMathHelper : MathHelper { }
```

봉인 클래스는 유틸리티 클래스, 보안에 민감한 클래스, 또는 하위 클래스 생성으로 동작이 변경되지 않도록 보장하려는 경우에 유용합니다. .NET의 `string` 타입은 sealed입니다.

### 5.2 봉인 메서드

개별 재정의된 메서드를 봉인하여 계층 구조에서 더 이상의 재정의를 방지할 수 있습니다.

```csharp
public class Animal
{
    public virtual void MakeSound()
    {
        Console.WriteLine("Some generic sound");
    }
}

public class Dog : Animal
{
    // 재정의 후 봉인: 추가 재정의 불가
    public sealed override void MakeSound()
    {
        Console.WriteLine("Bark!");
    }
}

public class GoldenRetriever : Dog
{
    // 오류: 봉인 메서드 'Dog.MakeSound()'를 재정의할 수 없음
    // public override void MakeSound() { }
}
```

### 5.3 Sealed를 사용해야 하는 경우

봉인은 다음 경우에 적절합니다:
- 클래스가 확장을 위해 설계되지 않은 경우(유틸리티 또는 헬퍼 클래스)
- 보안이나 정확성이 특정 동작이 변경되지 않는 것에 의존하는 경우
- 성능: JIT 컴파일러가 봉인 메서드 호출을 최적화할 수 있는 경우

```csharp
public abstract class PaymentProcessor
{
    public abstract decimal CalculateFee(decimal amount);

    // 템플릿 메서드 패턴: 변조 방지를 위해 봉인
    public sealed override string ToString()
    {
        return $"{GetType().Name} processor";
    }
}
```

## 6. `is`와 `as` 연산자

C#은 런타임에 안전한 타입 검사와 캐스팅을 위해 `is`와 `as` 연산자를 제공합니다.

### 6.1 `is` 연산자

`is` 연산자는 객체가 주어진 타입과 호환되는지 확인하고 `true` 또는 `false`를 반환합니다.

```csharp
public class Animal { }
public class Dog : Animal { }
public class Cat : Animal { }

Animal animal = new Dog();

if (animal is Dog)
{
    Console.WriteLine("It's a dog!");     // 이것이 출력됨
}

if (animal is Cat)
{
    Console.WriteLine("It's a cat!");     // 이것은 출력되지 않음
}

if (animal is Animal)
{
    Console.WriteLine("It's an animal!"); // 이것이 출력됨 (Dog은 Animal임)
}
```

### 6.2 `is`를 사용한 패턴 매칭 (C# 7+)

`is` 연산자는 변수를 선언하여 타입 검사와 캐스팅을 한 번에 수행할 수도 있습니다.

```csharp
Animal animal = new Dog { Name = "Rex" };

if (animal is Dog dog)
{
    // 'dog'은 이제 이 스코프에서 사용 가능한 Dog 변수
    Console.WriteLine($"Dog name: {dog.Name}");
}

// switch 문에서도 작동
void DescribeAnimal(Animal a)
{
    switch (a)
    {
        case Dog d:
            Console.WriteLine($"Dog: {d.Name}");
            break;
        case Cat c:
            Console.WriteLine($"Cat: {c.Name}");
            break;
        default:
            Console.WriteLine("Unknown animal");
            break;
    }
}
```

### 6.3 `as` 연산자

`as` 연산자는 캐스팅을 시도하고, 실패하면 예외를 던지는 대신 `null`을 반환합니다.

```csharp
Animal animal = new Dog();

Dog dog = animal as Dog;
if (dog != null)
{
    Console.WriteLine("Successfully cast to Dog.");
}

Cat cat = animal as Cat;  // null을 반환, 예외 없음
if (cat == null)
{
    Console.WriteLine("Cannot cast to Cat.");
}
```

### 6.4 `as` vs 직접 캐스팅

```csharp
Animal animal = new Dog();

// 직접 캐스팅: 실패하면 InvalidCastException 발생
try
{
    Cat cat = (Cat)animal;  // 예외 발생!
}
catch (InvalidCastException ex)
{
    Console.WriteLine($"Cast failed: {ex.Message}");
}

// 'as' 연산자: 실패하면 null 반환 (예외 없음)
Cat safeCat = animal as Cat;  // null, 예외 없음

// 패턴 매칭이 포함된 'is': 가장 깔끔한 접근법
if (animal is Cat patternCat)
{
    // patternCat을 안전하게 사용
}
```

## 7. 업캐스팅과 다운캐스팅

### 7.1 업캐스팅 (암시적)

업캐스팅(upcasting)은 파생 타입을 기본 타입으로 변환하는 것입니다. 항상 안전하며 암시적으로 발생합니다.

```csharp
Dog dog = new Dog { Name = "Buddy" };
Animal animal = dog;  // 암시적 업캐스트: Dog -> Animal

// 객체는 여전히 Dog이지만, 참조 타입은 Animal
Console.WriteLine(animal.Name);   // 작동: Name은 Animal에 정의됨
// animal.Bark();                  // 오류: Bark은 Animal에 정의되지 않음
```

### 7.2 다운캐스팅 (명시적)

다운캐스팅(downcasting)은 기본 타입 참조를 다시 파생 타입으로 변환합니다. 명시적 캐스팅이 필요하며 런타임에 실패할 수 있습니다.

```csharp
Animal animal = new Dog { Name = "Buddy" };

// 'is' 검사를 사용한 안전한 다운캐스트
if (animal is Dog dog)
{
    dog.Bark();  // 이제 Dog 전용 메서드를 호출할 수 있음
}

// 명시적 캐스팅으로 대체 (검사 없이는 위험)
Dog dog2 = (Dog)animal;  // 객체가 실제로 Dog이므로 작동
dog2.Bark();
```

### 7.3 다형성의 실제 동작

업캐스팅은 다형성(polymorphism)의 기반입니다. 서로 다른 파생 타입을 기본 타입 컬렉션에 저장하고 재정의된 메서드를 호출할 수 있습니다.

```csharp
List<Animal> animals = new List<Animal>
{
    new Dog { Name = "Rex" },
    new Cat { Name = "Whiskers" },
    new Dog { Name = "Buddy" }
};

foreach (Animal a in animals)
{
    a.Speak();  // 각 실제 타입에 맞는 올바른 override를 호출

    if (a is Dog d)
    {
        d.Bark();  // Dog 전용 동작
    }
}
```

## 8. `protected` 접근 한정자

`protected` 한정자는 멤버를 클래스 자체와 모든 파생 클래스에서 접근 가능하게 하지만, 외부 코드에서는 접근할 수 없게 합니다.

### 8.1 Protected 멤버

```csharp
public class BankAccount
{
    public string Owner { get; set; }
    protected decimal balance;  // 파생 클래스에서 접근 가능

    public BankAccount(string owner, decimal initialBalance)
    {
        Owner = owner;
        balance = initialBalance;
    }

    public decimal GetBalance() => balance;
}

public class SavingsAccount : BankAccount
{
    private decimal interestRate;

    public SavingsAccount(string owner, decimal balance, decimal rate)
        : base(owner, balance)
    {
        interestRate = rate;
    }

    public void ApplyInterest()
    {
        // protected이므로 'balance'에 접근 가능
        balance += balance * interestRate;
        Console.WriteLine($"New balance after interest: {balance:C}");
    }
}
```

```csharp
SavingsAccount sa = new SavingsAccount("Alice", 1000m, 0.05m);
sa.ApplyInterest();  // "New balance after interest: $1,050.00"

// sa.balance;  // 오류: 보호 수준으로 인해 'balance'에 접근할 수 없음
Console.WriteLine(sa.GetBalance());  // public 메서드를 대신 사용
```

### 8.2 Protected Internal과 Private Protected

C#은 추가적인 접근 한정자 조합을 제공합니다:

```csharp
public class MyBase
{
    // protected internal: 파생 클래스 또는 같은 어셈블리의 모든 코드에서 접근 가능
    protected internal int value1;

    // private protected: 같은 어셈블리의 파생 클래스에서만 접근 가능
    private protected int value2;
}
```

| 한정자 | 같은 클래스 | 파생 (같은 어셈블리) | 파생 (다른 어셈블리) | 같은 어셈블리 | 외부 |
|---|---|---|---|---|---|
| `public` | 예 | 예 | 예 | 예 | 예 |
| `protected` | 예 | 예 | 예 | 아니오 | 아니오 |
| `internal` | 예 | 예 | 아니오 | 예 | 아니오 |
| `protected internal` | 예 | 예 | 예 | 예 | 아니오 |
| `private protected` | 예 | 예 | 아니오 | 아니오 | 아니오 |
| `private` | 예 | 아니오 | 아니오 | 아니오 | 아니오 |

## 9. `Object` 클래스

C#의 모든 클래스는 암시적으로 `System.Object`를 상속합니다. 따라서 모든 객체는 동작을 커스터마이징하기 위해 재정의할 수 있는 공통 메서드 집합에 접근할 수 있습니다.

### 9.1 `ToString()` 재정의

```csharp
public class Product
{
    public string Name { get; set; }
    public decimal Price { get; set; }

    public override string ToString()
    {
        return $"{Name} (${Price:F2})";
    }
}

Product p = new Product { Name = "Laptop", Price = 999.99m };
Console.WriteLine(p);              // "Laptop ($999.99)"
Console.WriteLine(p.ToString());   // 같은 결과
```

### 9.2 `Equals()`와 `GetHashCode()` 재정의

`Equals`를 재정의할 때는 일관성을 유지하기 위해 `GetHashCode`도 함께 재정의해야 합니다(동일한 객체는 같은 해시 코드를 생성해야 합니다).

```csharp
public class Point
{
    public int X { get; set; }
    public int Y { get; set; }

    public override bool Equals(object obj)
    {
        if (obj is Point other)
        {
            return X == other.X && Y == other.Y;
        }
        return false;
    }

    public override int GetHashCode()
    {
        return HashCode.Combine(X, Y);
    }

    public override string ToString()
    {
        return $"({X}, {Y})";
    }
}
```

```csharp
Point a = new Point { X = 3, Y = 4 };
Point b = new Point { X = 3, Y = 4 };
Point c = new Point { X = 1, Y = 2 };

Console.WriteLine(a.Equals(b));    // True (같은 좌표)
Console.WriteLine(a.Equals(c));    // False

// Equals를 재정의하지 않으면 참조 동등성이 사용됨:
// a.Equals(b)는 서로 다른 객체이므로 False가 됨

// HashCode 일관성
Console.WriteLine(a.GetHashCode() == b.GetHashCode());  // True
```

### 9.3 완전한 재정의 패턴

```csharp
public class Student
{
    public string Id { get; set; }
    public string Name { get; set; }
    public double Gpa { get; set; }

    public override string ToString()
    {
        return $"Student {Id}: {Name} (GPA: {Gpa:F2})";
    }

    public override bool Equals(object obj)
    {
        if (ReferenceEquals(this, obj)) return true;
        if (obj is null) return false;
        if (GetType() != obj.GetType()) return false;

        Student other = (Student)obj;
        return Id == other.Id;
    }

    public override int GetHashCode()
    {
        return Id?.GetHashCode() ?? 0;
    }

    // 선택 사항: 동등성을 위한 연산자 오버로드
    public static bool operator ==(Student left, Student right)
    {
        if (left is null) return right is null;
        return left.Equals(right);
    }

    public static bool operator !=(Student left, Student right)
    {
        return !(left == right);
    }
}
```

## 10. 실전 예제: 도형 계층 구조

상속, 추상 클래스, 메서드 재정의, 다형성을 모두 보여주는 종합적인 도형 계층 구조를 만들어 봅시다.

### 10.1 추상 기본 클래스

```csharp
public abstract class Shape
{
    public string Color { get; set; }
    public string Name { get; protected set; }

    protected Shape(string color)
    {
        Color = color;
    }

    // 추상 멤버: 모든 도형이 정의해야 함
    public abstract double Area();
    public abstract double Perimeter();

    // 가상 메서드: 재정의할 수 있지만 기본 구현이 있음
    public virtual void Draw()
    {
        Console.WriteLine($"Drawing a {Color} {Name}");
    }

    public override string ToString()
    {
        return $"{Name} [Color={Color}, Area={Area():F2}, Perimeter={Perimeter():F2}]";
    }
}
```

### 10.2 구체적 도형 클래스

```csharp
public class Circle : Shape
{
    public double Radius { get; set; }

    public Circle(double radius, string color = "red") : base(color)
    {
        Radius = radius;
        Name = "Circle";
    }

    public override double Area() => Math.PI * Radius * Radius;
    public override double Perimeter() => 2 * Math.PI * Radius;
}

public class Rectangle : Shape
{
    public double Width { get; set; }
    public double Height { get; set; }

    public Rectangle(double width, double height, string color = "blue") : base(color)
    {
        Width = width;
        Height = height;
        Name = "Rectangle";
    }

    public override double Area() => Width * Height;
    public override double Perimeter() => 2 * (Width + Height);
}

public class Triangle : Shape
{
    public double SideA { get; set; }
    public double SideB { get; set; }
    public double SideC { get; set; }

    public Triangle(double a, double b, double c, string color = "green") : base(color)
    {
        SideA = a;
        SideB = b;
        SideC = c;
        Name = "Triangle";
    }

    public override double Area()
    {
        double s = Perimeter() / 2; // 반둘레
        return Math.Sqrt(s * (s - SideA) * (s - SideB) * (s - SideC));
    }

    public override double Perimeter() => SideA + SideB + SideC;
}
```

### 10.3 계층 구조 확장

```csharp
public class Square : Rectangle
{
    public Square(double side, string color = "purple")
        : base(side, side, color)
    {
        Name = "Square";
    }

    public override void Draw()
    {
        base.Draw();
        Console.WriteLine($"  Side length: {Width}");
    }
}

public sealed class EquilateralTriangle : Triangle
{
    public EquilateralTriangle(double side, string color = "orange")
        : base(side, side, side, color)
    {
        Name = "Equilateral Triangle";
    }
}
```

### 10.4 다형성을 활용한 계층 구조 사용

```csharp
class Program
{
    static void Main()
    {
        List<Shape> shapes = new List<Shape>
        {
            new Circle(5, "red"),
            new Rectangle(4, 6, "blue"),
            new Triangle(3, 4, 5, "green"),
            new Square(7, "purple"),
            new EquilateralTriangle(10, "orange")
        };

        Console.WriteLine("=== Shape Report ===");
        double totalArea = 0;

        foreach (Shape shape in shapes)
        {
            Console.WriteLine(shape);   // 재정의된 ToString() 호출
            shape.Draw();               // 재정의된 또는 기본 Draw() 호출
            totalArea += shape.Area();  // 다형적 호출
            Console.WriteLine();
        }

        Console.WriteLine($"Total area of all shapes: {totalArea:F2}");

        // 타입 검사 및 다운캐스팅
        Console.WriteLine("\n=== Circles Only ===");
        foreach (Shape shape in shapes)
        {
            if (shape is Circle circle)
            {
                Console.WriteLine($"Circle radius: {circle.Radius}");
            }
        }

        // 가장 큰 도형 찾기
        Shape largest = shapes.OrderByDescending(s => s.Area()).First();
        Console.WriteLine($"\nLargest shape: {largest}");
    }
}
```

출력:
```
=== Shape Report ===
Circle [Color=red, Area=78.54, Perimeter=31.42]
Drawing a red Circle

Rectangle [Color=blue, Area=24.00, Perimeter=20.00]
Drawing a blue Rectangle

Triangle [Color=green, Area=6.00, Perimeter=12.00]
Drawing a green Triangle

Square [Color=purple, Area=49.00, Perimeter=28.00]
Drawing a purple Square
  Side length: 7

Equilateral Triangle [Color=orange, Area=43.30, Perimeter=30.00]
Drawing a orange Equilateral Triangle

Total area of all shapes: 200.84

=== Circles Only ===
Circle radius: 5

Largest shape: Circle [Color=red, Area=78.54, Perimeter=31.42]
```

## 11. 연습 문제

1. **동물 계층 구조**: 추상 `Animal` 클래스에 추상 `Speak()`과 `Move()` 메서드, `Name` 속성, 그리고 구체적인 `Describe()` 메서드를 작성하세요. `Bird`, `Fish`, `Mammal` 클래스를 적절한 구현으로 파생시키세요. 그런 다음 `Bird`를 상속하지만 `Move()`를 재정의하여 "fly" 대신 "waddle"이라고 말하는 `Penguin` 클래스를 만드세요. 여러 동물을 `List<Animal>`에 저장하고 다형적으로 순회하세요.

2. **직원 급여**: `Name`과 `Id` 속성이 있는 기본 클래스 `Employee`와 추상 메서드 `CalculatePay()`를 만드세요. `SalariedEmployee`(고정 월급), `HourlyEmployee`(근무 시간 × 시급), `CommissionEmployee`(기본급 + 판매 수수료 비율)를 파생시키세요. `ToString()`과 `Equals()`를 재정의하세요(`Id` 기반 동등성). 직원 목록을 받아 총 급여가 포함된 급여 보고서를 출력하는 메서드를 작성하세요.

3. **세 가지 모두 재정의하기**: `Isbn`, `Title`, `Author` 속성이 있는 `Book` 클래스를 만드세요. `ToString()`, `Equals()`(ISBN 기반), `GetHashCode()`를 재정의하세요. 같은 ISBN을 가진 두 `Book` 객체가 동등하다고 간주되고, 딕셔너리 키로 올바르게 사용될 수 있으며, 의미 있는 문자열 표현을 보여주는 것을 시연하세요.

4. **Sealed 방지**: 클래스 계층 구조를 만드세요: `Account`(추상) -> `CheckingAccount` -> `PremiumCheckingAccount`. `CheckingAccount`에서 `Withdraw()` 메서드를 봉인하여 `PremiumCheckingAccount`가 출금 로직을 변경할 수 없지만 `EarnRewards()`와 같은 자체 메서드를 추가할 수 있게 하세요. 봉인된 메서드를 재정의하려고 하면 컴파일 오류가 발생하는 것을 시연하세요(주석 처리하고 설명하세요).

5. **타입 검사 동물원**: 2~3단계의 상속으로 최소 5개의 구체적 타입이 있는 동물원 동물 계층 구조를 만드세요. `is` 패턴 매칭과 `switch` 표현식을 사용하여 각 동물의 실제 타입에 따라 다른 보고서를 생성하는 `ZooReport(List<Animal> animals)` 메서드를 작성하세요(예: 포유류에는 "서식지" 보고서, 새에는 "날개 폭" 보고서, 파충류에는 "온도" 보고서).
